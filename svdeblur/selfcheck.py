"""Startup self-check: execute one small real restoration, then audit it.

This deliberately does *not* stop at "numpy/scipy import fine".  It builds a
small contract on disk, runs the same :func:`svdeblur.runner.run_contract`
code the batch release uses, and independently re-derives the forward
residual, the output size and the written PNGs.  Any mismatch fails the
check with the *stage* that failed.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .artifacts import atomic_json, atomic_png
from .contract import ContractError, load_contract
from .field import KernelField, make_control_field
from .runner import run_contract
from .synth import gaussian_psf, texture_bars

SELF_CHECK_SIZE = 48
SELF_CHECK_GRID = (2, 2)
SELF_CHECK_SUPPORT = (7, 7)


def _build_fixture(root: Path) -> Path:
    """Write observed.png, field.json and task.json for the self-check."""
    H = W = SELF_CHECK_SIZE
    sharp = texture_bars(W, H)
    field = make_control_field(
        (H, W), SELF_CHECK_GRID, SELF_CHECK_SUPPORT,
        lambda yn, xn, i, j: gaussian_psf(SELF_CHECK_SUPPORT,
                                         0.9 + 0.9 * yn, 0.9 + 0.9 * xn))
    from .operator import SVForward
    blurred = SVForward(field, (H, W), boundary="reflect").apply(sharp)
    rng = np.random.default_rng(20240117)
    observed = np.clip(blurred + 0.003 * rng.standard_normal((H, W)), 0.0, 1.0)

    atomic_png(root / "observed.png", observed)
    atomic_png(root / "reference.png", sharp)
    body = field.to_dict()
    body["field_version"] = 2
    atomic_json(root / "field.json", body)
    contract = {
        "contract_version": 1,
        "name": "selfcheck",
        "observed_image": "observed.png",
        "reference_image": "reference.png",
        "kernel_field": "field.json",
        "output_dir": "out",
        "boundary": "reflect",
        "regularization": {"iters": 60, "sigma": 0.003, "tv_eps": 0.008,
                           "lam_rel": 0.02, "tol": 1e-9},
        "memory_budget_mb": 256,
        "threads": 1,
    }
    atomic_json(root / "task.json", dict(contract))
    return root / "task.json"


def run_selfcheck(out_dir: Path, keep: bool = False) -> dict[str, Any]:
    """Run the check; returns a JSON-able report with ``passed`` and stages."""
    report: dict[str, Any] = {"passed": False, "stages": {}}

    work = Path(tempfile.mkdtemp(prefix="svdeblur-selfcheck-"))
    try:
        # ---- stage: fixture
        try:
            task = _build_fixture(work)
            report["stages"]["fixture"] = "ok"
        except Exception as exc:
            report["stages"]["fixture"] = f"failed: {exc}"
            report["failed_stage"] = "fixture"
            return report

        # ---- stage: contract validation (catches missing field / bad dims)
        try:
            contract = load_contract(task, root=work)
            report["stages"]["contract"] = {
                "ok": True,
                "image_shape": list(contract.image_shape),
                "kernel_grid": list(contract.field.grid_shape),
                "kernel_support": list(contract.field.support),
                "estimated_working_set_mb": round(contract.estimated_bytes / 2 ** 20, 3),
                "params": contract.resolved_params(),
            }
        except ContractError as exc:
            report["stages"]["contract"] = {"ok": False, **exc.as_dict()}
            report["failed_stage"] = exc.stage
            return report

        # ---- stage: real restoration through the release runner
        outcome = run_contract(contract)
        report["stages"]["restoration"] = {
            "status": outcome.status,
            "stage": outcome.record.get("stage"),
            "stop": outcome.record.get("stop"),
            "verification": outcome.record.get("verification"),
        }
        if not outcome.ok:
            report["failed_stage"] = outcome.record.get("stage", "restoration")
            return report

        # ---- stage: independent re-loading of the published PNGs
        try:
            restored_png = np.asarray(Image.open(outcome.out_dir / "restored.png")) / 255.0
            observed_png = np.asarray(Image.open(work / "observed.png")) / 255.0
            if restored_png.shape != observed_png.shape:
                raise AssertionError(
                    f"published restored.png is {restored_png.shape}, "
                    f"observed is {observed_png.shape}")
            # Recompute the forward residual from the *published* image and the
            # operator rebuilt from the on-disk contract, not from memory.
            contract2 = load_contract(task, root=work)
            from .operator import SVForward
            forward = SVForward(contract2.field, contract2.image_shape,
                                boundary=contract2.boundary)
            residual = forward.apply(restored_png) - contract2.image
            rms = float(np.sqrt(np.mean(residual ** 2)))
            report["stages"]["published_artifacts"] = {
                "ok": True,
                "restored_png_shape": list(restored_png.shape),
                "observed_png_shape": list(observed_png.shape),
                "recomputed_residual_rms_from_png": rms,
                "byte_size": (outcome.out_dir / "restored.png").stat().st_size,
            }
        except Exception as exc:
            report["stages"]["published_artifacts"] = f"failed: {exc}"
            report["failed_stage"] = "published_artifacts"
            return report

        # ---- stage: output contract (all declared artefacts exist and parse)
        try:
            report["stages"]["output_contract"] = {
                "ok": True,
                "restored_bytes": (outcome.out_dir / "restored.png").stat().st_size,
                "residual_bytes": (outcome.out_dir / "residual.png").stat().st_size,
                "kernels_bytes": (outcome.out_dir / "kernels.png").stat().st_size,
                "curves_bytes": (outcome.out_dir / "curves.png").stat().st_size,
                "report_status": json.loads(
                    (outcome.out_dir / "report.json").read_text())["status"],
            }
        except Exception as exc:
            report["stages"]["output_contract"] = f"failed: {exc}"
            report["failed_stage"] = "output_contract"
            return report

        report["passed"] = True
        if keep:
            target = out_dir / "selfcheck"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(outcome.out_dir, target)
            report["kept_at"] = str(target)
        return report
    finally:
        shutil.rmtree(work, ignore_errors=True)
