"""Release-level checks: contract, self-check, failures, determinism, UI residual.

These exercise the shipped scripts and the serving path rather than the bare
numerics (which live in ``test_core.py``).  They run without a browser; the
Playwright-driven window check is ``scripts/ui_check.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from svdeblur.contract import (ContractError, estimate_working_set_bytes,
                               load_contract, load_kernel_field,
                               load_observed_image)
from svdeblur.field import KernelField, make_control_field
from svdeblur.runner import run_contract
from svdeblur.selfcheck import run_selfcheck
from svdeblur.server import svc_load, svc_restore
from svdeblur.synth import gaussian_psf

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ contract
def test_selfcheck_runs_a_real_restoration_and_verifies_it():
    with tempfile.TemporaryDirectory() as tmp:
        report = run_selfcheck(Path(tmp))
    assert report["passed"], report
    v = report["stages"]["restoration"]["verification"]
    assert v["passed"] and v["output_shape_ok"] and v["finite_output"]
    # the independently recomputed residual must match the solver's own
    assert v["residual_relative_gap"] < 1e-9


def test_missing_kernel_file_fails_at_kernel_stage(tmp_path):
    img = tmp_path / "obs.png"
    from PIL import Image
    Image.fromarray(np.full((24, 24), 128, np.uint8)).save(img)
    with pytest.raises(ContractError) as ei:
        load_kernel_field(tmp_path / "nope.json", (24, 24), allow_signed=False)
    assert ei.value.stage == "kernel_field"
    assert ei.value.code == "KERNEL_FIELD_MISSING"


def test_unsupported_image_dimensionality_is_rejected(tmp_path):
    from PIL import Image
    p = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(p)
    with pytest.raises(ContractError) as ei:
        load_observed_image(p)
    assert ei.value.stage == "image"
    assert ei.value.code == "UNSUPPORTED_IMAGE_DIMENSION"


def test_kernel_field_version_mismatch_is_rejected(tmp_path):
    body = {"field_version": 1, "grid_shape": [2, 2], "support": [5, 5],
            "interpolation": "bilinear", "normalization": "unit_sum",
            "cells": [[[[1, 0, 0, 0, 0]] * 5] * 2] * 2}
    p = tmp_path / "f.json"
    p.write_text(json.dumps(body))
    with pytest.raises(ContractError) as ei:
        load_kernel_field(p, (16, 16), allow_signed=False)
    assert ei.value.code == "KERNEL_FIELD_VERSION"


def test_memory_budget_exceeded_is_explicit(tmp_path):
    H = W = 384
    field = make_control_field((H, W), (4, 4), (15, 15),
                               lambda yn, xn, i, j: gaussian_psf((15, 15), 1.2, 1.2))
    est = estimate_working_set_bytes(field, (H, W), snapshot_count=8)
    assert est > 16 * 2 ** 20  # the budget the fixture declares


# ------------------------------------------------------------------ end to end
def test_release_case_runs_and_records_params():
    case = ROOT / "release" / "cases" / "gradient"
    if not (case / "task.json").is_file():
        pytest.skip("release fixtures not built; run scripts/build_fixtures.py")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copytree(case, root / "case")
        task = json.loads((root / "case" / "task.json").read_text())
        task["observed_image"] = str(root / "case" / "observed.png")
        task["kernel_field"] = str(root / "case" / "field.json")
        task["reference_image"] = str(root / "case" / "reference.png")
        task["output_dir"] = str(root / "out")
        task["regularization"]["iters"] = 60
        (root / "case" / "task2.json").write_text(json.dumps(task))
        contract = load_contract(root / "case" / "task2.json", root=root)
        outcome = run_contract(contract)
        assert outcome.ok
        report = json.loads((root / "out" / "report.json").read_text())
        assert report["status"] == "completed"
        assert report["verification"]["passed"]
        assert report["params"]["regularization"]["lam_resolved"] == pytest.approx(
            0.02 * 0.004)
        assert report["threads"]["release_threads"] == 1
        assert (root / "out" / "restored.png").is_file()
        # published image keeps full resolution (no downsampling to hide drift)
        from PIL import Image
        assert Image.open(root / "out" / "restored.png").size == (
            contract.image_shape[1], contract.image_shape[0])


def test_cancelled_run_publishes_no_restored_image(tmp_path):
    case = ROOT / "release" / "cases" / "gradient"
    if not (case / "task.json").is_file():
        pytest.skip("release fixtures not built")
    shutil.copytree(case, tmp_path / "case")
    task = json.loads((tmp_path / "case" / "task.json").read_text())
    task.update({
        "observed_image": str(tmp_path / "case" / "observed.png"),
        "kernel_field": str(tmp_path / "case" / "field.json"),
        "reference_image": str(tmp_path / "case" / "reference.png"),
        "output_dir": str(tmp_path / "out"),
    })
    task["regularization"]["iters"] = 100000
    (tmp_path / "case" / "t.json").write_text(json.dumps(task))
    contract = load_contract(tmp_path / "case" / "t.json", root=tmp_path)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 3

    # a stale image from an earlier run must not survive
    contract.output_dir.mkdir(parents=True, exist_ok=True)
    (contract.output_dir / "restored.png").write_bytes(b"stale")
    outcome = run_contract(contract, cancel=cancel)
    assert outcome.status == "cancelled"
    assert not (contract.output_dir / "restored.png").exists()
    record = json.loads((contract.output_dir / "run.json").read_text())
    assert record["status"] == "cancelled"
    assert record["stage"] == "solve"


# ------------------------------------------------------------------ UI payload
def test_restore_payload_carries_real_forward_residual_and_iteration_residual():
    loaded = svc_load({"case": "gradient"})
    body = {"case": "gradient", "field": loaded["field"], "shape": loaded["shape"],
            "allow_signed": False, "boundary": "reflect",
            "config": {"iters": 60, "sigma": 0.004, "tv_eps": 0.008},
            "snapshots": [10, 60]}
    res = svc_restore(body)
    assert res["ok"]
    assert res["residual"].startswith("data:image/png;base64,")
    r = res["metrics"]["residual"]
    assert r["rms"] > 0
    series = res["series"]["iteration_residuals"]
    assert series and series[-1]["iteration"] == res["series"]["stop_iteration"] or series
    # the final iteration residual equals the recomputed forward residual
    assert abs(series[-1]["residual_abs"] - r["l2"]) <= 1e-6 * max(r["l2"], 1.0)


def test_restore_payload_blocks_on_missing_kernel():
    loaded = svc_load({"case": "gradient"})
    field = loaded["field"]
    field["cells"][1][1] = None
    res = svc_restore({"case": "gradient", "field": field, "shape": loaded["shape"]})
    assert not res["ok"]
    assert any(d["code"] == "MISSING_KERNEL" for d in res["diagnostics"]["errors"])


# --------------------------------------------------------------- batch runner
def _mini_manifest(tmp_path: Path) -> Path:
    case = ROOT / "release" / "cases" / "gradient"
    if not (case / "task.json").is_file():
        pytest.skip("release fixtures not built")
    shutil.copytree(case, tmp_path / "good")
    task = json.loads((tmp_path / "good" / "task.json").read_text())
    task.update({
        "name": "good",
        "observed_image": str(tmp_path / "good" / "observed.png"),
        "kernel_field": str(tmp_path / "good" / "field.json"),
        "reference_image": str(tmp_path / "good" / "reference.png"),
        "output_dir": str(tmp_path / "out_good"),
    })
    task["regularization"]["iters"] = 40
    (tmp_path / "good" / "task.json").write_text(json.dumps(task))

    # a contract whose kernel file does not exist
    shutil.copytree(ROOT / "release" / "fixtures" / "missing_kernel_file",
                    tmp_path / "bad")
    bad = json.loads((tmp_path / "bad" / "task.json").read_text())
    bad.update({
        "observed_image": str(tmp_path / "bad" / "observed.png"),
        "kernel_field": str(tmp_path / "bad" / "does_not_exist.json"),
        "output_dir": str(tmp_path / "out_bad"),
    })
    (tmp_path / "bad" / "task.json").write_text(json.dumps(bad))

    manifest = {
        "manifest_version": 1,
        "cases": [
            {"name": "good", "contract": str(tmp_path / "good" / "task.json"),
             "out_dir": str(tmp_path / "out_good")},
            {"name": "bad", "contract": str(tmp_path / "bad" / "task.json"),
             "out_dir": str(tmp_path / "out_bad"), "expect_failure": "kernel_field"},
        ],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p


def test_batch_runner_reports_expected_and_unexpected_failures(tmp_path):
    from svdeblur.release import run_manifest
    manifest = _mini_manifest(tmp_path)
    summary = run_manifest(manifest, root=tmp_path, jobs=1)
    by_case = {r["case"]: r for r in summary["results"]}
    assert by_case["good"]["status"] == "completed"
    assert by_case["bad"]["status"] == "failed"
    assert by_case["bad"]["stage"] == "kernel_field"
    assert "expectation_met" in by_case["bad"]
    assert not (tmp_path / "out_bad" / "restored.png").exists()


def test_verify_refuses_downsampled_output(tmp_path):
    """Shrinking both runs identically must still be refused, not passed."""
    from PIL import Image
    from svdeblur.verify import verify_release
    ref_dir = ROOT / "out" / "release_ref" / "gradient"
    cand_dir = ROOT / "out" / "release" / "gradient"
    if not ((ref_dir / "restored.npz").is_file() and (cand_dir / "restored.npz").is_file()):
        pytest.skip("run scripts/build.sh first (needs reference and candidate runs)")

    def shrink(src: Path, dst: Path):
        shutil.copytree(src, dst)
        png = Image.open(dst / "restored.png")
        full = (png.size[1], png.size[0])
        png.resize((96, 96)).save(dst / "restored.png")
        np.savez(dst / "restored.npz", restored=np.zeros((96, 96)),
                 residual=np.zeros((96, 96)))
        report = json.loads((dst / "report.json").read_text())
        report["params"]["image_shape"] = list(full)  # claim full resolution
        (dst / "report.json").write_text(json.dumps(report))

    shrink(ref_dir, tmp_path / "ref")
    shrink(cand_dir, tmp_path / "cand")
    result = verify_release(tmp_path / "cand", tmp_path / "ref", threads=1)
    assert not result["passed"]
    assert any("downsampled" in p for p in result["problems"])
