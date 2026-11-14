"""Execute one contract end to end.

Guarantees, in the order they matter for a release:

* Actually run the restoration; the self-check calls this same code path.
* Recompute the forward residual and output dimensions independently of the
  solver's own bookkeeping before anything is published.
* Publish artefacts atomically only after the solve finished.  A cancelled or
  failed run leaves no ``restored.png`` that could be mistaken for a result.
* Record the parameters actually used and, on failure, the stage that failed.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .artifacts import (atomic_json, atomic_png, curves_png, kernel_sheet,
                        residual_visual)
from .contract import CaseContract, ContractError, write_run_record
from .operator import SVForward
from .solve import SolveConfig, TVSolver
from .synth import psnr
from .threads import thread_config

ARTIFACT_FILES = ("restored.png", "residual.png", "kernels.png", "curves.png",
                  "restored.npz")


def record_load_failure(contract_path: Path, exc: ContractError,
                        root: Path | None = None) -> None:
    """Persist an input-contract failure so the clean-dir test can see the stage.

    The output directory is read best-effort from the raw contract; if even
    that is unreadable there is nothing to write and the CLI message is the
    only record.  Any restored image left from an earlier run is removed so a
    failed input cannot sit next to a stale "result".
    """
    out_dir = root / "out" / contract_path.stem if root else contract_path.parent / "out"
    declared: dict[str, Any] = {}
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if raw.get("output_dir"):
                base = root if root is not None else Path.cwd()
                out_dir = (base / raw["output_dir"]).resolve()
            declared = {
                "name": raw.get("name"),
                "boundary": raw.get("boundary"),
                "regularization": raw.get("regularization"),
                "observed_image": raw.get("observed_image"),
                "kernel_field": raw.get("kernel_field"),
                "memory_budget_mb": raw.get("memory_budget_mb"),
                "threads": raw.get("threads"),
            }
    except Exception:
        pass
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        _clear_previous_artifacts(out_dir)
        record = {
            "run_id": f"{contract_path.stem}-unstarted",
            "case": contract_path.stem,
            "status": "failed",
            "stage": exc.stage,
            "error": exc.as_dict(),
            "declared_params": declared,
            "contract_source": str(contract_path.resolve()),
        }
        atomic_json(out_dir / "report.json", record)
        write_run_record(record, out_dir)
    except OSError:
        pass


class Cancelled(RuntimeError):
    """The run was aborted on request; nothing is published."""


@dataclass
class RunOutcome:
    case: str
    status: str                  # "completed" | "cancelled" | "failed"
    out_dir: Path
    record: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status == "completed"


def _solver_config(contract: CaseContract) -> SolveConfig:
    reg = contract.regularization
    return SolveConfig(
        iters=int(reg["iters"]), lam=reg.get("lam"),
        lam_rel=float(reg.get("lam_rel", 0.02)),
        tv_eps=float(reg["tv_eps"]), sigma=float(reg["sigma"]),
        damping_mu=float(reg.get("damping_mu", 0.0)),
        tol=float(reg.get("tol", 1e-9)),
        noise_floor_factor=float(reg.get("noise_floor_factor", 0.8)),
    )


def _clear_previous_artifacts(out_dir: Path) -> None:
    for name in ARTIFACT_FILES + ("report.json",):
        p = out_dir / name
        if p.exists():
            p.unlink()


def independent_verify(observed: np.ndarray, restored: np.ndarray,
                       forward: SVForward, result) -> dict[str, Any]:
    """Recompute what the solver claims, using a fresh forward evaluation.

    * forward residual ||A x_hat - b|| recomputed from scratch;
    * output shape and finite-value check;
    * comparison against the solver's own final residual to catch drift.
    """
    residual = forward.apply(restored) - observed
    recomputed_abs = float(np.linalg.norm(residual))
    recomputed_rms = float(np.sqrt(np.mean(residual ** 2)))
    solver_abs = float(result.residual_abs[-1]) if result.residual_abs else float("nan")
    rel_gap = abs(recomputed_abs - solver_abs) / max(abs(solver_abs), 1e-15)
    shape_ok = restored.shape == observed.shape
    finite_ok = bool(np.isfinite(restored).all())
    residual_matches = rel_gap < 1e-9
    checks = {
        "output_shape": list(restored.shape),
        "expected_shape": list(observed.shape),
        "output_shape_ok": shape_ok,
        "finite_output": finite_ok,
        "recomputed_residual_abs": recomputed_abs,
        "recomputed_residual_rms": recomputed_rms,
        "solver_final_residual_abs": solver_abs,
        "residual_relative_gap": rel_gap,
        "residual_matches_solver": residual_matches,
        "passed": bool(shape_ok and finite_ok and residual_matches
                       and math.isfinite(recomputed_abs)),
    }
    if not checks["passed"]:
        bad = [k for k in ("output_shape_ok", "finite_output", "residual_matches_solver")
               if not checks[k]]
        raise ContractError(
            "verification", "SELF_CHECK_FAILED",
            "independent verification failed: " + ", ".join(bad)
            + f" (residual relative gap {rel_gap:.3e})")
    return checks


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise Cancelled("cancelled by request")


def run_contract(contract: CaseContract, *,
                 cancel: Callable[[], bool] | None = None,
                 log: Callable[[str], None] | None = None,
                 save_npy: bool = False) -> RunOutcome:
    out_dir = contract.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    params = contract.resolved_params()
    run_id = f"{contract.name}-{uuid.uuid4().hex[:10]}"
    started = time.time()
    record: dict[str, Any] = {
        "run_id": run_id,
        "case": contract.name,
        "status": "running",
        "stage": "start",
        "params": params,
        "threads": thread_config(),
        "contract_source": str(contract.source) if contract.source else None,
        "warnings": contract.warnings,
        "kernel_diagnostics": contract.diagnostics,
    }
    _clear_previous_artifacts(out_dir)
    write_run_record(record, out_dir)  # a crash from here leaves status=running only

    def emit(msg: str) -> None:
        if log is not None:
            log(msg)

    try:
        _check_cancel(cancel)
        record["stage"] = "solve"
        forward = SVForward(contract.field, contract.image_shape,
                            boundary=contract.boundary, allow_signed=False)
        cfg = _solver_config(contract)
        solver = TVSolver(forward, cfg, reference=contract.image)

        def progress(it: int, info: dict[str, Any]) -> None:
            _check_cancel(cancel)

        try:
            result = solver.run(contract.image, progress=progress)
        except MemoryError as exc:
            raise ContractError(
                "memory_budget", "MEMORY_ERROR",
                f"allocation failed inside the solver at the declared "
                f"{contract.memory_budget_mb} MB budget: {exc}") from exc

        _check_cancel(cancel)
        record["stage"] = "verify"
        restored = np.clip(result.image, 0.0, 1.0)
        verification = independent_verify(contract.image, restored, forward, result)
        record["verification"] = verification
        record["solver"] = result.as_metadata()
        record["stop"] = {"reason": result.stop_reason,
                          "iteration": result.stop_iteration,
                          "detail": result.stop_detail,
                          "noise_floor": result.noise_floor}

        residual = forward.apply(restored) - contract.image
        rstats = {
            "l2": float(np.linalg.norm(residual)),
            "rms": float(np.sqrt(np.mean(residual ** 2))),
            "relative": float(np.linalg.norm(residual)
                              / max(np.linalg.norm(contract.image), 1e-12)),
            "mean": float(residual.mean()),
            "max_abs": float(np.abs(residual).max()),
        }
        metrics = {
            "psnr_input": psnr(contract.image, contract.image),
            "residual_rms": rstats["rms"],
        }
        if contract.reference_sharp is not None:
            metrics["psnr_restored_vs_reference"] = psnr(restored, contract.reference_sharp)
            metrics["gain_db"] = (metrics["psnr_restored_vs_reference"]
                                  - psnr(contract.image, contract.reference_sharp))
        record["residual"] = rstats
        record["metrics"] = metrics

        _check_cancel(cancel)
        record["stage"] = "publish"
        atomic_png(out_dir / "restored.png", restored)
        atomic_png(out_dir / "residual.png", residual_visual(residual))
        kernel_sheet(contract.field, contract.image_shape, out_dir / "kernels.png",
                     f"{contract.name}: interpolated kernel field")
        curves_png(result.residuals, result.objectives, result.tv_terms,
                   result.stop_iteration, out_dir / "curves.png",
                   f"{contract.name}: residual / objective / TV")
        if save_npy:
            tmp = out_dir / ".restored.npz.tmp"
            with open(tmp, "wb") as fh:
                np.savez(fh, restored=restored, residual=residual)
            tmp.replace(out_dir / "restored.npz")
        record["artifacts"] = {
            "restored": "restored.png",
            "residual": "residual.png",
            "kernels": "kernels.png",
            "curves": "curves.png",
            "report": "report.json",
        }
        record["elapsed_s"] = round(time.time() - started, 3)
        record["status"] = "completed"
        record["stage"] = "done"
        atomic_json(out_dir / "report.json", record)
        write_run_record(record, out_dir)
        emit(f"[{contract.name}] completed: stop={result.stop_reason} "
             f"@{result.stop_iteration}, residual rms={rstats['rms']:.4e}")
        return RunOutcome(contract.name, "completed", out_dir, record)

    except Cancelled as exc:
        record["status"] = "cancelled"
        record["error"] = {"stage": record["stage"], "code": "CANCELLED",
                           "message": str(exc)}
        record["elapsed_s"] = round(time.time() - started, 3)
        _clear_previous_artifacts(out_dir)
        atomic_json(out_dir / "report.json", record)
        write_run_record(record, out_dir)
        emit(f"[{contract.name}] cancelled at stage {record['stage']}; no restored image written")
        return RunOutcome(contract.name, "cancelled", out_dir, record)
    except ContractError as exc:
        record["status"] = "failed"
        record["stage"] = exc.stage
        record["error"] = exc.as_dict()
        record["elapsed_s"] = round(time.time() - started, 3)
        _clear_previous_artifacts(out_dir)
        atomic_json(out_dir / "report.json", record)
        write_run_record(record, out_dir)
        emit(f"[{contract.name}] FAILED at stage {exc.stage} ({exc.code}): {exc.message}")
        return RunOutcome(contract.name, "failed", out_dir, record)
    except MemoryError as exc:
        record["status"] = "failed"
        record["stage"] = "solve"
        record["error"] = {"stage": "solve", "code": "MEMORY_ERROR",
                           "message": str(exc)}
        record["elapsed_s"] = round(time.time() - started, 3)
        _clear_previous_artifacts(out_dir)
        atomic_json(out_dir / "report.json", record)
        write_run_record(record, out_dir)
        emit(f"[{contract.name}] FAILED at stage solve (MEMORY_ERROR)")
        return RunOutcome(contract.name, "failed", out_dir, record)
