"""Operational safety checks in a clean temporary directory.

Covers the three scenarios that must fail cleanly and leave no artefact that
could be mistaken for a finished restoration:

* kernel-field version mismatch  -> stage ``kernel_field``, exit non-zero;
* read-only output directory     -> stage ``output``, exit non-zero;
* abort while the solver runs    -> status ``cancelled`` and NO restored.png.

Everything happens under a fresh ``tempfile.mkdtemp`` so the repository tree
and any existing output are untouched.  Run with ``python3 scripts/test_clean_dir.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0", "PYTHONPATH": str(ROOT),
    })
    return env


def _copy_case(target: Path, name: str = "gradient") -> None:
    src = ROOT / "release" / "cases" / name
    target.mkdir(parents=True, exist_ok=True)
    for f in ("observed.png", "field.json", "reference.png"):
        shutil.copy2(src / f, target / f)


def _contract(folder: Path, out_dir: Path, **over) -> Path:
    body = {
        "contract_version": 1,
        "name": over.pop("name", "clean"),
        "observed_image": str(folder / "observed.png"),
        "kernel_field": str(folder / "field.json"),
        "output_dir": str(out_dir),
        "boundary": "reflect",
        "regularization": {"iters": over.pop("iters", 40), "sigma": 0.004, "tv_eps": 0.008},
        "memory_budget_mb": 512,
        "threads": 1,
    }
    body.update(over)
    path = folder / "task.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _run(task: Path, timeout: int = 120):
    proc = subprocess.run(
        [sys.executable, "-m", "svdeblur.cli", "run", str(task),
         "--root", str(ROOT)],
        cwd=str(ROOT), env=_env(), capture_output=True, text=True, timeout=timeout)
    return proc


def _record(out_dir: Path) -> dict:
    p = out_dir / "run.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def check_version_mismatch(work: Path) -> dict:
    folder = work / "version"
    _copy_case(folder)
    field = json.loads((folder / "field.json").read_text())
    field["field_version"] = 1
    (folder / "field.json").write_text(json.dumps(field), encoding="utf-8")
    out = work / "out_version"
    task = _contract(folder, out)
    proc = _run(task)
    rec = _record(out)
    restored = (out / "restored.png").exists()
    ok = (proc.returncode != 0 and rec.get("stage") == "kernel_field"
          and rec.get("error", {}).get("code") == "KERNEL_FIELD_VERSION"
          and not restored)
    return {"name": "kernel_version_mismatch", "passed": ok,
            "exit": proc.returncode, "stage": rec.get("stage"),
            "code": rec.get("error", {}).get("code"),
            "restored_png_present": restored}


def check_readonly_output(work: Path) -> dict:
    folder = work / "readonly"
    _copy_case(folder)
    out = work / "out_readonly"
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o500)
    task = _contract(folder, out)
    try:
        proc = _run(task)
    finally:
        os.chmod(out, 0o700)
    stage = None
    code = None
    if proc.stdout.strip():
        try:
            body = json.loads(proc.stdout)
            stage, code = body.get("stage"), body.get("code")
        except json.JSONDecodeError:
            pass
    ok = proc.returncode != 0 and stage == "output" and code == "OUTPUT_READONLY"
    return {"name": "readonly_output", "passed": ok, "exit": proc.returncode,
            "stage": stage, "code": code,
            "restored_png_present": (out / "restored.png").exists()}


def check_abort(work: Path) -> dict:
    """Cancel mid-run and assert no restored image survives.

    Two paths are checked: a cooperative cancel marker (records
    status=cancelled cleanly) and a hard SIGTERM to the process group (the
    stronger guarantee: even an abrupt stop leaves no restored.png).
    """
    results = []

    folder = work / "abort"
    _copy_case(folder)
    out = work / "out_abort"
    out.mkdir(parents=True, exist_ok=True)
    # A stale image from a previous run must also be removed by a cancelled run.
    (out / "restored.png").write_bytes(b"stale-result-not-from-this-run")
    task = _contract(folder, out, iters=100000, name="abort")
    cancel_marker = work / "cancel.request"
    proc = subprocess.Popen(
        [sys.executable, "-m", "svdeblur.cli", "run", str(task),
         "--root", str(ROOT), "--cancel-file", str(cancel_marker)],
        cwd=str(ROOT), env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    time.sleep(2.0)
    cancel_marker.write_text("stop")
    try:
        out_text, _ = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out_text, _ = proc.communicate()
    rec = _record(out)
    restored = (out / "restored.png").exists()
    results.append({
        "name": "cancel_marker", "passed": (not restored
                                            and rec.get("status") == "cancelled"),
        "exit": proc.returncode, "record_status": rec.get("status"),
        "stage": rec.get("stage"), "restored_png_present": restored})

    # Abrupt SIGTERM: no graceful record expected, but still no restored image.
    folder2 = work / "abort_sigterm"
    _copy_case(folder2)
    out2 = work / "out_abort_sigterm"
    out2.mkdir(parents=True, exist_ok=True)
    (out2 / "restored.png").write_bytes(b"stale-result")
    task2 = _contract(folder2, out2, iters=100000, name="abort_sigterm")
    proc2 = subprocess.Popen(
        [sys.executable, "-m", "svdeblur.cli", "run", str(task2), "--root", str(ROOT)],
        cwd=str(ROOT), env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    time.sleep(2.0)
    os.killpg(proc2.pid, signal.SIGTERM)
    try:
        proc2.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc2.pid, signal.SIGKILL)
        proc2.communicate()
    restored2 = (out2 / "restored.png").exists()
    results.append({"name": "sigterm_mid_run", "passed": not restored2,
                    "exit": proc2.returncode,
                    "record_status": _record(out2).get("status"),
                    "restored_png_present": restored2})
    return {"name": "abort_mid_run",
            "passed": all(r["passed"] for r in results), "subchecks": results}


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="svdeblur-cleandir-"))
    try:
        results = [check_version_mismatch(work), check_readonly_output(work),
                   check_abort(work)]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    report = {"passed": all(r["passed"] for r in results), "checks": results}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
