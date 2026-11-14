"""Run a manifest of task contracts, one locked-thread subprocess each.

Each case is executed in a fresh interpreter so that BLAS state, the memory
budget and a cancellation signal are per-case.  Failures do not abort the
batch; they are collected and reported, and the overall exit status is
non-zero if any case failed.  The manifest's order is preserved and every
case's output lives under its own directory.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .threads import THREAD_ENV, RELEASE_THREADS

DEFAULT_TIMEOUT_S = 900


def _apply_thread_env(env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    out.update(THREAD_ENV)
    out["PYTHONHASHSEED"] = "0"
    # Cases may live outside the repository, so make the package importable in
    # the child regardless of the caller's PYTHONPATH.
    pkg_parent = str(Path(__file__).resolve().parents[1])
    existing = out.get("PYTHONPATH")
    out["PYTHONPATH"] = pkg_parent if not existing else pkg_parent + os.pathsep + existing
    return out


def run_manifest(manifest: Path, *, root: Path | None = None,
                 jobs: int = 1, timeout_s: int = DEFAULT_TIMEOUT_S,
                 stop_on_fail: bool = False) -> dict[str, Any]:
    root = (root or Path.cwd()).resolve()
    body = json.loads(manifest.read_text(encoding="utf-8"))
    cases = body.get("cases", [])
    if not cases:
        return {"ok": False, "error": "manifest has no 'cases'", "results": []}

    results: list[dict[str, Any]] = []
    base_env = _apply_thread_env(os.environ)
    started = time.time()

    queue = list(cases)
    running: list[dict[str, Any]] = []

    def launch(case: dict[str, Any]):
        contract = (root / case["contract"]).resolve()
        out_dir = (root / case["out_dir"]).resolve() if case.get("out_dir") else None
        cmd = [sys.executable, "-m", "svdeblur.cli", "run", str(contract),
               "--root", str(root)]
        if case.get("save_npy"):
            cmd.append("--save-npy")
        log_path = contract.parent / f".{contract.stem}.log"
        log = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(root), env=base_env,
                                stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        return {"case": case, "proc": proc, "out_dir": out_dir,
                "log_path": log_path, "log": log, "started": time.time(),
                "timed_out": False}

    def drain(slot, killed_for_timeout: bool = False):
        slot["log"].close()
        code = slot["proc"].returncode
        case, out_dir = slot["case"], slot["out_dir"]
        status = "completed" if (code == 0 and not killed_for_timeout) else "failed"
        entry = {"case": case.get("name", case["contract"]), "exit_code": code,
                 "status": status, "log": str(slot["log_path"])}
        if killed_for_timeout:
            entry["error"] = {"stage": "process", "code": "TIMEOUT",
                              "message": f"exceeded {timeout_s}s wall clock"}
        if out_dir is not None:
            rec = out_dir / "run.json"
            if rec.is_file():
                record = json.loads(rec.read_text(encoding="utf-8"))
                entry["status"] = record.get("status", status)
                entry["stage"] = record.get("stage")
                entry["run_id"] = record.get("run_id")
                if record.get("error"):
                    entry["error"] = record["error"]
            entry["out_dir"] = str(out_dir)
        if case.get("expect_failure"):
            want = case["expect_failure"]
            got_stage = entry.get("stage")
            entry["expected_failure"] = want
            entry["expectation_met"] = (entry["status"] == "failed" and got_stage == want)
        results.append(entry)

    try:
        while queue or running:
            while queue and len(running) < max(jobs, 1):
                running.append(launch(queue.pop(0)))
            still = []
            for slot in running:
                proc = slot["proc"]
                if proc.poll() is None:
                    if time.time() - slot["started"] > timeout_s:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            proc.wait()
                        drain(slot, killed_for_timeout=True)
                        if stop_on_fail:
                            queue.clear()
                        continue
                    still.append(slot)
                else:
                    drain(slot)
                    if stop_on_fail and results[-1]["status"] == "failed":
                        queue.clear()
            running = still
            if running:
                time.sleep(0.1)
    finally:
        for slot in running:
            proc = slot["proc"]
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            proc.wait()
            drain(slot, killed_for_timeout=True)

    expected_failures_met = all(
        r.get("expectation_met", True) for r in results if "expected_failure" in r)
    any_failed = any(r["status"] == "failed" and "expected_failure" not in r
                     for r in results)
    return {
        "ok": (not any_failed) and expected_failures_met,
        "cases": len(results),
        "elapsed_s": round(time.time() - started, 2),
        "threads": RELEASE_THREADS,
        "results": sorted(results, key=lambda r: r["case"]),
    }


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="svdeblur.release",
                                 description="Run every contract in a manifest")
    ap.add_argument("manifest")
    ap.add_argument("--root", default=None,
                    help="base dir for relative paths (default: manifest's dir)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                    help="per-case wall-clock cap in seconds")
    ap.add_argument("--stop-on-fail", action="store_true")
    args = ap.parse_args(argv)
    summary = run_manifest(Path(args.manifest), root=Path(args.root) if args.root else None,
                           jobs=args.jobs, timeout_s=args.timeout,
                           stop_on_fail=args.stop_on_fail)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
