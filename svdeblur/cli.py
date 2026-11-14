"""Command-line aid for the spatially varying deblurring window.

The release's primary interface is the graphical *变化去模糊* (change-aware
deblurring) window started by ``scripts/start_ui.sh``.  These commands exist to
prepare inputs, run batch jobs and validate a release; they never replace the
GUI.  Every failure prints the stage that failed and exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

import numpy as np

from .contract import (ContractError, load_contract, write_run_record)
from .runner import record_load_failure, run_contract


def _fail(exc: ContractError) -> int:
    print(json.dumps({"ok": False, "stage": exc.stage, "code": exc.code,
                      "error": exc.message}, indent=2))
    return 2


def cmd_run(args) -> int:
    try:
        root = Path(args.root) if args.root else Path.cwd()
        contract = load_contract(Path(args.contract), root=root)
    except ContractError as exc:
        record_load_failure(Path(args.contract), exc,
                            root=Path(args.root) if args.root else Path.cwd())
        return _fail(exc)

    stop = {"requested": False, "signal": None}

    def cancel() -> bool:
        if args.cancel_file and Path(args.cancel_file).exists():
            stop["requested"] = True
        return stop["requested"]

    def on_signal(signum, _frame):
        # Co-operative: the solver checks cancel() each iteration and unwinds
        # through the normal exception path, which records status=cancelled and
        # removes any restored image before the process exits.
        stop["requested"] = True
        stop["signal"] = signum

    old = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            old[sig] = signal.signal(sig, on_signal)
    try:
        outcome = run_contract(contract, cancel=cancel,
                               log=lambda m: print(m, flush=True),
                               save_npy=args.save_npy)
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
    print(json.dumps({"ok": outcome.ok, "status": outcome.status,
                      "stage": outcome.record.get("stage"),
                      "output_dir": str(outcome.out_dir),
                      "run_id": outcome.record.get("run_id")}, indent=2))
    return 0 if outcome.ok else 1


def cmd_selfcheck(args) -> int:
    from .selfcheck import run_selfcheck
    result = run_selfcheck(out_dir=Path(args.out), keep=args.keep)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def cmd_verify(args) -> int:
    from .verify import verify_release
    result = verify_release(Path(args.out), Path(args.reference),
                            threads=int(args.threads))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def cmd_export_field(args) -> int:
    """Export the field of a shipped sample as a contract-ready JSON."""
    from .pipeline import CASES, DIAGNOSTIC_CASES
    from .server import load_sample
    cases = {c["name"]: c for c in CASES + DIAGNOSTIC_CASES}
    if args.case not in cases:
        print(f"unknown case {args.case!r}; choose from {sorted(cases)}", file=sys.stderr)
        return 2
    payload = load_sample(args.case)
    body = payload["sample"].field.to_dict()
    body["field_version"] = 2
    body["source_case"] = args.case
    body["normalization"] = payload["sample"].field.normalization
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out} ({args.case})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="svdeblur.cli",
                                 description="Batch aid for the deblurring window")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="execute one task contract")
    p.add_argument("contract")
    p.add_argument("--root", default=None,
                   help="base dir for relative paths (default: the contract's directory)")
    p.add_argument("--save-npy", action="store_true",
                   help="also write restored.npz (float arrays) for tolerance checks")
    p.add_argument("--cancel-file", default=None,
                   help="cancel cleanly (status=cancelled, no restored image) "
                        "when this path appears")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("selfcheck", help="run a small real restoration end to end")
    p.add_argument("--out", default="out/selfcheck")
    p.add_argument("--keep", action="store_true", help="keep the self-check artefacts")
    p.set_defaults(fn=cmd_selfcheck)

    p = sub.add_parser("verify", help="compare two completed runs for numerical tolerance")
    p.add_argument("--out", required=True, help="output tree of the candidate release")
    p.add_argument("--reference", required=True, help="output tree of the reference release")
    p.add_argument("--threads", type=int, default=1)
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("export-field", help="write a shipped sample's kernel field as JSON")
    p.add_argument("--case", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_export_field)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
