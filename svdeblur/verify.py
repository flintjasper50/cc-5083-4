"""Compare two completed release runs under a declared tolerance.

The comparison is on full-resolution float arrays (``restored.npz``, written
with ``--save-npy``) as well as the published 8-bit PNGs.  Tolerance is
declared, not tuned after the fact:

* ``float_delta``   — max |a-b| on the float64 restoration.  Source: the two
  runs execute the identical deterministic instruction stream with one BLAS
  worker each, so the only expected difference is FMA/reduction ordering
  inside LAPACK/FFT.  Allowed 1e-10 (≈ 2.6e-8 of an 8-bit level).
* ``png_delta``     — after quantising to uint8 both runs must be identical
  unless ``--allow-png-delta`` is passed; a quantisation tie at a rounding
  boundary is documented rather than hidden.
* ``residual_delta``— max relative difference of the independently recomputed
  forward residual RMS; allowed 1e-9.

The verifier refuses to pass if either run downsampled its outputs: it checks
that ``restored.png`` has the same dimensions as ``observed.png`` for its own
run, so shrinking an image cannot be used to hide instability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

FLOAT_DELTA = 1e-10
RESIDUAL_REL_DELTA = 1e-9
PNG_DELTA = 0


def _load_run(out_dir: Path) -> dict[str, Any]:
    report_path = out_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"{out_dir} has no report.json (incomplete run)")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise RuntimeError(f"{out_dir} status is {report.get('status')!r}, not completed")
    npy = out_dir / "restored.npz"
    if not npy.is_file():
        raise FileNotFoundError(f"{out_dir} has no restored.npz; rerun with --save-npy")
    data = np.load(npy)
    return {
        "report": report,
        "restored": data["restored"],
        "residual": data["residual"],
        "png": np.asarray(Image.open(out_dir / "restored.png")),
    }


def verify_release(candidate: Path, reference: Path, threads: int = 1) -> dict[str, Any]:
    result: dict[str, Any] = {
        "threads": threads,
        "tolerance": {
            "float_delta_max": FLOAT_DELTA,
            "float_delta_source": "BLAS/FFT reduction-order only; runs are "
                                  "single-threaded and share the same code path",
            "png_delta_max": PNG_DELTA,
            "png_delta_source": "uint8 quantisation of a delta below 1e-7 of a level",
            "residual_relative_delta_max": RESIDUAL_REL_DELTA,
        },
        "passed": False,
    }
    try:
        ref = _load_run(reference)
        cand = _load_run(candidate)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    problems: list[str] = []
    if ref["restored"].shape != cand["restored"].shape:
        problems.append(f"shape mismatch: {ref['restored'].shape} vs "
                        f"{cand['restored'].shape}")
        result["problems"] = problems
        return result

    # Full-resolution requirement: no silent downsampling.
    for label, run in (("reference", ref), ("candidate", cand)):
        shape = tuple(run["report"].get("params", {}).get("image_shape", []))
        if shape and tuple(run["restored"].shape) != shape:
            problems.append(f"{label} restored.npz {run['restored'].shape} != "
                            f"observed {shape} (downsampled?)")
        if run["png"].shape != tuple(shape):
            problems.append(f"{label} restored.png {run['png'].shape} != observed {shape}")

    float_delta = float(np.max(np.abs(cand["restored"] - ref["restored"])))
    if float_delta > FLOAT_DELTA:
        problems.append(f"float delta {float_delta:.3e} > {FLOAT_DELTA:.0e}")

    png_delta = int(np.max(np.abs(cand["png"].astype(int) - ref["png"].astype(int))))
    if png_delta > PNG_DELTA:
        problems.append(f"png delta {png_delta} > {PNG_DELTA}")

    ref_rms = float(np.sqrt(np.mean(ref["residual"] ** 2)))
    cand_rms = float(np.sqrt(np.mean(cand["residual"] ** 2)))
    residual_rel = abs(cand_rms - ref_rms) / max(ref_rms, 1e-15)
    if residual_rel > RESIDUAL_REL_DELTA:
        problems.append(f"residual RMS relative delta {residual_rel:.3e} > "
                        f"{RESIDUAL_REL_DELTA:.0e}")

    result.update({
        "float_delta": float_delta,
        "png_delta": png_delta,
        "residual_rms": {"reference": ref_rms, "candidate": cand_rms,
                         "relative_delta": residual_rel},
        "shape": list(cand["restored"].shape),
        "problems": problems,
        "passed": not problems,
    })
    return result
