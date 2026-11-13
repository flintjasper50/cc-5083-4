"""Build the shipped dataset and benchmark artefacts.

Usage:
    python -m svdeblur.pipeline --out out

Writes, for the parameter set declared in CASES:
  data/<case>_{sharp,blurred,observed}.png
  out/<case>_restored.png, <case>_residual.png, <case>_kernel_field.png,
  out/<case>_curves.png, out/<case>_report.json
  out/benchmark.json  (all cases, one table)
  out/manifest.json   (sha256 of every artefact + full parameter set)

Re-running with the same arguments reproduces byte-identical PNGs and JSON
(no timestamps, fixed seeds, fixed iteration counts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .field import KernelField
from .operator import SVForward, apply_kernel_at, extend
from .solve import SolveConfig, TVSolver, residual_stats
from .synth import generate_sample, psnr, save_png, to_uint8


CASES = [
    {
        "name": "gradient",
        "texture": "mixed", "size": (192, 192), "grid": (3, 3), "support": (11, 11),
        "kind": "gaussian_gradient", "sigma": 0.004, "seed": 0,
        "config": dict(iters=300, lam_rel=0.02, tv_eps=0.008, sigma=0.004, tol=1e-9),
    },
    {
        "name": "rotation",
        "texture": "text", "size": (192, 192), "grid": (2, 3), "support": (13, 13),
        "kind": "motion_rotation", "sigma": 0.004, "seed": 1,
        "config": dict(iters=300, lam_rel=0.02, tv_eps=0.010, sigma=0.004, tol=1e-9),
    },
    {
        "name": "mixed",
        "texture": "rings", "size": (176, 176), "grid": (3, 3), "support": (9, 9),
        "kind": "mixed_modes", "sigma": 0.004, "seed": 2,
        "config": dict(iters=300, lam_rel=0.02, tv_eps=0.008, sigma=0.004, tol=1e-9),
    },
    {
        "name": "texture",
        "texture": "checker", "size": (160, 160), "grid": (4, 4), "support": (9, 9),
        "kind": "checker_swap", "sigma": 0.003, "seed": 3,
        "config": dict(iters=300, lam_rel=0.02, tv_eps=0.007, sigma=0.003, tol=1e-9),
    },
]

# Diagnostic scenarios: each is a declared, reproducible failure mode that the
# editor and the pipeline surface as an actionable diagnostic.
DIAGNOSTIC_CASES = [
    {
        "name": "diag_missing",
        "texture": "bars", "size": (160, 160), "grid": (3, 3), "support": (9, 9),
        "kind": "gaussian_gradient", "sigma": 0.004, "seed": 4,
        "missing_cells": [(1, 1)],
        "config": dict(iters=300, lam_rel=0.02, tv_eps=0.008, sigma=0.004, tol=1e-9),
        "expected_diagnostic": "MISSING_KERNEL",
    },
    {
        "name": "diag_negative",
        "texture": "text", "size": (160, 160), "grid": (2, 2), "support": (9, 9),
        "kind": "signed_demo", "sigma": 0.004, "seed": 5,
        "config": dict(iters=300, lam_rel=0.02, tv_eps=0.010, sigma=0.004, tol=1e-9),
        "expected_diagnostic": "NEGATIVE_WEIGHT",
    },
    {
        "name": "diag_oversized",
        "texture": "rings", "size": (128, 128), "grid": (2, 2), "support": (41, 41),
        "kind": "oversized_demo", "sigma": 0.004, "seed": 6,
        "config": dict(iters=250, lam_rel=0.02, tv_eps=0.008, sigma=0.004, tol=1e-9),
        "expected_diagnostic": "SUPPORT_TOO_LARGE",
    },
]


# --------------------------------------------------------------------------- checks
def operator_consistency(field: KernelField, shape: tuple[int, int],
                         boundary: str, allow_signed: bool) -> dict:
    """Verification of the operator used for this very sample."""
    A = SVForward(field, shape, boundary=boundary, allow_signed=allow_signed)
    rng = np.random.default_rng(11)
    x = rng.standard_normal(shape)
    y = rng.standard_normal(shape)
    inner_ab = float(np.sum(A.apply(x) * y))
    inner_xy = float(np.sum(x * A.apply_adjoint(y)))
    adjoint_rel = abs(inner_ab - inner_xy) / max(abs(inner_ab), 1e-12)

    k0 = field.interpolate_at(shape[0] // 2, shape[1] // 2, shape)["kernel"]

    sums = field.kernel_sums()
    norm_dev = float(np.nanmax(np.abs(sums - 1.0)))
    return {
        "adjoint_inner_product_relative_error": adjoint_rel,
        "adjoint_exact": adjoint_rel < 1e-9,
        "max_kernel_sum_deviation": norm_dev,
        "kernels_unit_sum": norm_dev < 1e-6,
        "boundary": boundary,
        "boundary_extension_note": (
            "reflect: mirrored padding; its adjoint folds overlapping "
            "copies back (verified by the inner-product test above)"
        ),
        "centre_kernel_sum": float(k0.sum()),
    }


def impulse_equivalence(field: KernelField, shape: tuple[int, int],
                        boundary: str = "zero") -> float:
    """max |A e_m^p - k(y,x)| over pixels for a single impulse.

    A spatially varying convolution must equal, pixel by pixel, the
    convolution with the interpolated local kernel; this measures that
    equality directly on the operator that produced the sample.
    """
    H, W = shape
    A = SVForward(field, shape, boundary=boundary, allow_signed=True)
    m = (H // 2, W // 2)
    imp = np.zeros((H, W))
    imp[m] = 1.0
    out = A.apply(imp)
    kh, kw = field.support
    r = ((kh - 1) // 2, (kw - 1) // 2)
    worst = 0.0
    for p in range(max(0, m[0] - r[0] - 1), min(H, m[0] + r[0] + 2)):
        for q in range(max(0, m[1] - r[1] - 1), min(W, m[1] + r[1] + 2)):
            k = field.interpolate_at(p, q, shape)["kernel"]
            dy, dx = p - m[0], q - m[1]
            if -r[0] <= dy <= r[0] and -r[1] <= dx <= r[1]:
                worst = max(worst, abs(out[p, q] - k[dy + r[0], dx + r[1]]))
    return float(worst)


def edge_handling_report(field: KernelField, shape: tuple[int, int],
                         sharp: np.ndarray, observed: np.ndarray) -> dict:
    """RMS residual per depth-from-edge ring, for the declared boundary rule."""
    A = SVForward(field, shape, boundary="reflect", allow_signed=True)
    r = A.apply(sharp) - observed
    H, W = shape
    dist = np.minimum(np.minimum(np.arange(H)[:, None], np.arange(H)[::-1][:, None]),
                      np.minimum(np.arange(W)[None, :], np.arange(W)[::-1][None, :]))
    rings = []
    for lo, hi in [(0, 4), (4, 12), (12, 24), (10000, 10000)]:
        sel = (dist >= lo) & (dist < hi) if hi < 10000 else dist >= 24
        if sel.any():
            rings.append({"ring_px": [lo, None if hi >= 10000 else hi],
                          "rms": float(np.sqrt(np.mean(r[sel] ** 2))),
                          "pixels": int(sel.sum())})
    return {"boundary": "reflect", "residual_by_depth_from_edge": rings}


# --------------------------------------------------------------------------- plots
def kernel_field_sheet(field: KernelField, shape: tuple[int, int],
                       path: Path, title: str) -> None:
    """Grid of local PSFs with the pixel location each was sampled from."""
    gh, gw = field.grid_shape
    kh, kw = field.support
    cell = max(48, kh * 4)
    pad = 18
    width = gw * cell + 2 * pad + 200
    height = gh * cell + 2 * pad + 46
    pic = Image.new("L", (width, height), 24)
    d = ImageDraw.Draw(pic)
    d.text((pad, 8), title, fill=235)
    pos = field.cell_positions(shape)
    for i in range(gh):
        for j in range(gw):
            x0, y0 = pad + j * cell, 34 + i * cell
            if not field.mask[i, j]:
                d.rectangle([x0, y0, x0 + cell - 8, y0 + cell - 8], outline=110)
                d.line([x0, y0, x0 + cell - 8, y0 + cell - 8], fill=110)
                d.text((x0 + 4, y0 + cell // 2 - 6), "missing", fill=180)
                continue
            k = field.normalized_kernels()[i, j]
            lo, hi = float(k.min()), float(k.max())
            span = max(hi - lo, 1e-9)
            pic_k = np.clip((k - lo) / span, 0, 1)
            pic_k = np.asarray(Image.fromarray(to_uint8(pic_k)).resize((cell - 8, cell - 8)))
            pic.paste(Image.fromarray(pic_k), (x0, y0))
            d.text((x0 + 2, y0 + cell - 20),
                   f"({int(pos[i, j, 0])},{int(pos[i, j, 1])}) s={k.sum():.3f}",
                   fill=200)
    # legend
    d.text((pad + gw * cell + 12, 34), "cell -> sampled", fill=210)
    d.text((pad + gw * cell + 12, 50), "at image (y,x)", fill=170)
    d.text((pad + gw * cell + 12, 76), "s = kernel sum", fill=170)
    d.text((pad + gw * cell + 12, 102), "bilinear tent", fill=170)
    d.text((pad + gw * cell + 12, 118), "weights between", fill=170)
    d.text((pad + gw * cell + 12, 134), "control points", fill=170)
    pic.save(path, format="PNG")


def curves_plot(residuals: list[float], objectives: list[float], tvs: list[float],
                stop_iteration: int, path: Path, title: str) -> None:
    W, H = 720, 300
    pic = Image.new("L", (W, H), 18)
    d = ImageDraw.Draw(pic)
    d.text((12, 8), title, fill=235)

    def panel(series, bounds, label, color, top, height):
        d.rectangle([40, top, W - 14, top + height], outline=60)
        lo, hi = bounds
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return
        n = len(series)
        for i in range(n - 1):
            x0 = 40 + (W - 54) * i / max(n - 1, 1)
            x1 = 40 + (W - 54) * (i + 1) / max(n - 1, 1)
            v0 = height - height * (series[i] - lo) / (hi - lo)
            v1 = height - height * (series[i + 1] - lo) / (hi - lo)
            d.line([x0, top + v0, x1, top + v1], fill=color, width=1)
        d.text((4, top + height // 2 - 6), label, fill=200)
        sx = 40 + (W - 54) * (stop_iteration - 1) / max(n - 1, 1)
        d.line([sx, top, sx, top + height], fill=235)
        d.text((min(sx + 4, W - 70), top + 2), f"stop {stop_iteration}", fill=235)

    pos = residuals
    panel(pos, (min(pos), max(pos)), "res", 210, 34, 96)
    panel(objectives, (min(objectives), max(objectives)), "obj", 150, 150, 96)
    panel(tvs, (min(tvs), max(tvs)), "tv", 120, 266, 34)
    d.text((40, 268), "vertical bar = declared stop iteration", fill=200)
    pic.save(path, format="PNG")


def residual_image(observed: np.ndarray, restored: np.ndarray, field: KernelField,
                   shape: tuple[int, int], path: Path) -> dict:
    A = SVForward(field, shape, boundary="reflect", allow_signed=True)
    r = A.apply(restored) - observed
    stats = residual_stats(observed, restored, A)
    # signed residual mapped to mid-grey so sign and structure are visible
    vis = np.clip(0.5 + r * 4.0, 0, 1)
    save_png(path, vis)
    return stats


# --------------------------------------------------------------------------- runner
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def run_case(case: dict, data_dir: Path, out_dir: Path) -> dict:
    allow_signed = case["kind"] in ("signed_demo", "signed")
    H, W = case["size"]
    name = case["name"]
    prefix = f"{name}_{case['kind']}"

    sample = generate_sample(
        texture=case["texture"], size=case["size"], grid_shape=case["grid"],
        support=case["support"], kernel_kind=case["kind"], sigma=case["sigma"],
        boundary="reflect", seed=case["seed"],
        missing_cells=case.get("missing_cells"), normalized=case.get("normalized", True),
    )
    # samples must exist even for diagnostic-only cases whose field is not
    # directly invertible (missing kernels); the operator check then reports
    # why restoration is blocked instead of silently patching the field.
    for tag, img in (("sharp", sample.sharp), ("blurred", sample.blurred),
                     ("observed", sample.observed)):
        save_png(data_dir / f"{prefix}_{tag}.png", img)

    diagnostics = KernelField.summarize(
        sample.field.diagnostics((H, W), allow_signed=allow_signed))

    consistency = operator_consistency(sample.field, (H, W), "reflect", allow_signed) \
        if diagnostics["ok"] else {"skipped": "field has blocking diagnostics"}
    consistency["impulse_equivalence_max_abs"] = impulse_equivalence(
        sample.field, (H, W), boundary="reflect") if diagnostics["ok"] else None
    consistency["edge_handling"] = edge_handling_report(
        sample.field, (H, W), sample.sharp, sample.observed) if diagnostics["ok"] else None

    report: dict = {
        "case": case,
        "sample_meta": sample.meta,
        "kernel_diagnostics": diagnostics,
        "operator_consistency": consistency,
    }

    if diagnostics["ok"]:
        cfg = SolveConfig(**case["config"])
        solver = TVSolver(SVForward(sample.field, (H, W), boundary="reflect",
                                    allow_signed=allow_signed),
                          cfg, reference=sample.observed)
        result = solver.run(sample.observed)
        metrics = {
            "psnr_input_vs_sharp": psnr(sample.observed, sample.sharp),
            "psnr_blurred_vs_sharp": psnr(sample.blurred, sample.sharp),
            "psnr_restored_vs_sharp": psnr(result.image, sample.sharp),
            "gain_db": psnr(result.image, sample.sharp) - psnr(sample.observed, sample.sharp),
        }
        save_png(out_dir / f"{prefix}_restored.png", result.image)
        residual_stats_out = residual_image(sample.observed, result.image, sample.field,
                                            (H, W), out_dir / f"{prefix}_residual.png")
        kernel_field_sheet(sample.field, (H, W), out_dir / f"{prefix}_kernel_field.png",
                           f"{prefix}: interpolated kernel field")
        curves_plot(result.residuals, result.objectives, result.tv_terms,
                    result.stop_iteration, out_dir / f"{prefix}_curves.png",
                    f"{prefix}: residual / objective / TV vs iteration")
        report.update({
            "solve": result.as_metadata(),
            "metrics": metrics,
            "residual_stats": residual_stats_out,
            "artefacts": [f"{prefix}_{tag}.png"
                          for tag in ("restored", "residual", "curves", "kernel_field")],
        })
    else:
        # still draw the declared field so the failure mode is inspectable
        kernel_field_sheet(sample.field, (H, W), out_dir / f"{prefix}_kernel_field.png",
                           f"{prefix}: field with blocking diagnostics")
        report.update({
            "solve": None,
            "metrics": None,
            "restored": None,
            "blocked_reason": [d["message"] for d in diagnostics["errors"]],
            "artefacts": [f"{prefix}_kernel_field.png"],
        })

    with open(out_dir / f"{prefix}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build SVDB dataset and benchmark")
    ap.add_argument("--out", default="out", help="output directory for artefacts")
    ap.add_argument("--data", default="data", help="output directory for PNG samples")
    ap.add_argument("--only", default=None, help="comma-separated case names")
    ap.add_argument("--diagnostics", action="store_true",
                    help="also build the missing/negative/oversized diagnostic cases")
    args = ap.parse_args(argv)
    data_dir, out_dir = Path(args.data), Path(args.out)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = None if not args.only else {s.strip() for s in args.only.split(",")}
    all_cases = CASES + (DIAGNOSTIC_CASES if args.diagnostics else [])
    reports = []
    for case in all_cases:
        if wanted and case["name"] not in wanted:
            continue
        rep = run_case(case, data_dir, out_dir)
        reports.append(rep)
        if rep["metrics"]:
            m = rep["metrics"]
            print(f"[{case['name']}] {case['kind']}: input {m['psnr_input_vs_sharp']:.2f} dB "
                  f"-> restored {m['psnr_restored_vs_sharp']:.2f} dB "
                  f"(+{m['gain_db']:.2f}); stop={rep['solve']['stop_reason']} "
                  f"@ it {rep['solve']['stop_iteration']}")
        else:
            print(f"[{case['name']}] blocked by diagnostics: "
                  f"{[d['code'] for d in rep['kernel_diagnostics']['errors']]}")

    benchmark = {
        "parameter_set": [c for c in all_cases if not wanted or c["name"] in wanted],
        "results": [{k: r[k] for k in ("case", "metrics", "solve", "operator_consistency",
                                       "residual_stats", "kernel_diagnostics", "blocked_reason")
                    if k in r}
                    for r in reports],
        "note": "all rows share the declared per-case parameter sets; identifiers are "
                "texture/size/grid/support/sigma/seed, so rows are individually re-runnable",
    }
    with open(out_dir / "benchmark.json", "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, sort_keys=True)

    manifest = {"files": {}}
    for p in sorted(list(data_dir.glob("*.png")) + list(out_dir.glob("*.png")) +
                    list(out_dir.glob("*.json"))):
        if p.name == "manifest.json":
            continue
        manifest["files"][str(p.relative_to(p.parent.parent))] = sha256(p)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote {len(manifest['files'])} artefacts to {data_dir}/ and {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
