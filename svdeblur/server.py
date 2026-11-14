"""JSON API over the SVDB core, served with the standard library only.

Endpoints (all JSON unless noted):
  GET  /api/cases                      available built-in samples + params
  POST /api/load        {case|sample}  load sample, return arrays as PNG data URLs
  POST /api/kernel_at   {y,x}          interpolated kernel + contributing cells
  POST /api/diagnostics {field,...}    kernel-field diagnostics (live editing)
  POST /api/restore     {field,...}    run the solver; returns iterations + final
  POST /api/export      {field,...}    run once, write all artefacts for that run
  GET  /export/<name>.png|json         static artefact download
  GET  /                              the application shell

Every export is produced from the parameter set of the request that triggered
it, and a run id ties the returned images, curves, kernels and JSON report
together so the UI can assert "same run".
"""

from __future__ import annotations

import base64
import io
import json
import threading
import traceback
import uuid
from dataclasses import dataclass, field as dc_field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .field import (
    BOUNDARY_MODES, INTERP_MODES, NORMALIZATIONS, Diagnostic, KernelField,
    bilinear_weights,
)
from .operator import SVForward, apply_kernel_at
from .solve import SolveConfig, TVSolver, residual_stats, STOP_REASONS, tv_value
from .synth import TEXTURES, generate_sample, psnr, to_uint8

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
EXPORT_DIR = ROOT / "out" / "web_exports"
MAX_SIDE = 256
MAX_ITERS = 1200

_lock = threading.Lock()


# --------------------------------------------------------------------------- helpers
def png_data_url(img: np.ndarray, scale: int = 1) -> str:
    arr = to_uint8(img)
    pic = Image.fromarray(arr, mode="L")
    if scale != 1:
        pic = pic.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    pic.save(buf, format="PNG", compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def array_bounds(img: np.ndarray) -> list[float]:
    return [float(img.min()), float(img.max())]


def _json_safe(v):
    """NaN/inf are not valid JSON; encode missing numeric data as null."""
    import math
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    return v


def diag_payload(field: KernelField, shape: tuple[int, int], allow_signed: bool) -> dict:
    items = field.diagnostics(shape, allow_signed=allow_signed)
    return {
        "items": [d.__dict__ for d in items],
        "errors": [d.__dict__ for d in items if d.severity == "error"],
        "warnings": [d.__dict__ for d in items if d.severity == "warning"],
        "ok": not any(d.severity == "error" for d in items),
        "kernel_sums": _json_safe(field.kernel_sums().tolist()),
        "mask": field.mask.astype(int).tolist(),
        "field_shape": list(shape),
    }


def parse_field(raw: dict[str, Any]) -> KernelField:
    return KernelField.from_dict(raw)


_CASE_CACHE: dict[str, dict[str, Any]] = {}
_CASE_LOCK = threading.Lock()


def case_spec(name: str) -> dict[str, Any]:
    from .pipeline import CASES, DIAGNOSTIC_CASES
    for c in CASES + DIAGNOSTIC_CASES:
        if c["name"] == name:
            return c
    raise KeyError(name)


def load_sample(name: str, override: dict[str, Any] | None = None) -> dict[str, Any]:
    with _CASE_LOCK:
        if name in _CASE_CACHE and not override:
            return _CASE_CACHE[name]
    case = case_spec(name)
    kw = dict(
        texture=case["texture"], size=tuple(case["size"]), grid_shape=tuple(case["grid"]),
        support=tuple(case["support"]), kernel_kind=case["kind"],
        sigma=case["sigma"], seed=case["seed"], boundary="reflect",
        missing_cells=case.get("missing_cells"),
    )
    if override:
        for key in ("support", "grid", "sigma", "texture", "kind"):
            if override.get(key) is not None:
                kw[{"grid": "grid_shape", "texture": "texture", "kind": "kernel_kind"}.get(key, key)] = \
                    tuple(override[key]) if key == "grid" else override[key]
    sample = generate_sample(**kw)
    allow_signed = case["kind"] in ("signed_demo",) or bool(override and override.get("allow_signed"))
    payload = {
        "name": name,
        "sample": sample,
        "allow_signed": allow_signed,
        "case": case,
    }
    if not override:
        _CASE_CACHE[name] = payload
    return payload


def serialize_sample(sample) -> dict[str, Any]:
    H, W = sample.sharp.shape
    return {
        "sharp": png_data_url(sample.sharp),
        "blurred": png_data_url(sample.blurred),
        "observed": png_data_url(sample.observed),
        "shape": [H, W],
        "sigma": sample.sigma,
        "meta": sample.meta,
        "field": sample.field.to_dict(),
        "diagnostics": diag_payload(sample.field, (H, W), True),
    }


# --------------------------------------------------------------------------- services
def svc_cases() -> dict[str, Any]:
    from .pipeline import CASES, DIAGNOSTIC_CASES
    return {
        "cases": [
            {"name": c["name"], "kind": c["kind"], "texture": c["texture"],
             "size": list(c["size"]), "grid": list(c["grid"]), "support": list(c["support"]),
             "sigma": c["sigma"], "seed": c["seed"],
             "diagnostic": c.get("expected_diagnostic"),
             "config": c["config"]}
            for c in CASES + DIAGNOSTIC_CASES
        ],
        "textures": sorted(TEXTURES),
        "interpolations": list(INTERP_MODES),
        "normalizations": list(NORMALIZATIONS),
        "boundaries": list(BOUNDARY_MODES),
        "stop_reasons": STOP_REASONS,
    }


def svc_load(body: dict[str, Any]) -> dict[str, Any]:
    name = body.get("case", "gradient")
    payload = load_sample(name, body.get("override") or None)
    data = serialize_sample(payload["sample"])
    data["allow_signed"] = payload["allow_signed"]
    data["requested_case"] = name
    data["config"] = payload["case"]["config"]
    return data


def svc_kernel_at(body: dict[str, Any]) -> dict[str, Any]:
    field = parse_field(body["field"])
    shape = tuple(int(v) for v in body["shape"])
    y, x = float(body["y"]), float(body["x"])
    info = field.interpolate_at(y, x, shape)
    cell_psfs = []
    for ci, cj, w in info["weights"]:
        present = bool(field.mask[ci, cj])
        cell_psfs.append({
            "cell": [ci, cj], "weight": w, "present": present,
            "kernel": field.normalized_kernels()[ci, cj].tolist() if present else None,
            "sum": float(field.kernels[ci, cj].sum()) if present else None,
        })
    return {
        "y": info["y"], "x": info["x"],
        "weights": info["weights"],
        "kernel": info["kernel"].tolist() if not info["missing"] else None,
        "kernel_sum": info["sum"],
        "missing": info["missing"],
        "missing_cells": info["missing_cells"],
        "cells": cell_psfs,
        "interpolation": field.interpolation,
        "normalization": field.normalization,
    }


def svc_diagnostics(body: dict[str, Any]) -> dict[str, Any]:
    field = parse_field(body["field"])
    shape = tuple(int(v) for v in body.get("shape", field.support))
    return diag_payload(field, shape, bool(body.get("allow_signed")))
    # note: shape is required from the UI for the support-vs-image checks


def _config(body: dict[str, Any], sample) -> SolveConfig:
    raw = dict(body.get("config") or {})
    sigma = float(raw.pop("sigma", sample.sigma))
    iters = int(min(max(int(raw.pop("iters", 300)), 1), MAX_ITERS))
    lam = raw.pop("lam", None)
    if lam is not None:
        lam = float(lam)
        if lam < 0:
            raise ValueError("lam must be non-negative")
    lam_rel = float(raw.pop("lam_rel", 0.02))
    return SolveConfig(
        iters=iters, lam=lam, lam_rel=lam_rel,
        tv_eps=float(raw.pop("tv_eps", 0.008)),
        sigma=sigma,
        damping_mu=float(raw.pop("damping_mu", 0.0)),
        tol=float(raw.pop("tol", 1e-9)),
        noise_floor_factor=float(raw.pop("noise_floor_factor", 0.8)),
    )


def svc_restore(body: dict[str, Any]) -> dict[str, Any]:
    name = body.get("case", "gradient")
    payload = load_sample(name)
    sample = payload["sample"]
    field = parse_field(body["field"]) if body.get("field") else sample.field
    shape = tuple(int(v) for v in body.get("shape", sample.observed.shape))
    if shape != sample.observed.shape:
        raise ValueError(f"shape {shape} does not match loaded sample {sample.observed.shape}")
    allow_signed = bool(body.get("allow_signed", payload["allow_signed"]))
    diags = diag_payload(field, shape, allow_signed)
    if not diags["ok"]:
        return {"ok": False, "diagnostics": diags,
                "blocked_reason": [d["message"] for d in diags["errors"]]}

    cfg = _config(body, sample)
    snap_iters = sorted({int(v) for v in body.get("snapshots", []) if 0 < int(v) <= cfg.iters})
    cfg.snapshot_iters = tuple(snap_iters)
    forward = SVForward(field, shape, boundary=str(body.get("boundary", "reflect")),
                        allow_signed=allow_signed)
    solver = TVSolver(forward, cfg, reference=sample.observed)
    result = solver.run(sample.observed)

    # The residual shown in the window is the real forward residual of the
    # restored image under the operator that was just inverted -- not a
    # client-side difference of two PNGs.
    residual = forward.apply(result.image) - sample.observed
    residual_stats_out = residual_stats(sample.observed, result.image, forward)
    iteration_residuals = [
        {"iteration": i + 1, "residual_abs": a, "residual_relative": r}
        for i, (a, r) in enumerate(zip(result.residual_abs, result.residuals))
    ]

    run_id = uuid.uuid4().hex[:12]
    frames = []
    for k in sorted(result.snapshots):
        frames.append({"iteration": k, "image": png_data_url(result.snapshots[k])})
    info = {"iterations": len(result.residuals), "residual_abs": result.residual_abs,
            "residuals": result.residuals, "objectives": result.objectives,
            "tv": result.tv_terms, "steps": result.step_sizes,
            "noise_floor": result.noise_floor, "stop_reason": result.stop_reason,
            "stop_detail": result.stop_detail, "stop_iteration": result.stop_iteration,
            "iteration_residuals": iteration_residuals}
    payload_out = {
        "ok": True,
        "run_id": run_id,
        "restored": png_data_url(result.image),
        "observed": png_data_url(sample.observed),
        "sharp": png_data_url(sample.sharp),
        "residual": png_data_url(np.clip(0.5 + residual * 4.0, 0.0, 1.0)),
        "frames": frames,
        "series": info,
        "metrics": {"psnr_restored_vs_sharp": psnr(result.image, sample.sharp),
                    "psnr_input_vs_sharp": psnr(sample.observed, sample.sharp),
                    "residual": residual_stats_out},
        "diagnostics": diags,
        "operator": forward.describe(),
        "config": cfg.as_dict(),
    }
    with _lock:
        _LAST_RUN[run_id] = {"result": result, "field": field, "sample": sample,
                             "forward": forward, "cfg": cfg}
        if len(_LAST_RUN) > 8:
            _LAST_RUN.pop(next(iter(_LAST_RUN)))
    return payload_out


_LAST_RUN: dict[str, dict[str, Any]] = {}


def svc_export(body: dict[str, Any]) -> dict[str, Any]:
    """Run once and write every artefact of that run under the same run id."""
    resp = svc_restore(body)
    if not resp.get("ok"):
        return resp
    run_id = resp["run_id"]
    entry = _LAST_RUN[run_id]
    result, field, sample, forward, cfg = (
        entry["result"], entry["field"], entry["sample"], entry["forward"], entry["cfg"])
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"web_{run_id}"

    def save(img, tag):
        Image.fromarray(to_uint8(img), mode="L").save(
            EXPORT_DIR / f"{prefix}_{tag}.png", format="PNG", compress_level=6)

    save(result.image, "restored")
    save(sample.observed, "observed")
    save(sample.sharp, "sharp")
    # signed residual visualisation
    r = forward.apply(result.image) - sample.observed
    save(np.clip(0.5 + r * 4.0, 0, 1), "residual")
    # kernel field sheet
    _kernel_sheet(field, sample.sharp.shape, EXPORT_DIR / f"{prefix}_kernels.png")
    # curves
    _curves(result, EXPORT_DIR / f"{prefix}_curves.png")
    # local kernels at an explicit grid of probe points
    probes = []
    shape = sample.sharp.shape
    for fy in (0.1, 0.5, 0.9):
        for fx in (0.1, 0.5, 0.9):
            y, x = (shape[0] - 1) * fy, (shape[1] - 1) * fx
            probes.append({"y": y, "x": x,
                           **{k: v for k, v in svc_kernel_at(
                               {"field": field.to_dict(), "shape": list(shape),
                                "y": y, "x": x}).items() if k != "cells"}})
    report = {
        "run_id": run_id,
        "case": body.get("case", "gradient"),
        "field": field.to_dict(),
        "config": cfg.as_dict(),
        "operator": forward.describe(),
        "solve": result.as_metadata(),
        "metrics": resp["metrics"],
        "diagnostics": resp["diagnostics"],
        "probes": probes,
        "artefacts": sorted(f"{prefix}_{t}.png" for t in
                            ("restored", "observed", "sharp", "residual", "kernels", "curves")),
        "sample_meta": sample.meta,
    }
    with open(EXPORT_DIR / f"{prefix}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return {"ok": True, "run_id": run_id, "artefacts": report["artefacts"],
            "report_url": f"/export/{prefix}_report.json",
            "restored_url": f"/export/{prefix}_restored.png",
            "curves_url": f"/export/{prefix}_curves.png",
            "kernels_url": f"/export/{prefix}_kernels.png",
            "residual_url": f"/export/{prefix}_residual.png"}


# --------------------------------------------------------------------------- drawing
from PIL import ImageDraw, ImageFont


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _kernel_sheet(field: KernelField, shape, path) -> None:
    gh, gw = field.grid_shape
    kh, kw = field.support
    cell = max(46, kh * 4)
    pad = 16
    W, H = gw * cell + 2 * pad + 150, gh * cell + 2 * pad + 40
    pic = Image.new("L", (W, H), 22)
    d = ImageDraw.Draw(pic)
    d.text((pad, 8), "interpolated kernel field", fill=235)
    pos = field.cell_positions(shape)
    kn = field.normalized_kernels()
    for i in range(gh):
        for j in range(gw):
            x0, y0 = pad + j * cell, 30 + i * cell
            if not field.mask[i, j]:
                d.rectangle([x0, y0, x0 + cell - 6, y0 + cell - 6], outline=120)
                d.line([x0, y0, x0 + cell - 6, y0 + cell - 6], fill=120)
                d.text((x0 + 2, y0 + cell // 2), "missing", fill=200)
                continue
            k = kn[i, j]
            lo, hi = float(k.min()), float(k.max())
            vis = np.clip((k - lo) / max(hi - lo, 1e-9), 0, 1)
            pic_k = Image.fromarray(to_uint8(vis)).resize((cell - 6, cell - 6))
            pic.paste(pic_k, (x0, y0))
            d.text((x0 + 2, y0 + cell - 18),
                   f"({int(pos[i, j, 0])},{int(pos[i, j, 1])})", fill=210)
    d.text((pad + gw * cell + 8, 30), f"grid {gh}x{gw}", fill=200)
    d.text((pad + gw * cell + 8, 48), f"support {kh}x{kw}", fill=200)
    d.text((pad + gw * cell + 8, 66), field.interpolation, fill=200)
    d.text((pad + gw * cell + 8, 84), field.normalization, fill=200)
    pic.save(path, format="PNG")


def _curves(result, path) -> None:
    W, H = 700, 260
    pic = Image.new("L", (W, H), 18)
    d = ImageDraw.Draw(pic)
    d.text((10, 6), "residual / objective / TV", fill=235)

    def panel(series, label, color, top, height):
        d.rectangle([44, top, W - 12, top + height], outline=60)
        lo, hi = min(series), max(series)
        if hi <= lo:
            hi = lo + 1.0
        n = len(series)
        for i in range(n - 1):
            x0 = 44 + (W - 56) * i / max(n - 1, 1)
            x1 = 44 + (W - 56) * (i + 1) / max(n - 1, 1)
            v0 = height - height * (series[i] - lo) / (hi - lo)
            v1 = height - height * (series[i + 1] - lo) / (hi - lo)
            d.line([x0, top + v0, x1, top + v1], fill=color)
        d.text((4, top + height // 2 - 5), label, fill=200)
        sx = 44 + (W - 56) * (result.stop_iteration - 1) / max(n - 1, 1)
        d.line([sx, top, sx, top + height], fill=235)
        d.text((min(sx + 3, W - 68), top + 1), f"stop {result.stop_iteration}", fill=235)

    panel(result.residuals, "res", 210, 26, 96)
    panel(result.objectives, "obj", 150, 130, 96)
    panel(result.tv_terms, "tv", 120, 234, 20)
    pic.save(path, format="PNG")


# --------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "SVDB/1.0"

    def log_message(self, fmt, *args):
        pass

    def _json(self, code: int, obj: Any):
        data = json.dumps(_json_safe(obj)).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path):
        if not path.is_file():
            return self._json(404, {"error": "not found"})
        data = path.read_bytes()
        ctype = {".png": "image/png", ".json": "application/json",
                 ".html": "text/html; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8"}.get(path.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                return self._file(WEB / "index.html")
            if self.path.startswith("/static/"):
                return self._file(WEB / self.path[len("/static/"):])
            if self.path.startswith("/export/"):
                name = Path(self.path[len("/export/"):]).name
                return self._file(EXPORT_DIR / name)
            if self.path.startswith("/data/"):
                name = Path(self.path[len("/data/"):]).name
                return self._file(ROOT / "data" / name)
            if self.path == "/api/cases":
                return self._json(200, svc_cases())
            return self._json(404, {"error": f"no route {self.path}"})
        except Exception:
            return self._json(500, {"error": traceback.format_exc()})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            routes = {
                "/api/load": svc_load,
                "/api/kernel_at": svc_kernel_at,
                "/api/diagnostics": svc_diagnostics,
                "/api/restore": svc_restore,
                "/api/export": svc_export,
            }
            fn = routes.get(self.path)
            if fn is None:
                return self._json(404, {"error": f"no route {self.path}"})
            return self._json(200, fn(body))
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except Exception:
            return self._json(500, {"error": traceback.format_exc()})


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    if not (WEB / "index.html").is_file():
        raise SystemExit(f"UI shell missing at {WEB / 'index.html'}")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"SVDB restoration window: http://{host}:{port}/  (Ctrl-C to stop)")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    serve(a.host, a.port)
