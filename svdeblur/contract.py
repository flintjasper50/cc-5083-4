"""Declared input contract for a restoration task.

A task is a JSON file with four explicit inputs: the observed image, the
kernel-field file, the regularisation parameters and the output directory.
:func:`load_contract` turns that file into a validated :class:`CaseContract`
and raises :class:`ContractError` at a named *stage* when an input is
missing, malformed, unsupported or over the memory budget.  Nothing is
silently patched: a missing kernel cell or a non-grayscale/non-uint8 image
is a hard failure with the stage recorded.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .field import KernelField, BOUNDARY_MODES
from .threads import verify_thread_config, RELEASE_THREADS

CONTRACT_VERSION = 1
SUPPORTED_IMAGE_MODE = "L"
SUPPORTED_IMAGE_DTYPE = "uint8"
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16

# Working-set model used again by the runner; kept here so the contract and
# the executor speak the same units.
BYTES_PER_FLOAT64 = 8


class ContractError(RuntimeError):
    """A task input is unusable; ``stage`` says which one and ``code`` why."""

    def __init__(self, stage: str, code: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"stage": self.stage, "code": self.code, "message": self.message}


@dataclass
class CaseContract:
    name: str
    observed_image: Path
    kernel_field: Path
    output_dir: Path
    boundary: str
    regularization: dict[str, Any]
    memory_budget_mb: int = 512
    threads: int = RELEASE_THREADS
    expected_failure: str | None = None  # stage name for documented fixtures
    source: Path | None = None
    image_shape: tuple[int, int] = (0, 0)
    image: np.ndarray | None = None
    field: KernelField | None = None
    reference_sharp: np.ndarray | None = None
    diagnostics: list[dict[str, Any]] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)
    estimated_bytes: int = 0

    # ------------------------------------------------------------- reporting
    def resolved_params(self) -> dict[str, Any]:
        reg = dict(self.regularization)
        if reg.get("lam") is None:
            reg["lam_resolved"] = float(reg.get("lam_rel", 0.02)) * float(reg["sigma"])
        else:
            reg["lam_resolved"] = float(reg["lam"])
        return {
            "contract_version": CONTRACT_VERSION,
            "case": self.name,
            "boundary": self.boundary,
            "regularization": reg,
            "memory_budget_mb": self.memory_budget_mb,
            "threads": self.threads,
            "observed_image": str(self.observed_image),
            "kernel_field": str(self.kernel_field),
            "image_shape": list(self.image_shape),
            "kernel_grid": list(self.field.grid_shape) if self.field else None,
            "kernel_support": list(self.field.support) if self.field else None,
            "kernel_normalization": self.field.normalization if self.field else None,
            "estimated_working_set_mb": round(self.estimated_bytes / 2 ** 20, 3),
        }


# ------------------------------------------------------------------ helpers
def _require(body: dict[str, Any], key: str, stage: str, sample: str) -> Any:
    if key not in body:
        raise ContractError(stage, "MISSING_FIELD", f"contract is missing '{key}' (e.g. {sample})")
    return body[key]


def load_observed_image(path: Path, expected_dtype: str = SUPPORTED_IMAGE_DTYPE,
                        expected_channels: int = 1) -> np.ndarray:
    """Load a grayscale uint8 image as float64 in [0, 1] or fail explicitly.

    Unsupported dimensionality (colour, alpha, palette, non-2-D, wrong dtype)
    raises :class:`ContractError` with stage ``image``.
    """
    if not path.is_file():
        raise ContractError("image", "IMAGE_MISSING", f"observed image not found: {path}")
    try:
        with Image.open(path) as pic:
            mode = pic.mode
            size = pic.size
            pic = pic.copy()
    except Exception as exc:  # unreadable / truncated / not an image
        raise ContractError("image", "IMAGE_UNREADABLE", f"cannot decode {path}: {exc}") from exc

    if mode != SUPPORTED_IMAGE_MODE:
        raise ContractError(
            "image", "UNSUPPORTED_IMAGE_DIMENSION",
            f"{path.name}: image mode {mode!r} is not single-channel 8-bit "
            f"({expected_channels} channel(s)); the contract accepts mode "
            f"{SUPPORTED_IMAGE_MODE!r} only. Convert with "
            f"'Image.open(...).convert(\"L\")' deliberately before running.")
    arr = np.asarray(pic)
    if arr.ndim != 2:
        raise ContractError(
            "image", "UNSUPPORTED_IMAGE_DIMENSION",
            f"{path.name}: decoded array has shape {arr.shape}; expected a 2-D "
            f"single-plane image (H, W).")
    if str(arr.dtype) != expected_dtype:
        raise ContractError(
            "image", "UNSUPPORTED_IMAGE_DTYPE",
            f"{path.name}: pixel dtype {arr.dtype} is not the declared "
            f"{expected_dtype!r}.")
    H, W = arr.shape
    if H < MIN_IMAGE_SIDE or W < MIN_IMAGE_SIDE:
        raise ContractError(
            "image", "UNSUPPORTED_IMAGE_SIZE",
            f"{path.name}: {W}x{H} is below the supported minimum side of "
            f"{MIN_IMAGE_SIDE}px (a varying field cannot be resolved).")
    if H > MAX_IMAGE_SIDE or W > MAX_IMAGE_SIDE:
        raise ContractError(
            "image", "UNSUPPORTED_IMAGE_SIZE",
            f"{path.name}: {W}x{H} exceeds the supported maximum side of "
            f"{MAX_IMAGE_SIDE}px; tile the problem instead of raising the budget.")
    if size != (W, H):
        raise ContractError("image", "UNSUPPORTED_IMAGE_DIMENSION",
                            f"{path.name}: decoded array {W}x{H} disagrees with header {size}")
    return arr.astype(np.float64) / 255.0


def estimate_working_set_bytes(field: KernelField, shape: tuple[int, int],
                               snapshot_count: int, fft_factor: float = 3.0) -> int:
    """Conservative peak working set for one restoration.

    Model: the active tent-weight maps (one H*W float64 per active cell),
    the solver's ten persistent H*W arrays, the snapshot copies, and an
    ``fft_factor`` multiplier for the extended buffers and FFT workspace
    allocated transiently by ``fftconvolve``.  This is a floor, not a
    guarantee; the runner also converts an allocation :class:`MemoryError`
    into an explicit budget failure.
    """
    H, W = shape
    pixels = H * W
    active = int(field.mask.sum())
    persistent = active + 10 + max(snapshot_count, 1)
    return int(BYTES_PER_FLOAT64 * pixels * (persistent * fft_factor))


def _parse_regularization(raw: Any) -> dict[str, Any]:
    stage = "regularization"
    if not isinstance(raw, dict):
        raise ContractError(stage, "BAD_REGULARIZATION", "regularization must be an object")
    needed = ("iters", "sigma", "tv_eps")
    for key in needed:
        if key not in raw:
            raise ContractError(stage, "MISSING_FIELD",
                                f"regularization is missing '{key}'")
    iters = int(raw["iters"])
    sigma = float(raw["sigma"])
    tv_eps = float(raw["tv_eps"])
    lam = raw.get("lam", None)
    lam_rel = float(raw.get("lam_rel", 0.02))
    damping_mu = float(raw.get("damping_mu", 0.0))
    tol = float(raw.get("tol", 1e-9))
    noise_floor_factor = float(raw.get("noise_floor_factor", 0.8))
    if iters < 1:
        raise ContractError(stage, "BAD_REGULARIZATION", f"iters={iters} must be >= 1")
    if not sigma > 0 or not np.isfinite(sigma):
        raise ContractError(stage, "BAD_REGULARIZATION", f"sigma={sigma} must be positive and finite")
    if not tv_eps > 0 or not np.isfinite(tv_eps):
        raise ContractError(stage, "BAD_REGULARIZATION", f"tv_eps={tv_eps} must be positive")
    if lam is not None:
        lam = float(lam)
        if not np.isfinite(lam) or lam < 0:
            raise ContractError(stage, "BAD_REGULARIZATION", f"lam={lam} must be finite and >= 0")
    if not 0.0 <= damping_mu:
        raise ContractError(stage, "BAD_REGULARIZATION", f"damping_mu={damping_mu} must be >= 0")
    return {
        "iters": iters, "sigma": sigma, "tv_eps": tv_eps, "lam": lam,
        "lam_rel": lam_rel, "damping_mu": damping_mu, "tol": tol,
        "noise_floor_factor": noise_floor_factor,
    }


def load_kernel_field(path: Path, shape: tuple[int, int],
                      allow_signed: bool) -> tuple[KernelField, list[dict[str, Any]]]:
    """Load a kernel-field JSON and reject version drift / holes / bad support."""
    stage = "kernel_field"
    if not path.is_file():
        raise ContractError(stage, "KERNEL_FIELD_MISSING", f"kernel field not found: {path}")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractError(stage, "KERNEL_FIELD_UNREADABLE",
                            f"cannot parse kernel field {path}: {exc}") from exc
    if not isinstance(body, dict):
        raise ContractError(stage, "KERNEL_FIELD_UNREADABLE", "kernel field JSON must be an object")
    version = body.get("field_version")
    if version is None:
        raise ContractError(
            stage, "KERNEL_FIELD_VERSION",
            f"{path.name}: no 'field_version'; re-export the field from the "
            f"v{CONTRACT_VERSION} release (expected field_version 2).")
    if int(version) != 2:
        raise ContractError(
            stage, "KERNEL_FIELD_VERSION",
            f"{path.name}: field_version {version} is not the supported value 2 "
            f"(version mismatch); regenerate the field with the release editor.")
    try:
        field = KernelField.from_dict(body)
    except Exception as exc:
        raise ContractError(stage, "KERNEL_FIELD_MALFORMED", f"{path.name}: {exc}") from exc

    diags = field.diagnostics(shape, allow_signed=allow_signed)
    blocking = [d for d in diags if d.severity == "error"]
    if blocking:
        codes = sorted({d.code for d in blocking})
        raise ContractError(
            stage, codes[0],
            f"{path.name}: " + "; ".join(d.message for d in blocking))
    return field, [d.__dict__ for d in diags]


def check_output_dir(path: Path) -> None:
    """Fail before any work if the output directory cannot be written."""
    stage = "output"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ContractError(stage, "OUTPUT_UNWRITABLE", f"cannot create {path}: {exc}") from exc
    probe = path / ".write_probe"
    try:
        probe.write_text("probe", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise ContractError(stage, "OUTPUT_READONLY",
                            f"output directory {path} is not writable: {exc}") from exc
    finally:
        if probe.exists():
            try:
                probe.unlink()
            except OSError:
                pass


def load_contract(path: Path, *, root: Path | None = None,
                  expected_failure: str | None = None) -> CaseContract:
    """Parse and validate a task contract; raises on the first failed stage."""
    root = (root or Path.cwd()).resolve()
    stage = "contract"
    if not path.is_file():
        raise ContractError(stage, "CONTRACT_MISSING", f"contract file not found: {path}")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractError(stage, "CONTRACT_UNREADABLE", f"cannot parse {path}: {exc}") from exc

    v = body.get("contract_version", CONTRACT_VERSION)
    if int(v) != CONTRACT_VERSION:
        raise ContractError(stage, "CONTRACT_VERSION",
                            f"{path.name}: contract_version {v} != {CONTRACT_VERSION}")

    name = str(_require(body, "name", stage, "gradient"))
    obs_rel = _require(body, "observed_image", stage, "cases/<name>_observed.png")
    kf_rel = _require(body, "kernel_field", stage, "kernels/<name>.json")
    out_rel = _require(body, "output_dir", stage, "out/release")
    boundary = str(body.get("boundary", "reflect"))
    if boundary not in BOUNDARY_MODES:
        raise ContractError(stage, "BAD_BOUNDARY",
                            f"boundary {boundary!r} not in {BOUNDARY_MODES}")
    allow_signed = bool(body.get("allow_signed", False))
    budget = int(body.get("memory_budget_mb", 512))
    if budget < 16:
        raise ContractError("memory_budget", "BAD_BUDGET", "memory_budget_mb must be >= 16")
    threads = int(body.get("threads", RELEASE_THREADS))
    if threads != RELEASE_THREADS:
        raise ContractError(
            "threads", "UNSUPPORTED_THREADS",
            f"contract requests {threads} threads; the release is locked to "
            f"{RELEASE_THREADS} to keep numerical results reproducible")
    reg = _parse_regularization(_require(body, "regularization", stage,
                                         '{"iters":300,"sigma":0.004,"tv_eps":0.008}'))

    observed_path = (root / obs_rel).resolve()
    kernel_path = (root / kf_rel).resolve()
    out_dir = (root / out_rel).resolve()

    image = load_observed_image(observed_path)
    shape = tuple(int(v) for v in image.shape)

    reference = None
    if body.get("reference_image"):
        try:
            reference = load_observed_image((root / body["reference_image"]).resolve())
        except ContractError as exc:
            raise ContractError("image", exc.code,
                                f"reference_image: {exc.message}") from exc
        if reference.shape != shape:
            raise ContractError(
                "image", "REFERENCE_SHAPE_MISMATCH",
                f"reference_image shape {reference.shape} != observed {shape}")

    field, diags = load_kernel_field(kernel_path, shape, allow_signed)

    snapshots = reg.get("snapshots") or 0
    estimated = estimate_working_set_bytes(field, shape, int(snapshots))
    if estimated > budget * 2 ** 20:
        raise ContractError(
            "memory_budget", "MEMORY_BUDGET_EXCEEDED",
            f"{name}: estimated working set {estimated / 2 ** 20:.1f} MB exceeds "
            f"memory_budget_mb={budget}. Lower the kernel grid/support or image "
            f"size, reduce snapshots, or raise memory_budget_mb deliberately.")

    check_output_dir(out_dir)

    ok, problems = verify_thread_config()
    if not ok:
        raise ContractError("threads", "THREAD_SETTING",
                            "thread policy not applied: " + "; ".join(problems))

    warnings = [d["message"] for d in diags if d["severity"] == "warning"]
    return CaseContract(
        name=name, observed_image=observed_path, kernel_field=kernel_path,
        output_dir=out_dir, boundary=boundary, regularization=reg,
        memory_budget_mb=budget, threads=threads,
        expected_failure=expected_failure, source=path.resolve(),
        image_shape=shape, image=image, field=field, reference_sharp=reference,
        diagnostics=diags, warnings=warnings, estimated_bytes=estimated)


def write_run_record(record: dict[str, Any], out_dir: Path) -> None:
    """Write a run record atomically so a crashed run cannot look finished."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "run.json"
    tmp = out_dir / ".run.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)
