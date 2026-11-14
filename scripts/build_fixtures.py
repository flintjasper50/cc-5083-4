"""Build the release fixtures: observed images, kernel fields, task contracts.

Writes into ``release/cases`` (the four shipped cases), ``release/fixtures``
(the three declared-failure scenarios used by the clean-directory tests) and
``release/manifest.json``.  The script is idempotent: rerunning overwrites in
place and produces byte-identical JSON for unchanged inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from svdeblur.artifacts import atomic_json, atomic_png
from svdeblur.pipeline import CASES, DIAGNOSTIC_CASES
from svdeblur.server import load_sample


ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "release"


def case_contract(name: str, observed: str, kernel_field: str, out_dir: str,
                  config: dict, *, boundary: str = "reflect",
                  budget_mb: int = 512,
                  reference: str | None = None,
                  allow_signed: bool = False) -> dict:
    body = {
        "contract_version": 1,
        "name": name,
        "observed_image": observed,
        "kernel_field": kernel_field,
        "output_dir": out_dir,
        "boundary": boundary,
        "regularization": config,
        "memory_budget_mb": budget_mb,
        "threads": 1,
        "allow_signed": allow_signed,
    }
    if reference:
        body["reference_image"] = reference
    return body


def write_case(case: dict, folder: Path) -> None:
    sample = load_sample(case["name"])["sample"]
    folder.mkdir(parents=True, exist_ok=True)
    atomic_png(folder / "observed.png", sample.observed)
    atomic_png(folder / "reference.png", sample.sharp)
    atomic_json(folder / "field.json", sample.field.to_dict())
    cfg = dict(case["config"])
    cfg.pop("tol", None)
    cfg.setdefault("tol", 1e-9)
    rel = folder.relative_to(ROOT).as_posix()
    body = case_contract(
        case["name"], f"{rel}/observed.png", f"{rel}/field.json",
        f"out/release/{case['name']}",
        cfg, reference=f"{rel}/reference.png",
        allow_signed=case["kind"] in ("signed_demo",))
    atomic_json(folder / "task.json", body)


def write_manifest() -> dict:
    entries = []
    for case in CASES:
        entries.append({
            "name": case["name"],
            "contract": f"release/cases/{case['name']}/task.json",
            "out_dir": f"out/release/{case['name']}",
            "save_npy": True,
        })
    manifest = {
        "manifest_version": 1,
        "threads": 1,
        "note": "All cases share the locked single-thread policy; out_dir is "
                "resolved from the repository root.",
        "cases": entries,
    }
    atomic_json(REL / "manifest.json", manifest)
    return manifest


def write_fixtures() -> list[dict]:
    """Declared-failure fixtures: each must fail at a named stage."""
    fixtures = []
    src = REL / "cases" / "gradient"

    def rel(folder: Path) -> str:
        return folder.relative_to(ROOT).as_posix()

    # 1. kernel-field version mismatch (old schema)
    folder = REL / "fixtures" / "kernel_version_mismatch"
    folder.mkdir(parents=True, exist_ok=True)
    atomic_png(folder / "observed.png",
               np.asarray(Image.open(src / "observed.png")) / 255.0)
    old = json.loads((src / "field.json").read_text())
    old["field_version"] = 1
    atomic_json(folder / "field.json", old)
    body = case_contract("kernel_version_mismatch", f"{rel(folder)}/observed.png",
                         f"{rel(folder)}/field.json",
                         "out/fixtures/kernel_version_mismatch",
                         {"iters": 40, "sigma": 0.004, "tv_eps": 0.008})
    atomic_json(folder / "task.json", body)
    fixtures.append({"name": "kernel_version_mismatch",
                     "contract": f"{rel(folder)}/task.json",
                     "out_dir": "out/fixtures/kernel_version_mismatch",
                     "expect_failure": "kernel_field"})

    # 2. unsupported image dimensionality (RGB)
    folder = REL / "fixtures" / "rgb_image"
    folder.mkdir(parents=True, exist_ok=True)
    gray = Image.open(src / "observed.png").convert("L")
    rgb = Image.merge("RGB", (gray, gray, gray))
    rgb.save(folder / "observed.png", format="PNG")
    atomic_json(folder / "field.json", json.loads((src / "field.json").read_text()))
    body = case_contract("rgb_image", f"{rel(folder)}/observed.png",
                         f"{rel(folder)}/field.json", "out/fixtures/rgb_image",
                         {"iters": 40, "sigma": 0.004, "tv_eps": 0.008})
    atomic_json(folder / "task.json", body)
    fixtures.append({"name": "rgb_image",
                     "contract": f"{rel(folder)}/task.json",
                     "out_dir": "out/fixtures/rgb_image",
                     "expect_failure": "image"})

    # 3. missing kernel file
    folder = REL / "fixtures" / "missing_kernel_file"
    folder.mkdir(parents=True, exist_ok=True)
    atomic_png(folder / "observed.png",
               np.asarray(Image.open(src / "observed.png")) / 255.0)
    body = case_contract("missing_kernel_file", f"{rel(folder)}/observed.png",
                         f"{rel(folder)}/does_not_exist.json",
                         "out/fixtures/missing_kernel_file",
                         {"iters": 40, "sigma": 0.004, "tv_eps": 0.008})
    atomic_json(folder / "task.json", body)
    fixtures.append({"name": "missing_kernel_file",
                     "contract": f"{rel(folder)}/task.json",
                     "out_dir": "out/fixtures/missing_kernel_file",
                     "expect_failure": "kernel_field"})

    # 4. memory budget too small
    folder = REL / "fixtures" / "memory_budget"
    folder.mkdir(parents=True, exist_ok=True)
    atomic_png(folder / "observed.png",
               np.asarray(Image.open(src / "observed.png")) / 255.0)
    atomic_json(folder / "field.json", json.loads((src / "field.json").read_text()))
    body = case_contract("memory_budget", f"{rel(folder)}/observed.png",
                         f"{rel(folder)}/field.json", "out/fixtures/memory_budget",
                         {"iters": 40, "sigma": 0.004, "tv_eps": 0.008},
                         budget_mb=16)
    atomic_json(folder / "task.json", body)
    fixtures.append({"name": "memory_budget",
                     "contract": f"{rel(folder)}/task.json",
                     "out_dir": "out/fixtures/memory_budget",
                     "expect_failure": "memory_budget"})

    atomic_json(REL / "fixtures" / "manifest.json", {
        "manifest_version": 1,
        "note": "Each case must fail at the named stage; the batch runner "
                "reports expectation_met for every entry.",
        "cases": fixtures,
    })
    return fixtures


def main(argv=None) -> int:
    for case in CASES:
        write_case(case, REL / "cases" / case["name"])
    manifest = write_manifest()
    fixtures = write_fixtures()
    print(json.dumps({"cases": [c["name"] for c in manifest["cases"]],
                      "fixtures": [f["name"] for f in fixtures]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
