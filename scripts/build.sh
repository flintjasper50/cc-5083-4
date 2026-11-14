#!/usr/bin/env bash
# Build/self-check/batch/verify for the release.  Order matters:
#   1. import the package (applies the locked thread policy)
#   2. run a real small restoration and audit it (not just "imports work")
#   3. build the shipped fixtures if they are absent
#   4. run the batch manifest, then compare against a reference tree
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

MODE="${1:-all}"
ROOT="$SVDB_ROOT"

echo "== versions =="
python3 - <<'PY'
import numpy, scipy, PIL, sys
print(f"python {sys.version.split()[0]}")
print(f"numpy  {numpy.__version__}")
print(f"scipy  {scipy.__version__}")
print(f"Pillow {PIL.__version__}")
PY

echo "== thread policy =="
python3 - <<'PY'
from svdeblur.threads import thread_config, verify_thread_config
ok, problems = verify_thread_config()
print("locked:", ok, thread_config()["env"])
if not ok:
    raise SystemExit("thread policy not applied: " + "; ".join(problems))
PY

echo "== self-check (real small restoration) =="
python3 -m svdeblur.cli selfcheck --out "$ROOT/out/selfcheck" --keep \
  | tee "$ROOT/out/selfcheck.json"
python3 - <<'PY'
import json, os
p = os.environ["SVDB_ROOT"] + "/out/selfcheck.json"
d = json.load(open(p))
if not d.get("passed"):
    raise SystemExit(f"self-check failed at stage {d.get('failed_stage')}")
print("self-check passed:", ", ".join(d["stages"].keys()))
PY

if [[ "$MODE" == "build" || ! -f "$ROOT/release/manifest.json" ]]; then
  echo "== build fixtures =="
  python3 "$ROOT/scripts/build_fixtures.py"
fi

echo "== batch release =="
rm -rf "$ROOT/out/release"
python3 -m svdeblur.release "$ROOT/release/manifest.json" --jobs "${SVDB_JOBS:-2}" \
  | tee "$ROOT/out/release.json"
python3 - <<'PY'
import json, os
d = json.load(open(os.environ["SVDB_ROOT"] + "/out/release.json"))
if not d.get("ok"):
    raise SystemExit("batch release reported a failure")
print("batch release ok:", d["cases"], "cases in", d["elapsed_s"], "s")
PY

echo "== declared-failure fixtures =="
rm -rf "$ROOT/out/fixtures"
python3 -m svdeblur.release "$ROOT/release/fixtures/manifest.json" --jobs 2 \
  | tee "$ROOT/out/fixtures.json"
python3 - <<'PY'
import json, os
d = json.load(open(os.environ["SVDB_ROOT"] + "/out/fixtures.json"))
bad = [r["case"] for r in d["results"] if not r.get("expectation_met")]
if bad:
    raise SystemExit(f"fixture(s) did not fail at the declared stage: {bad}")
print("all fixtures failed at the declared stage")
PY

if [[ "$MODE" == "all" ]]; then
  echo "== determinism / tolerance =="
  rm -rf "$ROOT/out/release_ref"
  cp -r "$ROOT/out/release" "$ROOT/out/release_ref"
  rm -rf "$ROOT/out/release"
  python3 -m svdeblur.release "$ROOT/release/manifest.json" --jobs "${SVDB_JOBS:-2}" \
    > "$ROOT/out/release_second.json"
  for case in gradient rotation mixed texture; do
    python3 -m svdeblur.cli verify --out "$ROOT/out/release/$case" \
      --reference "$ROOT/out/release_ref/$case" --threads 1 \
      | tee "$ROOT/out/verify_$case.json"
    python3 - "$ROOT/out/verify_$case.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if not d.get("passed"):
    raise SystemExit(f"tolerance check failed for {sys.argv[1]}: {d.get('problems')}")
print("tolerance ok:", d["float_delta"], d["png_delta"])
PY
  done
fi

echo "release build finished OK"
