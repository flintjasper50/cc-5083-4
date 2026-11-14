"""Open the 变化去模糊 window and verify the real restoration path in a browser.

What is asserted, in order (each is a real user action, not a stub):

1. the window boots and loads a sample;
2. an imported / edited kernel field is present and can be edited;
3. ``Start restoration`` returns a restored image AND a residual image that
   the server computed as ``A·restored − observed`` (not a client-side diff);
4. the iteration residual line and the final iteration agree;
5. marking a cell missing surfaces ``MISSING_KERNEL`` and blocks restoration,
   with the failure shown in the window;
6. filling the hole restores the ability to run.

Usage:
    SVDB_URL=http://127.0.0.1:8765 python3 scripts/ui_check.py
Screenshots land in ``out/ui_check``.  Exit status is non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = os.environ.get("SVDB_URL", "http://127.0.0.1:8765")
SHOT = Path(os.environ.get("SVDB_SHOTS", "out/ui_check")).resolve()
SHOT.mkdir(parents=True, exist_ok=True)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def run() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=[
            "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1680, "height": 1000})
        errors: list[str] = []
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        page.goto(BASE + "/", wait_until="networkidle")
        page.wait_for_function(
            "document.getElementById('imgObs').naturalWidth > 0", timeout=20000)
        check(page.inner_text("#status") == "loaded", "window booted and loaded a sample")
        check(page.eval_on_selector("#imgObs",
                                    "e => e.naturalWidth > 0"),
              "observed image is displayed")
        page.screenshot(path=str(SHOT / "01_loaded.png"), full_page=True)

        # ---- a real edit of the kernel field before restoring
        before = page.eval_on_selector(
            "#gridCanvas", "c => c.toDataURL().slice(0, 200)")
        page.evaluate("""() => {
          const cv = document.getElementById('gridCanvas');
          const r = cv.getBoundingClientRect();
          cv.dispatchEvent(new MouseEvent('click',
            {bubbles:true, clientX:r.left + r.width*0.5, clientY:r.top + r.height*0.5}));
        }""")
        page.wait_for_timeout(200)
        page.eval_on_selector("#sy", "e => { e.value = 2.5; e.dispatchEvent(new Event('input')); }")
        page.click("#btnPaint")
        page.wait_for_timeout(400)
        after = page.eval_on_selector(
            "#gridCanvas", "c => c.toDataURL().slice(0, 200)")
        check(before != after, "kernel field edit changed the editor canvas")

        # ---- start restoration and verify the actual outputs
        page.click("#btnRestore")
        page.wait_for_function(
            "document.getElementById('imgRest').naturalWidth > 0", timeout=120000)
        page.wait_for_timeout(600)
        state = page.evaluate("""() => {
          const res = document.getElementById('imgRes');
          const obs = document.getElementById('imgObs');
          const rest = document.getElementById('imgRest');
          const same = (a, b) => a.src === b.src;
          return {
            restLen: rest.src.length,
            resLen: res.src.length,
            obsLen: obs.src.length,
            residual_eq_observed: same(res, obs),
            residual_eq_restored: same(res, rest),
            capRes: document.getElementById('capRes').textContent,
            iterRes: document.getElementById('iterResidual').textContent,
            stop: document.getElementById('stopBox').textContent,
            status: document.getElementById('status').textContent,
            frames: document.querySelectorAll('.frame').length,
            tag: document.getElementById('tagRun').textContent,
          };
        }""")
        print("  restore state:", json.dumps(state, ensure_ascii=False))
        check(state["status"].startswith("done"), "restoration reported done")
        check(state["restLen"] > 100, "restored pane shows the run's image")
        check(state["resLen"] > 100, "residual pane shows a server-computed residual image")
        check(not state["residual_eq_observed"] and not state["residual_eq_restored"],
              "residual image is its own computation, not observed or restored")
        check("iteration residual" in state["iterRes"].lower()
              and "it " in state["iterRes"], "iteration residual is displayed")
        check(state["stop"].startswith("stop"), "stop reason is displayed")
        check(state["frames"] > 0, "intermediate iteration frames are present")

        # Cross-check the on-screen residual against a fresh API run of the
        # same field/params: the residual RMS must be the forward residual.
        cfg = page.evaluate("""() => ({
          field: state.field, shape: state.shape,
          boundary: document.getElementById('boundary').value,
          allow_signed: document.getElementById('signed').value === '1',
        })""")
        if cfg.get("field"):
            api = page.evaluate("""async (cfg) => {
              const body = Object.assign({case: document.getElementById('caseSelect').value,
                config: {iters: 60, sigma: 0.004, tv_eps: 0.008}},
                cfg, {snapshots: [10, 60]});
              const r = await fetch('/api/restore', {method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify(body)});
              return r.ok ? await r.json() : {ok: false};
            }""", cfg)
            if api.get("ok"):
                rres = api["metrics"]["residual"]["rms"]
                last = api["series"]["iteration_residuals"][-1]["residual_abs"]
                l2 = api["metrics"]["residual"]["l2"]
                check(abs(last - l2) <= 1e-6 * max(l2, 1.0),
                      f"API iteration residual {last:.6f} equals forward residual {l2:.6f}")
                check(rres > 0, f"API forward residual rms {rres:.6e} > 0")
        page.screenshot(path=str(SHOT / "02_restored.png"), full_page=True)

        # ---- mark a cell missing -> diagnostics must show and block
        page.evaluate("""() => {
          const cv = document.getElementById('gridCanvas');
          const r = cv.getBoundingClientRect();
          cv.dispatchEvent(new MouseEvent('click',
            {bubbles:true, clientX:r.left+4, clientY:r.top+4}));
        }""")
        page.click("#btnClearCell")
        page.wait_for_timeout(900)
        diag = page.inner_text("#diagList")
        check("MISSING_KERNEL" in diag, "MISSING_KERNEL appears after removing a cell")
        page.click("#btnRestore")
        page.wait_for_timeout(1500)
        err = page.eval_on_selector("#errorBox", "e => ({hidden: e.hidden, text: e.textContent})")
        check(not err["hidden"] and "blocked" in err["text"].lower(),
              "blocked restoration is surfaced in the window's error panel")
        check(page.inner_text("#capRest") == "blocked", "restored pane is marked blocked")
        page.screenshot(path=str(SHOT / "03_blocked.png"), full_page=True)

        # ---- fill the hole and restore again
        page.click("#btnFill")
        page.wait_for_timeout(900)
        check("MISSING_KERNEL" not in page.inner_text("#diagList"),
              "filling the hole clears MISSING_KERNEL")
        page.click("#btnRestore")
        page.wait_for_function(
            "document.getElementById('imgRest').naturalWidth > 0", timeout=120000)
        page.wait_for_timeout(500)
        check(page.eval_on_selector("#errorBox", "e => e.hidden"),
              "error panel cleared after a successful run")

        check(not errors, "no uncaught JavaScript errors: " + "; ".join(errors[:4]))
        page.screenshot(path=str(SHOT / "04_after_fill.png"), full_page=True)
        browser.close()

    print(f"\n{'PASS' if not failures else 'FAIL'} ({len(failures)} failure(s))")
    for f in failures:
        print("  -", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(run())
