import json, sys, time
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8765"
SHOT = "/workspace/out/ui"
import os
os.makedirs(SHOT, exist_ok=True)

def run():
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1680, "height": 1000})
        pg.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        pg.goto(BASE + "/", wait_until="networkidle")
        pg.wait_for_function("document.getElementById('imgObs').naturalWidth > 0", timeout=15000)
        pg.screenshot(path=f"{SHOT}/01_loaded.png", full_page=True)
        print("status after load:", pg.inner_text("#status"))
        print("case meta:", pg.inner_text("#caseMeta"))
        print("diagnostics:", pg.inner_text("#diagList")[:200].replace("\n", " | "))
        print("observed img:", pg.eval_on_selector("#imgObs", "e => e.naturalWidth + 'x' + e.naturalHeight"))
        print("grid canvas:", pg.eval_on_selector("#gridCanvas", "e => e.width + 'x' + e.height"))

        # probe via contextmenu on observed
        pg.evaluate("""() => {
          const stage = document.getElementById('stageObs');
          const r = stage.getBoundingClientRect();
          stage.dispatchEvent(new MouseEvent('contextmenu', {bubbles:true, clientX:r.left+r.width*0.5, clientY:r.top+r.height*0.5}));
        }""")
        pg.wait_for_timeout(900)
        print("kernel cap:", pg.inner_text("#capKernel"))
        print("psf meta:", pg.inner_text("#psfMeta").replace("\n", " | "))
        pg.screenshot(path=f"{SHOT}/02_probe.png", full_page=True)

        # restore
        pg.click("#btnRestore")
        pg.wait_for_function("document.getElementById('imgRest').naturalWidth > 0", timeout=60000)
        pg.wait_for_timeout(500)
        print("after restore status:", pg.inner_text("#status"))
        print("run tag:", pg.inner_text("#tagRun"))
        print("cfg tag:", pg.inner_text("#tagCfg"))
        print("metrics in/out/gain/rms:", pg.inner_text("#mIn"), pg.inner_text("#mOut"),
              pg.inner_text("#mGain"), pg.inner_text("#mRms"))
        print("stop:", pg.inner_text("#stopBox").replace("\n", " | "))
        print("frames:", pg.eval_on_selector_all(".frame", "els => els.length"))

        # every chart panel must contain its curve colour inside its own band
        chart = pg.evaluate("""() => {
          const cv = document.getElementById('chart');
          const dpr = window.devicePixelRatio || 1;
          const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
          const panels = [[6,70,[229,104,107]], [88,62,[90,169,230]], [162,40,[85,196,138]]];
          const out = [];
          for (const [top, h, [r,g,b]] of panels) {
            let hits = 0;
            const y0 = Math.floor(top * dpr), y1 = Math.min(cv.height, Math.ceil((top + h) * dpr));
            for (let y = y0; y < y1; y++) for (let x = 0; x < cv.width; x++) {
              const i = (y * cv.width + x) * 4;
              if (Math.abs(d[i]-r) < 40 && Math.abs(d[i+1]-g) < 40 && Math.abs(d[i+2]-b) < 40) hits++;
            }
            out.push(hits);
          }
          return {w: cv.width, h: cv.height, css: cv.style.height, bands: out};
        }""")
        print("chart:", chart)
        assert all(b > 30 for b in chart["bands"]), chart

        # overlay canvas must line up with the drawn image, not the whole stage
        ov = pg.evaluate("""() => {
          const stage = document.getElementById('stageObs');
          const cv = stage.querySelector('canvas.overlay');
          const img = stage.querySelector('img');
          const s = stage.getBoundingClientRect(), i = img.getBoundingClientRect();
          return {canvas:[cv.width, cv.height], stage:[s.width, s.height],
                  imgOffset:[i.left - s.left, i.top - s.top]};
        }""")
        print("overlay:", ov)
        pg.screenshot(path=f"{SHOT}/03_restored.png", full_page=True)

        # click an early frame
        if pg.eval_on_selector_all(".frame", "els => els.length") > 1:
            pg.eval_on_selector_all(".frame", "els => els[1].click()")
            pg.wait_for_timeout(400)
            print("frame cap:", pg.inner_text("#capRest"))
        pg.screenshot(path=f"{SHOT}/04_frame.png", full_page=True)

        # edit a cell: mark missing -> diagnostics should show MISSING_KERNEL and restore blocked
        pg.evaluate("""() => {
          const cv = document.getElementById('gridCanvas');
          const r = cv.getBoundingClientRect();
          cv.dispatchEvent(new MouseEvent('click', {bubbles:true, clientX:r.left+30, clientY:r.top+30}));
        }""")
        pg.click("#btnClearCell")
        pg.wait_for_timeout(800)
        print("diag after missing:", pg.inner_text("#diagList")[:260].replace("\n", " | "))
        pg.click("#btnRestore")
        pg.wait_for_timeout(1500)
        print("status after blocked restore:", pg.inner_text("#status"))
        print("stop after blocked:", pg.inner_text("#stopBox").replace("\n", " | "))
        pg.screenshot(path=f"{SHOT}/05_missing.png", full_page=True)

        # fill holes then restore again
        pg.click("#btnFill")
        pg.wait_for_timeout(700)
        print("diag after fill:", pg.inner_text("#diagList")[:200].replace("\n", " | "))
        pg.click("#btnRestore")
        pg.wait_for_function("document.getElementById('imgRest').naturalWidth > 0", timeout=60000)
        pg.wait_for_timeout(400)
        print("status after refill restore:", pg.inner_text("#status"))

        # export
        pg.click("#btnExport")
        pg.wait_for_selector("#exportLinks button", timeout=90000)
        pg.wait_for_timeout(1500)
        links = pg.eval_on_selector_all("#exportLinks button span", "els => els.map(e => e.textContent)")
        print("export links:", links)
        print("tag after export:", pg.inner_text("#tagRun"))
        pg.screenshot(path=f"{SHOT}/06_export.png", full_page=True)

        # kernel atlas sheet screenshots for the built-in diagnostic cases
        for case in ("diag_missing", "diag_negative", "diag_oversized"):
            pg.select_option("#caseSelect", case)
            pg.click("#btnLoad")
            pg.wait_for_function("document.getElementById('imgObs').naturalWidth > 0", timeout=20000)
            pg.wait_for_timeout(900)
            print(f"[{case}] diag:", pg.inner_text("#diagList")[:300].replace("\n", " | "))
            pg.screenshot(path=f"{SHOT}/07_{case}.png", full_page=True)

        b.close()
    print("\nJS errors:", len(errors))
    for e in errors[:20]:
        print(" -", e)
    return 0 if not errors else 1

if __name__ == "__main__":
    sys.exit(run())
