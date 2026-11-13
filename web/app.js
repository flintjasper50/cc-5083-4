/* SVDB restoration window — client logic.
 *
 * Single source of truth is `state.field` (the editable kernel field) plus the
 * solver controls.  Every restore and every export posts that exact state, and
 * the returned run_id is displayed in the header and stamped on all exports.
 */

const $ = (id) => document.getElementById(id);
const state = {
  cases: [], textures: [], boundaries: [],
  shape: [0, 0],
  field: null,            // {grid_shape, support, cells, interpolation, normalization}
  selected: [0, 0],
  probe: null,            // {y, x}
  obs: null,              // data URL of observed
  sharp: null,
  series: null,
  lastRun: null,
  runId: null,
  busy: false,
  probeCache: new Map(),
};

const el = {
  caseSelect: $("caseSelect"), caseMeta: $("caseMeta"),
  gridR: $("gridR"), gridC: $("gridC"), support: $("support"),
  interp: $("interp"), norm: $("norm"),
  sigma: $("sigma"), iters: $("iters"), lam: $("lam"), eps: $("eps"),
  mu: $("mu"), tau: $("tau"), boundary: $("boundary"), signed: $("signed"),
  showEdges: $("showEdges"), diagList: $("diagList"),
  gridCanvas: $("gridCanvas"), psfCanvas: $("psfCanvas"), psfMeta: $("psfMeta"),
  cy: $("cy"), cx: $("cx"), sy: $("sy"), sx: $("sx"), gridSel: $("gridSel"),
};

/* ----------------------------------------------------------- utilities */
function lambdaText(v) {
  if (v === null || v === undefined || !isFinite(v)) return "auto";
  return v.toExponential(2);
}
function fmt(v, d = 2) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  return v.toFixed(d);
}
function toast(msg, bad = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast show" + (bad ? " bad" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.className = "toast"; }, bad ? 5200 : 2600);
}
function setStatus(s) { $("status").textContent = s; }
async function api(path, body) {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const json = await res.json().catch(() => ({ error: "unparseable response" }));
  if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
  return json;
}
function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

/* ----------------------------------------------------------- field model */
function gridShape() { return state.field.grid_shape; }
function support() { return state.field.support; }

function cellAt(i, j) {
  const c = state.field.cells[i] && state.field.cells[i][j];
  return c ? c : null;
}
function setCell(i, j, k) {
  state.field.cells[i][j] = k;
}
function emptyCells() {
  const out = [];
  state.field.cells.forEach((row, i) => row.forEach((c, j) => { if (!c) out.push([i, j]); }));
  return out;
}
function normalizeKernel(k) {
  let s = 0;
  for (const row of k) for (const v of row) s += v;
  if (!isFinite(s) || Math.abs(s) < 1e-9) return null;
  return k.map((row) => row.map((v) => v / s));
}
function gaussianKernel(cell, supportPx) {
  const n = supportPx;
  const c = (n - 1) / 2;
  const cy = c + Number(el.cy.value), cx = c + Number(el.cx.value);
  const sy = Number(el.sy.value), sx = Number(el.sx.value);
  const k = [];
  for (let y = 0; y < n; y++) {
    const row = [];
    for (let x = 0; x < n; x++) {
      row.push(Math.exp(-((y - cy) ** 2 / (2 * sy * sy) + (x - cx) ** 2 / (2 * sx * sx))));
    }
    k.push(row);
  }
  return normalizeKernel(k) || k;
}
function resizeField(rows, cols, sup) {
  const old = state.field;
  const out = { grid_shape: [rows, cols], support: [sup, sup],
                interpolation: old.interpolation, normalization: old.normalization,
                cells: [] };
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      const src = old.cells[Math.min(i, old.grid_shape[0] - 1)]?.[Math.min(j, old.grid_shape[1] - 1)];
      if (!src) { row.push(null); continue; }
      if (src.length === sup) { row.push(src.map((r) => r.slice())); continue; }
      // centre crop / zero pad in 2-D
      const dst = Array.from({ length: sup }, () => Array(sup).fill(0));
      const sy0 = Math.floor((src.length - sup) / 2);
      const sx0 = Math.floor((src[0].length - sup) / 2);
      for (let y = 0; y < sup; y++) {
        for (let x = 0; x < sup; x++) {
          const y2 = y + sy0, x2 = x + sx0;
          if (y2 >= 0 && y2 < src.length && x2 >= 0 && x2 < src[0].length) dst[y][x] = src[y2][x2];
        }
      }
      row.push(normalizeKernel(dst));
    }
    out.cells.push(row);
  }
  state.field = out;
  state.selected = [Math.min(state.selected[0], rows - 1), Math.min(state.selected[1], cols - 1)];
}

/* ----------------------------------------------------------- canvas drawing */
const KERNEL_W = 260;
function drawGrid() {
  const cv = el.gridCanvas;
  const [gr, gc] = gridShape();
  const [sup] = support();
  const cell = 56;
  cv.width = gc * cell + 2;
  cv.height = gr * cell + 2;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const kn = state.field.cells;
  // normalise for display only
  for (let i = 0; i < gr; i++) {
    for (let j = 0; j < gc; j++) {
      const x0 = j * cell + 1, y0 = i * cell + 1;
      const k = kn[i][j];
      if (!k) {
        ctx.fillStyle = "#20272f";
        ctx.fillRect(x0, y0, cell - 2, cell - 2);
        ctx.strokeStyle = "#e5686b";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x0 + 4, y0 + 4); ctx.lineTo(x0 + cell - 6, y0 + cell - 6);
        ctx.moveTo(x0 + cell - 6, y0 + 4); ctx.lineTo(x0 + 4, y0 + cell - 6);
        ctx.stroke();
        ctx.fillStyle = "#e5686b";
        ctx.font = "9px ui-monospace,monospace";
        ctx.fillText("missing", x0 + 4, y0 + cell - 8);
      } else {
        const n = k.length;
        const px = (cell - 2) / n;
        let lo = Infinity, hi = -Infinity;
        for (const row of k) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
        if (hi <= lo) hi = lo + 1e-9;
        for (let y = 0; y < n; y++) {
          for (let x = 0; x < n; x++) {
            const t = (k[y][x] - lo) / (hi - lo);
            const g = Math.round(t * 255);
            ctx.fillStyle = `rgb(${g},${g},${Math.min(255, g + 30)})`;
            ctx.fillRect(x0 + x * px, y0 + y * px, Math.ceil(px), Math.ceil(px));
          }
        }
      }
      if (i === state.selected[0] && j === state.selected[1]) {
        ctx.strokeStyle = "#5aa9e6";
        ctx.lineWidth = 2;
        ctx.strokeRect(x0 - 1, y0 - 1, cell, cell);
      }
    }
  }
}

function drawProbeCanvas(k) {
  const cv = $("probeCanvas");
  const stage = $("stageKernel");
  if (!k) {
    cv.width = 4; cv.height = 4;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#0c1014"; ctx.fillRect(0, 0, 4, 4);
    return;
  }
  const box = stage.getBoundingClientRect();
  const n = k.length;
  const side = Math.max(64, Math.floor(Math.min(box.width, box.height) - 30));
  cv.width = side; cv.height = side;
  const ctx = cv.getContext("2d");
  const px = side / n;
  let lo = Infinity, hi = -Infinity;
  for (const row of k) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (hi <= lo) hi = lo + 1e-9;
  ctx.fillStyle = "#0c1014"; ctx.fillRect(0, 0, side, side);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const t = (k[y][x] - lo) / (hi - lo);
      const r = Math.round(28 + t * 215), g = Math.round(24 + t * 185), b2 = Math.round(30 + t * 170);
      ctx.fillStyle = `rgb(${r},${g},${b2})`;
      ctx.fillRect(x * px, y * px, Math.ceil(px), Math.ceil(px));
    }
  }
  // draw cell + grid gridlines for readability
  ctx.strokeStyle = "rgba(90,169,230,.35)";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, side - 1, side - 1);
  ctx.fillStyle = "#93a3b5";
  ctx.font = "10px ui-monospace,monospace";
  ctx.fillText(`${n}×${n} · Σ ${k.reduce((a, row) => a + row.reduce((b, v) => b + v, 0), 0).toFixed(3)}`,
               6, side - 6);
}

function drawPsf(k, metaText) {
  const cv = el.psfCanvas;
  if (!k) {
    cv.width = 8; cv.height = 8;
    const ctx = cv.getContext("2d"); ctx.fillStyle = "#0c1014"; ctx.fillRect(0, 0, 8, 8);
    el.psfMeta.innerHTML = metaText || "<b>missing</b>";
    return;
  }
  const n = k.length;
  const view = 132;
  cv.width = view; cv.height = view;
  const ctx = cv.getContext("2d");
  const px = view / n;
  let lo = Infinity, hi = -Infinity;
  for (const row of k) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (hi <= lo) hi = lo + 1e-9;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const t = (k[y][x] - lo) / (hi - lo);
      ctx.fillStyle = `rgb(${Math.round(30 + t * 210)},${Math.round(20 + t * 190)},${Math.round(24 + t * 190)})`;
      ctx.fillRect(x * px, y * px, Math.ceil(px), Math.ceil(px));
    }
  }
  let sum = 0, minv = Infinity, maxv = -Infinity;
  for (const row of k) for (const v of row) { sum += v; minv = Math.min(minv, v); maxv = Math.max(maxv, v); }
  el.psfMeta.innerHTML = `${metaText || ""}<br><b>Σ</b> ${fmt(sum, 4)}<br><b>min</b> ${fmt(minv, 3)}
    <br><b>max</b> ${fmt(maxv, 3)}<br><b>n</b> ${n}×${n}`;
}

/* ----------------------------------------------------------- probe */
function probeMarkup(info) {
  if (!info) return "probe";
  return `(${info.y.toFixed(1)}, ${info.x.toFixed(1)})`;
}

async function probeAt(y, x) {
  state.probe = { y, x };
  const [gr, gc] = gridShape();
  const overlay = elsOverlay();
  drawCellOverlay(overlay);
  if (emptyCells().length === (gr * gc)) {
    drawPsf(null, "no kernels"); drawProbeCanvas(null);
    return;
  }
  try {
    const info = await api("/api/kernel_at", {
      field: state.field, shape: state.shape, y, x,
    });
    $("capKernel").textContent = probeMarkup(info);
    if (info.missing) {
      drawPsf(null, `<b>undefined</b><br>interpolates into missing cell(s):<br>${
        info.missing_cells.map((c) => `(${c[0]},${c[1]})`).join(", ")}<br>Σ n/a`);
      drawProbeCanvas(null);
      return;
    }
    const w = info.weights.map(([i, j, v]) => `(${i},${j}) ${fmt(v, 2)}`).join(" · ");
    drawPsf(info.kernel, `<b>Σ</b> ${fmt(info.kernel_sum, 4)}<br><span style="color:#93a3b5">${w}</span>`);
    drawProbeCanvas(info.kernel);
  } catch (e) {
    toast(`kernel probe failed: ${e.message}`, true);
  }
}

/* crosshair placement in an image stage, in image pixel coordinates */
function placeCrosshair(stage, y, x) {
  const img = stage.querySelector("img");
  const ch = stage.querySelector(".crosshair");
  if (!img || !ch || !img.naturalWidth) return;
  const rect = img.getBoundingClientRect(), srect = stage.getBoundingClientRect();
  const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
  const dw = img.naturalWidth * scale, dh = img.naturalHeight * scale;
  const ox = (srect.width - dw) / 2, oy = (srect.height - dh) / 2;
  ch.style.display = "block";
  ch.style.left = `${ox + x * scale}px`;
  ch.style.top = `${oy + y * scale}px`;
}
function bindProbe(stage) {
  stage.addEventListener("contextmenu", (ev) => {
    ev.preventDefault();
    if (!state.shape[0]) return;
    const img = stage.querySelector("img");
    if (!img || !img.naturalWidth) return;
    // the <img> is object-fit:contain inside the stage; its own bounding box
    // is the exact drawn area, so map client coordinates through it directly
    const rect = img.getBoundingClientRect();
    const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
    const x = clamp((ev.clientX - rect.left) / scale, 0, img.naturalWidth - 1);
    const y = clamp((ev.clientY - rect.top) / scale, 0, img.naturalHeight - 1);
    placeCrosshair(stage, y, x);
    probeAt(y, x);
  });
}

/* overlay tent weight maps of each control cell (cheap, drawn from JS) */
function elsOverlay() {
  return {
    obs: ensureOverlay($("stageObs")),
    rest: ensureOverlay($("stageRest")),
    res: ensureOverlay($("stageRes")),
  };
}
function ensureOverlay(stage) {
  let cv = stage.querySelector("canvas.overlay");
  if (!cv) {
    cv = document.createElement("canvas");
    cv.className = "overlay";
    stage.appendChild(cv);
  }
  return cv;
}
function tentAxis(p, nPix, nCtrl) {
  if (nCtrl === 1) return [1];
  const t = p * (nCtrl - 1) / Math.max(nPix - 1, 1);
  const w = [];
  for (let c = 0; c < nCtrl; c++) w.push(Math.max(0, 1 - Math.abs(t - c)));
  return w;
}
function drawCellOverlay(set) {
  const [H, W] = state.shape;
  // the tent weight is 1 at a control point and falls to 0 at its neighbour;
  // we draw those boundaries plus markers for cells that carry no kernel.
  if (!H) return;
  const [gr, gc] = gridShape();
  for (const key of ["obs", "rest", "res"]) {
    const cv = set[key], stage = cv.parentElement;
    const img = stage.querySelector("img");
    if (!img || !img.naturalWidth) { cv.width = 0; continue; }
    const srect = stage.getBoundingClientRect();
    cv.width = Math.max(1, Math.round(srect.width));
    cv.height = Math.max(1, Math.round(srect.height));
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!el.showEdges.checked) continue;
    const iw = img.naturalWidth || W, ih = img.naturalHeight || H;
    const rect = img.getBoundingClientRect();
    const scale = Math.min(rect.width / iw, rect.height / ih);
    const dw = iw * scale, dh = ih * scale;
    const ox = rect.left - srect.left, oy = rect.top - srect.top;
    if (scale <= 0) continue;
    // draw cell boundary lines at control points (tent centres)
    ctx.save();
    ctx.strokeStyle = "rgba(90,169,230,.45)";
    ctx.lineWidth = 1;
    for (let c = 0; c < gr; c++) {
      const yy = oy + (c * (ih - 1) / Math.max(gr - 1, 1)) * scale;
      ctx.beginPath(); ctx.moveTo(ox, yy); ctx.lineTo(ox + dw, yy); ctx.stroke();
    }
    for (let c = 0; c < gc; c++) {
      const xx = ox + (c * (iw - 1) / Math.max(gc - 1, 1)) * scale;
      ctx.beginPath(); ctx.moveTo(xx, oy); ctx.lineTo(xx, oy + dh); ctx.stroke();
    }
    // prisms: the control point sits at the intersection; mark missing ones
    for (let i = 0; i < gr; i++) {
      for (let j = 0; j < gc; j++) {
        if (state.field.cells[i][j]) continue;
        const xx = ox + (j * (iw - 1) / Math.max(gc - 1, 1)) * scale;
        const yy = oy + (i * (ih - 1) / Math.max(gr - 1, 1)) * scale;
        ctx.fillStyle = "rgba(229,104,107,.9)";
        ctx.beginPath(); ctx.arc(xx, yy, 4, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.restore();
  }
}

/* ----------------------------------------------------------- diagnostics */
function renderDiagnostics(payload) {
  const list = el.diagList;
  list.innerHTML = "";
  if (!payload) { list.innerHTML = '<div class="empty-diag">no field loaded</div>'; return; }
  const items = payload.items || [];
  if (!items.length) {
    list.innerHTML = '<div class="empty-diag" style="color:#55c48a">✓ field is clean: '
      + 'unit-sum kernels, no holes, support within limits</div>';
    return;
  }
  for (const d of items) {
    const div = document.createElement("div");
    div.className = "diag " + d.severity;
    const cells = d.cells && d.cells.length
      ? `<div class="cells">cells: ${d.cells.map((c) => `(${c[0]},${c[1]})`).join(" ")}</div>` : "";
    div.innerHTML = `<span class="code">${d.code}</span><span class="msg"></span>${cells}`;
    div.querySelector(".msg").textContent = d.message;
    list.appendChild(div);
  }
}

let diagTimer = null;
function refreshDiagnostics() {
  if (!state.field) return;
  clearTimeout(diagTimer);
  diagTimer = setTimeout(async () => {
    try {
      const body = { field: state.field, shape: state.shape,
                     allow_signed: el.signed.value === "1" };
      const d = await api("/api/diagnostics", body);
      renderDiagnostics(d);
    } catch (e) { toast(`diagnostics failed: ${e.message}`, true); }
  }, 180);
}

/* ----------------------------------------------------------- restore */
function currentConfig() {
  const lamRaw = Number(el.lam.value);
  return {
    sigma: Number(el.sigma.value),
    iters: Number(el.iters.value),
    tv_eps: Number(el.eps.value),
    damping_mu: Number(el.mu.value),
    noise_floor_factor: Number(el.tau.value),
    ...(lamRaw > 0 ? { lam: lamRaw / 1000 } : {}),
  };
}

async function runRestore() {
  if (state.busy) return;
  if (!state.field) { toast("load a sample first", true); return; }
  state.busy = true;
  setStatus("running…");
  const t0 = performance.now();
  try {
    const body = {
      case: el.caseSelect.value,
      field: state.field,
      shape: state.shape,
      allow_signed: el.signed.value === "1",
      boundary: el.boundary.value,
      config: currentConfig(),
      snapshots: snapshotsFor(Number(el.iters.value)),
    };
    const res = await api("/api/restore", body);
    if (!res.ok) {
      renderDiagnostics(res.diagnostics);
      renderBlocked(res);
      toast(`restoration blocked: ${res.blocked_reason[0] || "see diagnostics"}`, true);
      setStatus("blocked");
      return;
    }
    state.lastRun = res;
    state.runId = res.run_id;
    renderRun(res);
    const secs = ((performance.now() - t0) / 1000).toFixed(2);
    setStatus(`done in ${secs}s`);
  } catch (e) {
    toast(`restore failed: ${e.message}`, true);
    setStatus("error");
  } finally {
    state.busy = false;
  }
}

function snapshotsFor(iters) {
  const want = [10, 25, 50, 100, 150, 200, 300, 400, 500, 700];
  const out = want.filter((v) => v <= iters);
  out.push(iters);
  return out;
}

function renderBlocked(res) {
  $("capRest").textContent = "blocked";
  $("imgRest").removeAttribute("src");
  $("imgRes").removeAttribute("src");
  strip.innerHTML = "";
  $("stopBox").innerHTML = "<b>stop</b> not started — kernel field has blocking diagnostics";
  $("mIn").textContent = $("mOut").textContent = $("mGain").textContent = $("mRms").textContent = "—";
  drawChart([], [], [], 0);
}

function renderRun(res) {
  $("tagRun").textContent = `run ${res.run_id}`;
  $("tagCfg").textContent = `σ ${fmt(res.config.sigma, 3)} · λ ${
    lambdaText(res.config.lam_resolved)} · ε ${fmt(res.config.tv_eps, 3)} · ${
    res.operator.boundary}`;
  $("imgRest").src = res.restored;
  $("imgObs").src = res.observed;
  $("capObs").textContent = `${res.operator.image_shape[1]}×${res.operator.image_shape[0]} · σ ${
    fmt(res.config.sigma, 3)}`;
  $("capRest").textContent = `run ${res.run_id}`;
  const m = res.metrics;
  $("mIn").innerHTML = `${fmt(m.psnr_input_vs_sharp, 2)}<small> dB</small>`;
  $("mOut").innerHTML = `${fmt(m.psnr_restored_vs_sharp, 2)}<small> dB</small>`;
  $("mGain").innerHTML = `${fmt(m.psnr_restored_vs_sharp - m.psnr_input_vs_sharp, 2)}<small> dB</small>`;
  $("mRms").innerHTML = `${fmt(m.residual.rms, 4)}`;
  $("stopBox").innerHTML = `<b>stop</b> ${res.series.stop_reason} @ iteration ${
    res.series.stop_iteration}<div class="why"></div>`;
  $("stopBox").querySelector(".why").textContent = res.series.stop_detail;
  drawChart(res.series.residuals, res.series.objectives, res.series.tv, res.series.stop_iteration);
  requestAnimationFrame(() => {
    drawChart(res.series.residuals, res.series.objectives, res.series.tv, res.series.stop_iteration);
  });

  // residual image (computed server-side at export only; approximate here from
  // the returned residual stats by showing the observed/restored difference)
  $("imgRes").src = res.observed;
  seResidualOverlay(res);

  $("opInfo").innerHTML = `grid ${res.operator.grid_shape.join("×")} · support ${
    res.operator.support.join("×")} · ${res.operator.interpolation}<br>norm ${
    res.operator.normalization} · boundary ${res.operator.boundary}<br>active cells ${
    res.operator.n_active_cells}`;

  renderStrip(res.frames);
  renderDiagnostics(res.diagnostics);
  document.querySelector("aside").scrollTop = 0;
  elsOverlay();
  drawCellOverlay(elsOverlay());
  drawGrid();
  if (state.probe) {
    probeAt(state.probe.y, state.probe.x);
  } else if (state.shape[0]) {
    probeAt((state.shape[0] - 1) / 2, (state.shape[1] - 1) / 2);
  }
}

/* draw the residual as a signed grey image inside the residual card */
function seResidualOverlay(res) {
  const stage = $("stageRes");
  let img = $("imgRes");
  img.style.visibility = "hidden";
  let cv = ensureOverlay(stage);
  cv.style.visibility = "visible";
  // load both images and difference them
  const a = new Image(), b = new Image();
  a.onload = () => {
    b.onload = () => {
      const off = document.createElement("canvas");
      const W = a.naturalWidth, Hh = a.naturalHeight;
      off.width = W; off.height = Hh;
      const octx = off.getContext("2d");
      octx.drawImage(a, 0, 0);
      const A = octx.getImageData(0, 0, W, Hh).data;
      octx.clearRect(0, 0, W, Hh); octx.drawImage(b, 0, 0);
      const B = octx.getImageData(0, 0, W, Hh).data;
      const out = octx.createImageData(W, Hh);
      for (let i = 0; i < A.length; i += 4) {
        const d = A[i] - B[i];           // observed - restored (uint8)
        const v = clamp(128 + d * 3, 0, 255);
        out.data[i] = out.data[i + 1] = out.data[i + 2] = v;
        out.data[i + 3] = 255;
      }
      octx.putImageData(out, 0, 0);
      img.src = off.toDataURL("image/png");
      img.style.visibility = "visible";
      cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
      img.onload = () => { cv.width = 0; drawCellOverlay(elsOverlay()); };
      $("capRes").textContent = `RMS ${fmt(res.metrics.residual.rms, 4)} · ±`;
    };
    b.src = res.observed;
  };
  a.src = res.restored;
}

/* ----------------------------------------------------------- charts */
function drawChart(residuals, objectives, tv, stopIter) {
  const cv = $("chart");
  const cssW = Math.max(240, Math.floor(cv.clientWidth || 300));
  const cssH = Math.max(212, Math.min(340, Math.round(cssW * 0.55)));
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(cssW * dpr);
  cv.height = Math.round(cssH * dpr);
  cv.style.height = cssH + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cssW, H = cssH;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#1e2630";
  ctx.fillRect(0, 0, W, H);
  const panels = [
    { s: residuals, label: "‖r‖/‖b‖", top: 6, h: 70, color: "#e5686b" },
    { s: objectives, label: "objective", top: 88, h: 62, color: "#5aa9e6" },
    { s: tv, label: "TV", top: 162, h: 40, color: "#55c48a" },
  ];
  for (const p of panels) {
    ctx.strokeStyle = "#33414f";
    ctx.lineWidth = 1;
    ctx.strokeRect(40, p.top, W - 52, p.h);
    ctx.fillStyle = "#93a3b5";
    ctx.font = "10px ui-monospace,monospace";
    ctx.fillText(p.label, 4, p.top + p.h / 2);
    if (!p.s || p.s.length < 2) continue;
    let lo = Infinity, hi = -Infinity;
    for (const v of p.s) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (hi <= lo) hi = lo + 1e-9;
    ctx.strokeStyle = p.color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i < p.s.length; i++) {
      const x = 40 + (W - 52) * i / (p.s.length - 1);
      const y = p.top + p.h - p.h * (p.s[i] - lo) / (hi - lo);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    if (stopIter > 0) {
      const sx = 40 + (W - 52) * clamp((stopIter - 1) / (p.s.length - 1), 0, 1);
      ctx.strokeStyle = "#e7eef7";
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(sx, p.top); ctx.lineTo(sx, p.top + p.h); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#e7eef7";
      ctx.fillText(`stop ${stopIter}`, Math.min(sx + 3, W - 60), p.top + 10);
    }
  }
}

/* ----------------------------------------------------------- filmstrip */
const strip = $("strip");
function renderStrip(frames) {
  strip.innerHTML = "";
  const label = document.createElement("div");
  label.className = "frame-label";
  label.innerHTML = "intermediate<br>iterations";
  strip.appendChild(label);
  if (!frames || !frames.length) {
    const empty = document.createElement("div");
    empty.className = "frame-label";
    empty.textContent = "no intermediate results (lower iterations or add snapshots)";
    strip.appendChild(empty);
    return;
  }
  for (const f of frames) {
    const div = document.createElement("div");
    div.className = "frame" + (f.iteration === state.lastRun?.series.stop_iteration ? " sel" : "");
    div.innerHTML = `<img src="${f.image}" alt=""><div class="fcap">it ${f.iteration}</div>`;
    div.title = "show this intermediate result in the restored pane";
    div.addEventListener("click", () => {
      $("imgRest").src = f.image;
      $("capRest").textContent = `iteration ${f.iteration} (run ${state.runId})`;
      drawCellOverlay(elsOverlay());
    });
    strip.appendChild(div);
  }
}

/* ----------------------------------------------------------- load */
function applyCaseMeta(c) {
  el.caseMeta.textContent =
    `${c.texture} · ${c.size[0]}×${c.size[1]} · grid ${c.grid.join("×")} · support ${
      c.support.join("×")} · σ ${fmt(c.sigma, 3)} · seed ${c.seed}${
      c.diagnostic ? " · diagnostic: " + c.diagnostic : ""}`;
}

async function loadCase(name) {
  setStatus("loading…");
  try {
    const res = await api("/api/load", { case: name });
    state.field = res.field;
    state.shape = res.shape;
    state.obs = res.observed;
    state.sharp = res.sharp;
    state.series = null;
    state.lastRun = null;
    state.runId = null;
    state.selected = [0, 0];
    state.probe = null;
    // reflect loaded metadata in the controls
    el.gridR.value = res.field.grid_shape[0]; $("oGridR").textContent = res.field.grid_shape[0];
    el.gridC.value = res.field.grid_shape[1]; $("oGridC").textContent = res.field.grid_shape[1];
    el.support.value = res.field.support[0]; $("oSupport").textContent = res.field.support[0];
    el.interp.value = res.field.interpolation;
    el.norm.value = res.field.normalization;
    el.sigma.value = res.sigma; $("oSigma").textContent = fmt(res.sigma, 3);
    if (res.config && res.config.iters) {
      el.iters.value = res.config.iters; $("oIters").textContent = res.config.iters;
    }
    if (res.config && res.config.tv_eps) {
      el.eps.value = res.config.tv_eps; $("oEps").textContent = res.config.tv_eps;
    }
    el.signed.value = res.allow_signed ? "1" : "0";
    $("imgObs").src = res.observed;
    $("capObs").textContent = `${res.shape[1]}×${res.shape[0]} · σ ${fmt(res.sigma, 3)}`;
    $("imgRest").removeAttribute("src");
    $("imgRes").removeAttribute("src");
    $("capRest").textContent = "no run";
    $("capRes").textContent = "no run";
    $("tagRun").textContent = "no run";
    $("tagCfg").textContent = "—";
    strip.innerHTML = "";
    renderDiagnostics(res.diagnostics);
    drawGrid();
    syncBrushFromSelected();
    elsOverlay();
    $("imgObs").onload = () => drawCellOverlay(elsOverlay());
    drawCellOverlay(elsOverlay());
    document.querySelector("aside").scrollTop = 0;
    if (res.diagnostics.errors.length) {
      toast(`loaded with ${res.diagnostics.errors.length} blocking diagnostic(s): ${
        res.diagnostics.errors.map((d) => d.code).join(", ")} — the field editor below is live`, true);
    } else {
      toast(`loaded ${name}`);
    }
    setStatus("loaded");
    applyCaseMeta(state.cases.find((c) => c.name === name) || res.meta);
  } catch (e) {
    toast(`load failed: ${e.message}`, true);
    setStatus("error");
  }
}

/* ----------------------------------------------------------- editor events */
function syncBrushFromSelected() {
  if (!state.field) return;
  const [gr, gc] = gridShape();
  const [i, j] = state.selected;
  el.gridSel.textContent = `cell (${i},${j})`;
  const k = state.field.cells[i]?.[j];
  if (!k) { drawPsf(null, "<b>missing</b>"); if (!state.probe) drawProbeCanvas(null); return; }
  // fit an axis-aligned Gaussian to the selected PSF for slider initialisation
  const n = k.length, c = (n - 1) / 2;
  let sy = 1, sx = 1, cy = 0, cx = 0, sw = 0;
  let m0 = 0, m1 = 0, m2 = 0;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const w = k[y][x]; cy += (y - c) * w; cx += (x - c) * w;
    m0 += w;
  }
  if (m0 > 1e-9) { cy /= m0; cx /= m0; }
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const w = k[y][x]; sy += w * (y - c - cy) ** 2; sx += w * (x - c - cx) ** 2;
  }
  if (m0 > 1e-9) { sy = Math.sqrt(Math.max(sy / m0, 0.2)); sx = Math.sqrt(Math.max(sx / m0, 0.2)); }
  el.cy.value = clamp(cy, -2, 2); el.cx.value = clamp(cx, -2, 2);
  el.sy.value = clamp(sy, 0.3, 5); el.sx.value = clamp(sx, 0.3, 5);
  updateBrushOutputs();
  if (!state.probe) drawPsf(k, `cell (${i},${j})`);
}

function updateBrushOutputs() {
  $("oCy").textContent = Number(el.cy.value).toFixed(1);
  $("oCx").textContent = Number(el.cx.value).toFixed(1);
  $("oSy").textContent = Number(el.sy.value).toFixed(1);
  $("oSx").textContent = Number(el.sx.value).toFixed(1);
}

function paintSelected() {
  const [i, j] = state.selected;
  const [sup] = support();
  setCell(i, j, gaussianKernel(null, sup));
  redrawAll();
  refreshDiagnostics();
  toast(`wrote PSF to cell (${i},${j})`);
}

function redrawAll() {
  drawGrid();
  syncBrushFromSelected();
  refreshDiagnostics();
  if (state.probe) probeAt(state.probe.y, state.probe.x);
}

function bindEditor() {
  el.gridCanvas.addEventListener("click", (ev) => {
    if (!state.field) return;
    const [gr, gc] = gridShape();
    const cell = 56;
    const rect = el.gridCanvas.getBoundingClientRect();
    const scale = rect.width / el.gridCanvas.width;
    const j = clamp(Math.floor((ev.clientX - rect.left) / (cell * scale)), 0, gc - 1);
    const i = clamp(Math.floor((ev.clientY - rect.top) / (cell * scale)), 0, gr - 1);
    state.selected = [i, j];
    redrawAll();
  });

  for (const id of ["cy", "cx", "sy", "sx"]) {
    $(id).addEventListener("input", updateBrushOutputs);
  }
  $("btnPaint").addEventListener("click", paintSelected);
  $("btnClearCell").addEventListener("click", () => {
    const [i, j] = state.selected;
    setCell(i, j, null);
    redrawAll();
    toast(`cell (${i},${j}) marked missing — restore will report MISSING_KERNEL`);
  });
  $("btnFillAll").addEventListener("click", () => {
    const [i, j] = state.selected;
    const src = state.field.cells[i]?.[j];
    if (!src) { toast("selected cell is missing; nothing to copy", true); return; }
    state.field.cells = state.field.cells.map((row) => row.map(() => src.map((r) => r.slice())));
    redrawAll();
    toast("filled every cell from the selected PSF");
  });
  $("btnSym").addEventListener("click", () => {
    const [i, j] = state.selected;
    const src = state.field.cells[i]?.[j];
    if (!src) { toast("selected cell is missing", true); return; }
    setCell(i, j, src.map((r) => r.slice().reverse()));
    redrawAll();
  });
  $("btnFill").addEventListener("click", () => {
    const missing = emptyCells();
    if (!missing.length) { toast("no missing cells"); return; }
    const [gr, gc] = gridShape();
    const valid = [];
    for (let i = 0; i < gr; i++) for (let j = 0; j < gc; j++)
      if (state.field.cells[i][j]) valid.push([i, j]);
    if (!valid.length) { toast("no valid kernels to copy from", true); return; }
    for (const [i, j] of missing) {
      let best = valid[0], bd = Infinity;
      for (const [a, b] of valid) {
        const d = (a - i) ** 2 + (b - j) ** 2;
        if (d < bd) { bd = d; best = [a, b]; }
      }
      setCell(i, j, state.field.cells[best[0]][best[1]].map((r) => r.slice()));
    }
    redrawAll();
    toast(`filled ${missing.length} missing cell(s) from nearest valid cell`);
  });

  $("btnResize").addEventListener("click", () => {
    if (!state.field) return;
    resizeField(Number(el.gridR.value), Number(el.gridC.value), Number(el.support.value));
    el.interp.value = state.field.interpolation;
    el.norm.value = state.field.normalization;
    redrawAll();
    toast(`field is now ${gridShape().join("×")} with ${support()[0]}px support`);
  });
  el.interp.addEventListener("change", () => { state.field.interpolation = el.interp.value; redrawAll(); });
  el.norm.addEventListener("change", () => { state.field.normalization = el.norm.value; redrawAll(); });
  el.signed.addEventListener("change", refreshDiagnostics);
  el.showEdges.addEventListener("change", () => drawCellOverlay(elsOverlay()));
}

/* ----------------------------------------------------------- control wiring */
function bindControls() {
  $("btnLoad").addEventListener("click", () => loadCase(el.caseSelect.value));
  el.caseSelect.addEventListener("change", () => applyCaseMeta(
    state.cases.find((c) => c.name === el.caseSelect.value) || {}));
  $("btnReload").addEventListener("click", async () => {
    try {
      const res = await api("/api/load", { case: el.caseSelect.value });
      state.field = res.field;
      state.selected = [0, 0];
      redrawAll();
      toast("field reloaded from the sample definition");
    } catch (e) { toast(e.message, true); }
  });

  $("btnRestore").addEventListener("click", runRestore);
  $("btnExport").addEventListener("click", doExport);

  for (const [id, out, fmtFn] of [
    [el.sigma, "oSigma", (v) => v.toFixed(3)],
    [el.iters, "oIters", (v) => String(v)],
    [el.eps, "oEps", (v) => v.toFixed(3)],
    [el.mu, "oMu", (v) => Number(v).toFixed(2)],
    [el.tau, "oTau", (v) => Number(v).toFixed(2)],
    [el.lam, "oLam", (v) => (Number(v) > 0 ? (Number(v) / 1000).toExponential(1) : "auto")],
    [el.gridR, "oGridR", (v) => String(v)],
    [el.gridC, "oGridC", (v) => String(v)],
    [el.support, "oSupport", (v) => String(v)],
  ]) {
    id.addEventListener("input", () => { $(out).textContent = fmtFn(id.value); });
  }
  ["stageObs", "stageRest", "stageRes"].forEach((id) => bindProbe($(id)));
}

/* ----------------------------------------------------------- export */
async function doExport() {
  if (state.busy) return;
  if (!state.field) { toast("load a sample first", true); return; }
  state.busy = true;
  setStatus("running + exporting…");
  try {
    const body = {
      case: el.caseSelect.value,
      field: state.field,
      shape: state.shape,
      allow_signed: el.signed.value === "1",
      boundary: el.boundary.value,
      config: currentConfig(),
      snapshots: snapshotsFor(Number(el.iters.value)),
    };
    const res = await api("/api/export", body);
    if (!res.ok) {
      renderDiagnostics(res.diagnostics);
      renderBlocked(res);
      toast(`export blocked: ${res.blocked_reason[0] || "see diagnostics"}`, true);
      setStatus("blocked");
      return;
    }
    $("tagRun").textContent = `run ${res.run_id} (exported)`;
    const links = $("exportLinks");
    links.innerHTML = "";
    const pairs = [
      ["restored image", res.restored_url],
      ["residual image", res.residual_url],
      ["kernel field sheet", res.kernels_url],
      ["iteration curves", res.curves_url],
      ["JSON report (same run)", res.report_url],
    ];
    for (const [label, url] of pairs) {
      const b = document.createElement("button");
      b.innerHTML = `${label}<span>${url}</span>`;
      b.addEventListener("click", () => {
        const a = document.createElement("a");
        a.href = url; a.download = url.split("/").pop();
        document.body.appendChild(a); a.click(); a.remove();
      });
      links.appendChild(b);
    }
    setStatus("exported");
    toast(`exported 5 artefacts from run ${res.run_id}`);
    // refresh the view with the same run's returned summary
    await runRestore();
  } catch (e) {
    toast(`export failed: ${e.message}`, true);
    setStatus("error");
  } finally {
    state.busy = false;
  }
}

/* ----------------------------------------------------------- boot */
async function boot() {
  try {
    const meta = await api("/api/cases");
    state.cases = meta.cases;
    state.textures = meta.textures;
    state.boundaries = meta.boundaries;
    el.boundary.innerHTML = meta.boundaries.map(
      (b) => `<option value="${b}">${b}</option>`).join("");
    el.boundary.value = "reflect";
    el.interp.innerHTML = meta.interpolations.map(
      (b) => `<option value="${b}">${b}</option>`).join("");
    el.norm.innerHTML = meta.normalizations.map(
      (b) => `<option value="${b}">${b}</option>`).join("");
    el.caseSelect.innerHTML = state.cases.map(
      (c) => `<option value="${c.name}">${c.name} — ${c.kind}${
        c.diagnostic ? " ⚠ " + c.diagnostic : ""}</option>`).join("");
    bindControls();
    bindEditor();
    document.querySelector("aside").scrollTop = 0;
    const first = state.cases.find((c) => c.name === "gradient") || state.cases[0];
    el.caseSelect.value = first.name;
    applyCaseMeta(first);
    await loadCase(first.name);
    drawChart([], [], [], 0);
  } catch (e) {
    toast(`boot failed: ${e.message} (is the server running?)`, true);
    setStatus("offline");
  }
}
boot();
