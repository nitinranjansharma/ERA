const tokens = ["The", "cat", "sat", "down"];
const Q = [
  [0.9, 0.3, 0.1, 0.0],
  [0.2, 1.0, 0.2, 0.1],
  [0.15, 0.9, 0.4, 0.1],
  [0.1, 0.2, 0.2, 0.8],
];
const K = [
  [1.0, 0.2, 0.0, 0.1],
  [0.2, 1.2, 0.3, 0.0],
  [0.1, 0.4, 1.1, 0.2],
  [0.0, 0.1, 0.3, 1.0],
];
const V = [
  [1.0, 0.0, 0.0],
  [0.0, 1.0, 0.0],
  [0.0, 0.0, 1.0],
  [0.2, 0.2, 0.6],
];
const DK = Q[0].length;

function dot(a, b) {
  return a.reduce((s, x, i) => s + x * b[i], 0);
}
function fmt(x) {
  if (!Number.isFinite(x)) return "−∞";
  return (Math.round(x * 100) / 100).toFixed(2);
}
function softmax(row) {
  const finite = row.filter(Number.isFinite);
  const m = Math.max(...finite);
  const exps = row.map((x) => (Number.isFinite(x) ? Math.exp(x - m) : 0));
  const z = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / z);
}

function scoresRaw() {
  return Q.map((q) => K.map((k) => dot(q, k)));
}

function pipeline(maskOn, scaleOn) {
  const raw = scoresRaw();
  const scaled = raw.map((row) => row.map((x) => (scaleOn ? x / Math.sqrt(DK) : x)));
  const masked = scaled.map((row, i) =>
    row.map((x, j) => (maskOn && j > i ? -Infinity : x))
  );
  const weights = masked.map(softmax);
  const out = weights.map((w) =>
    V[0].map((_, d) => w.reduce((s, wi, j) => s + wi * V[j][d], 0))
  );
  return { raw, scaled, masked, weights, out };
}

function table(rows, rowLabels, colLabels, heat) {
  let html = "<table><thead><tr><th></th>";
  colLabels.forEach((c) => (html += `<th>${c}</th>`));
  html += "</tr></thead><tbody>";
  rows.forEach((row, i) => {
    html += `<tr><td class="rowhead">${rowLabels[i]}</td>`;
    row.forEach((v) => {
      let cls = "";
      if (heat && Number.isFinite(v)) {
        if (v > 0.45) cls = "hot";
        else if (v > 0.25) cls = "warm";
        else cls = "cool";
      }
      if (!Number.isFinite(v)) cls = "dead";
      html += `<td class="${cls}">${fmt(v)}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  return html;
}

function initAttention() {
  const root = document.querySelector("#attention-lab");
  if (!root) return;
  const steps = ["Scores", "Scale", "Mask", "Softmax", "Read V"];
  let step = 0;
  let maskOn = true;
  let focus = 2;

  root.querySelector(".steps").innerHTML = steps
    .map((s, i) => `<button data-step="${i}" ${i === 0 ? 'class="active"' : ""}>${i + 1} ${s}</button>`)
    .join("");

  function render() {
    const p = pipeline(maskOn, true);
    const unscaled = pipeline(maskOn, false);
    const labels = tokens;
    let matrix;
    let heat = false;
    let shown;
    if (step === 0) shown = p.raw;
    if (step === 1) shown = p.scaled;
    if (step === 2) shown = p.masked;
    if (step === 3) {
      shown = p.weights;
      heat = true;
    }
    if (step === 4) {
      shown = p.out;
      heat = true;
    }
    const cols = step === 4 ? ["V0 red", "V1 green", "V2 blue"] : labels;
    matrix = table(shown, labels, cols, heat);

    const row = p.weights[focus];
    const best = row.indexOf(Math.max(...row));
    const mix = p.out[focus].map(fmt).join(", ");
    const rawRow = p.raw[focus].map(fmt).join(", ");
    const notes = [
      `Raw dots QKᵀ for every query. Row "${tokens[focus]}" · keys = [${rawRow}]. The cat key wins because the query was built to look like it. No order, no future-block, no sharpness control yet.`,
      `Same matrix divided by √dₖ = √${DK} = ${fmt(Math.sqrt(DK))}. With dₖ this small the change is mild. The saturation lab beside this one raises dₖ and the scale stops being optional.`,
      maskOn
        ? `Future positions are −∞ before the softmax. "${tokens[focus]}" can only read itself and tokens to its left. The 2017 encoder left this mask off. Every causal language model turns it on.`
        : `Mask off: every query reads every key, including the future. That is encoder self-attention in the 2017 paper, and it is illegal for a decoder that is still writing the sentence.`,
      `Softmax along the row. Weights on "${tokens[focus]}" land hardest on "${tokens[best]}" (${fmt(row[best])}). They are a distribution: the row sums to 1, and a −∞ score is exactly 0.`,
      `Output of "${tokens[focus]}" is the weighted sum of value rows: [${mix}]. Values here are primary colors, so the output color is the mixture the weights asked for. The winning content is the cat's value, not a copy of the query.`,
    ];
    root.querySelector("#score-table").innerHTML = matrix;
    root.querySelector("#readout").innerHTML = `<strong>${steps[step]}.</strong> ${notes[step]}`;
    root.querySelectorAll(".steps button").forEach((b) => {
      b.classList.toggle("active", Number(b.dataset.step) === step);
    });
    root.querySelector("#mask-toggle").setAttribute("aria-pressed", String(maskOn));
    root.querySelector("#mask-toggle").textContent = maskOn ? "Causal mask on" : "Causal mask off";
    const peakOff = Math.max(...unscaled.weights[focus].filter(Number.isFinite));
    const peakOn = Math.max(...p.weights[focus]);
    root.querySelector("#scale-note").textContent =
      `Unscaled peak on this row: ${fmt(peakOff)}. Scaled peak: ${fmt(peakOn)}.`;
  }

  root.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.step != null) step = Number(b.dataset.step);
    if (b.id === "mask-toggle") maskOn = !maskOn;
    if (b.dataset.focus != null) focus = Number(b.dataset.focus);
    render();
  });
  render();
}

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rng) {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function initScale() {
  const root = document.querySelector("#scale-lab");
  if (!root) return;
  const input = root.querySelector("input");
  const draw = () => {
    const d = Number(input.value);
    const rng = mulberry32(7);
    const trials = 400;
    let oneHot = 0;
    let oneHotScaled = 0;
    for (let t = 0; t < trials; t++) {
      const q = Array.from({ length: d }, () => gauss(rng));
      const keys = Array.from({ length: 8 }, () => Array.from({ length: d }, () => gauss(rng)));
      const dots = keys.map((k) => dot(q, k));
      const w = softmax(dots);
      const ws = softmax(dots.map((x) => x / Math.sqrt(d)));
      if (Math.max(...w) > 0.99) oneHot++;
      if (Math.max(...ws) > 0.99) oneHotScaled++;
    }
    root.querySelector("#scale-read").innerHTML =
      `dₖ = ${d}. Over 400 random queries against 8 keys, <strong>${oneHot}</strong> unscaled softmaxes collapse to a single key (max weight &gt; 0.99). After dividing by √dₖ, that happens <strong>${oneHotScaled}</strong> times. The scale is there so the softmax still has a gradient.`;
  };
  input.addEventListener("input", draw);
  draw();
}

function initRope() {
  const canvas = document.querySelector("#rope-canvas");
  if (!canvas) return;
  const slider = document.querySelector("#rope-offset");
  const note = document.querySelector("#rope-note");
  const ctx = canvas.getContext("2d");

  const q = [1.0, 0.25];
  const k = [0.85, 0.55];
  function rot(v, ang) {
    const c = Math.cos(ang);
    const s = Math.sin(ang);
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c];
  }
  function draw() {
    const offset = Number(slider.value);
    const theta = 0.35;
    const qr = rot(q, theta * 4);
    const kr = rot(k, theta * (4 - offset));
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "#e4d9c8";
    ctx.beginPath();
    ctx.moveTo(40, h / 2);
    ctx.lineTo(w - 20, h / 2);
    ctx.moveTo(w / 2, 16);
    ctx.lineTo(w / 2, h - 16);
    ctx.stroke();
    const cx = w / 2;
    const cy = h / 2;
    const scale = 78;
    function arrow(v, color, label) {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + v[0] * scale, cy - v[1] * scale);
      ctx.stroke();
      ctx.fillText(label, cx + v[0] * scale + 6, cy - v[1] * scale);
    }
    ctx.font = "13px IBM Plex Mono, monospace";
    arrow(rot(q, 0), "#c8b8a2", "q content");
    arrow(qr, "#1c3d66", "q at pos 4");
    arrow(kr, "#9a3418", "k at relative −" + offset);
    const rel = dot(q, rot(k, -theta * offset));
    const abs = dot(qr, kr);
    note.textContent =
      `Relative offset m−n = ${offset}. qₘ · kₙ = ${fmt(abs)}. The same product computed by rotating k only by the offset is ${fmt(rel)}. They match. The absolute positions dropped out.`;
  }
  slider.addEventListener("input", draw);
  draw();
}

function initAlibi() {
  const root = document.querySelector("#alibi-lab");
  if (!root) return;
  const slider = root.querySelector("input");
  const host = root.querySelector("#alibi-table");
  const note = root.querySelector("#alibi-note");
  const n = 6;
  function render() {
    const slope = Number(slider.value);
    const rows = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) {
        const content = j === i - 2 ? 1.2 : j === i ? 0.4 : 0.15;
        const bias = j <= i ? -slope * (i - j) : -Infinity;
        row.push(content + bias);
      }
      rows.push(softmax(row));
    }
    host.innerHTML = table(rows, ["t0", "t1", "t2", "t3", "t4", "t5"], ["t0", "t1", "t2", "t3", "t4", "t5"], true);
    note.textContent =
      `Slope m = ${slope.toFixed(2)}. Each step into the past subtracts ${slope.toFixed(2)} from the score before softmax. A steep head forgets quickly. A shallow head can still see the start of the sentence. There is no position table to run out of, which is why a model trained short can be run long.`;
  }
  slider.addEventListener("input", render);
  render();
}

function initCache() {
  const root = document.querySelector("#cache-lab");
  if (!root) return;
  const modes = { mha: 8, gqa: 2, mqa: 1 };
  let mode = "mha";
  function render() {
    const heads = 8;
    const kv = modes[mode];
    const share = heads / kv;
    const perToken = kv * 2;
    root.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
    let cols = "";
    for (let g = 0; g < kv; g++) {
      let qs = "";
      for (let i = 0; i < share; i++) qs += `<div class="qdot"></div>`;
      cols += `<div class="head-col"><div class="kvdot"></div>${qs}<div>KV ${g + 1}</div><div>${share} queries</div></div>`;
    }
    root.querySelector("#head-diagram").innerHTML = cols;
    const names = { mha: "Multi-head", gqa: "Grouped-query", mqa: "Multi-query" };
    root.querySelector("#cache-stat").innerHTML =
      `${names[mode]}. ${heads} query heads, ${kv} key/value heads. At every new token the cache stores ${perToken} vectors instead of ${heads * 2}. A 32k context keeps ${mode === "mha" ? "the full head set" : perToken + " vectors per layer per token"}. Prefill is still quadratic: sharing KV does not touch QKᵀ over the prompt.`;
  }
  root.addEventListener("click", (e) => {
    const b = e.target.closest("[data-mode]");
    if (!b) return;
    mode = b.dataset.mode;
    render();
  });
  render();
}

function initWindow() {
  const root = document.querySelector("#window-lab");
  if (!root) return;
  const n = 18;
  const query = 17;
  const win = root.querySelector("#win");
  const sinksOn = root.querySelector("#sinks");
  const host = root.querySelector("#window-cells");
  const note = root.querySelector("#window-note");
  function render() {
    const w = Number(win.value);
    const keepSinks = sinksOn.checked;
    let html = "";
    for (let i = 0; i < n; i++) {
      const inWin = i > query - w && i <= query;
      const sink = keepSinks && i < 2;
      const cls = ["cell", i === query ? "query" : "", inWin || sink ? "on" : "dim", sink && !inWin ? "sink" : ""].join(" ");
      html += `<div class="${cls}">${i === query ? "Q" : i}</div>`;
    }
    host.innerHTML = html;
    const kept = keepSinks ? Math.min(n, w + 2) : Math.min(n, w);
    note.textContent = keepSinks
      ? `Window of ${w} plus 2 sink tokens. The pale tokens in the middle are evicted. Generation stays stable because the softmax still has the sink tokens it learned to dump mass on. Those middle tokens cannot be quoted later. Cache length is about ${kept}, not ${n}.`
      : `Window of ${w}, sinks dropped. Early tokens are gone, including the ones that were acting as attention sinks. The softmax renormalizes over the window only, which is the failure StreamingLLM measured.`;
  }
  win.addEventListener("input", render);
  sinksOn.addEventListener("change", render);
  render();
}

function initDelta() {
  const root = document.querySelector("#delta-lab");
  if (!root) return;
  const kA = [1, 0];
  const kB = [0, 1];
  let add = [[0, 0], [0, 0]];
  let del = [[0, 0], [0, 0]];
  const log = [];

  function write(name, k, v, beta = 1) {
    add = add.map((row, r) => row.map((x, c) => x + v[r] * k[c]));
    const sk = [del[0][0] * k[0] + del[0][1] * k[1], del[1][0] * k[0] + del[1][1] * k[1]];
    const err = [sk[0] - v[0], sk[1] - v[1]];
    del = del.map((row, r) => row.map((x, c) => x - beta * err[r] * k[c]));
    log.unshift(name);
  }
  function read(S, k) {
    return [S[0][0] * k[0] + S[0][1] * k[1], S[1][0] * k[0] + S[1][1] * k[1]];
  }
  function render() {
    const aAdd = read(add, kA).map(fmt).join(", ");
    const aDel = read(del, kA).map(fmt).join(", ");
    const bAdd = read(add, kB).map(fmt).join(", ");
    const bDel = read(del, kB).map(fmt).join(", ");
    root.querySelector("#delta-read").innerHTML =
      `Read key A → additive state [${aAdd}], delta state [${aDel}].<br>Read key B → additive [${bAdd}], delta [${bDel}].<br><span style="color:#c8b8a2">${log.slice(0, 4).join(" · ") || "no writes yet"}</span>`;
  }
  root.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.write === "a1") write("write A ← (1, 0)", kA, [1, 0]);
    if (b.dataset.write === "a2") write("overwrite A ← (0, 1)", kA, [0, 1]);
    if (b.dataset.write === "b") write("write B ← (0, 1)", kB, [0, 1]);
    if (b.dataset.write === "reset") {
      add = [[0, 0], [0, 0]];
      del = [[0, 0], [0, 0]];
      log.length = 0;
    }
    render();
  });
  render();
}

function initFreq() {
  const root = document.querySelector("#freq-lab");
  if (!root) return;
  const input = root.querySelector("input");
  const host = root.querySelector("#freq-lanes");
  const bands = [
    { name: "highest", pi: 1, ntk: 1, yarn: 1 },
    { name: "high", pi: 1, ntk: 0.55, yarn: 0.15 },
    { name: "mid", pi: 1, ntk: 0.85, yarn: 0.55 },
    { name: "lowest", pi: 1, ntk: 1, yarn: 1 },
  ];
  function render() {
    const s = Number(input.value);
    host.innerHTML = bands
      .map((b) => {
        const parts = [
          ["PI", Math.min(100, (b.pi * s) * 12), "#9a3418"],
          ["NTK", Math.min(100, (1 + (s - 1) * b.ntk) * 12), "#8a5a10"],
          ["YaRN", Math.min(100, (1 + (s - 1) * b.yarn) * 12), "#0d5c56"],
        ];
        const ticks = parts
          .map(([label, w, color]) => `<i title="${label}" style="width:${w}%;background:${color}"></i>`)
          .join("");
        return `<div class="lane"><span>${b.name}</span><div class="ticks">${ticks}</div></div>`;
      })
      .join("");
    root.querySelector("#freq-note").textContent =
      `Scale ×${s}, drawn schematically. Position interpolation stretches every frequency by ${s}. NTK-aware scaling moves the slow frequencies much more than the fast ones. YaRN leaves the highest frequencies alone, fully interpolates the slowest, and ramps the middle. Bar length is stretch, not wavelength.`;
  }
  input.addEventListener("input", render);
  render();
}

function initCsa() {
  const root = document.querySelector("#csa-lab");
  if (!root) return;
  const n = 16;
  const m = 4;
  const scores = [0.2, 1.4, 0.4, 0.9];
  const host = root.querySelector("#csa-cells");
  const note = root.querySelector("#csa-note");
  let topk = 2;
  function render() {
    const keep = scores
      .map((s, i) => [s, i])
      .sort((a, b) => b[0] - a[0])
      .slice(0, topk)
      .map((x) => x[1]);
    let html = "";
    for (let i = 0; i < n; i++) {
      const block = Math.floor(i / m);
      const local = i >= n - m;
      const selected = keep.includes(block);
      const cls = ["cell", local ? "sink" : selected ? "on" : "dim"].join(" ");
      html += `<div class="${cls}">${i}</div>`;
    }
    host.innerHTML = html;
    note.textContent =
      `16 tokens compressed by m = 4 into 4 KV entries, then top-${topk} of those entries are selected. Gold cells are the recent window, kept uncompressed so the current block is not invisible to itself. Everything dimmed is not in the score. A token inside a selected block is visible only through the compressed summary.`;
  }
  root.addEventListener("click", (e) => {
    const b = e.target.closest("[data-k]");
    if (!b) return;
    topk = Number(b.dataset.k);
    root.querySelectorAll("[data-k]").forEach((x) => x.classList.toggle("active", x === b));
    render();
  });
  render();
}

function initNav() {
  const links = [...document.querySelectorAll(".rail a")];
  const sections = links
    .map((a) => document.querySelector(a.getAttribute("href")))
    .filter(Boolean);
  if (!sections.length || !("IntersectionObserver" in window)) return;
  const obs = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((e) => e.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + visible.target.id));
    },
    { rootMargin: "-20% 0px -60% 0px", threshold: [0.1, 0.4] }
  );
  sections.forEach((s) => obs.observe(s));
}

document.addEventListener("DOMContentLoaded", () => {
  initAttention();
  initScale();
  initRope();
  initAlibi();
  initCache();
  initWindow();
  initDelta();
  initFreq();
  initCsa();
  initNav();
});
