import {
  validateReplay,
  frameAtTime,
  formatTime,
  poseViewport,
} from "./replay.js";

const $ = (id) => document.getElementById(id);
const elements = [
  "notice",
  "error",
  "frame-status",
  "frame-tag",
  "stage-message",
  "mean-confidence",
  "confidence-description",
  "latency",
  "cadence",
  "current-time",
  "duration",
  "seek",
  "play",
  "restart",
  "next",
  "speed",
  "show-gt",
  "fit-view",
  "plot-mode",
  "gt-legend",
  "spectrum-shape",
  "joint-list",
  "failure-detail",
  "provenance-note",
  "meta-dataset",
  "meta-domain",
  "meta-split",
  "meta-checkpoint",
  "evaluation",
];
const ui = Object.fromEntries(elements.map((id) => [id, $(id)]));
const poseCanvas = $("pose-canvas"),
  spectrumCanvas = $("spectrum");
const poseCtx = poseCanvas.getContext("2d"),
  spectrumCtx = spectrumCanvas.getContext("2d");
let replay = null,
  index = 0,
  playing = false,
  anchorWall = 0,
  anchorTime = 0,
  speed = 1;
let jointRows = [],
  lastRendered = -1;
let viewport = { x: 0, y: 0, span: 1 };

function text(node, value) {
  node.textContent = value == null || value === "" ? "—" : String(value);
}
function showError(message) {
  text(ui.error, message);
  ui.error.hidden = false;
}
function setPlaying(value) {
  playing = Boolean(value && replay);
  anchorWall = performance.now();
  anchorTime = replay ? replay.frames[index].time : 0;
  ui.play.replaceChildren();
  const icon = document.createElement("span");
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = playing ? "Ⅱ" : "▶";
  ui.play.append(
    icon,
    document.createTextNode(playing ? " Pause replay" : " Play replay"),
  );
  ui.play.setAttribute("aria-pressed", String(playing));
}
function safeMeta(value) {
  return typeof value === "string" && value.length ? value : "Not supplied";
}
function loadReplay(data) {
  validateReplay(data);
  setPlaying(false);
  replay = data;
  index = 0;
  lastRendered = -1;
  viewport = poseViewport(replay.frames, ui["fit-view"].checked);
  ui.error.hidden = true;
  ui.seek.max = String(replay.frames.length - 1);
  ["seek", "play", "restart", "next"].forEach((id) => {
    ui[id].disabled = false;
  });
  const p = replay.provenance,
    synthetic = p.kind === "synthetic_diagnostic",
    recording = p.kind === "recording_inference",
    localDiagnostic = p.evaluation_scope === "same_condition_held_time";
  ui.notice.className = `notice${synthetic ? " diagnostic" : ""}`;
  ui.notice.replaceChildren();
  const badge = document.createElement("span");
  badge.className = "notice-label";
  badge.textContent = synthetic
    ? "SYNTHETIC DIAGNOSTIC"
    : localDiagnostic
      ? "LOCAL MOTION DIAGNOSTIC"
    : recording
      ? "RECORDED CSI INFERENCE"
      : "DATASET INFERENCE";
  const note = document.createElement("p");
  note.textContent = synthetic
    ? "Generated signals and illustrative joints verify the software path. This replay is not evidence of Wi-Fi pose accuracy or generalization."
    : localDiagnostic
      ? "Held time blocks from the same person, room and action. Dense reference poses are required. This result does not establish unseen-room or unseen-person performance."
    : recording
      ? "Inference from a CSI recording without pose labels. Subject and environment may be unknown; this replay does not establish held-out accuracy or supported deployment conditions."
      : "Recorded dataset inference. Inspect the dataset, split and checkpoint below. Replay provenance is supplied by the exporter; this viewer does not independently certify it.";
  ui.notice.append(badge, note);
  text(ui["provenance-note"], safeMeta(p.note));
  text(ui["confidence-description"], p.confidence_available === false
    ? "Unavailable. This model has no uncertainty estimator. Empty indicators are not confidence scores."
    : (p.confidence_definition || "Uncalibrated Gaussian mass within a fixed image radius. Not a probability of correctness."));
  text(
    ui["meta-dataset"],
    `${safeMeta(p.dataset)} / ${synthetic ? "diagnostic" : "recorded inference"}`,
  );
  text(
    ui["meta-domain"],
    `${safeMeta(p.subject)} / ${safeMeta(p.environment)}`,
  );
  text(ui["meta-split"], `${safeMeta(p.split)} / ${safeMeta(p.sequence)}`);
  text(ui["meta-checkpoint"], safeMeta(p.checkpoint));
  const intervals = replay.frames
    .slice(1)
    .map((f, i) => f.time - replay.frames[i].time)
    .sort((a, b) => a - b);
  const dt = intervals.length
    ? intervals[Math.floor(intervals.length / 2)]
    : null;
  text(ui.cadence, dt ? `${(1 / dt).toFixed(1)} Hz` : "—");
  text(
    ui.duration,
    formatTime(replay.frames.at(-1).time - replay.frames[0].time),
  );
  ui["show-gt"].checked = false;
  ui["show-gt"].disabled = !replay.frames.some((f) =>
    Array.isArray(f.ground_truth),
  );
  ui["joint-list"].replaceChildren();
  jointRows = replay.joint_names.map((name) => {
    const row = document.createElement("div");
    row.className = "joint-row";
    const info = document.createElement("div");
    info.className = "joint-info";
    const title = document.createElement("span");
    title.className = "joint-name";
    title.textContent = name.replaceAll("_", " ");
    title.title = name;
    const score = document.createElement("span");
    score.className = "joint-score";
    info.append(title, score);
    const bar = document.createElement("div");
    bar.className = "joint-bar";
    const fill = document.createElement("i");
    bar.append(fill);
    row.append(info, bar);
    ui["joint-list"].append(row);
    return { score, fill };
  });
  renderMetrics(synthetic ? null : replay.metrics);
  render();
}
function renderMetrics(metrics) {
  ui.evaluation.replaceChildren();
  ui.evaluation.hidden = true;
  if (!metrics) return;
  const supported = [
    [
      "pck",
      typeof metrics.pck_threshold === "number"
        ? `PCK @ ${metrics.pck_threshold.toFixed(2)}`
        : "PCK",
    ],
    ["pck_0.2", "PCK @ 0.2"],
    ["pck_0_2", "PCK @ 0.2"],
    ["nje", "NORMALIZED JOINT ERROR"],
    ["normalized_joint_error", "NORMALIZED JOINT ERROR"],
    ["latency_ms", "LATENCY / MS"],
    ["fps", "INFERENCE FPS"],
  ];
  for (const [key, label] of supported) {
    if (typeof metrics[key] !== "number" || !Number.isFinite(metrics[key]))
      continue;
    const item = document.createElement("span"),
      value = document.createElement("strong");
    value.textContent = metrics[key].toFixed(3);
    item.append(document.createTextNode(`${label} `), value);
    ui.evaluation.append(item);
    ui.evaluation.hidden = false;
  }
  if (!ui.evaluation.hidden) {
    const scope = document.createElement("span");
    scope.className = "evaluation-scope";
    scope.textContent =
      typeof metrics.scope === "string"
        ? metrics.scope
        : "Exporter-supplied evaluation. See the experiment report for scope.";
    ui.evaluation.prepend(scope);
  }
}
function canvasSize(canvas) {
  const rect = canvas.getBoundingClientRect(),
    ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * ratio)),
    height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width: rect.width, height: rect.height, ratio };
}
function drawPose(frame) {
  const { width: w, height: h, ratio } = canvasSize(poseCanvas);
  poseCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
  poseCtx.clearRect(0, 0, w, h);
  poseCtx.fillStyle = "#121f22";
  poseCtx.fillRect(0, 0, w, h);
  const side = Math.max(1, Math.min(w - 90, h - 100)),
    padX = (w - side) / 2,
    padY = (h - side) / 2,
    pw = side,
    ph = side;
  const project = (xy) => [
    padX + ((xy[0] - viewport.x) / viewport.span) * pw,
    padY + ((xy[1] - viewport.y) / viewport.span) * ph,
  ];
  poseCtx.lineWidth = 1;
  for (let i = 0; i <= 10; i++) {
    poseCtx.strokeStyle = i === 5 ? "#28403f" : "#1d3031";
    poseCtx.beginPath();
    poseCtx.moveTo(padX + (i * pw) / 10, padY);
    poseCtx.lineTo(padX + (i * pw) / 10, padY + ph);
    poseCtx.stroke();
    poseCtx.beginPath();
    poseCtx.moveTo(padX, padY + (i * ph) / 10);
    poseCtx.lineTo(padX + pw, padY + (i * ph) / 10);
    poseCtx.stroke();
  }
  poseCtx.font = "9px monospace";
  poseCtx.fillStyle = "#86a39d";
  poseCtx.textAlign = "right";
  for (let i = 0; i <= 4; i++)
    poseCtx.fillText(
      (viewport.y + (i / 4) * viewport.span).toFixed(2),
      padX - 10,
      padY + (i * ph) / 4 + 3,
    );
  poseCtx.textAlign = "center";
  for (let i = 0; i <= 4; i++)
    poseCtx.fillText(
      (viewport.x + (i / 4) * viewport.span).toFixed(2),
      padX + (i * pw) / 4,
      padY + ph + 15,
    );
  poseCtx.save();
  poseCtx.beginPath();
  poseCtx.rect(padX - 8, padY - 8, pw + 16, ph + 16);
  poseCtx.clip();
  if (frame && frame.status !== "invalid_signal" && Array.isArray(frame.pose)) {
    if (ui["show-gt"].checked && Array.isArray(frame.ground_truth))
      drawSkeleton(
        frame.ground_truth,
        null,
        "#d38b61",
        project,
        true,
        frame.ground_truth_valid,
      );
    drawSkeleton(frame.pose, replay.provenance.confidence_available === false ? null : frame.confidence, "#68dfda", project, false);
  }
  poseCtx.restore();
}
function drawSkeleton(
  points,
  confidence,
  color,
  project,
  reference,
  valid = null,
) {
  poseCtx.lineCap = "round";
  poseCtx.lineJoin = "round";
  poseCtx.setLineDash(reference ? [5, 5] : []);
  for (const [a, b] of replay.edges) {
    if (valid && (!valid[a] || !valid[b])) continue;
    const alpha = confidence
      ? 0.2 + 0.8 * Math.min(confidence[a], confidence[b])
      : 0.7;
    const pa = project(points[a]),
      pb = project(points[b]);
    poseCtx.globalAlpha = alpha;
    poseCtx.lineWidth = reference ? 1.7 : 3;
    poseCtx.strokeStyle = color;
    if (!reference) {
      poseCtx.shadowColor = color;
      poseCtx.shadowBlur = 9 * alpha;
    }
    poseCtx.beginPath();
    poseCtx.moveTo(...pa);
    poseCtx.lineTo(...pb);
    poseCtx.stroke();
    poseCtx.shadowBlur = 0;
  }
  poseCtx.setLineDash([]);
  points.forEach((point, i) => {
    if (valid && !valid[i]) return;
    const [x, y] = project(point),
      c = confidence ? confidence[i] : 0.7;
    poseCtx.globalAlpha = reference ? 0.7 : 0.3 + 0.7 * c;
    poseCtx.beginPath();
    poseCtx.arc(x, y, reference ? 2.7 : 4.1, 0, Math.PI * 2);
    poseCtx.fillStyle = reference ? color : "#142728";
    poseCtx.fill();
    poseCtx.lineWidth = reference ? 1 : 1.6;
    poseCtx.strokeStyle = color;
    poseCtx.stroke();
    if (!reference) {
      poseCtx.font = "7px monospace";
      poseCtx.fillStyle = "#7eaaa4";
      poseCtx.textAlign = "left";
      poseCtx.fillText(String(i).padStart(2, "0"), x + 8, y - 7);
    }
  });
  poseCtx.globalAlpha = 1;
}
function colorMap(v) {
  const stops = [
    [20, 37, 38],
    [38, 99, 97],
    [100, 177, 158],
    [218, 217, 128],
    [239, 155, 96],
  ];
  const position = Math.max(0, Math.min(1, v)) * 4,
    i = Math.min(3, Math.floor(position)),
    t = position - i;
  return stops[i].map((x, k) => Math.round(x + (stops[i + 1][k] - x) * t));
}
function drawSpectrum(frame) {
  const { width: w, height: h, ratio } = canvasSize(spectrumCanvas);
  spectrumCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
  spectrumCtx.fillStyle = "#182a2a";
  spectrumCtx.fillRect(0, 0, w, h);
  const matrix = frame?.spectrum;
  if (!matrix?.length || !matrix[0]?.length) {
    text(ui["spectrum-shape"], "NO SPECTRUM");
    return;
  }
  const rows = matrix.length,
    cols = matrix[0].length;
  text(ui["spectrum-shape"], `${rows} × ${cols}`);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const rgb = colorMap(matrix[r][c]);
      spectrumCtx.fillStyle = `rgb(${rgb.join(",")})`;
      spectrumCtx.fillRect(
        (c * w) / cols,
        (r * h) / rows,
        Math.ceil(w / cols),
        Math.ceil(h / rows),
      );
    }
}
function render() {
  if (!replay) {
    drawPose(null);
    drawSpectrum(null);
    return;
  }
  const f = replay.frames[index],
    invalid = f.status === "invalid_signal",
    confidenceAvailable = replay.provenance.confidence_available !== false,
    warning = ["low_confidence", "out_of_domain"].includes(f.status);
  ui.seek.value = String(index);
  text(ui["current-time"], formatTime(f.time - replay.frames[0].time));
  text(
    ui["frame-tag"],
    `FRAME ${String(index + 1).padStart(4, "0")} / ${String(replay.frames.length).padStart(4, "0")}`,
  );
  const statusLabels = {
    ok: "SIGNAL ACCEPTED",
    low_confidence: confidenceAvailable ? "LOW CONFIDENCE" : "CONFIDENCE UNAVAILABLE",
    invalid_signal: "SIGNAL REJECTED",
    out_of_domain: "DOMAIN WARNING",
  };
  text(ui["frame-status"], statusLabels[f.status]);
  ui["frame-status"].className =
    `status-badge ${invalid ? "danger" : warning ? "warning" : ""}`;
  ui["stage-message"].textContent = invalid
    ? "Pose withheld\nThe CSI window failed signal-quality checks."
    : "";
  text(
    ui["plot-mode"],
    ui["fit-view"].checked ? "SEQUENCE FIT / 2D" : "ANNOTATION PLANE / 2D",
  );
  poseCanvas.setAttribute(
    "aria-label",
    `2D pose, frame ${index + 1} of ${replay.frames.length}. ${statusLabels[f.status]}. ${invalid ? "Estimate withheld." : `${replay.joint_names.length} predicted joints. Reference labels ${ui["show-gt"].checked ? "shown" : "hidden"}.`}`,
  );
  const mean = f.confidence.reduce((a, b) => a + b, 0) / f.confidence.length;
  text(ui["mean-confidence"], invalid ? "Withheld" : confidenceAvailable ? mean.toFixed(3) : "Unavailable");
  text(
    ui.latency,
    f.latency_ms == null ? "Not measured" : `${f.latency_ms.toFixed(1)} ms`,
  );
  text(
    ui["gt-legend"],
    ui["show-gt"].checked && f.ground_truth
      ? "Reference label"
      : "Reference hidden",
  );
  if (ui["show-gt"].checked && f.ground_truth) {
    const dot = document.createElement("i");
    dot.className = "truth";
    ui["gt-legend"].prepend(dot);
  }
  jointRows.forEach(({ score, fill }, i) => {
    text(score, invalid || !confidenceAvailable ? "—" : f.confidence[i].toFixed(2));
    fill.style.width = `${invalid || !confidenceAvailable ? 0 : f.confidence[i] * 100}%`;
    fill.className = f.confidence[i] < 0.5 ? "low" : "";
  });
  ui["failure-detail"].className =
    `failure-detail ${invalid ? "danger" : warning ? "warning" : ""}`;
  const heading = document.createElement("strong"),
    detail = document.createElement("span");
  heading.textContent = invalid
    ? "Estimate withheld"
    : warning
      ? "Use with caution"
      : "Frame diagnostics";
  detail.textContent = f.reasons?.length
    ? f.reasons.join(" · ")
    : "No signal-quality flags were supplied for this frame. This does not establish pose accuracy.";
  ui["failure-detail"].replaceChildren(heading, detail);
  drawPose(f);
  drawSpectrum(f);
  lastRendered = index;
}
function tick(now) {
  if (playing && replay) {
    const target = anchorTime + ((now - anchorWall) / 1000) * speed;
    index = frameAtTime(replay.frames, target);
    if (target >= replay.frames.at(-1).time) setPlaying(false);
    if (index !== lastRendered) render();
  }
  requestAnimationFrame(tick);
}
ui.play.addEventListener("click", () => {
  if (replay && index === replay.frames.length - 1 && !playing) {
    index = 0;
    render();
  }
  setPlaying(!playing);
});
ui.restart.addEventListener("click", () => {
  index = 0;
  setPlaying(false);
  render();
});
ui.next.addEventListener("click", () => {
  setPlaying(false);
  if (replay) index = Math.min(index + 1, replay.frames.length - 1);
  render();
});
ui.seek.addEventListener("input", () => {
  index = Number(ui.seek.value);
  setPlaying(false);
  render();
});
ui.speed.addEventListener("change", () => {
  speed = Number(ui.speed.value);
  setPlaying(playing);
});
ui["show-gt"].addEventListener("change", render);
ui["fit-view"].addEventListener("change", () => {
  viewport = poseViewport(replay?.frames || [], ui["fit-view"].checked);
  render();
});
$("open-replay").addEventListener("click", () => $("replay-file").click());
$("replay-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    if (file.size > 100 * 1024 * 1024)
      throw new Error(
        "Choose a replay smaller than 100 MiB. Export a shorter sequence for inspection.",
      );
    loadReplay(JSON.parse(await file.text()));
  } catch (error) {
    showError(
      `Could not open replay. ${error.message} The previous replay is preserved.`,
    );
  }
  event.target.value = "";
});
document.addEventListener("keydown", (event) => {
  if (
    ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName) ||
    !replay
  )
    return;
  if (event.code === "Space") {
    event.preventDefault();
    ui.play.click();
  }
  if (event.code === "ArrowRight" || event.code === "ArrowLeft") {
    event.preventDefault();
    setPlaying(false);
    index = Math.max(
      0,
      Math.min(
        replay.frames.length - 1,
        index + (event.code === "ArrowRight" ? 1 : -1),
      ),
    );
    render();
  }
});
const resizeObserver = new ResizeObserver(() => render());
resizeObserver.observe(poseCanvas.parentElement);
resizeObserver.observe(spectrumCanvas.parentElement);
render();
requestAnimationFrame(tick);
fetch("/api/replay", { cache: "no-store" })
  .then(async (response) => {
    if (!response.ok)
      throw new Error(
        response.status === 404
          ? "No default replay is configured. Use Open replay to load an exported JSON file."
          : `Local server returned ${response.status}.`,
      );
    return response.json();
  })
  .then(loadReplay)
  .catch((error) => {
    ui.notice.className = "notice";
    ui.notice.querySelector(".notice-label").textContent = "READY";
    ui.notice.querySelector("p").textContent = error.message;
  });
