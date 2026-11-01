/* Replay validation is independent of the DOM so tools can test the file contract. */
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const object = (value) =>
  value !== null && typeof value === "object" && !Array.isArray(value);
export const STATUSES = new Set([
  "ok",
  "low_confidence",
  "invalid_signal",
  "out_of_domain",
]);

export function validateReplay(data) {
  if (!object(data) || data.schema_version !== 1)
    throw new Error("Use a GhostMesh replay with schema_version 1.");
  if (
    !object(data.provenance) ||
    ![
      "synthetic_diagnostic",
      "dataset_inference",
      "recording_inference",
    ].includes(data.provenance.kind)
  )
    throw new Error(
      "Replay provenance must identify dataset inference, recording inference or a synthetic diagnostic.",
    );
  if (data.provenance.coordinate_system !== "normalized_image")
    throw new Error("Replay coordinates must use normalized_image.");
  if (data.provenance.confidence_available != null &&
      typeof data.provenance.confidence_available !== "boolean")
    throw new Error("Confidence availability must be a boolean.");
  if (
    !Array.isArray(data.joint_names) ||
    data.joint_names.length < 1 ||
    data.joint_names.length > 100 ||
    data.joint_names.some(
      (n) => typeof n !== "string" || !n.length || n.length > 120,
    )
  )
    throw new Error("Replay needs 1–100 named joints.");
  const joints = data.joint_names.length;
  if (
    !Array.isArray(data.edges) ||
    data.edges.length > 300 ||
    data.edges.some(
      (e) =>
        !Array.isArray(e) ||
        e.length !== 2 ||
        e.some((i) => !Number.isInteger(i) || i < 0 || i >= joints),
    )
  )
    throw new Error("Replay skeleton edges must reference existing joints.");
  if (
    !Array.isArray(data.frames) ||
    data.frames.length < 1 ||
    data.frames.length > 200000
  )
    throw new Error("Replay needs 1–200,000 frames.");
  let previous = -Infinity;
  for (let i = 0; i < data.frames.length; i++) {
    const f = data.frames[i];
    const fail = (message) => {
      throw new Error(`Frame ${i + 1}: ${message}`);
    };
    if (!object(f) || !finite(f.time) || f.time < 0 || f.time <= previous)
      fail("timestamps must be finite, nonnegative and strictly increasing.");
    previous = f.time;
    if (!STATUSES.has(f.status)) fail("unknown signal status.");
    const poseOK = (p) =>
      Array.isArray(p) &&
      p.length === joints &&
      p.every((xy) => Array.isArray(xy) && xy.length === 2 && xy.every(finite));
    if (
      !poseOK(f.pose) &&
      !(
        f.status === "invalid_signal" &&
        (f.pose == null || (Array.isArray(f.pose) && !f.pose.length))
      )
    )
      fail("pose must contain a finite [x,y] coordinate for each joint.");
    if (
      !Array.isArray(f.confidence) ||
      f.confidence.length !== joints ||
      f.confidence.some((c) => !finite(c) || c < 0 || c > 1)
    )
      fail("joint confidence must be between zero and one.");
    if (f.ground_truth != null && !poseOK(f.ground_truth))
      fail("reference labels must match the skeleton.");
    if (
      f.ground_truth_valid != null &&
      (!Array.isArray(f.ground_truth_valid) ||
        f.ground_truth_valid.length !== joints ||
        f.ground_truth_valid.some((v) => typeof v !== "boolean"))
    )
      fail("reference validity must contain one boolean per joint.");
    if (
      f.spectrum != null &&
      (!Array.isArray(f.spectrum) ||
        f.spectrum.length > 2048 ||
        f.spectrum.some(
          (row) =>
            !Array.isArray(row) ||
            row.length > 2048 ||
            row.some((v) => !finite(v) || v < 0 || v > 1),
        ) ||
        (f.spectrum.length &&
          f.spectrum.some((row) => row.length !== f.spectrum[0].length)))
    )
      fail("spectrum must be a rectangular matrix normalized to [0,1].");
    if (f.latency_ms != null && (!finite(f.latency_ms) || f.latency_ms < 0))
      fail("latency must be nonnegative.");
    if (
      f.reasons != null &&
      (!Array.isArray(f.reasons) ||
        f.reasons.some((r) => typeof r !== "string" || r.length > 2000))
    )
      fail("diagnostic reasons must be text.");
  }
  if (data.metrics != null && !object(data.metrics))
    throw new Error("Evaluation metrics must be an object.");
  return data;
}

export function frameAtTime(frames, time) {
  let lo = 0,
    hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].time <= time) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

// A single viewport for the whole sequence avoids frame-by-frame camera motion.
// Only model predictions set this display crop; reference labels never influence it.
export function poseViewport(frames, fit = true) {
  if (!fit) return { x: 0, y: 0, span: 1 };
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const frame of frames) {
    if (frame.status === "invalid_signal" || !Array.isArray(frame.pose))
      continue;
    for (const [x, y] of frame.pose) {
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }
  if (!Number.isFinite(minX)) return { x: 0, y: 0, span: 1 };
  const span = Math.min(
    1,
    Math.max(0.35, Math.max(maxX - minX, maxY - minY) * 1.25),
  );
  return {
    x: Math.max(0, Math.min(1 - span, (minX + maxX - span) / 2)),
    y: Math.max(0, Math.min(1 - span, (minY + maxY - span) / 2)),
    span,
  };
}

export function formatTime(time) {
  const safe = Math.max(0, Number.isFinite(time) ? time : 0);
  const centiseconds = Math.round(safe * 100);
  const minutes = Math.floor(centiseconds / 6000);
  return `${String(minutes).padStart(2, "0")}:${((centiseconds % 6000) / 100).toFixed(2).padStart(5, "0")}`;
}
