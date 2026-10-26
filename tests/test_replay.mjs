import test from "node:test";
import assert from "node:assert/strict";
import {
  validateReplay,
  frameAtTime,
  formatTime,
  poseViewport,
} from "../ghostmesh/web/replay.js";

function fixture() {
  return {
    schema_version: 1,
    provenance: {
      kind: "synthetic_diagnostic",
      coordinate_system: "normalized_image",
    },
    joint_names: ["root", "tip"],
    edges: [[0, 1]],
    frames: [0, 0.05, 0.13, 0.2].map((time) => ({
      time,
      pose: [
        [0.4, 0.2],
        [0.6, 0.8],
      ],
      confidence: [0.8, 0.2],
      spectrum: [
        [0, 0.5, 1],
        [1, 0.5, 0],
      ],
      status: "ok",
      reasons: [],
      latency_ms: null,
    })),
  };
}

test("diagnostic and actual-inference provenance are explicit", () => {
  assert.equal(
    validateReplay(fixture()).provenance.kind,
    "synthetic_diagnostic",
  );
  const actual = fixture();
  actual.provenance.kind = "dataset_inference";
  assert.equal(validateReplay(actual), actual);
  actual.provenance.kind = "recording_inference";
  assert.equal(validateReplay(actual), actual);
  actual.provenance.kind = "live";
  assert.throws(() => validateReplay(actual), /provenance/);
});
test("timestamp seek follows actual irregular timestamps and clamps at ends", () => {
  const frames = fixture().frames;
  for (const [time, index] of [
    [-1, 0],
    [0, 0],
    [0.049, 0],
    [0.05, 1],
    [0.12, 1],
    [0.13, 2],
    [0.1999, 2],
    [0.2, 3],
    [100, 3],
  ])
    assert.equal(frameAtTime(frames, time), index);
});
test("an invalid signal can withhold pose but must keep the joint contract", () => {
  const data = fixture(),
    f = data.frames[1];
  f.status = "invalid_signal";
  f.pose = null;
  f.confidence = [0, 0];
  assert.equal(validateReplay(data), data);
  f.status = "ok";
  assert.throws(() => validateReplay(data), /pose/);
  f.status = "invalid_signal";
  f.confidence = [];
  assert.throws(() => validateReplay(data), /confidence/);
});
test("all failure statuses and valid reference masks are accepted", () => {
  const data = fixture();
  data.frames.forEach((f, i) => {
    f.status = ["ok", "low_confidence", "invalid_signal", "out_of_domain"][i];
    f.ground_truth = structuredClone(f.pose);
    f.ground_truth_valid = [true, false];
  });
  assert.equal(validateReplay(data), data);
});
test("temporal duplicates, reversal and nonfinite coordinates are rejected", () => {
  for (const change of [
    (d) => (d.frames[1].time = 0),
    (d) => (d.frames[2].time = 0.02),
    (d) => (d.frames[0].pose[0][0] = NaN),
    (d) => (d.frames[0].pose[0][0] = Infinity),
    (d) => (d.frames[0].confidence[1] = 1.1),
  ]) {
    const data = fixture();
    change(data);
    assert.throws(() => validateReplay(data));
  }
});
test("malformed spectra and incompatible skeletons are rejected", () => {
  for (const change of [
    (d) => (d.edges = [[0, 2]]),
    (d) => (d.frames[0].spectrum = [[0, 1], [0]]),
    (d) => (d.frames[0].spectrum = [[2]]),
    (d) => (d.frames[0].ground_truth = [[0, 0]]),
    (d) => (d.frames[0].ground_truth_valid = [true, 1]),
    (d) => (d.frames[0].status = "great"),
    (d) => (d.frames[0].latency_ms = -1),
  ]) {
    const data = fixture();
    change(data);
    assert.throws(() => validateReplay(data));
  }
});
test("format time is stable across minute changes and invalid values", () => {
  assert.equal(formatTime(0), "00:00.00");
  assert.equal(formatTime(65.2), "01:05.20");
  assert.equal(formatTime(59.999), "01:00.00");
  assert.equal(formatTime(NaN), "00:00.00");
  assert.equal(formatTime(-2), "00:00.00");
});

test("fit view is isotropic, fixed across sequence and independent of reference labels", () => {
  const data = fixture();
  data.frames[3].pose = [
    [0.2, 0.1],
    [0.7, 0.9],
  ];
  const view = poseViewport(data.frames);
  assert.deepEqual(view, { x: 0, y: 0, span: 1 });
  const small = fixture();
  small.frames.forEach((f) => {
    f.pose = [
      [0.3, 0.2],
      [0.4, 0.5],
    ];
    f.ground_truth = [
      [0, 0],
      [1, 1],
    ];
  });
  const fitted = poseViewport(small.frames);
  assert.ok(Math.abs(fitted.span - 0.375) < 1e-12);
  small.frames.forEach((f) => {
    f.ground_truth = [
      [0.4, 0.4],
      [0.6, 0.6],
    ];
  });
  assert.deepEqual(poseViewport(small.frames), fitted);
  assert.deepEqual(poseViewport(small.frames, false), { x: 0, y: 0, span: 1 });
  small.frames.forEach((f) => {
    f.status = "invalid_signal";
    f.pose = null;
  });
  assert.deepEqual(poseViewport(small.frames), { x: 0, y: 0, span: 1 });
});
