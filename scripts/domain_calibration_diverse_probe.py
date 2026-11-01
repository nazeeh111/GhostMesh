"""E01-E03 only: 108 labels across four people/all actions; fifth person held out."""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ROOMS = ("E01", "E02", "E03")
ARMS = ("frame_nll", "factorized_dynamic_nll")
SHRINK = (0., .25, .5, .75, 1.)
RIDGE = (.1, 1., 10., 100.)
HELPER_SHA = "847ee479af3012b6ac2926332d8e572da7cfe7f2cc1d3f9368fb096c185f54f6"
CONFIG = dict(rooms=ROOMS, arms=ARMS, calibration_labels=108, people_per_calibration=4,
              actions_per_person=27, labels_per_recording=1, rotations_per_room=5,
              reference_rule="31 + int.from_bytes(sha256(('ghostmesh-diverse108-v1|' + recordID).encode()).digest(), 'big') % 265",
              shrink_grid=SHRINK, ridge_grid=RIDGE, window=8, bins=19, threads=1,
              collection_burden="108 calibration recordings from4 people, versus12 recordings from2 people in the prior120-label design",
              hyperparameter_selection="Minimum mean NJE across all15 development person/room rotations; ties retain grid order",
              scope="Supervised calibration DEVELOPMENT ONLY, E01-E03, frozen single-seed models; no E04 or final records")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_end(record_id):
    require(len(record_id.split('/')) == 3 and record_id.split('/')[0] in ROOMS, "Forbidden reference record")
    return 31 + int.from_bytes(sha256(("ghostmesh-diverse108-v1|" + record_id).encode()).digest(), "big") % 265


def rotation_mask(subjects, held_person):
    people = set(subjects)
    require(len(people) == 5 and held_person in people, "Unexpected rotation people")
    mask = subjects != held_person
    require(int(mask.sum()) == 108 and int((~mask).sum()) == 27, "Expected four calibration people/all27actions")
    return mask


def run(root, manifest_path, out):
    require(not out.exists(), "Refusing to overwrite diverse calibration study")
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(root))
    study, cache = root / "work/domain-neural-v2", root / "work/domain-dev-v2"
    reg, original_results, receipt = read(study / "registration.json"), read(study / "results.json"), read(cache / "receipt.json")
    require(original_results["registration"] == reg and digest(cache / "receipt.json") == reg["cache_receipt_sha256"] and receipt["arrays"] == reg["cache_arrays_sha256"], "Original study/cache binding differs")
    require(all(k.split('/')[0] in ROOMS for k in receipt["sequences"]), "Forbidden source-cache room")
    inputs = {}
    def bind(path, expected=None):
        value = digest(path)
        require(expected is None or value == expected, f"Hash mismatch: {path}")
        inputs[str(path.resolve())] = value
    bind(manifest_path, receipt["manifest_sha256"])
    manifest = read(manifest_path)
    records = sorted([r for r in manifest["sequences"] if r["environment"] in ROOMS], key=lambda r: r["id"])
    require(len(records) == 405 and set(r["id"] for r in records) == set(receipt["sequences"]), "Source ROI differs")
    plan = []
    for row in records:
        path = (manifest_path.parent / row["path"]).resolve()
        require(path.parent == manifest_path.parent.resolve() and path.name == row["id"].replace('/', '-') + '.npz', "Unexpected source path")
        bind(path, row["sha256"])
        plan.append(dict(id=row["id"], room=row["environment"], subject=row["subject"], action=row["action"],
                         frame_index=reference_end(row["id"]), source_path=str(path), source_sha256=row["sha256"]))
    snapshot = study / "executed-source/scripts"
    for name in ("domain_neural_probe.py", "domain_probe.py"):
        bind(snapshot / name, reg["source_sha256"][f"scripts/{name}"])
        bind(root / "scripts" / name)
    require(ast.dump(ast.parse((snapshot / "domain_neural_probe.py").read_text())) == ast.dump(ast.parse((root / "scripts/domain_neural_probe.py").read_text())), "Neural implementation changed")
    helper_path = root / "work/domain-calibration-v2-checked/executed-source.py"
    bind(helper_path, HELPER_SHA)
    for name in ("ghostmesh/models.py", "ghostmesh/metrics.py"):
        bind(root / name, reg["source_sha256"][name])
    original_preprocess = manifest_path.parents[2] / "ghostmesh/preprocess.py"
    bind(original_preprocess)
    bind(root / "ghostmesh/preprocess.py", digest(original_preprocess))
    for path in (study / "registration.json", study / "results.json", cache / "receipt.json"):
        bind(path)
    for name, value in receipt["arrays"].items():
        require(Path(name).name == name, "Unsafe cache filename")
        bind(cache / name, value)
    for room in ROOMS:
        bind(study / f"{room}-training-stats.npz")
        for arm in ARMS:
            old = next(r for r in original_results["results"] if r["room"] == room and r["arm"] == arm)
            bind(study / f"{room}-{arm}/best.pt", old["checkpoint_sha256"])
    # Freeze all sources, record identities and deterministic endpoints BEFORE
    # decoding cached labels/features or original calibration recordings.
    out.mkdir(parents=True)
    write(out / "registration.json", dict(config=CONFIG, script_sha256=digest(__file__), inputs_sha256=inputs,
                                         reference_plan=plan, unique_reference_frames=405,
                                         freeze_stage="Before any source calibration array decode in this probe",
                                         runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__), threads=1)))
    (out / "executed-source.py").write_bytes(Path(__file__).read_bytes())
    rows, parameters, parity, coverage = [], {}, dict(reference_windows=0, cache_intersections=0, original_models=0), {}
    try:
        helper = load_module("diverse_checked_calibration", helper_path)
        neural = load_module("diverse_frozen_neural", snapshot / "domain_neural_probe.py")
        frozen = load_module("diverse_frozen_features", snapshot / "domain_probe.py")
        generic = load_module("diverse_generic_features", root / "scripts/domain_probe.py")
        require(neural.CONFIG == reg["config"], "Neural configuration differs")
        a = {k: np.load(cache / f"{k}.npy", allow_pickle=False) for k in ("room", "subject", "sequence", "end")}
        require(set(a["room"]) == set(ROOMS) and all(s.split('/')[0] in ROOMS for s in a["sequence"]), "Forbidden cached room")
        a.update({k: np.load(cache / f"{k}.npy", allow_pickle=False) for k in ("moments", "dynamic", "pose", "valid")})
        require(a["moments"].shape == a["dynamic"].shape == (14985, 8, 114) and a["valid"].all(), "Unexpected source cache")
        for room in ROOMS:
            reference = {k: [] for k in ("moments", "dynamic", "pose", "valid", "sequence", "subject", "end")}
            for entry in [p for p in plan if p["room"] == room]:
                with np.load(entry["source_path"], allow_pickle=False) as archive:
                    csi, y, v = archive["csi"], archive["pose"], archive["valid"]
                require(csi.shape == (297, 3, 114, 10) and y.shape == (297, 17, 2), "Unexpected source recording")
                end = entry["frame_index"]
                m, d = generic.features(csi[end - 7:end + 1], bins=19)
                om, od = frozen.features(csi[end - 7:end + 1])
                require(np.array_equal(m, om) and np.array_equal(d, od), "Old/generic reference feature mismatch")
                parity["reference_windows"] += 1
                existing = np.flatnonzero((a["sequence"] == entry["id"]) & (a["end"] == end))
                if len(existing):
                    i = existing.item()
                    require(np.array_equal(m, a["moments"][i]) and np.array_equal(d, a["dynamic"][i]) and np.array_equal(y[end], a["pose"][i]), "Reference/cache mismatch")
                    parity["cache_intersections"] += 1
                for key, value in dict(moments=m, dynamic=d, pose=y[end], valid=v[end], sequence=entry["id"], subject=entry["subject"], end=end).items():
                    reference[key].append(value)
            reference = {k: np.asarray(v) for k, v in reference.items()}
            require(len(reference["pose"]) == 135 and reference["valid"].all() and np.isfinite(reference["pose"]).all(), "Reference coverage differs")
            np.savez_compressed(out / f"{room}-references.npz", **reference)
            train, _, held = neural.masks(a, room)
            require(int(held.sum()) == 4995, "Unexpected development room coverage")
            norm = {}
            with np.load(study / f"{room}-training-stats.npz", allow_pickle=False) as stats:
                for key in ("moments", "dynamic"):
                    mean, scale = stats[key + "_mean"], stats[key + "_scale"]
                    require(np.array_equal(mean, a[key][train].mean((0, 1))) and np.array_equal(scale, np.maximum(a[key][train].std((0, 1)), .1)), "Source normalization differs")
                    norm[key] = (mean, scale)
            def normalized(values):
                return [((values[k] - norm[k][0]) / norm[k][1]).astype(np.float32) for k in ("moments", "dynamic")]
            hm, hd = normalized({k: a[k][held] for k in ("moments", "dynamic")})
            rm, rd = normalized(reference)
            predictions, ref_predictions = {}, {}
            for arm in ARMS:
                checkpoint = torch.load(study / f"{room}-{arm}/best.pt", map_location="cpu", weights_only=True)
                require(checkpoint["config"] == reg["config"] and checkpoint["arm"] == arm and checkpoint["source_sha256"] == reg["source_sha256"] and checkpoint["cache_receipt_sha256"] == reg["cache_receipt_sha256"], "Checkpoint binding differs")
                model = neural.Probe(arm, checkpoint["training_target_stats"])
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                model.eval()
                predictions[arm] = helper.predict(model, hm, hd).astype(float)
                checked = helper.score(predictions[arm], a["pose"][held], a["sequence"][held])
                old = next(r for r in original_results["results"] if r["room"] == room and r["arm"] == arm)["held_development_room"]
                require(abs(checked["nje"] - old["normalized_joint_error"]) < 1e-6 and abs(checked["pck20"] - old["pck"]) < 1e-6, "Original checkpoint accuracy differs")
                parity["original_models"] += 1
                ref_predictions[arm] = helper.predict(model, rm, rd).astype(float)
            source_mean = a["pose"][train].astype(float).mean(0)
            for person in sorted(set(reference["subject"])):
                cal = rotation_mask(reference["subject"], person)
                test = a["subject"][held] == person
                require(int(test.sum()) == 999 and person not in set(reference["subject"][cal]), "Held-person separation differs")
                y, seq = a["pose"][held][test], a["sequence"][held][test]
                cy = reference["pose"][cal]
                key = f"{room}-{person}"
                mean = cy.astype(float).mean(0)
                parameters[key] = dict(source_mean=source_mean.tolist(), target_mean=mean.tolist(), neural={})
                coverage[key] = dict(calibration_people=sorted(set(reference["subject"][cal])), held_person=person,
                                     labels=108, actions=27, calibration_recordings=108, validation_recordings=27,
                                     validation_windows=999, validation_joints=16983, rejected=0)
                def record(method, p):
                    rows.append(dict(room=room, held_person=person, method=method, scores=helper.score(p, y, seq)))
                record("source_mean", np.broadcast_to(source_mean, y.shape))
                record("target_mean", np.broadcast_to(mean, y.shape))
                for arm in ARMS:
                    p, cp = predictions[arm][test], ref_predictions[arm][cal]
                    offset = (cy[:, [11, 12]].mean(1) - cp[:, [11, 12]].mean(1)).mean(0)
                    residual = mean - cp.mean(0)
                    parameters[key]["neural"][arm] = dict(root_shift=offset.tolist(), joint_residual=residual.tolist(), calibration_prediction_mean=cp.mean(0).tolist(), shrink_grid=SHRINK)
                    record(arm + ":zero", p)
                    record(arm + ":root_shift", p + offset)
                    record(arm + ":joint_shift", p + residual)
                    for alpha in SHRINK:
                        record(arm + f":shrink_{alpha:g}", helper.shrink(p, cp, cy, alpha))
                for kind, cx, vx in (("last", rm[cal, -1], hm[test, -1]), ("temporal", rm[cal].reshape(108, -1), hm[test].reshape(999, -1))):
                    for alpha in RIDGE:
                        fit = helper.ridge_fit(cx, cy, alpha)
                        np.savez(out / f"{key}-ridge-{kind}-{alpha:g}.npz", **fit)
                        record(f"ridge_{kind}:{alpha:g}", helper.ridge_predict(fit, vx))
                print(json.dumps(dict(room=room, held_person=person, labels=108,
                                      target_mean_nje=next(r for r in rows if r["room"] == room and r["held_person"] == person and r["method"] == "target_mean")["scores"]["nje"])), flush=True)
        require(all(digest(Path(p)) == h for p, h in inputs.items()), "Frozen inputs changed")
        require(parity["reference_windows"] == 405 and parity["original_models"] == 6 and len(coverage) == 15 and len(rows) == 390, "Incomplete study")
        methods = [r["method"] for r in rows if r["room"] == "E01" and r["held_person"] == "S01"]
        aggregate = {}
        for method in methods:
            values = [r for r in rows if r["method"] == method]
            aggregate[method] = dict(macro_nje=float(np.mean([r["scores"]["nje"] for r in values])),
                                     macro_pck05=float(np.mean([r["scores"]["pck05"] for r in values])),
                                     macro_pck20=float(np.mean([r["scores"]["pck20"] for r in values])),
                                     macro_relative_motion_r2=float(np.mean([r["scores"]["motion"]["relative"]["r2"] for r in values])),
                                     per_room={room: float(np.mean([r["scores"]["nje"] for r in values if r["room"] == room])) for room in ROOMS})
        selected = {arm: min([arm + f":shrink_{alpha:g}" for alpha in SHRINK], key=lambda m: aggregate[m]["macro_nje"]) for arm in ARMS}
        selected["ridge"] = min([f"ridge_{kind}:{alpha:g}" for kind in ("last", "temporal") for alpha in RIDGE], key=lambda m: aggregate[m]["macro_nje"])
        write(out / "parameters.json", parameters)
        limits = "One source-model seed;15 rotations are correlated, only3 rooms/15 unique people. Hyperparameters selected on these same development results, not independent confirmation. Broader references change person/action/time coverage together and require4 people performing all27 actions, a higher collection burden than2people/6actions. All source categories are calibration-represented. No E04 arrays, final records, phase features or neural training. Annotation noise and stride8 (~0.8s) limit motion inference."
        write(out / "results.json", dict(config=CONFIG, coverage=coverage, parity_checks=parity, rows=rows,
                                         aggregates=aggregate, development_selection=selected, limits=limits, seconds=time.monotonic() - started))
        show = ["source_mean", "target_mean"] + [arm + ":zero" for arm in ARMS] + [arm + ":root_shift" for arm in ARMS] + [arm + ":joint_shift" for arm in ARMS] + list(selected.values())
        text = ["# Diverse108 source-room calibration proxy", "", CONFIG["scope"], "", "Five leave-one-person-out rotations in each of E01–E03. Each calibration set uses four people × 27 actions × one deterministic reference frame = 108 labels. Each held person contributes 27 recordings / 999 cached endpoints / 16,983 valid joints. There are 405 unique reference frames across the 15 correlated rotations, not 1,620 independent labels or 15 independent rooms.", "", "Collection requires 108 labeled recordings from four people, versus 12 recordings from two people in the earlier 120-label design: nine times as many recordings and twice as many people. This changes person, action and time coverage together, not one isolated causal factor. It does not change the reserved final calibration plan automatically.", "", "Every neural adapter and direct ridge receives identical references. Frozen weights and source-only normalization are reused. All 405 reference features equal the archived eight-frame implementation; all six source checkpoint accuracies match original reports. No E04/final arrays or phase calibration were accessed.", "", "| Method | Mean NJE | PCK@.05 | PCK@.2 | Mean relative motion R² | E01 NJE | E02 NJE | E03 NJE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method in dict.fromkeys(show):
            v = aggregate[method]
            text.append(f"|{method}|{v['macro_nje']:.6f}|{v['macro_pck05']:.6f}|{v['macro_pck20']:.6f}|{v['macro_relative_motion_r2']:.6f}|{v['per_room']['E01']:.6f}|{v['per_room']['E02']:.6f}|{v['per_room']['E03']:.6f}|")
        text += ["", "| Room/person | Target mean NJE | Selected frame NJE | Selected candidate NJE | Selected ridge NJE | Candidate relative motion R² |", "|---|---:|---:|---:|---:|---:|"]
        lookup = {(r["room"], r["held_person"], r["method"]): r["scores"] for r in rows}
        for key, c in coverage.items():
            room, person = key.split('-')
            values = [lookup[room, person, name] for name in ("target_mean", selected[ARMS[0]], selected[ARMS[1]], selected["ridge"])]
            text.append(f"|{key}|{values[0]['nje']:.6f}|{values[1]['nje']:.6f}|{values[2]['nje']:.6f}|{values[3]['nje']:.6f}|{values[2]['motion']['relative']['r2']:.6f}|")
        text += ["", "Grid selections minimize mean NJE over these same15development rotations; all grids/parameters/subject scores are retained. R² subtracts each recording's mean prediction and annotation;0is the static oracle recording mean, negative means fluctuations hurt squared error. Any static-reference benefit is distinct from useful dynamic prediction.", "", limits]
        with (out / "SUMMARY.md").open("x") as stream:
            stream.write("\n".join(text) + "\n")
        write(out / "artifact_manifest.json", dict(script_sha256=digest(__file__), outputs_sha256={p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}))
    except Exception as exc:
        write(out / "failure.json", dict(type=type(exc).__name__, error=str(exc), completed_rows=len(rows), parity_checks=parity))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.manifest.resolve(), args.out.resolve())
