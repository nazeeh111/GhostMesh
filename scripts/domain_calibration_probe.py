"""Small E01-E03 supervised calibration proxy; never opens E04 arrays."""
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
ACTIONS = ("A01", "A02", "A03", "A06", "A12", "A19")
ENDS = (31, 60, 89, 118, 147, 176, 205, 234, 263, 292)
SHRINK = (0., .25, .5, .75, 1.)
RIDGE = (.1, 1., 10., 100.)
CONFIG = dict(rooms=ROOMS, arms=ARMS, calibration_actions=ACTIONS, calibration_end_indices=ENDS,
              shrink_grid=SHRINK, ridge_grid=RIDGE, seed=90210, threads=1, window=8, bins=19,
              calibration_people="First two released-cohort people per proxy room",
              validation_people="Remaining three released-cohort people per proxy room",
              labels_per_room=120, selection="Minimum macro-room validation NJE; ties retain grid order",
              scope="DEVELOPMENT ONLY: supervised 120-label room calibration, E01-E03; no E04/final holdout")


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


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


def shrink(p, calibration_prediction, calibration_target, alpha):
    return calibration_target.astype(float).mean(0) + alpha * (p.astype(float) - calibration_prediction.astype(float).mean(0))


def ridge_fit(x, y, alpha):
    """Dual solve; alpha*n convention, fitted intercept, no validation fitting."""
    x, y = x.astype(float), y.reshape(len(y), -1).astype(float)
    xm, ym = x.mean(0), y.mean(0)
    xc = x - xm
    require(alpha > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "Invalid ridge inputs")
    # Explicit contractions avoid BLAS floating-status warnings observed on
    # this host, while retaining the same small positive-definite dual solve.
    gram = np.einsum("if,jf->ij", xc, xc, optimize=False) + alpha * len(x) * np.eye(len(x))
    dual = np.linalg.solve(gram, y - ym)
    residual = np.einsum("ij,jk->ik", gram, dual, optimize=False) - (y - ym)
    require(np.linalg.norm(residual) <= 1e-9 * max(np.linalg.norm(y - ym), 1.), "Ridge solve residual too large")
    weight = np.einsum("if,ij->fj", xc, dual, optimize=False)
    require(np.isfinite(weight).all(), "Nonfinite ridge weights")
    return dict(xmean=xm, ymean=ym, weight=weight)


def ridge_predict(fit, x):
    centered = x.astype(float) - fit["xmean"]
    return (np.einsum("if,fj->ij", centered, fit["weight"], optimize=False) + fit["ymean"]).reshape(-1, 17, 2)


def motion(p, y, groups):
    pr, yr = p[:, [11, 12]].mean(1, keepdims=True), y[:, [11, 12]].mean(1, keepdims=True)
    result = {}
    for name, pred, target in (("absolute", p, y), ("root", pr, yr), ("relative", p - pr, y - yr)):
        pc, yc = pred[groups], target[groups]
        pc, yc = pc - pc.mean(1, keepdims=True), yc - yc.mean(1, keepdims=True)
        pv, yv = float(np.mean(pc ** 2)), float(np.mean(yc ** 2))
        result[name] = dict(r2=1 - float(np.mean((pc - yc) ** 2)) / yv,
                            prediction_variance=pv, annotation_variance=yv,
                            variance_ratio=pv / yv,
                            alignment_cosine=float(np.mean(pc * yc) / np.sqrt(pv * yv)) if pv > 1e-25 else None)
    return result


def score(p, y, sequence):
    p, y = p.astype(float), y.astype(float)
    require(p.shape == y.shape and np.isfinite(p).all(), "Invalid prediction")
    names = sorted(set(sequence))
    groups = np.asarray([np.flatnonzero(sequence == name) for name in names])
    require(groups.shape == (len(names), 37), "Unexpected per-record endpoint count")
    scale = np.linalg.norm(y.max(1) - y.min(1), axis=-1)
    require(bool((scale > 1e-6).all()), "Degenerate reference box")
    error = np.linalg.norm(p - y, axis=-1) / scale[:, None]
    root = np.linalg.norm(p[:, [11, 12]].mean(1) - y[:, [11, 12]].mean(1), axis=-1)
    relative = np.linalg.norm(p - p[:, [11, 12]].mean(1, keepdims=True) - (y - y[:, [11, 12]].mean(1, keepdims=True)), axis=-1)
    rng, permutation = np.random.default_rng(CONFIG["seed"]), np.arange(len(y))
    for group in groups:
        permutation[group] = rng.permutation(group)
    shuffled = np.linalg.norm(p[permutation] - y, axis=-1) / scale[:, None]
    return dict(nje=float(error.mean()), pck05=float((error <= .05).mean()),
                pck10=float((error <= .1).mean()), pck20=float((error <= .2).mean()),
                root_coordinate_error=float(root.mean()), relative_coordinate_error=float(relative.mean()),
                windows=len(y), sequences=len(names), valid_joints=int(error.size),
                motion=motion(p, y, groups), shuffled_nje=float(shuffled.mean()),
                shuffle_nje_minus_aligned=float(shuffled.mean() - error.mean()))


@torch.inference_mode()
def predict(model, moments, dynamic):
    return np.concatenate([model(torch.from_numpy(moments[start:start + 128]),
                                 torch.from_numpy(dynamic[start:start + 128]))["pose"].numpy()
                           for start in range(0, len(moments), 128)])


def calibration_rows(manifest, room):
    require(room in ROOMS, "Forbidden calibration room")
    offset = ROOMS.index(room) * 10
    people = (f"S{offset + 1:02}", f"S{offset + 2:02}")
    rows = [r for r in manifest["sequences"] if r["environment"] == room
            and r["subject"] in people and r["action"] in ACTIONS]
    require(len(rows) == 12 and {r["id"] for r in rows} == {f"{room}/{s}/{a}" for s in people for a in ACTIONS}, "Calibration selection differs")
    return sorted(rows, key=lambda r: r["id"]), people


def run(root, manifest_path, out):
    require(not out.exists(), "Refusing to overwrite calibration probe")
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(root))
    cache, study = root / "work/domain-dev-v2", root / "work/domain-neural-v2"
    registered, completed, receipt = read(study / "registration.json"), read(study / "results.json"), read(cache / "receipt.json")
    require(completed["registration"] == registered, "Study registration differs")
    require(digest(cache / "receipt.json") == registered["cache_receipt_sha256"], "Cache receipt differs")
    require(receipt["arrays"] == registered["cache_arrays_sha256"], "Cache array bindings differ")
    require(all(s.split('/')[0] in ROOMS for s in receipt["sequences"]), "Forbidden room in cache receipt")
    inputs = {}
    def bind(path, expected=None):
        value = digest(path)
        require(expected is None or value == expected, f"Hash mismatch: {path}")
        inputs[str(path.resolve())] = value
        return value
    for path in (cache / "receipt.json", study / "registration.json", study / "results.json"):
        bind(path)
    bind(manifest_path, receipt["manifest_sha256"])
    manifest = read(manifest_path)
    snapshot = study / "executed-source/scripts"
    for name in ("domain_neural_probe.py", "domain_probe.py"):
        bind(snapshot / name, registered["source_sha256"][f"scripts/{name}"])
        bind(root / "scripts" / name)
    require(ast.dump(ast.parse((snapshot / "domain_neural_probe.py").read_text())) == ast.dump(ast.parse((root / "scripts/domain_neural_probe.py").read_text())), "Neural source changed semantically")
    for name in ("ghostmesh/models.py", "ghostmesh/metrics.py"):
        bind(root / name, registered["source_sha256"][name])
    # The generic feature helper was extended to arbitrary windows/bins. Match
    # the exact old preprocessing source, then assert numeric parity below.
    bind(root / "ghostmesh/preprocess.py", digest(manifest_path.parents[2] / "ghostmesh/preprocess.py"))
    neural = load_module("calibration_frozen_neural", snapshot / "domain_neural_probe.py")
    old_features = load_module("calibration_frozen_features", snapshot / "domain_probe.py")
    generic = load_module("calibration_generic_features", root / "scripts/domain_probe.py")
    require(neural.CONFIG == registered["config"], "Neural config differs")
    for name, value in receipt["arrays"].items():
        require(Path(name).name == name, "Unsafe cache path")
        bind(cache / name, value)
    a = {k: np.load(cache / f"{k}.npy", allow_pickle=False) for k in ("room", "sequence", "subject", "end")}
    require(set(a["room"]) == set(ROOMS) and all(s.split('/')[0] in ROOMS for s in a["sequence"]), "Forbidden cached room")
    a.update({k: np.load(cache / f"{k}.npy", allow_pickle=False) for k in ("moments", "dynamic", "pose", "valid")})
    require(a["moments"].shape == a["dynamic"].shape == (14985, 8, 114) and a["valid"].all(), "Unexpected cache/validity")
    source_rows = {}
    for room in ROOMS:
        selected, _ = calibration_rows(manifest, room)
        for row in selected:
            path = (manifest_path.parent / row["path"]).resolve()
            require(path.parent == manifest_path.parent.resolve() and path.name == row["id"].replace('/', '-') + ".npz", "Unexpected source path")
            bind(path, row["sha256"])
            source_rows[row["id"]] = dict(path=str(path), sha256=row["sha256"])
        bind(study / f"{room}-training-stats.npz")
        for arm in ARMS:
            original = next(r for r in completed["results"] if r["room"] == room and r["arm"] == arm)
            bind(study / f"{room}-{arm}/best.pt", original["checkpoint_sha256"])
    out.mkdir(parents=True)
    write(out / "registration.json", dict(config=CONFIG, script_sha256=digest(__file__), inputs_sha256=inputs,
                                         original_study_registration_sha256=digest(study / "registration.json"),
                                         calibration_sources=source_rows, labels_total=360,
                                         runtime=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__), cpu_threads=1)))
    (out / "executed-source.py").write_bytes(Path(__file__).read_bytes())
    rows, parameters, coverage, parity = [], {}, {}, dict(generic_windows=0, cache_endpoints=0, original_models=0)
    try:
        for room in ROOMS:
            selected, people = calibration_rows(manifest, room)
            source_train, _, held = neural.masks(a, room)
            test = held & ~np.isin(a["subject"], people)
            require(set(a["subject"][test]).isdisjoint(people) and len(set(a["subject"][test])) == 3, "Calibration/validation people overlap")
            require(int(test.sum()) == 2997, "Unexpected validation count")
            cal = {k: [] for k in ("moments", "dynamic", "pose", "valid", "sequence", "subject", "end")}
            for row in selected:
                with np.load(source_rows[row["id"]]["path"], allow_pickle=False) as archive:
                    csi, target, valid = archive["csi"], archive["pose"], archive["valid"]
                require(csi.shape == (297, 3, 114, 10) and target.shape == (297, 17, 2), "Unexpected calibration sequence shape")
                for end in ENDS:
                    m, d = generic.features(csi[end - 7:end + 1], bins=19)
                    old_m, old_d = old_features.features(csi[end - 7:end + 1])
                    require(np.array_equal(m, old_m) and np.array_equal(d, old_d), "Generic/executed feature mismatch")
                    parity["generic_windows"] += 1
                    existing = np.flatnonzero((a["sequence"] == row["id"]) & (a["end"] == end))
                    if len(existing):
                        i = existing.item()
                        require(np.array_equal(m, a["moments"][i]) and np.array_equal(d, a["dynamic"][i]) and np.array_equal(target[end], a["pose"][i]), "Original cached endpoint mismatch")
                        parity["cache_endpoints"] += 1
                    for key, value in dict(moments=m, dynamic=d, pose=target[end], valid=valid[end], sequence=row["id"], subject=row["subject"], end=end).items():
                        cal[key].append(value)
            cal = {k: np.asarray(v) for k, v in cal.items()}
            require(len(cal["pose"]) == 120 and cal["valid"].all() and np.isfinite(cal["pose"]).all(), "Calibration budget/validity differs")
            np.savez_compressed(out / f"{room}-calibration.npz", **cal)
            normalization = {}
            with np.load(study / f"{room}-training-stats.npz", allow_pickle=False) as stats:
                for key in ("moments", "dynamic"):
                    mean, scale = stats[key + "_mean"], stats[key + "_scale"]
                    require(np.array_equal(mean, a[key][source_train].mean((0, 1))) and np.array_equal(scale, np.maximum(a[key][source_train].std((0, 1)), .1)), "Source normalization mismatch")
                    normalization[key] = (mean, scale)
            def normalized(values):
                return [((values[k] - normalization[k][0]) / normalization[k][1]).astype(np.float32) for k in ("moments", "dynamic")]
            hm, hd = normalized({k: a[k][held] for k in ("moments", "dynamic")})
            cm, cd = normalized(cal)
            y, seq = a["pose"][test], a["sequence"][test]
            represented = np.isin(np.asarray([s.split('/')[-1] for s in seq]), ACTIONS)
            masks = dict(all=np.ones(len(y), bool), calibration_actions=represented, other_actions=~represented)
            coverage[room] = dict(calibration_people=people, validation_people=sorted(set(a["subject"][test])),
                                  calibration_labels=120, calibration_recordings=12, calibration_valid_joints=int(cal["valid"].sum()),
                                  validation_windows=int(test.sum()), validation_recordings=len(set(seq)),
                                  validation_calibration_action_windows=int(represented.sum()), validation_other_action_windows=int((~represented).sum()),
                                  rejected_calibration=0, rejected_validation=0, context=8, endpoint_stride=8)
            def record(method, prediction):
                row = dict(room=room, method=method, scores={name: score(prediction[mask], y[mask], seq[mask]) for name, mask in masks.items()})
                rows.append(row)
                print(json.dumps(dict(room=room, method=method, nje=row["scores"]["all"]["nje"], relative_motion_r2=row["scores"]["all"]["motion"]["relative"]["r2"])), flush=True)
            source_mean = a["pose"][source_train].astype(float).mean(0)
            target_mean = cal["pose"].astype(float).mean(0)
            parameters[room] = dict(source_mean=source_mean.tolist(), target_mean=target_mean.tolist(), neural={})
            record("source_mean", np.broadcast_to(source_mean, y.shape))
            record("target_mean", np.broadcast_to(target_mean, y.shape))
            for arm in ARMS:
                checkpoint = torch.load(study / f"{room}-{arm}/best.pt", map_location="cpu", weights_only=True)
                require(checkpoint["config"] == registered["config"] and checkpoint["arm"] == arm and checkpoint["cache_receipt_sha256"] == registered["cache_receipt_sha256"] and checkpoint["source_sha256"] == registered["source_sha256"], "Checkpoint binding mismatch")
                model = neural.Probe(arm, checkpoint["training_target_stats"])
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                model.eval()
                hp = predict(model, hm, hd)
                original = next(r for r in completed["results"] if r["room"] == room and r["arm"] == arm)["held_development_room"]
                reproduced = score(hp, a["pose"][held], a["sequence"][held])
                require(abs(reproduced["nje"] - original["normalized_joint_error"]) < 1e-6 and abs(reproduced["pck20"] - original["pck"]) < 1e-6, "Original model accuracy mismatch")
                parity["original_models"] += 1
                p, cp = hp[~np.isin(a["subject"][held], people)].astype(float), predict(model, cm, cd).astype(float)
                offset = (cal["pose"][:, [11, 12]].mean(1) - cp[:, [11, 12]].mean(1)).mean(0)
                residual = target_mean - cp.mean(0)
                parameters[room]["neural"][arm] = dict(root_shift=offset.tolist(), joint_residual=residual.tolist(), calibration_prediction_mean=cp.mean(0).tolist(), shrink_grid=SHRINK)
                record(arm + ":zero", p)
                record(arm + ":root_shift", p + offset)
                record(arm + ":joint_shift", p + residual)
                for alpha in SHRINK:
                    adapted = shrink(p, cp, cal["pose"], alpha)
                    require(alpha != 0 or np.allclose(adapted, target_mean, atol=1e-12), "Zero shrink is not target mean")
                    require(alpha != 1 or np.allclose(adapted, p + residual, atol=1e-12), "Unit shrink is not residual correction")
                    record(arm + f":shrink_{alpha:g}", adapted)
            vm, _ = normalized({k: a[k][test] for k in ("moments", "dynamic")})
            for kind, cx, vx in (("last", cm[:, -1], vm[:, -1]), ("temporal", cm.reshape(120, -1), vm.reshape(len(y), -1))):
                for alpha in RIDGE:
                    fit = ridge_fit(cx, cal["pose"], alpha)
                    np.savez(out / f"{room}-ridge-{kind}-{alpha:g}.npz", **fit)
                    record(f"ridge_{kind}:{alpha:g}", ridge_predict(fit, vx))
        require(all(digest(Path(path)) == value for path, value in inputs.items()), "Frozen inputs changed during probe")
        require(parity == dict(generic_windows=360, cache_endpoints=72, original_models=6), "Incomplete parity checks")
        methods = [r["method"] for r in rows if r["room"] == ROOMS[0]]
        aggregates = {method: dict(macro_nje=float(np.mean([r["scores"]["all"]["nje"] for r in rows if r["method"] == method])),
                                   macro_pck05=float(np.mean([r["scores"]["all"]["pck05"] for r in rows if r["method"] == method])),
                                   macro_relative_motion_r2=float(np.mean([r["scores"]["all"]["motion"]["relative"]["r2"] for r in rows if r["method"] == method]))) for method in methods}
        selected = {arm: min([arm + f":shrink_{alpha:g}" for alpha in SHRINK], key=lambda m: aggregates[m]["macro_nje"]) for arm in ARMS}
        selected["ridge"] = min([f"ridge_{kind}:{alpha:g}" for kind in ("last", "temporal") for alpha in RIDGE], key=lambda m: aggregates[m]["macro_nje"])
        write(out / "parameters.json", parameters)
        write(out / "results.json", dict(config=CONFIG, coverage=coverage, parity_checks=parity, rows=rows, aggregates=aggregates,
                                         development_selection=selected, seconds=time.monotonic() - started,
                                         limits="Single-seed, post-hoc development proxy. The same three proxy-validation rooms select these hyperparameters, so these scores are not independent tests. Labels are supplied annotations; constant means can yield apparent calibration gains. No E04/final recordings accessed. Within-record centering is diagnostic only; stride8 (~0.8s) can alias motion. One fixed within-record shuffle has no significance or causal claim."))
        text = ["# Source-room-only 120-label calibration proxy", "", CONFIG["scope"], "", "Calibration: first two people/room, six fixed actions, ten zero-based endpoints each. Validation: remaining three people, 81 recordings/2,997 endpoints/50,949 joints per room. Six represented action categories contribute666endpoints;21other categories contribute2,331. Every adapter and direct ridge sees the same120labels/room. No neural training or E04/final-holdout access.", "", "All360generic feature windows match the archived8-frame implementation exactly; all72intersecting original cache endpoints match; all6frozen models reproduce original held-room NJE/PCK within1e-6 before adaptation. Fitted output/Ridge parameters and three calibration-example archives are retained.", "", "Hyperparameters below minimize macro-NJE across these same three development rooms. This is development selection, not independent confirmation.", "", "| Method | Macro NJE | Macro PCK@.05 | Macro relative motion R² |", "|---|---:|---:|---:|"]
        chosen_methods = ["source_mean", "target_mean"] + [arm + ":zero" for arm in ARMS] + [arm + ":root_shift" for arm in ARMS] + [arm + ":joint_shift" for arm in ARMS] + list(selected.values())
        for method in dict.fromkeys(chosen_methods):
            item = aggregates[method]
            text.append(f"|{method}|{item['macro_nje']:.6f}|{item['macro_pck05']:.6f}|{item['macro_relative_motion_r2']:.6f}|")
        text += ["", "| Room | Method | All NJE | Calibration-action NJE | Other-action NJE | Relative motion R² | Within-record shuffle NJE delta |", "|---|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["method"] in ["target_mean"] + list(selected.values()):
                s = row["scores"]
                text.append(f"|{row['room']}|{row['method']}|{s['all']['nje']:.6f}|{s['calibration_actions']['nje']:.6f}|{s['other_actions']['nje']:.6f}|{s['all']['motion']['relative']['r2']:.6f}|{s['all']['shuffle_nje_minus_aligned']:.6f}|")
        text += ["", "R² removes each recording's mean prediction and annotation. Zero is the oracle static recording mean; negative means predicted fluctuations increase squared error. Root-relative variance/cosine and all grids are in results.json. Calibration actions are source-seen categories; other actions here are absent from the calibration set, not unseen source categories. These are supplied2D annotations, one training seed, overlapping input windows, and development-only comparisons; no reliable motion, causal sensing or unseen-room final claim follows automatically."]
        positive_motion = sum(r["scores"]["all"]["motion"]["relative"]["r2"] > 1e-8 for r in rows)
        text += ["", "## Interpretation", "",
                 f"Target mean macro NJE is {aggregates['target_mean']['macro_nje']:.6f}, versus source mean {aggregates['source_mean']['macro_nje']:.6f} and the uncalibrated frame control {aggregates['frame_nll:zero']['macro_nje']:.6f}. These fixed few-shot references do not automatically supply a transferable room prior.",
                 f"Across all {len(rows)} method/room rows, {positive_motion} have positive pooled within-record root-relative R². Compare every selected adapter against both the target mean and its unchanged source model; reducing noisy fluctuations alone is not recovered motion.",
                 "Selections above use the same proxy-validation scores they summarize. No final or E04 improvement is established. Pose distributions of two calibration people and six actions can differ from those of the three validation people and remaining actions. All outcomes, including negative calibration transfer, remain part of this probe."]
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
