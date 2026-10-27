"""Person-and-room-enrolled 60-label amplitude/phase proxy, E01-E03 only."""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ROOMS = ("E01", "E02", "E03")
CAL_ACTIONS = ("A02", "A06")
LINEAR_GRID = (.01, .1, 1., 10., 100.)
WIDTH_GRID = (.5, 1., 2.)
KERNEL_GRID = (.01, .1, 1.)
HELPER_SHA = "847ee479af3012b6ac2926332d8e572da7cfe7f2cc1d3f9368fb096c185f54f6"
CONFIG = dict(scope="DEVELOPMENT ONLY: same person and room enrolled with60 reference poses from A02/A06; other25actions held; no E04/final records",
              rooms=ROOMS, calibration_actions=CAL_ACTIONS, labels_per_recording=30,
              selection="floor(linspace(0,36,30)) of each recording's37 ordered cached endpoints",
              modalities={"amplitude": 3420, "amplitude_phase": 7980}, feature_scale_floor=.1,
              linear_grid=LINEAR_GRID, linear_regularizer="lambda*n; dual linear kernel",
              gaussian_width_grid=WIDTH_GRID, gaussian_ridge_grid=KERNEL_GRID,
              gaussian_kernel="exp(-squared_distance/(2*sigma^2)); sigma=calibration-only median pair distance*width; ridge=lambda",
              source_mean="Same8source-training people as frozen frame NN, excluding entire target room and source validation people",
              final_boundary="No E04 enrollment. A02/A06 enrollment would consume S40's only reserved test recordings; any future personal final protocol must remain separate.",
              selection_metric="Minimum mean NJE over same15development people; equal grids per modality; ties retain grid order",
              numeric="Torch float64 CPU,1thread; cached Gram/distance per person/modality; finite/solve-residual guards", seed=90210)


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


def header(path):
    with Path(path).open("rb") as stream:
        magic = stream.read(8)
        require(magic[:6] == b'\x93NUMPY', "Invalid NumPy header")
        length = stream.read(2 if magic[6] == 1 else 4)
        text = stream.read(int.from_bytes(length, 'little'))
    value = ast.literal_eval(text.decode('latin1').strip())
    return dict(sha256=sha256(magic + length + text).hexdigest(), shape=value['shape'], dtype=value['descr'], fortran=value['fortran_order'])


def enrollment(sequence, subject, ends, person):
    require(all(str(s).split('/')[0] in ROOMS for s in sequence), "Forbidden enrollment room")
    cal = []
    positions = np.floor(np.linspace(0, 36, 30)).astype(int)
    own = subject == person
    for action in CAL_ACTIONS:
        indices = np.flatnonzero(own & np.asarray([s.endswith('/' + action) for s in sequence]))
        require(len(indices) == 37 and np.array_equal(ends[indices], np.arange(7, 297, 8)), "Calibration recording ordering differs")
        cal.extend(indices[positions])
    test = np.flatnonzero(own & ~np.asarray([s.split('/')[-1] in CAL_ACTIONS for s in sequence]))
    cal = np.asarray(cal)
    require(len(set(cal.tolist())) == 60 and len(test) == 925 and set(sequence[cal]).isdisjoint(set(sequence[test])), "Enrollment/action leakage or count mismatch")
    return cal, test


def matrix_cache(calibration, validation):
    x, z = torch.as_tensor(calibration, dtype=torch.float64), torch.as_tensor(validation, dtype=torch.float64)
    mean, scale = x.mean(0), x.std(0, correction=0).clamp_min(.1)
    x, z = (x - mean) / scale, (z - mean) / scale
    gram, cross = x @ x.T, z @ x.T
    distance = (x.square().sum(1)[:, None] + x.square().sum(1)[None, :] - 2 * gram).clamp_min(0)
    cross_distance = (z.square().sum(1)[:, None] + x.square().sum(1)[None, :] - 2 * cross).clamp_min(0)
    pairs = distance[torch.triu_indices(len(x), len(x), 1).unbind()]
    median = float(torch.quantile(pairs.sqrt(), .5))
    require(all(torch.isfinite(v).all() for v in (x, z, gram, cross, distance, cross_distance)), "Nonfinite feature matrices")
    return dict(mean=mean, scale=scale, gram=gram, cross=cross, distance=distance,
                cross_distance=cross_distance, median_distance=max(median, 1e-12), median_floor_used=median <= 1e-12)


def dual_prediction(kernel, cross_kernel, target, ridge, centered):
    target = torch.as_tensor(target.reshape(len(target), -1), dtype=torch.float64)
    if centered:
        # Linear regression with a fitted target intercept and centered features.
        mean = target.mean(0)
        rowmean, grandmean = kernel.mean(1), kernel.mean()
        crossmean = cross_kernel.mean(1)
        centered_kernel = kernel - rowmean[:, None] - rowmean[None, :] + grandmean
        centered_cross = cross_kernel - crossmean[:, None] - rowmean[None, :] + grandmean
    else:
        # Gaussian KRR fits residuals around calibration target mean; no test fit.
        mean = target.mean(0)
        centered_kernel, centered_cross = kernel, cross_kernel
    system = centered_kernel + ridge * torch.eye(len(target), dtype=torch.float64)
    dual = torch.linalg.solve(system, target - mean)
    residual = torch.linalg.vector_norm(system @ dual - (target - mean))
    require(torch.isfinite(dual).all() and residual <= 1e-8 * max(float(torch.linalg.vector_norm(target - mean)), 1.), "Kernel solve failed finite/residual check")
    prediction = centered_cross @ dual + mean
    require(torch.isfinite(prediction).all(), "Nonfinite kernel prediction")
    return prediction.numpy().reshape(-1, 17, 2), dict(dual=dual.numpy(), target_mean=mean.numpy(), ridge=np.array(ridge))


def scores(prediction, target, sequence):
    names = sorted(set(sequence))
    group = np.asarray([np.flatnonzero(sequence == s) for s in names])
    require(group.shape == (len(names), 37), "Expected complete37-endpoint records")
    p, y = prediction.astype(float), target.astype(float)
    require(p.shape == y.shape and np.isfinite(p).all(), "Invalid predicted poses")
    box = np.linalg.norm(y.max(1) - y.min(1), axis=-1)
    require(bool((box > 1e-6).all()), "Degenerate reference box")
    error = np.linalg.norm(p - y, axis=-1) / box[:, None]
    rng, order = np.random.default_rng(CONFIG['seed']), np.arange(len(y))
    for g in group:
        order[g] = rng.permutation(g)
    shuffled = np.linalg.norm(p[order] - y, axis=-1) / box[:, None]
    motion = {}
    pr, yr = p[:, [11, 12]].mean(1, keepdims=True), y[:, [11, 12]].mean(1, keepdims=True)
    for name, a, b in (("absolute", p, y), ("root", pr, yr), ("relative", p - pr, y - yr)):
        a, b = a[group], b[group]
        a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
        pv, yv = np.mean(a*a, axis=tuple(range(1, a.ndim))), np.mean(b*b, axis=tuple(range(1, b.ndim)))
        mse = np.mean((a-b)**2, axis=tuple(range(1, a.ndim)))
        motion[name] = dict(r2=1-float(mse.mean())/float(yv.mean()) if yv.mean()>1e-12 else None,
                            prediction_variance=float(pv.mean()), annotation_variance=float(yv.mean()),
                            variance_ratio=float(pv.mean()/yv.mean()) if yv.mean()>1e-12 else None,
                            per_record_prediction_variance=pv.tolist(), per_record_annotation_variance=yv.tolist(),
                            per_record_r2=[float(1-e/v) if v>1e-12 else None for e,v in zip(mse,yv)])
    per_action = []
    for i, name in enumerate(names):
        g = group[i]
        per_action.append(dict(action=name.split('/')[-1], sequence=name, nje=float(error[g].mean()),
                               pck05=float((error[g]<=.05).mean()), pck10=float((error[g]<=.1).mean()), pck20=float((error[g]<=.2).mean()),
                               shuffle_nje_minus_aligned=float(shuffled[g].mean()-error[g].mean()),
                               motion_r2={k:v['per_record_r2'][i] for k,v in motion.items()},
                               motion_variance={k:dict(prediction=v['per_record_prediction_variance'][i],annotation=v['per_record_annotation_variance'][i]) for k,v in motion.items()}))
    return dict(nje=float(error.mean()), pck05=float((error<=.05).mean()), pck10=float((error<=.1).mean()),
                pck20=float((error<=.2).mean()), windows=len(y), valid_joints=int(error.size),
                motion=motion, shuffle_nje_minus_aligned=float(shuffled.mean()-error.mean()), per_action=per_action)


def run(root, out):
    require(not out.exists(), "Refusing to overwrite personal probe")
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(root))
    base, rawdir, phasedir = root/'work/domain-dev-v2', root/'work/domain-dev-raw8', root/'work/domain-dev-phase8'
    receipt, raw_receipt, phase_receipt = read(base/'receipt.json'), read(rawdir/'receipt.json'), read(phasedir/'receipt.json')
    require(raw_receipt['reference_cache_receipt_sha256']==phase_receipt['reference_cache_receipt_sha256']==digest(base/'receipt.json'), "Cache alignment receipts differ")
    require(raw_receipt['source_manifest_sha256']==receipt['manifest_sha256'], "Raw source manifest differs")
    require(all(k.split('/')[0] in ROOMS for k in receipt['sequences']) and set(phase_receipt['raw_sequence_ordered_file_hashes'])==set(receipt['sequences']), "Forbidden/misaligned phase ROI")
    inputs = {}
    def bind(path, expected=None):
        value=digest(path)
        require(expected is None or value==expected, f"Hash mismatch: {path}")
        inputs[str(path.resolve())]=value
    for path in (base/'receipt.json', rawdir/'receipt.json', phasedir/'receipt.json'):
        bind(path)
    for name, value in receipt['arrays'].items():
        require(Path(name).name==name, "Unsafe cached array path")
        bind(base/name, value)
    bind(rawdir/'raw.npy', raw_receipt['raw_sha256'])
    bind(phasedir/'phase.npy', phase_receipt['phase_sha256'])
    headers=dict(amplitude=header(rawdir/'raw.npy'), phase=header(phasedir/'phase.npy'))
    require(tuple(headers['amplitude']['shape'])==(14985,8,3,114,10) and tuple(headers['phase']['shape'])==(14985,8,4,114,10), "Unexpected feature headers")
    metadata={k:np.load(base/f'{k}.npy',allow_pickle=False) for k in ('room','subject','sequence','end')}
    require(set(metadata['room'])==set(ROOMS) and all(s.split('/')[0] in ROOMS for s in metadata['sequence']), "Forbidden source data")
    plan={person:dict(zip(('calibration_indices','validation_indices'),[v.tolist() for v in enrollment(metadata['sequence'],metadata['subject'],metadata['end'],person)])) for person in sorted(set(metadata['subject']))}
    require(len(plan)==15, "Expected15source people")
    study=root/'work/domain-neural-v2'
    neural_reg, completed=read(study/'registration.json'),read(study/'results.json')
    require(completed['registration']==neural_reg and neural_reg['cache_receipt_sha256']==digest(base/'receipt.json'), "Source NN registration differs")
    for path in (study/'registration.json',study/'results.json'):
        bind(path)
    snapshot=study/'executed-source/scripts/domain_neural_probe.py'
    bind(snapshot,neural_reg['source_sha256']['scripts/domain_neural_probe.py'])
    helperpath=root/'work/domain-calibration-v2-checked/executed-source.py'
    bind(helperpath,HELPER_SHA)
    for name in ('ghostmesh/models.py','ghostmesh/metrics.py'):
        bind(root/name,neural_reg['source_sha256'][name])
    for room in ROOMS:
        bind(study/f'{room}-training-stats.npz')
        old=next(r for r in completed['results'] if r['room']==room and r['arm']=='frame_nll')
        bind(study/f'{room}-frame_nll/best.pt',old['checkpoint_sha256'])
    out.mkdir(parents=True)
    write(out/'registration.json',dict(config=CONFIG,script_sha256=digest(__file__),inputs_sha256=inputs,headers=headers,
                                       enrollment_plan=plan,phase_definition=phase_receipt['phase_features'],
                                       freeze='Before decoding amplitude/phase/pose arrays; metadata only',runtime=dict(torch=str(torch.__version__),numpy=np.__version__,threads=1)))
    (out/'executed-source.py').write_bytes(Path(__file__).read_bytes())
    rows,coverage,kernel_metadata,checks=[],{},{},dict(source_models=0,kernel_systems=0,amplitude_prefix_matches=0)
    try:
        helper=load_module('personal_checked_helper',helperpath)
        neural=load_module('personal_frozen_neural',snapshot)
        pose=np.load(base/'pose.npy',allow_pickle=False)
        valid=np.load(base/'valid.npy',allow_pickle=False)
        require(pose.shape==(14985,17,2) and valid.all() and np.isfinite(pose).all(), "Unexpected labels/validity")
        raw=np.load(rawdir/'raw.npy',mmap_mode='r',allow_pickle=False)
        phase=np.load(phasedir/'phase.npy',mmap_mode='r',allow_pickle=False)
        require(raw.dtype==phase.dtype==np.float32, "Unexpected feature dtype")
        moments=np.load(base/'moments.npy',allow_pickle=False)
        dynamic=np.load(base/'dynamic.npy',allow_pickle=False)
        source_predictions, source_means={},{}
        arrays={**metadata,'pose':pose}
        for room in ROOMS:
            train,_,held=neural.masks(arrays,room)
            source_means[room]=pose[train].astype(float).mean(0)
            with np.load(study/f'{room}-training-stats.npz',allow_pickle=False) as norm:
                hm=((moments[held]-norm['moments_mean'])/norm['moments_scale']).astype(np.float32)
                hd=((dynamic[held]-norm['dynamic_mean'])/norm['dynamic_scale']).astype(np.float32)
            checkpoint=torch.load(study/f'{room}-frame_nll/best.pt',map_location='cpu',weights_only=True)
            require(checkpoint['config']==neural_reg['config'] and checkpoint['source_sha256']==neural_reg['source_sha256'], "Source checkpoint configuration differs")
            model=neural.Probe('frame_nll',checkpoint['training_target_stats'])
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            model.eval()
            prediction=helper.predict(model,hm,hd)
            old=next(r for r in completed['results'] if r['room']==room and r['arm']=='frame_nll')['held_development_room']
            check=helper.score(prediction,pose[held],metadata['sequence'][held])
            require(abs(check['nje']-old['normalized_joint_error'])<1e-6 and abs(check['pck20']-old['pck'])<1e-6, "Source NN accuracy mismatch")
            source_predictions[room]=dict(indices=np.flatnonzero(held),prediction=prediction)
            checks['source_models']+=1
        for person,selection in plan.items():
            cal,test=(np.asarray(selection[k]) for k in ('calibration_indices','validation_indices'))
            room=str(metadata['room'][cal[0]])
            require((metadata['room'][cal]==room).all() and (metadata['subject'][test]==person).all(), "Personal split differs")
            cy,y,seq=pose[cal],pose[test],metadata['sequence'][test]
            ampcal=np.array(raw[cal,-1]).reshape(60,-1)
            amptest=np.array(raw[test,-1]).reshape(925,-1)
            phasecal=np.array(phase[cal,-1]).reshape(60,-1)
            phasetest=np.array(phase[test,-1]).reshape(925,-1)
            require(all(np.isfinite(x).all() for x in (ampcal,amptest,phasecal,phasetest)), "Nonfinite enrolled features")
            np.savez(out/f'{person}-references.npz',amplitude=ampcal,phase=phasecal,target=cy,indices=cal)
            coverage[person]=dict(room=room,calibration_labels=60,calibration_recordings=2,calibration_actions=CAL_ACTIONS,
                                  test_recordings=25,test_endpoints=925,test_valid_joints=15725,discarded_calibration_record_endpoints=14)
            def record(method,prediction):
                rows.append(dict(person=person,room=room,method=method,scores=scores(prediction,y,seq)))
            record('personal_mean',np.broadcast_to(cy.astype(float).mean(0),y.shape))
            record('source_mean',np.broadcast_to(source_means[room],y.shape))
            sp=source_predictions[room]
            record('source_frame_nn',sp['prediction'][np.isin(sp['indices'],test)])
            amp_normalization=None
            kernel_metadata[person]={}
            for mode,cx,vx in (('amplitude',ampcal,amptest),('amplitude_phase',np.concatenate((ampcal,phasecal),1),np.concatenate((amptest,phasetest),1))):
                matrices=matrix_cache(cx,vx)
                if mode=='amplitude':
                    amp_normalization=(matrices['mean'].clone(),matrices['scale'].clone())
                else:
                    require(torch.allclose(matrices['mean'][:3420],amp_normalization[0],atol=1e-12,rtol=0) and torch.allclose(matrices['scale'][:3420],amp_normalization[1],atol=1e-12,rtol=0), "Amplitude normalization changed between modalities")
                    checks['amplitude_prefix_matches']+=1
                kernel_metadata[person][mode]=dict(feature_dimensions=cx.shape[1],median_pair_distance=matrices['median_distance'],median_floor_used=matrices['median_floor_used'])
                np.savez(out/f'{person}-{mode}-normalization.npz',mean=matrices['mean'].numpy(),scale=matrices['scale'].numpy(),median_distance=np.array(matrices['median_distance']))
                for lam in LINEAR_GRID:
                    prediction,fit=dual_prediction(matrices['gram'],matrices['cross'],cy,lam*len(cy),True)
                    np.savez(out/f'{person}-{mode}-linear-{lam:g}.npz',**fit)
                    checks['kernel_systems']+=1
                    record(f'{mode}:linear:{lam:g}',prediction)
                for width in WIDTH_GRID:
                    sigma=matrices['median_distance']*width
                    kernel=torch.exp(-matrices['distance']/(2*sigma*sigma))
                    cross=torch.exp(-matrices['cross_distance']/(2*sigma*sigma))
                    for lam in KERNEL_GRID:
                        prediction,fit=dual_prediction(kernel,cross,cy,lam,False)
                        np.savez(out/f'{person}-{mode}-rbf-{width:g}-{lam:g}.npz',sigma=np.array(sigma),**fit)
                        checks['kernel_systems']+=1
                        record(f'{mode}:rbf:{width:g}:{lam:g}',prediction)
            print(json.dumps(dict(person=person,labels=60,held_actions=25,rows=len(rows))),flush=True)
        require(all(digest(Path(p))==h for p,h in inputs.items()), "Frozen inputs changed")
        require(len(rows)==465 and checks==dict(source_models=3,kernel_systems=420,amplitude_prefix_matches=15), "Incomplete personal study")
        methods=[r['method'] for r in rows if r['person']=='S01']
        aggregates={}
        for method in methods:
            subset=[r for r in rows if r['method']==method]
            good=[r['scores']['motion']['relative']['r2'] for r in subset if r['scores']['motion']['relative']['r2'] is not None]
            aggregates[method]=dict(nje=float(np.mean([r['scores']['nje'] for r in subset])),pck05=float(np.mean([r['scores']['pck05'] for r in subset])),
                                     pck10=float(np.mean([r['scores']['pck10'] for r in subset])),pck20=float(np.mean([r['scores']['pck20'] for r in subset])),
                                     mean_relative_motion_r2=float(np.mean(good)),positive_motion_people=sum(v>0 for v in good),
                                     per_room={room:float(np.mean([r['scores']['nje'] for r in subset if r['room']==room])) for room in ROOMS})
        selected={mode:min([m for m in methods if m.startswith(mode+':')],key=lambda m:aggregates[m]['nje']) for mode in ('amplitude','amplitude_phase')}
        limits="Person-and-room-enrolled mode,60labeled poses from two ~30s A02/A06 recordings;25other entire action recordings held. No zero-shot, unseen-person, new-room or final-test claim. Same15source-development people choose hyperparameters, so optimistic selection is possible. Labels are model-generated annotations;8frame window preprocessing uses causal context;925endpoints overlap and37/record sample at~0.8s. Phase is inter-antenna cosine/sine, not calibrated AoA/ToF. No E04/final or new physical data accessed. A02/A06 enrollment would consume S40's only reserved recordings, so this does not replace or consume the existing final16 plan."
        write(out/'results.json',dict(config=CONFIG,coverage=coverage,checks=checks,kernel_metadata=kernel_metadata,rows=rows,aggregates=aggregates,development_selection=selected,limits=limits,seconds=time.monotonic()-started))
        shown=['source_mean','personal_mean','source_frame_nn']+list(selected.values())
        text=['# Person-enrolled amplitude/phase calibration proxy','',CONFIG['scope'],'',
              'Each of 15 people supplies 60 labels from two complete A02/A06 recordings. Their 25 other action recordings provide 925 held endpoints. Calibration and evaluation action records are disjoint. All labels, feature statistics and bandwidths fit only these references. This is explicitly a same-person, same-room setup, not the room-calibration/unseen-person experiment.','',
              '| Method | NJE | PCK@.05 | PCK@.1 | PCK@.2 | Mean relative motion R² | Positive motion people |','|---|---:|---:|---:|---:|---:|---:|']
        for method in shown:
            v=aggregates[method]
            text.append(f"|{method}|{v['nje']:.6f}|{v['pck05']:.6f}|{v['pck10']:.6f}|{v['pck20']:.6f}|{v['mean_relative_motion_r2']:.6f}|{v['positive_motion_people']}/15|")
        text += ['','| Person | Personal mean NJE | Selected amplitude NJE | Selected amplitude+phase NJE | Phase relative motion R² | Phase shuffle NJE delta |','|---|---:|---:|---:|---:|---:|']
        lookup={(r['person'],r['method']):r['scores'] for r in rows}
        for person in plan:
            a,b,c=(lookup[person,m] for m in ('personal_mean',selected['amplitude'],selected['amplitude_phase']))
            text.append(f"|{person}|{a['nje']:.6f}|{b['nje']:.6f}|{c['nje']:.6f}|{c['motion']['relative']['r2']:.6f}|{c['shuffle_nje_minus_aligned']:.6f}|")
        text += ['', 'Selections minimize mean NJE over the same 15 development people. All 420 kernel fits, normalization/dual parameters, reference row IDs and every per-action result are retained. Negative centered motion R² means worse fluctuations than a static oracle recording mean. Source NN is an unchanged source-only reference, not an equally personalized neural learner.','',limits]
        with (out/'SUMMARY.md').open('x') as stream:
            stream.write('\n'.join(text)+'\n')
        write(out/'artifact_manifest.json',dict(script_sha256=digest(__file__),outputs_sha256={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}))
    except Exception as exc:
        write(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),completed_rows=len(rows),checks=checks))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.root.resolve(),args.out.resolve())
