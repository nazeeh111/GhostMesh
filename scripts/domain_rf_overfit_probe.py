"""Train-set capacity diagnostic only. Frozen RF model; one E01 source recording."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from ghostmesh.rf_pose import RFPose, loss_components  # noqa: E402
from domain_rf_probe import RFSet, clip_groups  # noqa: E402
from domain_structured_probe import digest, motion_metrics  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def score(prediction, target):
    p, y = np.asarray(prediction, float).reshape(-1, 17, 2), np.asarray(target, float).reshape(-1, 17, 2)
    scale = np.linalg.norm(y.max(1) - y.min(1), axis=-1).clip(1e-6)
    d = np.linalg.norm(p - y, axis=-1) / scale[:, None]
    return {'nje': float(d.mean()), **{f'pck{t}': float((d <= t).mean()) for t in (.05, .1, .2)},
            'motion': motion_metrics(p, y, np.repeat('E01/S01/A01', len(y)))}


def gradient_evidence(model):
    groups = {'body_encoder': [], 'body_head': [], 'root_encoder': [], 'root_head': [], 'uncertainty': []}
    for name, p in model.named_parameters():
        if 'uncertainty' in name:
            key = 'uncertainty'
        elif name.startswith('root_encoder'):
            key = 'root_encoder'
        elif name.startswith('root_head'):
            key = 'root_head'
        elif name.startswith('relative_head'):
            key = 'body_head'
        else:
            key = 'body_encoder'
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise ValueError('Nonfinite gradient: ' + name)
            groups[key].append(float(p.grad.detach().square().sum().cpu()))
    return {k: float(np.sqrt(sum(v))) for k, v in groups.items()}


@torch.no_grad()
def evaluate(model, body, root, target):
    model.eval()
    out = model(body, root, return_sequence=True)
    if not all(torch.isfinite(v).all() for v in out.values()):
        raise ValueError('Nonfinite output')
    arrays = {k: v.cpu().numpy() for k, v in out.items()}
    terms = {k: float(v.cpu()) for k, v in loss_components(out, target).items()}
    report = {'dense': score(arrays['pose'], target.cpu().numpy()),
              'endpoint': score(arrays['pose'][:, -1], target.cpu().numpy()[:, -1]),
              'raw_dense': score(arrays['raw_pose'], target.cpu().numpy()),
              'clip_fraction': float(((arrays['raw_pose'] < 0) | (arrays['raw_pose'] > 1)).mean()),
              'loss_components': terms}
    report['loss'] = terms['relative_geometry'] + terms['root_geometry'] + .1 * terms['bone'] + .1 * terms['uncertainty']
    return report, arrays


def run(args):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if not 1 <= args.steps <= 2000:
        raise ValueError('Bounded 1..2000 optimization steps required')
    torch.set_num_threads(1)
    ref, stream, rf = [ROOT / 'work' / x for x in ('domain-dev-v2', 'domain-dev-stream8', 'domain-dev-rf8')]
    receipts = {str(p.relative_to(ROOT)): digest(p) for p in (ref/'receipt.json', stream/'receipt.json', rf/'receipt.json')}
    rr, sr, qr = [json.loads((p/'receipt.json').read_text()) for p in (ref, stream, rf)]
    if sr['reference_receipt_sha256'] != receipts[str((ref/'receipt.json').relative_to(ROOT))] or qr['reference_receipt_sha256'] != sr['reference_receipt_sha256']:
        raise ValueError('Cache reference receipt mismatch')
    sources = {str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), ROOT/'ghostmesh/rf_pose.py', ROOT/'ghostmesh/models.py', ROOT/'scripts/domain_rf_probe.py', ROOT/'scripts/domain_structured_probe.py')}
    for receipt in (sr, qr):
        for filename, expected in receipt['source_sha256'].items():
            if digest(ROOT/filename) != expected:
                raise ValueError('Cache primitive source changed: ' + filename)
    arrays = {}
    verified = {}
    for directory, receipt, names in ((ref, rr, ('sequence', 'room', 'subject', 'end', 'pose', 'valid')),
                                      (stream, sr, ('legacy', 'root_spectrum', 'pose_sequence', 'valid_sequence')),
                                      (rf, qr, ('quotient',))):
        for name in names:
            path = directory/(name+'.npy')
            actual = digest(path)
            expected = receipt['array_sha256'][name] if 'array_sha256' in receipt else receipt['arrays'][path.name]
            if actual != expected:
                raise ValueError('Cache content mismatch: ' + str(path))
            verified[str(path.relative_to(ROOT))] = actual
            arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    if set(arrays['room']) != {'E01', 'E02', 'E03'}:
        raise ValueError('Only source rooms allowed')
    indices = np.flatnonzero(arrays['sequence'] == 'E01/S01/A01')
    if len(indices) != 37 or not np.array_equal(arrays['end'][indices], np.arange(7, 296, 8)):
        raise ValueError('Unexpected record coverage')
    if not arrays['valid_sequence'][indices].all() or not np.array_equal(arrays['pose_sequence'][indices, -1], arrays['pose'][indices]):
        raise ValueError('Dense labels/endpoint mismatch')
    selected = {name: np.asarray(arrays[name][indices]).copy() for name in ('legacy', 'quotient', 'root_spectrum', 'pose_sequence')}
    if not all(np.isfinite(a).all() for a in selected.values()):
        raise ValueError('Nonfinite selected source record')
    config = {'scope': 'TRAIN-SET CAPACITY ONLY; no held-out accuracy/generalization claim; no E04 arrays',
              'record': 'E01/S01/A01', 'steps': args.steps, 'device': args.device, 'seed': 2027,
              'batch': 37, 'frames': 296, 'dense_context': '37 disjoint causal 8-frame blocks; no temporal history across block boundaries',
              'optimizer': 'AdamW lr=0.0003 weight_decay=0.01; frozen rf_loss; separate branch gradient clip=1',
              'selection': 'minimum TRAINING dense NJE only; last state retained regardless',
              'modes': ['legacy', 'quotient'], 'identical_initialization': True,
              'global_window_indices': indices.tolist(), 'zero_based_frame_indices': list(range(296)),
              'endpoints': arrays['end'][indices].tolist(), 'sources': sources, 'receipts': receipts,
              'verified_cache_sha256': verified, 'torch': str(torch.__version__), 'platform': platform.platform()}
    out.mkdir(parents=True)
    write(out/'registration.json', config)
    (out/'executed-source.py').write_bytes(Path(__file__).read_bytes())
    np.savez(out/'selected-source-record.npz', **selected)
    torch.manual_seed(2027)
    initial = RFPose(kernel='native')
    initial.initialize_from_source(selected['pose_sequence'])
    initial_state = snapshot(initial)
    torch.save({'state_dict': initial_state, 'config': config}, out/'initial.pt')
    source_mean = np.broadcast_to(selected['pose_sequence'].reshape(-1, 17, 2).mean(0), (296, 17, 2))
    target_flat = selected['pose_sequence'].reshape(-1, 17, 2).astype(float)
    target_root = target_flat[:, [11, 12]].mean(1, keepdims=True)
    target_parts = {'absolute': target_flat, 'root': target_root, 'relative': target_flat-target_root}
    target_variance = {key: float(np.mean((value-value.mean(0))**2)) for key, value in target_parts.items()}
    summary = {'config': config, 'target_mean': score(source_mean, selected['pose_sequence']),
               'target_within_record_mean_coordinate_variance': target_variance,
               'target_bbox_diagonal_mean': float(np.linalg.norm(target_flat.max(1)-target_flat.min(1), axis=-1).mean()), 'results': []}
    root_final = None
    for mode in config['modes']:
        folder = out/mode
        folder.mkdir()
        model = RFPose(kernel='native').to(args.device)
        model.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
        dataset = RFSet(selected['legacy'], selected['legacy'], selected['quotient'], selected['root_spectrum'], selected['pose_sequence'], np.arange(37), mode, dense=True)
        body, root, target = [torch.stack([dataset[i][j] for i in range(37)]).to(args.device) for j in range(3)]
        history, gradients, best = [], [], float('inf')
        started = time.monotonic()
        for step in range(args.steps+1):
            if step % args.interval == 0 or step == args.steps:
                row, predictions = evaluate(model, body, root, target)
                row.update(step=step, seconds=time.monotonic()-started)
                history.append(row)
                if row['dense']['nje'] < best:
                    best = row['dense']['nje']
                    torch.save({'state_dict': snapshot(model), 'optimizer': optimizer.state_dict(), 'step': step, 'config': config}, folder/'best-training.pt')
                    np.savez(folder/'best-training-predictions.npz', **predictions)
                    best_step = step
                write(folder/'trajectory.json', {'history': history, 'gradients': gradients})
                print(json.dumps({'mode': mode, 'step': step, 'nje': row['dense']['nje'], 'relative_r2': row['dense']['motion']['relative']['within_record_r2'], 'seconds': row['seconds']}), flush=True)
            if step == args.steps:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            terms = loss_components(model(body, root, return_sequence=True), target)
            loss = terms['relative_geometry'] + terms['root_geometry'] + .1*terms['bone'] + .1*terms['uncertainty']
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite objective')
            loss.backward()
            evidence = gradient_evidence(model)
            if step < 3 or (step+1) % args.interval == 0:
                gradients.append({'update': step+1, 'unclipped_gradient_l2': evidence})
            clip_groups(model)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Nonfinite parameter')
        last = snapshot(model)
        torch.save({'state_dict': last, 'optimizer': optimizer.state_dict(), 'step': args.steps, 'config': config}, folder/'last.pt')
        np.savez(folder/'last-predictions.npz', **predictions)
        root_state = {k: v for k, v in last.items() if k.startswith(('root_encoder.', 'root_head.'))}
        root_difference = max(float((v-root_final[k]).abs().max()) for k, v in root_state.items()) if root_final is not None else 0.
        root_final = root_state
        p, y = predictions['pose'].reshape(-1, 17, 2), selected['pose_sequence'].reshape(-1, 17, 2)
        rng = np.random.default_rng(2027)
        report = {'mode': mode, 'best_training_step': best_step, 'best_training_nje': best,
                  'last': history[-1], 'shuffled_last': score(p[rng.permutation(len(p))], y),
                  'circular_shift_37_frames_last': score(np.roll(p, 37, axis=0), y),
                  'root_parameter_max_difference_from_prior_mode': root_difference,
                  'parameter_max_change': {prefix: max(float((v-initial_state[k]).abs().max()) for k, v in last.items() if k.startswith(prefix)) for prefix in ('packet_encoder.', 'temporal.', 'relative_head.', 'root_encoder.', 'root_head.')},
                  'finite_all_steps': True, 'seconds': time.monotonic()-started}
        write(folder/'result.json', report)
        summary['results'].append(report)
        write(out/'results.json', summary)
    # Bind final source bytes and outputs, refusing a silent mid-run source mutation.
    if any(digest(ROOT/name) != expected for name, expected in sources.items()):
        raise ValueError('Source changed during run')
    write(out/'manifest.json', {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    run(parser.parse_args())
