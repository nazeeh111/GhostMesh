"""Inference-only input-time controls for a completed single-record capacity run."""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import json
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from domain_rf_overfit_probe import RFPose, RFSet, digest, write  # noqa: E402


def remap_frames(tensor, order):
    return tensor.reshape(-1, *tensor.shape[2:])[order].reshape(tensor.shape)


def run(run_dir, out):
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    config = json.loads((run_dir/'registration.json').read_text())
    if config['record'] != 'E01/S01/A01' or config['device'] != 'cpu':
        raise ValueError('This bounded inference diagnostic requires the registered CPU source record')
    if not (run_dir/'manifest.json').exists():
        raise ValueError('Training run must finish first')
    for filename, expected in config['sources'].items():
        source_path = run_dir/'executed-source.py' if filename == 'scripts/domain_rf_overfit_probe.py' else ROOT/filename
        if digest(source_path) != expected:
            raise ValueError('Frozen source mismatch: '+filename)
    # Reuse exact archived metric/evaluation code, independent of later style-only cleanup.
    spec = importlib.util.spec_from_file_location('frozen_capacity_helpers', run_dir/'executed-source.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    evaluate = frozen.evaluate
    manifest = json.loads((run_dir/'manifest.json').read_text())
    for filename in ('selected-source-record.npz', 'legacy/best-training.pt', 'legacy/last.pt', 'quotient/best-training.pt', 'quotient/last.pt'):
        if digest(run_dir/filename) != manifest[filename]:
            raise ValueError('Training artifact mismatch: '+filename)
    data = np.load(run_dir/'selected-source-record.npz', allow_pickle=False)
    order = np.random.default_rng(2027).permutation(296)
    shift = np.roll(np.arange(296), 37)
    registration = {'scope': 'TRAIN-SET inference-only controls, no test/generalization evidence',
                    'training_registration_sha256': digest(run_dir/'registration.json'),
                    'training_manifest_sha256': digest(run_dir/'manifest.json'),
                    'source_sha256': digest(Path(__file__)), 'seed': 2027,
                    'shuffle_order': order.tolist(), 'circular_shift_order': shift.tolist(),
                    'definition': 'Permute/shift cached individual input frames then restore37x8 blocks; labels remain original. Joint, body-only and root-only controls. This changes temporal block boundaries; neither causal nor generalization proof.'}
    out.mkdir(parents=True)
    write(out/'registration.json', registration)
    (out/'executed-source.py').write_bytes(Path(__file__).read_bytes())
    results = []
    for mode in ('legacy', 'quotient'):
        dataset = RFSet(data['legacy'], data['legacy'], data['quotient'], data['root_spectrum'], data['pose_sequence'], np.arange(37), mode, dense=True)
        body, root, target = [torch.stack([dataset[i][j] for i in range(37)]) for j in range(3)]
        for state in ('best-training', 'last'):
            checkpoint = torch.load(run_dir/mode/(state+'.pt'), map_location='cpu', weights_only=False)
            model = RFPose(kernel='native')
            model.load_state_dict(checkpoint['state_dict'])
            for control, mapping, body_change, root_change in (
                ('aligned', np.arange(296), False, False),
                ('joint-shuffle', order, True, True),
                ('body-shuffle', order, True, False),
                ('root-shuffle', order, False, True),
                ('joint-circular-shift-37', shift, True, True)):
                b = remap_frames(body, mapping) if body_change else body
                r = remap_frames(root, mapping) if root_change else root
                score, prediction = evaluate(model, b, r, target)
                if control == 'aligned':
                    stored = np.load(run_dir/mode/(state+'-predictions.npz'), allow_pickle=False)
                    for key in prediction:
                        np.testing.assert_array_equal(prediction[key], stored[key], err_msg='Checkpoint reload changed '+key)
                results.append({'mode': mode, 'checkpoint': state, 'step': checkpoint['step'], 'control': control, 'score': score})
                np.savez(out/f'{mode}-{state}-{control}.npz', **prediction)
    write(out/'results.json', {'registration': registration, 'results': results})
    write(out/'manifest.json', {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.run.resolve(), args.out.resolve())
