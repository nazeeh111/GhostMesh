"""Byte-only supplemental dependency binding for source30 comparison results.

No MAT/NPY decoder is imported or called. Prior results are never rewritten.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(reference, stream, quotient, cache, comparison, old_manifest, output):
    if output.exists():
        raise FileExistsError(output)
    checked = []

    def bind(path, expected, basis):
        actual = digest(path)
        if actual != expected:
            raise ValueError(f'Hash mismatch: {path}; expected by {basis}')
        checked.append({'path': str(path.resolve()), 'sha256': actual, 'expected_by': basis})
        return actual

    ref_path, stream_path, quotient_path = [p/'receipt.json' for p in (reference, stream, quotient)]
    ref, st, qt, source, result = [read(p) for p in (ref_path, stream_path, quotient_path, cache/'receipt.json', comparison)]
    ref_hash = digest(ref_path)
    for receipt, name in ((st, str(stream_path)), (qt, str(quotient_path))):
        if receipt['reference_receipt_sha256'] != ref_hash:
            raise ValueError('Cross-receipt reference mismatch: '+name)
        if (receipt['windows'], receipt['window']) != (ref['windows'], ref['window']):
            raise ValueError('Cross-receipt window scope mismatch: '+name)
    if (ref['windows'], ref['window'], ref['stride']) != (14985, 8, 8):
        raise ValueError('Unexpected frozen reference window coverage')
    old = read(old_manifest)
    source_ids = {r['id'] for r in old['sequences'] if r['environment'] in ('E01', 'E02', 'E03')}
    if len(source_ids) != 405 or set(ref['sequences']) != source_ids or any(x[:3] not in {'E01', 'E02', 'E03'} for x in source_ids):
        raise ValueError('Reference record scope is not the frozen405 source records')
    bind(old_manifest, ref['manifest_sha256'], str(ref_path)+':manifest_sha256')
    if result['old_manifest_sha256'] != ref['manifest_sha256']:
        raise ValueError('Completed comparison used another old manifest')
    bind(reference/'sequence.npy', ref['arrays']['sequence.npy'], str(ref_path)+':arrays.sequence.npy')
    bind(reference/'end.npy', ref['arrays']['end.npy'], str(ref_path)+':arrays.end.npy')
    bind(stream/'legacy.npy', st['array_sha256']['legacy'], str(stream_path)+':array_sha256.legacy')
    bind(quotient/'quotient.npy', qt['array_sha256']['quotient'], str(quotient_path)+':array_sha256.quotient')
    if 'ghostmesh/preprocess.py' not in st['source_sha256'] or any('ghostmesh/rf_features.py' not in x['source_sha256'] for x in (st, qt)):
        raise ValueError('Primitive source bindings are absent')
    for receipt, receipt_path in ((st, stream_path), (qt, quotient_path)):
        for name, expected in sorted(receipt['source_sha256'].items()):
            bind(ROOT/name, expected, str(receipt_path)+':source_sha256.'+name)
    if st['source_sha256']['ghostmesh/rf_features.py'] != qt['source_sha256']['ghostmesh/rf_features.py']:
        raise ValueError('Different RF primitives in reference caches')
    bind(ROOT/'scripts/domain_verify_source_cache.py', result['script_sha256'], str(comparison)+':script_sha256')
    bind(cache/'receipt.json', result['source_receipt_sha256'], str(comparison)+':source_receipt_sha256')
    bind(ROOT/'scripts/domain_full_source_cache.py', source['script_sha256'], str(cache/'receipt.json')+':script_sha256')
    if result['full_cache_array_hashes_verified'] != source['arrays']:
        raise ValueError('Comparison and source cache array bindings differ')
    expected_ids = [f'E{e:02d}/S{s:02d}/A{a:02d}' for e in range(1,4) for s in range(10*(e-1)+1,10*e+1) for a in range(1,28)]
    if [r['id'] for r in source['records']] != expected_ids or source['frames'] != 240570:
        raise ValueError('Unexpected expanded source identity scope')
    if any(r['offset'] != i*297 or r['frames'] != 297 for i,r in enumerate(source['records'])):
        raise ValueError('Unexpected expanded source row offsets')
    if (result['old_source_records_exact_amplitude_pose_validity'], result['sampled_windows_exact_legacy_and_quotient']) != (405,1215):
        raise ValueError('Completed comparison coverage differs')
    inputs = {str(p.resolve()): digest(p) for p in (ref_path, stream_path, quotient_path, cache/'receipt.json', comparison, old_manifest)}
    report = {'schema_version': 1, 'status': 'all_comparison_dependencies_bound',
              'scope': 'Supplemental byte-only reference dependency verification; E01-E03 caches only. No MAT/NPY arrays decoded; no E04 raw files or arrays opened.',
              'limitations': 'Does not rerun the405-record/1215-window comparison or rehash the new source arrays; binds its completed result and already verified array hashes.',
              'script_sha256': digest(Path(__file__)), 'input_receipts_and_manifest_sha256': inputs,
              'reference_records': 405, 'expanded_source_records':810, 'reference_windows':14985,
              'verified_actual_input_and_source_files': checked,
              'cross_receipt_reference_binding':True, 'primitive_source_bindings':True,
              'completed_comparison_result_and_source_cache_bound':True}
    with output.open('x') as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'stream', 'quotient', 'cache', 'comparison', 'old-manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.reference.resolve(), args.stream.resolve(), args.quotient.resolve(), args.cache.resolve(), args.comparison.resolve(), args.old_manifest.resolve(), args.output.resolve())
    print(json.dumps({'status':result['status'],'verified_actual_files':len(result['verified_actual_input_and_source_files'])}), flush=True)
