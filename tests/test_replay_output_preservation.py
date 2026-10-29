"""Actual replay writer bodies with inert inference/data dependencies.

These tests qualify output preservation, never model efficacy or signal/data
contracts. No Torch/NumPy import, model, dataset or package initialization runs.
"""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class _Pose:
    shape = (1, 17, 2)
    def __getitem__(self, _): return self
    def round(self, _): return self
    def tolist(self): return [[0, 0]] * 17


class _Recording:
    files = ['csi', 'timestamps']
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def __getitem__(self, name):
        return SimpleNamespace(shape=(1, 1, 1, 1)) if name == 'csi' else [0.0]


class ReplayOutputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.original = b'another writer owns these replay bytes\n'
        source = Path(__file__).resolve().parents[1] / 'ghostmesh/replay.py'
        tree = ast.parse(source.read_text())
        names = {'infer_recording', 'export_replay'}
        bodies = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in bodies}, names)
        package, prepare = ModuleType('_replay_output_control'), ModuleType('_replay_output_control.prepare')
        package.__path__ = []
        prepare.JOINT_NAMES, prepare.EDGES = ['joint'] * 17, []
        modules = patch.dict(sys.modules, {'_replay_output_control': package, '_replay_output_control.prepare': prepare})
        modules.start()
        self.addCleanup(modules.stop)
        checkpoint = {'purpose': 'software_check_only', 'manifest_sha256': 'fixed',
                      'input_shape': [1, 1, 1], 'config': {'data': {'window': 1}}}
        sequence = {'csi': SimpleNamespace(shape=(1, 1, 1, 1)), 'timestamps': [0.0], 'pose': _Pose(), 'valid': _Pose()}
        manifest = {'provenance': {'kind': 'synthetic_fixture'}, 'sequences': [
            {'id': 'fixture', 'split': 'test', 'environment': 'fixture-room', 'subject': 'fixture-person', 'path': 'fixture.npz'},
            {'id': 'training', 'split': 'train', 'environment': 'fixture-room', 'subject': 'training-person', 'path': 'unused.npz'}]}
        self.namespace = {'__package__': '_replay_output_control', 'Path': Path, 'json': json,
            'np': SimpleNamespace(load=lambda *a, **k: _Recording()),
            'torch': SimpleNamespace(set_num_threads=lambda *_: None),
            'load_checkpoint': lambda *_: (SimpleNamespace(joints=17, channels=1), checkpoint),
            'sha256': lambda *_: 'fixed', 'load_manifest': lambda *_: manifest, 'read_sequence': lambda *_: sequence,
            'predict_frames': lambda *a, **k: [{'status': 'ok'}]}
        exec(compile(ast.Module(body=bodies, type_ignores=[]), str(source), 'exec'), self.namespace)

    def writers(self):
        for name in ['infer_recording', 'export_replay']:
            output = self.root / name / 'replay.json'
            yield name, output, lambda path, name=name: self.namespace[name]('synthetic-input', 'synthetic-checkpoint', path)

    def test_new_outputs_preserve_replay_contract(self):
        for name, output, write in self.writers():
            with self.subTest(writer=name):
                result = write(output)
                replay = json.loads(output.read_text())
                self.assertEqual(result['frames'], 1)
                self.assertEqual(replay['schema_version'], 1)
                self.assertEqual(replay['frames'][0]['status'], 'ok')
                self.assertEqual('ground_truth' in replay['frames'][0], name == 'export_replay')
                self.assertEqual(replay['provenance']['kind'], 'synthetic_diagnostic')

    def test_initial_existing_outputs_are_preserved(self):
        for name, output, write in self.writers():
            with self.subTest(writer=name):
                output.parent.mkdir()
                output.write_bytes(self.original)
                with self.assertRaises(FileExistsError): write(output)
                self.assertEqual(output.read_bytes(), self.original)

    def test_outputs_created_after_existence_check_are_preserved(self):
        for name, output, write in self.writers():
            with self.subTest(writer=name):
                mkdir = Path.mkdir
                def other_writer(path, *args, **kwargs):
                    result = mkdir(path, *args, **kwargs)
                    if path == output.parent: output.write_bytes(self.original)
                    return result
                with patch.object(Path, 'mkdir', new=other_writer):
                    with self.assertRaises(FileExistsError): write(output)
                self.assertEqual(output.read_bytes(), self.original)

    def test_dangling_output_symlinks_do_not_create_their_targets(self):
        for name, output, write in self.writers():
            with self.subTest(writer=name):
                output.parent.mkdir()
                target = output.parent / 'must-remain-absent.json'
                try: output.symlink_to(target)
                except (NotImplementedError, OSError) as error:
                    self.skipTest(f'This platform cannot create the test symlink: {error}')
                with self.assertRaises(FileExistsError): write(output)
                self.assertTrue(output.is_symlink())
                self.assertFalse(target.exists())

    def test_serialization_errors_do_not_create_outputs(self):
        self.namespace['predict_frames'] = lambda *a, **k: [{'status': 'ok', 'invalid': float('nan')}]
        for name, output, write in self.writers():
            with self.subTest(writer=name):
                with self.assertRaises(ValueError): write(output)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
