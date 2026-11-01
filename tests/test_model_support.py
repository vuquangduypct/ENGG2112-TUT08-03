"""Regression checks without model downloads or access to private retinal data."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote
import re
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.model_support import feature_cache_metadata, load_feature_cache

REPO = Path(__file__).resolve().parents[1]


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'one.png'
        self.image.write_bytes(b'original synthetic image bytes')
        self.split = self.root / 'split.csv'
        self.prepared = self.root / 'prepared.csv'
        self.config = self.root / 'config.json'
        for p in (self.split, self.prepared, self.config):
            p.write_text('synthetic metadata', encoding='utf-8')
        self.frame = pd.DataFrame({'id_code': ['one'], 'diagnosis': [0],
                                   'split': ['train'], 'image_path': [str(self.image)]})
        self.cache = self.root / 'features.npz'
        self.metadata = self.current_metadata()
        self.save_cache()

    def current_metadata(self):
        return feature_cache_metadata(self.frame, self.split, self.prepared, self.config,
                                      {'weights': 'fixed'}, {'mean': [0.5]})

    def save_cache(self, **overrides):
        arrays = {'metadata': json.dumps(self.metadata), 'id_code': np.array(['one']),
                  'labels': np.array([0]), 'features': np.zeros((1, 49, 512), np.float32)}
        arrays.update(overrides)
        np.savez_compressed(self.cache, **arrays)

    def load(self, metadata=None):
        return load_feature_cache(self.cache, self.metadata if metadata is None else metadata,
                                  self.frame.id_code, self.frame.diagnosis)

    def test_matching_cache_is_reused(self):
        self.assertIsNotNone(self.load())

    def test_changed_image_bytes_invalidate_cache_with_unchanged_split(self):
        self.image.write_bytes(b'different prepared image')
        self.assertIsNone(self.load(self.current_metadata()))

    def test_changed_config_and_manifest_invalidate_cache(self):
        for path in (self.config, self.prepared, self.split):
            with self.subTest(path=path.name):
                old = path.read_bytes()
                path.write_bytes(b'changed')
                self.assertIsNone(self.load(self.current_metadata()))
                path.write_bytes(old)

    def test_transform_or_encoder_change_invalidates_cache(self):
        for key in ('transform', 'encoder'):
            with self.subTest(key=key):
                changed = dict(self.metadata, **{key: {'changed': True}})
                self.assertIsNone(self.load(changed))

    def test_wrong_labels_ids_shape_and_nonfinite_features_rejected(self):
        for override in ({'labels': np.array([4])}, {'labels': np.array([0.5])},
                         {'id_code': np.array(['other'])},
                         {'features': np.zeros((1, 48, 512), np.float32)},
                         {'features': np.full((1, 49, 512), np.nan, np.float32)}):
            with self.subTest(field=list(override)):
                self.save_cache(**override)
                self.assertIsNone(self.load())

    def test_legacy_missing_and_corrupt_caches_are_rebuilt(self):
        self.save_cache(metadata=json.dumps({'manifest_sha256': 'legacy'}))
        self.assertIsNone(self.load())
        self.cache.write_bytes(b'broken file')
        self.assertIsNone(self.load())
        self.cache.unlink()
        self.assertIsNone(self.load())

    def test_test_rows_rejected_before_image_access(self):
        self.frame['split'] = 'test'
        self.frame['image_path'] = str(self.root / 'nonexistent-test-image.png')
        with self.assertRaisesRegex(ValueError, 'train/validation'):
            self.current_metadata()

    def test_actual_notebook_cache_blocks_rebuild_and_reuse(self):
        for name in ('Random_Forest', 'CNN_BiLSTM'):
            with self.subTest(notebook=name):
                notebook = json.loads((REPO / f'jupyter notebooks/ENGG2112_{name}.ipynb').read_text(encoding='utf-8'))
                code = ''.join(notebook['cells'][4]['source'])
                block = code[code.index('from src.model_support'):code.index('split_indices =')]
                calls = []

                def extract(frame):
                    calls.append(len(frame))
                    return np.zeros((len(frame), 49, 512), np.float32)

                self.cache.write_bytes(b'invalid legacy cache')
                scope = dict(np=np, json=json, feature_frame=self.frame, SPLIT_PATH=self.split,
                             PREPARED_MANIFEST_PATH=self.prepared, PREPROCESSING_CONFIG_PATH=self.config,
                             torch=SimpleNamespace(__version__='test'), ENCODER_ID='fixed',
                             IMAGENET_MEAN=(0.485, 0.456, 0.406), IMAGENET_STD=(0.229, 0.224, 0.225),
                             FEATURE_CACHE=self.cache, extract_feature_sequences=extract)
                with patch.dict('sys.modules', {'torchvision': SimpleNamespace(__version__='test')}):
                    exec(block, scope)
                    exec(block, scope)
                    self.assertEqual(calls, [1])
                    self.image.write_bytes(self.image.read_bytes() + b'changed')
                    exec(block, scope)
                    self.assertEqual(calls, [1, 1])
                    self.assertEqual(scope['feature_labels'].tolist(), [0])


class NotebookIntegrationTests(unittest.TestCase):
    def test_root_discovery_from_root_new_folder_and_nested_folder(self):
        for path in (REPO / 'jupyter notebooks').glob('*.ipynb'):
            notebook = json.loads(path.read_text(encoding='utf-8'))
            definitions = []
            for c in notebook['cells']:
                if c['cell_type'] == 'code':
                    tree = ast.parse(''.join(c['source']))
                    definitions += [n for n in tree.body if isinstance(n, ast.FunctionDef)
                                    and n.name == 'find_project_root']
            self.assertEqual(len(definitions), 1, path.name)
            scope = {'Path': Path}
            exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), scope)
            for start in (REPO, REPO / 'jupyter notebooks', REPO / 'jupyter notebooks/nested'):
                self.assertEqual(scope['find_project_root'](start), REPO, path.name)
            with tempfile.TemporaryDirectory() as d:
                handoff = Path(d)
                (handoff / 'data').mkdir()
                (handoff / 'data/train.csv').touch()
                (handoff / 'artifacts/splits').mkdir(parents=True)
                self.assertEqual(scope['find_project_root'](handoff / 'notebooks'), handoff)

    def test_readme_local_links_exist(self):
        text = (REPO / 'README.md').read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if not target.startswith(('http:', 'https:', '#')):
                self.assertTrue((REPO / unquote(target)).exists(), target)

    def test_model_test_switches_stay_locked(self):
        for name in ('Model_Training', 'Random_Forest', 'CNN_BiLSTM'):
            p = REPO / f'jupyter notebooks/ENGG2112_{name}.ipynb'
            notebook = json.loads(p.read_text(encoding='utf-8'))
            switches = []
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                for node in ast.parse(''.join(cell['source'])).body:
                    if isinstance(node, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == 'CONFIRM_REVIEWED_FROZEN_SPLIT'
                            for t in node.targets):
                        switches.append(ast.literal_eval(node.value))
            self.assertEqual(switches, [False], name)


if __name__ == '__main__':
    unittest.main()
