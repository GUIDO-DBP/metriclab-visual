"""Integration tests against prepared local images and the actual C++ DLL."""
import concurrent.futures
import io
import json
import unittest

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app, state
from backend.core import DATA, ROOT, describe_bytes, extractor, load_vectors


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()
        cls.manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
        cls.image = ROOT / cls.manifest['query'][0]['path']
        cls.raw = cls.image.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def upload(self, endpoint, **data):
        return self.client.post(endpoint, files={'file': ('sample.jpg', self.raw, 'image/jpeg')}, data=data)

    def test_descriptor_compatibility_and_split(self):
        self.assertTrue(np.allclose(describe_bytes(self.raw), extractor.describe(self.image)))
        self.assertFalse({r['path'] for r in self.manifest['train']} & {r['path'] for r in self.manifest['query']})
        self.assertEqual(load_vectors(DATA / 'query.bin').shape, (len(self.manifest['query']), 144))

    def test_real_indices_match_oracle(self):
        for k in [5, 10, 20]:
            response = self.upload('/api/compare', k=k)
            self.assertEqual(response.status_code, 200, response.text)
            runs = response.json()['runs']
            self.assertEqual(len(runs), 3)
            self.assertEqual(runs[0]['distances'], len(self.manifest['train']))
            for run in runs:
                self.assertTrue(run['oracle_match'])
                self.assertTrue(np.allclose([n['distance'] for n in runs[0]['neighbors']], [n['distance'] for n in run['neighbors']]))
                self.assertEqual(len(run['neighbors']), k)
        self.assertTrue(response.json()['trace'])

    def test_selected_algorithm_and_trace(self):
        for algorithm in [0, 1, 2]:
            result = self.upload('/api/search', algorithm=algorithm).json()
            self.assertEqual([r['algorithm'] for r in result['runs']], [algorithm])
            self.assertEqual(bool(result['trace']), algorithm == 1)
        tree = self.client.get('/api/tree').json()
        # Leaf buckets may hold multiple objects: node count need not equal n.
        self.assertGreater(len(tree), 0)
        self.assertLessEqual(len(tree), len(self.manifest['train']))
        self.assertTrue(all(child == -1 or 0 <= child < len(tree) for node in tree for child in [node['left'], node['right']]))
        self.assertEqual(tree[0]['id'], 0)

    def test_models_and_experiments(self):
        available = [m['key'] for m in self.client.get('/api/status').json()['models'] if m['available']]
        response = self.upload('/api/classify', compare='true')
        self.assertEqual(response.status_code, 200, response.text)
        preds = response.json()['predictions']
        self.assertEqual([p['key'] for p in preds], available)
        for pred in preds:
            self.assertIn(pred['category'], {r['category'] for r in self.manifest['train']})
            self.assertTrue(all(0 <= p['value'] <= 1 for p in pred['probabilities']))
            if pred['key'] in {'gini', 'entropy'}:
                self.assertEqual(pred['decisions'][-1]['feature'], -2)
            if pred['key'] == 'knn':
                self.assertEqual(len(pred['neighbors']), 5)
        experiments = self.client.get('/api/experiments').json()
        self.assertEqual(experiments['fingerprint'], self.manifest['fingerprint'])
        for m in experiments['metrics']:
            self.assertEqual(sum(map(sum, m['confusion'])), len(self.manifest['query']))

    def test_validation_and_paths(self):
        self.assertEqual(self.upload('/api/search', k=0).status_code, 422)
        self.assertEqual(self.upload('/api/search', k=21).status_code, 422)
        self.assertEqual(self.upload('/api/search', algorithm=3).status_code, 422)
        self.assertEqual(self.upload('/api/classify', model='unknown').status_code, 422)
        self.assertEqual(self.client.post('/api/search', files={'file': ('broken.png', b'not an image')}).status_code, 422)
        image = io.BytesIO(); Image.new('RGB', (5,5)).save(image, format='GIF')
        self.assertEqual(self.client.post('/api/search', files={'file': ('fake.jpg', image.getvalue())}).status_code, 422)
        self.assertEqual(self.client.get('/api/images/bad/0').status_code, 404)
        self.assertEqual(self.client.get('/api/images/train/-1').status_code, 404)
        self.assertEqual(self.client.get('/api/images/query/0').status_code, 200)

    def test_missing_model_and_missing_index(self):
        bundle = state['bundle']; engine = state['engine']
        try:
            state['bundle'] = None
            self.assertEqual(self.upload('/api/classify', model='gini').status_code, 409)
            state['engine'] = None
            self.assertFalse(self.client.get('/api/status').json()['ready'])
            self.assertEqual(self.upload('/api/search').status_code, 503)
        finally:
            state['bundle'] = bundle; state['engine'] = engine

    def test_parallel_queries(self):
        vector = describe_bytes(self.raw)
        engine = state['engine']
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: engine.search(vector, 5, 1), range(8)))
        expected = results[0]['runs'][0]['neighbors']
        self.assertTrue(all(r['runs'][0]['neighbors'] == expected for r in results))


if __name__ == '__main__':
    unittest.main()
