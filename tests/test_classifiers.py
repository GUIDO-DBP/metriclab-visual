"""Tests of trained artifacts, real predictions, explanations and evaluation."""
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from backend.core import DATA, ROOT, describe_bytes, load_vectors
from backend.main import app, state
from backend.modeling import PRIMARY_KEYS, artifact_directory


class Classifiers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()
        cls.manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
        cls.raw = (ROOT / cls.manifest['query'][0]['path']).read_bytes()
        cls.vector = describe_bytes(cls.raw)
        cls.artifacts = artifact_directory(DATA)
        cls.bundle = state['bundle']

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def upload(self, endpoint='/api/classify', **data):
        return self.client.post(endpoint, files={'file': ('sample.jpg',self.raw,'image/jpeg')}, data=data)

    def test_four_algorithms_same_vector_and_criterion(self):
        with patch('backend.main.describe_bytes', wraps=describe_bytes) as extractor:
            result = self.upload(compare='true').json()
            self.assertEqual(extractor.call_count, 1)
        self.assertEqual([p['key'] for p in result['predictions']], ['gini','forest','xgboost','lightgbm'])
        self.assertEqual(result['unavailable'], [])
        self.assertEqual(sum(result['agreement'].values()), 4)
        entropy = self.upload(compare='true', model='forest', criterion='entropy').json()
        self.assertEqual(entropy['vector_sha256'], result['vector_sha256'])
        self.assertEqual([p['key'] for p in entropy['predictions']], ['entropy','forest','xgboost','lightgbm'])
        self.assertEqual(entropy['predictions'][0]['criterion'], 'entropy')
        self.assertEqual(self.upload(criterion='invalid').status_code, 422)

    def test_all_models_real_probabilities_and_latency(self):
        self.assertTrue(self.client.get('/api/status').json()['classification_ready'])
        for key in PRIMARY_KEYS:
            with self.subTest(model=key):
                response = self.upload(model=key)
                self.assertEqual(response.status_code,200,response.text)
                prediction = response.json()['predictions'][0]
                self.assertGreater(prediction['inference_ms'],0)
                probs = prediction['probabilities']
                self.assertEqual(len(probs),101)
                self.assertAlmostEqual(sum(p['value'] for p in probs),1,places=5)
                self.assertTrue(all(0 <= p['value'] <= 1 for p in probs))
                actual = int(self.bundle['models'][key].predict(self.vector[None])[0])
                self.assertEqual(prediction['category'],self.bundle['classes'][actual])

    def test_serialized_and_native_models_preserve_probabilities(self):
        from xgboost import XGBClassifier
        from lightgbm import Booster
        loaded = joblib.load(self.artifacts/'models.joblib')
        sample = load_vectors(DATA/'query.bin')[:3]
        for key in PRIMARY_KEYS:
            np.testing.assert_allclose(loaded['models'][key].predict_proba(sample), self.bundle['models'][key].predict_proba(sample),atol=1e-10)
            self.assertIn('params',loaded['metadata'][key])
            self.assertIn('versions',loaded['metadata'][key])
        xgb = XGBClassifier(); xgb.load_model(self.artifacts/'xgboost.json')
        np.testing.assert_allclose(xgb.predict_proba(sample),self.bundle['models']['xgboost'].predict_proba(sample),rtol=1e-6)
        lgb = Booster(model_file=str(self.artifacts/'lightgbm.txt'))
        np.testing.assert_allclose(lgb.predict(sample),self.bundle['models']['lightgbm'].predict_proba(sample),rtol=1e-6)

    def test_reported_metrics_match_complete_test_set(self):
        experiments = self.client.get('/api/experiments').json()
        self.assertEqual(experiments['run_id'],self.bundle['run_id'])
        self.assertEqual(experiments['fingerprint'],self.manifest['fingerprint'])
        self.assertEqual([r['key'] for r in experiments['metrics']],list(PRIMARY_KEYS))
        labels = {name:i for i,name in enumerate(self.bundle['classes'])}
        truth = np.array([labels[r['category']] for r in self.manifest['query']])
        vectors = load_vectors(DATA/'query.bin')
        for metric in experiments['metrics']:
            pred = self.bundle['models'][metric['key']].predict(vectors)
            p,r,f,_ = precision_recall_fscore_support(truth,pred,average='macro',zero_division=0)
            self.assertAlmostEqual(metric['accuracy'],accuracy_score(truth,pred))
            self.assertAlmostEqual(metric['balanced_accuracy'],balanced_accuracy_score(truth,pred))
            self.assertAlmostEqual(metric['precision_macro'],p)
            self.assertAlmostEqual(metric['recall_macro'],r)
            self.assertAlmostEqual(metric['f1_macro'],f)
            self.assertEqual(sum(map(sum,metric['confusion'])),len(truth))
            self.assertGreater(metric['inference_ms'],0)
            self.assertEqual(metric['inference_samples'],128)

    def test_real_tree_nodes_and_image_paths(self):
        for key,index in [('gini',0),('entropy',0),('forest',0),('forest',9)]:
            with self.subTest(model=key,tree=index):
                model = self.bundle['models'][key]
                estimator = model.estimators_[index] if key == 'forest' else model
                response = self.client.get(f'/api/models/{key}/tree?tree_index={index}')
                self.assertEqual(response.status_code,200,response.text)
                tree = response.json()
                self.assertLessEqual(len(tree['nodes']),15)
                for node in tree['nodes']:
                    n = node['id']
                    self.assertEqual(node['left'],int(estimator.tree_.children_left[n]))
                    self.assertAlmostEqual(node['impurity'],float(estimator.tree_.impurity[n]))
                    self.assertAlmostEqual(sum(c['proportion'] for c in node['classes']),1)
                path_response = self.upload(f'/api/models/{key}/path',tree_index=index)
                self.assertEqual(path_response.status_code,200,path_response.text)
                nodes = path_response.json()['path']
                self.assertTrue(nodes[-1]['leaf'])
                for node,next_node in zip(nodes,nodes[1:]):
                    expected = node['left'] if self.vector[node['feature']] <= node['threshold'] else node['right']
                    self.assertEqual(next_node['id'],expected)
        for url in ['/api/models/gini/tree?root=99999999','/api/models/gini/tree?max_depth=20','/api/models/forest/tree?tree_index=200','/api/models/xgboost/tree']:
            self.assertEqual(self.client.get(url).status_code,422)

    def test_importances_from_trained_models(self):
        for key in PRIMARY_KEYS:
            result = self.client.get(f'/api/models/{key}/importance').json()
            self.assertEqual(len(result['features']),144)
            self.assertAlmostEqual(sum(r['value'] for r in result['features']),1,places=5)
            expected = np.asarray(self.bundle['models'][key].feature_importances_,dtype=float)
            expected /= expected.sum()
            for feature in result['features']:
                self.assertAlmostEqual(feature['value'],expected[feature['index']])
        self.assertEqual(self.client.get('/api/models/forest').json()['tree_count'],200)
        self.assertEqual(self.client.get('/api/models/xgboost').json()['boosting_rounds'],80)

    def test_classifier_independent_of_metric_engine_and_unavailable_states(self):
        engine,bundle = state['engine'],state['bundle']
        try:
            state['engine'] = None
            self.assertTrue(self.client.get('/api/status').json()['classification_ready'])
            self.assertEqual(self.upload(model='xgboost').status_code,200)
            self.assertEqual(self.upload('/api/search').status_code,503)
            state['bundle'] = None
            comparison = self.upload(compare='true').json()
            self.assertEqual(comparison['predictions'],[])
            self.assertEqual(len(comparison['unavailable']),4)
            self.assertEqual(self.upload(model='gini').status_code,409)
        finally:
            state['engine'],state['bundle'] = engine,bundle


if __name__ == '__main__':
    unittest.main()
