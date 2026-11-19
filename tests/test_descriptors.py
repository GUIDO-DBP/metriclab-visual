import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from backend.core import ROOT,DATA,DESCRIPTOR
from backend.descriptors import RESNET_ID,ResNetRegistry,digest
from backend.main import app,state,describe_bytes


class Descriptors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context=TestClient(app); cls.client=cls.context.__enter__()
        cls.manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
        cls.raw=(ROOT/cls.manifest['query'][0]['path']).read_bytes()

    @classmethod
    def tearDownClass(cls): cls.context.__exit__(None,None,None)

    def upload(self,descriptor=DESCRIPTOR,endpoint='/api/classify',**data):
        return self.client.post(endpoint,files={'file':('sample.jpg',self.raw,'image/jpeg')},data={'descriptor_id':descriptor,**data})

    def test_all_models_both_descriptors_with_real_category_examples(self):
        for descriptor,size in [(DESCRIPTOR,144),(RESNET_ID,512)]:
            for key in ['gini','entropy','forest','xgboost','lightgbm']:
                with self.subTest(descriptor=descriptor,model=key):
                    response=self.upload(descriptor,model=key)
                    self.assertEqual(response.status_code,200,response.text)
                    result=response.json(); prediction=result['predictions'][0]
                    self.assertEqual(len(result['descriptor']),size)
                    self.assertEqual(prediction['descriptor_id'],descriptor)
                    self.assertEqual(len(prediction['probabilities']),101)
                    self.assertAlmostEqual(sum(p['value'] for p in prediction['probabilities']),1,places=5)
                    examples=prediction['examples']; self.assertEqual(len(examples),6)
                    self.assertEqual(len({e['id'] for e in examples}),6)
                    for example in examples:
                        row=self.manifest['train'][example['id']]
                        self.assertEqual(example['split'],'train')
                        self.assertEqual(row['category'],prediction['category'])
                        self.assertEqual(example['category'],prediction['category'])
                        self.assertNotIn('distance',example)
                        image=self.client.get(example['url'])
                        self.assertEqual(image.status_code,200)
                        self.assertEqual(image.content,(ROOT/row['path']).read_bytes())

    def test_comparison_extracts_once_and_matches_the_saved_embedding(self):
        with patch('backend.main.describe_bytes',wraps=describe_bytes) as extract:
            hsv=self.upload(compare='true').json()
            self.assertEqual(extract.call_count,1)
        manager=state['resnet']
        with patch.object(manager,'extract',wraps=manager.extract) as extract:
            cnn=self.upload(RESNET_ID,compare='true').json()
            self.assertEqual(extract.call_count,1)
        for response in [hsv,cnn]:
            self.assertEqual([p['key'] for p in response['predictions']],['gini','forest','xgboost','lightgbm'])
            self.assertTrue(all(p['descriptor_id']==response['descriptor_id'] for p in response['predictions']))
        parent=ROOT/'runtime/experiments/resnet18/20261118T195917Z-42d3213e'
        np.testing.assert_allclose(cnn['descriptor'],np.load(parent/'query_resnet18.npy',mmap_mode='r')[0],rtol=2e-4,atol=2e-5)
        self.assertNotEqual(hsv['vector_sha256'],cnn['vector_sha256'])
        self.assertIn(cnn['extraction']['device'],['cpu','cuda'])

    def test_dimension_guard_unknown_descriptor_and_reversible_disable(self):
        self.assertEqual(self.upload('unknown').status_code,422)
        self.assertEqual(self.upload(RESNET_ID,model='knn').status_code,422)
        manager=state['resnet']
        with patch.object(manager,'extract',return_value=(np.zeros(144,dtype=np.float32),{'device':'cpu','cache_hit':False})):
            self.assertEqual(self.upload(RESNET_ID).status_code,422)
        with patch.dict(os.environ,{'METRICLAB_ENABLE_RESNET':'0'}):
            disabled=ResNetRegistry(self.manifest)
            self.assertFalse(disabled.available)
            try:
                state['resnet']=disabled
                self.assertEqual(self.upload(RESNET_ID).status_code,409)
                self.assertEqual(self.upload(DESCRIPTOR).status_code,200)
            finally: state['resnet']=manager
        with patch.dict(os.environ,{'METRICLAB_RESNET_PYTHON':str(ROOT/'runtime/does-not-exist/python.exe')}):
            self.assertFalse(ResNetRegistry(self.manifest).available)

    def test_explanations_metrics_and_gallery_api_are_descriptor_scoped(self):
        for descriptor,size in [(DESCRIPTOR,144),(RESNET_ID,512)]:
            response=self.client.get(f'/api/models/gini/importance?descriptor_id={descriptor}').json()
            self.assertEqual(len(response['features']),size)
            self.assertAlmostEqual(sum(f['value'] for f in response['features']),1)
            tree=self.client.get(f'/api/models/gini/tree?descriptor_id={descriptor}').json()
            if descriptor==RESNET_ID:
                self.assertIn('embedding',tree['nodes'][0]['feature_name'])
                self.assertIn('embedding',response['groups'])
            path=self.upload(descriptor,endpoint='/api/models/gini/path').json()['path']
            self.assertTrue(path[-1]['leaf'])
        cv=self.client.get(f'/api/experiments?descriptor_id={RESNET_ID}&evaluation=training_cv').json()
        self.assertEqual(len(cv['cv_metrics']),5)
        self.assertGreater(cv['internal_confirmation']['selected']['f1_macro'],cv['internal_confirmation']['original']['f1_macro'])
        historic=self.client.get(f'/api/experiments?descriptor_id={RESNET_ID}').json()
        self.assertIn('NO es una evaluación independiente nueva',historic['protocol'])
        self.assertEqual(len(historic['metrics']),5)
        self.assertGreater(next(m for m in historic['metrics'] if m['key']=='lightgbm')['accuracy'],.8)
        category=self.manifest['train'][0]['category']
        examples=self.client.get(f'/api/categories/{category}/images').json()['images']
        self.assertEqual(len(examples),6)
        self.assertEqual(self.client.get('/api/categories/not-a-category/images').status_code,404)
        self.assertEqual(self.client.get(f'/api/categories/{category}/images?limit=100').status_code,422)

    def test_original_model_and_experiment_artifacts_preserved(self):
        backup=max((ROOT/'runtime/integration_backups').iterdir())
        protected=json.loads((backup/'immutable_artifacts.json').read_text(encoding='utf-8'))
        catalog_path=ROOT/'resources/manifest.json'
        excluded=json.loads(catalog_path.read_text(encoding='utf-8')).get('excluded_private_logs',[]) if catalog_path.exists() else []
        for path,expected in protected.items():
            if not (ROOT/path).exists() and path in excluded:
                self.assertEqual(Path(path).suffix,'.log')
                continue  # Historical logs with private local paths are not publicly distributed.
            self.assertEqual(digest(ROOT/path),expected,path)


if __name__=='__main__': unittest.main()
