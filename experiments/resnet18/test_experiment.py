"""Validation of the completed real experiment, separate from application tests."""
import gc
import os
import time
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .common import OUTPUT_ROOT, assert_protected, load_json, save_json, validate_embeddings


class Experiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        specified = os.getenv('RESNET_EXPERIMENT_RUN')
        cls.directory = Path(specified) if specified else max(p for p in OUTPUT_ROOT.iterdir() if (p/'completion.json').exists())
        cls.results = load_json(cls.directory/'results.json')

    def test_exact_split_and_full_row_label_correspondence(self):
        checks = validate_embeddings(self.directory,rehash_images=True)
        self.assertEqual(checks['train']['shape'],[6941,512])
        self.assertEqual(checks['query']['shape'],[1736,512])
        self.assertEqual(len(load_json(self.directory/'classes.json')),101)

    def test_four_models_real_metrics_and_no_test_tuning(self):
        truth = np.load(self.directory/'query_labels.npy')
        for row in self.results['models']:
            self.assertFalse(row['test_used_for_fit'])
            self.assertFalse(row['hyperparameter_search'])
            self.assertTrue(row['persistence_verified'])
            self.assertEqual(sum(map(sum,row['confusion'])),1736)
            prediction = np.load(self.directory/'models'/f"{row['descriptor']}_{row['key']}_predictions.npy")
            self.assertAlmostEqual(row['accuracy'],accuracy_score(truth,prediction))
            self.assertAlmostEqual(row['balanced_accuracy'],balanced_accuracy_score(truth,prediction))
            self.assertAlmostEqual(row['f1_macro'],f1_score(truth,prediction,average='macro'))
            self.assertGreater(row['train_s'],0)
            self.assertGreater(row['inference_ms'],0)
            self.assertEqual(row['inference_samples'],min(self.results['protocol']['benchmark_samples'],1736))
        for key in ['forest','xgboost']:
            old = next(r for r in self.results['models'] if r['key']==key and r['descriptor']=='hsv')
            new = next(r for r in self.results['models'] if r['key']==key and r['descriptor']=='resnet18')
            self.assertEqual(old['params'],new['params'])
            self.assertAlmostEqual(old['historical_accuracy_delta_pp'],100*(old['accuracy']-old['historical_accuracy']))
            if key == 'forest': self.assertAlmostEqual(old['accuracy'],old['historical_accuracy'])
        self.assertEqual(len(self.results['models']),4)

    def test_frozen_pretrained_network_and_real_image_checks(self):
        extraction = self.results['extraction']
        self.assertEqual(extraction['weights'],'ResNet18_Weights.IMAGENET1K_V1')
        self.assertFalse(extraction['fine_tuning'])
        self.assertEqual(extraction['trainable_parameters'],0)
        self.assertEqual(extraction['state_sha256_before'],extraction['state_sha256_after'])
        self.assertTrue(extraction['checkpoint_sha256'].startswith('f37072fd'))
        self.assertEqual(len(extraction['row_reextraction_checks']),6)

    def test_cpu_extractor_with_official_preprocessing_matches_saved_vectors(self):
        import torch
        from .extract import build_extractor, preprocess_row, state_hash
        network, preprocessing, device, metadata = build_extractor('cpu',2)
        self.assertEqual(device,'cpu')
        self.assertFalse(network.training)
        self.assertTrue(all(not p.requires_grad for p in network.parameters()))
        self.assertEqual(preprocessing.crop_size,[224])
        self.assertEqual(preprocessing.resize_size,[256])
        self.assertEqual(preprocessing.mean,[.485,.456,.406])
        self.assertEqual(preprocessing.std,[.229,.224,.225])
        manifest = load_json(self.directory/'split.json')
        before = state_hash(network)
        deltas = []
        with torch.inference_mode():
            for split,index in [('train',0),('query',1735)]:
                vector = network(preprocess_row(manifest[split][index],preprocessing).unsqueeze(0)).numpy()[0]
                stored = np.load(self.directory/f'{split}_resnet18.npy',mmap_mode='r')[index]
                self.assertEqual(vector.shape,(512,))
                np.testing.assert_allclose(vector,stored,rtol=3e-4,atol=3e-5)
                deltas.append(float(np.max(np.abs(vector-stored))))
            timings = []
            for i in np.linspace(0,1735,16,dtype=int):
                start = time.perf_counter_ns()
                network(preprocess_row(manifest['query'][i],preprocessing).unsqueeze(0)).numpy()
                timings.append((time.perf_counter_ns()-start)/1e6)
        self.assertEqual(before,state_hash(network))
        save_json(self.directory/'cpu_validation.json',{'device':device,'cpu_matches_saved_embeddings':True,
            'max_absolute_errors':deltas,'samples':len(timings),'mean_extraction_ms':float(np.mean(timings)),
            'p95_extraction_ms':float(np.percentile(timings,95)),'threads':2,
            'includes':'image load + official preprocessing + full frozen CNN; no classifier',
            'note':'16-image CPU functionality/latency check; GPU benchmark uses 128 images.'})
        del network
        gc.collect()

    def test_application_frontend_and_active_models_unchanged(self):
        count = assert_protected(load_json(self.directory/'protected_files.json'))
        self.assertGreater(count,50)
        complete = load_json(self.directory/'completion.json')
        self.assertTrue(complete['complete'])
        self.assertFalse(complete['models_activated'])
        self.assertFalse(complete['frontend_changed'])


if __name__ == '__main__': unittest.main()
