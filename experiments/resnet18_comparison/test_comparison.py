import os
import unittest
from pathlib import Path

import joblib
import numpy as np

from experiments.resnet18.common import ROOT, assert_protected, load_json, sha256, validate_embeddings
from .core import KEYS, OUTPUT_ROOT, assert_parent, calculate_metrics, parent_directory, validate_folds


class Comparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        specified=os.getenv('RESNET_COMPARISON_RUN')
        cls.directory=Path(specified) if specified else max(p for p in OUTPUT_ROOT.iterdir() if (p/'completion.json').exists())
        cls.parent=parent_directory(cls.directory)
        cls.protocol=load_json(cls.directory/'protocol.json')

    def test_existing_application_parent_models_and_embeddings_unchanged(self):
        self.assertGreater(assert_protected(load_json(self.directory/'protected_files.json')),50)
        self.assertGreater(assert_parent(load_json(self.directory/'parent_files.json')),20)
        checks=validate_embeddings(self.parent,rehash_images=True)
        self.assertEqual(checks['train']['shape'],[6941,512])
        self.assertEqual(checks['query']['shape'],[1736,512])
        complete=load_json(self.directory/'completion.json')
        self.assertTrue(complete['rf_xgboost_full_models_reused'])
        self.assertFalse(complete['models_activated'])

    def test_folds_are_training_only_disjoint_stratified_and_shared(self):
        specification=validate_folds(self.directory)
        self.assertEqual(specification['seed'],23)
        coverage=np.zeros(6941,dtype=int)
        labels=np.load(self.parent/'train_labels.npy')
        for fold in specification['folds']:
            a,b=np.array(fold['fit']),np.array(fold['validation'])
            self.assertEqual(np.intersect1d(a,b).size,0)
            self.assertEqual(len(np.unique(labels[a])),101)
            self.assertEqual(len(np.unique(labels[b])),101)
            coverage[b]+=1
        np.testing.assert_array_equal(coverage,np.ones(6941,dtype=int))
        for key in KEYS:
            summary=load_json(self.directory/'cv'/key/'summary.json')
            self.assertEqual(summary['folds_sha256'],sha256(self.directory/'folds.json'))
            self.assertFalse(summary['historical_test_used'])

    def test_all_cv_metrics_means_std_and_oof_are_supported_by_predictions(self):
        truth=np.load(self.parent/'train_labels.npy')
        specification=load_json(self.directory/'folds.json')
        for key in KEYS:
            summary=load_json(self.directory/'cv'/key/'summary.json')
            oof=np.full(6941,-1,dtype=np.int64)
            for fold,record in zip(specification['folds'],summary['folds']):
                prediction=np.load(self.directory/'cv'/key/f"fold_{fold['fold']}_predictions.npy")
                oof[fold['validation']]=prediction
                self.assertFalse(record['historical_test_opened'])
                self.assertEqual(record['parameters_sha256'],sha256(self.directory/'protocol.json'))
                metrics=calculate_metrics(truth[fold['validation']],prediction)
                for metric,value in metrics.items(): self.assertAlmostEqual(record[metric],value)
            np.testing.assert_array_equal(oof,np.load(self.directory/'cv'/key/'oof_predictions.npy'))
            for metric,value in calculate_metrics(truth,oof).items():
                self.assertAlmostEqual(summary['oof'][metric],value)
                values=[r[metric] for r in summary['folds']]
                self.assertAlmostEqual(summary['mean'][metric],np.mean(values))
                self.assertAlmostEqual(summary['std'][metric],np.std(values,ddof=1))

    def test_candidate_is_fixed_from_cv_and_not_historical_test(self):
        selected=load_json(self.directory/'cv_selection.json')
        rows=[load_json(self.directory/'cv'/key/'summary.json') for key in KEYS]
        expected=sorted(rows,key=lambda r:(-r['mean']['f1_macro'],-r['mean']['balanced_accuracy'],r['key']))[0]['key']
        self.assertEqual(selected['candidate'],expected)
        self.assertFalse(selected['historical_test_metrics_read_for_selection'])
        self.assertFalse(self.protocol['test_for_parameter_or_model_selection'])
        self.assertFalse(self.protocol['hyperparameter_search'])
        for key in KEYS:
            result=load_json(self.directory/'final'/key/'results.json')
            self.assertEqual(result['cv_selection_sha256'],sha256(self.directory/'cv_selection.json'))
        gini=dict(self.protocol['parameters']['gini']); entropy=dict(self.protocol['parameters']['entropy'])
        self.assertEqual(gini.pop('criterion'),'gini'); self.assertEqual(entropy.pop('criterion'),'entropy')
        self.assertEqual(gini,entropy)

    def test_historical_test_is_marked_reused_and_models_are_persistent(self):
        truth=np.load(self.parent/'query_labels.npy')
        x=np.load(self.parent/'query_resnet18.npy',mmap_mode='r')
        for key in KEYS:
            row=load_json(self.directory/'final'/key/'results.json')
            self.assertIn('NOT a new independent validation',row['test_role'])
            predicted=np.load(self.directory/'final'/key/'predictions.npy')
            for metric,value in calculate_metrics(truth,predicted).items(): self.assertAlmostEqual(row[metric],value)
            self.assertEqual(sum(map(sum,row['confusion'])),1736)
            self.assertEqual(row['parameters'],self.protocol['parameters'][key])
            self.assertEqual(row['model_sha256'],sha256(ROOT/row['model_path']))
            self.assertEqual(row['full_model_reused'],key in {'forest','xgboost'})
            self.assertGreater(row['inference_ms'],0)
            self.assertEqual(row['inference_samples'],self.protocol['benchmark_samples'])
            if key in {'gini','entropy','lightgbm'}:
                bundle=joblib.load(ROOT/row['model_path'])
                self.assertEqual(bundle['dimensions'],512)
                probability=bundle['model'].predict_proba(x[:3])
                self.assertEqual(probability.shape,(3,101))
                np.testing.assert_allclose(probability.sum(axis=1),1,atol=1e-6)
                np.testing.assert_array_equal(bundle['model'].predict(x[:3]),predicted[:3])

    def test_real_gallery_proposal_uses_training_images_with_correct_category(self):
        proposal=load_json(self.directory/'gallery_examples_proposal.json')
        self.assertFalse(proposal['active'])
        train=load_json(self.parent/'split.json')['train']
        self.assertEqual(len(proposal['categories']),101)
        for category,items in proposal['categories'].items():
            self.assertEqual(len(items),6)
            for item in items:
                row=train[item['image_id']]
                self.assertEqual(item['split'],'train')
                self.assertEqual(row['category'],category)
                self.assertEqual(item['category'],category)
                self.assertEqual(row['path'],item['path'])
                self.assertTrue((ROOT/item['path']).is_file())
                self.assertEqual(item['url'],f"/api/images/train/{item['image_id']}")


if __name__=='__main__': unittest.main()
