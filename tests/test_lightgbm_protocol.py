import json
import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runtime/experiments/lightgbm_resnet/20261118T211652Z-600d0ec2'
def read(name): return json.loads((RUN/name).read_text(encoding='utf-8'))

class LightGBMProtocol(unittest.TestCase):
    def test_selection_uses_only_development_folds(self):
        p=read('protocol.json'); s=read('selection.json')
        dev=set(p['development']); hold=set(p['internal_confirmation'])
        self.assertFalse(dev & hold)
        self.assertEqual(dev | hold,set(range(6941)))
        self.assertEqual(len(hold),1389)
        seen=[]
        for fold in p['folds']:
            fit,val=set(fold['fit']),set(fold['validation'])
            self.assertFalse(fit & val); self.assertEqual(fit | val,dev)
            self.assertFalse((fit | val) & hold); seen.extend(val)
        self.assertEqual(sorted(seen),sorted(dev))
        candidates=read('search.json')
        winner=sorted(candidates,key=lambda r:(-r['mean']['f1_macro'],-r['mean']['balanced_accuracy'],r['candidate']))[0]
        self.assertEqual(s['candidate'],winner['candidate'])
        self.assertEqual(s['parameters'],winner['parameters'])
        self.assertFalse(s['test_used']); self.assertFalse(s['internal_confirmation_used'])
        self.assertFalse(p['test_for_selection']); self.assertFalse(p['confirmation_for_selection'])

    def test_recorded_metrics_match_predictions_and_labels(self):
        p=read('protocol.json'); parent=ROOT/p['parent_run']
        y=np.load(parent/'train_labels.npy')
        def check(actual,expected):
            truth,pred=actual
            self.assertAlmostEqual(accuracy_score(truth,pred),expected['accuracy'])
            self.assertAlmostEqual(balanced_accuracy_score(truth,pred),expected['balanced_accuracy'])
            self.assertAlmostEqual(f1_score(truth,pred,average='macro',zero_division=0),expected['f1_macro'])
        for candidate in read('search.json'):
            for k,fold in enumerate(p['folds']):
                pred=np.load(RUN/f"candidate_{candidate['candidate']}_fold_{k}_predictions.npy")
                check((y[fold['validation']],pred),candidate['folds'][k])
        for name,expected in read('internal_confirmation.json').items():
            check((y[p['internal_confirmation']],np.load(RUN/f'confirmation_{name}_predictions.npy')),expected)
        check((np.load(parent/'query_labels.npy'),np.load(RUN/'historical_predictions.npy')),read('results.json'))

if __name__=='__main__': unittest.main()
