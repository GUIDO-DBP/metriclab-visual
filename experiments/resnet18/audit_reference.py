"""Reproduce the original XGBoost settings, not a hyperparameter search.

The paired descriptor experiment limits BOTH classifiers to two threads. If
that replay differs from the historical result, verify the ORIGINAL settings
unchanged (including original thread count), without selecting a new setting.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from .common import BASE_DATA, load_bin, load_json, save_json


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run
    baseline=load_json(run/'baseline.json')
    params=dict(baseline['models']['xgboost']['training_params'])
    if isinstance(params.get('missing'),str) and params['missing'].lower()=='nan': params['missing']=np.nan
    pointer=load_json(BASE_DATA/'active_classifiers.json')
    original=XGBClassifier(); original.load_model(BASE_DATA/pointer['directory']/'xgboost.json')
    x_test=load_bin(BASE_DATA/'query.bin'); y_test=np.load(run/'query_labels.npy')
    original_prediction=original.predict(x_test)
    # Parameters fixed to the historical configuration, not changed after scoring.
    replay=XGBClassifier(**params)
    replay.fit(load_bin(BASE_DATA/'train.bin'),np.load(run/'train_labels.npy'))
    replay_prediction=replay.predict(x_test)
    fresh=np.load(run/'models/hsv_xgboost_predictions.npy')
    result={'purpose':'Reproducibility audit only; no model/configuration selected or activated.',
        'historical_threads':params['n_jobs'], 'paired_experiment_threads':load_json(run/'protocol.json')['threads'],
        'historical_native_accuracy':float(accuracy_score(y_test,original_prediction)),
        'historical_params_replay_accuracy':float(accuracy_score(y_test,replay_prediction)),
        'paired_experiment_baseline_accuracy':float(accuracy_score(y_test,fresh)),
        'disagreements_original_vs_original_params_replay':int(np.sum(original_prediction!=replay_prediction)),
        'disagreements_original_vs_paired_baseline':int(np.sum(original_prediction!=fresh)),
        'test_used_for_fitting_or_parameter_selection':False}
    save_json(run/'historical_reproducibility_audit.json',result)
    print(result)


if __name__=='__main__': main()
