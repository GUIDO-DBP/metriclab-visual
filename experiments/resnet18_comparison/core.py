import copy
import gc
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from experiments.resnet18.common import ROOT, load_json, sha256

KEYS = ('gini', 'entropy', 'forest', 'xgboost', 'lightgbm')
NAMES = {'gini':'Decision Tree — Gini','entropy':'Decision Tree — Entropía',
         'forest':'Random Forest','xgboost':'XGBoost','lightgbm':'LightGBM'}
OUTPUT_ROOT = ROOT / 'runtime/experiments/resnet18_comparison'


def parent_directory(run):
    parent = (ROOT / load_json(Path(run)/'protocol.json')['parent_run']).resolve()
    if not parent.is_relative_to(ROOT/'runtime/experiments/resnet18'):
        raise ValueError('Invalid source experiment.')
    return parent


def freeze_parent(parent):
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(Path(parent).rglob('*')) if p.is_file()}


def assert_parent(snapshot):
    for relative, expected in snapshot.items():
        if sha256(ROOT/relative) != expected:
            raise ValueError(f'Prior experiment changed: {relative}')
    return len(snapshot)


def make_model(run, key):
    params = copy.deepcopy(load_json(Path(run)/'protocol.json')['parameters'][key])
    if key in {'gini','entropy'}:
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**params)
    if key == 'forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    if key == 'lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    from xgboost import XGBClassifier
    if isinstance(params.get('missing'),str) and params['missing'].lower() == 'nan':
        params['missing'] = np.nan
    return XGBClassifier(**params)


def calculate_metrics(truth, predicted, include_confusion=False):
    result = {'accuracy':float(accuracy_score(truth,predicted)),
              'balanced_accuracy':float(balanced_accuracy_score(truth,predicted)),
              'f1_macro':float(f1_score(truth,predicted,average='macro',zero_division=0))}
    if include_confusion:
        result['confusion'] = confusion_matrix(truth,predicted,labels=np.arange(101)).tolist()
    return result


def validate_folds(run):
    specification = load_json(Path(run)/'folds.json')
    if specification['universe_split'] != 'train' or specification['universe_n'] != 6941:
        raise ValueError('CV must refer only to the original training matrix.')
    coverage = np.zeros(6941,dtype=int)
    labels = np.load(parent_directory(run)/'train_labels.npy')
    for fold in specification['folds']:
        fit,validation = np.asarray(fold['fit'],dtype=int),np.asarray(fold['validation'],dtype=int)
        if (fit<0).any() or (validation<0).any() or (fit>=6941).any() or (validation>=6941).any():
            raise ValueError('Fold row outside original training split.')
        if np.intersect1d(fit,validation).size or len(fit)+len(validation)!=6941:
            raise ValueError('Overlapping/incomplete fold.')
        if not np.array_equal(np.sort(np.r_[fit,validation]),np.arange(6941)):
            raise ValueError('Fold does not partition the full training split.')
        if len(np.unique(labels[fit]))!=101 or len(np.unique(labels[validation]))!=101:
            raise ValueError('A class is missing in a fold.')
        coverage[validation]+=1
    if not np.all(coverage==1) or len(specification['folds'])!=5:
        raise ValueError('Each training row must have exactly one out-of-fold prediction.')
    return specification


def unload(model):
    del model
    gc.collect()
