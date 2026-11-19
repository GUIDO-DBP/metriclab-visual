"""Append-only comparison run; parent embeddings/models are read-only."""
import argparse
import copy
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.resnet18.common import (DESCRIPTOR, ROOT, OUTPUT_ROOT as PARENT_ROOT,
    assert_protected, load_json, protected_files, save_json, sha256, validate_embeddings)
from .core import KEYS, OUTPUT_ROOT, assert_parent, freeze_parent, validate_folds


def worker(run,module,*args):
    env=os.environ.copy(); env['PYTHONIOENCODING']='utf-8'
    with (run/'execution.log').open('a',encoding='utf-8') as log:
        command=[sys.executable,'-m',module,'--run',str(run),*map(str,args)]
        with subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1) as process:
            for line in process.stdout:
                log.write(line); log.flush(); print(line,end='',flush=True)
            if process.wait(): raise RuntimeError(f'{module} failed; see execution.log. No app/parent model was overwritten.')


def create_run(parent,threads,benchmark_samples):
    checks=validate_embeddings(parent,rehash_images=True)
    baseline=load_json(parent/'baseline.json')
    parameters={}
    sources={}
    for key in KEYS:
        if key in {'forest','xgboost'}:
            previous=load_json(parent/'models'/f'resnet18_{key}_results.json')
            params=copy.deepcopy(previous['params'])
            sources[key]='saved existing ResNet full-training model parameters'
            if params['n_jobs']!=threads: raise ValueError('Preserve the prior RF/XGBoost two-thread configuration.')
        else:
            params=copy.deepcopy(baseline['models'][key].get('training_params') or baseline['models'][key]['params'])
            if key=='lightgbm': params['n_jobs']=threads
            sources[key]='historical fixed classifier configuration, unchanged except bounded n_jobs'
        parameters[key]=params
    name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    run=OUTPUT_ROOT/name; run.mkdir(parents=True,exist_ok=False)
    save_json(run/'protected_files.json',protected_files())
    save_json(run/'parent_files.json',freeze_parent(parent))
    protocol={'run_id':name,'created_at':datetime.now(timezone.utc).isoformat(),
        'parent_run':parent.relative_to(ROOT).as_posix(), 'descriptor':DESCRIPTOR,'dimensions':512,
        'split_fingerprint':checks['original_fingerprint'],'n_original_train':6941,'n_historical_test':1736,
        'cv':{'method':'StratifiedKFold','n_splits':5,'shuffle':True,'seed':23,
              'scope':'original training partition ONLY','estimators':'new unfitted estimator per model/fold'},
        'parameters':parameters,'parameter_sources':sources,'model_seed':0,
        'threads':threads,'benchmark_samples':benchmark_samples,'hyperparameter_search':False,
        'candidate_rule':'largest mean CV validation F1 macro; tie by mean balanced accuracy, then key',
        'historical_test_role':'reused previously evaluated set; descriptive comparison, not new independent validation',
        'test_for_parameter_or_model_selection':False,'activate_models':False,
        'versions':{p:version(p) for p in ['numpy','scikit-learn','xgboost','lightgbm','joblib','scipy']},
        'python':platform.python_version()}
    save_json(run/'protocol.json',protocol)  # Fixed before creating/fitting any CV estimators.
    labels=np.load(parent/'train_labels.npy')
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=23)
    folds=[{'fold':i,'fit':fit.tolist(),'validation':validation.tolist()}
           for i,(fit,validation) in enumerate(splitter.split(np.zeros(len(labels)),labels),1)]
    save_json(run/'folds.json',{'universe_split':'train','universe_n':6941,
        'indices':'zero-based rows of parent train_resnet18.npy, NOT mixed global train/test indices',
        'seed':23,'folds':folds})
    validate_folds(run)
    return run


def freeze_recommendation(run):
    summaries=[load_json(run/'cv'/key/'summary.json') for key in KEYS]
    ranking=sorted(summaries,key=lambda row:(-row['mean']['f1_macro'],-row['mean']['balanced_accuracy'],row['key']))
    selection={'created_at':datetime.now(timezone.utc).isoformat(),'source':'five-fold CV inside original training only',
        'candidate':ranking[0]['key'],'ranking_by_cv_only':[r['key'] for r in ranking],
        'rule':load_json(run/'protocol.json')['candidate_rule'],
        'cv_summaries_sha256':{key:sha256(run/'cv'/key/'summary.json') for key in KEYS},
        'historical_test_metrics_read_for_selection':False,'model_activated':False}
    if (run/'cv_selection.json').exists():
        previous=load_json(run/'cv_selection.json')
        for field in ['candidate','ranking_by_cv_only','cv_summaries_sha256']:
            if previous[field]!=selection[field]: raise ValueError('Frozen CV recommendation cannot change.')
    else: save_json(run/'cv_selection.json',selection)
    print('Training-only CV candidate:',selection['candidate'],'(no application activation)',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--run',type=Path,help='Resume a previously created extension; do not overwrite the parent.')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--benchmark-samples',type=int,default=128)
    args=parser.parse_args()
    if args.threads!=2 or not 1<=args.benchmark_samples<=1736: parser.error('Use the original two-thread model settings and 1–1736 benchmark samples.')
    if args.run:
        run=args.run.resolve()
        if not run.is_relative_to(OUTPUT_ROOT.resolve()): raise ValueError('Resume directory is outside extension outputs.')
        protocol=load_json(run/'protocol.json')
        if protocol['threads']!=args.threads or protocol['benchmark_samples']!=args.benchmark_samples: raise ValueError('Resume with original settings.')
        parent=(ROOT/protocol['parent_run']).resolve()
    else:
        parent=args.parent.resolve() if args.parent else max(p for p in PARENT_ROOT.iterdir() if (p/'completion.json').exists())
        if not parent.is_relative_to(PARENT_ROOT.resolve()): raise ValueError('Use an existing completed ResNet experiment.')
        run=create_run(parent,args.threads,args.benchmark_samples)
    assert_protected(load_json(run/'protected_files.json'))
    assert_parent(load_json(run/'parent_files.json'))
    print('Extension directory:',run,flush=True)
    for key in KEYS:
        if not (run/'cv'/key/'summary.json').exists(): worker(run,'experiments.resnet18_comparison.cross_validate','--model',key)
    freeze_recommendation(run)
    for key in KEYS:
        if not (run/'final'/key/'results.json').exists(): worker(run,'experiments.resnet18_comparison.final_models','--model',key)
    worker(run,'experiments.resnet18_comparison.report')
    app_count=assert_protected(load_json(run/'protected_files.json'))
    parent_count=assert_parent(load_json(run/'parent_files.json'))
    save_json(run/'completion.json',{'complete':True,'app_files_unchanged':app_count,'parent_files_unchanged':parent_count,
        'original_embeddings_reused':True,'rf_xgboost_full_models_reused':True,'hsv_models_unchanged':True,
        'historical_test_is_new_validation':False,'models_activated':False,
        'finished_at':datetime.now(timezone.utc).isoformat()})
    print('Comparison complete; prior models/embeddings and application unchanged.',flush=True)


if __name__=='__main__': main()
