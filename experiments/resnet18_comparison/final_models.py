"""Full-training fits for NEW models only; read-only references for RF/XGBoost."""
import argparse
import gc
import time
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.resnet18.common import ROOT, load_json, save_json, sha256
from .core import KEYS, NAMES, calculate_metrics, make_model, parent_directory


def evaluate(run,key):
    run=Path(run)
    if not (run/'cv_selection.json').exists():
        raise ValueError('Complete train-only CV and freeze the recommendation before test evaluation.')
    protocol=load_json(run/'protocol.json')
    parent=parent_directory(run)
    directory=run/'final'/key
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'results.json').exists(): raise ValueError('Final results already exist.')
    classes=load_json(parent/'classes.json')
    x={s:np.load(parent/f'{s}_resnet18.npy',mmap_mode='r') for s in ['train','query']}
    y={s:np.load(parent/f'{s}_labels.npy') for s in ['train','query']}
    reused=key in {'forest','xgboost'}
    if reused:
        source=parent/'models'/f'resnet18_{key}.joblib'
        previous=load_json(parent/'models'/f'resnet18_{key}_results.json')
        if sha256(source)!=previous['model_sha256']: raise ValueError('Original ResNet model changed.')
        bundle=joblib.load(source)
        model=bundle['model']
        if bundle['classes']!=classes or model.n_features_in_!=512: raise ValueError('Model not compatible with parent embeddings.')
        if previous['params']!=protocol['parameters'][key]: raise ValueError('CV and reused full model parameters differ.')
        training_seconds=previous['train_s']
    else:
        print(f'Training full {key}: 6941 images, same frozen 512D descriptor',flush=True)
        model=make_model(run,key)
        with threadpool_limits(limits=protocol['threads']):
            start=time.perf_counter(); model.fit(x['train'],y['train']); training_seconds=time.perf_counter()-start
        source=directory/'model.joblib'
        joblib.dump({'model':model,'classes':classes,'descriptor':protocol['descriptor'],'dimensions':512,
                     'source_run':run.name,'split_fingerprint':protocol['split_fingerprint']},source,compress=3)
        if key=='lightgbm': model.booster_.save_model(str(directory/'lightgbm.txt'))
        expected=model.predict_proba(x['query'][:3])
        del model
        gc.collect()
        bundle=joblib.load(source); model=bundle['model']
        np.testing.assert_allclose(model.predict_proba(x['query'][:3]),expected,rtol=1e-7,atol=1e-9)
    with threadpool_limits(limits=protocol['threads']):
        prediction=model.predict(x['query'])
        if reused:
            np.testing.assert_array_equal(prediction,np.load(parent/'models'/f'resnet18_{key}_predictions.npy'))
        if not np.array_equal(model.classes_,np.arange(101)): raise ValueError('Class mapping mismatch.')
        model.predict_proba(x['query'][:1])
        times=[]
        indices=np.linspace(0,1735,protocol['benchmark_samples'],dtype=int)
        for index in indices:
            start=time.perf_counter_ns(); probs=model.predict_proba(x['query'][index:index+1]); times.append((time.perf_counter_ns()-start)/1e6)
            if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1),1,atol=1e-5): raise ValueError('Invalid probabilities.')
    np.save(directory/'predictions.npy',prediction,allow_pickle=False)
    result={'key':key,'name':NAMES[key], 'descriptor':protocol['descriptor'],'dimensions':512,
        'n_train':6941,'n_test':1736,'n_classes':101,'test_role':'previously used test; descriptive comparison, NOT a new independent validation',
        'full_model_reused':reused,'fit_s':training_seconds,'fit_time_source':'parent experiment' if reused else 'this extension',
        'model_path':source.relative_to(ROOT).as_posix(),'model_sha256':sha256(source),
        'parameters':protocol['parameters'][key], 'persistence_verified':True,
        'inference_ms':float(np.mean(times)),'inference_p95_ms':float(np.percentile(times,95)),
        'inference_samples':len(times),'inference_indices':indices.tolist(),
        'inference_timing':'predict_proba for one 512D vector, warm-up excluded, extractor/IO/HTTP excluded',
        'cv_selection_sha256':sha256(run/'cv_selection.json'),
        **calculate_metrics(y['query'],prediction,include_confusion=True)}
    save_json(directory/'results.json',result)
    print(f"{key}: historical test accuracy={result['accuracy']:.5f}; balanced={result['balanced_accuracy']:.5f}; F1={result['f1_macro']:.5f}; inference={result['inference_ms']:.3f}ms; reused={reused}",flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True); parser.add_argument('--model',choices=KEYS,required=True)
    args=parser.parse_args(); evaluate(args.run,args.model)


if __name__=='__main__': main()
