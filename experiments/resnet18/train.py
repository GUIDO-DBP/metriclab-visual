"""One process/model to bound RAM; fixed historical params, same saved split."""
import argparse
import time
from pathlib import Path
from importlib.metadata import version

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from threadpoolctl import threadpool_limits

from .common import BASE_DATA, load_bin, load_json, save_json, sha256, validate_embeddings


def train(run, descriptor, key, threads, benchmark_samples):
    run = Path(run)
    slug = f'{descriptor}_{key}'
    destination = run / 'models'
    destination.mkdir(exist_ok=True)
    result_path = destination / f'{slug}_results.json'
    if result_path.exists():
        raise ValueError('Model results already exist; never overwrite an earlier model.')
    validate_embeddings(run)
    classes = load_json(run / 'classes.json')
    labels = {s: np.load(run / f'{s}_labels.npy') for s in ['train','query']}
    x = {s: (np.load(run / f'{s}_resnet18.npy',mmap_mode='r') if descriptor == 'resnet18' else load_bin(BASE_DATA / f'{s}.bin')) for s in ['train','query']}
    params = dict(load_json(run / 'baseline.json')['models'][key].get('training_params') or load_json(run / 'baseline.json')['models'][key]['params'])
    params['n_jobs'] = threads
    if key == 'forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
    else:
        from xgboost import XGBClassifier
        if isinstance(params.get('missing'),str) and params['missing'].lower() == 'nan': params['missing'] = np.nan
        model = XGBClassifier(**params)
    print(f'Training {slug}: {x["train"].shape}, {threads} CPU threads; no tuning.',flush=True)
    with threadpool_limits(limits=threads):
        start = time.perf_counter()
        model.fit(x['train'],labels['train'])
        train_s = time.perf_counter()-start
        predicted = model.predict(x['query'])
        model.predict_proba(x['query'][:1])  # warm-up
        timings = []
        indices = np.linspace(0,len(labels['query'])-1,min(benchmark_samples,len(labels['query'])),dtype=int)
        for index in indices:
            start = time.perf_counter_ns()
            probability = model.predict_proba(x['query'][index:index+1])
            timings.append((time.perf_counter_ns()-start)/1e6)
            if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1),1,atol=1e-5):
                raise ValueError('Invalid probabilities.')
    if not np.array_equal(model.classes_,np.arange(len(classes))):
        raise ValueError('Model class encoding does not match saved labels.')
    np.save(destination / f'{slug}_predictions.npy',predicted,allow_pickle=False)
    model_path = destination / f'{slug}.joblib'
    joblib.dump({'model':model,'classes':classes,'descriptor':descriptor,'dimensions':x['train'].shape[1]},model_path,compress=3)
    if key == 'xgboost': model.save_model(destination / f'{slug}.json')
    expected_probs = model.predict_proba(x['query'][:3])
    # Drop the large forest before reloading it for persistence verification.
    import gc
    del model
    gc.collect()
    restored = joblib.load(model_path)
    np.testing.assert_allclose(restored['model'].predict_proba(x['query'][:3]),expected_probs,rtol=1e-7,atol=1e-9)
    metadata_params = {k: (str(v) if isinstance(v,float) and not np.isfinite(v) else v) for k,v in params.items()}
    result = {'descriptor':descriptor,'key':key,'dimensions':int(x['train'].shape[1]),
        'n_train':len(labels['train']),'n_test':len(labels['query']),'n_classes':len(classes),
        'accuracy':float(accuracy_score(labels['query'],predicted)),
        'balanced_accuracy':float(balanced_accuracy_score(labels['query'],predicted)),
        'precision_macro':float(precision_score(labels['query'],predicted,average='macro',zero_division=0)),
        'recall_macro':float(recall_score(labels['query'],predicted,average='macro',zero_division=0)),
        'f1_macro':float(f1_score(labels['query'],predicted,average='macro',zero_division=0)),
        'train_s':train_s,'inference_ms':float(np.mean(timings)),'inference_p95_ms':float(np.percentile(timings,95)),
        'inference_samples':len(timings),'inference_indices':indices.tolist(),'inference_operation':'single-image predict_proba, warm-up excluded; extractor/IO excluded',
        'params':metadata_params,'threads':threads,'test_used_for_fit':False,'hyperparameter_search':False,
        'versions':{p:version(p) for p in ['scikit-learn','xgboost','numpy','joblib']},
        'confusion':confusion_matrix(labels['query'],predicted,labels=np.arange(len(classes))).tolist(),
        'model_sha256':sha256(model_path),'persistence_verified':True}
    if descriptor == 'hsv':
        historical = next(r for r in load_json(run / 'baseline.json')['metrics'] if r['key'] == key)
        result['historical_accuracy'] = historical['accuracy']
        result['historical_accuracy_delta_pp'] = 100*(result['accuracy']-historical['accuracy'])
    save_json(result_path,result)
    print(f"{slug}: accuracy={result['accuracy']:.6f}; balanced={result['balanced_accuracy']:.6f}; F1={result['f1_macro']:.6f}; fit={train_s:.3f}s; infer={result['inference_ms']:.3f}ms",flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--descriptor',choices=['hsv','resnet18'],required=True)
    parser.add_argument('--model',choices=['forest','xgboost'],required=True)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--benchmark-samples',type=int,default=128)
    args = parser.parse_args()
    if min(args.threads,args.benchmark_samples) < 1: parser.error('Limits must be positive.')
    train(args.run,args.descriptor,args.model,args.threads,args.benchmark_samples)


if __name__ == '__main__': main()
