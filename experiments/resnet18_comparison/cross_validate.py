"""Only train_resnet18/train_labels are opened by this fitting worker."""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.resnet18.common import load_json, save_json, sha256
from .core import KEYS, calculate_metrics, make_model, parent_directory, validate_folds


def cross_validate(run,key):
    run = Path(run)
    parent = parent_directory(run)
    folds = validate_folds(run)
    protocol = load_json(run/'protocol.json')
    x = np.load(parent/'train_resnet18.npy',mmap_mode='r')
    y = np.load(parent/'train_labels.npy')
    directory = run/'cv'/key
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.json').exists():
        raise ValueError('Complete CV results already exist; never overwrite.')
    out_of_fold = np.full(len(y),-1,dtype=np.int64)
    records = []
    for fold in folds['folds']:
        number = fold['fold']
        target = directory/f'fold_{number}.json'
        fit = np.asarray(fold['fit'],dtype=int)
        validation = np.asarray(fold['validation'],dtype=int)
        if target.exists():
            record = load_json(target)
            if record['folds_sha256'] != sha256(run/'folds.json'):
                raise ValueError('Checkpoint belongs to other folds.')
            prediction = np.load(directory/f'fold_{number}_predictions.npy')
        else:
            print(f'{key}: fold {number}/5; fit={len(fit)}, validation={len(validation)}; training partition only',flush=True)
            model = make_model(run,key)  # NEW estimator; never use the already-fit full model.
            with threadpool_limits(limits=protocol['threads']):
                start = time.perf_counter()
                model.fit(x[fit],y[fit])
                seconds = time.perf_counter()-start
                prediction = model.predict(x[validation])
            record = {'fold':number,'key':key,'n_fit':len(fit),'n_validation':len(validation),
                'fit_s':seconds, 'folds_sha256':sha256(run/'folds.json'),
                'parameters_sha256':sha256(run/'protocol.json'),
                'input_split':'train','historical_test_opened':False,
                **calculate_metrics(y[validation],prediction)}
            np.save(directory/f'fold_{number}_predictions.npy',prediction,allow_pickle=False)
            save_json(target,record)
            del model
            gc.collect()
        out_of_fold[validation] = prediction
        records.append(record)
        print(f"{key} fold {number}: accuracy={record['accuracy']:.5f}; balanced={record['balanced_accuracy']:.5f}; F1={record['f1_macro']:.5f}; fit={record['fit_s']:.2f}s",flush=True)
    if (out_of_fold<0).any(): raise ValueError('Incomplete out-of-fold predictions.')
    np.save(directory/'oof_predictions.npy',out_of_fold,allow_pickle=False)
    summary = {'key':key,'n_folds':5,'universe':'original training only','n_images':len(y),
        'parameters':protocol['parameters'][key], 'folds_sha256':sha256(run/'folds.json'),
        'folds':records,'oof':calculate_metrics(y,out_of_fold,include_confusion=True),
        'mean':{m:float(np.mean([r[m] for r in records])) for m in ['accuracy','balanced_accuracy','f1_macro']},
        'std':{m:float(np.std([r[m] for r in records],ddof=1)) for m in ['accuracy','balanced_accuracy','f1_macro']},
        'fit_s_total':sum(r['fit_s'] for r in records),'historical_test_used':False,
        'dispersion_note':'Sample SD across five overlapping-training folds; not a confidence interval.'}
    save_json(directory/'summary.json',summary)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--model',choices=KEYS,required=True)
    args=parser.parse_args(); cross_validate(args.run,args.model)


if __name__=='__main__': main()
