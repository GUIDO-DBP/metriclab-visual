import argparse
import contextlib
import gc
import io
import time
import uuid
import warnings
from collections import Counter
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,log_loss
from sklearn.model_selection import StratifiedKFold,train_test_split
from threadpoolctl import threadpool_limits

from experiments.resnet18.common import ROOT,DESCRIPTOR,load_json,save_json,sha256,validate_embeddings
from experiments.resnet18_comparison.core import calculate_metrics

OUTPUT=ROOT/'runtime/experiments/lightgbm_resnet'


def fit_logged(params,x,y,v,z,path):
    curves={}; stream=io.StringIO()
    model=lgb.LGBMClassifier(**{**params,'verbosity':1})
    with warnings.catch_warnings(record=True) as captured,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream),threadpool_limits(limits=2):
        warnings.simplefilter('always')
        start=time.perf_counter()
        model.fit(x,y,eval_set=[(x,y),(v,z)],eval_names=['fit','validation'],
                  eval_metric=['multi_logloss','multi_error'],callbacks=[lgb.record_evaluation(curves)])
        seconds=time.perf_counter()-start
    path.with_suffix('.log').write_text(stream.getvalue(),encoding='utf-8')
    info={'curves':curves,'python_warnings':[str(w.message) for w in captured],
          'no_positive_gain_messages':stream.getvalue().count('No further splits with positive gain'),
          'fit_s':seconds}
    save_json(path.with_suffix('.json'),info)
    return model,info


def distribution(model,x,y,classes):
    probabilities=model.predict_proba(x); pred=model.classes_[np.argmax(probabilities,axis=1)]
    margins=model.predict(x,raw_score=True)
    return {**calculate_metrics(y,pred), 'predicted_class_counts':dict(Counter(classes[int(i)] for i in pred)),
            'true_class_counts':dict(Counter(classes[int(i)] for i in y)),
            'predicted_classes':int(len(np.unique(pred))), 'max_abs_raw_margin':float(np.max(np.abs(margins))),
            'mean_max_probability':float(np.mean(probabilities.max(axis=1))),
            'finite_probabilities':bool(np.isfinite(probabilities).all()),
            'probability_sum_max_error':float(np.max(np.abs(probabilities.sum(axis=1)-1))),
            'per_category':classification_report(y,pred,labels=np.arange(101),target_names=classes,output_dict=True,zero_division=0)}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--comparison',type=Path,required=True)
    args=parser.parse_args(); comparison=args.comparison.resolve()
    old=load_json(comparison/'protocol.json'); parent=ROOT/old['parent_run']
    validate_embeddings(parent,rehash_images=True)
    run=OUTPUT/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]); run.mkdir(parents=True)
    base=dict(old['parameters']['lightgbm'])
    # Fixed configurations. Query integrity was checked above; its values/labels
    # are not used for fitting, diagnosis, candidate evaluation or selection.
    changes=[{},
        {'reg_lambda':1.,'min_child_weight':.1,'learning_rate':.05,'n_estimators':120},
        {'reg_lambda':5.,'reg_alpha':.1,'min_child_weight':1.,'learning_rate':.1,'n_estimators':120},
        {'num_leaves':31,'max_depth':8,'min_child_samples':20,'learning_rate':.05,'n_estimators':160,'reg_lambda':5.,'reg_alpha':.1,'min_child_weight':1.},
        {'num_leaves':7,'max_depth':3,'min_child_samples':40,'learning_rate':.1,'n_estimators':120,'reg_lambda':5.,'reg_alpha':.1,'min_child_weight':1.},
        {'num_leaves':31,'max_depth':-1,'min_child_samples':10,'learning_rate':.05,'n_estimators':160,'reg_lambda':1.,'min_child_weight':.1},
        {'num_leaves':15,'max_depth':6,'min_child_samples':40,'learning_rate':.03,'n_estimators':240,'reg_lambda':10.,'reg_alpha':.5,'min_child_weight':1.}]
    candidates=[{**base,**c} for c in changes]
    y=np.load(parent/'train_labels.npy'); x=np.load(parent/'train_resnet18.npy',mmap_mode='r'); classes=load_json(parent/'classes.json')
    development,confirmation=train_test_split(np.arange(6941),test_size=.2,stratify=y,random_state=41)
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=29)
    folds=[{'fit':development[a].tolist(),'validation':development[b].tolist()} for a,b in cv.split(development,y[development])]
    save_json(run/'protocol.json',{'run_id':run.name,'parent_run':parent.relative_to(ROOT).as_posix(),
        'comparison_run':comparison.relative_to(ROOT).as_posix(),'descriptor':DESCRIPTOR,'split_fingerprint':old['split_fingerprint'],
        'candidates':candidates,'development':development.tolist(),'internal_confirmation':confirmation.tolist(),'folds':folds,
        'selection':'highest mean development CV F1 macro; tie by balanced accuracy','model_seed':0,
        'test_for_selection':False,'confirmation_for_selection':False,'versions':{p:version(p) for p in ['lightgbm','scikit-learn','numpy']}})
    print('LightGBM experiment:',run,flush=True)
    # Diagnosis on a development-only fold. Confirmation and historical test remain untouched.
    a,b=np.asarray(folds[0]['fit']),np.asarray(folds[0]['validation'])
    original,info=fit_logged(base,x[a],y[a],x[b],y[b],run/'diagnostic_curves')
    checkpoints=[]
    for n in [1,5,10,20,40,60,80]:
        p=original.predict_proba(x[b],num_iteration=n)
        margins=original.predict(x[b],num_iteration=n,raw_score=True)
        checkpoints.append({'iteration':n,**calculate_metrics(y[b],np.argmax(p,axis=1)),
            'log_loss':float(log_loss(y[b],p,labels=np.arange(101))), 'max_abs_margin':float(np.abs(margins).max())})
    save_json(run/'diagnosis.json',{'input_shape':list(x.shape),'classes':classes,'label_range':[int(y.min()),int(y.max())],
        'model_classes':original.classes_.tolist(),'n_features_in':int(original.n_features_in_),
        'training':distribution(original,x[a],y[a],classes),'development_validation':distribution(original,x[b],y[b],classes),
        'checkpoints':checkpoints,'warnings':info['python_warnings'],'test_opened':False})
    del original; gc.collect()
    rows=[]
    for i,params in enumerate(candidates):
        scores=[]
        for k,fold in enumerate(folds):
            a,b=np.asarray(fold['fit']),np.asarray(fold['validation'])
            model,info=fit_logged(params,x[a],y[a],x[b],y[b],run/f'candidate_{i}_fold_{k}')
            pred=model.predict(x[b]); metric=calculate_metrics(y[b],pred); scores.append(metric)
            np.save(run/f'candidate_{i}_fold_{k}_predictions.npy',pred,allow_pickle=False)
            del model; gc.collect()
        row={'candidate':i,'parameters':params,'folds':scores,
             'mean':{m:float(np.mean([s[m] for s in scores])) for m in scores[0]},
             'std':{m:float(np.std([s[m] for s in scores],ddof=1)) for m in scores[0]}}
        rows.append(row); save_json(run/'search.json',rows)
        print(f"Candidate {i}: CV F1={row['mean']['f1_macro']:.4f}, balanced={row['mean']['balanced_accuracy']:.4f}",flush=True)
    winner=sorted(rows,key=lambda r:(-r['mean']['f1_macro'],-r['mean']['balanced_accuracy'],r['candidate']))[0]
    save_json(run/'selection.json',{'candidate':winner['candidate'],'parameters':winner['parameters'],
        'cv_mean':winner['mean'],'test_used':False,'internal_confirmation_used':False,'frozen_at':datetime.now(timezone.utc).isoformat()})
    # One independent WITHIN-training confirmation; never used to choose a candidate.
    confirmations={}
    for name,params in [('original',base),('selected',winner['parameters'])]:
        model,info=fit_logged(params,x[development],y[development],x[confirmation],y[confirmation],run/f'confirmation_{name}_curves')
        confirmations[name]=distribution(model,x[confirmation],y[confirmation],classes)
        np.save(run/f'confirmation_{name}_predictions.npy',model.predict(x[confirmation]),allow_pickle=False)
        del model; gc.collect()
    save_json(run/'internal_confirmation.json',confirmations)
    improved=confirmations['selected']['f1_macro']>confirmations['original']['f1_macro']
    # The fixed winner is fitted once on ALL original training rows; no test-based changes.
    selected=lgb.LGBMClassifier(**winner['parameters'])
    start=time.perf_counter(); selected.fit(x,y); fit_s=time.perf_counter()-start
    model_path=run/'model.joblib'
    joblib.dump({'model':selected,'classes':classes,'descriptor':DESCRIPTOR,'dimensions':512,
                 'split_fingerprint':old['split_fingerprint'],'source_run':run.name},model_path,compress=3)
    selected.booster_.save_model(str(run/'lightgbm.txt'))
    # Historical test used once at the end for DESCRIPTIVE comparison only.
    q=np.load(parent/'query_resnet18.npy',mmap_mode='r'); truth=np.load(parent/'query_labels.npy')
    pred=selected.predict(q); np.save(run/'historical_predictions.npy',pred,allow_pickle=False)
    result={**distribution(selected,q,truth,classes),'fit_s':fit_s,
        'confusion':confusion_matrix(truth,pred,labels=np.arange(101)).tolist(),'test_role':'historical test reused, NOT new validation',
        'key':'lightgbm','model_path':model_path.relative_to(ROOT).as_posix(),'model_sha256':sha256(model_path),
        'parameters':winner['parameters'],'improved_in_training_confirmation':improved}
    selected.predict_proba(q[:1]); times=[]
    for n in np.linspace(0,1735,128,dtype=int):
        start=time.perf_counter_ns(); selected.predict_proba(q[n:n+1]); times.append((time.perf_counter_ns()-start)/1e6)
    result['inference_ms']=float(np.mean(times)); result['inference_samples']=128
    reference=selected.predict_proba(q[:3]); del selected; gc.collect()
    restored=joblib.load(model_path); np.testing.assert_allclose(restored['model'].predict_proba(q[:3]),reference,atol=1e-10)
    save_json(run/'results.json',result)
    print('LightGBM complete:',{k:result[k] for k in ['accuracy','balanced_accuracy','f1_macro','improved_in_training_confirmation']},flush=True)
    plot(run)


def plot(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,color in [('original','#b56677'),('selected','#5d8e84')]:
        curves=load_json(run/f'confirmation_{name}_curves.json')['curves']
        for axis,metric in zip(axes,['multi_logloss','multi_error']):
            axis.plot(curves['fit'][metric],color=color,linestyle='--',label=name+' fit')
            axis.plot(curves['validation'][metric],color=color,label=name+' confirmation')
            axis.set_title(metric); axis.set_xlabel('Boosting iteration'); axis.grid(alpha=.15)
    axes[0].set_yscale('symlog'); axes[0].legend(fontsize=8)
    fig.savefig(run/'training_curves.png',dpi=170); plt.close(fig)


if __name__=='__main__': main()
