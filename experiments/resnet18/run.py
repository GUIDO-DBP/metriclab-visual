"""Offline orchestrator: no activation, no app dependencies or files changed."""
import argparse
import os
import subprocess
import sys
import uuid
from datetime import datetime,timezone
from pathlib import Path

from .common import (BASE_DATA, OUTPUT_ROOT, ROOT, assert_protected, load_json,
                     protected_files, save_json, validate_embeddings, validate_original)


def worker(module,*args,python=None):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['TORCH_HOME'] = str(ROOT / 'runtime/experiments/torch-cache')
    with (CURRENT_RUN / 'execution.log').open('a',encoding='utf-8') as log:
        command = [str(python or sys.executable),'-m',module,*map(str,args)]
        print('Running:',module,' '.join(map(str,args)),flush=True)
        with subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1) as process:
            for line in process.stdout:
                log.write(line); log.flush(); print(line,end='',flush=True)
            if process.wait(): raise RuntimeError(f'{module} failed; see execution.log. Existing app unchanged.')


def main():
    global CURRENT_RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,help='Resume an existing independent experiment directory.')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--benchmark-samples',type=int,default=128)
    args = parser.parse_args()
    if min(args.batch_size,args.threads,args.benchmark_samples)<1: parser.error('Limits must be positive.')
    manifest,_,classes = validate_original()
    if args.run:
        run = args.run.resolve()
        if not run.is_relative_to(OUTPUT_ROOT.resolve()) or not (run/'split.json').is_file():
            raise ValueError('Resume directory must be an existing independent ResNet experiment.')
        if load_json(run/'split.json') != manifest: raise ValueError('The original split has changed.')
        protocol = load_json(run/'protocol.json')
        for key in ['batch_size','threads','benchmark_samples']:
            if getattr(args,key) != protocol[key]:
                raise ValueError(f'Resume must preserve original {key}={protocol[key]}.')
        snapshot = load_json(run/'protected_files.json')
        assert_protected(snapshot)
    else:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        run = OUTPUT_ROOT/run_id
        run.mkdir(parents=True,exist_ok=False)
        snapshot = protected_files()
        save_json(run/'protected_files.json',snapshot)
        save_json(run/'split.json',manifest)
        save_json(run/'classes.json',classes)
        pointer = load_json(BASE_DATA/'active_classifiers.json')
        source = (BASE_DATA/pointer['directory']).resolve()
        save_json(run/'baseline.json',load_json(source/'experiments.json'))
        save_json(run/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),
            'device':args.device,'batch_size':args.batch_size,'threads':args.threads,'benchmark_samples':args.benchmark_samples,
            'source_manifest_fingerprint':manifest['fingerprint'],'original_active_run':pointer,
            'models':['forest','xgboost'],'weights':'ResNet18_Weights.IMAGENET1K_V1',
            'parameters':'Historical RF/XGBoost training parameters copied before evaluation; n_jobs limited equally.',
            'fine_tuning':False,'test_used_for_hyperparameters':False,'application_activation':False})
    CURRENT_RUN = run
    print('Experiment directory:',run,flush=True)
    if not (run/'extraction.json').exists():
        worker('experiments.resnet18.extract','--run',run,'--device',args.device,'--batch-size',args.batch_size,'--threads',args.threads,'--benchmark-samples',args.benchmark_samples)
    checks = validate_embeddings(run,rehash_images=True)
    save_json(run/'alignment_checks.json',checks)
    # Run each model in a separate process, releasing memory before the next.
    for descriptor in ['resnet18','hsv']:
        for key in ['forest','xgboost']:
            if not (run/'models'/f'{descriptor}_{key}_results.json').exists():
                worker('experiments.resnet18.train','--run',run,'--descriptor',descriptor,'--model',key,'--threads',args.threads,'--benchmark-samples',args.benchmark_samples)
    replay = load_json(run/'models/hsv_xgboost_results.json')
    if abs(replay['historical_accuracy_delta_pp']) > 1e-9 and not (run/'historical_reproducibility_audit.json').exists():
        worker('experiments.resnet18.audit_reference','--run',run)
    if not (run/'hsv_extraction_latency.json').exists():
        worker('experiments.resnet18.hsv_latency','--run',run,'--benchmark-samples',args.benchmark_samples,
               python=ROOT/'.venv/Scripts/python.exe')
    worker('experiments.resnet18.report','--run',run)
    count = assert_protected(snapshot)
    save_json(run/'completion.json',{'complete':True,'protected_files_unchanged':count,
        'original_split_unchanged':True,'models_activated':False,'frontend_changed':False,
        'finished_at':datetime.now(timezone.utc).isoformat()})
    print(f'Experiment complete; {count} main application/source/artifact files unchanged.',flush=True)


if __name__ == '__main__': main()
