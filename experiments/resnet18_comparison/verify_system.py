"""Regression checks through the unchanged app environment, not the ML workers."""
import argparse
import os
import subprocess
import time
from pathlib import Path

from experiments.resnet18.common import ROOT, assert_protected, load_json, save_json
from .core import assert_parent


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run.resolve()
    if not (run/'completion.json').exists(): raise ValueError('Complete model/CV work first.')
    env=os.environ.copy(); env['PYTHONIOENCODING']='utf-8'
    commands=[
        ('backend_integration',[str(ROOT/'.venv/Scripts/python.exe'),'-m','unittest','discover','-s','tests','-v'],ROOT),
        ('cpp_oracle',[str(ROOT/'runtime/oracle_test.exe')],ROOT),
        ('frontend_build',[os.environ.get('COMSPEC',r'C:\Windows\System32\cmd.exe'),'/d','/c','npm.cmd','run','build'],ROOT/'frontend'),
    ]
    records=[]
    for name,command,cwd in commands:
        start=time.perf_counter()
        result=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (run/f'verify_{name}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        records.append({'name':name,'command':command,'exit_code':result.returncode,'elapsed_s':time.perf_counter()-start})
        print(name,'exit',result.returncode,flush=True)
        if result.returncode: raise RuntimeError(f'Regression check failed: {name}; inspect its log.')
    # Uses the current frontend and image endpoint, without starting/replacing a user server.
    script='''
import json,sys
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app
proposal=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
item=next(iter(proposal['categories'].values()))[0]
with TestClient(app) as client:
    status=client.get('/api/status').json()
    assert status['classification_ready'] and status['metric_ready']
    assert client.get('/').status_code==200
    photo=client.get(item['url'])
    assert photo.status_code==200 and photo.headers['content-type'].startswith('image/') and len(photo.content)>100
    print(json.dumps({'frontend_http':200,'classifiers_ready':True,'metric_engine_ready':True,'real_training_image_url':item['url'],'category':item['category'],'image_bytes':len(photo.content)}))
'''
    result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-c',script,str(run/'gallery_examples_proposal.json')],
                          cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (run/'verify_frontend_and_image_api.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode: raise RuntimeError('Frontend/image API regression check failed.')
    records.append({'name':'frontend_and_real_image_api','exit_code':result.returncode,'result':result.stdout.strip()})
    count=assert_protected(load_json(run/'protected_files.json'))
    parent_count=assert_parent(load_json(run/'parent_files.json'))
    save_json(run/'system_verification.json',{'checks':records,'application_files_unchanged':count,
        'parent_files_unchanged':parent_count,'all_checks_passed':True,'application_changed':False})
    print('All existing-system checks passed; files/models unchanged.',flush=True)


if __name__=='__main__': main()
