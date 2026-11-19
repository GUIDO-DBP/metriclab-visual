"""Portable preflight and optional real inference; never trains or rewrites artifacts."""
import argparse
import json
import sys
from pathlib import Path
from scripts.resource_io import ROOT,CATALOG,read_json,sha256

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); parser.add_argument('--skip-resnet',action='store_true'); parser.add_argument('--require-engine',action='store_true')
    args=parser.parse_args(); checks=[]
    from backend.core import DATA,DESCRIPTOR,load_vectors
    from backend.descriptors import ResNetRegistry,RESNET_ID
    from backend.modeling import artifact_directory
    import numpy as np
    import joblib
    manifest=read_json(DATA/'manifest.json'); catalog=read_json(CATALOG)
    assert manifest['fingerprint']==catalog['split_fingerprint'],'La división no coincide con esta entrega.'
    assert len(manifest['train'])==6941 and len(manifest['query'])==1736
    classes=sorted({r['category'] for r in manifest['train']}); assert len(classes)==101
    from scripts.dataset_resources import validate
    validate(ROOT); checks.append('8677 image SHA256 / historical split')
    for split,count in [('train',6941),('query',1736)]:
        values=load_vectors(DATA/f'{split}.bin'); assert values.shape==(count,144) and np.isfinite(values).all()
    bundle=joblib.load(artifact_directory(DATA)/'models.joblib')
    assert bundle['classes']==classes and bundle['fingerprint']==manifest['fingerprint']
    for key in ['gini','entropy','forest','xgboost','lightgbm']:
        model=bundle['models'][key]; assert model.n_features_in_==144 and np.array_equal(model.classes_,np.arange(101))
    checks.append('5 HSV classifiers / 101 classes / 144 dimensions')
    if not args.skip_resnet:
        registry=ResNetRegistry(manifest)
        try:
            assert registry.available,registry.reason
            for key in ['gini','entropy','forest','xgboost','lightgbm']: registry.model(key)
            checks.append('5 ResNet classifiers / SHA256 / 101 classes / 512 dimensions')
        finally: registry.close()
    assert (ROOT/'frontend/dist/index.html').is_file(),'Falta compilar frontend: cd frontend; npm.cmd run build'
    if args.smoke:
        from fastapi.testclient import TestClient
        from backend.main import app
        with TestClient(app) as client:
            raw=(ROOT/manifest['query'][0]['path']).read_bytes()
            for descriptor in ([DESCRIPTOR] if args.skip_resnet else [DESCRIPTOR,RESNET_ID]):
                for key in ['gini','entropy','forest','xgboost','lightgbm']:
                    r=client.post('/api/classify',files={'file':('test.jpg',raw,'image/jpeg')},data={'descriptor_id':descriptor,'model':key})
                    assert r.status_code==200,r.text
                    pred=r.json()['predictions'][0]; assert len(pred['probabilities'])==101 and len(pred['examples'])==6
                    for example in pred['examples']:
                        assert example['split']=='train' and manifest['train'][example['id']]['category']==pred['category']
                        assert client.get(example['url']).status_code==200
                assert len(client.get('/api/experiments',params={'descriptor_id':descriptor}).json()['metrics'])==5
            if args.require_engine: assert client.get('/api/status').json()['metric_ready']
        checks.append('real HTTP inference / 10 configurations / images / metrics' if not args.skip_resnet else 'real HSV HTTP inference / images / metrics')
    result={'passed':True,'python':sys.version.split()[0],'checks':checks,'training_performed':False}
    target=ROOT/'runtime/installation-verification.json'; target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
