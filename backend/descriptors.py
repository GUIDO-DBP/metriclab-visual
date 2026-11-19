"""Lazy, versioned ResNet integration. HSV remains the default and independent."""
import base64
import hashlib
import json
import os
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np

from .core import ROOT, DESCRIPTOR as HSV_ID

RESNET_ID='torchvision-resnet18-imagenet1k-v1-avgpool512-fp32'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def local_path(relative):
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT/'runtime') or not path.is_file():
        raise ValueError('Artefacto fuera de runtime o no disponible.')
    return path


class ResNetRegistry:
    def __init__(self,manifest):
        self.available=False; self.reason='ResNet-18 pendiente de configuración.'
        self.config=None; self.cache=OrderedDict(); self.vectors=OrderedDict()
        self.lock=threading.RLock(); self.worker=None; self.worker_log=None; self.device='sin iniciar'
        try:
            if os.getenv('METRICLAB_ENABLE_RESNET','1')=='0':
                raise ValueError('ResNet-18 desactivada por configuración; HSV sigue disponible.')
            path=ROOT/os.getenv('METRICLAB_DESCRIPTOR_REGISTRY','runtime/descriptor_registry.json')
            self.config=json.loads(path.read_text(encoding='utf-8'))
            cfg=self.config
            if cfg['descriptor']!=RESNET_ID or cfg['dimensions']!=512 or cfg['split_fingerprint']!=manifest['fingerprint']:
                raise ValueError('Registro ResNet incompatible con el dataset.')
            if cfg['classes']!=sorted({r['category'] for r in manifest['train']}):
                raise ValueError('Codificación de clases ResNet incompatible.')
            self.python=Path(os.getenv('METRICLAB_RESNET_PYTHON',str(ROOT/'runtime/resnet-env/Scripts/python.exe')))
            if not self.python.is_file(): raise ValueError('Falta el entorno experimental de ResNet. Consulta README.md.')
            probe=subprocess.run([str(self.python),'-c',"import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision') else 1)"],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if probe.returncode: raise ValueError('PyTorch/torchvision no están instalados en el entorno ResNet. HSV sigue disponible.')
            checkpoint=local_path(cfg['checkpoint_path'])
            if digest(checkpoint)!=cfg['checkpoint_sha256']: raise ValueError('Pesos ResNet ausentes o con checksum distinto.')
            for entry in cfg['models'].values():
                if digest(local_path(entry['path']))!=entry['sha256']: raise ValueError('Checksum de modelo ResNet distinto.')
            self.available=True; self.reason=''
        except Exception as exc:
            self.reason=str(exc)

    def model(self,key):
        if not self.available: raise ValueError(self.reason)
        if key not in self.config['models']: raise ValueError('Modelo ResNet no disponible.')
        with self.lock:
            if key in self.cache:
                bundle=self.cache.pop(key); self.cache[key]=bundle
            else:
                # One ResNet classifier at a time, alongside the preserved HSV bundle.
                self.cache.clear()
                entry=self.config['models'][key]; path=local_path(entry['path'])
                if digest(path)!=entry['sha256']: raise ValueError('El archivo del modelo cambió desde el registro.')
                bundle=joblib.load(path)
                model=bundle['model']
                if model.n_features_in_!=512 or bundle['classes']!=self.config['classes'] or bundle.get('descriptor') not in {'resnet18',RESNET_ID}:
                    raise ValueError('Modelo/vector ResNet incompatible.')
                if not np.array_equal(model.classes_,np.arange(101)): raise ValueError('Etiquetas del modelo incompatibles.')
                expected={'gini':'DecisionTreeClassifier','entropy':'DecisionTreeClassifier','forest':'RandomForestClassifier','xgboost':'XGBClassifier','lightgbm':'LGBMClassifier'}[key]
                if type(model).__name__!=expected: raise ValueError('El algoritmo no coincide con el modelo registrado.')
                if key in {'gini','entropy'} and model.criterion!=('gini' if key=='gini' else 'entropy'):
                    raise ValueError('Criterio de árbol incompatible.')
                self.cache[key]=bundle
            return bundle['model'],bundle['classes']

    def _start(self):
        if self.worker and self.worker.poll() is None: return
        env=os.environ.copy(); env['PYTHONIOENCODING']='utf-8'; env['TORCH_HOME']=str(ROOT/'runtime/experiments/torch-cache')
        self.worker_log=(ROOT/'runtime/resnet_worker.log').open('a',encoding='utf-8')
        self.worker=subprocess.Popen([str(self.python),'-u','-m','backend.resnet_worker'],cwd=ROOT,env=env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.worker_log,text=True,encoding='utf-8',bufsize=1,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        ready=self.worker.stdout.readline()
        if not ready:
            self.available=False; self.reason='No se pudo iniciar ResNet. Consulta runtime/resnet_worker.log.'
            raise ValueError(self.reason)
        result=json.loads(ready)
        if not result.get('ready'):
            self.available=False; self.reason=result.get('error','Error al cargar ResNet.')
            raise ValueError(self.reason)
        self.device=result['device']

    def extract(self,raw):
        if not self.available: raise ValueError(self.reason)
        key=RESNET_ID+':'+hashlib.sha256(raw).hexdigest()
        with self.lock:
            if key in self.vectors:
                value=self.vectors.pop(key); self.vectors[key]=value
                return value.copy(),{'device':self.device,'cache_hit':True}
            self._start()
            self.worker.stdin.write(json.dumps({'image':base64.b64encode(raw).decode('ascii')})+'\n'); self.worker.stdin.flush()
            line=self.worker.stdout.readline()
            if not line: raise ValueError('El proceso ResNet terminó durante la extracción.')
            result=json.loads(line)
            if 'error' in result: raise ValueError(result['error'])
            value=np.asarray(result['vector'],dtype=np.float32)
            if value.shape!=(512,) or not np.isfinite(value).all(): raise ValueError('Vector ResNet inválido.')
            self.vectors[key]=value
            while len(self.vectors)>4: self.vectors.popitem(last=False)
            return value.copy(),{'device':result['device'],'cache_hit':False}

    def experiments(self,evaluation):
        if not self.available: raise ValueError(self.reason)
        if evaluation=='training_cv':
            original=json.loads(local_path(self.config['comparison_results']).read_text(encoding='utf-8'))
            tuning=json.loads(local_path(self.config['tuning_protocol']).read_text(encoding='utf-8'))
            confirmation=json.loads(local_path(self.config['tuning_confirmation']).read_text(encoding='utf-8'))
            selection=json.loads(local_path(self.config['tuning_selection']).read_text(encoding='utf-8'))
            return {'metrics':[],'descriptor_id':RESNET_ID,'evaluation_scope':'training_cv',
                'cv_metrics':original['cv_training_only'],'internal_confirmation':confirmation,'tuning_selection':selection,
                'protocol':'CV de 5 folds original, solo entrenamiento. LightGBM ajustado: búsqueda de 3 folds en desarrollo y confirmación interna reservada; no mezclar estos alcances.',
                'n_train':6941,'n_test':len(tuning['internal_confirmation']),'n_classes':101,'run_id':self.config['run_id']}
        classes=self.config['classes']; metrics=[]
        for key in ['gini','entropy','forest','xgboost','lightgbm']:
            row=json.loads(local_path(self.config['models'][key]['metrics_path']).read_text(encoding='utf-8'))
            matrix=np.asarray(row['confusion'],dtype=float); diag=np.diag(matrix)
            precision=np.divide(diag,matrix.sum(axis=0),out=np.zeros(101),where=matrix.sum(axis=0)>0)
            recall=np.divide(diag,matrix.sum(axis=1),out=np.zeros(101),where=matrix.sum(axis=1)>0)
            metrics.append({**row,'model':self.config['models'][key]['name'],'train_s':row.get('fit_s'),
                'precision_macro':float(precision.mean()),'recall_macro':float(recall.mean()),
                'reused':row.get('full_model_reused',False)})
        return {'metrics':metrics,'classes':classes,'descriptor_id':RESNET_ID,'dimensions':512,
            'evaluation_scope':'historical_test','n_train':6941,'n_test':1736,'n_classes':101,
            'fingerprint':self.config['split_fingerprint'],'run_id':self.config['run_id'],
            'protocol':'Prueba histórica reutilizada: NO es una evaluación independiente nueva. Modelos completos por descriptor; LightGBM elegido solo con validación de entrenamiento.',
            'timing_protocol':'Inferencia predict_proba, excluye extracción CNN, IO y HTTP; tiempos de diferentes ejecuciones versionadas.',
            'versions':self.config['versions']}

    def close(self):
        with self.lock:
            self.cache.clear(); self.vectors.clear()
            if self.worker and self.worker.poll() is None:
                try:
                    self.worker.stdin.write('{"close":true}\n'); self.worker.stdin.flush(); self.worker.wait(timeout=5)
                except Exception: self.worker.terminate(); self.worker.wait(timeout=5)
            if self.worker:
                for pipe in (self.worker.stdin,self.worker.stdout):
                    if pipe: pipe.close()
                self.worker=None
            if self.worker_log: self.worker_log.close()
