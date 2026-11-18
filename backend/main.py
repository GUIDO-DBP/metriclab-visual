import json
from collections import Counter
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .core import DATA, ROOT, DESCRIPTOR, NAMES, Engine, describe_bytes, load_vectors

MODEL_NAMES = {'gini': 'Decision Tree — Gini', 'entropy': 'Decision Tree — Entropía',
               'forest': 'Random Forest', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'knn': 'k-NN / VP-Tree'}
state = {'manifest': None, 'engine': None, 'bundle': None, 'error': None}


@asynccontextmanager
async def lifespan(app):
    try:
        if (DATA / 'manifest.json').exists():
            manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
            if manifest['descriptor'] != DESCRIPTOR:
                raise ValueError('Descriptor incompatible; vuelve a preparar los datos.')
            train = load_vectors(DATA / 'train.bin')
            if train.shape != (len(manifest['train']), 144) or not np.isfinite(train).all():
                raise ValueError('Vectores incompatibles con el manifiesto.')
            state['manifest'] = manifest
            state['engine'] = Engine(train)
            # Load only the locally generated trusted model bundle.
            if (DATA / 'models.joblib').exists():
                bundle = joblib.load(DATA / 'models.joblib')
                if bundle['fingerprint'] == manifest['fingerprint']:
                    state['bundle'] = bundle
    except Exception as exc:
        state['error'] = str(exc)
    yield
    if state['engine']:
        state['engine'].close()
    state.update(manifest=None, engine=None, bundle=None, error=None)


app = FastAPI(title='MetricLab Visual', lifespan=lifespan)


def ready():
    if state['engine'] is None:
        raise HTTPException(503, 'Índice no construido. Ejecuta scripts/build-engine.ps1 y python -m scripts.prepare --train; reinicia el backend. ' + (state['error'] or ''))
    return state['manifest'], state['engine']


@app.get('/api/status')
def status():
    manifest = state['manifest']
    bundle = state['bundle'] or {}
    return {'ready': state['engine'] is not None, 'error': state['error'],
            'dataset': manifest['dataset'] if manifest else 'Sin dataset preparado',
            'kind': manifest['kind'] if manifest else None,
            'indexed': len(manifest['train']) if manifest else 0,
            'queries': len(manifest['query']) if manifest else 0,
            'categories': sorted({r['category'] for r in manifest['train']}) if manifest else [],
            'dimensions': 144, 'descriptor': DESCRIPTOR,
            'models': [{'key': key, 'name': name,
                        'available': bool(state['engine']) if key == 'knn' else key in bundle.get('models', {}),
                        'reason': bundle.get('unavailable', {}).get(key, 'Modelo pendiente: ejecuta python -m scripts.prepare --train.')} for key, name in MODEL_NAMES.items()]}


@app.get('/api/dataset')
def dataset():
    manifest, _ = ready()
    samples = list({r['category']: r for r in reversed(manifest['query'])}.values())
    samples.sort(key=lambda r: r['category'])
    return {**status(), 'seed': manifest['seed'], 'query_fraction': manifest['query_fraction'],
            'max_per_class': manifest['max_per_class'], 'skipped': len(manifest['skipped']),
            'samples': [{**r, 'url': f"/api/images/query/{r['id']}"} for r in samples[:20]],
            'counts': dict(Counter(r['category'] for r in manifest['train']))}


@app.get('/api/images/{split}/{image_id}')
def image(split: str, image_id: int):
    manifest, _ = ready()
    if split not in {'train', 'query'} or not 0 <= image_id < len(manifest[split]):
        raise HTTPException(404, 'Imagen inexistente.')
    path = (ROOT / manifest[split][image_id]['path']).resolve()
    if not path.is_relative_to(ROOT / 'dataset') and not path.is_relative_to(ROOT / 'runtime'):
        raise HTTPException(403, 'La imagen debe estar en dataset/ o runtime/.')
    if not path.is_file():
        raise HTTPException(404, 'El archivo de imagen no está disponible.')
    return FileResponse(path)


def enrich(result, manifest):
    for run in result['runs']:
        run['name'] = NAMES[run['algorithm']]
        for item in run['neighbors']:
            row = manifest['train'][item['id']]
            item.update(category=row['category'], url=f"/api/images/train/{item['id']}")
    return result


async def vector_from(file):
    raw = await file.read(10*1024*1024+1)
    if len(raw) > 10*1024*1024:
        raise HTTPException(413, 'La imagen supera el límite de 10 MB.')
    try:
        return await run_in_threadpool(describe_bytes, raw)
    except Exception as exc:
        raise HTTPException(422, f'Error de procesamiento: {exc}') from exc


async def search_impl(file, k, algorithm):
    manifest, engine = ready()
    if algorithm not in {-1, 0, 1, 2} or not 1 <= k <= 20:
        raise HTTPException(422, 'Algoritmo o k inválido (1–20).')
    vector = await vector_from(file)
    result = await run_in_threadpool(engine.search, vector, min(k, len(manifest['train'])), algorithm)
    return {**enrich(result, manifest), 'descriptor': vector.tolist(), 'dataset': manifest['dataset'], 'kind': manifest['kind']}


@app.post('/api/search')
async def search(file: UploadFile = File(...), k: int = Form(10), algorithm: int = Form(1)):
    return await search_impl(file, k, algorithm)


@app.post('/api/compare')
async def compare(file: UploadFile = File(...), k: int = Form(10)):
    return await search_impl(file, k, -1)


@app.get('/api/tree')
def tree():
    _, engine = ready()
    return engine.structure


def predict(vector, key, k):
    manifest, engine = ready()
    if key == 'knn':
        result = enrich(engine.search(vector, min(k, len(manifest['train'])), 1), manifest)
        neighbors = result['runs'][0]['neighbors']
        labels = [n['category'] for n in neighbors]
        votes = Counter(labels)
        winner = max(labels, key=lambda label: votes[label])
        return {'key': key, 'name': MODEL_NAMES[key], 'category': winner, 'neighbors': neighbors,
                'probability_label': 'Proporción de votos (no probabilidad calibrada)',
                'probabilities': [{'category': label, 'value': n/len(labels)} for label, n in votes.most_common()]}
    bundle = state['bundle']
    if not bundle or key not in bundle['models']:
        raise HTTPException(409, 'Modelo no disponible; prepara los datos con --train. Para boosting instala su biblioteca y vuelve a entrenar.')
    model = bundle['models'][key]
    result = {'key': key, 'name': MODEL_NAMES[key], 'category': bundle['classes'][int(model.predict(vector[None])[0])],
              'probability_label': 'Probabilidad estimada por el modelo', 'probabilities': []}
    if hasattr(model, 'predict_proba'):
        result['probabilities'] = sorted([{'category': bundle['classes'][int(c)], 'value': float(v)} for c, v in
                                         zip(model.classes_, model.predict_proba(vector[None])[0])], key=lambda x: -x['value'])[:8]
    if key in {'gini', 'entropy'}:
        nodes = model.decision_path(vector[None]).indices
        result['decisions'] = [{'node': int(n), 'feature': int(model.tree_.feature[n]),
                                'threshold': float(model.tree_.threshold[n]),
                                'value': float(vector[model.tree_.feature[n]]) if model.tree_.feature[n] >= 0 else None,
                                'impurity': float(model.tree_.impurity[n])} for n in nodes]
    return result


@app.post('/api/classify')
async def classify(file: UploadFile = File(...), model: str = Form('gini'), compare: bool = Form(False), k: int = Form(5)):
    ready()
    if model not in MODEL_NAMES or not 1 <= k <= 20:
        raise HTTPException(422, 'Modelo o k inválido.')
    vector = await vector_from(file)
    keys = [r['key'] for r in status()['models'] if r['available']] if compare else [model]
    results = [await run_in_threadpool(predict, vector, key, k) for key in keys]
    return {'predictions': results, 'descriptor': vector.tolist()}


@app.get('/api/experiments')
def experiments():
    manifest, _ = ready()
    path = DATA / 'experiments.json'
    if not path.exists():
        return {'metrics': [], 'message': 'Evaluación pendiente. Ejecuta python -m scripts.prepare --train.'}
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['fingerprint'] != manifest['fingerprint']:
        return {'metrics': [], 'message': 'La evaluación pertenece a otra preparación. Vuelve a entrenar.'}
    return result


# Production frontend, optional: API also works before npm run build.
if (ROOT / 'frontend/dist').exists():
    app.mount('/', StaticFiles(directory=ROOT / 'frontend/dist', html=True), name='frontend')
