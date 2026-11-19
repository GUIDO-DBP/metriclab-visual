import json
import hashlib
import time
from collections import Counter
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .core import DATA, ROOT, DESCRIPTOR, NAMES, Engine, describe_bytes, load_vectors
from .modeling import (MODEL_NAMES, FAMILIES, PRIMARY_KEYS, artifact_directory,
                       decision_path, feature_name, tree_structure)
from .descriptors import ResNetRegistry, RESNET_ID

state = {'manifest': None, 'engine': None, 'bundle': None, 'error': None,
         'model_error': None, 'metric_error': None, 'artifacts': None, 'resnet': None}


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
            state['resnet'] = ResNetRegistry(manifest)
            # Classifiers and the complementary C++ laboratory are independent.
            try:
                artifacts = artifact_directory(DATA)
                state['artifacts'] = artifacts
                if (artifacts / 'models.joblib').exists():
                    bundle = joblib.load(artifacts / 'models.joblib')
                    if bundle['fingerprint'] != manifest['fingerprint']:
                        raise ValueError('Los modelos pertenecen a otra preparación de datos. Vuelve a entrenar.')
                    if bundle.get('descriptor', DESCRIPTOR) != DESCRIPTOR:
                        raise ValueError('Los modelos usan otro descriptor.')
                    state['bundle'] = bundle
            except Exception as exc:
                state['model_error'] = str(exc)
            try:
                state['engine'] = Engine(train)
            except Exception as exc:
                state['metric_error'] = str(exc)
    except Exception as exc:
        state['error'] = str(exc)
    yield
    if state['engine']:
        state['engine'].close()
    if state['resnet']:
        state['resnet'].close()
    state.update(manifest=None, engine=None, bundle=None, error=None,
                 model_error=None, metric_error=None, artifacts=None, resnet=None)


app = FastAPI(title='MetricLab Visual', lifespan=lifespan)


def ready():
    if state['engine'] is None:
        raise HTTPException(503, 'Índice no construido. Ejecuta scripts/build-engine.ps1 y python -m scripts.prepare --train; reinicia el backend. ' + (state['error'] or ''))
    return state['manifest'], state['engine']


def dataset_ready():
    if state['manifest'] is None:
        raise HTTPException(503, 'Dataset no preparado. Prepara las imágenes y reinicia la aplicación.')
    return state['manifest']


def require_descriptor(descriptor_id):
    if descriptor_id not in {DESCRIPTOR,RESNET_ID}:
        raise HTTPException(422,'Descriptor desconocido.')
    if descriptor_id==RESNET_ID and (not state['resnet'] or not state['resnet'].available):
        raise HTTPException(409,state['resnet'].reason if state['resnet'] else 'ResNet no disponible.')


def trained_model(key,descriptor_id=DESCRIPTOR):
    require_descriptor(descriptor_id)
    if key not in PRIMARY_KEYS:
        raise HTTPException(422, 'Selecciona un clasificador de árboles disponible.')
    if descriptor_id==RESNET_ID:
        try: return state['resnet'].model(key)
        except ValueError as exc: raise HTTPException(409,str(exc)) from exc
    bundle = state['bundle']
    if not bundle or key not in bundle['models']:
        raise HTTPException(409, 'Modelo pendiente. Ejecuta el entrenamiento de clasificadores y reinicia la aplicación.')
    return bundle['models'][key], bundle['classes']


@app.get('/api/status')
def status(descriptor_id: str = DESCRIPTOR):
    if descriptor_id not in {DESCRIPTOR,RESNET_ID}: raise HTTPException(422,'Descriptor desconocido.')
    manifest = state['manifest']
    bundle = state['bundle'] or {}
    resnet=state['resnet']
    if descriptor_id==RESNET_ID:
        available=bool(resnet and resnet.available)
        return {**status(DESCRIPTOR),'classification_ready':available,
            'descriptor':RESNET_ID,'descriptor_id':RESNET_ID,'dimensions':512,
            'descriptor_name':'ResNet-18','run_id':resnet.config['run_id'] if available else None,
            'model_error':None if available else (resnet.reason if resnet else 'ResNet pendiente'),
            'models':[{'key':key,'name':name,'family':FAMILIES[key],'primary':key!='knn',
                       'available':available and key!='knn','reason':'' if available and key!='knn' else 'Modelo no disponible para este descriptor.'}
                      for key,name in MODEL_NAMES.items()]}
    primary_available = [key for key in PRIMARY_KEYS if key in bundle.get('models', {})]
    return {'ready': manifest is not None, 'error': state['error'],
            'classification_ready': len({FAMILIES[key] for key in primary_available}) == 4,
            'metric_ready': state['engine'] is not None,
            'model_error': state['model_error'], 'metric_error': state['metric_error'],
            'run_id': bundle.get('run_id'), 'algorithm_count': 4,
            'dataset': manifest['dataset'] if manifest else 'Sin dataset preparado',
            'kind': manifest['kind'] if manifest else None,
            'indexed': len(manifest['train']) if manifest else 0,
            'queries': len(manifest['query']) if manifest else 0,
            'categories': sorted({r['category'] for r in manifest['train']}) if manifest else [],
            'dimensions': 144, 'descriptor': DESCRIPTOR,'descriptor_id':DESCRIPTOR,'descriptor_name':'HSV + Sobel',
            'models': [{'key': key, 'name': name, 'family': FAMILIES[key], 'primary': key != 'knn',
                        'available': bool(state['engine']) if key == 'knn' else key in bundle.get('models', {}),
                        'reason': '' if (bool(state['engine']) if key == 'knn' else key in bundle.get('models', {})) else bundle.get('unavailable', {}).get(key, 'Modelo pendiente: ejecuta python -m scripts.train_classifiers.')} for key, name in MODEL_NAMES.items()]}


@app.get('/api/descriptors')
def descriptors():
    resnet=state['resnet']
    return {'default':DESCRIPTOR,'descriptors':[
        {'id':DESCRIPTOR,'name':'HSV + Sobel','dimensions':144,'available':state['bundle'] is not None,'reason':state['model_error'] or ''},
        {'id':RESNET_ID,'name':'ResNet-18','dimensions':512,'available':bool(resnet and resnet.available),
         'reason':resnet.reason if resnet else 'Registro ResNet pendiente.','device':resnet.device if resnet else 'sin iniciar'}]}


@app.get('/api/dataset')
def dataset(descriptor_id: str = DESCRIPTOR):
    manifest = dataset_ready()
    samples = list({r['category']: r for r in reversed(manifest['query'])}.values())
    samples.sort(key=lambda r: r['category'])
    counts = {s: dict(Counter(r['category'] for r in manifest[s])) for s in ['train', 'query']}
    values = list(counts['train'].values())
    return {**status(descriptor_id), 'seed': manifest['seed'], 'query_fraction': manifest['query_fraction'],
            'total_processed': len(manifest['train']) + len(manifest['query']),
            'class_counts': counts, 'imbalance': {'min_train': min(values), 'max_train': max(values),
                                                'ratio': max(values)/min(values)},
            'fingerprint': manifest['fingerprint'], 'image_size': 224 if descriptor_id==RESNET_ID else manifest['image_size'],
            'max_per_class': manifest['max_per_class'], 'skipped': len(manifest['skipped']),
            'samples': [{**r, 'url': f"/api/images/query/{r['id']}"} for r in samples[:20]],
            'counts': counts['train']}


def category_examples(category,limit=6):
    manifest=dataset_ready()
    return [{'id':r['id'],'category':r['category'],'split':'train','url':f"/api/images/train/{r['id']}"}
            for r in manifest['train'] if r['category']==category][:limit]


@app.get('/api/categories/{category}/images')
def category_images(category: str,limit: int=Query(6,ge=1,le=6)):
    images=category_examples(category,limit)
    if not images: raise HTTPException(404,'Categoría no encontrada.')
    return {'category':category,'split':'train','images':images,
            'note':'Ejemplos reales de la categoría; no son resultados de similitud ni garantizan la predicción.'}


@app.get('/api/images/{split}/{image_id}')
def image(split: str, image_id: int):
    manifest = dataset_ready()
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


async def extract_query(file,descriptor_id=DESCRIPTOR):
    require_descriptor(descriptor_id)
    raw = await file.read(10*1024*1024+1)
    if len(raw) > 10*1024*1024:
        raise HTTPException(413, 'La imagen supera el límite de 10 MB.')
    try:
        start=time.perf_counter()
        if descriptor_id==DESCRIPTOR:
            vector=await run_in_threadpool(describe_bytes, raw)
            info={'device':'cpu','cache_hit':False}
        else:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {'JPEG','PNG'} or image.width*image.height>24_000_000:
                    raise ValueError('Solo JPG/PNG, máximo 24 megapíxeles.')
                image.verify()
            vector,info=await run_in_threadpool(state['resnet'].extract,raw)
        expected=144 if descriptor_id==DESCRIPTOR else 512
        if vector.shape!=(expected,) or not np.isfinite(vector).all(): raise ValueError('Vector incompatible con el descriptor.')
        return vector,{**info,'extraction_ms':(time.perf_counter()-start)*1000}
    except Exception as exc:
        raise HTTPException(422, f'Error de procesamiento: {exc}') from exc


async def vector_from(file,descriptor_id=DESCRIPTOR):
    return (await extract_query(file,descriptor_id))[0]


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


def predict(vector, key, k, descriptor_id=DESCRIPTOR):
    if key == 'knn':
        manifest, engine = ready()
        start = time.perf_counter_ns()
        result = enrich(engine.search(vector, min(k, len(manifest['train'])), 1), manifest)
        neighbors = result['runs'][0]['neighbors']
        labels = [n['category'] for n in neighbors]
        votes = Counter(labels)
        winner = max(labels, key=lambda label: votes[label])
        return {'key': key, 'name': MODEL_NAMES[key], 'category': winner, 'neighbors': neighbors,
                'family': 'secondary', 'inference_ms': (time.perf_counter_ns()-start)/1e6,
                'probability_label': 'Proporción de votos (no probabilidad calibrada)',
                'probabilities': [{'category': label, 'value': n/len(labels)} for label, n in votes.most_common()]}
    model, classes = trained_model(key,descriptor_id)
    if model.n_features_in_!=len(vector): raise HTTPException(409,'El modelo no es compatible con el vector seleccionado.')
    start = time.perf_counter_ns()
    probabilities = model.predict_proba(vector[None])[0]
    inference_ms = (time.perf_counter_ns()-start)/1e6
    winner = int(model.classes_[int(np.argmax(probabilities))])
    result = {'key': key, 'name': MODEL_NAMES[key], 'family': FAMILIES[key],
              'category': classes[winner], 'inference_ms': inference_ms,
              'criterion': key if key in {'gini', 'entropy'} else None,
              'descriptor_id':descriptor_id,'examples':category_examples(classes[winner]),
                'model_run_id':state['resnet'].config['models'][key]['metadata']['source_run'] if descriptor_id==RESNET_ID else state['bundle'].get('run_id'),
              'probability_label': 'Probabilidad estimada por el modelo (sin calibración)',
              'probabilities': sorted([{'category': classes[int(c)], 'value': float(v)} for c, v in
                                      zip(model.classes_, probabilities)], key=lambda x: -x['value'])}
    if key in {'gini', 'entropy'}:
        result['decisions'] = decision_path(model, key, classes, vector,descriptor_id=descriptor_id)
    return result


@app.post('/api/classify')
async def classify(file: UploadFile = File(...), model: str = Form('gini'), compare: bool = Form(False),
                   k: int = Form(5), criterion: str = Form('gini'), descriptor_id: str = Form(DESCRIPTOR)):
    total_start=time.perf_counter()
    manifest = dataset_ready()
    if model not in MODEL_NAMES or not 1 <= k <= 20 or criterion not in {'gini', 'entropy'}:
        raise HTTPException(422, 'Modelo o k inválido.')
    if model=='knn' and descriptor_id!=DESCRIPTOR: raise HTTPException(422,'k-NN métrico conserva el descriptor HSV original.')
    vector,extraction = await extract_query(file,descriptor_id)
    chosen_criterion = model if model in {'gini', 'entropy'} else criterion
    keys = [chosen_criterion, 'forest', 'xgboost', 'lightgbm'] if compare else [model]
    available = {r['key']: r for r in status(descriptor_id)['models']}
    unavailable = [available[key] for key in keys if not available[key]['available']]
    if not compare and unavailable:
        raise HTTPException(409, unavailable[0]['reason'])
    keys = [key for key in keys if available[key]['available']]
    results = [await run_in_threadpool(predict, vector, key, k,descriptor_id) for key in keys]
    return {'predictions': results, 'descriptor': vector.tolist(), 'unavailable': unavailable,
            'agreement': dict(Counter(r['category'] for r in results)),
            'unanimous': len({r['category'] for r in results}) == 1,
            'vector_sha256': hashlib.sha256(vector.tobytes()).hexdigest(),
            'dataset': manifest['dataset'], 'run_id': status(descriptor_id)['run_id'],
            'descriptor_id':descriptor_id,'dimensions':len(vector),'extraction':extraction,
              'preprocessing':({'weights':'ResNet18_Weights.IMAGENET1K_V1','resize':256,'center_crop':224,'rgb':True,'mean':[.485,.456,.406],'std':[.229,.224,.225],'layer':'avgpool / fc Identity','checkpoint_sha256':state['resnet'].config['checkpoint_sha256']} if descriptor_id==RESNET_ID else {'extractor':DESCRIPTOR,'image_size':128}),
              'evaluation_scope':'historical_test_reused',
            'total_ms':(time.perf_counter()-total_start)*1000}


@app.get('/api/models/{key}/importance')
def importance(key: str,descriptor_id: str=DESCRIPTOR):
    model, _ = trained_model(key,descriptor_id)
    values = np.asarray(model.feature_importances_, dtype=float)
    total = float(values.sum())
    normalized = values / total if total else values
    measure = 'Ganancia' if key in {'xgboost', 'lightgbm'} else 'Reducción de impureza'
    return {'key': key, 'measure': measure,
            'note': 'Importancias del modelo entrenado, normalizadas para sumar 1. Comparables dentro de este modelo; no indican causalidad.',
            'descriptor_id':descriptor_id,
            'groups': {'embedding':float(normalized.sum())} if descriptor_id==RESNET_ID else {'hsv': float(normalized[:128].sum()), 'edges': float(normalized[128:].sum())},
            'features': sorted([{'index': i, 'name': feature_name(i,descriptor_id), 'value': float(v),
                                 'raw_value': float(values[i]), 'group': 'Embedding' if descriptor_id==RESNET_ID else ('HSV' if i < 128 else 'Bordes')}
                                for i, v in enumerate(normalized)], key=lambda row: -row['value'])}


@app.get('/api/models/{key}/tree')
def model_tree(key: str, tree_index: int = Query(0, ge=0), root: int = Query(0, ge=0), max_depth: int = Query(3, ge=1, le=4),descriptor_id: str=DESCRIPTOR):
    model, classes = trained_model(key,descriptor_id)
    try:
        return tree_structure(model, key, classes, tree_index, root, max_depth,descriptor_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/models/{key}/path')
async def model_path(key: str, file: UploadFile = File(...), tree_index: int = Form(0),descriptor_id: str=Form(DESCRIPTOR)):
    model, classes = trained_model(key,descriptor_id)
    vector = await vector_from(file,descriptor_id)
    try:
        path = await run_in_threadpool(decision_path, model, key, classes, vector, tree_index,descriptor_id)
        return {'key': key, 'tree_index': tree_index, 'path': path,
                'note': 'Ruta de este árbol individual. En Random Forest, la predicción final combina todos los árboles.'}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get('/api/models/{key}')
def model_info(key: str,descriptor_id: str=DESCRIPTOR):
    model, classes = trained_model(key,descriptor_id)
    info = {'key': key, 'name': MODEL_NAMES[key], 'n_classes': len(classes),
            'dimensions': int(model.n_features_in_),'descriptor_id':descriptor_id,
            'metadata': state['resnet'].config['models'][key]['metadata'] if descriptor_id==RESNET_ID else state['bundle'].get('metadata', {}).get(key, {})}
    if key == 'forest':
        info.update(tree_count=len(model.estimators_), explorable_trees=min(10, len(model.estimators_)))
    elif key == 'xgboost':
        info.update(boosting_rounds=model.get_booster().num_boosted_rounds())
    elif key == 'lightgbm':
        info.update(boosting_rounds=model.booster_.current_iteration(), tree_count=model.booster_.num_trees())
    else:
        info.update(tree_count=1, criterion=model.criterion, total_nodes=model.tree_.node_count)
    return info


@app.get('/api/experiments')
def experiments(descriptor_id: str=DESCRIPTOR,evaluation: str='historical_test'):
    manifest = dataset_ready()
    require_descriptor(descriptor_id)
    if evaluation not in {'historical_test','training_cv'}: raise HTTPException(422,'Evaluación desconocida.')
    if descriptor_id==RESNET_ID: return state['resnet'].experiments(evaluation)
    if evaluation=='training_cv': return {'metrics':[],'message':'No hay CV interna guardada para HSV en esta versión.','evaluation_scope':'training_cv','descriptor_id':DESCRIPTOR}
    path = (state['artifacts'] or DATA) / 'experiments.json'
    if not path.exists():
        return {'metrics': [], 'message': 'Evaluación pendiente. Ejecuta python -m scripts.prepare --train.'}
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['fingerprint'] != manifest['fingerprint']:
        return {'metrics': [], 'message': 'La evaluación pertenece a otra preparación. Vuelve a entrenar.'}
    if result.get('run_id') != (state['bundle'] or {}).get('run_id'):
        return {'metrics': [], 'message': 'Los modelos y la evaluación pertenecen a distintas ejecuciones.'}
    return {**result,'descriptor_id':DESCRIPTOR,'evaluation_scope':'historical_test',
            'protocol':result['protocol']+' Prueba histórica reutilizada: no es una evaluación independiente nueva.'}


# Production frontend, optional: API also works before npm run build.
if (ROOT / 'frontend/dist').exists():
    app.mount('/', StaticFiles(directory=ROOT / 'frontend/dist', html=True), name='frontend')
