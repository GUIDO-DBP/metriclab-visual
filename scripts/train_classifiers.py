"""Train the missing classifiers and evaluate all on the SAME saved split.

No test-set tuning or re-extraction. Existing valid DT/RF are reused by default.
Versioned artifacts + atomic pointer keep the previous delivery recoverable.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

from backend.core import DATA, DESCRIPTOR, load_vectors
from backend.modeling import MODEL_NAMES, PRIMARY_KEYS, artifact_directory


def validate_split(vectors, manifest):
    if manifest['descriptor'] != DESCRIPTOR:
        raise ValueError('Descriptor incompatible. No se entrenará con vectores mezclados.')
    paths = []
    for split in ['train', 'query']:
        matrix = vectors[split]
        if matrix.shape != (len(manifest[split]), 144) or not np.isfinite(matrix).all():
            raise ValueError(f'Vectores {split} incompatibles con el manifiesto.')
        paths.append({r['path'] for r in manifest[split]})
        if len(paths[-1]) != len(manifest[split]):
            raise ValueError('Hay rutas duplicadas dentro de una partición.')
    if paths[0] & paths[1]:
        raise ValueError('Las particiones de entrenamiento y prueba se solapan.')
    rows = {split: manifest[split] for split in ['train', 'query']}
    fingerprint = hashlib.sha256(vectors['train'].tobytes() + vectors['query'].tobytes() + json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if fingerprint != manifest['fingerprint']:
        raise ValueError('La huella de los vectores/etiquetas no coincide con el manifiesto.')


def constructors(n_classes, threads):
    # Reuse the original sklearn definitions, with bounded parallelism.
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    return {
        'gini': lambda: DecisionTreeClassifier(criterion='gini', random_state=0),
        'entropy': lambda: DecisionTreeClassifier(criterion='entropy', random_state=0),
        'forest': lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=threads),
        'xgboost': lambda: XGBClassifier(objective='multi:softprob', num_class=n_classes,
            n_estimators=80, max_depth=4, learning_rate=.1, subsample=.9, colsample_bytree=.9,
            tree_method='hist', max_bin=64, n_jobs=threads, random_state=0, eval_metric='mlogloss', verbosity=0),
        'lightgbm': lambda: LGBMClassifier(objective='multiclass', num_class=n_classes,
            n_estimators=80, num_leaves=15, max_depth=6, learning_rate=.1,
            min_child_samples=20, max_bin=63, n_jobs=threads, random_state=0,
            deterministic=True, force_col_wise=True, importance_type='gain', verbosity=-1),
    }


def json_value(value):
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Metadata, e.g. XGBoost missing=NaN; strict JSON.
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def train(vectors, manifest, *, retrain=False, threads=4, benchmark_samples=128):
    validate_split(vectors, manifest)
    enc = LabelEncoder().fit([r['category'] for r in manifest['train']])
    y_train = enc.transform([r['category'] for r in manifest['train']])
    y_test = enc.transform([r['category'] for r in manifest['query']])
    classes = enc.classes_.tolist()
    if len(classes) < 2:
        raise ValueError('Se necesitan al menos dos categorías para comparar clasificadores.')
    threads = max(1, min(threads, os.cpu_count() or 1))
    ctors = constructors(len(classes), threads)
    previous_dir = artifact_directory(DATA)
    previous = {}
    previous_metrics = {}
    if (previous_dir / 'models.joblib').exists():
        candidate = joblib.load(previous_dir / 'models.joblib')
        if candidate.get('fingerprint') == manifest['fingerprint'] and candidate.get('classes') == classes:
            previous = candidate
            if (previous_dir / 'experiments.json').exists():
                old_exp = json.loads((previous_dir / 'experiments.json').read_text(encoding='utf-8'))
                if old_exp.get('fingerprint') == manifest['fingerprint']:
                    previous_metrics = {r['key']: r for r in old_exp['metrics'] + old_exp.get('secondary_metrics', [])}

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    out = DATA / 'runs' / run_id
    out.mkdir(parents=True, exist_ok=False)
    versions = {p: version(p) for p in ['numpy', 'scikit-learn', 'xgboost', 'lightgbm', 'joblib', 'opencv-python-headless']}
    versions['python'] = platform.python_version()
    models, metrics, metadata = {}, [], {}
    timing_indices = np.linspace(0, len(y_test)-1, min(benchmark_samples, len(y_test)), dtype=int)
    for key in PRIMARY_KEYS:
        model = previous.get('models', {}).get(key)
        reusable = model is not None and key in {'gini', 'entropy', 'forest'} and model.n_features_in_ == 144
        if key in {'gini', 'entropy'} and reusable:
            reusable = model.criterion == ('gini' if key == 'gini' else 'entropy')
        reused = reusable and not retrain
        if reused:
            print(f'Reutilizando {MODEL_NAMES[key]} entrenado en la entrega anterior.', flush=True)
            seconds = previous_metrics.get(key, {}).get('train_s')
            training_params = previous.get('metadata', {}).get(key, {}).get('training_params') or json_value(model.get_params())
            if key == 'forest':
                model.set_params(n_jobs=threads)  # Execution policy only, no refit.
        else:
            print(f'Entrenando {MODEL_NAMES[key]} con {threads} hilos como máximo…', flush=True)
            model = ctors[key]()
            training_params = json_value(model.get_params())
            start = time.perf_counter()
            model.fit(vectors['train'], y_train)
            seconds = time.perf_counter()-start
        predicted = model.predict(vectors['query'])
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average='macro', zero_division=0)
        # Unit-query latency: same predict_proba operation as the web application.
        model.predict_proba(vectors['query'][:1])  # warm-up outside timing
        timings = []
        for i in timing_indices:
            start = time.perf_counter_ns()
            probabilities = model.predict_proba(vectors['query'][i:i+1])
            timings.append((time.perf_counter_ns()-start)/1e6)
            if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
                raise ValueError(f'Probabilidades inválidas en {key}.')
        row = {'key': key, 'model': MODEL_NAMES[key], 'accuracy': float(accuracy_score(y_test, predicted)),
               'balanced_accuracy': float(balanced_accuracy_score(y_test, predicted)),
               'precision_macro': float(precision), 'recall_macro': float(recall), 'f1_macro': float(f1),
               'train_s': seconds, 'inference_ms': float(np.mean(timings)),
               'inference_p95_ms': float(np.percentile(timings, 95)), 'inference_samples': len(timings),
               'n_train': len(y_train), 'n_test': len(y_test), 'n_classes': len(classes),
               'reused': reused, 'confusion': confusion_matrix(y_test, predicted, labels=range(len(classes))).tolist()}
        metrics.append(row)
        models[key] = model
        metadata[key] = {'name': MODEL_NAMES[key], 'implementation': type(model).__module__ + '.' + type(model).__name__,
                         'params': json_value(model.get_params()), 'training_params': training_params,
                         'training_time_source': 'previous_delivery' if reused else 'this_run', 'reused': reused,
                         'train_s': seconds, 'versions': versions, 'run_id': run_id}
        if key == 'xgboost':
            model.save_model(out / 'xgboost.json')
        if key == 'lightgbm':
            model.booster_.save_model(str(out / 'lightgbm.txt'))
        print(f"{MODEL_NAMES[key]}: accuracy={row['accuracy']:.4f}, F1 macro={row['f1_macro']:.4f}, inferencia media={row['inference_ms']:.3f} ms", flush=True)

    bundle = {'schema_version': 2, 'run_id': run_id, 'fingerprint': manifest['fingerprint'],
              'descriptor': DESCRIPTOR, 'models': models, 'classes': classes, 'metadata': metadata, 'unavailable': {}}
    joblib.dump(bundle, out / 'models.joblib', compress=3)
    # Verify serialization before publishing the run; compare complete probabilities.
    loaded = joblib.load(out / 'models.joblib')
    for key, model in models.items():
        if not np.allclose(model.predict_proba(vectors['query'][:3]), loaded['models'][key].predict_proba(vectors['query'][:3]), atol=1e-10):
            raise ValueError(f'La persistencia alteró las probabilidades de {key}.')
    experiments = {'schema_version': 2, 'run_id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(),
        'fingerprint': manifest['fingerprint'], 'dataset': manifest['dataset'], 'kind': manifest['kind'],
        'descriptor': DESCRIPTOR, 'dimensions': 144, 'n_train': len(y_train), 'n_test': len(y_test),
        'n_classes': len(classes), 'classes': classes, 'metrics': metrics,
        'secondary_metrics': [previous_metrics['knn']] if 'knn' in previous_metrics else [],
        'versions': versions, 'models': metadata,
        'class_counts': {s: dict(Counter(r['category'] for r in manifest[s])) for s in ['train', 'query']},
        'protocol': 'División estratificada existente 80/20, semilla 7. Semilla de modelos 0. Parámetros fijados antes de evaluar; sin ajuste sobre prueba ni validación cruzada. Ambos criterios de Decision Tree comparten configuración.',
        'timing_protocol': f'Latencia media de predict_proba para una imagen, {len(timing_indices)} imágenes de prueba espaciadas uniformemente, tras una consulta de calentamiento. Excluye extracción, JSON, HTTP y visualización. Máximo {threads} hilos.'}
    (out / 'experiments.json').write_text(json.dumps(experiments, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    pointer = DATA / 'active_classifiers.tmp'
    pointer.write_text(json.dumps({'directory': out.relative_to(DATA).as_posix(), 'run_id': run_id}), encoding='utf-8')
    pointer.replace(DATA / 'active_classifiers.json')
    print(f'Activada ejecución {run_id}. Artefactos anteriores conservados.', flush=True)
    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true', help='Reentrenar también los árboles y el bosque existentes.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--benchmark-samples', type=int, default=128)
    args = parser.parse_args()
    if args.threads < 1 or args.benchmark_samples < 1:
        parser.error('Los límites deben ser positivos.')
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    train({s: load_vectors(DATA / f'{s}.bin') for s in ['train', 'query']}, manifest,
          retrain=args.retrain, threads=args.threads, benchmark_samples=args.benchmark_samples)


if __name__ == '__main__':
    main()
