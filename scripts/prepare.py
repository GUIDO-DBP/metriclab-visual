"""Preparación reproducible offline. Ejecutar con python -m scripts.prepare."""
import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from backend.core import DATA, ROOT, DESCRIPTOR, Engine, extractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='dataset/101_ObjectCategories')
    ap.add_argument('--kind', choices=['caltech101', 'synthetic'], default='caltech101')
    ap.add_argument('--max-per-class', type=int, default=0)
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--train-only', action='store_true', help='Reutiliza los vectores preparados y vuelve a entrenar/evaluar.')
    args = ap.parse_args()
    if args.train_only:
        from backend.core import load_vectors
        manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
        train({split: load_vectors(DATA / f'{split}.bin') for split in ['train', 'query']}, manifest)
        return
    root = (ROOT / args.images).resolve()
    if not root.is_relative_to(ROOT):
        raise SystemExit('Copia el dataset dentro del proyecto para mantener rutas portables.')
    if not any(root.is_relative_to(ROOT / folder) for folder in ['dataset', 'runtime']):
        raise SystemExit('Coloca las imágenes dentro de dataset/ o runtime/.')
    rng = random.Random(7)
    rows = {'train': [], 'query': []}
    vectors = {'train': [], 'query': []}
    skipped = []
    categories = sorted(p for p in root.iterdir() if p.is_dir() and p.name not in extractor.SKIP)
    for category_index, category in enumerate(categories):
        if category_index % 10 == 0:
            print(f'Procesando categoría {category_index+1}/{len(categories)}: {category.name}', flush=True)
        files = sorted(p for p in category.iterdir() if p.suffix.lower() in extractor.EXTS)
        rng.shuffle(files)
        if args.max_per_class:
            files = files[:args.max_per_class]
        valid = []
        for path in files:
            value = extractor.describe(path)
            if value is None:
                skipped.append(path.relative_to(ROOT).as_posix())
            else:
                valid.append((path, value))
        if len(valid) < 2:
            skipped.extend(p.relative_to(ROOT).as_posix() for p, _ in valid)
            continue
        n_query = min(len(valid)-1, max(1, round(len(valid)*.2)))
        for i, (path, value) in enumerate(valid):
            split = 'query' if i < n_query else 'train'
            rows[split].append({'id': len(rows[split]), 'category': category.name,
                                'path': path.relative_to(ROOT).as_posix()})
            vectors[split].append(value)
    if not vectors['train'] or not vectors['query']:
        raise SystemExit('Se necesitan al menos dos imágenes válidas por categoría.')
    DATA.mkdir(parents=True, exist_ok=True)
    for split in rows:
        vectors[split] = np.vstack(vectors[split])
        extractor.write_bin(DATA / f'{split}.bin', vectors[split])
    fingerprint = hashlib.sha256(vectors['train'].tobytes()+vectors['query'].tobytes()+json.dumps(rows, sort_keys=True).encode()).hexdigest()
    manifest = {'dataset': 'Caltech-101' if args.kind == 'caltech101' else 'Demostración sintética',
                'kind': args.kind, 'descriptor': DESCRIPTOR, 'dimensions': 144, 'image_size': 128,
                'seed': 7, 'query_fraction': .2, 'max_per_class': args.max_per_class,
                'fingerprint': fingerprint, 'skipped': skipped, **rows}
    (DATA / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    print(f"Preparadas {len(rows['train'])} indexadas y {len(rows['query'])} consultas; {len(skipped)} omitidas.", flush=True)
    if args.train:
        train(vectors, manifest)


def train(vectors, manifest):
    # Reuse the project's constructors; optional imports remain optional.
    import importlib.util
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    spec = importlib.util.spec_from_file_location('original_classify', ROOT / 'metriclab/demo/classify.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    enc = LabelEncoder().fit([r['category'] for r in manifest['train']])
    y = enc.transform([r['category'] for r in manifest['train']])
    truth = enc.transform([r['category'] for r in manifest['query']])
    classes = enc.classes_.tolist()
    models, metrics, unavailable = {}, [], {}

    def evaluate(key, name, predictions, seconds):
        metrics.append({'key': key, 'model': name, 'accuracy': float(accuracy_score(truth, predictions)),
                        'balanced_accuracy': float(balanced_accuracy_score(truth, predictions)),
                        'f1_macro': float(f1_score(truth, predictions, average='macro')),
                        'train_s': seconds,
                        'confusion': confusion_matrix(truth, predictions, labels=range(len(classes))).tolist()})

    for key, (name, constructor, reason) in zip(['gini', 'entropy', 'forest', 'xgboost', 'lightgbm'], mod.build_models()):
        if constructor is None:
            unavailable[key] = reason
            continue
        try:
            print(f'Entrenando {name}…', flush=True)
            model = constructor()
            start = time.perf_counter()
            model.fit(vectors['train'], y)
            seconds = time.perf_counter()-start
            pred = model.predict(vectors['query'])
            evaluate(key, name, pred, seconds)
            models[key] = model
        except Exception as exc:
            unavailable[key] = f'Error al entrenar: {exc}'
    engine = Engine(vectors['train'])
    preds = []
    for value in vectors['query']:
        neighbors = engine.search(value, min(5, len(y)), 1)['runs'][0]['neighbors']
        labels = [int(y[n['id']]) for n in neighbors]
        counts = Counter(labels)
        preds.append(max(labels, key=lambda label: counts[label]))
    engine.close()
    evaluate('knn', 'k-NN / VP-Tree (k=5)', preds, 0.0)
    bundle = {'fingerprint': manifest['fingerprint'], 'models': models, 'classes': classes, 'unavailable': unavailable}
    joblib.dump(bundle, DATA / 'models.joblib', compress=3)
    experiments = {'fingerprint': manifest['fingerprint'], 'dataset': manifest['dataset'], 'kind': manifest['kind'],
                   'n_train': len(y), 'n_test': len(truth), 'classes': classes, 'metrics': metrics,
                   'protocol': 'Separación estratificada 80/20, semilla 7. Sin ajuste de hiperparámetros. k-NN k=5. No es validación cruzada.'}
    (DATA / 'experiments.json').write_text(json.dumps(experiments, ensure_ascii=False), encoding='utf-8')
    print('Modelos y evaluación guardados.', flush=True)


if __name__ == '__main__':
    main()
