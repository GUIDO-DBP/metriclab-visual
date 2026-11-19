"""Preparación reproducible offline. Ejecutar con python -m scripts.prepare."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from backend.core import DATA, ROOT, DESCRIPTOR, extractor


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
    # Single training pipeline, preserving the prepared split and prior artifacts.
    from scripts.train_classifiers import train as train_classifiers
    return train_classifiers(vectors, manifest)


if __name__ == '__main__':
    main()
