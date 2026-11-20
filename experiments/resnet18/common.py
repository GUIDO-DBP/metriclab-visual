import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE_DATA = ROOT / 'runtime/data'
OUTPUT_ROOT = ROOT / 'runtime/experiments/resnet18'
DESCRIPTOR = 'torchvision-resnet18-imagenet1k-v1-avgpool512-fp32'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_bin(path):
    with Path(path).open('rb') as f:
        if f.read(4) != b'MLV1':
            raise ValueError('Not an MLV1 feature file.')
        n, d = struct.unpack('<II', f.read(8))
        return np.frombuffer(f.read(), dtype='<f4').reshape(n, d).copy()


def image_path(row):
    path = (ROOT / row['path']).resolve()
    if not path.is_relative_to((ROOT / 'dataset').resolve()) or not path.is_file():
        raise ValueError(f"Missing/invalid dataset path: {row['path']}")
    return path


def validate_original():
    manifest = load_json(BASE_DATA / 'manifest.json')
    if manifest['kind'] != 'caltech101' or manifest['descriptor'] != 'hsv128-edge16-v1':
        raise ValueError('This experiment requires the original real Caltech/HSV preparation.')
    if (len(manifest['train']), len(manifest['query'])) != (6941, 1736):
        raise ValueError('Expected the existing 6941/1736 split; no new split will be made.')
    matrices = {s: load_bin(BASE_DATA / f'{s}.bin') for s in ['train', 'query']}
    for split, matrix in matrices.items():
        if matrix.shape != (len(manifest[split]), 144) or not np.isfinite(matrix).all():
            raise ValueError('Original vectors do not match manifest.')
        if [r['id'] for r in manifest[split]] != list(range(len(matrix))):
            raise ValueError('Original row IDs are not aligned with the vectors.')
        paths = [r['path'] for r in manifest[split]]
        if len(paths) != len(set(paths)):
            raise ValueError('Duplicate paths within split.')
    if {r['path'] for r in manifest['train']} & {r['path'] for r in manifest['query']}:
        raise ValueError('Training and test paths overlap.')
    ordered = {s: manifest[s] for s in ['train', 'query']}
    fingerprint = hashlib.sha256(matrices['train'].tobytes() + matrices['query'].tobytes() + json.dumps(ordered, sort_keys=True).encode()).hexdigest()
    if fingerprint != manifest['fingerprint']:
        raise ValueError('Original split fingerprint does not match.')
    classes = sorted({r['category'] for r in manifest['train']})
    if len(classes) != 101 or not {r['category'] for r in manifest['query']} <= set(classes):
        raise ValueError('Incompatible labels between training and test.')
    return manifest, matrices, classes


def protected_files():
    """Freeze main application sources and active artifacts without changing them."""
    paths = []
    for folder in ['backend', 'metriclab', 'scripts', 'frontend/src', 'frontend/dist']:
        paths.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in ['README.md', 'requirements.txt', 'requirements/runtime.lock.txt', 'requirements/tested-direct.txt', 'requirements/optional-postgres.txt',
                 'requirements/resnet.txt', 'requirements/documentation.txt',
                 'frontend/index.html', 'frontend/package.json', 'frontend/package-lock.json',
                 'runtime/data/manifest.json', 'runtime/data/train.bin', 'runtime/data/query.bin',
                 'runtime/data/models.joblib', 'runtime/data/experiments.json', 'runtime/data/active_classifiers.json']:
        if (ROOT / name).is_file():
            paths.append(ROOT / name)
    pointer = load_json(BASE_DATA / 'active_classifiers.json')
    active = (BASE_DATA / pointer['directory']).resolve()
    if not active.is_relative_to(BASE_DATA):
        raise ValueError('Invalid active artifact path.')
    paths.extend(p for p in active.iterdir() if p.is_file())
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(paths))}


def assert_protected(snapshot):
    current = protected_files()
    if current != snapshot:
        changed = sorted(k for k in set(current) | set(snapshot) if current.get(k) != snapshot.get(k))
        raise ValueError(f'Main application changed during the experiment: {changed}')
    return len(current)


def validate_embeddings(run, rehash_images=False):
    run = Path(run)
    manifest, _, classes = validate_original()
    saved = load_json(run / 'split.json')
    if saved != manifest or load_json(run / 'classes.json') != classes:
        raise ValueError('Experimental split/classes differ from the original preparation.')
    meta = load_json(run / 'extraction.json')
    encoded = {name: i for i, name in enumerate(classes)}
    seen_hashes = {}
    checks = {}
    for split in ['train', 'query']:
        vectors = np.load(run / f'{split}_resnet18.npy', mmap_mode='r')
        labels = np.load(run / f'{split}_labels.npy')
        rows = load_json(run / f'{split}_rows.json')
        if vectors.shape != (len(manifest[split]), 512) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
            raise ValueError('Invalid embedding matrix.')
        if labels.tolist() != [encoded[r['category']] for r in manifest[split]]:
            raise ValueError('Labels not aligned with original manifest.')
        if len(rows) != len(labels):
            raise ValueError('Incomplete embedding row metadata.')
        for i, (actual, original) in enumerate(zip(rows, manifest[split])):
            if {k: actual[k] for k in ['id', 'category', 'path']} != original or actual['row'] != i or actual['label'] != int(labels[i]):
                raise ValueError('Embedding row does not match its original image/label.')
            if rehash_images and sha256(image_path(actual)) != actual['image_sha256']:
                raise ValueError('Image changed since extraction.')
            seen_hashes.setdefault(actual['image_sha256'], set()).add(split)
        if sha256(run / f'{split}_resnet18.npy') != meta['splits'][split]['vectors_sha256']:
            raise ValueError('Embedding checksum mismatch.')
        checks[split] = {'shape': list(vectors.shape), 'rows_labels_aligned': True}
    checks['identical_content_across_splits'] = sum(len(splits) > 1 for splits in seen_hashes.values())
    checks['original_fingerprint'] = manifest['fingerprint']
    return checks
