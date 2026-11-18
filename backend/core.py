"""Descriptor original y adaptador del motor. Sin algoritmos sustitutos."""
import ctypes as ct
import importlib.util
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = (ROOT / os.getenv('METRICLAB_DATA', 'runtime/data')).resolve()
spec = importlib.util.spec_from_file_location('original_features', ROOT / 'metriclab/demo/extract_features.py')
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)
DESCRIPTOR = 'hsv128-edge16-v1'
NAMES = ['Secuencial', 'VP-Tree', 'LAESA']


def describe_bytes(raw):
    # Decode once with Pillow to bound pixel dimensions before OpenCV allocates.
    import io
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in {'JPEG', 'PNG'}:
            raise ValueError('Solo se admiten archivos JPG y PNG.')
        if image.width * image.height > 24_000_000:
            raise ValueError('La imagen supera el límite de 24 megapíxeles.')
        image.verify()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('No se pudo decodificar la imagen.')
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    return np.concatenate([extractor.color_hist(img), extractor.edge_hist(img)]).astype(np.float32)


def load_vectors(path):
    import struct
    with open(path, 'rb') as f:
        if f.read(4) != b'MLV1':
            raise ValueError('Archivo de vectores incompatible.')
        n, d = struct.unpack('<II', f.read(8))
        return np.frombuffer(f.read(), dtype='<f4').reshape(n, d).copy()


class Engine:
    def __init__(self, vectors):
        self.lock = threading.Lock()
        library = ROOT / 'runtime' / ('metriclab.dll' if os.name == 'nt' else 'libmetriclab.so')
        self.lib = ct.CDLL(str(library))
        ptr = ct.POINTER(ct.c_float)
        self.lib.ml_create.argtypes = [ptr, ct.c_uint, ct.c_uint]
        self.lib.ml_create.restype = ct.c_void_p
        self.lib.ml_error.restype = ct.c_char_p
        self.lib.ml_search.argtypes = [ct.c_void_p, ptr, ct.c_uint, ct.c_int]
        self.lib.ml_search.restype = ct.c_char_p
        self.lib.ml_structure.argtypes = [ct.c_void_p]
        self.lib.ml_structure.restype = ct.c_char_p
        self.lib.ml_destroy.argtypes = [ct.c_void_p]
        self.handle = self.lib.ml_create(vectors.ctypes.data_as(ptr), *vectors.shape)
        if not self.handle:
            raise RuntimeError(self.lib.ml_error().decode())
        self.structure = json.loads(self.lib.ml_structure(self.handle))

    def search(self, vector, k=10, selected=-1):
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        if vector.shape != (144,) or not np.isfinite(vector).all():
            raise ValueError('Vector inválido: se requieren 144 valores finitos.')
        with self.lock:
            raw = self.lib.ml_search(self.handle, vector.ctypes.data_as(ct.POINTER(ct.c_float)), k, selected)
            if raw is None:
                raise RuntimeError(self.lib.ml_error().decode())
            return json.loads(raw)

    def close(self):
        if self.handle:
            self.lib.ml_destroy(self.handle)
            self.handle = None
