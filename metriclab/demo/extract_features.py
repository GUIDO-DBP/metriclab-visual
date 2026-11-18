#!/usr/bin/env python3
"""Extractor de caracteristicas para MetricLab Visual.

Lee una carpeta con una subcarpeta por categoria (la estructura de
Caltech-101: 101_ObjectCategories/<categoria>/image_XXXX.jpg), convierte cada
imagen en un vector y separa:

  - COLECCION (train): las imagenes que se indexan en el motor C++ y con las
    que se entrenan los clasificadores.
  - CONSULTAS (query): imagenes que NUNCA entran al indice ni al
    entrenamiento. Se usan como busquedas de prueba y como conjunto de
    evaluacion de los clasificadores.

La separacion es ESTRATIFICADA: de cada categoria se toma la misma
proporcion para consulta. Caltech-101 esta muy desbalanceado y un muestreo
al azar mediria sobre todo las clases grandes.

Descriptor (vector de 144 numeros):
  - 128: histograma de color HSV (8 tonos x 4 saturaciones x 4 brillos)
  -  16: histograma de orientacion de bordes, ponderado por su intensidad
Cada parte se normaliza para sumar 1, asi ninguna domina por escala.

Salida (en la carpeta indicada):
  train.bin / query.bin        vectores para el motor C++
  train_meta.csv / query_meta.csv  id, categoria, ruta de la imagen

Formato binario: 'MLV1' | uint32 n | uint32 dim | n*dim float32 (little endian)

Uso:
  python3 extract_features.py IMAGENES/ SALIDA/ [--query-frac 0.2]
          [--max-per-class 80] [--seed 7]
"""
import argparse
import csv
import random
import struct
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 128
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# Caltech-101 trae una carpeta de "fondo" que no es una categoria de objeto.
SKIP = {"BACKGROUND_Google"}


def color_hist(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return h / max(h.sum(), 1e-9)


def edge_hist(img_bgr, bins=16):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi  # orientacion sin sentido: [0, pi)
    h, _ = np.histogram(ang, bins=bins, range=(0, np.pi), weights=mag)
    return h / max(h.sum(), 1e-9)


def describe(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # las grises pasan a 3 canales
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return np.concatenate([color_hist(img), edge_hist(img)]).astype(np.float32)


def write_bin(path, mat):
    mat = np.ascontiguousarray(mat, dtype="<f4")
    with open(path, "wb") as fh:
        fh.write(b"MLV1")
        fh.write(struct.pack("<II", mat.shape[0], mat.shape[1]))
        fh.write(mat.tobytes())


def write_meta(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "categoria", "ruta"])
        for i, (lab, p) in enumerate(rows):
            w.writerow([i, lab, p])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("out")
    ap.add_argument("--query-frac", type=float, default=0.2)
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="limita imagenes por categoria (0 = sin limite)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_v, train_m, query_v, query_m = [], [], [], []
    categories = sorted(d for d in root.iterdir() if d.is_dir() and d.name not in SKIP)
    if not categories:
        raise SystemExit(f"No hay subcarpetas de categoria en {root}")

    fallidas = 0
    for cat in categories:
        files = sorted(f for f in cat.iterdir() if f.suffix.lower() in EXTS)
        rng.shuffle(files)
        if args.max_per_class:
            files = files[: args.max_per_class]
        n_q = max(1, round(len(files) * args.query_frac))
        for idx, f in enumerate(files):
            v = describe(f)
            if v is None:
                fallidas += 1
                continue
            if idx < n_q:
                query_v.append(v); query_m.append((cat.name, str(f)))
            else:
                train_v.append(v); train_m.append((cat.name, str(f)))

    train = np.vstack(train_v)
    query = np.vstack(query_v)
    write_bin(out / "train.bin", train)
    write_bin(out / "query.bin", query)
    write_meta(out / "train_meta.csv", train_m)
    write_meta(out / "query_meta.csv", query_m)

    print(f"categorias: {len(categories)}")
    print(f"coleccion (indexada / entrenamiento): {train.shape[0]} imagenes")
    print(f"consultas (nunca indexadas):          {query.shape[0]} imagenes")
    print(f"dimension del vector:                 {train.shape[1]}")
    if fallidas:
        print(f"imagenes que no se pudieron leer:     {fallidas}")


if __name__ == "__main__":
    main()
