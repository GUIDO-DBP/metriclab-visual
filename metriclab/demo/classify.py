#!/usr/bin/env python3
"""Modulo de clasificacion de MetricLab Visual.

Compara los cuatro algoritmos pedidos por el docente, mas un quinto que
conecta la clasificacion con el indice metrico:

  1. Decision Tree con criterio Gini
  2. Decision Tree con criterio entropia
  3. Random Forest
  4. XGBoost       (requiere: pip install xgboost)
  5. LightGBM      (requiere: pip install lightgbm)
  6. k-NN sobre el VP-Tree: NO se entrena. Toma los 5 vecinos que devolvio el
     motor C++ y vota la categoria. Es el puente entre las dos mitades del
     proyecto: clasifica usando directamente el indice metrico.

Datos: los mismos vectores y la misma separacion que la busqueda. La
COLECCION es el conjunto de entrenamiento; las CONSULTAS son el conjunto de
prueba y nunca se usan para entrenar.

Como Caltech-101 esta desbalanceado, ademas de la exactitud (accuracy) se
reportan la exactitud balanceada y el F1 macro, que dan el mismo peso a cada
categoria sin importar cuantas imagenes tenga.

Si XGBoost o LightGBM no estan instalados, se omiten con aviso y el resto
se ejecuta igual.

Uso:  python3 classify.py CARPETA_VECTORES [--k 5]
"""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0


def load_bin(path):
    with open(path, "rb") as fh:
        if fh.read(4) != b"MLV1":
            raise SystemExit(f"{path}: formato desconocido")
        n, dim = np.frombuffer(fh.read(8), dtype="<u4")
        return np.frombuffer(fh.read(), dtype="<f4").reshape(int(n), int(dim))


def load_labels(path):
    with open(path) as fh:
        return [row["categoria"] for row in csv.DictReader(fh)]


def build_models():
    """Devuelve (nombre, constructor | None, motivo_si_no_disponible)."""
    models = [
        ("Decision Tree (Gini)",
         lambda: DecisionTreeClassifier(criterion="gini", random_state=SEED), None),
        ("Decision Tree (entropía)",
         lambda: DecisionTreeClassifier(criterion="entropy", random_state=SEED), None),
        ("Random Forest",
         lambda: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1), None),
    ]
    try:
        from xgboost import XGBClassifier
        models.append(("XGBoost", lambda: XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbosity=0), None))
    except ImportError:
        models.append(("XGBoost", None, "no instalado (pip install xgboost)"))
    try:
        from lightgbm import LGBMClassifier
        models.append(("LightGBM", lambda: LGBMClassifier(
            n_estimators=200, learning_rate=0.1, num_leaves=31,
            random_state=SEED, n_jobs=-1, verbose=-1), None))
    except ImportError:
        models.append(("LightGBM", None, "no instalado (pip install lightgbm)"))
    return models


def knn_from_index(vecinos_csv, train_y, n_query, k):
    """Voto mayoritario entre los k vecinos que encontro el VP-Tree.
    Empate: gana la categoria cuyo representante esta mas cerca."""
    neigh = defaultdict(list)
    with open(vecinos_csv) as fh:
        for row in csv.DictReader(fh):
            neigh[int(row["consulta"])].append((int(row["rango"]), int(row["id_coleccion"])))
    preds = []
    for q in range(n_query):
        top = [train_y[i] for _, i in sorted(neigh[q])[:k]]
        votes = Counter(top)
        best = max(votes.values())
        tied = {c for c, v in votes.items() if v == best}
        preds.append(next(c for c in top if c in tied))  # el mas cercano entre empatados
    return np.array(preds)


def precision_at_k(vecinos_csv, train_y, query_y, k):
    """Precision@k de la BUSQUEDA, promediada por categoria (macro)."""
    neigh = defaultdict(list)
    with open(vecinos_csv) as fh:
        for row in csv.DictReader(fh):
            neigh[int(row["consulta"])].append((int(row["rango"]), int(row["id_coleccion"])))
    per_class = defaultdict(list)
    for q, lab in enumerate(query_y):
        top = [train_y[i] for _, i in sorted(neigh[q])[:k]]
        per_class[lab].append(sum(t == lab for t in top) / k)
    return float(np.mean([np.mean(v) for v in per_class.values()])), \
        {c: float(np.mean(v)) for c, v in per_class.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vectores")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    d = Path(args.vectores)

    X_tr, X_te = load_bin(d / "train.bin"), load_bin(d / "query.bin")
    y_tr, y_te = np.array(load_labels(d / "train_meta.csv")), np.array(load_labels(d / "query_meta.csv"))

    enc = LabelEncoder().fit(np.concatenate([y_tr, y_te]))
    yi_tr, yi_te = enc.transform(y_tr), enc.transform(y_te)
    classes = list(enc.classes_)

    print(f"entrenamiento: {len(y_tr)}  prueba: {len(y_te)}  categorias: {len(classes)}\n")
    print(f"{'modelo':<30}{'accuracy':>10}{'acc. bal.':>11}{'F1 macro':>10}{'entreno s':>11}")
    print("-" * 72)

    rows, confusions = [], {}

    for name, ctor, reason in build_models():
        if ctor is None:
            print(f"{name:<30}{'—':>10}{'—':>11}{'—':>10}   {reason}")
            rows.append({"modelo": name, "estado": reason})
            continue
        model = ctor()
        t0 = time.perf_counter()
        model.fit(X_tr, yi_tr)
        t_fit = time.perf_counter() - t0
        pred = model.predict(X_te)
        rows.append(_metrics(name, yi_te, pred, t_fit))
        confusions[name] = confusion_matrix(yi_te, pred, labels=range(len(classes)))
        _print(rows[-1])

    # k-NN usando el indice metrico (sin entrenamiento)
    name = f"k-NN sobre VP-Tree (k={args.k})"
    t0 = time.perf_counter()
    pred = enc.transform(knn_from_index(d / "vecinos.csv", y_tr, len(y_te), args.k))
    t_knn = time.perf_counter() - t0
    rows.append(_metrics(name, yi_te, pred, 0.0))
    rows[-1]["nota"] = "sin entrenamiento: usa los vecinos del motor C++"
    confusions[name] = confusion_matrix(yi_te, pred, labels=range(len(classes)))
    _print(rows[-1])

    p_at_k, p_per_class = precision_at_k(d / "vecinos.csv", y_tr, y_te, args.k)
    print(f"\nPrecision@{args.k} de la busqueda (macro por categoria): {p_at_k:.3f}")

    # --- salidas ---
    with open(d / "clasificacion.csv", "w", newline="") as fh:
        keys = ["modelo", "estado", "accuracy", "balanced_accuracy", "f1_macro", "entreno_s", "nota"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    with open(d / "resumen.json", "w") as fh:
        json.dump({"k": args.k, "precision_at_k": p_at_k,
                   "precision_por_categoria": p_per_class,
                   "categorias": classes,
                   "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                   "clasificadores": rows}, fh, ensure_ascii=False, indent=2)

    _plot_confusions(confusions, classes, d / "matrices_confusion.png")
    print(f"\nescrito clasificacion.csv, resumen.json, matrices_confusion.png en {d}")


def _metrics(name, y_true, y_pred, t_fit):
    return {"modelo": name, "estado": "ok",
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "entreno_s": round(t_fit, 3)}


def _print(r):
    print(f"{r['modelo']:<30}{r['accuracy']:>10.3f}{r['balanced_accuracy']:>11.3f}"
          f"{r['f1_macro']:>10.3f}{r['entreno_s']:>11.3f}")


def _plot_confusions(confusions, classes, path):
    n = len(confusions)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, cm) in zip(axes.flat, confusions.items()):
        ax.axis("on")
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=60, fontsize=7, ha="right")
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("predicho", fontsize=8)
        ax.set_ylabel("real", fontsize=8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j]:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
