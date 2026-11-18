#!/usr/bin/env python3
"""Grafica resultados.csv.

MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

Uso:  python3 scripts/plot_results.py [resultados.csv]
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "resultados.csv"

rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        r["n"] = int(r["n"]); r["dim"] = int(r["dim"])
        r["rho"] = float(r["rho"])
        r["dist_por_consulta"] = float(r["dist_por_consulta"])
        r["poda_pct"] = float(r["poda_pct"])
        rows.append(r)

COLORS = {
    "LinearScan": "#8a8a8a",
    "VP-Tree[random]": "#4c78a8",
    "VP-Tree[max-varianza]": "#2b6cb0",
    "VP-Tree[max-rango]": "#7aa6d6",
    "VP-Tree[fft]": "#b5cce5",
    "LAESA[m=16]": "#e07b39",
    "LAESA[m=48]": "#b85c1e",
}


def serie(dataset, xkey):
    out = defaultdict(list)
    for r in rows:
        if r["dataset"] != dataset:
            continue
        out[r["indice"]].append((r[xkey], r["poda_pct"], r["dist_por_consulta"]))
    for k in out:
        out[k].sort()
    return out


fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# --- 1. poda vs dimension, datos agrupados -------------------------------
ax = axes[0]
for name, pts in serie("agrupado", "dim").items():
    if name == "LinearScan":
        continue
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
            label=name, color=COLORS.get(name), lw=1.8, ms=4)
ax.set_xscale("log", base=2)
ax.set_xlabel("dimension")
ax.set_ylabel("% de objetos evitados")
ax.set_title("Datos AGRUPADOS\nla poda resiste la dimension")
ax.set_ylim(0, 100)
ax.grid(alpha=0.25)
ax.legend(fontsize=7)

# --- 2. poda vs dimension, datos uniformes -------------------------------
ax = axes[1]
for name, pts in serie("uniforme", "dim").items():
    if name == "LinearScan":
        continue
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
            label=name, color=COLORS.get(name), lw=1.8, ms=4)
ax.set_xscale("log", base=2)
ax.set_xlabel("dimension")
ax.set_ylabel("% de objetos evitados")
ax.set_title("Datos UNIFORMES\nla poda se derrumba")
ax.set_ylim(0, 100)
ax.grid(alpha=0.25)
ax.legend(fontsize=7)

# --- 3. dimensionalidad intrinseca vs poda -------------------------------
ax = axes[2]
for tag, marker in (("agrupado", "o"), ("uniforme", "s")):
    xs, ys = [], []
    for r in rows:
        if r["dataset"] != tag or r["indice"] != "VP-Tree[max-varianza]":
            continue
        xs.append(r["rho"]); ys.append(r["poda_pct"])
    ax.scatter(xs, ys, marker=marker, s=48, label=tag, alpha=0.85)
ax.set_xscale("log")
ax.set_xlabel(r"dimensionalidad intrinseca  $\rho=\mu^2/2\sigma^2$")
ax.set_ylabel("% de objetos evitados")
ax.set_title("rho NO predice por si solo\n(VP-Tree max-varianza)")
ax.set_ylim(-5, 100)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("resultados.png", dpi=150)
print("escrito resultados.png")
