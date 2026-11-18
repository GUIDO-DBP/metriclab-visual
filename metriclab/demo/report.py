#!/usr/bin/env python3
"""Genera un reporte HTML autocontenido con los resultados de la demo.

Un solo archivo .html (imagenes incrustadas) que se puede abrir en cualquier
navegador sin conexion y mostrar al docente.

Uso:  python3 report.py CARPETA_VECTORES salida.html [--sinteticos]
"""
import argparse
import base64
import csv
import html
import io
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def thumb(path, size=104):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_file(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def esc(s):
    return html.escape(str(s))


def bar(value, vmax, label):
    pct = 0 if vmax <= 0 else max(0.0, min(100.0, 100.0 * value / vmax))
    return (f'<div class="bar" title="{esc(label)}"><span style="width:{pct:.1f}%"></span>'
            f'<em>{esc(label)}</em></div>')


CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f3f2ef;--border:#e2e0da;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#77756f;--series-1:#2a78d6;
--good:#1a7f37;--bad:#c4302b;--warn-bg:#fff4d6;--warn-border:#e3b341}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242422;--border:#3a3936;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9a988f;--series-1:#3987e5;
--good:#3fb950;--bad:#f0605b;--warn-bg:#3a2f12;--warn-border:#a07c1d}}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5}
.viz-root{background:var(--surface-1);color:var(--text-primary);min-height:100vh}
main{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:20px;margin:48px 0 6px;padding-top:8px;border-top:1px solid var(--border)}
.sub{color:var(--text-secondary);margin:0 0 20px}
.lead{color:var(--text-secondary);max-width:760px}
.banner{background:var(--warn-bg);border:1px solid var(--warn-border);border-radius:8px;
padding:12px 16px;margin:20px 0;font-size:14px}
.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}
.step{background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:12px 14px;font-size:14px}
.step b{display:block;font-size:13px;color:var(--text-muted);font-weight:600;margin-bottom:2px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}
.kpi{border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.kpi .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
.kpi .l{font-size:13px;color:var(--text-secondary)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--text-secondary);border-bottom:1px solid var(--border);padding:8px 10px;font-size:13px}
td{border-bottom:1px solid var(--border);padding:7px 10px;vertical-align:middle}
td.num{text-align:right}
tr.off td{color:var(--text-muted)}
.bar{position:relative;height:20px;background:var(--surface-2);border-radius:4px;min-width:160px}
.bar span{position:absolute;left:0;top:0;bottom:0;background:var(--series-1);border-radius:0 4px 4px 0;opacity:.85}
.bar em{position:absolute;left:8px;top:0;line-height:20px;font-style:normal;font-size:12px;color:var(--text-primary);mix-blend-mode:normal}
.ok{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}
.gal{display:grid;gap:14px;margin-top:14px}
.row{display:grid;grid-template-columns:132px 1fr;gap:14px;align-items:start;padding:12px;border:1px solid var(--border);border-radius:10px}
.q img{border:3px solid var(--text-primary);border-radius:6px;display:block}
.q .t{font-size:12px;color:var(--text-secondary);margin-top:4px}
.res{display:flex;gap:10px;flex-wrap:wrap}
.res figure{margin:0;width:110px}
.res img{border-radius:6px;display:block;border:3px solid transparent}
.res .hit img{border-color:var(--good)}.res .miss img{border-color:var(--bad)}
.res figcaption{font-size:11.5px;color:var(--text-secondary);line-height:1.3;margin-top:3px}
.res .miss figcaption b{color:var(--bad)}
.legend{font-size:13px;color:var(--text-secondary);margin:6px 0}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin:0 4px 0 12px}
img.full{max-width:100%;border:1px solid var(--border);border-radius:8px;background:#fff}
.note{font-size:14px;color:var(--text-secondary);max-width:820px}
ul.honest li{margin:6px 0}
code{background:var(--surface-2);padding:1px 5px;border-radius:4px;font-size:13px}
.trace td{font-size:13px}
@media (max-width:760px){.flow,.kpis{grid-template-columns:1fr 1fr}.row{grid-template-columns:1fr}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vectores")
    ap.add_argument("salida")
    ap.add_argument("--sinteticos", action="store_true")
    ap.add_argument("--galeria", type=int, default=8)
    args = ap.parse_args()
    d = Path(args.vectores)

    train_meta = read_csv(d / "train_meta.csv")
    query_meta = read_csv(d / "query_meta.csv")
    metricas = read_csv(d / "metricas.csv")
    clasif = read_csv(d / "clasificacion.csv")
    resumen = json.loads((d / "resumen.json").read_text())
    trace = json.loads((d / "traza.json").read_text())

    neigh = defaultdict(list)
    for r in read_csv(d / "vecinos.csv"):
        neigh[int(r["consulta"])].append((int(r["rango"]), int(r["id_coleccion"]), float(r["distancia"])))

    k = resumen["k"]                                       # vecinos para galeria y k-NN
    k_search = max(len(v) for v in neigh.values())         # vecinos que busco el motor
    n_train = len(train_meta)
    tq_path = d / "traza_consulta.txt"
    trace_q = int(tq_path.read_text().strip()) if tq_path.exists() else 0

    # ---------------- KPIs ----------------
    vp = next(m for m in metricas if m["indice"] == "VP-Tree[max-varianza]")
    best_laesa = min((m for m in metricas if m["indice"].startswith("LAESA")),
                     key=lambda m: float(m["dist_por_consulta"]))
    ok_rows = [c for c in clasif if c["estado"] == "ok"]
    best_clf = max(ok_rows, key=lambda c: float(c["f1_macro"]))
    todos_ok = all(m["coincide_oraculo"] == "si" for m in metricas)

    # ---------------- galeria ----------------
    picked, seen = [], set()
    for qi, qm in enumerate(query_meta):
        if qm["categoria"] not in seen:
            seen.add(qm["categoria"]); picked.append(qi)
        if len(picked) >= args.galeria:
            break

    gal = []
    for qi in picked:
        qm = query_meta[qi]
        figs = []
        for rank, tid, dist in sorted(neigh[qi])[:k]:
            tm = train_meta[tid]
            hit = tm["categoria"] == qm["categoria"]
            figs.append(
                f'<figure class="{"hit" if hit else "miss"}"><img src="{thumb(tm["ruta"])}" width="104" height="104" alt="">'
                f'<figcaption>#{rank + 1} · <b>{esc(tm["categoria"])}</b><br>d = {dist:.3f}</figcaption></figure>')
        gal.append(
            f'<div class="row"><div class="q"><img src="{thumb(qm["ruta"], 120)}" width="120" height="120" alt="">'
            f'<div class="t">consulta: <b>{esc(qm["categoria"])}</b></div></div>'
            f'<div class="res">{"".join(figs)}</div></div>')

    # ---------------- tabla busqueda ----------------
    max_d = n_train
    srows = []
    for m in metricas:
        dq = float(m["dist_por_consulta"])
        srows.append(
            f'<tr><td>{esc(m["indice"])}</td>'
            f'<td>{bar(dq, max_d, f"{dq:.0f} de {n_train}")}</td>'
            f'<td class="num">{float(m["poda_pct"]):.1f} %</td>'
            f'<td class="num">{float(m["us_por_consulta"]):.0f}</td>'
            f'<td class="num">{int(float(m["dist_construccion"])):,}</td>'
            f'<td class="num">{float(m["memoria_kb"]):.1f}</td>'
            f'<td class="{"ok" if m["coincide_oraculo"] == "si" else "no"}">{"sí" if m["coincide_oraculo"] == "si" else "NO"}</td></tr>')

    # ---------------- tabla clasificacion ----------------
    crows = []
    for c in clasif:
        if c["estado"] != "ok":
            crows.append(f'<tr class="off"><td>{esc(c["modelo"])}</td><td colspan="4">pendiente: {esc(c["estado"])}</td></tr>')
            continue
        acc = float(c["accuracy"])
        nota = f'<br><span style="font-size:12px;color:var(--text-muted)">{esc(c["nota"])}</span>' if c.get("nota") else ""
        crows.append(
            f'<tr><td>{esc(c["modelo"])}{nota}</td>'
            f'<td>{bar(acc, 1.0, f"{acc * 100:.0f} %")}</td>'
            f'<td class="num">{float(c["balanced_accuracy"]) * 100:.0f} %</td>'
            f'<td class="num">{float(c["f1_macro"]):.3f}</td>'
            f'<td class="num">{float(c["entreno_s"]):.3f}</td></tr>')

    # ---------------- traza ----------------
    kind_es = {"visit": "visita nodo", "dist": "calcula distancia", "prune": "DESCARTA",
               "candidate": "nuevo candidato", "descend": "desciende"}
    # Se muestran los primeros pasos y luego una ventana alrededor de cada poda,
    # que es lo que interesa explicar.
    prune_pos = [i for i, e in enumerate(trace) if e["kind"] == "prune"]
    keep = set(range(min(6, len(trace))))
    for p in prune_pos[:4]:
        keep.update(range(max(0, p - 3), p + 1))
    trows, last = [], -1
    for i in sorted(keep):
        e = trace[i]
        if last >= 0 and i != last + 1:
            trows.append(f'<tr class="off"><td></td><td colspan="6">… {i - last - 1} pasos omitidos …</td></tr>')
        last = i
        tau = "∞" if e["tau"] is None else f'{e["tau"]:.3f}'
        obj = "" if e["object"] < 0 else (train_meta[e["object"]]["categoria"] + f' #{e["object"]}')
        dist = f'{e["dist"]:.3f}' if e["kind"] in ("dist", "candidate", "prune") and e["dist"] else ""
        style = ' style="font-weight:600"' if e["kind"] == "prune" else ""
        trows.append(f'<tr{style}><td class="num">{e["step"]}</td><td>{kind_es.get(e["kind"], e["kind"])}</td>'
                     f'<td class="num">{e["node"] if e["node"] >= 0 else ""}</td><td>{esc(obj)}</td>'
                     f'<td class="num">{dist}</td><td class="num">{tau}</td><td>{esc(e["reason"])}</td></tr>')
    n_prune = sum(1 for e in trace if e["kind"] == "prune")
    n_dist = sum(1 for e in trace if e["kind"] == "dist")

    pendientes = [c["modelo"] for c in clasif if c["estado"] != "ok"]
    banner = ""
    if args.sinteticos:
        banner = ('<div class="banner"><b>Datos de demostración.</b> Estas imágenes son sintéticas, '
                  'generadas con la misma estructura de carpetas que Caltech-101. Sirven para comprobar '
                  'que todo el flujo funciona; <b>no son resultados de Caltech-101</b> y no deben '
                  'citarse como tales en el informe.</div>')

    html_out = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MetricLab Visual · avance</title><style>{CSS}</style></head>
<body class="viz-root"><main>
<h1>MetricLab Visual</h1>
<p class="sub">Búsqueda de imágenes por similitud y clasificación · demostración del avance</p>
{banner}

<div class="flow">
 <div class="step"><b>1 · Imagen</b>Se lee la foto y se reduce a 128×128.</div>
 <div class="step"><b>2 · Vector</b>144 números: color HSV (128) + orientación de bordes (16).</div>
 <div class="step"><b>3 · Motor C++</b>Secuencial, VP-Tree y LAESA buscan los {k_search} más parecidos.</div>
 <div class="step"><b>4 · Clasificación</b>Árbol de decisión, Random Forest, boosting y k-NN predicen la categoría.</div>
</div>

<div class="kpis">
 <div class="kpi"><div class="v">{n_train} / {len(query_meta)}</div><div class="l">imágenes indexadas / consultas, {len(resumen["categorias"])} categorías</div></div>
 <div class="kpi"><div class="v">{resumen["precision_at_k"] * 100:.0f} %</div><div class="l">Precision@{k} de la búsqueda (promedio por categoría)</div></div>
 <div class="kpi"><div class="v">{float(best_laesa["poda_pct"]):.0f} %</div><div class="l">comparaciones evitadas por el mejor índice ({esc(best_laesa["indice"])})</div></div>
 <div class="kpi"><div class="v">{float(best_clf["f1_macro"]):.2f}</div><div class="l">mejor F1 macro: {esc(best_clf["modelo"])}</div></div>
</div>

<h2>Búsqueda por similitud</h2>
<p class="lead">Cada fila es una consulta que <b>nunca estuvo en el índice</b>. A la derecha, los {k} primeros resultados
del VP-Tree en orden de distancia. El borde indica si el resultado es de la misma categoría que la consulta.</p>
<div class="legend">borde:<i style="background:var(--good)"></i>misma categoría<i style="background:var(--bad)"></i>otra categoría</div>
<div class="gal">{"".join(gal)}</div>

<h2>Rendimiento de los índices</h2>
<p class="lead">La medida principal es cuántas distancias se calculan por consulta: no depende de la máquina.
La búsqueda secuencial siempre calcula {n_train}. La última columna confirma que cada índice devuelve
<b>exactamente</b> los mismos vecinos que la búsqueda secuencial (prueba del oráculo).</p>
<table><thead><tr><th>Índice</th><th>Distancias por consulta</th><th style="text-align:right">Evitadas</th>
<th style="text-align:right">µs/consulta</th><th style="text-align:right">Distancias al construir</th>
<th style="text-align:right">Memoria KB</th><th>Oráculo</th></tr></thead><tbody>{"".join(srows)}</tbody></table>
<p class="note">Observación: con solo {n_train} imágenes el VP-Tree descarta poco
({float(vp["poda_pct"]):.1f} %), mientras LAESA llega a {float(best_laesa["poda_pct"]):.0f} %. En mediciones
anteriores con datos controlados, la poda del VP-Tree pasó de 53 % con 1 000 objetos a 82 % con 10 000.
Caltech-101 tiene unas veinte veces más imágenes que esta demo, así que este es el punto que hay que volver a medir con datos reales.
Los microsegundos varían entre ejecuciones; por eso no son la medida principal.</p>

<h2>Clasificación</h2>
<p class="lead">Los mismos vectores. Se entrena con las {resumen["n_train"]} imágenes de la colección y se evalúa
con las {resumen["n_test"]} consultas. La exactitud balanceada y el F1 macro dan el mismo peso a cada
categoría, porque las categorías tienen tamaños distintos.</p>
<table><thead><tr><th>Modelo</th><th>Exactitud</th><th style="text-align:right">Exactitud balanceada</th>
<th style="text-align:right">F1 macro</th><th style="text-align:right">Entrenamiento (s)</th></tr></thead>
<tbody>{"".join(crows)}</tbody></table>
<p class="note">El k-NN no se entrena: toma los vecinos que encontró el VP-Tree y vota. Es el punto donde la
clasificación y el índice métrico se conectan.</p>
<img class="full" src="{b64_file(d / "matrices_confusion.png", "image/png")}" alt="Matrices de confusión por modelo">
<p class="note">Matrices de confusión: filas = categoría real, columnas = categoría predicha. La diagonal son aciertos.</p>

<h2>Recorrido del VP-Tree paso a paso</h2>
<p class="lead">Búsqueda de los {k_search} vecinos de una consulta de la categoría
<b>{esc(query_meta[trace_q]["categoria"])}</b>, tal como la reportó el motor C++ (<code>traza.json</code>).
Se eligió la consulta donde el árbol descartó más regiones: calculó {n_dist} distancias y descartó {n_prune} subárboles.
τ es la distancia al peor de los {k_search} mejores encontrados hasta ese momento; mientras vale ∞ no se puede
descartar nada, y cuanto antes baja, más se poda. La regla viene de la desigualdad triangular: si la consulta
está a distancia d del pivote y el corte del nodo es μ, el hijo interno se descarta cuando d − τ ≥ μ y el externo cuando d + τ &lt; μ.
Esta traza es la que usará la visualización interactiva de la página final.</p>
<table class="trace"><thead><tr><th>Paso</th><th>Acción</th><th>Nodo</th><th>Objeto</th><th style="text-align:right">Distancia</th>
<th style="text-align:right">τ</th><th>Motivo</th></tr></thead><tbody>{"".join(trows)}</tbody></table>

<h2>Qué falta y qué consultar</h2>
<ul class="honest">
 <li>{"Todos los índices coinciden con el oráculo." if todos_ok else "<b>Algún índice NO coincide con el oráculo: revisar antes de usar estas cifras.</b>"}</li>
 {"<li>Pendiente de ejecutar: " + ", ".join(esc(p) for p in pendientes) + ". El código ya está; solo requiere instalarlos.</li>" if pendientes else ""}
 <li>Pasar de estas imágenes de prueba a Caltech-101: el mismo comando, cambiando la carpeta de entrada.</li>
 <li>Conectar el motor con pybind11 y construir la página interactiva (subir imagen, elegir método, ver el recorrido).</li>
 <li>Para el docente: ¿los clasificadores deben implementarse desde cero o basta con bibliotecas? ¿Se esperan los cuatro?</li>
</ul>
</main></body></html>"""

    Path(args.salida).write_text(html_out, encoding="utf-8")
    print(f"escrito {args.salida} ({Path(args.salida).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
