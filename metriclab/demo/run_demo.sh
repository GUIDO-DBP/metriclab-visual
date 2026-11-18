#!/usr/bin/env bash
# Ejecuta todo el flujo de MetricLab Visual de punta a punta.
#
#   ./run_demo.sh                              -> imagenes de prueba sinteticas
#   ./run_demo.sh /ruta/101_ObjectCategories   -> Caltech-101 real
#
# Variables opcionales:
#   MAX_POR_CLASE=80 ./run_demo.sh /ruta/...   limita imagenes por categoria
set -euo pipefail
cd "$(dirname "$0")"

IMG="${1:-}"
OUT=work/vectores
FLAGS=""
EXTRA=""

if [ -z "$IMG" ]; then
  IMG=work/imagenes
  FLAGS="--sinteticos"
  [ -d "$IMG" ] || python3 make_demo_images.py "$IMG"
fi
[ -n "${MAX_POR_CLASE:-}" ] && EXTRA="--max-per-class $MAX_POR_CLASE"

echo "== 1/4 compilando el motor C++"
cmake -S .. -B ../build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build ../build -j4 >/dev/null

echo "== 2/4 extrayendo vectores de $IMG"
mkdir -p "$OUT"
python3 extract_features.py "$IMG" "$OUT" $EXTRA

echo "== 3/4 busqueda con el motor C++"
../build/search_cli "$OUT/train.bin" "$OUT/query.bin" "$OUT" 10

echo "== 4/4 clasificacion"
python3 classify.py "$OUT"

python3 report.py "$OUT" work/reporte.html $FLAGS
echo
echo "Listo. Abre demo/work/reporte.html en el navegador."
