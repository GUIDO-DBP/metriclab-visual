# Auditoría del workspace — 16 de noviembre de 2026

Se inspeccionaron los fuentes, headers, pruebas, extractor, clasificadores, scripts y el texto del reporte HTML antes de integrar la aplicación.

| Componente original | Hallazgo y decisión |
| --- | --- |
| `metriclab/include` y `src` | Interfaz común MetricIndex, métrica instrumentada, búsqueda secuencial, VP-Tree con cuatro políticas de pivote y LAESA. Se reutilizan sin reemplazar los algoritmos. |
| `metriclab/tests/oracle_test.cpp` | Pruebas de k-NN y rango sobre datos agrupados, uniformes, casi duplicados y distintas dimensiones; verificación triangular. Compila y pasa. |
| `metriclab/apps/search_cli.cpp` | CLI de archivos MLV1; construye ocho configuraciones y exporta CSV y una traza de ejemplo. Se conserva. El puente nuevo evita reconstruir todo por cada petición web. |
| `metriclab/demo/extract_features.py` | Descriptor 144D HSV + bordes, normalización independiente y tamaño 128 × 128. Se importan directamente sus funciones, sin nueva fórmula incompatible. |
| `metriclab/demo/classify.py` | Constructores sklearn, XGBoost/LightGBM opcionales, k-NN a partir del CSV. Se reutilizan sus constructores; la web obtiene vecinos directamente del índice y guarda modelos para inferencia posterior. |
| `MetricLab_reporte_demo.html` | Demostración sintética: 400 indexadas, 100 consultas, ocho categorías. Sus métricas, miniaturas y traza no son resultados de Caltech-101. Se conserva sin alteración y no se importan sus números a la web. |
| `dataset/101_ObjectCategories` | 9.145 archivos de imagen incluyendo BACKGROUND_Google. Al excluir esa carpeta: 8.677 imágenes válidas de 101 clases; partición nueva reproducible 6.941 / 1.736. |
| Modelos / vectores guardados | No existían al iniciar. Se generaron en `runtime/data/`. |
| Entorno | Python, Node y g++ disponibles. FastAPI y OpenCV ausentes inicialmente; instalados en `.venv`. No se encontró CMake en PATH. PostgreSQL no configurado ni probado. |

La DLL nueva expone creación/liberación, búsqueda y topología a través de una ABI C; Python usa ctypes. La extensión `VpTree::writeStructure` es de lectura y no cambia la construcción ni la búsqueda. Se añadió un target CMake opcional manteniendo los ejecutables anteriores.

Los resultados de la aplicación provienen exclusivamente de vectores procesados y de ejecuciones del motor/modelos. La interfaz muestra indisponibilidad cuando no existe un componente, sin datos de relleno.
