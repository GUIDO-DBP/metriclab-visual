# Diagnóstico y ajuste controlado de LightGBM + ResNet-18

Ejecución real: `runtime/experiments/lightgbm_resnet/20261118T211652Z-600d0ec2`.
Consulta [RESULTADOS.md](RESULTADOS.md) para el diagnóstico, tablas y limitaciones.

```powershell
.\runtime\resnet-env\Scripts\python.exe -m experiments.lightgbm_resnet.run --comparison runtime/experiments/resnet18_comparison/20261118T204643Z-8efb68c2
```

Cada ejecución crea un directorio nuevo. No reextrae embeddings ni modifica los modelos anteriores. Verifica vectores, etiquetas, división y hashes de imágenes originales. Separa 5.552 registros de desarrollo y 1.389 de confirmación interna reservada mediante estratificación y semilla 41. Busca siete configuraciones prefijadas con tres folds estratificados en desarrollo (semilla 29), seleccionando F1 macro medio y balanced accuracy como desempate. Guarda la selección antes de abrir la confirmación o medir la prueba histórica. No usa early stopping ni el test para elegir etapas.

`protocol.json` contiene parámetros e índices; `search.json`, métricas de cada fold; `selection.json`, elección congelada; `internal_confirmation.json`, predicciones/distribuciones/métricas por clase; `diagnosis.json`, curvas parciales y diagnóstico numérico. Se conservan logs y curvas completas por ajuste, predicciones `.npy`, modelo joblib y modelo nativo. `results.json` identifica expresamente la prueba histórica reutilizada. El coste de la búsqueda no es el tiempo de entrenamiento final.

La reproducibilidad del método y las métricas guardadas se verifica con `tests/test_lightgbm_protocol.py`. Las semillas no garantizan tiempos idénticos entre equipos.
