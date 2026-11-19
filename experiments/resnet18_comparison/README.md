# Extensión independiente: cuatro clasificadores ResNet-18

Reutiliza **los archivos existentes** `train_resnet18.npy` y `query_resnet18.npy` del experimento padre. No extrae otra vez, no modifica ResNet, no cambia la división y no activa modelos en FastAPI. RF/XGBoost completos se conservan en lectura; solo se añaden los ajustes completos Gini, entropía y LightGBM.

## Metodología

- 6.941 imágenes de entrenamiento, 1.736 de prueba histórica, 101 clases, descriptor congelado oficial de 512 dimensiones.
- Cinco folds de `StratifiedKFold`, shuffle, semilla 23, **solo dentro de las 6.941 imágenes de entrenamiento**. Mismos folds para los cuatro algoritmos y ambos criterios de Decision Tree.
- Cada fold construye un estimador nuevo y lo ajusta con las filas de entrenamiento de ese fold. Los modelos completos RF/XGBoost anteriores nunca producen las predicciones OOF.
- Semilla de modelos 0. RF/XGBoost usan exactamente sus parámetros ResNet guardados; Gini/entropía y LightGBM copian las configuraciones fijas históricas, con dos hilos como máximo. Sin búsqueda, selección de características ni early stopping.
- Regla definida antes de calcular resultados: mayor **F1 macro medio de validación interna**, desempate por balanced accuracy y nombre. La propuesta se congela antes de evaluar los modelos nuevos sobre prueba. No se activa automáticamente.
- Se reportan media y desviación muestral por fold, métricas OOF y la comparación de prueba **reutilizada** por separado. La desviación no es un intervalo de confianza y los folds no son réplicas independientes de entrenamiento.
- El descriptor ya fue explorado anteriormente mirando la prueba histórica. Esta CV evalúa estabilidad condicional a ese descriptor; no es una evaluación externa ni una corrección retroactiva de esa exploración.

## Ejecutar en Windows

Se utiliza el entorno experimental de la fase anterior:

```powershell
.\runtime\resnet-env\Scripts\python.exe -m pip install -r experiments/resnet18_comparison/requirements.txt
.\runtime\resnet-env\Scripts\python.exe -m experiments.resnet18_comparison.run --parent runtime/experiments/resnet18/20261118T195917Z-42d3213e
```

No hace falta CUDA ni PyTorch para esta etapa: los embeddings ya están calculados. Los ajustes de los clasificadores son CPU con dos hilos. El entorno de la aplicación `.venv` y sus dependencias no se modifican.

Cada ejecución crea un directorio nuevo en `runtime/experiments/resnet18_comparison/<run_id>`. Para reanudar una ejecución interrumpida:

```powershell
.\runtime\resnet-env\Scripts\python.exe -m experiments.resnet18_comparison.run --run runtime/experiments/resnet18_comparison/<run_id>
```

No cambia parámetros al retomar. Los checkpoints completos no se sobrescriben. Los ajustes se ejecutan secuencialmente para limitar RAM. Los modelos CV no se conservan como grandes bundles: se guardan sus índices, parámetros y predicciones para reproducibilidad. Los tres modelos completos nuevos sí se guardan y verifican mediante recarga.

## Artefactos

```text
protocol.json                    parámetros, semillas y regla de recomendación previa
folds.json                       índices exactos dentro de TRAIN, para los cinco folds
protected_files.json             hashes de aplicación/modelos HSV activos
parent_files.json                hashes de todos los artefactos del experimento padre
cv/<modelo>/fold_N.json           métricas, tiempos y hashes por fold
cv/<modelo>/fold_N_predictions.npy
cv/<modelo>/oof_predictions.npy   6941 predicciones, cada fila fuera de su entrenamiento
cv/<modelo>/summary.json          medias, desviaciones y métricas OOF
cv_selection.json                recomendación desde CV, congelada antes de prueba
final/<modelo>/results.json       evaluación descriptiva de prueba histórica
final/<modelo>/predictions.npy    1736 predicciones
final/{gini,entropy,lightgbm}/model.joblib
final/lightgbm/lightgbm.txt       formato nativo adicional
model_catalog_proposal.json      inventario para futura integración, NO activo
results.json / comparison CSVs / REPORT.md / cv_stability.png
completion.json                  integridad y no activación
```

Los tiempos de entrenamiento de RF/XGBoost completos proceden del experimento padre, claramente marcados; no se reentrenan esos modelos. La inferencia de los cinco modelos se mide otra vez en esta etapa con 128 consultas unitarias a `predict_proba`, tras calentamiento y bajo dos hilos, sin CNN, IO ni HTTP.

## Verificación y plan

Las pruebas de la extensión se ejecutan después de completar una ejecución:

```powershell
.\runtime\resnet-env\Scripts\python.exe -m unittest experiments.resnet18_comparison.test_comparison -v
.\runtime\resnet-env\Scripts\python.exe -m experiments.resnet18_comparison.verify_system --run runtime/experiments/resnet18_comparison/<run_id>
```

El [plan de integración](PLAN_INTEGRACION.md) detalla el registro de descriptores, validación de dimensiones, carga diferida CPU/CUDA, selector en la página, galería de **imágenes reales del entrenamiento** por categoría predicha y los cambios necesarios en métricas/explorador. No implementa cambios en la aplicación principal.

El verificador ejecuta las pruebas existentes del backend, el oráculo C++ y el build del frontend; además comprueba HTTP 200 de la página y de una imagen real de entrenamiento mediante la API actual. Confirma al terminar los hashes de aplicación y experimento padre y guarda los logs. No inicia ni sustituye un servidor del usuario.

Fuentes: [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), [evaluación y selección de modelos](https://scikit-learn.org/stable/modules/cross_validation.html).
