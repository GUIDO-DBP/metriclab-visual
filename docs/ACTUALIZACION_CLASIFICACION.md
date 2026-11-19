# Actualización de segunda unidad — 18 de noviembre de 2026

## Auditoría antes de modificar

Se revisaron frontend, backend, DLL/fuentes C++, demos, preparación, modelos guardados, imágenes, pruebas y dependencias. La versión de partida pasó las 10.503 comprobaciones C++, sus siete pruebas de integración y la compilación TypeScript/Vite. No existía repositorio Git; se creó un respaldo en el commit `621253d` y se trabajó en `feature/clasificacion-u2`.

El respaldo `runtime/backups/antes-clasificacion-2026-11-18.zip` conserva manifiesto, vectores, modelos y evaluación anteriores. El HTML sintético y el código/experimentos de `metriclab/` no se eliminaron ni cambiaron.

Hallazgos comprobados:

- `backend/core.py` importa las funciones del extractor original `metriclab/demo/extract_features.py`. Las cargas HTTP producen el mismo vector que el extractor de archivos.
- Los árboles guardados son `sklearn.tree.DecisionTreeClassifier` con Gini/entropía, semilla 0 y configuraciones comparables. Random Forest tiene 200 estimadores.
- La preparación tiene 6.941 filas de entrenamiento y 1.736 de prueba, 144 columnas y 101 categorías. Rutas disjuntas, valores finitos y SHA-256 de vectores/etiquetas/partición verificado contra el manifiesto. Se mantuvo esa división sin reextraer características.
- Las bibliotecas XGBoost y LightGBM no estaban instaladas. Se instalaron sus paquetes oficiales y se verificaron entrenando y prediciendo.
- Los metadatos previos no incluían precision/recall macro, latencia ni parámetros/versiones completos. Se incorporaron en una nueva evaluación, conservando los artefactos históricos.

La independencia comprobada es por archivo/ruta y por exclusión del conjunto de prueba de `fit`. No se realizó una auditoría de imágenes visualmente duplicadas o de contenido parecido entre clases de Caltech.

## Organización actual

La navegación principal contiene Inicio, Clasificar imagen, Comparación de clasificadores, Explorador de árboles y Dataset/metodología. Una entrada secundaria conserva **Laboratorio de búsqueda métrica — Trabajo complementario**.

La carga, galería, modal, estadísticas, descriptor y visualizador de VP-Tree se extrajeron/reutilizaron como componentes. La aplicación sigue en el mismo proyecto, API y script de inicio; no se creó otro producto.

| Archivo | Cambio |
| --- | --- |
| `frontend/src/main.tsx`, `App.tsx` | Punto de entrada ligero, navegación centrada en clasificación, cuatro tarjetas, carga reutilizada y comparación sobre la misma fotografía |
| `frontend/src/components.tsx`, `api.ts`, `types.ts` | Controles reutilizados, contrato de API y tipos compartidos |
| `frontend/src/ClassifierEvaluation.tsx` | Cinco configuraciones agrupadas en cuatro algoritmos, métricas completas, matrices filtrables y descarga JSON |
| `frontend/src/ClassifierExplorer.tsx` | Nodos reales, ramas, clases/pesos, Gini/entropía, diez árboles RF y rutas de imagen; importancias reales para los cuatro algoritmos |
| `frontend/src/MetricLaboratory.tsx`, `MetricTree.tsx` | Búsqueda/benchmark/VP-Tree/k-NN conservados como complemento |
| `frontend/src/classification.css`, `index.html` | Extensión de la identidad azul/violeta y diseño responsive; estilos originales conservados |
| `backend/main.py` | Modelos persistentes, inferencia medida, comparación de cuatro con una extracción, estados independientes de C++ y endpoints de explicación |
| `backend/modeling.py` | Selección de artefactos activos, nombres de características, nodos y rutas leídos de los modelos |
| `scripts/train_classifiers.py` | Entrenamiento/evaluación versionados, reutilización de DT/RF, formatos nativos y verificación de persistencia |
| `scripts/prepare.py` | Conserva extracción y división; delega entrenamiento al pipeline único |
| `scripts/start.ps1` | Mantiene el comando de inicio; C++ deja de bloquear clasificación; propaga fallos de inicio |
| `scripts/import_postgres.py` | Importa la evaluación activa en lugar de confundirla con la histórica |
| `requirements*.txt` | Boosting pasa a dependencias principales; PostgreSQL sigue opcional; versiones verificadas registradas |
| `tests/`, `frontend/e2e/` | Conservación/adaptación de pruebas anteriores y ampliación para los cuatro clasificadores |
| `README.md`, documentación nueva | Instalación Windows, guion de exposición, arquitectura, resultados y límites actualizados |

## Entrenamiento y persistencia

Los modelos sklearn válidos anteriores se reutilizaron. En el bosque se limitó `n_jobs` para la ejecución actual; los parámetros originales del entrenamiento se conservan por separado. No se reentrena al predecir.

XGBoost usa clasificación multiclase `multi:softprob`, 80 etapas, profundidad máxima 4, histogramas con 64 bins, tasa 0,1, submuestreo y columnas de 0,9. LightGBM usa `multiclass`, 80 etapas, 15 hojas, profundidad máxima 6, 63 bins, tasa 0,1 y modo determinista por columnas. Semilla de modelos 0 y hasta cuatro hilos. Son parámetros fijos de recursos razonables, no un ajuste guiado por prueba.

Los resultados de cinco configuraciones se guardan en una ejecución versionada. El lector activo utiliza `active_classifiers.json`; por compatibilidad puede leer el formato raíz de la entrega anterior cuando no existe ese archivo. La huella y el descriptor impiden cargar un modelo de otra preparación. El identificador de ejecución evita mezclar una evaluación con otros modelos.

Antes de activar un entrenamiento se guardan y recargan los modelos y se comparan sus probabilidades. Se incluyen `xgboost.json` y `lightgbm.txt` nativos además del bundle joblib. Los metadatos se serializan como JSON estricto: el valor especial `missing=NaN` de XGBoost queda descrito como texto y no provoca fallos de API.

## Contratos de la interfaz

`/api/classify` extrae una vez el vector. `compare=true` ejecuta el criterio elegido de Decision Tree más RF, XGBoost y LightGBM. k-NN queda fuera de ese grupo y se ejecuta solo si se solicita explícitamente. Se devuelven todas las probabilidades, no porcentajes creados en el frontend.

La tabla informa coincidencias y desacuerdos, sin suponer una etiqueta verdadera para una imagen cargada. La confianza de cada modelo está sin calibrar. Los tiempos de inferencia miden únicamente `predict_proba`; extraer el vector y explicar rutas quedan fuera de la medida.

El visor recibe un subárbol real de hasta 31 nodos mediante `max_depth<=4` (la interfaz solicita 15 nodos con profundidad 3). Cada nodo contiene IDs reales de hijos, característica, umbral, impureza, número de muestras, peso y distribución de clases. Es posible abrir una rama y remontar, con lo que se evita renderizar miles de nodos simultáneamente.

La ruta se obtiene con `decision_path`; no se dibujan decisiones de entrenamiento ficticias. En RF corresponde al árbol seleccionado, no al resultado agregado. Las proporciones de clase se convierten a pesos con `weighted_n_node_samples`, manteniendo la interpretación correcta de bootstrap.

Para boosting se muestran etapas/árboles reales y las importancias calculadas por el modelo. No se inventa una ruta única de un ensamble multiclase. Las importancias son ganancia (boosting) o reducción de impureza (DT/RF), normalizadas dentro de cada modelo.

## Conservación y límites

La búsqueda conserva sus algoritmos C++, contadores, verificación secuencial, topología y trazas. Su disponibilidad se informa en `metric_ready`, separada de `classification_ready`. No se añadieron índices nuevos.

La clasificación sigue limitada por histogramas de color y bordes. No hay validación cruzada, calibración ni selección de hiperparámetros. Las imágenes de prueba no intervienen en esas decisiones. Los conteos son muy desiguales (25–640 entrenamientos por clase), por eso se informan promedios macro.

PostgreSQL continúa como migración opcional sin instancia probada. Los nuevos descriptores requieren otra identidad de preparación y otros modelos. No se incluyeron para retrasar la prioridad de los cuatro clasificadores.
