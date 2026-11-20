# Metodología y resultados conservados

Se usan las mismas 6.941 imágenes de entrenamiento y 1.736 de prueba de Caltech-101, 101 clases sin fondo. HSV + Sobel produce 144 características; ResNet-18 congelada produce el embedding de 512 componentes previo a la clasificación con pesos/preprocesamiento oficiales. No se atribuye significado visual individual a esos componentes.

**La prueba de 1.736 imágenes es histórica y fue reutilizada.** Ni la reorganización ni su presentación constituyen una evaluación independiente nueva. Las tablas comparan ejecuciones versionadas; no son mediciones simultáneas ni demuestran superioridad estadística. La réplica HSV/XGBoost de dos hilos obtuvo 48,33 %, separada del 48,79 % del modelo activo.

| Evidencia conservada | Contenido |
| --- | --- |
| [Resumen científico](../resources/experiment_summary.json) | Métricas, parámetros, versiones y validación. |
| [HSV](PRUEBAS_CLASIFICACION.md) | Entrenamiento original, cinco configuraciones y métricas por categoría. |
| [ResNet inicial](../experiments/resnet18/RESULTADOS.md) | Embeddings, etiquetas, Random Forest y XGBoost frente a HSV. |
| [Comparación ResNet](../experiments/resnet18_comparison/RESULTADOS.md) | Árboles Gini/entropía, LightGBM inicial y validación cruzada interna. |
| [Diagnóstico LightGBM](../experiments/lightgbm_resnet/RESULTADOS.md) | Siete configuraciones, regularización, confirmación reservada y modelo final. |
| [Paquete research](https://github.com/GUIDO-DBP/metriclab-visual/releases/tag/resources-v1) | Embeddings, etiquetas, predicciones, curvas e inventarios con SHA256. |

La comparación ResNet original usó cinco folds internos. El diagnóstico LightGBM reservó 1.389 imágenes de entrenamiento como confirmación y buscó siete configuraciones con tres folds en las otras 5.552; seleccionó antes de abrir confirmación o consultar prueba histórica. Son protocolos distintos, no una CV común para todos los modelos finales. No se dispone de CV HSV equivalente en estos artefactos.

LightGBM inicial obtuvo accuracy 19,76 % y F1 macro 0,1139. La versión regularizada obtuvo en confirmación interna 87,62 % y 0,8224; después se ajustó con las 6.941 imágenes y se evaluó sobre la prueba histórica: 87,33 % y 0,8160. Ambas versiones se conservan. La limpieza no entrena ni cambia resultados.

Accuracy mide la proporción de aciertos. Balanced accuracy promedia recall por categoría; F1 macro promedia F1 por categoría. La matriz tiene filas verdaderas y columnas predichas; la diagonal cuenta aciertos. Los resúmenes de recall, soporte y confusiones se calculan desde matrices reales.

Interfaz e informes conservan accuracy, balanced accuracy, F1 macro, ajuste e inferencia. El tiempo del clasificador excluye extracción; la primera consulta ResNet incluye cargar el worker/pesos. Los tiempos históricos dependen de entorno, hardware e hilos: no son un benchmark uniforme del sistema completo.

Probabilidades sin calibrar, sin rechazo de desconocidos; no se auditó posible solapamiento con ImageNet. Una evaluación independiente futura requiere datos no utilizados en los experimentos anteriores. Los [manuales e informe preliminar](ARCHIVO_HISTORICO.md) no son el informe definitivo.
