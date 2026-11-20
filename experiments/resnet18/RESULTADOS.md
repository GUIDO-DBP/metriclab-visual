# Resultados verificados del experimento ResNet-18

> Los artefactos `runtime/` indicados abajo son rutas locales, no archivos del árbol GitHub. Se obtienen con el paquete `research` de la [Release resources-v1](https://github.com/GUIDO-DBP/metriclab-visual/releases/tag/resources-v1); véase la [instalación](../../docs/INSTALACION_WINDOWS.md). Las cifras y protocolos originales se conservan.

Ejecucion: `20261118T195917Z-42d3213e`. Resultado independiente; la aplicacion y el frontend siguen con HSV + Sobel.

# Experimento independiente: ResNet-18 frente a HSV + Sobel

Dispositivo de extracción: **cuda** (NVIDIA GeForce RTX 4060 Laptop GPU).
División original: **6.941 entrenamiento / 1.736 prueba / 101 clases**, sin cambios.
CNN preentrenada y congelada: `ResNet18_Weights.IMAGENET1K_V1`, `weights.transforms()`, `fc=Identity`, 512 dimensiones FP32.

| Descriptor | Modelo | Accuracy | Balanced accuracy | F1 macro | Entrenamiento (s) | Inferencia clasificador (ms/imagen) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HSV + Sobel (144D) | Random Forest | 46.49 % | 24.77 % | 0.2585 | 7.560 | 24.456 |
| ResNet-18 (512D) | Random Forest | 86.12 % | 77.41 % | 0.7903 | 36.733 | 26.031 |
| HSV + Sobel (144D) | XGBoost | 48.33 % | 27.80 % | 0.2850 | 13.478 | 1.462 |
| ResNet-18 (512D) | XGBoost | 87.15 % | 80.55 % | 0.8184 | 42.869 | 1.374 |

Tiempos de entrenamiento e inferencia de las cuatro filas medidos de nuevo en este experimento; mismas semillas y parámetros por clasificador, salvo dimensión de entrada. Hasta dos hilos CPU por defecto.
Inferencia: media de 128 llamadas unitarias a predict_proba tras calentamiento; excluye extractor, imágenes, HTTP y JSON.

Extracción individual ResNet-18 (imagen + preprocesamiento oficial + transferencia + CNN): **7.276 ms** de media; HSV/Sobel: **1.965 ms**. La inferencia del clasificador en la tabla no incluye este costo.
Extracción masiva de los 8.677 embeddings: **53.58 s**, batch 8; excluye descarga, carga y calentamiento inicial del modelo.

## Referencia histórica y reproducibilidad
Los resultados guardados de la aplicación permanecen intactos:
- Random Forest HSV original: accuracy 46.49 %, balanced accuracy 24.77 %, F1 macro 0.2585.
- XGBoost HSV original: accuracy 48.79 %, balanced accuracy 28.21 %, F1 macro 0.2914.
La ejecución CPU también se comprobó: dos filas reales coinciden con los vectores guardados (error máximo 0.00000739). En una comprobación funcional de 16 imágenes con dos hilos, la extracción CPU tomó 93.385 ms de media. Esta muestra CPU es más pequeña que el benchmark GPU y no se trata como una comparación de latencia exhaustiva.
La referencia XGBoost original utilizó 4 hilos y la comparación pareada actual limita ambos descriptores a 2. La réplica HSV actual da 48.33 %; no se la sustituye por el 48.79 % anterior.
Una auditoría adicional repitió los parámetros históricos sin cambios (incluidos los hilos): obtuvo 48.79 % y 0 predicciones diferentes frente al modelo original. No se seleccionaron hiperparámetros ni se cambió la configuración del experimento para elevar la métrica.

## Cambio de precisión
- Random Forest: accuracy **+39.63 puntos porcentuales**; balanced accuracy +52.63 puntos; F1 macro +0.5319. Corrige 705 predicciones anteriores y empeora 17.
- XGBoost: accuracy **+38.82 puntos porcentuales**; balanced accuracy +52.75 puntos; F1 macro +0.5334. Corrige 709 predicciones anteriores y empeora 35.
- Frente al XGBoost histórico de 48.79 %, el nuevo descriptor mejora **+38.36 puntos** de accuracy; sus tiempos históricos no se tratan como mediciones simultáneas.

## Interpretación y recomendación
Las activaciones preentrenadas capturan patrones visuales aprendidos en ImageNet, mientras HSV/Sobel describe principalmente color y orientación de bordes. Las métricas macro permiten valorar las clases pequeñas.
La mejor configuración ResNet de este experimento es XGBoost, con 87.15 % de accuracy.
Se recomienda integrar posteriormente ResNet-18 como descriptor alternativo para clasificación: mejora accuracy, balanced accuracy y F1 macro con ambos clasificadores en esta prueba. XGBoost ofrece aquí el mejor accuracy y F1 macro. Mantener HSV como referencia, registrar la nueva identidad de descriptor y asociar modelos de 512D sin mezclar los artefactos de 144D. Confirmar después con validación independiente. No se ha activado el descriptor ni modificado el frontend.

## Verificación y límites
- 66 archivos protegidos de aplicación, frontend, separación y modelos activos conservan su SHA-256.
- Mismo orden de archivos y etiquetas, sin imágenes descartadas; embeddings finitos de (6941,512) y (1736,512).
- Las 8.677 imágenes tienen SHA-256 registrado. Se verificó la alineación de todas las etiquetas y se reextrajeron seis filas reales individualmente para compararlas con su vector por lote.
- Ningún parámetro de la CNN fue entrenable; el estado de pesos y BatchNorm tiene el mismo hash antes y después.
- Métricas recalculadas desde las 1.736 predicciones guardadas; modelos guardados/recargados con probabilidades equivalentes.
- Los parámetros se copiaron de la referencia antes de evaluar. No se usó prueba para fit, early stopping, selección de características ni ajuste de hiperparámetros.
- Contenidos de imagen idénticos entre particiones de la división conservada: 0.
- Una sola división fija, sin validación cruzada ni calibración. La prueba ya se había utilizado en el proyecto: para confirmar una integración futura se recomienda validación dentro de entrenamiento y una evaluación independiente final.
- ResNet fue preentrenada en ImageNet; no se auditó posible solapamiento de imágenes con ese corpus de origen.

Fuentes oficiales: [ResNet-18 y transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [arquitectura torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).


## Reproducir y consultar artefactos

- [Gu?a y comandos del experimento](README.md).
- Resultados completos JSON: `runtime/experiments/resnet18/20261118T195917Z-42d3213e/results.json`.
- Comparaci?n CSV: `runtime/experiments/resnet18/20261118T195917Z-42d3213e/comparison.csv`.
- Reporte de integridad: `runtime/experiments/resnet18/20261118T195917Z-42d3213e/completion.json`.
- Auditor?a de reproducibilidad hist?rica: `runtime/experiments/resnet18/20261118T195917Z-42d3213e/historical_reproducibility_audit.json`.

Comparaci?n calculada: `runtime/experiments/resnet18/20261118T195917Z-42d3213e/comparison.png`

## Pruebas realizadas

Comando: `runtime/resnet-env/Scripts/python.exe -m unittest experiments.resnet18.test_experiment -v`.

**Cinco pruebas aprobadas**: integridad de aplicaci?n/modelos activos; ejecuci?n CPU y correspondencia con embeddings GPU; separaci?n y alineaci?n de todas las filas; m?tricas reales y par?metros comparables; red preentrenada congelada y verificaciones de imagen. La CNN no se entren? ni se afin?.

Las comprobaciones no cambian los modelos principales. Los artefactos grandes permanecen bajo `runtime/`, ya excluido de Git, y el c?digo de este experimento queda bajo `experiments/resnet18/`.
