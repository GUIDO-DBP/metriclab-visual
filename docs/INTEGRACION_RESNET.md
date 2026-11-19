# Integración reversible de ResNet-18

Se implementó el alcance autorizado de `experiments/resnet18_comparison/PLAN_INTEGRACION.md` sobre la aplicación existente. El plan histórico se conserva como documento de su experimento; este archivo describe el estado posterior a la integración.

## Uso y arquitectura

Iniciar con `scripts/start.ps1`. HSV + Sobel (144D) sigue como descriptor predeterminado. El selector permite ResNet-18 (512D); conserva el archivo cargado e invalida predicción/vector/rutas anteriores. Los cuatro algoritmos y los dos criterios de Decision Tree están disponibles en ambos descriptores. Cambiar modelo o criterio actualiza la predicción y su galería cuando ya había un resultado. Comparar extrae una sola vez y permite abrir los ejemplos de cada fila.

El registro `runtime/descriptor_registry.json` referencia artefactos existentes con SHA256, clases, fingerprint, dimensiones, versiones y origen. HSV mantiene `runtime/data/active_classifiers.json` sin cambios. ResNet usa Gini/entropía del experimento de comparación, RF/XGBoost originales y LightGBM regularizado versionado. El LightGBM anterior sigue conservado y referenciado como `previous_lightgbm`; no se reescribió ningún bundle.

FastAPI no importa torch. `backend/descriptors.py` arranca al primer uso un extractor persistente en el intérprete experimental; `backend/resnet_worker.py` usa los pesos oficiales `IMAGENET1K_V1`, RGB, resize 256, center crop 224, normalización ImageNet y fc Identity en evaluación/inference_mode. CUDA se comprueba mediante operación real en el extractor compartido; CPU es alternativa. No entrena ni descarga pesos durante las consultas. El checkpoint local se comprueba antes de ofrecer el descriptor.

Se admite un clasificador ResNet en caché junto al bundle HSV previo y cuatro vectores como máximo. La clave contiene el ID versionado del descriptor y SHA de la imagen. No se guardan las imágenes subidas. Se valida `n_features_in_`, clases 0–100, tipo de algoritmo, criterio y dimensiones/finitud del vector. No se cambia el motor C++: búsqueda, k-NN, índices y laboratorio siguen usando HSV144.

Una observación de memoria después de consultas de navegador registró **494,60 MiB RSS** del backend y **1.106,55 MiB** del worker principal; el árbol de procesos sumaba **1.612,76 MiB**, incluidos lanzadores auxiliares (`runtime/resnet_memory_snapshot.json`). Es una muestra puntual con CUDA activa, no un máximo ni la VRAM ni una prueba de carga.

`/api/classify` recibe `descriptor_id`; devuelve el descriptor, metadatos de extracción/dispositivo/caché, duración total, inferencia por clasificador, ejecución de origen y preprocesamiento. El tiempo de extracción fría incluye iniciar el worker/cargar la red. Los tiempos de inferencia de las tablas excluyen ese coste. Los modelos explicables reciben el mismo ID para que nombres de características, rutas, árboles e importancias correspondan al descriptor seleccionado.

`/api/categories/{category}/images` devuelve como máximo seis IDs del manifiesto **train**, con categoría exacta. La interfaz reutiliza tarjetas y modal existentes. No inventa distancias/rangos; explica que son ejemplos de la categoría y no recuperación por similitud. Las pruebas comparan categoría, split, URL y bytes con los archivos del manifiesto. No hay nuevas fotos sintéticas ni de servicios externos.

Las comparaciones se filtran por descriptor. Se muestran por separado prueba histórica reutilizada y validación interna. La CV original de cinco folds, la búsqueda posterior de tres folds y la confirmación reservada no se presentan como el mismo protocolo. No existen resultados de CV HSV en estos artefactos: se indica expresamente.

## Configuración y reversión

Las dependencias torch/torchvision permanecen en `runtime/resnet-env`; no son requisitos del servidor HSV. Consultar `experiments/resnet18/README.md` para instalar el entorno CPU/CUDA y los pesos. El registro y los artefactos runtime no van a Git; en otra máquina hay que proporcionarlos y configurarlos explícitamente:

```powershell
.\.venv\Scripts\python.exe -m scripts.configure_resnet --comparison runtime/experiments/resnet18_comparison/20261118T204643Z-8efb68c2 --lightgbm runtime/experiments/lightgbm_resnet/20261118T211652Z-600d0ec2
```

La configuración se escribe mediante reemplazo atómico y conserva el registro previo si existía. No genera ni entrena modelos.

Para desactivar ResNet y reiniciar normalmente:

```powershell
$env:METRICLAB_ENABLE_RESNET='0'
.\scripts\start.ps1
```

Para reactivar, definirlo como `1` y reiniciar. `METRICLAB_RESNET_PYTHON` permite elegir otro intérprete compatible; `METRICLAB_DESCRIPTOR_REGISTRY` otro registro; `METRICLAB_RESNET_DEVICE` admite `auto`, `cpu` o `cuda`. Faltando intérprete, torch, registro o checkpoint válido, solo ResNet queda indisponible y HSV sigue funcionando. Los controles frontend deshabilitan la opción no disponible. Un fallo tardío del worker se comunica como error; se conserva la posibilidad de elegir HSV.

La copia previa a esta integración está en `runtime/integration_backups/20261118T211632Z/application_before.zip`, con hashes del estado previo y de los 143 artefactos inmutables. Permite revertir también el código local; conservar antes cualquier cambio posterior del usuario. Las antiguas pruebas de experimentos que comparan hashes del código de la aplicación describen su estado de entonces y no deben interpretarse como prohibición de esta integración autorizada.

## Resultados ResNet activados: prueba histórica reutilizada

| Algoritmo / criterio | Accuracy | Balanced accuracy | F1 macro | Entrenamiento (s) | Inferencia (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decision Tree Gini | 57,09 % | 42,46 % | 0,4237 | 15,273 | 0,075 |
| Decision Tree entropía | 52,88 % | 35,78 % | 0,3553 | 59,565 | 0,057 |
| Random Forest | 86,12 % | 77,41 % | 0,7903 | 36,733 | 22,700 |
| XGBoost | 87,15 % | 80,55 % | 0,8184 | 42,869 | 3,107 |
| LightGBM regularizado | 87,33 % | 80,91 % | 0,8160 | 48,617 | 0,687 |

Se conservan los tiempos de sus ejecuciones respectivas; no constituyen un benchmark simultáneo. La inferencia mide `predict_proba`, no la CNN ni HTTP. Las mismas 1.736 imágenes ya intervinieron en la exploración del descriptor: esta tabla no es una validación independiente nueva. El ajuste posterior de LightGBM solo usó entrenamiento; detalle en `experiments/lightgbm_resnet/RESULTADOS.md`.

## Verificación y límites

`python -m unittest discover -s tests -v` pasó **21 pruebas**: modelos de ambos descriptores, galerías reales, vector extraído frente al guardado, extracción única, controles de dimensión, desactivación y entorno ausente, métricas/protocolos, hashes anteriores y laboratorio métrico. `npm.cmd run build` pasó TypeScript y generó el frontend. **Las siete pruebas Playwright pasaron en Edge**, validando los flujos anteriores y ResNet, ambos criterios, cambios de modelo/fila, imágenes cargadas y categoría, ampliación, alcance experimental, reversión HSV y ancho móvil. Artefactos en `runtime/integration_tests.log`, `runtime/e2e_final.log` y capturas `runtime/resnet-classification-*.png`.

`python -m scripts.verify_resnet_cpu` realizó una extracción CPU explícita y la comparó con el embedding conservado: 512 componentes, diferencia absoluta máxima **0,0000047684**, primera extracción **5,282 s** incluyendo carga. CUDA se verificó también mediante la API y el flujo de navegador. La primera consulta puede tardar varios segundos. La carga de un modelo no residente se incluye en el total de solicitud, no en `predict_proba`.

Pendientes: conjunto confirmatorio externo, calibración de probabilidades y benchmark de concurrencia/memoria sostenida; no se afirma capacidad de producción multiusuario. La caché limita referencias, pero el asignador de memoria puede conservar reservas. No se comprobó instalación limpia en otra máquina. Persisten categorías con pocos casos y precisión limitada. La búsqueda de siete configuraciones no demuestra un óptimo global ni una causa única por parámetro.
