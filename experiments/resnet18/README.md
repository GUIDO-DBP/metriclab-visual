# Experimento aislado: ResNet-18 como descriptor visual

Este módulo **no se importa desde la aplicación**. No cambia el frontend, el descriptor activo, los requisitos principales, la división de datos ni los modelos que usa FastAPI.

ResNet-18 se carga con `ResNet18_Weights.IMAGENET1K_V1`, se congela, se pone en `eval()` y se sustituye exclusivamente `fc` por `Identity`. Su salida es el vector de **512 activaciones anteriores a la capa de clasificación**, después de average pooling y flatten. No se entrena ni se afina la CNN. Se usa FP32, sin AMP ni normalización adicional de embeddings.

El preprocesamiento es el objeto oficial `weights.transforms()`, aplicado a imágenes PIL RGB: resize a lado corto 256, center crop 224, valores 0–1 y normalización ImageNet. Se guarda su representación y el SHA-256 de los pesos oficiales. [Documentación torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

## Datos y protocolo fijado antes de evaluar

- Fuente: `runtime/data/manifest.json`, exactamente 6.941 entrenamientos y 1.736 pruebas de las 101 categorías de Caltech ya preparadas.
- Se comprueba el fingerprint de los vectores HSV/etiquetas existentes, los IDs y que las rutas no se solapen. No se llama a train_test_split ni se omiten imágenes.
- Se conserva el orden completo del manifiesto. Cada embedding tiene fila, ID, ruta, categoría, etiqueta codificada y SHA-256 del archivo original.
- Random Forest y XGBoost copian sus parámetros de entrenamiento históricos de la ejecución activa, antes de ver resultados. Misma semilla, mismo número de árboles y mismos parámetros para HSV y ResNet; únicamente se limita `n_jobs` por igual para recursos.
- No se usa prueba para fit, selección de características, hiperparámetros, early stopping ni entrenamiento de la CNN.
- Se entrenan primero RF/XGBoost sobre 512D y después se repite la referencia HSV144D, guardando esos modelos de referencia también fuera de la aplicación. Así los tiempos no dependen de una medición histórica de otra sesión.
- Los clasificadores usan CPU con dos hilos por defecto; CUDA acelera el extractor, si pasa pruebas reales de tensores y convoluciones. `--device cpu` permite ejecutar toda la extracción en CPU. `--device auto` cae a CPU si la comprobación CUDA falla; `--device cuda` exige CUDA funcional.

Si la réplica HSV de XGBoost difiere de la cifra histórica al limitar los hilos, se ejecuta una auditoría adicional con la configuración **original exacta**, incluyendo sus cuatro hilos. Se informa esa diferencia y el resultado de la auditoría por separado; no se elige una configuración nueva ni se modifica la comparación pareada de dos hilos.

## Entorno independiente en Windows

Desde la raíz del workspace:

```powershell
python -m venv --system-site-packages runtime/resnet-env
```

En esta máquina se verificó Python 3.14.4, una RTX 4060 Laptop de 8 GB y PyTorch 2.14.1 + torchvision 0.29.1, con CUDA 13.0:

```powershell
.\runtime\resnet-env\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.\runtime\resnet-env\Scripts\python.exe -m pip install -r experiments/resnet18/requirements.txt
```

Para CPU, instala las mismas versiones desde el índice oficial CPU en ese entorno:

```powershell
.\runtime\resnet-env\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

El entorno de la aplicación `.venv` se usa únicamente para medir su extractor HSV original y debe seguir instalado. Los paquetes científicos existentes se reutilizaron mediante system-site-packages; un entorno aislado sin esa opción requiere instalar todas las dependencias del archivo experimental. No se modifica `.venv` al instalar PyTorch.

La primera extracción descarga los pesos oficiales (~45 MB). La caché se fija en `runtime/experiments/torch-cache`, dentro del workspace. Las ruedas CUDA de PyTorch pueden ocupar varios GB; no son modelos adicionales entrenados por el proyecto.

## Ejecutar

```powershell
.\runtime\resnet-env\Scripts\python.exe -m experiments.resnet18.run --device auto --batch-size 8 --threads 2 --benchmark-samples 128
```

O fuerza CPU:

```powershell
.\runtime\resnet-env\Scripts\python.exe -m experiments.resnet18.run --device cpu --batch-size 8 --threads 2 --benchmark-samples 128
```

Cada ejecución crea una carpeta nueva en `runtime/experiments/resnet18/<run_id>`. Los cuatro entrenamientos se ejecutan secuencialmente en procesos distintos para liberar memoria. No se carga el bundle completo de los modelos principales para copiar parámetros; se leen sus metadatos JSON.

Si se interrumpe, se puede retomar una carpeta ya creada:

```powershell
.\runtime\resnet-env\Scripts\python.exe -m experiments.resnet18.run --run runtime/experiments/resnet18/<run_id> --device auto
```

Al retomar, usa los mismos límites originales de batch, hilos y muestras de benchmark. Se validan los resultados completos existentes y no se sobrescriben modelos. Una extracción incompleta se inicia de nuevo; no se aceptan embeddings parciales como resultados completos.

## Archivos independientes

```text
split.json / classes.json       copia exacta de separación y codificación
train_resnet18.npy              (6941,512), float32
query_resnet18.npy              (1736,512), float32
train_labels.npy                etiquetas, mismo orden
query_labels.npy                etiquetas, mismo orden
train_rows.json                 imagen/ID/categoría/hash por fila
query_rows.json                 imagen/ID/categoría/hash por fila
extraction.json                 pesos, transforms, dispositivo y tiempos
alignment_checks.json           validación completa y coincidencias de contenido
baseline.json                   métricas y parámetros históricos de referencia
models/*                       cuatro modelos independientes y predicciones
results.json / comparison.csv   métricas comparables y cambios en puntos
REPORT.md / comparison.png      informe y gráfica calculados
protected_files.json            hashes iniciales de aplicación y artefactos activos
completion.json                 prueba final de que siguen sin cambios
execution.log                   ejecución y progreso
```

## Verificación

Después de completar un experimento:

```powershell
.\runtime\resnet-env\Scripts\python.exe -m unittest experiments.resnet18.test_experiment -v
```

Las pruebas verifican las 8.677 correspondencias imagen/etiqueta y sus hashes, las métricas desde las predicciones reales, igualdad de parámetros entre descriptores, preservación de los archivos principales y pesos congelados. Además ejecutan el extractor en **CPU real** y comparan dos filas con sus embeddings guardados, registrando una comprobación de latencia CPU de 16 imágenes en `cpu_validation.json`.

Durante la extracción se reextraen seis filas reales de manera individual y se comparan con los resultados por lote; el estado de la CNN (incluido BatchNorm) debe mantener el mismo SHA-256. Cada clasificador se guarda, libera y recarga para comprobar sus probabilidades.

Accuracy, balanced accuracy y F1 se calculan sobre las 1.736 pruebas completas. Inferencia es la media de 128 llamadas unitarias a `predict_proba`, tras calentamiento, excluyendo extracción/IO/HTTP. Se informa aparte la extracción ResNet y HSV. El benchmark GPU sincroniza CUDA para evitar tiempos artificialmente cortos.

## Interpretación

La decisión de integrar depende de ganancia en accuracy y métricas macro, costo de extracción, disponibilidad CPU/GPU y confirmación posterior con validación independiente. La división se conserva aunque se encuentren imágenes de contenido idéntico entre particiones: ese conteo se informa como límite del experimento y no se corrige a escondidas.

Es aprendizaje por transferencia desde ImageNet; no se auditó el posible solapamiento de Caltech con ese corpus de preentrenamiento. Este experimento no ofrece validación cruzada, probabilidades calibradas ni una garantía sobre cualquier fotografía externa. No activa automáticamente un descriptor aunque mejore.
