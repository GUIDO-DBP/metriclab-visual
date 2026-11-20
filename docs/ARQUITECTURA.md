# Arquitectura y límites de módulos

La aplicación principal clasifica fotografías; el laboratorio de búsqueda métrica es complementario.

| Módulo | Responsabilidad y dependencias |
| --- | --- |
| `frontend/src` | React/TypeScript: descriptor/modelo, carga, evaluación, árboles y galerías. Obtiene resultados reales de FastAPI. |
| `backend/main.py` | API y frontend compilado. Orquesta clasificación y rutas de imágenes. |
| `backend/core.py` | Manifiesto y vectores HSV. Importa `metriclab/demo/extract_features.py`; carga opcionalmente la DLL C++. |
| `backend/modeling.py` | Modelos HSV y explicaciones de árboles/importancias. |
| `backend/descriptors.py` | Registro ResNet, validación SHA256/clases/dimensiones y worker persistente separado. |
| `backend/resnet_worker.py` | Extracción oficial ResNet-18 congelada; no entrena ni descarga durante la consulta. |
| `metriclab/` | Secuencial, VP-Tree y LAESA, puente ABI, oráculo y demo histórica. Backend, build y tests mantienen dependencias reales. |
| `experiments/` | Scripts reproducibles e informes; no se ejecutan al instalar. |
| `resources/` | Catálogo portable, inventarios, hashes y evidencia científica resumida. |

`runtime/data/active_classifiers.json` selecciona modelos HSV144; `runtime/descriptor_registry.json` selecciona modelos ResNet512 y valida su procedencia. Ningún modelo recibe el descriptor del otro. C++ sigue usando HSV144.

Modelos y embeddings se distribuyen por Releases. Las fotografías conservan IDs, categorías y partición del manifiesto. Galerías muestran exclusivamente seis filas de entrenamiento de la categoría predicha; no son recuperación por similitud. Solo una muestra identificada puede aportar categoría verdadera.

Dataset, entornos, cachés y compilaciones permanecen ignorados. PostgreSQL y PDF no son requisitos para clasificar. [Instalación](INSTALACION_WINDOWS.md) · [Integración reversible](INTEGRACION_RESNET.md).
