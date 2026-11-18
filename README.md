# MetricLab Visual

Aplicación local para buscar imágenes con índices métricos C++ y clasificarlas con modelos Python. Interfaz React + TypeScript, FastAPI, Lucide y Recharts. No requiere servicios de IA, claves API ni PostgreSQL para funcionar.

## Estado de esta entrega

- Se conservó el motor existente: Secuencial, VP-Tree (pivote de máxima varianza) y LAESA (16 pivotes).
- El descriptor utiliza **exactamente el extractor original**: imagen BGR de 128 × 128, histograma HSV de 128 dimensiones y orientación de bordes de 16. Cada parte se normaliza por separado. Distancia euclidiana.
- Caltech-101 local preparado: **6.941 imágenes indexadas, 1.736 consultas reservadas, 101 categorías**. Se excluye `BACKGROUND_Google`. No hay imágenes de consulta dentro del índice ni del entrenamiento.
- Decision Tree con Gini y entropía, Random Forest de 200 árboles y k-NN mediante el VP-Tree están operativos y evaluados.
- XGBoost y LightGBM no están instalados en esta entrega: aparecen deshabilitados con instrucciones, nunca con predicciones inventadas.
- Galería ampliable, carga por archivo o arrastre, vista previa reutilizada entre secciones, histogramas, comparación de índices, topología y traza real del árbol, probabilidades/votos, caminos reales de Decision Tree y matrices de confusión.
- PostgreSQL tiene esquema e importador opcionales. El backend todavía lee los artefactos locales, incluso después de importar a PostgreSQL.

El reporte `MetricLab_reporte_demo.html` es una demostración **sintética de 400 imágenes indexadas y 100 consultas**, con ocho categorías. Sus cifras no se utilizan como resultados de Caltech. Véase [auditoría](docs/AUDITORIA.md).

## Ejecutar ahora en Windows / VS Code

Abre esta carpeta en VS Code. En una terminal PowerShell, desde la raíz:

```powershell
.\scripts\start.ps1
```

Abre **http://127.0.0.1:8000**. El frontend compilado y la API se sirven juntos. La API interactiva está en **http://127.0.0.1:8000/docs**. Detén el servidor con `Ctrl+C`.

Si PowerShell bloquea scripts locales, puedes ejecutar directamente:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

No es necesario activar el entorno virtual. Si ya hay un servidor en el puerto 8000, utiliza el existente o detenlo antes de iniciar otro.

## Instalación desde cero

Requisitos: Python de 64 bits (recomendado 3.12–3.14), Node.js 22.12 o posterior y g++ de MSYS2 UCRT64. Versiones exactas verificadas en [pruebas](docs/PRUEBAS.md).

1. Instala Python y Node.js. Instala MSYS2 y, desde su terminal **UCRT64**, ejecuta:

```sh
pacman -S --needed mingw-w64-ucrt-x86_64-gcc
```

2. Añade `C:\msys64\ucrt64\bin` a `PATH` (el script también busca esa ubicación). Crea un entorno Python **aislado** e instala las dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Compila y valida el motor:

```powershell
.\scripts\build-engine.ps1
```

El script genera `runtime/metriclab.dll` y ejecuta el oráculo original. Usa enlace estático del runtime de MinGW para evitar DLL adicionales en otra computadora. Detén el backend antes de recompilar la DLL.

Si no puedes usar scripts PowerShell, la compilación equivalente es:

```powershell
g++ -std=c++17 -O2 -shared -static -I metriclab/include metriclab/apps/web_bridge.cpp metriclab/src/vp_tree.cpp metriclab/src/laesa.cpp metriclab/src/trace.cpp -o runtime/metriclab.dll
```

Alternativa CMake, si lo tienes instalado:

```powershell
cmake -S metriclab -B runtime/build -G "MinGW Makefiles"
cmake --build runtime/build --config Release
ctest --test-dir runtime/build --output-on-failure
Copy-Item runtime/build/libmetriclab.dll runtime/metriclab.dll
```

Con CMake puede ser necesario mantener `C:\msys64\ucrt64\bin` en `PATH` para sus DLL de ejecución. La ruta principal comprobada en esta entrega es el script de g++.

4. Coloca Caltech-101 descomprimido así (ya está presente en este workspace):

```text
dataset/101_ObjectCategories/
  accordion/image_0001.jpg
  airplanes/image_0001.jpg
  ...
```

5. Prepara vectores, separación, modelos y evaluación:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare --train
```

Se procesan todas las imágenes válidas de las carpetas de objetos. La separación 80/20 se hace por categoría, semilla 7, con al menos una consulta y una imagen de entrenamiento por clase. Las clases con menos de dos imágenes se omiten. Para una preparación rápida explícitamente parcial:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare --max-per-class 40 --train
```

El límite queda registrado y visible en Dataset. Para usar otra carpeta, cópiala dentro de `dataset/` y añade `--images dataset/mi_coleccion`. El parámetro `--kind caltech101` declara el origen: úsalo solo para imágenes reales de Caltech. Para datos sintéticos utiliza `--kind synthetic`.

6. Instala y compila el frontend:

```powershell
cd frontend
npm.cmd ci
npm.cmd run build
cd ..
.\scripts\start.ps1
```

La interfaz puede abrirse antes de preparar datos: mostrará el estado pendiente. No es necesario configurar una base de datos.

## Desarrollo con recarga del frontend

Terminal 1, en la raíz:

```powershell
.\scripts\start.ps1 -Dev
```

Terminal 2:

```powershell
cd frontend
npm.cmd run dev
```

Abre la URL que imprime Vite (normalmente http://127.0.0.1:5173). Vite envía `/api` al backend del puerto 8000. No abras `index.html` mediante `file://`. `npm run preview` sirve solo archivos estáticos; para la aplicación integrada usa FastAPI o Vite dev.

## Cómo presentar el proyecto

1. En Inicio, entra en Búsqueda y elige una consulta reservada de las miniaturas, o carga una foto JPG/PNG.
2. Ejecuta VP-Tree y observa imágenes, distancias y métricas. Prueba Secuencial y LAESA con la misma imagen.
3. En Clasificación se conserva la imagen. Selecciona un modelo o compara todos los disponibles. Abre las decisiones del árbol; para k-NN observa las cinco imágenes que votaron.
4. En Laboratorio ejecuta los tres índices y avanza en la traza. Pulsa nodos del diagrama para explorar ramas y usa Remontar para regresar. El árbol tiene miles de nodos: se muestran cuatro niveles a la vez.
5. Revisa métricas de clasificadores y abre la matriz de confusión. La evaluación procede del conjunto reservado completo, no de la fotografía cargada.
6. En Dataset explica la separación y la diferencia entre búsqueda y clasificación.

Se aceptan JPG/PNG de hasta 10 MB y 24 megapíxeles. El backend decodifica y valida el contenido real. Las imágenes subidas se procesan en memoria y no se guardan.

## Modelos opcionales

```powershell
.\.venv\Scripts\python.exe -m pip install xgboost lightgbm
.\.venv\Scripts\python.exe -m scripts.prepare --train-only
```

Reinicia FastAPI después de preparar o entrenar. `--train-only` reutiliza los vectores actuales. Los errores de bibliotecas o entrenamiento quedan registrados como no disponibles. La preparación puede hacerse sin `--train` si solo necesitas búsqueda y k-NN interactivo.

La clasificación usa las bibliotecas existentes del proyecto; no son implementaciones propias desde cero de árboles/boosting. Si el docente exige implementarlos manualmente, ese trabajo sigue pendiente. El motor métrico sí utiliza las implementaciones C++ propias originales.

## Demostración sintética separada

El generador original puede preparar una colección alternativa, sin sobrescribir Caltech:

```powershell
.\.venv\Scripts\python.exe metriclab/demo/make_demo_images.py runtime/demo-images
$env:METRICLAB_DATA = 'runtime/demo-data'
.\.venv\Scripts\python.exe -m scripts.prepare --images runtime/demo-images --kind synthetic --train
.\scripts\start.ps1
```

El aviso de demostración aparece en todas las secciones. Para volver a Caltech, detén el servidor y ejecuta:

```powershell
Remove-Item Env:METRICLAB_DATA -ErrorAction SilentlyContinue
.\scripts\start.ps1
```

Cada preparación guarda una huella de vectores, etiquetas y separación. Se rechazan modelos y evaluaciones de otra huella para evitar mezclar datos. Detén el servidor antes de regenerar una preparación. No ejecutes dos preparaciones simultáneas sobre la misma carpeta de salida.

## Arquitectura y API

```text
frontend/             React + TypeScript + Vite; CSS responsive
backend/core.py       descriptor original y adaptador ctypes
backend/main.py       FastAPI, validación, búsquedas y predicciones
metriclab/            motor original, demos, benchmarks y pruebas
  apps/web_bridge.cpp puente C ABI para índices persistentes
scripts/              compilación, preparación, inicio e importación
runtime/data/         manifiesto, vectores, modelos y experimentos locales
docs/                 auditoría, pruebas y esquema PostgreSQL
tests/                integración real del backend
frontend/e2e/         pruebas del navegador
```

Se eligió **ctypes + una ABI C mínima**, en lugar de pybind11, para evitar dependencias de compilación de Python. El puente copia los vectores una vez y conserva los tres índices en memoria. Un bloqueo serializa el acceso a sus contadores. El tiempo de consulta mide únicamente `searchKnn`, sin JSON, transporte HTTP, comparación contra el oráculo ni instrumentación de traza. El costo de construcción se mide al iniciar el backend. Para aprender cómo se monta el ciclo de vida de la API, consulta [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/); el proxy de desarrollo sigue [Vite server options](https://vite.dev/config/server-options/).

No se modificaron las reglas de búsqueda ni de poda. La única extensión de VP-Tree es una lectura de topología para el diagrama. Se verifica la lista ordenada de distancias con tolerancia `1e-9`: si hay empates, distintos IDs a la misma distancia son válidos. Si falla la verificación, no se presenta el resultado como correcto.

| Endpoint | Función |
| --- | --- |
| `GET /api/status` | Estado, dataset, índices y modelos disponibles |
| `GET /api/dataset` | Categorías, conteos, separación y consultas de muestra |
| `GET /api/images/{train\|query}/{id}` | Imagen por ID validado; no acepta rutas arbitrarias |
| `POST /api/search` | Multipart: `file`, `k` (1–20), `algorithm` (0 secuencial, 1 VP-Tree, 2 LAESA) |
| `POST /api/compare` | Multipart: `file`, `k`; ejecuta los tres índices y devuelve la traza del VP-Tree |
| `GET /api/tree` | Topología real del VP-Tree |
| `POST /api/classify` | Multipart: `file`, `model`, `compare` y `k` (votos k-NN) |
| `GET /api/experiments` | Métricas calculadas y matrices de confusión con huella compatible |

Los modelos aceptados son `gini`, `entropy`, `forest`, `xgboost`, `lightgbm`, `knn`. La traza de consulta se devuelve junto con búsqueda VP-Tree o comparación, asociada a esa misma ejecución. No hay almacenamiento global de la última imagen que pueda cruzar consultas de usuarios.

Para añadir HOG/embeddings en el futuro, introduce un identificador de descriptor y una nueva preparación, dimensión y modelos compatibles. No mezcles sus vectores con `hsv128-edge16-v1`. Esta versión expone solamente el descriptor disponible.

## PostgreSQL opcional

El esquema [docs/schema.sql](docs/schema.sql) separa datasets por huella, metadatos de imágenes con rutas relativas, descriptores y evaluaciones. No requiere extensiones vectoriales: PostgreSQL almacena datos, C++ sigue buscando.

```powershell
.\.venv\Scripts\python.exe -m pip install "psycopg[binary]"
# Define DATABASE_URL en el entorno local sin compartir su contraseña.
.\.venv\Scripts\python.exe -m scripts.import_postgres
```

El importador crea tablas e importa los artefactos de forma transaccional e idempotente. No se ha ejecutado contra un servidor PostgreSQL en esta entrega. Para una migración completa falta añadir un repositorio de lectura de PostgreSQL en el backend, que cargue esos mismos vectores al motor. Los experimentos interactivos no se guardan en la base automáticamente.

Variables documentadas en `.env.example`: `METRICLAB_DATA` selecciona los artefactos y `DATABASE_URL` solo se usa en el importador. Se leen del entorno, el archivo `.env` no se carga automáticamente. Las rutas de imágenes guardadas son relativas a la raíz del proyecto.

## Pruebas

```powershell
.\runtime\oracle_test.exe
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd frontend
npm.cmd run build
# Con FastAPI iniciado en 8000 y Microsoft Edge instalado:
npx.cmd playwright test
```

Los E2E actuales esperan los cuatro modelos de esta entrega y las 101 categorías de Caltech preparado completo. Consulta [docs/PRUEBAS.md](docs/PRUEBAS.md) para resultados y limitaciones. Capturas generadas en `runtime/home-desktop.png`, `runtime/search-desktop.png`, `runtime/lab-desktop.png`, `runtime/home-mobile.png` y `runtime/search-mobile.png`.

## Qué falta y qué proporcionar

- Para XGBoost/LightGBM: sus bibliotecas y un nuevo entrenamiento; no faltan imágenes de Caltech para el flujo actual.
- Para PostgreSQL: una instancia y credenciales locales, más la integración de lectura si deseas sustituir archivos.
- Para descriptores CNN/HOG: especificación y pesos/preprocesamiento, junto con nueva preparación y evaluación.
- Para reproducir exactamente las imágenes de la demo histórica: sus archivos originales o ejecutar el generador con los mismos parámetros; el HTML contiene resultados ilustrativos y miniaturas, no un índice reutilizable.
- Para el informe final: confirmar el protocolo académico (p. ej. validación cruzada, muestras por clase e implementaciones manuales de clasificadores). Las mediciones actuales son un experimento reproducible con división estratificada, no una afirmación de rendimiento del estado del arte.

Los artefactos `runtime/`, el entorno virtual y el dataset se excluyen de Git por tamaño y portabilidad. Para otra computadora copia el dataset o proporciónalo de nuevo, instala y ejecuta los pasos de preparación. Los modelos `joblib` deben provenir únicamente de una preparación local confiable.
