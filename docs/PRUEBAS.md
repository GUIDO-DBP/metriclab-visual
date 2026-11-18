# Verificación de la entrega

Ejecutada en Windows el 16 de noviembre de 2026. Dataset local Caltech-101 completo, excluyendo fondo. Semilla de separación 7. Descriptor original de 144 dimensiones.

## Motor original

El ejecutable `runtime/oracle_test.exe` pasó **10.503 comprobaciones con cero fallos**: búsqueda exacta k-NN/rango sobre diferentes distribuciones y dimensiones, empates/casi duplicados y validaciones de desigualdad triangular.

## Integración Python / C++ / FastAPI

`python -m unittest discover -s tests -v`: **7 pruebas aprobadas**.

- Descriptor de una carga HTTP igual al extractor original y separación de rutas train/query sin intersección.
- Secuencial, VP-Tree y LAESA coinciden en distancias para k=5, 10 y 20; la secuencial calcula una distancia por imagen indexada.
- Selección real de cada algoritmo, traza únicamente para VP-Tree y topología con referencias válidas. Las hojas pueden contener varios objetos, de modo que nodos e imágenes no tienen que coincidir.
- Inferencia de todos los modelos disponibles, clases válidas, probabilidades dentro de rango, camino del árbol hasta hoja y cinco vecinos de k-NN.
- Matrices de confusión con conteo total igual a las 1.736 consultas.
- Rechazo de k/algoritmo/modelo inválidos, archivos corruptos, GIF renombrado como JPG e IDs de imagen incorrectos.
- Estados claros de modelo no entrenado (409), índice ausente (503) y consultas paralelas reproducibles mediante acceso serializado al motor.

## Frontend y navegador

`npm.cmd run build`: TypeScript estricto y compilación Vite aprobados.

`npx.cmd playwright test`, Microsoft Edge instalado, modo headless: **2 escenarios aprobados**.

- Escritorio: carga de imagen real, búsqueda y diez fotografías recuperadas, verificación con secuencial, ampliación/cierre con Escape, comparación de cuatro clasificadores, decisiones reales, comparación de índices, avance en la traza y 101 categorías en Dataset. Sin excepciones JavaScript.
- Móvil de 390 × 844: navegación, carga y búsqueda con resultados reales, sin desbordamiento horizontal.

Se inspeccionaron capturas de escritorio y móvil. Los archivos de captura permanecen en `runtime/`. El navegador utiliza la aplicación compilada servida por FastAPI, no respuestas simuladas.

## Evaluación calculada sobre Caltech local

6.941 imágenes de entrenamiento y 1.736 de evaluación. Estos resultados NO pertenecen a la demo sintética del HTML.

| Modelo | Accuracy | Balanced accuracy | F1 macro |
| --- | ---: | ---: | ---: |
| Decision Tree — Gini | 32,26 % | 15,90 % | 0,1541 |
| Decision Tree — Entropía | 30,59 % | 13,77 % | 0,1376 |
| Random Forest | 46,49 % | 24,77 % | 0,2585 |
| k-NN / VP-Tree, k=5 | 38,08 % | 18,59 % | 0,1908 |

El dataset está desbalanceado. La diferencia entre accuracy y balanced accuracy muestra por qué no basta una sola cifra. Este descriptor de color/bordes tiene capacidad semántica limitada; una fotografía externa puede clasificarse incorrectamente. No hay calibración de probabilidades, búsqueda de hiperparámetros ni validación cruzada. Las fracciones de voto de k-NN no se presentan como probabilidades calibradas.

Los tiempos exactos y matrices están en `runtime/data/experiments.json`. Los tiempos de índices son de cada consulta en vivo, excluyen JSON, HTTP, verificación y trazas. Pueden variar por máquina/carga; una sola consulta no constituye un benchmark agregado.

## Límites de lo verificado

- XGBoost y LightGBM: bibliotecas ausentes; estados deshabilitados comprobados en la interfaz. Su entrenamiento no se ejecutó.
- PostgreSQL: esquema e importador preparados, sin servidor/credenciales; no se probó la migración contra una instancia.
- CMake: target añadido pero no ejecutado porque no está instalado; compilación g++ real comprobada.
- El entorno de esta sesión usa `.venv` con acceso a paquetes científicos ya instalados en el equipo; las instrucciones del README crean un entorno aislado. No se comprobó una segunda instalación limpia en otra máquina.
- Las fuentes web son opcionales: el CSS usa fuentes locales de respaldo si no hay Internet.
- El generador sintético y su modo separado están documentados, pero las pruebas E2E de esta entrega se ejecutaron sobre Caltech real.

Stack verificado: Python 3.14, Node.js 24.15.0, g++ 16.2.0 (MSYS2 UCRT64), FastAPI 0.163.1, OpenCV 5.0.0.93, NumPy 2.5.3, scikit-learn 1.9.1, React 19, Vite 6.4.6, Playwright 1.65.1. `frontend/package-lock.json` fija las versiones npm instaladas.
