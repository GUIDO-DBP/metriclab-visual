# Demo funcional de MetricLab Visual

Flujo completo: **imágenes → vectores → búsqueda con el motor C++ → clasificación → reporte HTML.**

## Ejecutar

Requisitos: Python 3 con `numpy scikit-learn opencv-python pillow matplotlib`, un compilador C++17 y CMake.
Para los cuatro clasificadores, además:

```bash
pip install xgboost lightgbm
```

**Con imágenes de prueba** (no hace falta descargar nada):

```bash
cd demo
./run_demo.sh
```

**Con Caltech-101.** Descarga y descomprime el dataset, luego apunta a la carpeta `101_ObjectCategories`:

```bash
./run_demo.sh /ruta/a/101_ObjectCategories
# primera prueba más rápida, limitando cada categoría a 80 imágenes:
MAX_POR_CLASE=80 ./run_demo.sh /ruta/a/101_ObjectCategories
```

La carpeta `BACKGROUND_Google` se ignora automáticamente: es ruido de fondo, no una categoría de objeto.

Al terminar, abre `demo/work/reporte.html` en el navegador.

### En Windows

El script es de bash: funciona en Git Bash o WSL. Si prefieres PowerShell, ejecuta los pasos a mano:

```powershell
cmake -S .. -B ..\build
cmake --build ..\build --config Release
python extract_features.py C:\ruta\101_ObjectCategories work\vectores
..\build\Release\search_cli.exe work\vectores\train.bin work\vectores\query.bin work\vectores 10
python classify.py work\vectores
python report.py work\vectores work\reporte.html
```

## Qué hace cada archivo

| Archivo | Función |
|---|---|
| `make_demo_images.py` | Genera imágenes de prueba con la misma estructura de carpetas que Caltech-101 |
| `extract_features.py` | Convierte cada imagen en un vector de 144 números y separa colección y consultas |
| `../apps/search_cli.cpp` | Motor C++: construye los índices, busca, verifica contra el oráculo y exporta vecinos y traza |
| `classify.py` | Decision Tree (Gini y entropía), Random Forest, XGBoost, LightGBM y k-NN sobre el VP-Tree |
| `report.py` | Reporte HTML autocontenido para mostrar los resultados |

## Decisiones que conviene saber explicar

- **Las consultas nunca entran al índice ni al entrenamiento.** Si una consulta estuviera indexada, su vecino más cercano sería ella misma a distancia cero.
- **La separación es estratificada:** se toma la misma proporción de cada categoría, porque Caltech-101 está desbalanceado.
- **Además de la exactitud se reportan exactitud balanceada y F1 macro**, que pesan igual a todas las categorías.
- **El k-NN no se entrena:** vota con los vecinos que devolvió el VP-Tree. Es el punto que une la clasificación con el índice métrico.
- **Puente C++ ↔ Python por archivos binarios** de forma provisional; la versión final usará pybind11.

## Advertencia

Las imágenes de prueba son sintéticas. Validan que el flujo funciona, **no** predicen el rendimiento sobre Caltech-101. No citar sus cifras en el informe como resultados reales.
