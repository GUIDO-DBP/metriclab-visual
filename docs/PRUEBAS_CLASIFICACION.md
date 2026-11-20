# Evidencias de la actualización — 18 de noviembre de 2026

## Entorno y protocolo

Windows x64, Python 3.14.4, Node.js 24.15.0, g++ MSYS2 UCRT64 16.2.0, scikit-learn 1.9.1, XGBoost 3.4.1 y LightGBM 4.7.0. Otras versiones directas en `requirements/tested-direct.txt`; npm fija dependencias en `package-lock.json`.

Caltech-101: 6.941 imágenes de entrenamiento y 1.736 de prueba, 101 categorías, descriptor original de 144 dimensiones. División anterior intacta, semilla 7. Modelos con semilla 0. Sin ajuste de hiperparámetros ni selección de características usando prueba.

Ejecución activa de esta entrega: `20261118T134358Z-421958bc`. El directorio contiene modelos joblib, formatos nativos de boosting, parámetros, versiones, latencias y matrices. Los artefactos raíz de la primera entrega y un ZIP de respaldo permanecen disponibles.

## Resultados calculados

| Configuración | Accuracy | Balanced accuracy | Precision macro | Recall macro | F1 macro | Entrenamiento (s) | Inferencia media (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision Tree — Gini | 0,322581 | 0,158992 | 0,155056 | 0,158992 | 0,154074 | 1,024450¹ | 0,053623 |
| Decision Tree — Entropía | 0,305876 | 0,137657 | 0,141178 | 0,137657 | 0,137634 | 2,812866¹ | 0,052765 |
| Random Forest | 0,464862 | 0,247748 | 0,358776 | 0,247748 | 0,258480 | 1,159984¹ | 22,949248 |
| XGBoost | 0,487903 | 0,282080 | 0,337795 | 0,282080 | 0,291416 | 10,691721 | 1,381695 |
| LightGBM | 0,461982 | 0,236895 | 0,293057 | 0,236895 | 0,244960 | 5,999019 | 0,812082 |

¹ Modelo conservado: tiempo registrado en su entrenamiento anterior. No es una medición nueva simultánea al entrenamiento de boosting. Los parámetros originales y de ejecución se guardan por separado.

Accuracy/precision/recall/F1 se recalcularon sobre **todas** las imágenes de prueba. Latencia: 128 consultas unitarias a `predict_proba`, espaciadas sobre prueba y después de calentamiento; sin extracción, HTTP, JSON ni dibujo. Se usa hasta cuatro hilos. Las latencias en vivo pueden diferir.

Los resultados muestran el límite del descriptor: accuracy de aproximadamente 30–49 % y métricas macro menores en un dataset desbalanceado. No se presentan como precisión garantizada para fotografías arbitrarias. Tampoco se mezclan con la demostración sintética del HTML previo.

## Backend: 14 pruebas aprobadas

Comando: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.

Las siete pruebas anteriores se conservaron/adaptaron para separar clasificación principal y complemento. Se añadieron otras siete para los clasificadores:

- Extracción compatible con el código original, rutas train/test disjuntas y tamaño de matrices correcto.
- Los tres índices reales coinciden con secuencial para k=5/10/20. Topología/trazas válidas, consultas concurrentes y estados de error.
- Cada modelo produce una predicción igual a `predict` del objeto entrenado, 101 probabilidades finitas que suman 1 y latencia positiva.
- Comparación con cuatro resultados exactos; Gini/entropía como criterio seleccionable. Se comprueba que la extracción se llama **una sola vez** y se conserva la misma huella de vector.
- Persistencia joblib sin cambios en probabilidades; carga de XGBoost JSON y LightGBM TXT nativos produce probabilidades equivalentes.
- Todas las métricas publicadas se verifican recalculando predicciones sobre prueba completa, con la misma huella y ejecución. Cada matriz suma 1.736 observaciones.
- Nodos, umbrales, impurezas y pesos coinciden con los estimadores; rutas de imagen siguen los hijos reales y terminan en una hoja. Se prueban ambos criterios y árboles 1 y 10 del bosque.
- Las 144 importancias coinciden con las del modelo. Se verifica un bosque de 200 árboles y XGBoost de 80 etapas.
- La clasificación de boosting funciona con `engine=None`; la búsqueda devuelve 503. Un modelo ausente devuelve 409 y no se inventan predicciones.
- Rechazo de índices de árbol, profundidad y modelos inválidos; imágenes corruptas o formatos no admitidos; k e IDs fuera de rango.
- k-NN conserva una ejecución explícita de cinco vecinos, identificado como secundario.

## Frontend y navegador

`npm.cmd run build`: TypeScript estricto y Vite aprobados. Se conservó la separación de chunks de gráficas.

`npm.cmd run test:e2e`: seis escenarios reales en Microsoft Edge instalado, con FastAPI local y modelos guardados, sin respuestas simuladas:

1. Búsqueda complementaria, diez vecinos, modal/Escape, comparación de cuatro, índices, avance en traza y 101 categorías.
2. Pantalla móvil de 390 × 844: carga, clasificación, comparación y navegación por evaluación/explorador/dataset sin desbordamiento del documento.
3. Inferencia individual de los cuatro algoritmos, cambio a entropía y comparación sobre la misma foto, con categorías mostradas iguales a la respuesta real.
4. Explorador Gini/entropía, nodos, ruta hasta hoja, árbol 10 del bosque e importancias reales de ambos modelos de boosting.
5. Cinco configuraciones evaluadas, matriz filtrable, exportación JSON con 1.736 pruebas y consulta de metodología.
6. Archivo corrupto, mensaje de procesamiento, nueva carga por arrastre y recuperación con predicción válida.

Las pruebas de flujo y explorador revisan también ausencia de excepciones JavaScript. Capturas inspeccionadas: `runtime/home-desktop.png`, `runtime/classification-mobile.png`, `runtime/classification-comparison-desktop.png`, `runtime/classifier-explorer-desktop.png` y `runtime/classifier-evaluation-desktop.png`.

## Motor anterior y alcance de verificación

El oráculo C++ volvió a pasar **10.503 comprobaciones, cero fallos**. Sus fuentes, demos y benchmarks no fueron modificados. Las búsquedas de integración ejecutan la DLL real.

Se comprobó el inicio del backend y se corrigió la detección de fallos del script cuando el puerto está ocupado. Para verificar la aplicación actual se reinició una instancia previa del mismo proyecto identificada en el puerto 8000.

No se verificó una instalación limpia en una segunda máquina, el camino de CMake, una instancia PostgreSQL ni calibración/validación cruzada. El entorno local reutiliza paquetes científicos ya instalados; la guía de instalación crea un entorno aislado.

Durante las pruebas se corrigió la serialización de `NaN` en los parámetros de XGBoost: era metadato de valores faltantes, no una probabilidad ni métrica inválida. La API y los artefactos nuevos utilizan JSON compatible.
