# Resultados verificados de la ampliacion ResNet-18

[Plan de integracion](PLAN_INTEGRACION.md) | [Guia de ejecucion](README.md)

# ResNet-18: comparación de cuatro algoritmos y estabilidad interna

Ejecución: `20261118T204643Z-8efb68c2`. Padre conservado: `20261118T195917Z-42d3213e`.
**La aplicación principal no ha cambiado.** Se reutilizaron los embeddings de 512D y los modelos completos Random Forest/XGBoost; se añadieron Decision Tree (Gini y entropía) y LightGBM.

## Validación interna: solo las 6.941 imágenes de entrenamiento
Cinco particiones estratificadas compartidas por todos los modelos, shuffle y semilla 23. Estimador nuevo en cada partición; parámetros fijos, sin búsqueda ni early stopping. Ambas opciones de Decision Tree son criterios del mismo algoritmo.

| Modelo | Accuracy media ± DE | Balanced accuracy media ± DE | F1 macro media ± DE |
| --- | ---: | ---: | ---: |
| Decision Tree — Gini | 57.05 ± 0.53 % | 42.67 ± 1.03 % | 0.4259 ± 0.0148 |
| Decision Tree — Entropía | 49.70 ± 1.76 % | 32.27 ± 2.21 % | 0.3208 ± 0.0215 |
| Random Forest | 84.53 ± 0.90 % | 76.05 ± 1.08 % | 0.7743 ± 0.0105 |
| XGBoost | 86.88 ± 1.05 % | 80.85 ± 1.25 % | 0.8170 ± 0.0137 |
| LightGBM | 24.36 ± 7.49 % | 18.20 ± 5.24 % | 0.1700 ± 0.0434 |

DE: desviación estándar muestral entre cinco particiones con entrenamientos solapados; para accuracy/balanced accuracy se expresa en puntos porcentuales. No es un intervalo de confianza ni una prueba de superioridad estadística. Cada imagen de entrenamiento tiene exactamente una predicción out-of-fold guardada.
La regla fijada antes de evaluar recomendó **XGBoost** por mayor F1 macro medio de validación interna (desempate por balanced accuracy). La recomendación quedó congelada antes de evaluar los modelos nuevos en prueba. No se activó ningún modelo.
Random Forest y XGBoost sostienen resultados altos con el descriptor congelado. Los árboles individuales son menos precisos. LightGBM con los parámetros heredados muestra baja precisión y alta dispersión entre folds: no debe promoverse con esta configuración sin un diagnóstico o ajuste posterior exclusivamente dentro de entrenamiento. No se modificaron parámetros para elevar su resultado después de observar la prueba histórica. Esta comparación no implica que ResNet beneficie por igual a todos los clasificadores.

## Comparación descriptiva en las 1.736 imágenes de prueba YA UTILIZADAS
**Esta tabla no es una validación independiente nueva.** Mantiene la separación original para comparar resultados de las cuatro familias bajo el mismo descriptor. Los resultados RF/XGBoost corresponden a sus modelos completos existentes, sin reentrenarlos.

| Modelo | Accuracy | Balanced accuracy | F1 macro | Entrenamiento (s) | Inferencia (ms/imagen) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decision Tree — Gini | 57.09 % | 42.46 % | 0.4237 | 15.273 | 0.075 |
| Decision Tree — Entropía | 52.88 % | 35.78 % | 0.3553 | 59.565 | 0.057 |
| Random Forest | 86.12 % | 77.41 % | 0.7903 | 36.733¹ | 22.700 |
| XGBoost | 87.15 % | 80.55 % | 0.8184 | 42.869¹ | 3.107 |
| LightGBM | 19.76 % | 10.32 % | 0.1139 | 13.782 | 0.437 |

¹ Tiempo de entrenamiento conservado del experimento padre. RF/XGBoost no volvieron a ajustarse con los 6.941 ejemplos completos. Sus clones de validación sí se entrenaron únicamente con las filas correspondientes de cada partición.
La inferencia de los cinco modelos se volvió a medir en esta ejecución: 128 consultas unitarias a predict_proba después de calentamiento, dos hilos CPU, mismo orden de imágenes. Excluye extracción ResNet, lectura de fotos, HTTP y visualización.

## Trazabilidad y comprobaciones
- 66 archivos protegidos de la aplicación y 38 archivos del experimento padre conservan su SHA-256.
- Los embeddings no se recalcularon, normalizaron ni mezclaron con HSV; se comprobó su checksum y alineación con rutas/etiquetas.
- Modelos nuevos versionados, RF/XGBoost referenciados en lectura; ningún modelo HSV ni referencia activa se sobrescribió.
- Folds e índices guardados, parámetros y versiones fijados en protocol.json, predicciones por fold/OOF y por prueba disponibles para recalcular métricas.
- CNN congelada de ImageNet, sin fine tuning ni extracción dependiente de etiquetas. Precalcular sus embeddings no ajusta un transformador a las etiquetas del conjunto de validación.

## Límites e integración posterior
La validación interna evalúa estabilidad condicional al descriptor ResNet previamente elegido. Ese enfoque ya fue explorado con la prueba histórica; ni esta CV ni reutilizar la prueba eliminan ese antecedente. Antes de una afirmación de generalización final conviene una evaluación externa o un protocolo prospectivo independiente.
El candidato de integración se justifica por CV, no por elegir el mayor accuracy de la tabla de prueba reutilizada. El plan propone selector HSV/ResNet, modelos asociados al descriptor y una galería de ejemplos reales de la clase predicha; aún no implementa esos cambios en la página.

Referencias: [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), [validación y selección de modelos](https://scikit-learn.org/stable/modules/cross_validation.html).


## Evidencias y artefactos

- Seis pruebas de la extension aprobadas.
- Las 14 pruebas existentes del backend y las 10.503 comprobaciones C++ pasan.
- Build del frontend aprobado; pagina e imagen real de entrenamiento responden HTTP 200.
- 66 archivos de aplicacion y 38 artefactos padre conservan sus hashes.

- [Resultados JSON](../../runtime/experiments/resnet18_comparison/20261118T204643Z-8efb68c2/results.json)
- [Comparacion CV](../../runtime/experiments/resnet18_comparison/20261118T204643Z-8efb68c2/training_cv_comparison.csv)
- [Prueba historica reutilizada](../../runtime/experiments/resnet18_comparison/20261118T204643Z-8efb68c2/historical_test_comparison.csv)
- [Seleccion congelada desde CV](../../runtime/experiments/resnet18_comparison/20261118T204643Z-8efb68c2/cv_selection.json)
- [Verificacion del sistema existente](../../runtime/experiments/resnet18_comparison/20261118T204643Z-8efb68c2/system_verification.json)
- [Catalogo propuesto, no activo](../../runtime/experiments/resnet18_comparison/20261118T204643Z-8efb68c2/model_catalog_proposal.json)
- [Ejemplos de fotos reales por categoria](../../runtime/experiments/resnet18_comparison/20261118T204643Z-8efb68c2/gallery_examples_proposal.json)

![Estabilidad de validacion interna](../../runtime/experiments/resnet18_comparison/20261118T204643Z-8efb68c2/cv_stability.png)
