# LightGBM + ResNet-18: diagnóstico, mejora e integración

Ejecución real `20261118T211652Z-600d0ec2`, sobre los embeddings originales, sin reentrenar la CNN. Se conservaron los experimentos ResNet `20261118T195917Z-42d3213e` y comparación `20261118T204643Z-8efb68c2`, así como todos los modelos HSV.

## Integridad y diagnóstico

Se verificaron SHA de vectores e imágenes, alineación fila/imagen/etiqueta y el manifiesto original: **6.941 × 512** para entrenamiento, **1.736 × 512** para prueba histórica, float32 finito, 101 categorías y códigos contiguos 0–100. `n_features_in_` y `classes_` del LightGBM original coinciden. No se encontró intercambio de descriptor ni de etiquetas. El chequeo de integridad lee ambos conjuntos; las filas de prueba no se usan para el diagnóstico ni para seleccionar parámetros. El campo histórico `test_opened: false` de `diagnosis.json` se refiere al diagnóstico, no a la auditoría previa de integridad.

El ajuste original usaba 15 hojas, profundidad 6, mínimo 20 muestras, learning rate 0,1 y 80 etapas, pero `min_child_weight=0.001` y ninguna regularización L1/L2. En el primer fold de desarrollo, accuracy alcanzaba **60,89 % en la etapa 10**, luego caía a aproximadamente **13,56 % en la 20**. El máximo margen bruto pasaba de 135 a **37.373**. Al final, la probabilidad máxima media era **0,99939**, aunque las predicciones eran deficientes. Esto muestra actualizaciones desproporcionadas y saturación, no ausencia de información en el embedding.

En la confirmación interna reservada, el original terminó prediciendo solo **30 clases**; `Leopards` recibió 680 de 1.389 predicciones y `watch` 366. El máximo margen fue **81.799,80**. El modelo seleccionado predijo las **101 clases**, con margen máximo **9,01** y probabilidad máxima media **0,80857**. Estas probabilidades siguen sin calibrar.

Las curvas de confirmación original muestran pérdida de entrenamiento **1,5325 → 33,2383** y validación **2,6084 → 33,0340**. Tras regularizar: entrenamiento **3,5532 → 0,0283** y validación **3,6016 → 0,5191**. Persiste una brecha de generalización; la solución no elimina el sobreajuste.

Se capturaron **6.718** mensajes `No further splits with positive gain` en el diagnóstico original y **12.062** en la confirmación seleccionada. El segundo ajusta más etapas y esas cifras no son tasas comparables. La advertencia indica restricciones para seguir dividiendo; por sí sola no explica el colapso, pues también aparece en el modelo preciso. Se registró además la deprecación de `eval_set` en LightGBM 4.7.0; no impidió el ajuste. Los logs completos quedan junto a cada curva JSON.

La evidencia es compatible con inestabilidad de los valores de hoja en este problema multiclase. La documentación oficial describe `min_child_weight` como el mínimo Hessiano por hoja y `reg_lambda` como regularización L2 ([LGBMClassifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)). Las modificaciones fueron conjuntas: **no se ha demostrado por ablación qué parámetro individual causó o resolvió el problema**.

![Curvas reales de entrenamiento y validación](../../runtime/experiments/lightgbm_resnet/20261118T211652Z-600d0ec2/training_curves.png)

## Selección exclusivamente con entrenamiento

Estratificación fija: 5.552 imágenes de desarrollo y 1.389 de confirmación reservada. Siete configuraciones prefijadas, tres folds en desarrollo, selección por F1 macro medio; balanced accuracy como desempate. Sin early stopping ni consulta de métricas de prueba para seleccionar. La confirmación se abrió después de guardar `selection.json`. No se eligió una nueva configuración a partir de ella.

| Candidato | Hojas / profundidad / mínimo muestras | LR / etapas | Hessiano mínimo / L1 / L2 | Accuracy CV | F1 macro CV ± DE |
| --- | --- | --- | --- | ---: | ---: |
| 0 original | 15 / 6 / 20 | 0,10 / 80 | 0,001 / 0 / 0 | 34,29 % | 0,2693 ± 0,3195 |
| 1 | 15 / 6 / 20 | 0,05 / 120 | 0,1 / 0 / 1 | 83,74 % | 0,7672 ± 0,0080 |
| **2 seleccionado** | **15 / 6 / 20** | **0,10 / 120** | **1 / 0,1 / 5** | **85,03 %** | **0,7858 ± 0,0059** |
| 3 | 31 / 8 / 20 | 0,05 / 160 | 1 / 0,1 / 5 | 84,19 % | 0,7716 ± 0,0096 |
| 4 | 7 / 3 / 40 | 0,10 / 120 | 1 / 0,1 / 5 | 85,23 % | 0,7851 ± 0,0047 |
| 5 | 31 / sin límite / 10 | 0,05 / 160 | 0,1 / 0 / 1 | 83,70 % | 0,7638 ± 0,0090 |
| 6 | 15 / 6 / 40 | 0,03 / 240 | 1 / 0,5 / 10 | 83,41 % | 0,7612 ± 0,0091 |

Los candidatos 2 y 4 están próximos; no se afirma superioridad estadística. La DE entre folds no es un intervalo de confianza. La búsqueda fue pequeña y controlada, no exhaustiva. Balanced accuracy CV seleccionada: **78,27 %**. La CV anterior de cinco folds utilizaba más datos por ajuste y otro protocolo: sus cifras no se sustituyen ni se mezclan con esta búsqueda.

| Confirmación interna, 1.389 filas de entrenamiento | Accuracy | Balanced accuracy | F1 macro |
| --- | ---: | ---: | ---: |
| Original | 4,32 % | 2,73 % | 0,0302 |
| Seleccionado | **87,62 %** | **81,98 %** | **0,8224** |

Esta confirmación aporta evidencia de mejora sin usar la prueba histórica para escoger parámetros. Las métricas originales dependen mucho de las filas usadas para ajustar; el 4,32 % no sustituye al 19,76 % histórico.

Se revisaron las 101 categorías. Entre las más débiles en confirmación seleccionada: `anchor` y `water_lilly` F1 0,20 (7 y 6 casos), `ant` 0,3333 (7), `lobster` y `octopus` 0,4615 (7 y 6). Los soportes pequeños limitan las conclusiones por clase. `internal_confirmation.json` y `results.json` conservan precision, recall, F1, soporte y distribución de predicciones de todas las categorías. No se retocó el modelo según las clases difíciles de prueba.

## Prueba histórica reutilizada

Solo después de congelar la selección y confirmar la mejora se ajustó el modelo final con las 6.941 imágenes. Se volvieron a medir las **mismas 1.736 imágenes históricas**: **no es una evaluación independiente nueva**. Accuracy **87,33 %**, balanced accuracy **80,91 %**, F1 macro **0,8160**, frente a **19,76 % / 10,32 % / 0,1139** del LightGBM original. Entrenamiento final **48,617 s**, inferencia media **0,687 ms**, sobre 128 consultas tras calentamiento, excluyendo CNN, IO y HTTP. Son tiempos de esta ejecución, no una comparación simultánea con los otros modelos.

Se recomienda incorporar el descriptor conservando XGBoost, Random Forest y el LightGBM estabilizado como opciones. La diferencia histórica entre LightGBM y XGBoost es pequeña y no justifica declarar un ganador nuevo. Para una comparación confirmatoria futura se necesita un conjunto externo que no haya intervenido en este ciclo de exploración.

Modelo nuevo: `runtime/experiments/lightgbm_resnet/20261118T211652Z-600d0ec2/model.joblib`; SHA256 `7768e11659ce2d250d7470d2516093646ed3fa2db4d77899b8757af10582bc83`. También se guarda `lightgbm.txt`. Configuración completa en `selection.json` y `results.json`. Los modelos anteriores siguen intactos.
