# 05. Entrenamiento, validación y persistencia

[TOC]

## 5.1. Aprender no es inferir

Entrenar significa construir parámetros o reglas a partir de ejemplos etiquetados. Inferir significa aplicar un modelo ya aprendido a una entrada. Instalar restaura esos modelos y comprueba compatibilidad: no es necesario repetir el entrenamiento para abrir MetricLab.

En los árboles, el ajuste decide variables, umbrales y distribuciones de hojas. El bosque ajusta muchos árboles con bootstrap y selección de variables. Boosting incorpora árboles según gradientes de una pérdida. La CNN ResNet ya estaba preentrenada y permaneció congelada; entrenar un clasificador con sus activaciones no equivale a volver a entrenar esa CNN.

## 5.2. División original y trazabilidad

Los 6.941 registros de entrenamiento y 1.736 de prueba conservan nombres, IDs y contenido. HSV y ResNet describen las mismas imágenes en el mismo orden dentro de cada split. Guardar una matriz sin conservar su lista de filas sería insuficiente: una permutación de etiquetas puede producir un entrenamiento formalmente válido y semánticamente incorrecto.

Cada experimento conserva protocolo, parámetros, versiones, predicciones y resultados. Las nuevas ejecuciones crean carpetas versionadas, no reemplazan los modelos activos por tener nombres parecidos. RF/XGBoost ResNet completos se reutilizaron al ampliar la comparación. Los clones de validación cruzada sí se ajustaron de nuevo porque cada fold requiere un modelo que no haya visto su validación.

## 5.3. Validación cruzada por folds

En cinco folds, el entrenamiento se divide en cinco partes estratificadas. Se ajusta un modelo con cuatro y se predice la restante; luego cambia la parte reservada. Al finalizar, cada imagen tiene una predicción out-of-fold de un modelo que no la utilizó para aprender. El resultado se resume por media y desviación muestral entre folds.

Ejemplo pequeño: 100 ejemplos repartidos en cinco partes de 20. En cada ajuste se usan 80 y se evalúan 20. Los cinco conjuntos de ajuste se solapan, por lo que sus resultados no son cinco experimentos totalmente independientes. La desviación expresa dispersión observada, no un intervalo de confianza automático.

En la comparación ResNet original se usó `StratifiedKFold(n_splits=5, shuffle=True, random_state=23)`. Las cinco configuraciones —dos criterios y tres conjuntos de árboles— compartieron índices. Se mantuvieron parámetros fijos y no se utilizó early stopping. XGBoost obtuvo el mayor F1 macro medio, 0,8170 ± 0,0137; el LightGBM original, 0,1700 ± 0,0434.

## 5.4. Búsqueda controlada de LightGBM

La investigación posterior separó entrenamiento en **5.552 desarrollo y 1.389 confirmación**, estratificando con semilla 41. Dentro de desarrollo se crearon tres folds con semilla 29. Se definieron siete configuraciones antes de medirlas. La regla de selección fue mayor F1 macro medio y, en empate, balanced accuracy.

La selección se guardó en `selection.json` antes de evaluar confirmación. La configuración 2 logró F1 medio 0,785848 y DE 0,005886. Después se ajustaron original y seleccionado con los 5.552 registros de desarrollo y se compararon una sola vez sobre los 1.389 reservados. Esos resultados confirmaron una mejora; no se volvieron a usar para buscar otra configuración.

Finalmente se ajustó el seleccionado con los 6.941 registros completos y se volvió a medir la prueba histórica. Los tiempos de búsqueda y los ajustes internos no son el tiempo de ajuste final de 48,617 s. Mezclarlos produciría una comparación de costos engañosa.

## 5.5. Evitar fuga de información

La separación no debe usarse únicamente al final del script. Se mantiene durante selección de parámetros, transformaciones que aprendan de datos, calibración y decisiones de versión. En este proyecto el extractor ResNet está fijo: calcular previamente todas sus activaciones no usa las etiquetas ni ajusta estadísticas con Caltech. Si se afinara la red o se aprendiera un normalizador con todos los splits, el razonamiento cambiaría y habría que situar ese ajuste dentro de cada entrenamiento correspondiente.

Los chequeos de integridad sí leen vectores, rutas y etiquetas de prueba para verificar correspondencia; no calculan con ellos un criterio de selección. Una comprobación de checksum no equivale a entrenar. Sin embargo, observar repetidamente accuracy de prueba para decidir el enfoque general sí introduce un antecedente de exploración que debe reconocerse.

No se usaron las 1.736 imágenes para elegir los parámetros nuevos de LightGBM. Eso no convierte la prueba completa en inédita: había sido consultada durante etapas anteriores. La CV es interna y condicional al descriptor previamente elegido. Una evaluación externa futura debe planificarse antes de observar sus resultados.

## 5.6. Guardar y cargar modelos

Los bundles joblib contienen estimador, lista de clases y metadatos. XGBoost y LightGBM también conservan formatos nativos para controles de equivalencia. El registro apunta a rutas relativas dentro de `runtime/`, con SHA-256. Al inferir se verifican identidad del descriptor, dimensión y clases; el alias histórico `resnet18` se admite únicamente para bundles identificados por el registro y sus hashes.

Los archivos joblib usan mecanismos de serialización que pueden ejecutar código al cargar. **Solo se deben cargar paquetes de procedencia confiable cuyo hash coincida con el catálogo obtenido del repositorio esperado.** El hash detecta alteraciones respecto de ese catálogo, no acredita que un desconocido sea confiable. No se permite al usuario subir modelos desde la página.

El entorno importa: sklearn no garantiza persistencia compatible entre versiones diferentes. La entrega conserva dependencias fijadas en `requirements-lock.txt`. Si se cambia Python o las bibliotecas, primero hay que comprobar carga y predicciones; no se soluciona un fallo entrenando silenciosamente nuevos modelos durante la instalación.

## 5.7. CPU, CUDA y tiempos

La CNN puede extraer en CUDA cuando una operación real comprueba que funciona, y usa CPU cuando no hay CUDA adecuada en modo automático. Los clasificadores guardados se ejecutan sobre CPU con sus parámetros de hilos. Tener GPU no convierte automáticamente todo el pipeline en GPU.

La primera extracción incluye cargar PyTorch, pesos y red. El costo de una imagen con el extractor caliente es distinto. La aplicación separa extracción, `predict_proba` y total de solicitud. El total puede incluir carga de un clasificador que no estaba en caché. No se suman tiempos de distintas ejecuciones para prometer una latencia de interfaz.

## 5.8. Repetir experimentos de forma opcional

Primero instalar los paquetes models y research, las imágenes y el entorno experimental completo. Este último requiere más bibliotecas que el worker mínimo. Los siguientes comandos son de investigación y **sí realizan trabajo de entrenamiento**, por lo que no forman parte del arranque normal:

```powershell
# Dentro del entorno experimental compatible, instalar requisitos de experiments.
.\runtime\resnet-env\Scripts\python.exe -m pip install -r experiments/resnet18/requirements.txt lightgbm==4.7.0
.\runtime\resnet-env\Scripts\python.exe -m experiments.resnet18_comparison.run --help
.\runtime\resnet-env\Scripts\python.exe -m experiments.lightgbm_resnet.run --comparison runtime/experiments/resnet18_comparison/20261118T204643Z-8efb68c2
```

Consultar el README de cada experimento antes de ejecutar: conserva comandos, requisitos y selección de directorios. No se modifica el algoritmo para reproducir el informe. Los tests históricos que comparan hashes del código de aplicación reflejan el estado anterior a la integración autorizada y no deben interpretarse como un requisito de que el frontend actual siga idéntico al de ese día.

## 5.9. Reproducibilidad y límites

Repetir el método requiere las mismas filas, etiquetas, pesos, parámetros y versiones. Las semillas fijan fuentes de aleatoriedad contempladas, pero no prometen el mismo tiempo en otro equipo. Algunos detalles numéricos o de ejecución pueden variar. El proyecto guarda predicciones para recalcular métricas y hashes para distinguir artefactos, en lugar de depender únicamente de una captura de pantalla.

**Actividad:** explica por qué evaluar un modelo ya entrenado con las 6.941 filas dentro de un fold de esas mismas filas no constituye validación cruzada. Después indica qué archivos verificarías si dos estudiantes obtienen accuracy distinta con el «mismo» modelo.

## Referencias

Documentación primaria: [validación cruzada sklearn](https://scikit-learn.org/stable/modules/cross_validation.html), [persistencia de modelos](https://scikit-learn.org/stable/model_persistence.html). Protocolos del proyecto: `experiments/resnet18_comparison/` y `experiments/lightgbm_resnet/`.
