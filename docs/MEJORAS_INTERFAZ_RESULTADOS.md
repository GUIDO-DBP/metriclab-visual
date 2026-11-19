# Mejoras controladas de interfaz e interpretación

Fecha: 19 de noviembre de 2026. Cambios incrementales sobre MetricLab Visual; se conservan algoritmos, modelos, descriptores, métricas, instalación y laboratorio C++.

## Contexto de la evaluación

El aviso naranja se sustituye por un bloque informativo azul/violeta. Mantiene visibles el número de imágenes y el hecho de que el conjunto ya se utilizó en experimentos anteriores. La frase «Estos resultados no son una evaluación independiente nueva» sigue visible incluso con la explicación cerrada. «Consultar explicación metodológica» abre el protocolo guardado y explica la diferencia entre prueba histórica y validación interna. No se cambian los datos ni se oculta la limitación.

La gráfica y la tabla conservan sus métricas. El eje utiliza Gini, Entropía, RF, XGB y LGBM para evitar etiquetas superpuestas en móvil; la leyenda explicativa y el tooltip mantienen los nombres completos. Se suprime la animación de entrada de las barras para que no aparezcan temporalmente incompletas al cambiar tamaño o descriptor.

## Interpretación automática y alcance

`frontend/src/evaluationAnalysis.ts` calcula resúmenes transitorios, sin mutar la respuesta API. Se muestran los máximos de accuracy y F1 macro, incluyendo empates. Una regla visible considera cercanos dos resultados si la diferencia de accuracy es de hasta **1 punto porcentual** y la de F1 macro de hasta **0,01**, simultáneamente. Es una regla de presentación; no es una prueba estadística, intervalo de confianza, criterio de entrenamiento ni selección de modelos.

La interpretación utiliza exclusivamente las métricas del descriptor seleccionado. En ResNet, los datos guardados tienen mayor accuracy en LightGBM y mayor F1 macro en XGBoost. Se describen estas observaciones sin declarar superioridad estadística. Los resultados HSV y ResNet permanecen separados, y la pestaña de validación interna conserva sus propios protocolos.

La vista se reinicia al cambiar descriptor o alcance de evaluación y solo admite un fallback cuyo descriptor y alcance coincidan. El explorador también reinicia su contexto al cambiar descriptor para no mostrar momentáneamente datos de la selección anterior.

Las categorías difíciles y la confusión más frecuente corresponden al modelo elegido en «Modelo evaluado», identificado en el texto. No se construye un promedio ficticio de matrices de diferentes clasificadores.

## Resumen y matriz completa

Desde cada matriz real se calculan soporte por fila, aciertos de diagonal, recall, errores y pares direccionales de confusión. Se muestran cinco categorías con mayor recall y cinco con menor recall, incluyendo **aciertos / imágenes**; los empates se ordenan por nombre y las clases sin soporte no se clasifican por recall. También aparecen los cinco errores direccionales más frecuentes por conteo y el soporte de su categoría de origen.

Un desplegable incluye soporte, aciertos y recall de las 101 clases. Se conserva la matriz completa de 101 × 101, con desplazamiento y filtro visual. Los resúmenes se calculan con todas las clases y no cambian al filtrar. La matriz utiliza cantidades de imágenes: el color se escala respecto del mayor conteo de cada fila, no expresa porcentajes por sí mismo.

Una matriz con dimensiones incorrectas, valores negativos, fraccionarios o no finitos no produce un resumen inventado. Se comunica que el resumen no está disponible. Se distingue entre la frecuencia absoluta de errores y la proporción de fallos de una categoría pequeña.

## Clasificación y categoría verdadera

Se conservan la carga de archivos, los dos descriptores, cuatro algoritmos, criterios Gini/entropía, probabilidades sin calibrar y galería de seis fotografías train de la categoría predicha.

El componente de carga conserva las cuatro miniaturas y añade un selector de las muestras identificadas que ya devuelve la API. No se incorporan imágenes nuevas ni nuevas particiones. Al elegir una muestra, se descarga su imagen y se conserva su ID/categoría del manifiesto. Se muestra la etiqueta verdadera junto a la predicción y se indica acierto o error de ese caso individual.

La procedencia se conserva al cambiar modelo, criterio o descriptor de la misma fotografía. Al cargar o arrastrar otro archivo se borra: el nombre del archivo, la predicción y el parecido con ejemplos **no se utilizan para inferir la verdad**. Incluso un archivo externo con los mismos bytes que una muestra se considera desconocido si no fue seleccionado como muestra identificada. No se envía la etiqueta verdadera al clasificador.

Las fotos de la galería siguen siendo ejemplos de entrenamiento de la predicción, incluso cuando el modelo falla y la etiqueta verdadera es otra. El texto no las presenta como vecinos por similitud ni como prueba del acierto. La carga de muestras tiene estado de progreso y cancelación al salir del componente.

## Explorador real y legibilidad

Una guía desplegable explica raíz (nodo 0), característica, umbral, ramas izquierda ≤ y derecha >, impureza Gini/entropía y hoja. El inspector explica el nodo seleccionado con su condición y datos reales. Advierte que valores mostrados están redondeados y que la impureza no es una probabilidad de error externo.

En ResNet se habla de componentes numéricos del embedding, sin atribuirles partes visuales. En el bosque se diferencia la respuesta de un árbol individual del promedio de todos los árboles. Las rutas antiguas se invalidan al cambiar archivo/modelo/árbol/descriptor; una respuesta tardía de otro contexto no puede sustituir la ruta actual.

Se aumenta el contraste y tamaño de textos secundarios en evaluación y exploración, se ajustan espacios y tablas, y se mantiene la paleta existente. La matriz conserva cabeceras visibles, desplazamiento accesible por teclado y foco visible. Las tablas resumidas se adaptan a móvil y la tabla completa de soportes permite scroll. El diagrama mantiene lectura mediante desplazamiento y no comprime ilegiblemente sus etiquetas.

## Verificación y conservación

La compilación se comprueba tanto en el proyecto como en el checkout de instalación independiente, usando el mismo lock y sin dependencias nuevas. Las pruebas incluyen los cálculos matemáticos, las diez matrices reales (cinco configuraciones por descriptor), soporte contra el manifiesto, máximos, cercanía, filtros, navegación, galerías, etiquetas conocidas/desconocidas y descarte de rutas tardías.

Las capturas de escritorio y móvil se generan localmente bajo `runtime/ui_improvements/`, excluido de Git. Se comparan SHA256 de 229 archivos protegidos: resultados, vectores, modelos, checkpoint, registro, catálogo y locks. No se modifican ni reentrenan modelos ni se alteran recursos publicados en Releases. Los procedimientos de instalación y el motor complementario se mantienen.

Resultados de la verificación final:

| Comprobación | Resultado |
| --- | --- |
| Backend, suite existente | 25 pruebas aprobadas. |
| Playwright | 14 aprobadas: 10 flujos de navegador y 4 pruebas de cálculo/datos. |
| Compilación TypeScript/Vite | Correcta en el proyecto y en el checkout independiente; mismo bundle generado. |
| Matrices | Las diez matrices reales mantienen 1.736 imágenes y los soportes del manifiesto; accuracy y balanced accuracy coinciden con los valores guardados. |
| Categoría verdadera | Caso correcto y caso incorrecto reales; sustitución por archivo externo elimina la etiqueta conocida. |
| Responsive | Escritorio y 390 × 844; matriz navegable por teclado y sin desbordamiento horizontal de página. |
| Preservación local | 229 hashes comprobados; cero modificaciones. |
| Releases | Verificación de lectura: tamaños y digests originales conservados. No se suben assets nuevos. |
| Publicación | Sin dependencias nuevas ni archivos temporales seleccionados; commit de interfaz separado. |

Se utilizó el entorno independiente verificado en la entrega anterior para recompilar y revisar la aplicación. No se repitió una instalación completamente nueva ni se probó otro equipo físico. Continúa la advertencia de deprecación de Starlette/TestClient sobre httpx y el aviso NO_COLOR del runner, sin errores de ejecución ni fallos en las pruebas.

## Actualizar una instalación existente

No es necesario descargar modelos, dataset, embeddings ni pesos nuevamente. Conservar el entorno y los recursos instalados, detener la instancia anterior si ocupa el puerto y ejecutar:

```powershell
git pull --ff-only
cd frontend
npm.cmd run build
cd ..
.\scripts\start.ps1
```

En el proyecto local que ya contiene esta entrega basta recargar la página después de la compilación. El proceso de instalación desde cero sigue siendo el mismo de `09_INSTALACION_EJECUCION.md`.

## Límites que continúan

La prueba es histórica reutilizada. La cercanía observada no demuestra equivalencia o superioridad estadística. Cinco casos mostrados no agotan los empates ni todas las confusiones. Soportes pequeños hacen más incierto interpretar un recall alto. Las probabilidades no están calibradas y un caso acertado no sustituye las métricas globales.

La verdad visible solo corresponde a muestras identificadas por la aplicación. No se implementa reconocimiento automático de etiquetas verdaderas de archivos externos. Las características del embedding no tienen semántica visual demostrada. La verificación en otro directorio del mismo equipo no certifica otra computadora física.
