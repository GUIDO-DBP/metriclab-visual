# Verificación de los PDF

Fecha: 19 de noviembre de 2026. Se generaron once PDF reales mediante Markdown → HTML independiente → impresión PDF de Edge/Chromium. No se cambiaron extensiones para simular documentos.

| Documento | Páginas | Bytes | Hallazgos geométricos |
| --- | ---: | ---: | ---: |
| 01_INTRODUCCION_GENERAL.pdf | 4 | 173118 | 0 |
| 02_DATASET_CALTECH101.pdf | 5 | 1895669 | 0 |
| 03_EXTRACCION_CARACTERISTICAS.pdf | 6 | 277099 | 0 |
| 04_ALGORITMOS_CLASIFICACION.pdf | 6 | 282911 | 0 |
| 05_ENTRENAMIENTO_MODELOS.pdf | 4 | 117167 | 0 |
| 06_ESTADISTICAS_RESULTADOS.pdf | 6 | 402944 | 0 |
| 07_ARQUITECTURA_CODIGO.pdf | 5 | 218781 | 0 |
| 08_MANUAL_USUARIO.pdf | 8 | 826847 | 0 |
| 09_INSTALACION_EJECUCION.pdf | 5 | 143215 | 0 |
| 10_EXPERIMENTOS_CONCLUSIONES.pdf | 5 | 221910 | 0 |
| INFORME_GENERAL_METRICLAB_VISUAL.pdf | 21 | 404654 | 0 |

Los diez manuales suman 54 páginas; el informe académico, 21. Se conserva el Markdown editable. El informe usa portada con marcadores, Times New Roman 12, interlineado doble, márgenes de una pulgada, paginación superior, jerarquía, tablas/figuras y referencias con sangría francesa según la presentación académica APA 7. Los campos personales e institucionales deben completarse antes de presentar.

Controles: imágenes cargadas en todos los HTML; tablas y preformateados sin desbordamiento horizontal en el ancho de impresión; apertura y extracción de texto de todas las páginas; detección de páginas vacías, caracteres de sustitución y texto fuera de página. Se comprobaron enlaces y no hay enlaces privados file://. Se generaron hojas de contacto de todas las páginas y se inspeccionaron visualmente, incluyendo manual de usuario e informe.

Las fórmulas se renderizaron con MathText a alta resolución; las capturas son reales y las figuras provienen de manifiestos/resultados. No se observaron textos cortados ni tablas truncadas en los controles descritos. Esto no equivale a una certificación editorial exhaustiva: una revisión académica final debe completar autores y adaptar requisitos particulares del docente.

Reproducir: scripts/build_documentation.py, scripts/build_pdfs.mjs y scripts/check_pdfs.py. Archivos temporales y contactos en documentacion_entrega/qa quedan excluidos; PDF definitivos y fuentes sí se publican.
