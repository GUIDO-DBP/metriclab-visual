# Documentación de entrega de MetricLab Visual

Fuentes editables y PDF reales generados desde ellas. Los diez manuales suman 54 páginas y el informe académico tiene 21. El inventario y los controles se registran en `docs/VERIFICACION_PDF.md`.

| Tema | Markdown | PDF |
| --- | --- | --- |
| 01 Introducción | [Leer](01_INTRODUCCION_GENERAL.md) | [PDF](pdf/01_INTRODUCCION_GENERAL.pdf) |
| 02 Dataset | [Leer](02_DATASET_CALTECH101.md) | [PDF](pdf/02_DATASET_CALTECH101.pdf) |
| 03 Características | [Leer](03_EXTRACCION_CARACTERISTICAS.md) | [PDF](pdf/03_EXTRACCION_CARACTERISTICAS.pdf) |
| 04 Clasificadores | [Leer](04_ALGORITMOS_CLASIFICACION.md) | [PDF](pdf/04_ALGORITMOS_CLASIFICACION.pdf) |
| 05 Entrenamiento | [Leer](05_ENTRENAMIENTO_MODELOS.md) | [PDF](pdf/05_ENTRENAMIENTO_MODELOS.pdf) |
| 06 Métricas/resultados | [Leer](06_ESTADISTICAS_RESULTADOS.md) | [PDF](pdf/06_ESTADISTICAS_RESULTADOS.pdf) |
| 07 Arquitectura | [Leer](07_ARQUITECTURA_CODIGO.md) | [PDF](pdf/07_ARQUITECTURA_CODIGO.pdf) |
| 08 Manual de usuario | [Leer](08_MANUAL_USUARIO.md) | [PDF](pdf/08_MANUAL_USUARIO.pdf) |
| 09 Instalación | [Leer](09_INSTALACION_EJECUCION.md) | [PDF](pdf/09_INSTALACION_EJECUCION.pdf) |
| 10 Experimentos | [Leer](10_EXPERIMENTOS_CONCLUSIONES.md) | [PDF](pdf/10_EXPERIMENTOS_CONCLUSIONES.pdf) |
| Informe académico | [Fuente editable](INFORME_GENERAL_METRICLAB_VISUAL.md) | [Informe PDF](INFORME_GENERAL_METRICLAB_VISUAL.pdf) |

[Bibliografía APA 7](BIBLIOGRAFIA.md) · [Licencias y atribuciones](../docs/LICENCIAS_Y_RECURSOS.md) · [Auditoría de distribución](../docs/AUDITORIA_GITHUB.md)

Los campos de nombres, docente e institución son marcadores deliberados; completar antes de presentar. Las capturas son de la aplicación real y las figuras numéricas proceden de sus artefactos. Los ejemplos de cálculo didácticos se identifican como tales. La prueba histórica no se presenta como evaluación nueva.

## Regenerar documentación

Opcional; no necesario para ejecutar la aplicación. Instalar las dependencias de `requirements-docs.txt` en un entorno de documentación y usar el entorno backend para figuras. Playwright y Edge deben estar disponibles para imprimir.

```powershell
.\.venv\Scripts\python.exe -m scripts.documentation_assets
node scripts/capture_manual.mjs
.\.venv\Scripts\python.exe -m scripts.assemble_report
python -m scripts.build_documentation
node scripts/build_pdfs.mjs
python -m scripts.check_pdfs
```

Los comandos de figuras requieren los paquetes de evidencia y las imágenes. La captura requiere una instancia local. `build_documentation` y `check_pdfs` deben ejecutarse con el intérprete del entorno que tenga Markdown/PyMuPDF. La carpeta `.build/` contiene HTML temporal y `qa/` contactos de revisión; ambas se excluyen de Git. Los PDF finales y assets seleccionados sí se conservan.
