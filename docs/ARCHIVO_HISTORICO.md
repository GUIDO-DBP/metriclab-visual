# Materiales de estudio y antecedentes

Respaldo público: [archive/materiales-estudio-2026-11-20](https://github.com/GUIDO-DBP/metriclab-visual/tree/archive/materiales-estudio-2026-11-20), fijado en [`65839e4c4e3b31f92b77f1c3a1965e12edc88d62`](https://github.com/GUIDO-DBP/metriclab-visual/commit/65839e4c4e3b31f92b77f1c3a1965e12edc88d62). No se reescribió el historial.

| Material | Acceso y motivo |
| --- | --- |
| Diez manuales Markdown/PDF, bibliografía, figuras e informe preliminar | [documentacion_entrega](https://github.com/GUIDO-DBP/metriclab-visual/tree/archive/materiales-estudio-2026-11-20/documentacion_entrega). Material de estudio; el informe preliminar no debe confundirse con la entrega final. |
| Reporte HTML sintético | [MetricLab_reporte_demo.html](https://github.com/GUIDO-DBP/metriclab-visual/blob/archive/materiales-estudio-2026-11-20/MetricLab_reporte_demo.html). Etapa anterior de búsqueda; sus cifras no son resultados de Caltech-101. |
| Propuesta antigua | [Plan_Proyecto_MetricLab_Visual_APA7.docx](https://github.com/GUIDO-DBP/metriclab-visual/blob/archive/materiales-estudio-2026-11-20/Plan_Proyecto_MetricLab_Visual_APA7.docx). Objetivos preliminares. |
| Generadores de manuales/PDF y auditorías inicial/PDF | [scripts](https://github.com/GUIDO-DBP/metriclab-visual/tree/archive/materiales-estudio-2026-11-20/scripts) y [docs](https://github.com/GUIDO-DBP/metriclab-visual/tree/archive/materiales-estudio-2026-11-20/docs). No son dependencias de ejecución, instaladores ni pruebas vigentes. |

El ZIP local `documentacion_entrega.zip` contiene 93 archivos coincidentes byte por byte con archivos locales, sin adicionales ni diferencias. Ya estaba excluido de Git y no pertenece a Releases. Se conserva localmente sin duplicarlo en la entrega.

Los materiales retirados del índice permanecen físicamente en el workspace del autor, ignorados. Un clon nuevo recibe solo la entrega; el compañero puede descargar el respaldo. Para recuperar los materiales sin cambiar el checkout actual:

```powershell
git fetch origin
git worktree add ..\metriclab-materiales origin/archive/materiales-estudio-2026-11-20
```

Los generadores históricos se ejecutan desde ese checkout con sus requisitos originales; necesitan sus recursos runtime, no asumen los de otro directorio. Guías vigentes: [instalación](INSTALACION_WINDOWS.md), [arquitectura](ARQUITECTURA.md) y [metodología](METODOLOGIA_RESULTADOS.md). No se generó un nuevo informe APA.

La descripción histórica de la Release menciona la antigua ruta de instalación en `documentacion_entrega/`. No se modifica esa Release: para instalar se utiliza el README actual; la guía antigua sigue accesible en el respaldo enlazado arriba.
