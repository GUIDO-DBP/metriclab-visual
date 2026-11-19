# Publicación GitHub comprobada

Fecha: 19 de noviembre de 2026. No se desplegó una web en Netlify ni otro servidor.

## Repositorio y conservación

Repositorio: https://github.com/GUIDO-DBP/metriclab-visual

El usuario creó el repositorio público y vacío. Se conservó esa visibilidad. Se publicó en `main` el historial local existente, incluido el respaldo `621253d` y el commit de entrega `3068c78`, seguido de comprobaciones y documentación final. La rama local de trabajo sigue siendo `feature/clasificacion-u2`. **No hubo force push** ni cambios en otros repositorios.

Se realizó un clon completo desde GitHub en `runtime/public-clone-check/`; descargó los archivos del repositorio, incluidos los once PDF, y conservó el historial. Los once PDF clonados se abrieron y pasaron de nuevo los controles de páginas. No se copió `.git` manualmente ni se sustituyó el historial por un repositorio nuevo.

## Release y descargas reales

Release publicada: https://github.com/GUIDO-DBP/metriclab-visual/releases/tag/resources-v1

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| [models v2](https://github.com/GUIDO-DBP/metriclab-visual/releases/download/resources-v1/metriclab-models-2026-11-19-v2.zip) | 77.552.623 | `af02460acb86b34584b8f43dce5b17b2ae97723fcee31038ccfc899025c7ca3e` |
| [research v2](https://github.com/GUIDO-DBP/metriclab-visual/releases/download/resources-v1/metriclab-research-2026-11-19-v2.zip) | 90.779.931 | `ec299abf9a93ed9ba4b4f23b150be77bc0fea13b1f01aaaf9cbc67bd97f3a17a` |
| [SHA256SUMS](https://github.com/GUIDO-DBP/metriclab-visual/releases/download/resources-v1/SHA256SUMS.txt) | 206 | `68a65d27f885f5d8fd02f10fe64798487c7edc79d8d2aa68a2e7c69a96a68300` |

GitHub informó los tres assets como `uploaded`; sus tamaños y digests coinciden con el catálogo. Además de consultar metadatos, el instalador descargó **ambos ZIP desde la Release pública en el clon remoto**, verificó sus SHA256 y validó/instaló 24 y 172 archivos respectivamente. No se limitó la comprobación a que existiera un enlace.

Los ZIP preliminares v1 no se publican: contenían logs históricos con rutas privadas. El catálogo final es `2026-11-19-v2`; el tag `resources-v1` identifica la primera Release pública, con esos paquetes v2 revisados. Los originales locales se conservaron.

## Comandos para el compañero

```powershell
git clone https://github.com/GUIDO-DBP/metriclab-visual.git
cd metriclab-visual
.\scripts\setup.ps1
.\scripts\download_resources.ps1 -Repository GUIDO-DBP/metriclab-visual -Weights
.\.venv\Scripts\python.exe -m scripts.dataset_resources --download
.\scripts\verify_installation.ps1 -Smoke
.\scripts\start.ps1
```

`research` es opcional para inferencia normal; se obtiene con `download_resources.ps1 -Kind research -Repository GUIDO-DBP/metriclab-visual`. El dataset se descarga de CaltechDATA y el checkpoint desde PyTorch, con hashes. No se redistribuyen completos en Git ni se incluyen entornos. No se necesita entrenar para iniciar.

## Documentación y límites

Diez manuales Markdown/PDF: 54 páginas. Informe académico editable y PDF: 21 páginas, con portada de campos pendientes, referencias y presentación APA 7. Véase `documentacion_entrega/README.md`. Los autores, docente e institución deben completarse antes de presentar.

La prueba funcional independiente se hizo en un segundo directorio del mismo equipo, con entornos nuevos sin paquetes globales y torch CPU: 25 tests backend, 7 Edge y 10.503 comprobaciones C++ correctas. Esto no demuestra funcionamiento en otra computadora física o GPU. Véase `PRUEBA_INSTALACION_LIMPIA.md`.

Los datos de precisión siguen siendo prueba histórica reutilizada y validación interna según sus protocolos. Publicar el código no convierte esas cifras en una nueva validación. No se reentrenó ni modificó ningún algoritmo para esta entrega. Las licencias y condiciones de datos, bibliotecas, pesos y código propio se distinguen en `LICENCIAS_Y_RECURSOS.md`.
