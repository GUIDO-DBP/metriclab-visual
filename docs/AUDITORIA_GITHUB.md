# Auditoría de preparación GitHub

Fecha: 19 de noviembre de 2026. Repositorio de destino: `GUIDO-DBP/metriclab-visual`; rama local `feature/clasificacion-u2`. Se conserva el commit anterior `621253d` y el historial, sin force push. Se hizo una copia ZIP del código antes de modificarlo, en `runtime/github_preparation_backup/`, fuera de la publicación.

## Contenido seleccionado para Git

| Grupo | Archivos | MB decimales | Decisión |
| --- | ---: | ---: | --- |
| (raíz) | 12 | 0.386 | Código, metadatos pequeños o documentación seleccionada. |
| backend | 6 | 0.041 | Código, metadatos pequeños o documentación seleccionada. |
| docs | 11 | 0.053 | Código, metadatos pequeños o documentación seleccionada. |
| documentacion_entrega | 57 | 9.765 | Código, metadatos pequeños o documentación seleccionada. |
| experiments | 29 | 0.151 | Código, metadatos pequeños o documentación seleccionada. |
| frontend | 21 | 0.218 | Código, metadatos pequeños o documentación seleccionada. |
| metriclab | 23 | 0.125 | Código, metadatos pequeños o documentación seleccionada. |
| resources | 6 | 3.002 | Código, metadatos pequeños o documentación seleccionada. |
| scripts | 22 | 0.091 | Código, metadatos pequeños o documentación seleccionada. |
| tests | 5 | 0.028 | Código, metadatos pequeños o documentación seleccionada. |

El inventario exacto puede variar al añadir este informe; `resources/distribution_audit.json` registra la última pasada. No se añaden archivos por debajo del límite de Git solo porque quepan. PDF, capturas definitivas y figuras educativas se conservan; las temporales de QA no.

## Grupos excluidos y recuperación

| Grupo local inicial | Tamaño medido antes de las pruebas nuevas | Motivo / recuperación |
| --- | ---: | --- |
| .venv | 208.6 MB | Entorno ligado al equipo. Rehacer con setup y lock; no usa paquetes globales en una instalación nueva. |
| frontend | 138.5 MB | El total inicial incluye node_modules y dist. Solo se publica código/config/lock; npm ci y build regeneran dependencias y compilación. |
| dataset | 138.6 MB | 138,6 MB de fotografías/archivos originales. Descargar Caltech oficial, verificar archivo e imágenes; no se copia completo a Git. |
| runtime | 5951.9 MB | Incluye entornos, cachés y artefactos esenciales. No se borra: los recursos necesarios se empaquetan por allowlist, el resto queda local. |

Se excluyen también .git del contenido de trabajo, cachés Python/pip/npm, binarios compilados, .env reales, archivos temporales, logs, copias de seguridad, ruedas y checkpoints descargables. El historial Git se transfiere por Git, no como copia manual de .git. Las reglas están en `.gitignore` y `.gitattributes`.

## Paquetes de recursos

| Paquete | Archivos | Descarga MB | Tamaño sin comprimir MB | SHA256 |
| --- | ---: | ---: | ---: | --- |
| models | 24 | 77.553 | 105.507 | `af02460acb86b34584b8f43dce5b17b2ae97723fcee31038ccfc899025c7ca3e` |
| research | 172 | 90.780 | 125.215 | `ec299abf9a93ed9ba4b4f23b150be77bc0fea13b1f01aaaf9cbc67bd97f3a17a` |

`models` contiene el bundle HSV activo, formatos nativos, manifiesto, vectores HSV, registro y cinco modelos ResNet con métricas/protocolos que utiliza la API. `research` añade embeddings, etiquetas, predicciones, históricos, curvas e inventarios de preservación; no es obligatorio para una consulta normal. Los pesos CNN se descargan de PyTorch, no se duplican en los paquetes. La DLL se compila desde el código.

Se eligieron Releases frente a LFS: separan código de recursos y permiten descargar modelos sin instalar LFS ni reentrenar. GitHub bloquea archivos Git de más de 100 MiB y advierte desde 50 MiB; los assets de Release deben ser menores de 2 GiB ([documentación oficial](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)). Ambos ZIP están muy por debajo de ese límite. No se afirma que el acceso sea funcional hasta comprobar su publicación; ver `PUBLICACION_GITHUB.md`.

## Privacidad, integridad y límites

La auditoría encontró rutas privadas en tres logs históricos de investigación. Se excluyeron todos los logs de los ZIP públicos; los originales locales conservan sus hashes. La versión preliminar v1 de paquetes no se publica. El catálogo final v2 declara las omisiones; la prueba de preservación comprueba todos los originales donde existen y permite únicamente logs declarados ausentes en instalaciones distribuidas. No se alteraron modelos ni resultados para redaccionar logs.

Se buscaron credenciales con patrones conocidos, claves privadas y rutas privadas tanto en selección actual como en fuentes del historial y textos de ZIP. Los ejemplos `usuario:clave` de `.env.example` son placeholders, no credenciales. La búsqueda automática no demuestra ausencia de todo secreto concebible; se revisó además el contenido seleccionado y los enlaces de PDF. No se imprimen valores potencialmente secretos.

Resultado de la última pasada: **0 hallazgos pendientes**. Los paquetes contienen inventario por archivo y checksum; los instaladores verifican antes de mutar y rechazan rutas inseguras o archivos existentes diferentes. Joblib solo debe cargarse desde procedencia confiable.

Las licencias se documentan en `LICENCIAS_Y_RECURSOS.md`: Caltech CC BY 4.0 verificada mediante el DOI; código de bibliotecas y condiciones de pesos tratados por separado. No se asignó una licencia abierta nueva al código propio ni se inventaron identidades. La distribución no incluye corpus ImageNet ni fotos privadas de usuarios.
