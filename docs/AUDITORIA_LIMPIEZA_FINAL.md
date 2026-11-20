# Auditoría de limpieza de la entrega

Fecha: 20 de noviembre de 2026. Base: `65839e4c4e3b31f92b77f1c3a1965e12edc88d62`, estado inicial limpio. Trabajo en `chore/limpieza-entrega-academica`; integración propuesta por pull request a `main`, sin merge automático.

## Alcance y conservación

Se inventariaron los **197 archivos rastreados**, tamaños de blobs Git, SHA256 locales, funciones y referencias candidatas: [inventario completo](INVENTARIO_LIMPIEZA.csv). Se revisaron imports, scripts de instalación/distribución, dependientes del extractor C++, manifiestos y entornos ignorados. El CSV describe el estado anterior; sus referencias son candidatos históricos, no enlaces vigentes.

| Decisión | Archivos y justificación |
| --- | --- |
| Conservar | Backend, frontend, pruebas, experimentos y resultados, recursos, C++ y extractor HSV: mantienen ejecución, evidencia o dependientes reales. No cambió ninguna línea de backend/frontend/tests/C++; tampoco modelos, embeddings, partición, métricas ni catálogos. |
| Archivar y retirar de la entrega | 57 archivos de manuales/figuras/PDF, dos documentos antiguos de raíz, dos auditorías inicial/PDF y siete generadores históricos: 68 archivos. No son requisitos de ejecución; no se presentan como informe definitivo. |
| Reorganizar | Cinco requirements pasan a `requirements/`; las copias originales locales se conservan ignoradas. `requirements.txt` incluye el lock completo. No cambiaron versiones fijadas. |
| Añadir | Guías breves de instalación, arquitectura, metodología, archivo y dependencias; dos capturas reales de la aplicación; README con datos académicos confirmados. |
| Evitar duplicación | ZIP local de documentación: 93 entradas idénticas a archivos locales, cero diferencias/adicionales. Ya estaba ignorado y permanece local, fuera de Releases y Git. |

El [respaldo público](https://github.com/GUIDO-DBP/metriclab-visual/tree/archive/materiales-estudio-2026-11-20) se publicó y verificó **antes** de retirar archivos del índice. Apunta exactamente al commit base. Los 68 archivos siguen físicamente presentes y con su contenido original; no se borró material del compañero. [Acceso y recuperación](ARCHIVO_HISTORICO.md).

## Dependencias y enlaces

[Clasificación detallada](../requirements/README.md): runtime completo fijado; worker ResNet separado; unittest/httpx y Playwright para pruebas; generación PDF opcional; PostgreSQL opcional; registro de versiones directas redundante como instalador pero conservado como evidencia. El lock, las versiones directas y las dependencias del worker conservan sus blobs originales. No se actualizó el lock npm.

Se actualizaron setup, guías, licencias y lista de protección del experimento para las nuevas rutas. Los informes científicos conservan cifras/protocolos; sus antiguos enlaces a archivos ignorados `runtime/` se expresan como rutas locales con acceso al paquete research. El README enlaza capturas rastreadas y el archivo público, no materiales retirados. Las auditorías de publicación anteriores permanecen identificadas como evidencia histórica.

## Verificación

| Comprobación realizada | Resultado |
| --- | --- |
| Backend: unittest discover en tests | 25/25 aprobadas. |
| Frontend: setup, npm ci y TypeScript/Vite | Aprobados en workspace y clon; versiones conservadas. |
| Playwright/Edge en workspace, puerto 8003 | 14/14: diez flujos de navegador y cuatro comprobaciones de análisis/matrices. |
| Smoke original y clon CPU, con motor requerido | Diez configuraciones por instalación, 101 clases, 144D/512D, seis fotos train correctas por predicción y respuestas HTTP correctas. |
| Imágenes | 8.677 SHA256 verificados; manifiesto del clon idéntico al original, split 6.941/1.736. |
| C++ | Oráculo original y compilación nueva en clon: 10.503 comprobaciones, cero fallos. |
| Documentación archivada | Once PDF abiertos: 54 páginas de manuales y 21 del informe preliminar. |
| Enlaces y fuentes | Todos los enlaces Markdown relativos apuntan a archivos/carpetas rastreados; sintaxis Python correcta; sin referencias ejecutables a rutas retiradas. |
| Integridad | 291 archivos protegidos con SHA256 sin cambios; los 68 archivos archivados locales conservan su SHA256 original. |
| Selección Git | Sin dataset/runtime, ZIP, temporales ni entornos; cero hallazgos de los patrones de credenciales/rutas privadas revisados. |
| Recursos publicados | Tres assets de resources-v1 conservan tamaños y digests del catálogo; no se editaron Release ni tag. |

**Clon remoto de la rama de limpieza:** se obtuvo en `runtime/cleanup-check` y se comprobó el código funcional del commit `35efdac`; los cambios posteriores de esta rama documentan la verificación. Se crearon `.venv` y `runtime/resnet-env` nuevos, ambos sin paquetes globales. Setup completó pip check, torch 2.14.1+cpu/torchvision 0.29.1+cpu, NumPy 2.5.3, Pillow 12.3.0, npm ci y build. Se reutilizó caché de descargas de pip/npm, no paquetes instalados de otros entornos. Se repitió el comando normal tras un fallo del arnés al redirigir stderr de pip en PowerShell.

El clon descargó modelos desde la Release y pesos desde PyTorch, verificando sus SHA256. Extrajo las fotografías del ZIP oficial ya descargado, comprobando el checksum del ZIP y cada imagen; no se copió el dataset expandido. El comando `dataset_resources --download` se comprobó también en el workspace, reutilizando el ZIP oficial previamente verificado. No se presenta como una nueva descarga de Caltech en este ensayo. El clon no necesitó research para clasificar.

Se inició el clon mediante `start.ps1 -Port 8004`, con CPU. El flujo adicional de Edge comprobó modelos, galerías, comparación y vuelta a HSV. En la primera pasada su aserción de tabla agotó los cinco segundos predeterminados mientras la interfaz seguía cargando; el servidor completó HTTP 200. La repetición con espera de 30 segundos en una configuración de prueba local ignorada aprobó **1/1**. Se conserva esta observación: no se cambió la aplicación ni el timeout publicado para ocultarla. El estado Git del clon quedó limpio; no se entrenó ningún modelo.

## Tamaño y estructura de entrega

| Árbol rastreado | Archivos | Tamaño decimal |
| --- | ---: | ---: |
| Base, antes de limpiar | 197 | 13,78 MB (13.783.275 bytes) |
| Entrega reorganizada | 138 | Aproximadamente 4,51 MB |

Reducción aproximada del **67,3 %** del contenido del checkout, no del historial Git. Medición mediante tamaños de blobs Git, sin contar `.git`, dataset ni runtime. La raíz conserva README, configuración, `requirements.txt` y nueve carpetas con funciones separadas: backend, frontend, experiments, resources, scripts, tests, metriclab, docs y requirements.

Evidencias detalladas del arnés permanecen localmente en `runtime/cleanup_audit/`; no se publican caches ni logs. El [inventario previo](INVENTARIO_LIMPIEZA.csv), esta auditoría, los informes científicos y el respaldo Git hacen revisables las decisiones sin duplicar los manuales en la entrega.

## Riesgos y reversión

La prueba histórica de 1.736 imágenes sigue reutilizada; limpiar Git no crea validación independiente. La portabilidad se comprueba en otro directorio del mismo Windows, no en un segundo equipo. Edge y compilador MSYS2 son dependencias del equipo. El TestClient emite una deprecación de Starlette/httpx; no impide las pruebas. La espera predeterminada de cinco segundos del test adicional puede ser insuficiente al cargar modelos en CPU; no se ha optimizado ese rendimiento en esta limpieza. La descripción histórica de la Release conserva la ruta antigua del manual; el README actual y el archivo público indican su ubicación vigente.

No se reescribe historial ni se reducen objetos Git históricos: la reducción corresponde al árbol visible/checkout. Dataset y runtime ignorados se mantienen y no se cuentan como tamaño de entrega. Los assets de Releases y sus SHA256 no se modifican.

Para volver a estudiar la versión anterior, usar un worktree de la rama de archivo. Para deshacer una integración futura, revertir los commits de limpieza mediante nuevos commits y otro PR; no usar reset destructivo ni force push. Recuperar archivos retirados desde el respaldo si se necesitan, conservando la instalación vigente y sin sobrescribir datos locales diferentes.

El informe académico definitivo se añadirá después, como una fuente y su PDF final claramente identificados, citando los experimentos versionados y la limitación de prueba reutilizada. No se genera un informe APA en esta limpieza.
