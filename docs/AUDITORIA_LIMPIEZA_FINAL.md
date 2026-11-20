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

La comprobación final de clon aislado, tamaños y enlaces se incorporará antes de publicar el pull request. Comprobaciones ya ejecutadas: backend 25/25, setup y compilación TypeScript/Vite, smoke de diez configuraciones y galerías reales, 8.677 imágenes verificadas. No hubo entrenamiento.

## Riesgos y reversión

La prueba histórica de 1.736 imágenes sigue reutilizada; limpiar Git no crea validación independiente. La portabilidad se comprueba en otro directorio del mismo Windows, no en un segundo equipo. Edge y compilador MSYS2 son dependencias del equipo. El TestClient emite una deprecación de Starlette/httpx; no impide las pruebas.

No se reescribe historial ni se reducen objetos Git históricos: la reducción corresponde al árbol visible/checkout. Dataset y runtime ignorados se mantienen y no se cuentan como tamaño de entrega. Los assets de Releases y sus SHA256 no se modifican.

Para volver a estudiar la versión anterior, usar un worktree de la rama de archivo. Para deshacer una integración futura, revertir los commits de limpieza mediante nuevos commits y otro PR; no usar reset destructivo ni force push. Recuperar archivos retirados desde el respaldo si se necesitan, conservando la instalación vigente y sin sobrescribir datos locales diferentes.

El informe académico definitivo se añadirá después, como una fuente y su PDF final claramente identificados, citando los experimentos versionados y la limitación de prueba reutilizada. No se genera un informe APA en esta limpieza.
