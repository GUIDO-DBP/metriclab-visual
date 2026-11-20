# Licencias, procedencia y atribuciones

Verificación: 19 de noviembre de 2026. Este inventario distingue código, fotografías, pesos y modelos aprendidos; una licencia de biblioteca no otorga automáticamente derechos sobre todos sus datos.

## Caltech-101

Li, F.-F., Andreeto, M., Ranzato, M., y Perona, P. (2022). *Caltech 101*, versión 1.0. CaltechDATA. DOI: [10.22002/D1.20086](https://doi.org/10.22002/D1.20086). Los [metadatos primarios del DOI](https://api.datacite.org/dois/10.22002/D1.20086) declaran **Creative Commons Attribution 4.0 International**, identificador `cc-by-4.0`, [texto de licencia](https://creativecommons.org/licenses/by/4.0/legalcode).

Se verificó esta declaración en DataCite porque la página HTML de Caltech mostraba un apartado de derechos sin detallar la licencia. No se tomó la licencia de una ficha de terceros. El resumen verificable se conserva en `resources/caltech_provenance.json`.

El dataset completo se obtiene desde su fuente oficial y no se incorpora al árbol Git ni a los ZIP de modelos. Las fotografías de figuras y miniaturas de capturas corresponden a ese dataset; se redimensionaron, recortaron por encuadre o dispusieron en composiciones para explicar el proyecto. Las etiquetas originales se conservan. Atribución: autores anteriores, CaltechDATA, CC BY 4.0. La licencia no garantiza que estén despejados derechos de imagen, marcas u otros derechos de terceros.

## ResNet-18 y pesos oficiales

El extractor usa `ResNet18_Weights.IMAGENET1K_V1` desde [torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). El checkpoint se descarga directamente de `https://download.pytorch.org/models/resnet18-f37072fd.pth`; su SHA256 esperado es `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

El código de torchvision se publica bajo [BSD](https://github.com/pytorch/vision/blob/main/LICENSE). La [documentación de modelos](https://docs.pytorch.org/vision/stable/models.html) advierte que los pesos pueden tener condiciones derivadas de sus datasets. No se atribuye automáticamente a los pesos toda la licencia del código. No se redistribuye el checkpoint en Git ni en Releases de este proyecto; cada instalación lo obtiene de PyTorch. Consultar también las condiciones de [ImageNet](https://www.image-net.org/).

La entrega no afirma autorización para cualquier uso comercial de pesos/datos ni sustituye las condiciones originales. El caso de uso documentado es académico. El posible solapamiento de imágenes de Caltech con ImageNet no se auditó.

## Bibliotecas

| Componente | Licencia/procedencia que debe conservarse |
| --- | --- |
| React, Vite, TypeScript, FastAPI | MIT en sus proyectos oficiales. |
| scikit-learn, NumPy, SciPy | BSD y avisos de componentes incluidos en sus distribuciones. |
| XGBoost | Apache 2.0 y avisos de sus dependencias. |
| LightGBM | MIT, [licencia oficial](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE). |
| OpenCV | Apache 2.0 en versiones actuales; revisar avisos de la rueda instalada. |
| Pillow | HPND/avisos PIL y componentes incluidos. |
| PyTorch/torchvision | Conservar avisos de las ruedas; pueden contener componentes con distintas expresiones SPDX. |
| Matplotlib | Licencia propia compatible con PSF y avisos incluidos. |
| Playwright | Apache 2.0; se usa para pruebas y generación de PDF. |
| PyMuPDF | Herramienta opcional de revisión PDF; AGPL/comercial según su distribución. No es dependencia de ejecución de la web. |

Los entornos y ruedas no se redistribuyen. Pip/npm instalan las bibliotecas con sus avisos. `requirements/runtime.lock.txt` fija la instalación backend comprobada; `package-lock.json` registra paquetes frontend. Los generadores de documentación no cambian la licencia de las capturas propias ni atribuyen a este proyecto el código de terceros.

## Código propio, modelos y evidencias

No se ha asignado una licencia abierta nueva al código propio ni se han inventado titulares. La publicación y la entrega académica fueron autorizadas por el responsable del proyecto. Cualquier concesión adicional para redistribución o uso comercial debe decidirla el titular; una publicación en GitHub por sí sola no equivale a una licencia MIT.

Los clasificadores y embeddings fueron generados por los experimentos documentados; no son copias del checkpoint CNN. Se distribuyen para ejecutar y auditar esta entrega académica, conservando hashes, parámetros, atribución al dataset y procedencia del extractor. No se afirma una revisión jurídica exhaustiva de derivados o usos posteriores. Los integrantes verificados para esta entrega figuran en el README; los documentos preliminares archivados pueden conservar campos pendientes.

No se publica el corpus ImageNet. Los ZIP no contienen fotos del dataset, checkpoints, credenciales, datos personales de usuarios ni entornos. Los artefactos históricos se conservan sin cambiar resultados para esta distribución.
