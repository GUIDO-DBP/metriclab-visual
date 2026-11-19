# 09. Instalación Windows desde cero

[TOC]

## 9.1. Qué se descarga y qué no se entrena

El repositorio es https://github.com/GUIDO-DBP/metriclab-visual. Git contiene código, manuales, figuras, catálogos y hashes. Los modelos se distribuyen como ZIP versionado mediante Releases o una copia directa del mismo archivo con su hash. Caltech se obtiene desde la fuente oficial y los pesos ResNet desde PyTorch. No se incluyen entornos ni `node_modules`.

Los comandos normales de instalación **no entrenan** modelos. No ejecutes `prepare --train` para resolver un recurso ausente. Los modelos actuales requieren los manifiestos y la separación de esta entrega; sustituirlos por otra preparación puede invalidar su fingerprint.

## 9.2. Requisitos del equipo

Windows x64, Python **3.14 x64** y Node.js **22.12 o posterior**. Se verificó Python 3.14.4; el entorno actual usa Node 24.15.0. Instala desde [python.org](https://www.python.org/downloads/windows/) y [nodejs.org](https://nodejs.org/en/download). Habilita Python en PATH. Git se obtiene desde [git-scm.com](https://git-scm.com/downloads/win) o se puede descargar el ZIP del repositorio.

Reserva varios GB libres para dependencias y recursos. CPU es la opción inicial portable. GPU NVIDIA es opcional y requiere controlador compatible con la rueda CUDA elegida. El consumo depende del entorno; no se garantiza un mínimo de RAM suficiente para cualquier carga concurrente.

```powershell
python --version
python -c "import struct; print(struct.calcsize('P')*8)"
node --version
git --version
```

Debe indicar Python 3.14 y 64 bits. Evita el alias de Microsoft Store si no abre el Python instalado.

## 9.3. Clonar y preparar dependencias

```powershell
git clone https://github.com/GUIDO-DBP/metriclab-visual.git
cd metriclab-visual
.\scripts\setup.ps1
```

El script crea `.venv` sin paquetes globales, instala `requirements-lock.txt`, comprueba dependencias, crea `runtime/resnet-env` e instala torch/torchvision oficiales CPU y el preprocesamiento fijado. Ejecuta `npm ci` y compila el frontend. Repetirlo conserva datos y modelos; `-Refresh` vuelve a comprobar instalaciones de dependencias.

Si PowerShell bloquea scripts descargados, revisa su contenido y usa una política **solo para esa sesión**, sin cambiar permanentemente la configuración del equipo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup.ps1
```

No es necesario activar `.venv`: los scripts utilizan su intérprete explícito. `.env.example` es una plantilla y no se carga automáticamente. No contiene una contraseña real.

## 9.4. Instalar modelos ya entrenados

El paquete obligatorio es `metriclab-models-2026-11-19-v2.zip`; research es opcional para auditar o repetir experimentos. El catálogo `resources/manifest.json` contiene SHA256 y tamaños exactos. La instalación valida el ZIP completo y sus archivos, rechaza rutas fuera de runtime y se detiene si encuentra un archivo existente diferente.

Cuando la Release `resources-v1` esté publicada, la descarga automática usa su API y verifica el archivo contra el catálogo local:

```powershell
.\scripts\download_resources.ps1 -Repository GUIDO-DBP/metriclab-visual -Weights
```

Alternativa verificable sin depender del acceso a Releases: recibe el ZIP original del autor, guárdalo en `runtime/downloads/` y ejecuta:

```powershell
.\scripts\download_resources.ps1 -Archive runtime/downloads/metriclab-models-2026-11-19-v2.zip -Weights
```

Para evidencias completas:

```powershell
.\scripts\download_resources.ps1 -Kind research -Repository GUIDO-DBP/metriclab-visual
```

Si todavía no hay Release o no tienes permiso, el instalador lo informa; no sustituye el paquete por modelos reentrenados ni utiliza una URL inventada. Consulta el estado real de publicación en `docs/PUBLICACION_GITHUB.md`.

## 9.5. Dataset oficial

```powershell
.\.venv\Scripts\python.exe -m scripts.dataset_resources --download
.\.venv\Scripts\python.exe -m scripts.dataset_resources --verify
```

El ZIP oficial tiene MD5 `3138e1922a9193bfa496528edbbc45d0`. Después de obtenerlo se verifican SHA256 de las 8.677 imágenes esperadas y se instalan en `dataset/101_ObjectCategories`. El MD5 se usa porque es el checksum publicado por la fuente; el SHA256 individual añade verificación del contenido esperado por los modelos.

Si falla la descarga por red o por una redirección temporal del servidor, descarga desde https://data.caltech.edu/records/mzrjq-6wc02 y usa:

```powershell
.\.venv\Scripts\python.exe -m scripts.dataset_resources --archive runtime/downloads/caltech-101.zip
```

Si ya tienes una carpeta, proporciona `--source` con su ruta entre comillas. El script verifica antes de copiar y no modifica la separación histórica. No se requiere volver a ejecutar el extractor HSV ni el entrenamiento.

## 9.6. CPU y GPU

CPU no requiere CUDA Toolkit ni GPU. Para forzarla durante extracción:

```powershell
$env:METRICLAB_RESNET_DEVICE='cpu'
.\scripts\start.ps1
```

Si existe GPU NVIDIA y un controlador compatible, se puede instalar la variante comprobada del entorno original:

```powershell
.\scripts\setup.ps1 -TorchBackend cu130 -Refresh
$env:METRICLAB_RESNET_DEVICE='auto'
```

Se fijan torch 2.14.1 y torchvision 0.29.1; el índice oficial determina CPU o CUDA. Cambiar solo `-TorchBackend` sin `-Refresh` conserva un entorno que ya tiene las versiones requeridas. Una rueda CUDA también puede ejecutar CPU. `auto` comprueba una operación real; `cuda` explícito falla si no es válida. No se garantiza que cu130 sea compatible con cualquier GPU o controlador; CPU es la alternativa comprobada.

## 9.7. Motor C++ opcional

La clasificación funciona sin DLL. Para el laboratorio, instala MSYS2 y el compilador UCRT64 según [msys2.org](https://www.msys2.org/). En su terminal UCRT64:

```text
pacman -S --needed mingw-w64-ucrt-x86_64-gcc
```

Añade el `bin` de UCRT64 a PATH si corresponde a tu instalación y vuelve a PowerShell:

```powershell
.\scripts\build-engine.ps1
```

Compila la DLL y el oráculo, que debe finalizar sin fallos. No se distribuye una DLL copiada a ciegas desde otro equipo; se conserva el código y su comprobación. No hay que compilar C++ para usar los cuatro clasificadores.

## 9.8. Verificar e iniciar

```powershell
.\scripts\verify_installation.ps1 -Smoke
.\scripts\start.ps1
```

El verificador comprueba imágenes, fingerprint, 101 clases, cinco configuraciones por descriptor, hashes, frontend y predicciones reales con galerías. Escribe `runtime/installation-verification.json`. `-RequireEngine` exige también motor; `-SkipResNet` permite comprobar solo HSV si se instaló con `setup -SkipResNet`.

Abre **http://127.0.0.1:8000**. Carga una fotografía y prueba HSV y ResNet. La primera consulta ResNet tarda más por inicialización. Para detener, Ctrl+C en la terminal. Si el puerto está ocupado:

```powershell
.\scripts\start.ps1 -Port 8002
```

En desarrollo se puede usar `start.ps1 -Dev` y, en otra terminal, `cd frontend; npm.cmd run dev`. La entrega normal sirve el frontend compilado desde FastAPI y no requiere mantener dos servidores.

## 9.9. Diagnóstico de errores

| Síntoma | Comprobación |
| --- | --- |
| `python` no encontrado | Reabrir terminal tras instalar y revisar PATH/alias. |
| No existe rueda compatible | Usar Python 3.14 x64 y las versiones fijadas. |
| Release 404 | Confirmar publicación, tag y permiso; usar ZIP local verificado. |
| Hash distinto | No sobrescribir; obtener de nuevo el recurso auténtico. |
| Fotos ausentes | Ejecutar instalación/verificación de Caltech. |
| ResNet deshabilitado | Revisar intérprete, torch/torchvision, checkpoint y registro. |
| Error CUDA | Probar CPU y reiniciar; comprobar controlador. |
| Página sin interfaz | Ejecutar `npm.cmd ci` y `npm.cmd run build` dentro de frontend. |
| Laboratorio no disponible | Compilar C++; los clasificadores siguen independientes. |

Los modelos joblib deben proceder del repositorio/Release esperado y coincidir con el catálogo. No cargues modelos de desconocidos. Una conexión privada puede autenticarse con GitHub CLI o token en variable de entorno; no pegues tokens en scripts, URLs, capturas ni commits.

## 9.10. Qué se ha comprobado realmente

El registro final de prueba limpia se encuentra en `docs/PRUEBA_INSTALACION_LIMPIA.md`. Describe checkout, entornos sin paquetes globales, origen de recursos y resultados. No se afirma ejecución en otra computadora física ni otra GPU si solo se ha probado un segundo directorio del mismo equipo.
