# Instalación reproducible en Windows

Requiere Python 3.14 x64, Node.js 22.12+ y Git. Ejecutar desde PowerShell en la raíz. `setup.ps1` crea `.venv` para FastAPI/clasificadores y `runtime/resnet-env` para torch/torchvision; instala versiones fijadas y compila el frontend. No entrena.

```powershell
git clone https://github.com/GUIDO-DBP/metriclab-visual.git
cd metriclab-visual
.\scripts\setup.ps1
.\scripts\download_resources.ps1 -Repository GUIDO-DBP/metriclab-visual -Weights
.\.venv\Scripts\python.exe -m scripts.dataset_resources --download
.\scripts\verify_installation.ps1 -Smoke
.\scripts\start.ps1
```

Visitar http://127.0.0.1:8000. Ctrl+C detiene el proceso. Para otro puerto: `start.ps1 -Port 8002`. Si PowerShell bloquea scripts, revisar su contenido y usar `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` solo en esa sesión.

El paquete `models` de la [Release resources-v1](https://github.com/GUIDO-DBP/metriclab-visual/releases/tag/resources-v1) incluye modelos, manifiesto, registro de descriptores y métricas. Dataset y checkpoint se descargan de sus fuentes oficiales. El [catálogo](../resources/manifest.json) verifica tamaños y SHA256; el manifiesto de imágenes comprueba los 8.677 archivos y mantiene la separación 6.941/1.736.

La descarga es aditiva: reutiliza archivos idénticos y rechaza conflictos o hashes distintos. No reemplazar artefactos para eludir la validación. Cargar joblib solo de procedencia confiable.

Para ZIP de Release transportados sin red:

```powershell
.\scripts\download_resources.ps1 -Archive runtime/downloads/metriclab-models-2026-11-19-v2.zip
.\.venv\Scripts\python.exe -m scripts.dataset_resources --archive runtime/downloads/caltech-101.zip
```

El ZIP de modelos no contiene pesos: se descargan con `-Weights` o se proporciona el checkpoint exacto en la caché del catálogo. Para dataset existente, `dataset_resources --source RUTA_A_101_ObjectCategories` verifica cada imagen antes de instalarla.

## CPU, CUDA y configuración

CPU es predeterminado en setup. Para GPU/controlador compatibles, `setup.ps1 -TorchBackend cu130 -Refresh` instala CUDA. La aplicación comprueba una operación real y permite CPU; CUDA se utiliza para extracción, no automáticamente para los clasificadores.

```powershell
$env:METRICLAB_RESNET_DEVICE='cpu'
.\scripts\start.ps1
```

`.env.example` documenta opciones pero no se carga automáticamente. Para desactivar ResNet: `$env:METRICLAB_ENABLE_RESNET='0'` y reiniciar. `setup.ps1 -SkipResNet` prepara solo backend/HSV; no es instalación completa de ambos descriptores.

## Pruebas y laboratorio complementario

`verify_installation.ps1 -Smoke` comprueba modelos, clases, dimensiones, split, hashes, diez configuraciones HTTP y galerías. Para pruebas completas, instalar además research:

```powershell
.\scripts\download_resources.ps1 -Kind research -Repository GUIDO-DBP/metriclab-visual
.\scripts\build-engine.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\scripts\verify_installation.ps1 -Smoke -RequireEngine
cd frontend
npm.cmd run test:e2e
```

El motor requiere g++ UCRT64 de MSYS2; el build compila DLL y ejecuta su oráculo. Clasificación funciona sin DLL, aunque el laboratorio no estará disponible. Playwright utiliza Edge y un servidor en puerto 8000; `PLAYWRIGHT_BASE_URL` permite otro puerto.

Para dependencias, revisar `pip check` y ejecutar setup; no actualizar paquetes individualmente. Para fotos: `dataset_resources --verify`. Para frontend: `npm.cmd ci` y `npm.cmd run build` dentro de frontend. Para GPU incompatible: CPU. PostgreSQL es opcional.

[Dependencias](../requirements/README.md) · [Arquitectura](ARQUITECTURA.md) · [Prueba aislada anterior](PRUEBA_INSTALACION_LIMPIA.md) · [Auditoría actual](AUDITORIA_LIMPIEZA_FINAL.md).
