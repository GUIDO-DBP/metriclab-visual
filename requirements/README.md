# Dependencias de MetricLab Visual

Se conservan las versiones comprobadas con los modelos publicados. No actualizar scikit-learn, XGBoost o LightGBM sin revisar la compatibilidad de sus artefactos.

| Archivo | Función |
| --- | --- |
| `../requirements.txt` | Entrada única backend: incluye `runtime.lock.txt`. Sustituye los rangos anteriores por la instalación realmente comprobada. |
| `runtime.lock.txt` | Backend y herramientas de experimentación/pruebas, con dependencias transitivas fijadas. Copia byte por byte del anterior `requirements-lock.txt`. Lo usa `scripts/setup.ps1`. |
| `resnet.txt` | NumPy 2.5.3 y Pillow 12.3.0 del worker aislado. Setup instala antes torch 2.14.1 y torchvision 0.29.1 desde el índice oficial CPU o cu130. No añade torch al backend. |
| `tested-direct.txt` | Registro de 13 versiones directas verificadas; redundante como instalador, conservado como evidencia. No reemplaza el lock completo. |
| `optional-postgres.txt` | Importador PostgreSQL opcional, innecesario para la web. Conserva el rango original, que no es un lock probado de PostgreSQL. |
| `documentation.txt` | Markdown y PyMuPDF fijados para manuales históricos. No interviene en setup ni ejecución; los generadores están en la rama de archivo. |

Las pruebas Python usan `unittest` de la biblioteca estándar y httpx incluido en el lock. El desarrollo frontend (TypeScript, Vite y Playwright) está separado en `frontend/package.json` y fijado por `frontend/package-lock.json`; instalar con `npm.cmd ci`. Edge debe estar instalado para el navegador configurado.

Los requirements propios de `experiments/resnet18/` y `experiments/resnet18_comparison/` se conservan porque documentan sus entornos históricos. Ejecutar entrenamiento no forma parte de instalar la aplicación.

Los seis archivos anteriores de raíz siguen disponibles en la rama de archivo. Se actualizaron rutas ejecutables y guías vigentes; los JSON científicos históricos conservan sus nombres y hashes originales.
