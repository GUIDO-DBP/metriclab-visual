# Comprobación de instalación independiente

Fecha: 19 de noviembre de 2026. La prueba usa un segundo directorio del mismo equipo, `runtime/repro-check/source/`. **No demuestra ejecución en otra computadora física** ni compatibilidad con otra GPU. Se complementa con la comprobación del checkout publicado descrita en `PUBLICACION_GITHUB.md`.

## Aislamiento y recursos

Se creó un entorno backend nuevo con Python 3.14.4 x64, `include-system-site-packages = false`. Pip instaló las versiones fijadas; `pip check` informó que no había requisitos rotos. No se copió `.venv` original. Se generó `requirements/runtime.lock.txt` a partir de esta instalación, incluyendo dependencias transitivas.

Se creó también un entorno ResNet nuevo, sin paquetes globales, con **torch 2.14.1+cpu**, **torchvision 0.29.1+cpu**, NumPy 2.5.3 y Pillow 12.3.0. No se utilizó el entorno CUDA original ni la GPU para este ensayo. El endpoint del servidor nuevo confirmó `device: cpu` después de clasificar.

El dataset se descargó de la fuente oficial: ZIP de **137.414.764 bytes**, MD5 `3138e1922a9193bfa496528edbbc45d0`, SHA256 `331234750fc7f77520e50d9565e8b6907b03565e30320da1401130db08c61f91`. El servidor redirigía inicialmente a una URL firmada caducada por caché; se verificó la descarga evitando esa caché. El instalador incorpora ese tratamiento sin guardar URLs firmadas temporales. Se extrajeron las imágenes desde el ZIP, sin copiarlas silenciosamente del dataset local anterior, y se comprobaron los 8.677 SHA256 esperados.

Los pesos se descargaron de PyTorch a la caché nueva. Una conexión se interrumpió; el reintento completó los **46.830.571 bytes** y coincidió con SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. No se copiaron de la caché anterior.

Los modelos se instalaron desde ZIP versionados mediante el mismo código que utiliza un compañero. Se verificaron el hash exterior y el inventario de cada paquete. Después se probó la versión final v2: conserva recursos existentes idénticos y excluye logs históricos con rutas privadas. Los originales del workspace permanecen intactos. El informe de publicación registra adicionalmente la obtención desde la Release real.

## Procedimiento comprobado

Se preparó un checkout de fuentes en el segundo directorio, con entornos nuevos. Se ejecutó `scripts/setup.ps1 -SkipResNet` para comprobar el backend y ejecutar **npm ci** y **npm run build** allí; el entorno ResNet CPU ya se había instalado explícitamente en ese directorio. No se reutilizó `node_modules` del proyecto original. Esto comprueba los pasos de dependencias, aunque no debe confundirse con que todas las descargas se hicieron en una única invocación sin intervención ante errores de red.

Después de publicar se obtuvo allí el historial desde GitHub y se hizo checkout del commit `2a0bbe6`, sin modificar los entornos ni recursos ignorados. La verificación completa `verify_installation.ps1 -Smoke -RequireEngine` pasó sobre ese código publicado y el estado Git del checkout quedó limpio. Adicionalmente, un clon remoto separado descargó ambos ZIP y los pesos mediante el instalador; no se limitó la prueba a paquetes locales.

Se ejecutó `scripts/verify_installation.ps1 -Smoke`: comprobó separación, imágenes, modelos y galerías. Se inició la aplicación mediante **scripts/start.ps1 -Port 8002**, con frontend compilado local y backend del entorno nuevo. El puerto 8000 original se mantuvo separado.

El motor C++ se compiló desde las fuentes del segundo directorio utilizando el compilador MSYS2 existente en el equipo. No se copió la DLL original. Después se reinició la instancia de prueba para cargarla. El compilador es una dependencia del equipo, no un componente instalado por pip.

## Resultados reales

| Comprobación | Resultado |
| --- | --- |
| Dependencias backend nuevas y pip check | Correctas. |
| Torch/torchvision CPU aislados | Carga y extracción correctas. |
| Imágenes | 8.677 verificadas por SHA256; 101 categorías. |
| Modelos | Cinco configuraciones con cada descriptor; 144D/512D y 101 clases. |
| Inferencia HTTP y galerías | Diez configuraciones reales; seis fotos train correctas por predicción. |
| Frontend | npm ci y compilación TypeScript/Vite correctos. |
| Motor C++ nuevo | 10.503 comprobaciones; cero fallos. |
| Tests backend en entorno nuevo | 25 aprobados en la pasada final; incluye redirecciones sin filtrar credenciales. |
| Playwright/Edge en puerto 8002 | 7 aprobados; navegación, carga, modelos, comparación, galerías, ampliación, matrices, explorador y móvil. |
| Paquetes finales v2 sin logs | Instalación idempotente y cinco pruebas de descriptores repetidas correctamente. |
| Entrenamiento realizado para instalar | Ninguno. |

Las pruebas de instalación cubren hashes inválidos, rutas ZIP inseguras, conflictos y conservación de archivos existentes. Las pruebas históricas de preservación verifican los artefactos originales donde están disponibles; solo permiten ausencia de logs explícitamente declarados en el catálogo público. En el workspace original se verifican todos, incluidos esos logs.

## Límites precisos

No se ha probado una segunda máquina, otro sistema operativo ni otro controlador NVIDIA. Se usó Edge y un compilador ya instalados en este equipo. Pip/npm pudieron reutilizar caché de **descargas**, pero los paquetes instalados se obtuvieron en entornos nuevos, sin importar paquetes de los entornos originales. La instalación no requiere esas cachés para resolver rutas o localizar modelos.

La dependencia Starlette/TestClient emitió una advertencia de deprecación sobre httpx, sin fallar las pruebas. No se cambió el framework ni el algoritmo para ocultarla. No se afirma ausencia de cualquier aviso futuro. La prueba confirma funcionalidad local reproducible bajo las versiones descritas, no capacidad de producción ni una nueva validación de precisión.
