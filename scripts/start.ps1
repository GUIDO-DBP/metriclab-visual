param([switch]$Dev,[ValidateRange(1024,65535)][int]$Port=8000)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    if (-not (Test-Path '.venv/Scripts/python.exe')) { throw 'Primero instala las dependencias siguiendo README.md.' }
    if (-not (Test-Path 'runtime/metriclab.dll')) { Write-Host 'El laboratorio metrico complementario requiere compilar el motor; la clasificacion puede funcionar sin el.' }
    if (-not $Dev -and -not (Test-Path 'frontend/dist/index.html')) {
        Push-Location frontend
        try { & npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw 'Falló la compilación del frontend.' } } finally { Pop-Location }
    }
    Write-Host "MetricLab Visual: http://127.0.0.1:$Port (Ctrl+C para detener)"
    if ($Dev) { Write-Host 'Frontend en otra terminal: cd frontend; npm.cmd run dev' }
    & .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'El backend no pudo iniciar. Comprueba si ya existe una instancia en el puerto 8000.' }
} finally { Pop-Location }
