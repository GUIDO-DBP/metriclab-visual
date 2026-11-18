param([switch]$Dev)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    if (-not (Test-Path '.venv/Scripts/python.exe')) { throw 'Primero instala las dependencias siguiendo README.md.' }
    if (-not (Test-Path 'runtime/metriclab.dll')) { throw 'Ejecuta .\scripts\build-engine.ps1 antes de iniciar.' }
    if (-not $Dev -and -not (Test-Path 'frontend/dist/index.html')) {
        Push-Location frontend
        try { & npm.cmd run build; if ($LASTEXITCODE -ne 0) { throw 'Falló la compilación del frontend.' } } finally { Pop-Location }
    }
    Write-Host 'MetricLab Visual: http://127.0.0.1:8000 (Ctrl+C para detener)'
    if ($Dev) { Write-Host 'Frontend en otra terminal: cd frontend; npm.cmd run dev' }
    & .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
} finally { Pop-Location }
