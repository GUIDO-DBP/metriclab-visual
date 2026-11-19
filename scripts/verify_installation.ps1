param([switch]$Smoke,[switch]$SkipResNet,[switch]$RequireEngine)
$ErrorActionPreference='Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    if (-not (Test-Path '.venv/Scripts/python.exe')) { throw 'Ejecuta primero .\scripts\setup.ps1.' }
    $arguments=@('-m','scripts.verify_installation')
    if ($Smoke) { $arguments+='--smoke' }
    if ($SkipResNet) { $arguments+='--skip-resnet' }
    if ($RequireEngine) { $arguments+='--require-engine' }
    & .\.venv\Scripts\python.exe @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Verificación incompleta. Revisa el recurso o dependencia que falta; no se reentrena automáticamente.' }
} finally { Pop-Location }
