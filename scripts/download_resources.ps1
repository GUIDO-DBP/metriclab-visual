param([ValidateSet('models','research')][string]$Kind='models',[string]$Archive,[string]$Repository,[string]$Tag='resources-v1',[switch]$Weights)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    $python=Join-Path $projectRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw 'Ejecuta primero .\scripts\setup.ps1.' }
    $arguments=@('-m','scripts.download_resources','--kind',$Kind,'--tag',$Tag)
    if ($Archive) { $arguments+=@('--archive',$Archive) }
    if ($Repository) { $arguments+=@('--repository',$Repository) }
    if ($Weights) { $arguments+='--weights' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'No se instalaron los recursos. Revisa el mensaje anterior; no se reentrenó ningún modelo.' }
} finally { Pop-Location }
