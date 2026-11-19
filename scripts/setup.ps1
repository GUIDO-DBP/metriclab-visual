param([ValidateSet('cpu','cu130')][string]$TorchBackend='cpu',[switch]$SkipResNet,[switch]$Refresh)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
function Invoke-Checked([string]$Executable,[string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Falló $Executable. Revisa el mensaje anterior antes de continuar." }
}
try {
    $pythonCommand=Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { throw 'Instala Python 3.14 x64 desde python.org y añade python a PATH.' }
    Invoke-Checked $pythonCommand.Source @('-c','import sys,struct; assert sys.version_info[:2]==(3,14) and struct.calcsize(''P'')==8, ''Se requiere Python 3.14 x64 para esta entrega''')
    $node=Get-Command node -ErrorAction SilentlyContinue
    if (-not $node) { throw 'Instala Node.js 22.12 o posterior desde nodejs.org.' }
    Invoke-Checked $node.Source @('-e','const [a,b]=process.versions.node.split(''.'').map(Number); if(a<22||(a===22&&b<12))process.exit(1)')
    if (-not (Test-Path '.venv/Scripts/python.exe')) { Invoke-Checked $pythonCommand.Source @('-m','venv','.venv') }
    $python=Join-Path $projectRoot '.venv/Scripts/python.exe'
    $requirementsHash=(Get-FileHash 'requirements-lock.txt' -Algorithm SHA256).Hash
    $statePath=Join-Path $projectRoot 'runtime/setup-state.json'
    $setupState=$null
    if (Test-Path -LiteralPath $statePath) { $setupState=Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json }
    if ($Refresh -or -not $setupState -or $setupState.requirements -ne $requirementsHash) {
        Write-Host 'Instalando versiones compatibles con los modelos guardados...'
        Invoke-Checked $python @('-m','pip','install','-r','requirements-lock.txt')
    }
    Invoke-Checked $python @('-m','pip','check')
    if (-not $SkipResNet) {
        $resnetPython=Join-Path $projectRoot 'runtime/resnet-env/Scripts/python.exe'
        if (-not (Test-Path -LiteralPath $resnetPython)) { Invoke-Checked $pythonCommand.Source @('-m','venv','runtime/resnet-env') }
        Write-Host "Preparando extractor ResNet ($TorchBackend); CUDA requiere GPU y controlador compatibles."
        & $resnetPython -c 'import importlib.util as u,importlib.metadata as m,sys; found=u.find_spec(''torch'') and u.find_spec(''torchvision''); sys.exit(0 if found and m.version(''torch'').split(''+'')[0]==''2.14.1'' and m.version(''torchvision'').split(''+'')[0]==''0.29.1'' else 1)'
        $torchReady=$LASTEXITCODE -eq 0
        if ($Refresh -or -not $torchReady) {
            $torchArguments=@('-m','pip','install','torch==2.14.1','torchvision==0.29.1','--index-url',"https://download.pytorch.org/whl/$TorchBackend")
            if ($Refresh) { $torchArguments+='--force-reinstall' }
            Invoke-Checked $resnetPython $torchArguments
        }
        Invoke-Checked $resnetPython @('-m','pip','install','-r','requirements-resnet-runtime.txt')
        Invoke-Checked $resnetPython @('-c','import torch,torchvision,numpy,PIL; assert torch.__version__.split(''+'')[0]==''2.14.1'' and torchvision.__version__.split(''+'')[0]==''0.29.1''; print(''ResNet disponible; CUDA:'',torch.cuda.is_available())')
    }
    Push-Location frontend
    try {
        $lockHash=(Get-FileHash 'package-lock.json' -Algorithm SHA256).Hash
        if ($Refresh -or -not (Test-Path 'node_modules') -or -not $setupState -or $setupState.frontend -ne $lockHash) { Invoke-Checked 'npm.cmd' @('ci') }
        Invoke-Checked 'npm.cmd' @('run','build')
    } finally { Pop-Location }
    New-Item -ItemType Directory -Force runtime | Out-Null
    @{requirements=$requirementsHash;frontend=$lockHash;torchBackend=$TorchBackend} | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    Write-Host 'Dependencias listas. Ahora instala los paquetes de modelos, descarga el dataset y verifica la instalación.'
    Write-Host 'No se entrenó ningún modelo. Consulta documentacion_entrega/09_INSTALACION_EJECUCION.md.'
} finally { Pop-Location }
