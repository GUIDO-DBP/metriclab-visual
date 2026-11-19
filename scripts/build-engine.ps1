$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    New-Item -ItemType Directory -Force runtime | Out-Null
    $compiler = Get-Command g++ -ErrorAction SilentlyContinue
    if (-not $compiler -and (Test-Path 'C:\msys64\ucrt64\bin\g++.exe')) {
        $compiler = Get-Command 'C:\msys64\ucrt64\bin\g++.exe'
    }
    if (-not $compiler) { throw 'Instala g++ de MSYS2 UCRT64 y añade su carpeta bin a PATH.' }
    $sources = @('metriclab/src/vp_tree.cpp', 'metriclab/src/laesa.cpp', 'metriclab/src/trace.cpp')
    & $compiler.Source -std=c++17 -O2 -shared -static -I metriclab/include metriclab/apps/web_bridge.cpp @sources -o runtime/metriclab.dll
    if ($LASTEXITCODE -ne 0) { throw 'Falló la compilación del puente C++.' }
    & $compiler.Source -std=c++17 -O2 -static -I metriclab/include metriclab/tests/oracle_test.cpp @sources -o runtime/oracle_test.exe
    if ($LASTEXITCODE -ne 0) { throw 'Falló la compilación del oráculo.' }
    & .\runtime\oracle_test.exe
    if ($LASTEXITCODE -ne 0) { throw 'El motor no pasó las pruebas.' }
} finally { Pop-Location }
