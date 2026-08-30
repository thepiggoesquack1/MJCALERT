[CmdletBinding()]
param(
    [switch]$SkipDependencyInstall,
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

Write-Host "Python runtime:"
& $PythonExecutable -c "import sys; print(sys.executable); print(sys.version)"

if (-not $SkipDependencyInstall) {
    & $PythonExecutable -m pip install --upgrade pip
    & $PythonExecutable -m pip install -e ".[liveatc,control,dev,build]"
}

$TestTemp = Join-Path $ProjectRoot ".pytest-build-temp"
& $PythonExecutable -m pytest -q --basetemp="$TestTemp" -p no:cacheprovider
& $PythonExecutable -m ruff check .
& $PythonExecutable -m mypy src

& $PythonExecutable -m PyInstaller --noconfirm --clean "MRY Alert Control.spec"

$Output = Join-Path $ProjectRoot "dist\MRY Alert Control"
Copy-Item -LiteralPath "config.example.yaml" -Destination (Join-Path $Output "config.example.yaml") -Force
if (Test-Path -LiteralPath "config.yaml") {
    # Copy the existing configuration byte-for-byte; never synthesize or edit user settings.
    Copy-Item -LiteralPath "config.yaml" -Destination (Join-Path $Output "config.yaml") -Force
}
New-Item -ItemType Directory -Force -Path (Join-Path $Output "data") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Output "logs\gui") | Out-Null

if (Test-Path -LiteralPath $TestTemp) {
    Remove-Item -LiteralPath $TestTemp -Recurse -Force
}

Write-Host ""
Write-Host "Build complete:"
Write-Host (Join-Path $Output "MRY Alert Control.exe")
