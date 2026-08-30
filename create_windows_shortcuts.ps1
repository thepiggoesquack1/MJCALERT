[CmdletBinding()]
param(
    [switch]$Desktop,
    [switch]$StartMenu
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Executable = Join-Path $ProjectRoot "dist\MRY Alert Control\MRY Alert Control.exe"
if (-not (Test-Path -LiteralPath $Executable)) {
    throw "Build the application first. Executable not found: $Executable"
}
if (-not $Desktop -and -not $StartMenu) {
    throw "Specify -Desktop, -StartMenu, or both."
}

$Shell = New-Object -ComObject WScript.Shell
$Targets = @()
if ($Desktop) {
    $Targets += [Environment]::GetFolderPath("Desktop")
}
if ($StartMenu) {
    $Targets += Join-Path ([Environment]::GetFolderPath("Programs")) "MRY Jet Center"
}
foreach ($Folder in $Targets) {
    New-Item -ItemType Directory -Force -Path $Folder | Out-Null
    $Shortcut = $Shell.CreateShortcut((Join-Path $Folder "MRY Alert Control.lnk"))
    $Shortcut.TargetPath = $Executable
    $Shortcut.WorkingDirectory = Split-Path -Parent $Executable
    $Shortcut.Description = "MRY Jet Center Alert backend control"
    $Shortcut.Save()
}
Write-Host "Requested shortcuts created."
