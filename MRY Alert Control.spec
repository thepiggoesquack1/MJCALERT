# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.utils.hooks import collect_submodules

project_root = SPECPATH
source_root = os.path.join(project_root, "src")
packaging_hooks = os.path.join(project_root, "packaging_hooks")
hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("websockets")
    + collect_submodules("mry_alert")
)

control_analysis = Analysis(
    [os.path.join(source_root, "mry_alert", "control", "__main__.py")],
    pathex=[source_root],
    binaries=[],
    datas=[],
    hiddenimports=hidden,
    hookspath=[packaging_hooks],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch"],
    noarchive=False,
)
control_pyz = PYZ(control_analysis.pure)
control_exe = EXE(
    control_pyz,
    control_analysis.scripts,
    [],
    exclude_binaries=True,
    name="MRY Alert Control",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

backend_analysis = Analysis(
    [os.path.join(source_root, "mry_alert", "__main__.py")],
    pathex=[source_root],
    binaries=[],
    datas=[],
    hiddenimports=hidden,
    hookspath=[packaging_hooks],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "PySide6"],
    noarchive=False,
)
backend_pyz = PYZ(backend_analysis.pure)
backend_exe = EXE(
    backend_pyz,
    backend_analysis.scripts,
    [],
    exclude_binaries=True,
    name="MRY Alert Backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

bundle = COLLECT(
    control_exe,
    backend_exe,
    control_analysis.binaries,
    control_analysis.datas,
    backend_analysis.binaries,
    backend_analysis.datas,
    strip=False,
    upx=True,
    name="MRY Alert Control",
)
