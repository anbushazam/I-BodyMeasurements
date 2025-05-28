# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['measurement.py'],
    pathex=[],
    binaries=[],
    datas=[('HRNet\\\\lib', 'HRNet\\\\lib'), ('HRNet\\\\experiments', 'HRNet\\\\experiments'), ('weight_prediction_model_gb.pkl', '.'), ('scaler.pkl', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='measurement',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
