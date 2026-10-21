@echo off
rem SPDX-License-Identifier: LGPL-2.1-or-later
rem Copyright (C) 2026 Hany El Imam
rem
rem Step 3 on Windows with an Nvidia card: installs what is missing and trains.
rem
rem   1. steps 1 and 2 must be done (TEST_SET and DATA at the top of the package);
rem   2. change the network size below if wanted;
rem   3. double click this file.
rem
rem It uses the private Python environment .venv at the top of the package
rem (made by INSTALL_WIN.md, or here if missing, with Python 3.12), installs
rem PyTorch for Nvidia, numpy and Pillow in it, then trains.  If it stops,
rem double click again: it goes on from the last save.  The model ends up in
rem TRAINED\<C>x<L>\sstv_dnn_<C>x<L>.bin
setlocal
cd /d "%~dp0\.."

rem ---- settings ---------------------------------------------------------
set CHANNELS=20
set LAYERS=6
set STEPS=200000
rem PyTorch for CUDA 13.0: Nvidia RTX 20xx and newer, GTX 16xx, driver 580 or
rem newer.  For an older card (GTX 9xx, GTX 10xx) use cu126 instead.
set CUDA=cu130
rem ------------------------------------------------------------------------
set PY=.venv\Scripts\python.exe

echo === 1/4 Python environment ===
rem PyTorch for CUDA exists only for 64-bit Python, and the newest Python
rem versions get it later: Python 3.12 is the safe choice.
if not exist "%PY%" (
    py -3.12 --version >nul 2>&1
    if errorlevel 1 (
        echo Python 3.12 is missing: installing it...
        winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        py -3.12 --version >nul 2>&1
        if errorlevel 1 (
            echo.
            echo Python 3.12 is installed but Windows does not see it yet:
            echo close this window and double click the file again.
            pause
            exit /b 1
        )
    )
    echo Making the private environment .venv ...
    py -3.12 -m venv .venv
)
"%PY%" --version

echo === 2/4 PyTorch for Nvidia (the first time it downloads a few GB) ===
"%PY%" -c "import torch" >nul 2>&1
if errorlevel 1 (
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/%CUDA%
)
"%PY%" -m pip install numpy pillow

echo === 3/4 Nvidia card ===
"%PY%" -c "import torch,sys; ok=torch.cuda.is_available(); print('PyTorch', torch.__version__, '- card:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND'); sys.exit(0 if ok else 1)"
if errorlevel 1 (
    echo.
    echo PyTorch does not see the Nvidia card. Update the Nvidia driver and try
    echo again; with a GTX 9xx/10xx card set CUDA=cu126 at the top of this file
    echo and remove PyTorch first:  .venv\Scripts\python.exe -m pip uninstall -y torch
    pause
    exit /b 1
)

echo === 4/4 Training %CHANNELS% x %LAYERS%, %STEPS% steps ===
if not exist "DATA\clean" (
    echo DATA is missing: do step 2 first ^(2_training_data\README.md^).
    pause
    exit /b 1
)
if not exist "TEST_SET\reference\clean" (
    echo TEST_SET is missing: do step 1 first ^(1_test_set\README.md^).
    pause
    exit /b 1
)
rem train.py keeps the PC awake while it runs
"%PY%" 3_train\train.py --channels %CHANNELS% --layers %LAYERS% --steps %STEPS% --resume
echo.
echo Finished. The model is in TRAINED\%CHANNELS%x%LAYERS%\sstv_dnn_%CHANNELS%x%LAYERS%.bin
pause
