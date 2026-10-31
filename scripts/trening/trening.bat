@echo off
rem FotoMedik - doucavanje FBCNN-a (JPEG ciscenje) na tvojim fotkama.
rem Pokreni dvoklikom. Sve ostaje u ovom folderu; dobre fotke se samo citaju.
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

echo.
echo === FotoMedik: doucavanje FBCNN-a ===
echo.

rem --- 1. Python 3.12 -------------------------------------------------------
py -3.12 -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo Python 3.12 nije instaliran. Instaliram ga preko winget-a...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    py -3.12 -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo Python se nije instalirao. Instaliraj Python 3.12 sa python.org pa pokreni ponovo.
        pause
        exit /b 1
    )
)

rem --- 2. Okruzenje (prvi put ~3 GB: torch za NVIDIA karticu) ----------------
if not exist venv\Scripts\python.exe (
    echo Pravim okruzenje u folderu venv...
    py -3.12 -m venv venv || goto :fail
)
set PY=venv\Scripts\python.exe
%PY% -c "import torch" >nul 2>nul
if errorlevel 1 (
    echo Instaliram torch za NVIDIA karticu - ovo traje, preuzima se oko 3 GB...
    %PY% -m pip install --upgrade pip || goto :fail
    %PY% -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || goto :fail
)
echo Osvezavam FotoMedik kod...
%PY% -m pip install --upgrade --quiet onnx onnxscript "fotomedik @ https://github.com/aluckyv/fotomedik/archive/refs/heads/main.zip" || goto :fail
%PY% -c "import torch; print('Graficka:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NEMA - radice na procesoru (sporo)')"

rem --- 3. Parovi za ucenje --------------------------------------------------
if exist podaci\train_000.npz (
    echo.
    echo Parovi vec postoje u folderu "podaci". Obrisi taj folder ako hoces nove.
) else (
    echo.
    set "DOBRE="
    set /p DOBRE="Folder sa DOBRIM fotkama (samo se citaju): "
    call :pairs
    if errorlevel 1 goto :fail
)

rem --- 4. Ucenje ------------------------------------------------------------
echo.
echo Ucenje pocinje. Moze da se prekine sa Ctrl+C - najbolje stanje se cuva i ubacuje u program.
echo Ponovno pokretanje nastavlja gde je stalo.
set RESUME=
if exist izlaz\last.pt set RESUME=--resume
%PY% -m fotomedik.train.finetune podaci izlaz --install %RESUME% %*
if errorlevel 1 goto :fail
echo.
echo Gotovo. U FotoMediku ukljuci "JPEG clean-up" - koristi doucen model.
pause
exit /b 0

:pairs
if "%DOBRE%"=="" (
    echo Nisi uneo folder.
    exit /b 1
)
%PY% -m fotomedik.train.pairs "%DOBRE%" podaci --per 3 --patches 48
exit /b %errorlevel%

:fail
echo.
echo Nesto nije uspelo - poruka je iznad. Posalji je Claude-u.
pause
exit /b 1
