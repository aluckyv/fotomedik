@echo off
rem FotoMedik - doucavanje FBCNN-a koji vidi i slicicu (vodjena mreza) na tvojim fotkama.
rem Pokreni dvoklikom. Sve ostaje u ovom folderu; dobre fotke se samo citaju.
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

echo.
echo === FotoMedik: doucavanje FBCNN-a sa slicicom (vodjena mreza) ===
echo.

rem --- 1. Python 3.12 -------------------------------------------------------
call :havepy
if "%PYOK%"=="42" goto :pyready
echo Python 3.12 nije instaliran. Instaliram ga...
rem novi Python install manager ("py install"), pa winget kao rezerva
py install 3.12 2>nul
call :havepy
if "%PYOK%"=="42" goto :pyready
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
call :havepy
if "%PYOK%"=="42" goto :pyready
echo.
echo Python 3.12 se nije instalirao sam. Otvori Command Prompt i otkucaj:
echo     py install 3.12
echo pa pokreni ovaj fajl ponovo.
pause
exit /b 1
:pyready

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
%PY% -m pip install --upgrade --quiet onnx onnxscript numpy pillow click || goto :fail
rem uvek najnoviji kod (ista oznaka verzije, pa bez --force-reinstall pip ne bi zamenio stari)
%PY% -m pip install --quiet --force-reinstall --no-deps "fotomedik @ https://github.com/aluckyv/fotomedik/archive/refs/heads/main.zip" || goto :fail
%PY% -c "import torch; print('Graficka:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NEMA - radice na procesoru (sporo)')"

rem --- 3. Parovi za ucenje --------------------------------------------------
if exist podaci_slicica\train_000.npz (
    echo.
    echo Parovi vec postoje u folderu "podaci_slicica". Obrisi taj folder ako hoces nove.
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
set "KORACI=20000"
if exist izlaz_slicica\last.pt (
    set RESUME=--resume
    echo Ucenje je vec radjeno. Za nastavak upisi veci broj koraka od proslog puta, npr. 40000.
    set /p KORACI="Do koliko koraka da uci (Enter = 20000): "
)
set INIT=
if exist izlaz\best.pt set INIT=--init izlaz\best.pt
%PY% -m fotomedik.train.finetune podaci_slicica izlaz_slicica --guided %INIT% --install --iters %KORACI% %RESUME% %*
if errorlevel 1 goto :fail
echo.
echo Gotovo. U FotoMediku ukljuci samo "JPEG clean-up" - koristi vodjenu mrezu (ona sama gradi rupe iz slicice).
pause
exit /b 0

:havepy
set "PYOK="
for /f "delims=" %%v in ('py -3.12 -c "print(42)" 2^>nul') do set "PYOK=%%v"
exit /b 0

:pairs
if "%DOBRE%"=="" (
    echo Nisi uneo folder.
    exit /b 1
)
%PY% -m fotomedik.train.pairs "%DOBRE%" podaci_slicica --per 3 --patches 40
exit /b %errorlevel%

:fail
echo.
echo Nesto nije uspelo - poruka je iznad. Posalji je Claude-u.
pause
exit /b 1
