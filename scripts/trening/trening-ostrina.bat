@echo off
rem FotoMedik - doucavanje FBCNN-a protiv mekoce (kazna za izgubljene ivice i teksturu).
rem Pokreni dvoklikom kad trening.bat ili trening-slicica.bat zavrsi. Sve ostaje u ovom folderu.
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

echo.
echo === FotoMedik: ucenje protiv mekoce ===
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

rem --- 3. Ucenje protiv mekoce ------------------------------------------------
rem Nastavlja od najbolje mreze (vodjena ako postoji, inace obicna) sa dodatnom kaznom
rem za izgubljene ivice i teksturu. Rezultat ide u izlaz_ostro; stara mreza ostaje u svom folderu.
set "PODACI=podaci"
set "OD=izlaz\best.pt"
set "VODJENA="
if exist izlaz_slicica\best.pt if exist podaci_slicica\train_000.npz (
    set "PODACI=podaci_slicica"
    set "OD=izlaz_slicica\best.pt"
    set "VODJENA=--guided"
)
if not exist "%OD%" (
    echo Nema naucene mreze ^(%OD%^). Prvo pokreni trening.bat ili trening-slicica.bat.
    goto :fail
)
echo.
echo Ucim protiv mekoce od %OD% na parovima iz %PODACI%.
set RESUME=
set "KORACI=6000"
if exist izlaz_ostro\last.pt (
    set RESUME=--resume
    set /p KORACI="Do koliko koraka da uci (Enter = 6000): "
)
%PY% -m fotomedik.train.finetune %PODACI% izlaz_ostro %VODJENA% --init %OD% --sharp 0.5 --install --iters %KORACI% %RESUME% %*
if errorlevel 1 goto :fail
echo.
echo Gotovo. "JPEG clean-up" sada koristi ostriju mrezu.
echo Ako ti se ne svidi: kopiraj stari .onnx iz izlaz ili izlaz_slicica u %%LOCALAPPDATA%%\FotoMedik\models.
pause
exit /b 0

:havepy
set "PYOK="
for /f "delims=" %%v in ('py -3.12 -c "print(42)" 2^>nul') do set "PYOK=%%v"
exit /b 0

:fail
echo.
echo Nesto nije uspelo - poruka je iznad. Posalji je Claude-u.
pause
exit /b 1
