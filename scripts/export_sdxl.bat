@echo off
rem Exports the SDXL inpainting model to ONNX fp16 for FotoMedik's diffusion step, on this
rem computer (the GitHub machine runs out of memory). The same steps as
rem .github/workflows/export-sdxl.yml. Needs Python 3.11 or 3.12 (py launcher), about 40 GB
rem free next to this file, 32 GB of memory or more, and about an hour.
rem Everything goes into the "sdxl" folder next to this file; the result is in sdxl\fp16.
setlocal
cd /d "%~dp0"
set W=%~dp0sdxl
if not exist "%W%" mkdir "%W%"
set HF_HOME=%W%\hf
set LOG=%W%\export-log.txt
echo FotoMedik SDXL export - log in %LOG%
echo started %date% %time% > "%LOG%"

py -3.12 -c "print()" >nul 2>&1 && (set PY=py -3.12) || (set PY=py -3.11)
%PY% -c "import sys; print(sys.version)" >> "%LOG%" 2>&1 || (echo Python 3.11 or 3.12 is missing: install it from python.org & pause & exit /b 1)

if not exist "%W%\venv\Scripts\python.exe" %PY% -m venv "%W%\venv" >> "%LOG%" 2>&1
set P="%W%\venv\Scripts\python.exe"

echo [1/4] Installing the export tools...
%P% -m pip install --upgrade pip >> "%LOG%" 2>&1
%P% -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu >> "%LOG%" 2>&1 || goto fail
%P% -m pip install "numpy==1.26.4" "torch==2.5.1" "torchvision==0.20.1" "diffusers==0.31.0" "transformers==4.46.3" "optimum[exporters]==1.23.3" "onnx==1.17.0" "onnxruntime==1.20.1" onnxconverter-common accelerate safetensors >> "%LOG%" 2>&1 || goto fail

echo [2/4] Downloading the model and exporting it (fp32, the longest part)...
if not exist "%W%\fp32\unet\model.onnx" (
  "%W%\venv\Scripts\optimum-cli.exe" export onnx --model diffusers/stable-diffusion-xl-1.0-inpainting-0.1 --task inpainting --no-post-process "%W%\fp32" >> "%LOG%" 2>&1 || goto fail
)

echo [3/4] Converting to fp16...
if exist "%W%\fp16" rmdir /s /q "%W%\fp16"
%P% -m onnxruntime.transformers.models.stable_diffusion.optimize_pipeline -i "%W%\fp32" -o "%W%\fp16" --float16 --use_external_data_format --disable_group_norm --disable_skip_group_norm --disable_nhwc_conv --disable_bias_splitgelu --disable_bias_add --disable_packed_kv --disable_packed_qkv >> "%LOG%" 2>&1 || goto fail

echo [4/4] Checking that every part loads...
%P% -c "import onnxruntime as ort, glob; fs=sorted(glob.glob(r'%W%\fp16\*\model.onnx')); [print(f, [i.name for i in ort.InferenceSession(f, providers=['CPUExecutionProvider']).get_inputs()]) for f in fs]; assert fs" >> "%LOG%" 2>&1 || goto fail

echo Done. The model is in %W%\fp16 - tell Claude.
echo done %date% %time% >> "%LOG%"
pause
exit /b 0

:fail
echo Something failed. The log is %LOG% - tell Claude.
echo FAILED %date% %time% >> "%LOG%"
pause
exit /b 1
