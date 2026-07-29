@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Creating local virtual environment...
  py -3 -m venv .venv
)

".venv\Scripts\python.exe" -c "import gradio, huggingface_hub, pandas" >nul 2>nul
if errorlevel 1 (
  echo Installing GUI dependencies. This only needs to happen once...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Failed to install dependencies.
    pause
    exit /b 1
  )
)

set K230_DEPLOY_PORT=7861
set K230_DEPLOY_PORT_RANGE=20
set GRADIO_ANALYTICS_ENABLED=False
echo.
echo Starting K230 deploy GUI...
echo.
".venv\Scripts\python.exe" "k230_deploy_gui.py"

echo.
echo GUI stopped.
pause
