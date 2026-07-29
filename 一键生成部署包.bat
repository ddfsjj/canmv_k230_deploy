@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "scripts\vq_deploy_packager_gui.py"
) else (
  py "scripts\vq_deploy_packager_gui.py"
)
