@echo off
setlocal

set "PROJECT_DIR=%~dp0mehulu"
set "PYTHON_EXE=%PROJECT_DIR%\.venv311\Scripts\python.exe"
set "SCRIPT=%PROJECT_DIR%\realtime_detection.py"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  echo Create it first with: py -3.11 -m venv "%PROJECT_DIR%\.venv311"
  pause
  exit /b 1
)

if not exist "%SCRIPT%" (
  echo [ERROR] Script not found: %SCRIPT%
  pause
  exit /b 1
)

echo Starting realtime detection...
"%PYTHON_EXE%" "%SCRIPT%"

if errorlevel 1 (
  echo.
  echo [ERROR] Realtime detection exited with an error.
  pause
  exit /b 1
)

endlocal
