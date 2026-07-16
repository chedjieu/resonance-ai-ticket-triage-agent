@echo off
setlocal
set "PROJECT=%~dp0..\.."
set "PYTHON=%PROJECT%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" "%PROJECT%\deploy\bin\zip_wrapper.py" %*
exit /b %ERRORLEVEL%
