@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_CROSS.ps1"
set "CROSS_EXIT=%ERRORLEVEL%"
if not "%CROSS_EXIT%"=="0" (
  echo.
  echo CROSS launcher exited with status %CROSS_EXIT%.
  echo Review the message above before closing this window.
  pause
)
exit /b %CROSS_EXIT%
