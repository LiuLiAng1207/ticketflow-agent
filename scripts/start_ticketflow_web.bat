@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
if "%API_HOST%"=="" set "API_HOST=127.0.0.1"
if "%API_PORT%"=="" set "API_PORT=8000"
if "%VITE_TICKETFLOW_API_BASE%"=="" set "VITE_TICKETFLOW_API_BASE=http://%API_HOST%:%API_PORT%"

echo [TicketFlow] Opening API and React frontend in separate windows.
start "TicketFlow API" cmd /k "cd /d %CD% && call scripts\start_ticketflow_api.bat"
start "TicketFlow Web" cmd /k "cd /d %CD% && call scripts\start_ticketflow_frontend.bat"
start "" "http://127.0.0.1:5173"
