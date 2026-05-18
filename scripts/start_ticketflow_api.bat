@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
if "%API_HOST%"=="" set "API_HOST=127.0.0.1"
if "%API_PORT%"=="" set "API_PORT=8000"
if "%KG_BACKEND%"=="" set "KG_BACKEND=memory"
echo [TicketFlow] Starting FastAPI at http://%API_HOST%:%API_PORT%
python -c "from ticketflow.api import main; main()"
