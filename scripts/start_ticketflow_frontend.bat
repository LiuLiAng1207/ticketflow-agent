@echo off
setlocal
cd /d "%~dp0\..\frontend"
if "%npm_config_cache%"=="" set "npm_config_cache=%CD%\.npm-cache"
if not exist node_modules (
  echo [TicketFlow] Installing frontend dependencies...
  call npm install
  if errorlevel 1 exit /b %errorlevel%
)
if "%VITE_TICKETFLOW_API_BASE%"=="" set "VITE_TICKETFLOW_API_BASE=http://127.0.0.1:8000"
echo [TicketFlow] Starting React console at http://127.0.0.1:5173
call npm run dev
