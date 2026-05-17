@echo off
setlocal

cd /d "%~dp0"

if exist "D:\Anaconda\Scripts\activate.bat" (
  call "D:\Anaconda\Scripts\activate.bat" base
)

start "" powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 3; Start-Process 'http://localhost:8501'"

streamlit run src\ticketflow\app.py --server.headless true --browser.gatherUsageStats false
