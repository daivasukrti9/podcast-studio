@echo off
title Podcast Studio y Libreria Podcast
echo ============================================================================
echo   Reiniciando Podcast Studio (Diseno Vidrio Pastel) y Libreria Podcast
echo   URL Local: http://127.0.0.1:8085
echo ============================================================================
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8085 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
timeout /t 1 /nobreak >nul
start "" "http://127.0.0.1:8085/?v=%RANDOM%"
python app.py
pause
