@echo off
title SEMANA 6 - Tres en Raya IA + OpenCV
cd /d "%~dp0"

echo ===============================================
echo  Proyecto Semana 6 - Tres en Raya IA + OpenCV
echo ===============================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo Creando entorno virtual...
    py -3.14 -m venv venv
    if errorlevel 1 (
        echo No se pudo con Python 3.14, probando con python...
        python -m venv venv
    )
)

call "venv\Scripts\activate.bat"

echo.
echo Instalando librerias necesarias...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Iniciando proyecto...
python app.py

pause
