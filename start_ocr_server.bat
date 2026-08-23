@echo off
:: EyM Sanskrit OCR Server Launcher
:: Starts the Parinamika Flask backend for use with index.html
echo ================================================================
echo  EyM Sanskrit OCR — Parinamika Model Server
echo  Brahmavidya: Foundation for Indian Arts
echo ================================================================
echo.

:: Check for virtual environment
if exist "%~dp0venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call "%~dp0venv\Scripts\activate.bat"
) else if exist "%~dp0env\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (env)...
    call "%~dp0env\Scripts\activate.bat"
) else (
    echo [WARN] No virtual environment found. Using system Python.
    echo [HINT] Create one with: python -m venv venv
    echo [HINT] Then install: pip install -r requirements_ocr.txt
    echo.
)

:: Check for model files
if not exist "%~dp0modelss\" (
    echo [WARN] modelss\ folder not found.
    echo [INFO] Download the Parinamika model from:
    echo        https://drive.google.com/file/d/1KJ6vORY-Ybi_ldvdj2cDGAYsnRG4wdCR/view
    echo [INFO] Extract into: %~dp0modelss\
    echo [INFO] Server will start in Tesseract fallback mode.
    echo.
)

echo [INFO] Starting EyM OCR server on http://127.0.0.1:5001 ...
echo [INFO] Press Ctrl+C to stop.
echo.

cd /d "%~dp0"
python ocr_server.py

pause
