@echo off
REM Quick start script for AI Contraception Counseling System (Windows)

echo =========================================
echo AI Contraception Counseling System Setup
echo =========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo Virtual environment created!
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
set /p INSTALL="Would you like to install dependencies now? (y/n): "
if /i "%INSTALL%"=="y" (
    echo Installing dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo Dependencies installed!
) else (
    echo Skipping dependency installation.
    echo To install later, run: pip install -r requirements.txt
)
echo.

REM Set up environment file
if not exist .env (
    echo Creating .env file from template...
    copy .env.template .env
    echo .env file created! Please edit it and add your API keys.
) else (
    echo .env file already exists.
)
echo.

REM Create necessary directories
echo Ensuring all directories exist...
mkdir data\who 2>nul
mkdir data\bcs 2>nul
mkdir data\synthetic 2>nul
mkdir data\processed 2>nul
mkdir data\memory 2>nul
mkdir results\tables 2>nul
mkdir results\plots 2>nul
mkdir results\logs 2>nul
echo Directories created!
echo.

echo =========================================
echo Setup Complete!
echo =========================================
echo.
echo Next steps:
echo 1. Edit .env file and add your API keys
echo 2. Review and customize configs\config.yaml
echo 3. Place WHO and BCS+ PDFs in data\who\ and data\bcs\
echo 4. Run data preprocessing: python src\rag\preprocess_documents.py
echo 5. Start the API: uvicorn src.api.main:app --reload
echo.
echo For more information, see README.md
echo.
pause
