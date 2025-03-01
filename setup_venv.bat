@echo off
echo =====================================
echo FICC Spread Analysis - Setup Script
echo =====================================

rem Set virtual environment name
set VENV_NAME=ficc_env

rem Create directory structure if it doesn't exist
mkdir ficc_ai\data
mkdir ficc_ai\models
mkdir ficc_ai\visualizations

rem Create virtual environment
echo Creating virtual environment: %VENV_NAME%
python -m venv %VENV_NAME%

rem Activate virtual environment
echo Activating virtual environment
call %VENV_NAME%\Scripts\activate

rem Install dependencies
echo Installing dependencies from requirements.txt
pip install -r ficc_ai\requirements.txt

echo =====================================
echo Setup complete!
echo.
echo To activate the virtual environment manually:
echo %VENV_NAME%\Scripts\activate
echo.
echo To run the FICC analysis application:
echo python ficc_ai\ficc_ai.py
echo =====================================
