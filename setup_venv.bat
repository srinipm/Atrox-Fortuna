@echo off
echo Setting up virtual environment for RAG Code Search...

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install requirements
echo Installing required packages...
pip install -r requirements.txt

echo Setup complete!
echo.
echo To activate the virtual environment, run:
echo venv\Scripts\activate.bat
echo.
echo To deactivate the virtual environment, run:
echo deactivate
echo.
echo To use the RAG tool, make sure the virtual environment is activated.
