@echo off
echo === FinSight setup ===
python -m venv venv || (echo Python not found. Install Python 3.10+ from python.org and tick "Add to PATH". & pause & exit /b 1)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo Install failed - see the error above. & pause & exit /b 1)
if not exist .env copy .env.example .env >nul
echo.
echo Setup complete. Now run:  run.bat
echo (Optional: open .env in Notepad and paste your API keys.)
pause
