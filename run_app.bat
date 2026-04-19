@echo off
REM Brain Tumor Detection Streamlit App Launcher

echo Activating virtual environment and launching app...
cd /d %~dp0

call venv\Scripts\activate.bat

REM Use full path to venv python to avoid system python
%cd%\venv\Scripts\python.exe -m streamlit run app.py

pause
