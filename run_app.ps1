# Brain Tumor Detection - Streamlit Launcher (PowerShell)

Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host "Starting Streamlit app..." -ForegroundColor Green
Write-Host ""

# Use explicit python from venv
& .\venv\Scripts\python.exe -m streamlit run app.py
