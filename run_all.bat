@echo off
setlocal enabledelayedexpansion

title NeuralFact - Launch All Services (RAG, Streamlit & Showcase)
color 0A

echo ===============================================================================
echo                NEURALFACT: LLM RAG & HALLUCINATION DETECTOR
echo                   [ Launching Full Suite in Background ]
echo ===============================================================================
echo.

cd /d "%~dp0"

if exist "venv\Scripts\streamlit.exe" (
    set "STREAMLIT_EXE=venv\Scripts\streamlit.exe"
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    set "STREAMLIT_EXE=streamlit"
    set "PYTHON_EXE=python"
)

:: 1. Check Ollama server
echo [*] Checking Ollama local LLM service (http://127.0.0.1:11434)...
powershell -NoProfile -Command "(Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -ErrorAction SilentlyContinue) | Out-Null"
if %ERRORLEVEL% NEQ 0 (
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        echo [*] Starting local Ollama server in background...
        start /b "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve >nul 2>&1
        timeout /t 3 >nul
    ) else (
        echo [!] Ollama binary not found at default path. Will fallback to simulated generation if needed.
    )
) else (
    echo [V] Ollama server is already running and ready!
)

:: 2. Start Showcase Web App (port 8080)
echo [*] Launching Netflix Technical Showcase on http://localhost:8080 ...
start "NeuralFact Showcase Server" /min "%PYTHON_EXE%" -m http.server 8080 --directory showcase

:: 3. Start Streamlit Application (port 8501)
echo [*] Launching Streamlit Interactive RAG App on http://localhost:8501 ...
start "NeuralFact Streamlit App" "%STREAMLIT_EXE%" run app/streamlit_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

echo.
echo ===============================================================================
echo [V] All services have been launched!
echo     - Streamlit Interactive RAG:  http://localhost:8501
echo     - Technical Showcase UI:      http://localhost:8080
echo.
echo [*] To stop all running servers at any time, simply run: stop_all.bat
echo ===============================================================================
timeout /t 5

