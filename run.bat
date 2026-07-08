@echo off
REM Activates the project venv and launches demo.py.
REM demo.py loads ACE-Step 1.5 in-process, so no separate API server is needed.
REM Once the Gradio server is up, opens Firefox fullscreen (kiosk) on it.

setlocal
set "ROOT=%~dp0"
set "URL=http://localhost:7860"
set "FIREFOX=C:\Program Files\Mozilla Firefox\firefox.exe"

if "%~1"=="--browser" goto browser

if not exist "%ROOT%.venv\Scripts\activate.bat" (
    echo .venv not found - run install.ps1 first
    exit /b 1
)

call "%ROOT%.venv\Scripts\activate.bat"

REM Launch the browser watcher in the background, then run the app in
REM the foreground so Ctrl+C still stops it.
start "" /b cmd /c ""%~f0" --browser"
python "%ROOT%demo.py"
exit /b

:browser
REM Poll until the Gradio server answers, then open Firefox in kiosk mode.
:wait_server
curl -s -o nul "%URL%" && goto open_browser
timeout /t 1 /nobreak >nul
goto wait_server

:open_browser
start "" "%FIREFOX%" -kiosk "%URL%"
exit /b
