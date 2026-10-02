@echo off
rem Double-click to start the OmniVoice server in WSL. Copy this file anywhere on Windows (e.g. the Desktop).
rem The first run installs uv and clones the project into WSL if they are missing.
rem REPO is the project folder inside WSL; change it if you cloned somewhere else.
set REPO=~/omnivoice-fastapi
set REPO_URL=https://github.com/builtbybasit/omnivoice-fastapi.git
title OmniVoice server

echo ==^> Starting WSL (can take a few seconds)
wsl.exe --cd ~ bash -lc "echo ==\> Checking uv; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"
wsl.exe --cd ~ bash -lc "echo ==\> Checking the project in %REPO%; [ -d %REPO% ] || git clone %REPO_URL% %REPO%"
wsl.exe --cd ~ bash -lc "export PATH=$HOME/.local/bin:$PATH; cd %REPO% && bash scripts/run-wsl.sh"
pause
