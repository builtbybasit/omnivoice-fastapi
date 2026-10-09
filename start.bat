@echo off
rem Double-click to start the OmniVoice server in WSL. Copy this file anywhere on Windows (e.g. the Desktop).
rem Installs uv and clones the project into WSL if they are missing, and pulls the latest changes on every start.
rem REPO is the project folder inside WSL; change it if you cloned somewhere else.
set REPO=~/omnivoice-fastapi
set REPO_URL=https://github.com/builtbybasit/omnivoice-fastapi.git
title OmniVoice server

echo ==^> Starting WSL (can take a few seconds)
wsl.exe --cd ~ -e bash -lc "echo ==\> Checking uv; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"
wsl.exe --cd ~ -e bash -lc "echo ==\> Updating the project in %REPO%; if [ -d %REPO% ]; then git -C %REPO% pull --ff-only || echo Could not update, starting the version already there.; else git clone %REPO_URL% %REPO%; fi"
rem A .env next to this file replaces the one in WSL on every start. The trailing "." stops the path's last \ from escaping the quote.
if exist "%~dp0.env" wsl.exe --cd "%~dp0." -e bash -lc "echo ==\> Copying .env into %REPO%; cp .env %REPO%/.env"
wsl.exe --cd ~ -e bash -lc "export PATH=$HOME/.local/bin:$PATH; cd %REPO% && bash scripts/run-wsl.sh"
pause
