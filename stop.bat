@echo off
rem Double-click to stop the OmniVoice server or benchmark left running in WSL, freeing GPU and RAM.
rem REPO is the project folder inside WSL; change it if you cloned somewhere else.
set REPO=~/omnivoice-fastapi
title OmniVoice stop

echo ==^> Stopping OmniVoice in WSL
wsl.exe --cd ~ -e bash -lc "cd %REPO% && bash scripts/stop.sh"
pause
