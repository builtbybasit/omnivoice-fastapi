@echo off
rem Double-click to benchmark generation speed in WSL. Run start.bat once first: it installs uv and clones the project.
rem REPO is the project folder inside WSL; change it if you cloned somewhere else.
set REPO=~/omnivoice-fastapi
title OmniVoice benchmark

set VOICE=
set /p VOICE=Voice id to benchmark (press Enter for a designed voice): 
set ARGS=
if not "%VOICE%"=="" set ARGS=--voice %VOICE%
set MODE=
set /p MODE=Find the largest batch the GPU can handle instead of timing batches 1-8? (y/N): 
if /i "%MODE%"=="y" set ARGS=%ARGS% --find-max

echo ==^> Starting WSL (can take a few seconds)
wsl.exe --cd ~ -e bash -lc "echo ==\> Updating the project in %REPO%; git -C %REPO% pull --ff-only || echo Could not update, using the version already there."
wsl.exe --cd ~ -e bash -lc "export PATH=$HOME/.local/bin:$PATH; cd %REPO% && bash scripts/run-wsl.sh --setup-only && echo && echo ==\> Running the benchmark: each case runs after a warm-up, so this takes a few minutes && OMNIVOICE_ENABLE_FLASHINFER=true uv run --no-sync python scripts/benchmark.py %ARGS%"
pause
