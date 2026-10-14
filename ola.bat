@echo off
chcp 65001 > nul
if "%~1"=="" (
    python -m src.cli chat
) else (
    python -m src.cli route --prompt "%*"
)
