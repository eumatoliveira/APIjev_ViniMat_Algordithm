@echo off
chcp 65001 > nul
if "%~1"=="" (
    python -m src.cli gemma --prompt "Explique como funciona a matriz de convolucao Sobel para imagens."
) else (
    python -m src.cli gemma --prompt "%*"
)
