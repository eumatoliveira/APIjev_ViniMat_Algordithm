@echo off
chcp 65001 > nul
echo ======================================================================
echo    MATROUTE-AI — INICIANDO AMBIENTE WEB DE TESTES (PORTA 8000)
echo ======================================================================
echo Abrindo navegador em http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
python -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000
