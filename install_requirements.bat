@echo off
setlocal

set VENV_PY=.venv\Scripts\python.exe

if not exist %VENV_PY% (
  echo [INFO] .venv nao encontrado. Criando ambiente virtual em .venv...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERRO] Falha ao criar .venv.
    exit /b 1
  )
)

echo [INFO] Atualizando pip...
%VENV_PY% -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERRO] Falha ao atualizar pip.
  exit /b 1
)

echo [INFO] Instalando torch (CPU) primeiro...
%VENV_PY% -m pip install --default-timeout=300 --index-url https://download.pytorch.org/whl/cpu torch
if errorlevel 1 (
  echo [ERRO] Falha ao instalar torch (CPU).
  exit /b 1
)

echo [INFO] Instalando demais dependencias do requirements.txt...
%VENV_PY% -m pip install --default-timeout=300 -r requirements.txt
if errorlevel 1 (
  echo [ERRO] Falha na instalacao das dependencias do requirements.txt.
  exit /b 1
)

echo [OK] Dependencias instaladas com sucesso no ambiente .venv.
endlocal
