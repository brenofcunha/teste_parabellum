[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "[INFO] Iniciando ativacao do ambiente virtual (.venv)..."
$activateScript = ".venv/Scripts/Activate.ps1"

if (-not (Test-Path -Path $activateScript)) {
    Write-Host "[INFO] .venv nao encontrado. Criando ambiente virtual em .venv..."
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERRO] Falha ao criar .venv." -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path -Path $activateScript)) {
    Write-Host "[ERRO] Arquivo de ativacao nao encontrado em .venv/Scripts/Activate.ps1" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Ativando .venv..."
. $activateScript

if ($env:VIRTUAL_ENV -and $env:VIRTUAL_ENV -match "\\.venv$") {
    Write-Host "[OK] Ambiente virtual .venv ativado com sucesso." -ForegroundColor Green
    return
}

Write-Host "[ALERTA] Nao foi possivel confirmar VIRTUAL_ENV=.venv nesta sessao." -ForegroundColor Yellow
Write-Host "[ALERTA] Execute manualmente: . .\\.venv\\Scripts\\Activate.ps1"
