# Windows dev entrypoints — mirrors the Makefile targets.
# Usage: .\dev.ps1 <infra|dev|test|lint|seed>
param(
    [Parameter(Position = 0)]
    [ValidateSet("infra", "dev", "dashboard", "test", "lint", "seed", "clean")]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
$Py = ".\.venv\Scripts\python.exe"

switch ($Target) {
    "infra" {
        # Redis lives in WSL; 0.0.0.0 binding is required for Windows to reach it.
        wsl -d Ubuntu -- bash -lc "sudo service redis-server start; redis-cli ping"
    }
    "dev" {
        & $Py -m uvicorn gateway.main:app --reload --port 8000
    }
    "dashboard" {
        Push-Location dashboard
        npm run dev
        Pop-Location
    }
    "test" {
        $env:APP_ENV = "test"
        $env:DATABASE_URL = "sqlite+aiosqlite:///:memory:"
        & $Py -m pytest --cov=gateway --cov-report=term-missing
    }
    "lint" {
        & $Py -m ruff check gateway tests
        & $Py -m ruff format --check gateway tests
    }
    "seed" {
        & $Py -m gateway.cli seed
    }
    "clean" {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
            .pytest_cache, .ruff_cache, htmlcov, gateway.db, gateway.db-wal, gateway.db-shm
    }
    default {
        Write-Host "Usage: .\dev.ps1 <infra|dev|dashboard|test|lint|seed|clean>"
    }
}
