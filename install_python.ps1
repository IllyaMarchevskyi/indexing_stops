#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Host "Missing '$Name'. Install it and try again." -ForegroundColor Red
        exit 1
    }
}

Require-Command "winget"

Write-Host "Installing Python 3..." -ForegroundColor Cyan
winget install --id Python.Python.3.12 -e --source winget

Write-Host "Installing pyserial..." -ForegroundColor Cyan
python -m ensurepip --upgrade | Out-Null
python -m pip install --upgrade pip pyserial

Write-Host ""
Write-Host "Done. Restart PowerShell/terminal to refresh PATH." -ForegroundColor Green
