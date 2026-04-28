Param()

$ErrorActionPreference = "Stop"

Write-Host "Configuring Git hooks path to .githooks ..."

git config core.hooksPath .githooks

Write-Host "Done. Pre-push checks will now run from .githooks/pre-push"
Write-Host "You can verify with: git config --get core.hooksPath"
