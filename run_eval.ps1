# One-Click Dual-Profile Evaluator for VS Code Terminal
param(
    [Parameter(Position=0)]
    [ValidateSet("champion", "turbo", "speed", "fast")]
    [string]$Mode = "champion"
)

# Set the active evaluation profile
$env:EVAL_MODE = $Mode

if (-not $env:OPENAI_API_KEY) {
    # Check if .env file exists and load it
    if (Test-Path ".env") {
        Get-Content ".env" | ForEach-Object {
            if ($_ -match '^\s*([^#=]+)\s*=\s*(.*)\s*$') {
                [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
            }
        }
    }
}

$env:EVAL_EMBEDDER_MODULE = "main"
$env:EVAL_GENERATOR_MODULE = "main"
$env:PYTHONPATH = "C:\Users\ashee\Desktop\rag-local-eval-loop;C:\Users\ashee\Desktop\voice-rag"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Write-Host "==========================================================" -ForegroundColor Cyan
if ($Mode -eq "turbo") {
    Write-Host "  ⚡ PROFILE: TURBO (Ultra-Low Latency Sub-200ms)         " -ForegroundColor Green
} else {
    Write-Host "  🏆 PROFILE: CHAMPION (High Accuracy & Faithfulness)     " -ForegroundColor Yellow
}
Write-Host "==========================================================" -ForegroundColor Cyan

& ".\venv\Scripts\python.exe" -m eval.runner `
    --rag-root "C:\Users\ashee\Desktop\voice-rag" `
    --num-answerable 50 `
    --num-unanswerable 50 `
    --workers 1 `
    --judge-workers 16
