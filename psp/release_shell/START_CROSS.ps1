$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "CROSS_UX.ps1")

function Pause-Cross { Write-Host ""; Read-Host "Press Enter to return to the menu" | Out-Null }
function Confirm-Cross([string]$Message) {
    $answer = Read-Host "$Message [y/N]"
    return $answer -match '^(?i:y|yes)$'
}
function Ensure-CrossEnvironment {
    $python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    $requiredVersion = "1.0.0"
    $needsInstall = -not (Test-Path $python)
    $actionLabel = "installation"

    if (-not $needsInstall) {
        $installedVersion = (& $python -c "import importlib.metadata as m; print(m.version('cross-psp-two-stream'))" 2>$null | Select-Object -Last 1).Trim()
        if ($LASTEXITCODE -eq 0 -and $installedVersion -eq $requiredVersion) {
            Write-Host ("CROSS environment ready: PSP {0}" -f $installedVersion) -ForegroundColor Green
            Write-Host ("Environment: {0}" -f $python)
            return $python
        }
        Write-Host ("CROSS environment needs upgrade: installed PSP '{0}', required '{1}'." -f $installedVersion, $requiredVersion) -ForegroundColor Yellow
        $needsInstall = $true
        $actionLabel = "upgrade"
    } else {
        Write-Host "CROSS environment is not installed yet." -ForegroundColor Yellow
    }

    if (-not (Test-Path ".\INSTALL.ps1")) { throw "INSTALL.ps1 is missing." }
    if (-not (Confirm-Cross ("Run environment {0} now?" -f $actionLabel))) { return $null }
    Write-Host ""
    Write-Host ("Starting environment {0}. Progress stages will be shown below." -f $actionLabel) -ForegroundColor Cyan
    & powershell -NoProfile -ExecutionPolicy Bypass -File ".\INSTALL.ps1"
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "Environment setup failed with exit code $code." }
    if (-not (Test-Path $python)) { throw "Environment setup completed without creating .venv." }
    $installedVersion = (& $python -c "import importlib.metadata as m; print(m.version('cross-psp-two-stream'))" | Select-Object -Last 1).Trim()
    if ($LASTEXITCODE -ne 0 -or $installedVersion -ne $requiredVersion) { throw "Environment verification failed: PSP version is '$installedVersion', expected '$requiredVersion'." }
    Write-Host ""
    Write-Host "CROSS ENVIRONMENT INSTALLED/UPGRADED AND VERIFIED" -ForegroundColor Green
    Write-Host ("PSP version: {0}" -f $installedVersion)
    Write-Host ("Environment: {0}" -f $python)
    return $python
}
function Invoke-CrossChildScript([string]$ScriptName, [string]$Label) {
    $scriptPath = Join-Path $PSScriptRoot $ScriptName
    if (-not (Test-Path $scriptPath)) { throw "$ScriptName is missing." }
    Write-Host ""
    Write-Host ("STARTING: {0}" -f $Label) -ForegroundColor Cyan
    & powershell -NoProfile -ExecutionPolicy Bypass -File $scriptPath
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Host ""
        Write-Host ("{0} FAILED / FAIL-CLOSED (status {1})" -f $Label, $code) -ForegroundColor Red
        return $false
    }
    Write-Host ""
    Write-Host ("{0}: child process returned SUCCESS" -f $Label) -ForegroundColor Green
    return $true
}
function Run-Wizard([string]$Mode) {
    $python = Ensure-CrossEnvironment
    if ($null -eq $python) {
        Write-Host "No analysis started because the environment is not installed." -ForegroundColor Yellow
        Pause-Cross
        return
    }
    Write-Host ""
    Write-Host ("STARTING WIZARD MODE: {0}" -f $Mode.ToUpperInvariant()) -ForegroundColor Cyan
    & $python -m cross_psp_two_stream.cli wizard --mode $Mode
    $code = $LASTEXITCODE
    if ($code -eq 3) { return }
    if ($code -ne 0) {
        Write-Host ("Wizard finished with FAIL-CLOSED/error status {0}." -f $code) -ForegroundColor Red
    } else {
        Write-Host "Wizard returned successfully. If an analysis was run, its primary result was printed and opened automatically above." -ForegroundColor Green
    }
    Pause-Cross
}
function Reproduce-Article {
    while ($true) {
        Clear-Host
        Write-Host "REPRODUCE THE ARTICLE"
        Write-Host "1) Fast exact analysis/report reproduction (seconds; no simulation rerun)"
        Write-Host "2) Verify full 726-point campaign preflight (no 726-point execution)"
        Write-Host "3) Full 726-point from-scratch reproduction (resumable; long)"
        Write-Host "B) Back"
        $choice = (Read-Host "Choose").Trim().ToUpperInvariant()
        if ($choice -eq "B") { return }
        switch ($choice) {
            "1" {
                Write-Host "This regenerates the published REPORT.html from frozen authoritative result archives."
                if (Confirm-Cross "Start fast reproduction?") {
                    $python = Ensure-CrossEnvironment
                    if ($null -ne $python) { [void](Invoke-CrossChildScript "REPRODUCE_ARTICLE_FAST.ps1" "FAST ARTICLE REPRODUCTION") }
                }
                Pause-Cross
            }
            "2" {
                Write-Host "This validates all three frozen campaign contracts: 432 + 291 + 3 = 726 points."
                if (Confirm-Cross "Run 726-point preflight?") {
                    $python = Ensure-CrossEnvironment
                    if ($null -ne $python) { [void](Invoke-CrossChildScript "CHECK_FULL_726_PREFLIGHT.ps1" "726-POINT PREFLIGHT") }
                }
                Pause-Cross
            }
            "3" {
                Write-Host "Workload: 726 simulation points. The run is checkpointed/resumable and long."
                if (Confirm-Cross "Start the full 726-point rerun?") {
                    $python = Ensure-CrossEnvironment
                    if ($null -ne $python) { [void](Invoke-CrossChildScript "REPRODUCE_ARTICLE_FULL_726.ps1" "FULL 726-POINT REPRODUCTION") }
                }
                Pause-Cross
            }
            default { Write-Host "Unknown choice."; Pause-Cross }
        }
    }
}
function Main-Cross {
    while ($true) {
        Clear-Host
        Write-Host "CROSS Research Release 1.0.0 RC1"
        Write-Host "1) Reproduce the article"
        Write-Host "2) Analyze custom traffic on qualified reference physics"
        Write-Host "3) Analyze a custom physical configuration"
        Write-Host "4) Build/run a robustness campaign"
        Write-Host "L) Load an existing saved custom configuration"
        Write-Host "Q) Quit"
        Write-Host ""
        Write-Host "Successful analyses/reproductions automatically open the primary result file."
        Write-Host "Results are model-derived research diagnostics, not deployment advice."
        $choice = (Read-Host "Choose").Trim().ToUpperInvariant()
        switch ($choice) {
            "1" { Reproduce-Article }
            "2" { Run-Wizard "traffic" }
            "3" { Run-Wizard "physics" }
            "4" { Run-Wizard "robustness" }
            "L" { Run-Wizard "load" }
            "Q" { return }
            default { Write-Host "Unknown choice."; Pause-Cross }
        }
    }
}

try {
    Main-Cross
    exit 0
}
catch {
    Write-Host ""
    Write-Host "CROSS launcher error:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "The launcher will stay open so the error can be read."
    Pause-Cross
    exit 1
}
