<#
.SYNOPSIS
    Task runner script for mtool-json-translate-local-lm.

.DESCRIPTION
    Provides functions for Static Code Analysis (sca), formatting (format),
    and autofixing (fix). Can be executed as a script with a target argument
    or dot-sourced to load functions into your current PowerShell session.

.EXAMPLE
    .\Make.ps1 sca
    .\Make.ps1 format
    .\Make.ps1 fix
    .\Make.ps1 all

.EXAMPLE
    . .\Make.ps1
    sca
    format
    fix
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("sca", "format", "fix", "all", "check", "lint", "help", "")]
    [string]$Target = "help",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

# Project paths to inspect/format
$script:ProjectTargets = @(
    "src",
    "clean_game_text.py",
    "main.py",
    "validate_translation.py"
)

function Get-PythonInterpreter {
    <#
    .SYNOPSIS
        Resolves the Python interpreter path with fallback heuristics.
    #>
    # 1. Check active virtual environment in current shell
    if ($env:VIRTUAL_ENV) {
        $candidate = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    # 2. Check local repository .venv
    $localVenv = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path $localVenv) {
        return $localVenv
    }

    # 3. Check known PyCharm / project virtual environment
    $defaultVenv = "C:\Users\inoy\PycharmProjects\mtool_translate\.venv\Scripts\python.exe"
    if (Test-Path $defaultVenv) {
        return $defaultVenv
    }

    # 4. Fallback to system PATH python
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    Write-Error "Python interpreter not found! Please activate a virtual environment or ensure Python is on PATH."
    exit 1
}

function Format-Header {
    param([string]$Title)
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Cyan
}

function sca {
    <#
    .SYNOPSIS
        Runs Static Code Analysis (Ruff check, Mypy type-checking, Pylint, Isort check).
    #>
    [CmdletBinding()]
    param()

    Format-Header "Static Code Analysis (SCA)"
    $python = Get-PythonInterpreter
    Write-Host "Using Python: $python`n" -ForegroundColor DarkGray

    $failedTools = @()

    # 1. Ruff Check
    Write-Host "[1/4] Running Ruff linter..." -ForegroundColor Yellow
    & $python -m ruff check $script:ProjectTargets
    if ($LASTEXITCODE -ne 0) {
        $failedTools += "ruff"
    } else {
        Write-Host "  [OK] Ruff check passed cleanly." -ForegroundColor Green
    }

    # 2. Mypy Type Check
    Write-Host "`n[2/4] Running Mypy type checker..." -ForegroundColor Yellow
    & $python -m mypy $script:ProjectTargets
    if ($LASTEXITCODE -ne 0) {
        $failedTools += "mypy"
    } else {
        Write-Host "  [OK] Mypy check passed cleanly." -ForegroundColor Green
    }

    # 3. Pylint Analysis
    Write-Host "`n[3/4] Running Pylint code analysis..." -ForegroundColor Yellow
    & $python -m pylint $script:ProjectTargets
    if ($LASTEXITCODE -ne 0) {
        $failedTools += "pylint"
    } else {
        Write-Host "  [OK] Pylint check passed cleanly." -ForegroundColor Green
    }

    # 4. Isort Check
    Write-Host "`n[4/4] Checking import sorting with Isort..." -ForegroundColor Yellow
    & $python -m isort --check-only --diff $script:ProjectTargets
    if ($LASTEXITCODE -ne 0) {
        $failedTools += "isort"
    } else {
        Write-Host "  [OK] Isort check passed cleanly." -ForegroundColor Green
    }

    # Summary
    Write-Host "`n--------------------------------------------------------" -ForegroundColor DarkGray
    if ($failedTools.Count -gt 0) {
        Write-Host "SCA FAILED: The following tool(s) reported issues: $($failedTools -join ', ')" -ForegroundColor Red
        Write-Host "Tip: Run '.\Make.ps1 fix' to automatically resolve formatting and common lint issues." -ForegroundColor Yellow
        if ($MyInvocation.InvocationName -ne '.') {
            exit 1
        }
        return $false
    } else {
        Write-Host "SCA PASSED: All static analysis checks completed successfully!" -ForegroundColor Green
        return $true
    }
}

function format {
    <#
    .SYNOPSIS
        Formats project code files with Ruff format and Isort.
    #>
    [CmdletBinding()]
    param()

    Format-Header "Code Formatter"
    $python = Get-PythonInterpreter
    Write-Host "Using Python: $python`n" -ForegroundColor DarkGray

    # 1. Isort import sorting
    Write-Host "[1/2] Sorting imports with Isort..." -ForegroundColor Yellow
    & $python -m isort $script:ProjectTargets

    # 2. Ruff format
    Write-Host "`n[2/2] Formatting code with Ruff..." -ForegroundColor Yellow
    & $python -m ruff format $script:ProjectTargets

    Write-Host "`nFormatting complete!" -ForegroundColor Green
    return $true
}

function fix {
    <#
    .SYNOPSIS
        Automatically fixes lint issues, sorts imports, and formats code.
    #>
    [CmdletBinding()]
    param()

    Format-Header "Auto-Fix & Format"
    $python = Get-PythonInterpreter
    Write-Host "Using Python: $python`n" -ForegroundColor DarkGray

    # 1. Ruff check --fix
    Write-Host "[1/3] Fixing lint errors with Ruff..." -ForegroundColor Yellow
    & $python -m ruff check --fix $script:ProjectTargets

    # 2. Isort
    Write-Host "`n[2/3] Sorting imports with Isort..." -ForegroundColor Yellow
    & $python -m isort $script:ProjectTargets

    # 3. Ruff format
    Write-Host "`n[3/3] Formatting code with Ruff..." -ForegroundColor Yellow
    & $python -m ruff format $script:ProjectTargets

    Write-Host "`nAuto-fix and formatting complete!" -ForegroundColor Green
    return $true
}

function Invoke-All {
    <#
    .SYNOPSIS
        Runs autofix, formatting, followed by a full static code analysis check.
    #>
    Format-Header "Running Full Pipeline (fix -> format -> sca)"
    fix
    format
    sca
}

function Show-Help {
    Format-Header "MTool Translator Task Runner (Make.ps1)"
    Write-Host "Available targets:" -ForegroundColor White
    Write-Host "  sca        " -NoNewline -ForegroundColor Green
    Write-Host "- Run static code analysis (Ruff check, Mypy, Pylint, Isort check)"
    Write-Host "  format     " -NoNewline -ForegroundColor Green
    Write-Host "- Format Python code using Ruff format and Isort"
    Write-Host "  fix        " -NoNewline -ForegroundColor Green
    Write-Host "- Automatically fix safe lint issues, sort imports, and format code"
    Write-Host "  all        " -NoNewline -ForegroundColor Green
    Write-Host "- Run fix, format, and then sca"
    Write-Host "  help       " -NoNewline -ForegroundColor Green
    Write-Host "- Show this help message"
    Write-Host "`nUsage examples:" -ForegroundColor White
    Write-Host "  .\Make.ps1 sca" -ForegroundColor DarkCyan
    Write-Host "  .\Make.ps1 format" -ForegroundColor DarkCyan
    Write-Host "  .\Make.ps1 fix" -ForegroundColor DarkCyan
    Write-Host "  .\Make.ps1 all" -ForegroundColor DarkCyan
    Write-Host "`nOr dot-source for interactive shell usage:" -ForegroundColor White
    Write-Host "  . .\Make.ps1" -ForegroundColor DarkCyan
    Write-Host "  sca" -ForegroundColor DarkCyan
}

# Aliases
Set-Alias -Name check -Value sca -Scope Script -ErrorAction SilentlyContinue
Set-Alias -Name lint -Value sca -Scope Script -ErrorAction SilentlyContinue

# Execute target if script is run directly (not dot-sourced)
if ($MyInvocation.InvocationName -ne '.' -and $MyInvocation.Line -notmatch '^\s*\.\s+') {
    switch ($Target.ToLowerInvariant()) {
        "sca"    { sca }
        "format" { format }
        "fix"    { fix }
        "all"    { Invoke-All }
        "check"  { sca }
        "lint"   { sca }
        "help"   { Show-Help }
        ""       { Show-Help }
        default  { Show-Help }
    }
}
