<#
.SYNOPSIS
    High-performance atomic block-patching tool for Windows PowerShell.
.DESCRIPTION
    Wraps patch_block.exe / patch_block.py to provide zero-escaping block editing for AI agents and developers.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$File,

    [Parameter(Mandatory = $false)]
    [string]$Search,

    [Parameter(Mandatory = $false)]
    [string]$Replace = "",

    [Parameter(Mandatory = $false)]
    [string]$SearchFile,

    [Parameter(Mandatory = $false)]
    [string]$ReplaceFile,

    [Parameter(Mandatory = $false)]
    [string]$PayloadFile,

    [switch]$AllowMultiple,

    [switch]$CleanTmp
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $scriptDir "patch_block.exe"

if (-not (Test-Path -LiteralPath $exe)) {
    $py = Join-Path $scriptDir "patch_block.py"
    $exe = "python"
    $baseArgs = @($py)
} else {
    $baseArgs = @()
}

$cliArgs = @()
if ($PayloadFile) {
    $cliArgs += @("--payload-file", (Resolve-Path -LiteralPath $PayloadFile).Path)
} else {
    $resolvedFile = if (Test-Path -LiteralPath $File) { (Resolve-Path -LiteralPath $File).Path } else { $File }
    $cliArgs += @("--file", $resolvedFile)

    if ($SearchFile) {
        $resolvedSearch = (Resolve-Path -LiteralPath $SearchFile).Path
        $cliArgs += @("--search-file", $resolvedSearch)
    } elseif ($Search) {
        $cliArgs += @("--search", $Search)
    }

    if ($ReplaceFile) {
        $resolvedReplace = (Resolve-Path -LiteralPath $ReplaceFile).Path
        $cliArgs += @("--replace-file", $resolvedReplace)
    } elseif ($PSBoundParameters.ContainsKey('Replace')) {
        $cliArgs += @("--replace", $Replace)
    }
}

if ($AllowMultiple) { $cliArgs += "--allow-multiple" }
if ($CleanTmp) { $cliArgs += "--clean-tmp" }

& $exe @baseArgs @cliArgs
exit $LASTEXITCODE
