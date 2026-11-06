# Fixed configurations, new init seeds, no search. ASCII for PowerShell 5.1.
param(
    [string]$PythonPath = (Get-Command python -CommandType Application).Source,
    [ValidateSet('digits', 'usps', 'coil20', 'mnist_10000', 'har')]
    [string[]]$Datasets = @('digits', 'usps', 'coil20', 'mnist_10000', 'har'),
    [int[]]$DataSeeds = @(0, 1, 2, 3, 4),
    [ValidateSet('ours', 'fkm', 'ffcm', 'nnfc', 'kfed', 'fedsc')]
    [string[]]$Methods = @('ours', 'fkm', 'ffcm', 'nnfc', 'kfed', 'fedsc'),
    [string]$InitSeeds = '100,101,102,103,104,105,106,107,108,109',
    [switch]$DryRun,
    [switch]$SummarizeOnly
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$entryPath = Join-Path $PSScriptRoot 'run_paired_wilcoxon.py'
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Python not found: $PythonPath" }
if (-not $DataSeeds.Count -or @($DataSeeds | Where-Object { $_ -notin 0..4 }).Count -or
    @($DataSeeds | Select-Object -Unique).Count -ne $DataSeeds.Count) {
    throw 'Use distinct DataSeeds from 0..4.'
}
if (-not $Datasets.Count -or @($Datasets | Select-Object -Unique).Count -ne $Datasets.Count -or
    -not $Methods.Count -or @($Methods | Select-Object -Unique).Count -ne $Methods.Count) {
    throw 'Use nonempty, distinct datasets and methods.'
}
$seeds = @($InitSeeds.Split(',') | ForEach-Object { [int]$_.Trim() })
if (-not $seeds.Count -or @($seeds | Where-Object { $_ -notin 100..109 }).Count -or
    @($seeds | Select-Object -Unique).Count -ne $seeds.Count) {
    throw 'Use distinct evaluation init seeds from 100..109.'
}
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$failedTasks = [System.Collections.Generic.List[string]]::new()

if (-not $SummarizeOnly) {
    foreach ($dataset in $Datasets) {
        $clients = if ($dataset -in @('usps', 'har')) { 50 } else { 20 }
        foreach ($dataSeed in $DataSeeds) {
            foreach ($method in $Methods) {
                $cli = @('--dataset', $dataset, '--grid-split', "$dataSeed", '--method', $method,
                    '--beta', '0.5', '--clients', "$clients", '--init-seeds', $InitSeeds,
                    '--data-dir', "$projectPath/data", '--resume')
                if ($DryRun) {
                    Write-Host "PLAN: $dataset / split=$dataSeed / $method / init=$InitSeeds"
                    continue
                }
                Write-Host "`nSTART: $dataset / split=$dataSeed / $method"
                & $PythonPath -B $entryPath @cli
                if ($LASTEXITCODE -ne 0) {
                    $failedTasks.Add("$dataset / split=$dataSeed / $method")
                    Write-Warning 'Task incomplete. Saved successes are retained; continuing independent tasks.'
                }
            }
        }
    }
}
if ($failedTasks.Count) {
    throw "Incomplete task(s): $($failedTasks -join '; '). Run the same command to resume."
}
$fullScope = $Datasets.Count -eq 5 -and $DataSeeds.Count -eq 5 -and $Methods.Count -eq 6 -and $seeds.Count -eq 10
if ($DryRun) {
    $fits = $Datasets.Count * $DataSeeds.Count * $Methods.Count * $seeds.Count
    if ($SummarizeOnly) { Write-Host 'PLAN: validate all 1500 fits and summarize; no training.' }
    else { Write-Host "`nPlan: $fits fixed fits. No training or files." }
}
elseif ($SummarizeOnly -or $fullScope) {
    & $PythonPath -B $entryPath --summarize-only --data-dir "$projectPath/data"
    if ($LASTEXITCODE -ne 0) { throw 'Summary failed. A complete 1500-fit comparison is required.' }
}
else {
    Write-Host 'Requested subset completed. The final summary requires all datasets, splits, methods and ten seeds.'
}
