# Remaining validation only. ASCII for Windows PowerShell 5.1.
param(
    [string]$PythonPath = (Get-Command python -CommandType Application).Source,
    [int[]]$DataSeeds = @(0, 1, 2, 3, 4),
    [string]$InitSeeds = '40,41,42,43,44',
    [ValidateSet('subject', 'transport')]
    [string[]]$Stages = @('subject', 'transport'),
    [ValidateSet('digits', 'usps', 'coil20', 'mnist_10000', 'har')]
    [string[]]$Datasets = @('har', 'usps', 'digits', 'coil20', 'mnist_10000'),
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Python not found: $PythonPath" }
if (-not $DataSeeds.Count -or @($DataSeeds | Where-Object { $_ -notin 0..4 }).Count -or
    @($DataSeeds | Select-Object -Unique).Count -ne $DataSeeds.Count) {
    throw 'Use distinct DataSeeds from 0..4.'
}
$seeds = @($InitSeeds.Split(',') | ForEach-Object { [int]$_.Trim() })
if (-not $seeds.Count -or @($seeds | Where-Object { $_ -notin 40..44 }).Count -or
    @($seeds | Select-Object -Unique).Count -ne $seeds.Count) { throw 'Use distinct init seeds from 40..44.' }
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$failedTasks = [System.Collections.Generic.List[string]]::new()

function Invoke-Validation {
    param([string]$Label, [string]$Entry, [string[]]$CliArgs)
    if ($DryRun) {
        Write-Host "PLAN: $Label"
        Write-Host "$PythonPath $projectPath/$Entry $($CliArgs -join ' ')"
        return
    }
    Write-Host "`nSTART: $Label"
    & $PythonPath "$projectPath/$Entry" @CliArgs
    if ($LASTEXITCODE -ne 0) {
        $failedTasks.Add($Label)
        Write-Warning "Incomplete: $Label. Failure rows are retained; continuing independent tasks."
    }
}

if ('subject' -in $Stages) {
    $failuresBeforePrepare = $failedTasks.Count
    Invoke-Validation -Label 'Official HAR / frozen natural-subject folds' `
        -Entry 'data/prepare_har_subject_splits.py' `
        -CliArgs @('--output-dir', "$projectPath/data", '--folds', ($DataSeeds -join ','))
    if ($failedTasks.Count -gt $failuresBeforePrepare) { throw 'Subject data preparation failed; validation did not start.' }
    foreach ($fold in $DataSeeds) {
        $outDir = Join-Path $projectPath "experiments/results/har_subject/fold_${fold}_clients24_fixed"
        Invoke-Validation -Label "HAR subject fold=$fold / eleven fixed conditions, no search" `
            -Entry 'experiments/scripts/run_har_subject_reference.py' `
            -CliArgs @('--fold', "$fold", '--data-dir', "$projectPath/data",
                '--source-root', "$projectPath/experiments/results/har", '--out-dir', $outDir,
                '--init-seeds', $InitSeeds, '--resume')
    }
}

if ('transport' -in $Stages) {
    foreach ($dataset in $Datasets) {
        $clients = if ($dataset -in @('usps', 'har')) { 50 } else { 20 }
        foreach ($dataSeed in $DataSeeds) {
            $sourceName = switch ($dataset) {
                'digits' { "federated_${dataSeed}_heterogeneity" }
                'usps' { "federated_${dataSeed}_clients50_heterogeneity" }
                'har' { "federated_${dataSeed}_clients50_ours_tpe" }
                default { "federated_${dataSeed}_ours" }
            }
            $sourceDir = Join-Path $projectPath "experiments/results/$dataset/$sourceName"
            $outDir = Join-Path $projectPath "experiments/results/$dataset/federated_${dataSeed}_clients${clients}_full_roles"
            Invoke-Validation -Label "$dataset / seed=$dataSeed / full native-process x dense-partial CKKS" `
                -Entry 'experiments/scripts/run_ours_process_training.py' `
                -CliArgs @('--dataset', $dataset, '--grid-split', "$dataSeed", '--beta', '0.5',
                    '--clients', "$clients", '--data-dir', "$projectPath/data", '--source-dir', $sourceDir,
                    '--out-dir', $outDir, '--init-seeds', $InitSeeds, '--eigen-solvers', 'dense', 'partial', '--resume')
        }
    }
}

if ($failedTasks.Count) { throw "Batch attempted all independent tasks; $($failedTasks.Count) incomplete task(s): $($failedTasks -join '; ')" }
if ($DryRun) { Write-Host "`nPlan only; no downloads, frozen data or models were created." }
else { Write-Host "`nRequested final validation studies completed." }
