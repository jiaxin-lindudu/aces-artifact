# HE-focused revision studies. ASCII for Windows PowerShell 5.1.
param(
    [string]$PythonPath = (Get-Command python -CommandType Application).Source,
    [int[]]$DataSeeds = @(0, 1, 2, 3, 4),
    [string]$InitSeeds = '40,41,42,43,44',
    [ValidateSet('roles', 'solver', 'covtype', 'large-silo', 'central')]
    [string[]]$Stages = @('roles', 'solver', 'covtype', 'large-silo', 'central'),
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectPath
if (-not (Test-Path -LiteralPath $PythonPath)) { throw "Python not found: $PythonPath" }
if (@($DataSeeds | Where-Object { $_ -notin 0..4 }).Count) { throw 'DataSeeds must be in 0..4.' }
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$failedTasks = [System.Collections.Generic.List[string]]::new()

function Invoke-Study {
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
        Write-Warning "FAILED: $Label. Successful records are preserved; continuing independent tasks."
    }
}

foreach ($stage in @('roles', 'solver')) {
    if ($stage -notin $Stages) { continue }
    foreach ($datasetName in @('digits', 'usps', 'coil20', 'mnist_10000')) {
        $clients = if ($datasetName -eq 'usps') { 50 } else { 20 }
        foreach ($dataSeed in $DataSeeds) {
            $sourceName = switch ($datasetName) {
                'digits' { "federated_${dataSeed}_heterogeneity" }
                'usps' { "federated_${dataSeed}_clients50_heterogeneity" }
                default { "federated_${dataSeed}_ours" }
            }
            $sourceDir = Join-Path $projectPath "experiments/results/$datasetName/$sourceName"
            $outDir = Join-Path $projectPath "experiments/results/$datasetName/federated_${dataSeed}_clients${clients}_he_revision"
            $options = @('--dataset', $datasetName, '--grid-split', "$dataSeed", '--clients', "$clients",
                '--beta', '0.5', '--init-seeds', $InitSeeds, '--select-on', 'acc',
                '--data-dir', "$projectPath/data", '--source-dir', $sourceDir, '--out-dir', $outDir)
            if ($stage -eq 'roles') {
                Invoke-Study -Label "$datasetName / seed=$dataSeed / three-role snapshot" `
                    -Entry 'experiments/scripts/run_ckks_role_validation.py' -CliArgs ($options + @('--resume'))
            } else {
                Invoke-Study -Label "$datasetName / seed=$dataSeed / dense-partial CKKS" `
                    -Entry 'experiments/scripts/run_ours_solver_comparison.py' `
                    -CliArgs ($options + @('--study', 'solver', '--aggregation', 'ckks'))
            }
        }
    }
}

$needsCovtype = @($Stages | Where-Object { $_ -in @('covtype', 'large-silo', 'central') }).Count -gt 0
if ($needsCovtype) {
    $priorFailures = $failedTasks.Count
    Invoke-Study -Label 'Covtype-10k / frozen Stage 1' -Entry 'data/prepare_centralized_splits.py' `
        -CliArgs @('--datasets', 'covtype_10000', '--data-seeds', ($DataSeeds -join ','), '--output-dir', "$projectPath/data")
    Invoke-Study -Label 'Covtype-10k / frozen Stage 2' -Entry 'data/prepare_federated_partitions.py' `
        -CliArgs @('--datasets', 'covtype_10000', '--data-seeds', ($DataSeeds -join ','), '--betas', '0.5',
            '--clients', '20', '--input-dir', "$projectPath/data", '--output-dir', "$projectPath/data")
    if ($failedTasks.Count -gt $priorFailures) { throw 'Covtype preparation failed. See errors above; no Covtype fitting started.' }
}

foreach ($dataSeed in $DataSeeds) {
    $covtypeSource = Join-Path $projectPath "experiments/results/covtype_10000/federated_${dataSeed}_reference"
    if ('covtype' -in $Stages) {
        Invoke-Study -Label "Covtype-10k / seed=$dataSeed / six methods, <=4 candidates each" `
            -Entry 'experiments/scripts/run_covtype_reference.py' `
            -CliArgs @('--grid-split', "$dataSeed", '--clients', '20', '--beta', '0.5',
                '--init-seeds', $InitSeeds, '--select-on', 'acc', '--eigen-solver', 'dense',
                '--data-dir', "$projectPath/data", '--out-dir', $covtypeSource, '--resume')
    }
    if ('large-silo' -in $Stages) {
        $outDir = Join-Path $projectPath "experiments/results/covtype_10000/federated_${dataSeed}_large_silo"
        Invoke-Study -Label "Covtype-10k / seed=$dataSeed / actual large-silo cost" `
            -Entry 'experiments/scripts/run_ours_solver_comparison.py' `
            -CliArgs @('--study', 'large-silo', '--dataset', 'covtype_10000', '--grid-split', "$dataSeed",
                '--clients', '20', '--beta', '0.5', '--init-seeds', $InitSeeds, '--select-on', 'acc',
                '--aggregation', 'ckks', '--view-clients', '3', '--local-samples', '128', '512', '1024', '2048',
                '--benchmark-rounds', '3', '--data-dir', "$projectPath/data", '--source-dir', $covtypeSource,
                '--out-dir', $outDir)
    }
}

if ('central' -in $Stages) {
    foreach ($dataSeed in $DataSeeds) {
        $outDir = Join-Path $projectPath "experiments/results/covtype_10000/seed_${dataSeed}_full"
        Invoke-Study -Label "Covtype-10k / seed=$dataSeed / centralized references" `
            -Entry 'centralized/run_clustering_ceiling.py' `
            -CliArgs @('--dataset', 'covtype_10000', '--data-seed', "$dataSeed", '--init-seeds', $InitSeeds,
                '--data-dir', "$projectPath/data", '--select-on', 'acc', '--recheck-max-iter', '1000', '--out-dir', $outDir)
    }
}

if ($DryRun) { Write-Host "`nPlan only; no data, results or training created." }
elseif ($failedTasks.Count) {
    Write-Host "`nIncomplete tasks:"
    $failedTasks | ForEach-Object { Write-Host "  $_" }
    throw "$($failedTasks.Count) task(s) failed. Review the Python errors above and rerun this command."
} else { Write-Host "`nAll requested HE-revision studies completed." }
