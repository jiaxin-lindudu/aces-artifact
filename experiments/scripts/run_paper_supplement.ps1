# Fixed-parameter paper supplements. ASCII for Windows PowerShell 5.1.
param(
    [string]$PythonPath = (Get-Command python -CommandType Application).Source,
    [int[]]$DataSeeds = @(0, 1, 2, 3, 4),
    [ValidateSet('digits', 'usps', 'coil20', 'mnist_10000')]
    [string[]]$Datasets = @('digits', 'usps', 'coil20', 'mnist_10000'),
    [ValidateSet('ours', 'ffcm', 'fedsc', 'kfed', 'fkm', 'nnfc', 'fastlloyd')]
    [string[]]$Methods = @('ours', 'ffcm', 'fedsc', 'kfed', 'fkm', 'nnfc', 'fastlloyd'),
    [ValidateSet('cost', 'normalization', 'budget', 'clients')]
    [string[]]$Stages = @('cost', 'normalization', 'budget', 'clients'),
    [ValidateSet(10, 20, 30, 50)]
    [int[]]$ClientCounts = @(10, 20, 30, 50),
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectPath
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$failedTasks = [System.Collections.Generic.List[string]]::new()
$script:plannedTasks = 0

# One shared invocation path for independent, resumable study entries.
function Invoke-Study {
    param([string]$Label, [string]$Entry, [string[]]$CliArgs)
    $script:plannedTasks++
    if ($DryRun) {
        Write-Host "PLAN: $Label"
        return
    }
    Write-Host "`nSTART: $Label"
    & $PythonPath "$projectPath/$Entry" @CliArgs
    if ($LASTEXITCODE -ne 0) {
        $failedTasks.Add($Label)
        Write-Warning "FAILED: $Label. Completed rows are preserved; continuing independent tasks."
    }
}

# Preflight existing reference inputs before starting a long batch.
foreach ($datasetName in $Datasets) {
    $referenceClients = if ($datasetName -eq 'usps') { 50 } else { 20 }
    foreach ($dataSeed in $DataSeeds) {
        if ($dataSeed -notin 0..4) { throw 'DataSeeds must be in 0..4.' }
        $sourceName = switch ($datasetName) {
            'digits' { "federated_${dataSeed}_heterogeneity" }
            'usps' { "federated_${dataSeed}_clients50_heterogeneity" }
            default { "federated_${dataSeed}_ours" }
        }
        $sourceDir = Join-Path $projectPath "experiments/results/$datasetName/$sourceName"
        $needed = if ('clients' -in $Stages) { @($Methods) } else { @() }
        if ('cost' -in $Stages) { $needed += @('ours', 'ffcm') }
        if (('normalization' -in $Stages) -or ('budget' -in $Stages)) { $needed += @('ours') }
        foreach ($method in ($needed | Select-Object -Unique)) {
            $stem = "grid_${method}_beta0.5_clients${referenceClients}"
            foreach ($suffix in @('.csv', '_mean.csv')) {
                if (-not (Test-Path -LiteralPath (Join-Path $sourceDir "$stem$suffix"))) {
                    throw "Missing reference CSV: $sourceDir/$stem$suffix"
                }
            }
        }
    }
}

# Only Stage 2 is extended: Stage 1 FIT/TEST and preprocessing stay frozen.
if ('clients' -in $Stages) {
    foreach ($clientCount in $ClientCounts) {
        Invoke-Study -Label "prepare frozen M=$clientCount" `
            -Entry 'data/prepare_federated_partitions.py' `
            -CliArgs @('--datasets', ($Datasets -join ','), '--data-seeds', ($DataSeeds -join ','),
                '--betas', '0.5', '--clients', "$clientCount", '--input-dir', "$projectPath/data",
                '--output-dir', "$projectPath/data", '--quiet')
    }
    if ($failedTasks.Count) {
        throw 'Stage 2 preparation failed. See errors above; no training started.'
    }
}

foreach ($stage in @('cost', 'normalization', 'budget', 'clients')) {
    if ($stage -notin $Stages) { continue }
    foreach ($datasetName in $Datasets) {
        if (($stage -eq 'budget') -and ($datasetName -notin @('usps', 'mnist_10000'))) { continue }
        $referenceClients = if ($datasetName -eq 'usps') { 50 } else { 20 }
        foreach ($dataSeed in $DataSeeds) {
            $sourceName = switch ($datasetName) {
                'digits' { "federated_${dataSeed}_heterogeneity" }
                'usps' { "federated_${dataSeed}_clients50_heterogeneity" }
                default { "federated_${dataSeed}_ours" }
            }
            $sourceDir = Join-Path $projectPath "experiments/results/$datasetName/$sourceName"
            $common = @('--dataset', $datasetName, '--grid-split', "$dataSeed", '--beta', '0.5',
                '--init-seeds', '40,41,42,43,44', '--select-on', 'acc',
                '--data-dir', "$projectPath/data", '--source-dir', $sourceDir)
            if ($stage -eq 'cost') {
                $outDir = Join-Path $projectPath "experiments/results/$datasetName/federated_${dataSeed}_clients${referenceClients}_ckks_cost"
                Invoke-Study -Label "$datasetName / seed=$dataSeed / four-arm cost" `
                    -Entry 'experiments/scripts/run_ckks_cost_comparison.py' `
                    -CliArgs ($common + @('--clients', "$referenceClients", '--out-dir', $outDir))
            } elseif ($stage -in @('normalization', 'budget')) {
                $outDir = Join-Path $projectPath "experiments/results/$datasetName/federated_${dataSeed}_clients${referenceClients}_fixed_checks"
                $options = @('--study', $stage, '--clients', "$referenceClients", '--out-dir', $outDir, '--resume')
                if ($stage -eq 'budget') {
                    if ($datasetName -eq 'mnist_10000') { $options += @('--rounds', '10', '20', '40') }
                    else { $options += @('--round-multipliers', '0.5', '1', '2', '4') }
                }
                Invoke-Study -Label "$datasetName / seed=$dataSeed / $stage" `
                    -Entry 'experiments/scripts/run_ours_fixed_checks.py' -CliArgs ($common + $options)
            } else {
                foreach ($clientCount in $ClientCounts) {
                    $outDir = Join-Path $projectPath "experiments/results/$datasetName/federated_${dataSeed}_clients${clientCount}_fixed_clients"
                    foreach ($method in $Methods) {
                        Invoke-Study -Label "$datasetName / seed=$dataSeed / M=$clientCount / $method" `
                            -Entry 'experiments/scripts/run_federated_best.py' `
                            -CliArgs ($common + @('--method', $method, '--study', 'partition',
                                '--reference-beta', '0.5', '--reference-clients', "$referenceClients",
                                '--clients', "$clientCount", '--out-dir', $outDir, '--resume'))
                    }
                }
            }
        }
    }
}
Write-Host "`nTasks scheduled: $script:plannedTasks"
if ($failedTasks.Count) {
    Write-Host 'Failed tasks (rerun the same command to resume):'
    $failedTasks | ForEach-Object { Write-Host "  $_" }
    throw "$($failedTasks.Count) independent task(s) failed. See the Python errors above."
}
Write-Host 'Paper supplement batch completed.'
