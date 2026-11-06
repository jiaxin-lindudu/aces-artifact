# Run the remaining fixed-parameter studies using the existing Python entries.
param(
    [string]$PythonPath = (Get-Command python -CommandType Application).Source
)

$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectPath
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

$datasets = @(
    @{ Name = 'digits'; Clients = 20; Source = 'federated_{0}_heterogeneity' },
    @{ Name = 'usps'; Clients = 50; Source = 'federated_{0}_clients50_heterogeneity' },
    @{ Name = 'coil20'; Clients = 20; Source = 'federated_{0}_ours' },
    @{ Name = 'mnist_10000'; Clients = 20; Source = 'federated_{0}_ours' }
)

# Shared by the study loops below: one CLI invocation, one output-completion check.
function Invoke-Experiment {
    param(
        [hashtable]$Dataset,
        [int]$DataSeed,
        [string]$Script,
        [string]$Stem,
        [string[]]$ExtraArgs,
        [string]$Beta = '0.5',
        [string]$ResultGroup = 'fixed_best'
    )

    $datasetName = $Dataset.Name
    $clientCount = $Dataset.Clients
    $sourceName = $Dataset.Source -f $DataSeed
    $sourceDir = Join-Path $projectPath "experiments/results/$datasetName/$sourceName"
    $outputDir = Join-Path $projectPath "experiments/results/$datasetName/federated_${DataSeed}_clients${clientCount}_${ResultGroup}"
    $rawFile = Join-Path $outputDir "${Stem}.csv"
    $meanFile = Join-Path $outputDir "${Stem}_mean.csv"
    if ((Test-Path -LiteralPath $rawFile) -and (Test-Path -LiteralPath $meanFile)) {
        Write-Host "跳过已有结果：$datasetName / seed=$DataSeed / $Stem"
        return
    }

    Write-Host "`n开始：$datasetName / seed=$DataSeed / $Stem"
    & $PythonPath "$projectPath/experiments/scripts/$Script" `
        --dataset $datasetName `
        --grid-split $DataSeed `
        --beta $Beta `
        --clients $clientCount `
        --data-dir "$projectPath/data" `
        --init-seeds '40,41,42,43,44' `
        --select-on acc `
        --source-dir $sourceDir `
        --out-dir $outputDir `
        @ExtraArgs

    if ($LASTEXITCODE -ne 0) {
        throw "运行失败：$datasetName / seed=$DataSeed / $Stem"
    }
}

# Prepare only missing Stage 2 assignments. Keep the original frozen FIT/TEST.
& $PythonPath "$projectPath/data/prepare_federated_partitions.py" `
    --datasets 'coil20,mnist_10000' `
    --data-seeds '0,1,2,3,4' `
    --betas '0.1,0.5,1.0' `
    --clients 20 `
    --input-dir "$projectPath/data" `
    --output-dir "$projectPath/data"
if ($LASTEXITCODE -ne 0) { throw 'COIL20/MNIST Stage 2 preparation failed' }

# 1. Complete the Ours studies; completed first-batch outputs are skipped.
# Add FFCM/FedSC on exactly the same quantity/feature views.
foreach ($dataset in $datasets) {
    $clientCount = $dataset.Clients
    foreach ($dataSeed in 0..4) {
        foreach ($study in @('mechanism', 'participation', 'privacy', 'space', 'anchors', 'quantity', 'feature', 'budget')) {
            $extraArgs = @('--study', $study)
            if ($study -in @('mechanism', 'space', 'anchors')) {
                $script = 'ablate_ours_best.py'
                $tag = if ($study -eq 'mechanism') { 'ablation' } else { $study }
                $stem = "grid_ours_${tag}_beta0.5_clients${clientCount}"
            } elseif ($study -in @('participation', 'privacy')) {
                $script = 'run_ours_privacy_participation.py'
                $stem = "grid_ours_${study}_beta0.5_clients${clientCount}"
            } elseif ($study -in @('quantity', 'feature')) {
                $script = 'run_ours_heterogeneity.py'
                $stem = "grid_ours_${study}_sourcebeta0.5_clients${clientCount}"
            } else {
                $script = 'run_fastlloyd_budget.py'
                $stem = "grid_fastlloyd_budget_beta0.5_clients${clientCount}"
                $extraArgs = @()
            }
            Invoke-Experiment -Dataset $dataset -DataSeed $dataSeed `
                -Script $script -Stem $stem -ExtraArgs $extraArgs
        }
        foreach ($method in @('ffcm', 'fedsc')) {
            foreach ($study in @('quantity', 'feature')) {
                Invoke-Experiment -Dataset $dataset -DataSeed $dataSeed `
                    -Script 'run_ours_heterogeneity.py' `
                    -Stem "grid_${method}_${study}_sourcebeta0.5_clients${clientCount}" `
                    -ExtraArgs @('--method', $method, '--study', $study)
            }
        }
    }
}

# 2. Missing COIL20/MNIST label-skew comparisons: transfer beta=.5 best configs.
# These are fixed-configuration robustness runs, not new grid-search winners.
foreach ($dataset in $datasets) {
    if ($dataset.Name -notin @('coil20', 'mnist_10000')) { continue }
    $clientCount = $dataset.Clients
    foreach ($dataSeed in 0..4) {
        foreach ($betaValue in @('0.1', '1.0')) {
            foreach ($method in @('ours', 'ffcm', 'nnfc', 'kfed', 'fedsc', 'fastlloyd', 'fkm')) {
                Invoke-Experiment -Dataset $dataset -DataSeed $dataSeed `
                    -Script 'run_federated_best.py' `
                    -Stem "grid_${method}_fixed_beta${betaValue}_clients${clientCount}" `
                    -ExtraArgs @('--method', $method, '--study', 'partition', '--reference-beta', '0.5') `
                    -Beta $betaValue -ResultGroup 'fixed_transfer'
            }
        }
    }
}

# 3. Native partial participation for FKM/FedSC; no invented one-shot rounds.
foreach ($dataset in $datasets) {
    $clientCount = $dataset.Clients
    foreach ($dataSeed in 0..4) {
        foreach ($method in @('fedsc', 'fkm')) {
            Invoke-Experiment -Dataset $dataset -DataSeed $dataSeed `
                -Script 'run_federated_best.py' `
                -Stem "grid_${method}_participation_beta0.5_clients${clientCount}" `
                -ExtraArgs @('--method', $method, '--study', 'participation')
        }
    }
}

# 4. Run the larger CKKS scale studies last.
foreach ($dataset in $datasets) {
    $clientCount = $dataset.Clients
    foreach ($dataSeed in 0..4) {
        foreach ($study in @('samples', 'clients', 'clusters')) {
            Invoke-Experiment -Dataset $dataset -DataSeed $dataSeed `
                -Script 'run_ours_scale.py' `
                -Stem "grid_ours_scale_${study}_beta0.5_clients${clientCount}" `
                -ExtraArgs @('--study', $study)
        }
    }
}

Write-Host "`n所有已配置的挂机实验完成。"
