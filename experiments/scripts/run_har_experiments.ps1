# HAR frozen splits, adaptive Ours, fixed-budget baselines and centralized references.
# ASCII source for Windows PowerShell 5.1.
param(
    [string]$PythonPath = (Get-Command python -CommandType Application).Source,
    [int[]]$DataSeeds = @(0, 1, 2, 3, 4),
    [string]$InitSeeds = '40,41,42,43,44',
    [int]$Clients = 50,
    [ValidateSet('prepare', 'federated', 'central')]
    [string[]]$Stages = @('prepare', 'federated', 'central'),
    [ValidateSet('ours', 'ffcm', 'fedsc', 'kfed', 'fkm', 'nnfc')]
    [string[]]$Methods = @('ours', 'ffcm', 'fedsc', 'kfed', 'fkm', 'nnfc'),
    [int]$MaxNewTrials = 8,
    [int]$Patience = 3,
    [double]$MinImprovement = 0.002,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Python not found: $PythonPath" }
if ($Clients -lt 1) { throw 'Clients must be positive.' }
if (-not $DataSeeds.Count -or @($DataSeeds | Where-Object { $_ -notin 0..4 }).Count -or
    @($DataSeeds | Select-Object -Unique).Count -ne $DataSeeds.Count) {
    throw 'Use distinct DataSeeds from 0..4.'
}
$expectedInitSeeds = @($InitSeeds.Split(',') | ForEach-Object { [int]$_.Trim() } | Sort-Object)
if ($expectedInitSeeds.Count -ne 5 -or @($expectedInitSeeds | Select-Object -Unique).Count -ne $expectedInitSeeds.Count) {
    throw 'Use five distinct initialization seeds.'
}
if ($MaxNewTrials -lt 2 -or $Patience -lt 1 -or $MinImprovement -lt 0) {
    throw 'MaxNewTrials must be at least 2, Patience positive, and MinImprovement nonnegative.'
}
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

function Invoke-Study {
    param([string]$Label, [string]$Entry, [string[]]$CliArgs)
    if ($DryRun) {
        Write-Host "PLAN: $Label"
        Write-Host "$PythonPath $projectPath/$Entry $($CliArgs -join ' ')"
        return
    }
    Write-Host "`nSTART: $Label"
    & $PythonPath "$projectPath/$Entry" @CliArgs
    if ($LASTEXITCODE -ne 0) { throw "HAR experiment failed: $Label" }
}

if ('prepare' -in $Stages) {
    Invoke-Study -Label 'HAR / frozen Stage 1 / FIT 8239, TEST 2060, PCA 64' `
        -Entry 'data/prepare_centralized_splits.py' `
        -CliArgs @('--datasets', 'har', '--data-seeds', ($DataSeeds -join ','),
            '--apply-pca', '--output-dir', "$projectPath/data")
    Invoke-Study -Label "HAR / frozen Stage 2 / beta=0.5, clients=$Clients" `
        -Entry 'data/prepare_federated_partitions.py' `
        -CliArgs @('--datasets', 'har', '--data-seeds', ($DataSeeds -join ','),
            '--betas', '0.5', '--clients', "$Clients",
            '--input-dir', "$projectPath/data", '--output-dir', "$projectPath/data")
}

if ('federated' -in $Stages) {
    $baselineMethods = @($Methods | Where-Object { $_ -ne 'ours' })
    foreach ($dataSeed in $DataSeeds) {
        if ('ours' -in $Methods) {
            $outDir = Join-Path $projectPath "experiments/results/har/federated_${dataSeed}_clients${Clients}_ours_tpe"
            Invoke-Study -Label "HAR / seed=$dataSeed / Ours adaptive FIT-only TPE" `
                -Entry 'experiments/scripts/tune_covtype_ours.py' `
                -CliArgs @('--dataset', 'har', '--grid-split', "$dataSeed", '--clients', "$Clients",
                    '--beta', '0.5', '--init-seeds', $InitSeeds, '--select-on', 'acc',
                    '--eigen-solver', 'dense', '--max-new-trials', "$MaxNewTrials",
                    '--patience', "$Patience", '--min-improvement', "$MinImprovement",
                    '--data-dir', "$projectPath/data", '--out-dir', $outDir, '--resume')
        }
        if ($baselineMethods.Count) {
            $outDir = Join-Path $projectPath "experiments/results/har/federated_${dataSeed}_clients${Clients}_reference"
            Invoke-Study -Label "HAR / seed=$dataSeed / fixed candidates: $($baselineMethods -join ',')" `
                -Entry 'experiments/scripts/run_covtype_reference.py' `
                -CliArgs @('--dataset', 'har', '--grid-split', "$dataSeed", '--clients', "$Clients",
                    '--beta', '0.5', '--init-seeds', $InitSeeds, '--select-on', 'acc',
                    '--methods', ($baselineMethods -join ','), '--eigen-solver', 'dense',
                    '--data-dir', "$projectPath/data", '--out-dir', $outDir, '--resume')
        }
    }
}

if ('central' -in $Stages) {
    foreach ($dataSeed in $DataSeeds) {
        $outDir = Join-Path $projectPath "experiments/results/har/seed_${dataSeed}_full"
        $ceilingPath = Join-Path $outDir "centralized_ceiling_har_seed${dataSeed}.csv"
        $metadataPath = Join-Path $outDir "centralized_run_metadata_har_seed${dataSeed}.json"
        if (-not $DryRun -and ((Test-Path -LiteralPath $ceilingPath) -or (Test-Path -LiteralPath $metadataPath))) {
            if (-not ((Test-Path -LiteralPath $ceilingPath) -and (Test-Path -LiteralPath $metadataPath))) {
                throw "Incomplete centralized output: $outDir. Review it before rerunning."
            }
            $rows = @(Import-Csv -LiteralPath $ceilingPath)
            $metadata = Get-Content -LiteralPath $metadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $centralMethods = @('central_kmeans', 'central_fcm', 'central_spectral')
            $complete = $rows.Count -eq (3 * $expectedInitSeeds.Count) -and
                $metadata.dataset -eq 'har' -and $metadata.data_seed -eq $dataSeed -and
                $metadata.select_on -eq 'acc' -and $metadata.budget_mode -eq 'auto' -and
                ($metadata.budgets -join ',') -eq '1,2,5,10,30,100,300,1000' -and
                (($metadata.init_seeds | Sort-Object) -join ',') -eq ($expectedInitSeeds -join ',')
            foreach ($method in $centralMethods) {
                $part = @($rows | Where-Object { $_.method -eq $method })
                $complete = $complete -and $part.Count -eq $expectedInitSeeds.Count -and
                    (($part | ForEach-Object { [int]$_.init_seed } | Sort-Object) -join ',') -eq ($expectedInitSeeds -join ',')
            }
            $complete = $complete -and -not @($rows | Where-Object {
                $_.dataset -ne 'har' -or [int]$_.data_seed -ne $dataSeed
            }).Count
            if (-not $complete) {
                throw "Centralized output differs from the requested HAR protocol: $outDir. Existing results were preserved."
            }
            Write-Host "SKIP: HAR / seed=$dataSeed / complete centralized references"
            continue
        }
        Invoke-Study -Label "HAR / seed=$dataSeed / centralized K-means, FCM, Spectral" `
            -Entry 'centralized/run_clustering_ceiling.py' `
            -CliArgs @('--dataset', 'har', '--data-seed', "$dataSeed", '--init-seeds', $InitSeeds,
                '--data-dir', "$projectPath/data", '--select-on', 'acc',
                '--recheck-max-iter', '1000', '--out-dir', $outDir)
    }
}

if ($DryRun) { Write-Host "`nPlan only; no downloads, data files, results or training created." }
else { Write-Host "`nRequested HAR experiments completed." }
