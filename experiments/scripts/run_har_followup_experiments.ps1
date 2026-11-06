# Fixed-best HAR follow-up studies on the existing frozen FIT/TEST splits.
# ASCII source for Windows PowerShell 5.1.
param(
    [string]$PythonPath = (Get-Command python -CommandType Application).Source,
    [int[]]$DataSeeds = @(0, 1, 2, 3, 4),
    [string]$InitSeeds = '40,41,42,43,44',
    [int]$Clients = 50,
    [ValidateSet('ablation', 'participation', 'cost', 'privacy', 'heterogeneity',
        'label-skew', 'clients', 'budget', 'roles', 'solver', 'large-silo', 'scale')]
    [string[]]$Stages = @('ablation', 'participation', 'cost', 'privacy', 'heterogeneity',
        'label-skew', 'clients', 'budget', 'roles', 'solver', 'large-silo', 'scale'),
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
$seeds = @($InitSeeds.Split(',') | ForEach-Object { [int]$_.Trim() })
if ($seeds.Count -ne 5 -or @($seeds | Select-Object -Unique).Count -ne $seeds.Count) {
    throw 'Use five distinct initialization seeds.'
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
    if ($LASTEXITCODE -ne 0) { throw "HAR follow-up failed: $Label" }
}

$methods = @('ours', 'ffcm', 'fedsc', 'kfed', 'fkm', 'nnfc')
foreach ($stage in $Stages) {
    if ($stage -eq 'label-skew') {
        Invoke-Study -Label "HAR / frozen FIT label-skew partitions / clients=$Clients" `
            -Entry 'data/prepare_federated_partitions.py' `
            -CliArgs @('--datasets', 'har', '--data-seeds', ($DataSeeds -join ','),
                '--betas', '0.1,1.0', '--clients', "$Clients",
                '--input-dir', "$projectPath/data", '--output-dir', "$projectPath/data")
    } elseif ($stage -eq 'clients') {
        foreach ($targetClients in @(10, 20, 30, 50)) {
            if ($targetClients -eq $Clients) { continue }
            Invoke-Study -Label "HAR / frozen FIT client-count partition / clients=$targetClients" `
                -Entry 'data/prepare_federated_partitions.py' `
                -CliArgs @('--datasets', 'har', '--data-seeds', ($DataSeeds -join ','),
                    '--betas', '0.5', '--clients', "$targetClients",
                    '--input-dir', "$projectPath/data", '--output-dir', "$projectPath/data")
        }
    }
    foreach ($dataSeed in $DataSeeds) {
        $oursSource = Join-Path $projectPath "experiments/results/har/federated_${dataSeed}_clients${Clients}_ours_tpe"
        $referenceSource = Join-Path $projectPath "experiments/results/har/federated_${dataSeed}_clients${Clients}_reference"
        $outDir = Join-Path $projectPath "experiments/results/har/federated_${dataSeed}_clients${Clients}_fixed_best"
        $identityOptions = @('--dataset', 'har', '--grid-split', "$dataSeed",
            '--init-seeds', $InitSeeds, '--data-dir', "$projectPath/data")
        $options = $identityOptions + @('--beta', '0.5', '--clients', "$Clients",
            '--select-on', 'acc', '--source-dir', $oursSource)
        if ($stage -eq 'ablation') {
            foreach ($study in @('mechanism', 'space', 'anchors')) {
                $studyOptions = @('--study', $study, '--out-dir', $outDir, '--resume')
                if ($study -eq 'mechanism') { $studyOptions += '--include-normalization' }
                Invoke-Study -Label "HAR / seed=$dataSeed / fixed-best $study ablations" `
                    -Entry 'experiments/scripts/ablate_ours_best.py' `
                    -CliArgs ($options + $studyOptions)
            }
        } elseif ($stage -eq 'participation') {
            Invoke-Study -Label "HAR / seed=$dataSeed / fixed-best partial participation" `
                -Entry 'experiments/scripts/run_ours_privacy_participation.py' `
                -CliArgs ($options + @('--study', 'participation', '--out-dir', $outDir, '--reuse-full', '--resume'))
            foreach ($method in @('fkm', 'fedsc')) {
                Invoke-Study -Label "HAR / seed=$dataSeed / $method native partial participation" `
                    -Entry 'experiments/scripts/run_federated_best.py' `
                    -CliArgs ($identityOptions + @('--beta', '0.5', '--clients', "$Clients",
                        '--reference-beta', '0.5', '--reference-clients', "$Clients",
                        '--method', $method, '--study', 'participation', '--select-on', 'acc',
                        '--source-dir', $referenceSource, '--out-dir', $outDir, '--resume'))
            }
        } elseif ($stage -eq 'cost') {
            $outDir = Join-Path $projectPath "experiments/results/har/federated_${dataSeed}_clients${Clients}_ckks_cost"
            Invoke-Study -Label "HAR / seed=$dataSeed / Ours and FFCM, plaintext and CKKS" `
                -Entry 'experiments/scripts/run_ckks_cost_comparison.py' `
                -CliArgs ($options + @('--ffcm-source-dir', $referenceSource, '--out-dir', $outDir))
        } elseif ($stage -eq 'privacy') {
            Invoke-Study -Label "HAR / seed=$dataSeed / plaintext-CKKS privacy diagnostics" `
                -Entry 'experiments/scripts/run_ours_privacy_participation.py' `
                -CliArgs ($options + @('--study', 'privacy', '--out-dir', $outDir, '--resume'))
            Invoke-Study -Label "HAR / seed=$dataSeed / matched plaintext-CKKS privacy control" `
                -Entry 'experiments/scripts/run_ours_privacy_control.py' `
                -CliArgs ($identityOptions + @('--beta', '0.5', '--clients', "$Clients",
                    '--source-dir', $outDir, '--out-dir', $outDir, '--resume'))
        } elseif ($stage -eq 'heterogeneity') {
            $outDir = Join-Path $projectPath "experiments/results/har/federated_${dataSeed}_clients${Clients}_heterogeneity"
            foreach ($method in @('ours', 'ffcm', 'fedsc')) {
                $source = if ($method -eq 'ours') { $oursSource } else { $referenceSource }
                foreach ($study in @('quantity', 'feature')) {
                    Invoke-Study -Label "HAR / seed=$dataSeed / $method fixed-best $study heterogeneity" `
                        -Entry 'experiments/scripts/run_ours_heterogeneity.py' `
                        -CliArgs ($identityOptions + @('--beta', '0.5', '--clients', "$Clients",
                            '--method', $method, '--study', $study, '--select-on', 'acc',
                            '--source-dir', $source, '--out-dir', $outDir, '--resume'))
                }
            }
        } elseif ($stage -eq 'label-skew' -or $stage -eq 'clients') {
            $targetBetas = if ($stage -eq 'label-skew') { @('0.1', '1.0') } else { @('0.5') }
            $clientCounts = if ($stage -eq 'clients') { @(10, 20, 30, 50) } else { @($Clients) }
            foreach ($targetBeta in $targetBetas) {
                foreach ($targetClients in $clientCounts) {
                    $outDir = Join-Path $projectPath "experiments/results/har/federated_${dataSeed}_clients${targetClients}_fixed_transfer"
                    foreach ($method in $methods) {
                        $source = if ($method -eq 'ours') { $oursSource } else { $referenceSource }
                        Invoke-Study -Label "HAR / seed=$dataSeed / $method fixed transfer / beta=$targetBeta, clients=$targetClients" `
                            -Entry 'experiments/scripts/run_federated_best.py' `
                            -CliArgs ($identityOptions + @('--beta', $targetBeta, '--clients', "$targetClients",
                                '--reference-beta', '0.5', '--reference-clients', "$Clients",
                                '--method', $method, '--study', 'partition', '--select-on', 'acc',
                                '--source-dir', $source, '--out-dir', $outDir, '--resume'))
                    }
                }
            }
        } elseif ($stage -eq 'budget') {
            Invoke-Study -Label "HAR / seed=$dataSeed / fixed-best round budgets 0.5T, T, 2T, 4T" `
                -Entry 'experiments/scripts/run_ours_fixed_checks.py' `
                -CliArgs ($options + @('--study', 'budget', '--round-multipliers', '0.5', '1', '2', '4',
                    '--out-dir', $outDir, '--resume'))
        } elseif ($stage -eq 'roles') {
            $outDir = Join-Path $projectPath "experiments/results/har/federated_${dataSeed}_clients${Clients}_he_revision"
            Invoke-Study -Label "HAR / seed=$dataSeed / three-role CKKS transport snapshot" `
                -Entry 'experiments/scripts/run_ckks_role_validation.py' `
                -CliArgs ($options + @('--selection-seeds', $InitSeeds, '--out-dir', $outDir, '--resume'))
        } elseif ($stage -eq 'solver') {
            $outDir = Join-Path $projectPath "experiments/results/har/federated_${dataSeed}_clients${Clients}_he_revision"
            Invoke-Study -Label "HAR / seed=$dataSeed / dense-partial solver CKKS comparison" `
                -Entry 'experiments/scripts/run_ours_solver_comparison.py' `
                -CliArgs ($options + @('--study', 'solver', '--aggregation', 'ckks', '--out-dir', $outDir))
        } elseif ($stage -eq 'large-silo') {
            $outDir = Join-Path $projectPath "experiments/results/har/federated_${dataSeed}_clients${Clients}_large_silo"
            Invoke-Study -Label "HAR / seed=$dataSeed / CKKS cost, 3 silos with nested unique FIT prefixes" `
                -Entry 'experiments/scripts/run_ours_solver_comparison.py' `
                -CliArgs ($options + @('--study', 'large-silo', '--aggregation', 'ckks',
                    '--ffcm-source-dir', $referenceSource, '--view-clients', '3',
                    '--local-samples', '128', '512', '1024', '2048', '--benchmark-rounds', '3',
                    '--out-dir', $outDir))
        } elseif ($stage -eq 'scale') {
            $outDir = Join-Path $projectPath "experiments/results/har/federated_${dataSeed}_clients${Clients}_scale"
            foreach ($study in @('samples', 'clients', 'clusters')) {
                $label = if ($study -eq 'clients') {
                    'CKKS fixed-local-N cost views; total FIT grows with client count'
                } else { "CKKS $study cost views" }
                Invoke-Study -Label "HAR / seed=$dataSeed / $label" `
                    -Entry 'experiments/scripts/run_ours_scale.py' `
                    -CliArgs ($options + @('--study', $study, '--out-dir', $outDir, '--resume'))
            }
        }
    }
}

if ($DryRun) { Write-Host "`nPlan only; no data, models or result files were created." }
else { Write-Host "`nThis HAR follow-up batch completed for the requested stages." }
