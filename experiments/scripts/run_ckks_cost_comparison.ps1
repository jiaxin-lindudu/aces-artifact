# Four fixed-best conditions; resume completed condition/seed rows in raw CSV.
param(
    [string]$PythonPath = (Get-Command python -CommandType Application).Source,
    [int[]]$DataSeeds = @(0, 1, 2, 3, 4),
    [switch]$DryRun
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
    @{ Name = 'usps'; Clients = 50; Source = 'federated_{0}_clients50_heterogeneity' }
)
foreach ($dataset in $datasets) {
    $datasetName = $dataset.Name
    $clientCount = $dataset.Clients
    foreach ($dataSeed in $DataSeeds) {
        $sourceName = $dataset.Source -f $dataSeed
        $sourceDir = Join-Path $projectPath "experiments/results/$datasetName/$sourceName"
        $outputDir = Join-Path $projectPath "experiments/results/$datasetName/federated_${dataSeed}_clients${clientCount}_ckks_cost"
        $extra = @()
        if ($DryRun) { $extra = @('--dry-run') }
        Write-Host "`nStarting: $datasetName / CKKS cost / data_seed=$dataSeed"
        & $PythonPath "$projectPath/experiments/scripts/run_ckks_cost_comparison.py" `
            --dataset $datasetName `
            --grid-split $dataSeed `
            --beta 0.5 `
            --clients $clientCount `
            --init-seeds '40,41,42,43,44' `
            --select-on acc `
            --data-dir "$projectPath/data" `
            --source-dir $sourceDir `
            --out-dir $outputDir `
            @extra
        if ($LASTEXITCODE -ne 0) {
            throw "CKKS cost comparison failed: $datasetName / data_seed=$dataSeed. See the Python error above."
        }
    }
}
Write-Host "`nCKKS cost comparison completed."
