# ACES reproducibility artifact

This artifact accompanies *ACES: Shared Anchors Make Spectral Federated Clustering Encryptable with Fixed-Length Cluster Statistics*. It includes the ACES implementation, baseline implementations, experiment entry points, fixed-configuration selection inputs, paper result CSVs, and plotting scripts.

## Setup

The reference environment is Windows x64 with Python 3.10. Create and activate a clean virtual environment, then run these commands from the repository root:

```powershell
python -m pip install -r requirements.txt
python -B experiments/scripts/grid_search_ours.py --dataset digits --dry-run
```

The requirements install CPU PyTorch. Timing and peak memory depend on the machine and installed numerical libraries.

## Data preparation

The scripts download public benchmark inputs as needed. Digits is supplied by scikit-learn, USPS uses OpenML dataset 41082, COIL20 uses OpenML coil-20 version 2 with a MAT fallback, HAR uses OpenML dataset 1478, and the natural-subject HAR study uses the official UCI archive. The MNIST-10k loader selects a fixed stratified subset from the combined original MNIST train and test sets, then creates the experiment's own splits. Raw data are available from their upstream sources under their respective terms.

```powershell
python data/prepare_centralized_splits.py --datasets digits,usps,coil20,mnist_10000,har --data-seeds 0,1,2,3,4 --apply-pca
python data/prepare_federated_partitions.py --datasets digits,coil20,mnist_10000 --data-seeds 0,1,2,3,4 --betas 0.5 --clients 20
python data/prepare_federated_partitions.py --datasets usps,har --data-seeds 0,1,2,3,4 --betas 0.5 --clients 50
python data/prepare_har_subject_splits.py
```

The first stage freezes FIT/TEST and fits preprocessing on FIT. The second stage assigns only FIT records to clients. The natural-subject study has 24 FIT clients per fold and uses subject ownership. See [data preparation](data/README.md) for split details and [experiment commands](experiments/scripts/README.md) for additional partitions, fixed-configuration runs, and ablations.

## Reproduce results

The original model selection uses initialization seeds 40–44. The paired validation uses seeds 100–109 and the selected per-split configurations from the included reference CSVs. The stored measurements can be inspected without training.

| Study | Entry point or output |
| --- | --- |
| ACES and federated baselines | `experiments/scripts/grid_search_*.py` and the included reference CSVs |
| Centralized references | `centralized/run_clustering_ceiling.py` |
| Subject-owned HAR | `experiments/scripts/run_har_subject_reference.py` after preparing the natural folds |
| Mechanism, normalization, heterogeneity, participation | `ablate_ours_best.py`, `run_ours_fixed_checks.py`, `run_ours_heterogeneity.py`, and `run_ours_privacy_participation.py` in `experiments/scripts/` |
| Encryption cost, roles, and privacy probes | `run_ckks_cost_comparison.py`, `run_ours_process_training.py`, `run_ckks_role_validation.py`, and `run_ours_privacy_control.py` in `experiments/scripts/` |
| Paired validation and Holm-adjusted tests | `experiments/scripts/run_paired_wilcoxon.ps1` and `experiments/results/paired_wilcoxon.csv` |

To check the 1,500-run plan and regenerate the statistical summary from the supplied runs, first prepare the benchmark splits above:

```powershell
$pythonPath = (Get-Command python -CommandType Application).Source
& ./experiments/scripts/run_paired_wilcoxon.ps1 -PythonPath $pythonPath -DryRun
python -B experiments/scripts/run_paired_wilcoxon.py --summarize-only
```

The batch script resumes from existing paired runs. To rerun one fit independently, supply a fresh output directory:

```powershell
python -B experiments/scripts/run_paired_wilcoxon.py --dataset digits --method ours --grid-split 0 --clients 20 --init-seeds 100 --out-dir reproduction/digits
```

Paper plots can be regenerated directly from the supplied CSVs:

```powershell
python -B experiments/figures/plot_paper_results.py
python -B experiments/figures/plot_paper_supplement.py
```

Figures are written to `paper/icde2027/figures/`. The baseline implementations contain attribution to their upstream publications and reference implementations.
