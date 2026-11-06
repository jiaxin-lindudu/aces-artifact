"""Publication figures from the completed fixed-parameter supplement CSVs.

No training or intermediate data files. Means average initializations within
each split; uncertainty is the sample SD of the five split means.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'experiments/results'
OUTPUT = ROOT / 'paper/icde2027/figures'
DATASETS = ('digits', 'usps', 'coil20', 'mnist_10000', 'har')
NAMES = ('Digits', 'USPS', 'COIL20', 'MNIST-10k', 'HAR')
REFERENCE_CLIENTS = dict(digits=20, usps=50, coil20=20, mnist_10000=20, har=50)
METHODS = {
    'ours': ('ACES', '#235C7E', 'o'),
    'ffcm': ('FFCM', '#279989', 's'),
    'fedsc': ('FedSC', '#CB7846', '^'),
    'kfed': ('k-FED', '#8C6BA5', 'D'),
}
SOURCES = []


def read_split(dataset, seed, suffix, stem, clients=None):
    clients = REFERENCE_CLIENTS[dataset] if clients is None else clients
    if dataset == 'har' and suffix == 'fixed_clients':
        suffix = 'fixed_transfer'
    if dataset == 'har' and suffix == 'fixed_checks':
        suffix = 'fixed_best'
    path = RESULTS / dataset / f'federated_{seed}_clients{clients}_{suffix}' / (stem + '.csv')
    block = pd.read_csv(path)
    recorded_clients = block['source_clients'] if 'source_clients' in block else block['clients']
    if (not block.dataset.eq(dataset).all() or not block.data_seed.eq(seed).all()
            or not recorded_clients.eq(clients).all() or not block.status.eq('completed').all()
            or not np.isfinite(block[['acc', 'test_acc']].to_numpy(float)).all()):
        raise ValueError(f'Incomplete or mismatched source: {path}')
    SOURCES.append(path)
    return block


def finish(fig, name, files):
    for ax in fig.axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2.5, width=.65, pad=3)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E5E9EE', linewidth=.6)
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(ROOT)).encode('utf-8'))
        digest.update(path.read_bytes())
    fig.savefig(OUTPUT / name, facecolor='white', metadata={
        'Subject': 'Source CSV SHA256=' + digest.hexdigest(),
        'Creator': 'ACES / Matplotlib', 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    print(name)


def plot_privacy_execution_cost():
    """Single-column dense-CKKS wall time, paired on frozen split and seed."""
    first = len(SOURCES)
    with plt.rc_context({
            'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': 8.2, 'axes.labelsize': 8.2, 'xtick.labelsize': 8,
            'ytick.labelsize': 8.2, 'axes.linewidth': .6, 'pdf.fonttype': 42,
            'text.color': '#253343', 'axes.labelcolor': '#253343',
            'xtick.color': '#536171', 'ytick.color': '#536171'}):
        fig, ax = plt.subplots(figsize=(3.45, 1.72))
        for position, dataset in enumerate(DATASETS):
            clients = REFERENCE_CLIENTS[dataset]
            raw = pd.concat([read_split(dataset, seed, 'full_roles',
                f'grid_ours_full_roles_beta0.5_clients{clients}') for seed in range(5)])
            native = raw[raw.condition.eq('native_dense')].set_index(
                ['data_seed', 'init_seed']).sort_index()
            process = raw[raw.condition.eq('process_dense')].set_index(
                ['data_seed', 'init_seed']).sort_index()
            expected = {(split, seed) for split in range(5) for seed in range(40, 45)}
            if (len(native) != 25 or len(process) != 25
                    or set(native.index) != expected or not native.index.equals(process.index)
                    or not np.isfinite(np.r_[native.wall_seconds, process.wall_seconds]).all()):
                raise ValueError('Incomplete dense-CKKS execution comparison: ' + dataset)
            split = [frame.wall_seconds.groupby('data_seed').mean()
                     for frame in (native, process)]
            ax.plot([value.mean() for value in split],
                    [position - .10, position + .10], color='#B8C2CB', lw=.85, zorder=1)
            for values, offset, label, color, marker, face in (
                    (split[0], -.10, 'Native CKKS', '#235C7E', 'o', 'white'),
                    (split[1], .10, 'Three-role CKKS', '#CB7846', 's', '#CB7846')):
                ax.errorbar(values.mean(), position + offset, xerr=values.std(ddof=1),
                            fmt=marker, color=color, markerfacecolor=face,
                            markersize=3.5, markeredgewidth=.9, capsize=1.8,
                            elinewidth=.85, label=label if position == 0 else None, zorder=3)
        ax.set_yticks(range(5), NAMES)
        ax.set_ylim(4.45, -.45)
        ax.set_xlim(0, 45)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_xlabel('End-to-end wall time (s)', labelpad=3)
        ax.grid(axis='x', color='#E5E9EE', linewidth=.55)
        fig.legend(*ax.get_legend_handles_labels(), ncol=2, frameon=False,
                   loc='upper center', bbox_to_anchor=(.57, 1.01),
                   handlelength=1.0, columnspacing=1.0, handletextpad=.4, fontsize=8)
        fig.subplots_adjust(left=.215, right=.98, top=.77, bottom=.28)
        finish(fig, 'privacy_execution_cost.pdf', SOURCES[first:])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
        'axes.titleweight': 'medium', 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'axes.linewidth': .65, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'text.color': '#253343', 'axes.labelcolor': '#253343',
        'xtick.color': '#536171', 'ytick.color': '#536171',
    })

    # Four native budgets are preserved: only within-method protection changes.
    first = len(SOURCES)
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 2.05))
    for dataset, name, ax in zip(DATASETS, NAMES, axes.flat):
        clients = REFERENCE_CLIENTS[dataset]
        raw = pd.concat([read_split(dataset, seed, 'ckks_cost',
            f'grid_ckks_cost_beta0.5_clients{clients}') for seed in range(5)])
        split = raw.groupby(['condition', 'data_seed'])[['training_core_seconds', 'test_acc']].mean()
        stats = split.groupby('condition').agg(['mean', 'std'])
        for method in ('ours', 'ffcm'):
            label, color, marker = METHODS[method]
            pair = stats.loc[[method + '_plain', method + '_ckks']]
            x = pair[('training_core_seconds', 'mean')].to_numpy()
            y = pair[('test_acc', 'mean')].to_numpy() * 100
            ax.plot(x, y, color=color, linewidth=1.7, zorder=2)
            for position, condition in enumerate(('plain', 'ckks')):
                ax.errorbar(x[position], y[position],
                    xerr=pair[('training_core_seconds', 'std')].iloc[position],
                    yerr=pair[('test_acc', 'std')].iloc[position] * 100,
                    fmt=marker, markersize=5.5, color=color,
                    markerfacecolor='white' if condition == 'plain' else color,
                    markeredgewidth=1.2, elinewidth=.85, capsize=2, zorder=3)
        ax.set_title(name, pad=7)
        ax.set_xlabel('Training core (s)')
        ax.margins(x=.16, y=.23)
        ax.set_xlim(left=0)
    axes[0].set_ylabel('TEST ACC (%)')
    legend = [Line2D([], [], color=METHODS[m][1], lw=1.7, label=METHODS[m][0])
              for m in ('ours', 'ffcm')]
    legend += [Line2D([], [], color='#536171', marker='o', lw=0, markersize=5,
                     markerfacecolor=face, label=label)
               for face, label in [('white', 'Plaintext'), ('#536171', 'CKKS')]]
    fig.legend(handles=legend, ncol=4, frameon=False, loc='upper center',
               bbox_to_anchor=(.53, 1.01), handlelength=1.6, columnspacing=1.5)
    fig.subplots_adjust(left=.075, right=.99, top=.71, bottom=.26, wspace=.48)
    finish(fig, 'quality_cost.pdf', SOURCES[first:])

    first = len(SOURCES)
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 2.05))
    client_counts = (10, 20, 30, 50)
    for dataset, name, ax in zip(DATASETS, NAMES, axes.flat):
        for method, (label, color, marker) in METHODS.items():
            records = []
            for clients in client_counts:
                for seed in range(5):
                    records.append(read_split(dataset, seed, 'fixed_clients',
                        f'grid_{method}_fixed_beta0.5_clients{clients}', clients))
            raw = pd.concat(records)
            split = raw.groupby(['clients', 'data_seed']).test_acc.mean().unstack('clients') * 100
            y = split.mean().reindex(client_counts).to_numpy()
            sd = split.std(ddof=1).reindex(client_counts).to_numpy()
            ax.plot(client_counts, y, color=color, marker=marker, markersize=4,
                    linewidth=1.7, label=label)
            ax.fill_between(client_counts, y - sd, y + sd, color=color, alpha=.10, linewidth=0)
        ax.set_title(name, pad=7)
        ax.set_xticks(client_counts)
        ax.margins(x=.05, y=.18)
    axes[0].set_ylabel('TEST ACC (%)')
    fig.supxlabel('Clients (fixed total FIT)', fontsize=9, y=.025)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc='upper center',
               bbox_to_anchor=(.53, 1.01), handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(left=.075, right=.99, top=.71, bottom=.27, wspace=.45)
    finish(fig, 'client_count.pdf', SOURCES[first:])

    first = len(SOURCES)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2), sharey=True)
    deltas = {}
    for dataset in DATASETS:
        clients = REFERENCE_CLIENTS[dataset]
        if dataset == 'har':
            raw = pd.concat([read_split(dataset, seed, 'fixed_best',
                f'grid_ours_ablation_beta0.5_clients{clients}') for seed in range(5)])
            raw = raw.rename(columns={'variant': 'condition'})
        else:
            raw = pd.concat([read_split(dataset, seed, 'fixed_checks',
                f'grid_ours_normalization_beta0.5_clients{clients}') for seed in range(5)])
        pair = raw.pivot(index=['data_seed', 'init_seed'], columns='condition', values=['acc', 'test_acc'])
        deltas[dataset] = (pair.xs('full', axis=1, level=1)
                           - pair.xs('no_normalization', axis=1, level=1)).groupby('data_seed').mean() * 100
    for ax, metric, title in zip(axes, ('acc', 'test_acc'), ('(a) FIT', '(b) TEST')):
        for position, dataset in enumerate(DATASETS):
            values = deltas[dataset][metric].to_numpy()
            ax.scatter(values, position + np.linspace(-.12, .12, 5),
                       s=14, color='#A6B8C2', linewidths=0, zorder=2)
            ax.errorbar(values.mean(), position, xerr=values.std(ddof=1), fmt='D',
                        markersize=4.5, color='#235C7E', capsize=3, elinewidth=1.3, zorder=3)
        ax.axvline(0, color='#8999A6', linewidth=.8, linestyle=(0, (3, 3)), zorder=1)
        ax.set_title(title, pad=7)
        ax.set_xlabel('Normalized - disabled ACC (pp)')
        ax.set_ylim(4.48, -.48)
        values = np.concatenate([part[metric].to_numpy() for part in deltas.values()])
        ax.set_xlim(min(-4.5, values.min() - 1), max(10.6, values.max() + 1))
        ax.grid(axis='x', color='#E5E9EE', linewidth=.6)
    axes[0].set_yticks(range(5), NAMES)
    fig.subplots_adjust(left=.14, right=.985, top=.84, bottom=.28, wspace=.20)
    finish(fig, 'normalization.pdf', SOURCES[first:])

    # Use newly trained longest runs only. The common first 80 USPS rounds
    # include all five splits; one split's additional rounds to 160 are excluded.
    first = len(SOURCES)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.0))
    for column, (dataset, name) in enumerate((('usps', 'USPS'), ('mnist_10000', 'MNIST-10k'), ('har', 'HAR'))):
        clients = REFERENCE_CLIENTS[dataset]
        longest = []
        for seed in range(5):
            raw = read_split(dataset, seed, 'fixed_checks',
                f'grid_ours_budget_check_beta0.5_clients{clients}')
            block = raw[raw.round_budget.eq(raw.round_budget.max())].sort_values('init_seed')
            if len(block) != 5 or not block.result_origin.eq('trained').all():
                raise ValueError('The longest-budget curves require five actual trained histories')
            longest.extend(block.to_dict('records'))
        count = min(int(row['rounds_run']) for row in longest)
        traces = {metric: np.array([json.loads(row[metric + '_trace'])[:count]
                                   for row in longest], dtype=float)
                  for metric in ('train_accuracy', 'test_accuracy', 'round_times')}
        if any(values.shape != (25, count) or not np.isfinite(values).all()
               for values in traces.values()):
            raise ValueError('Budget history is incomplete or nonfinite')
        rounds = np.arange(1, count + 1)
        for metric, color, linestyle, label in (
                ('train_accuracy', '#748694', (0, (4, 2)), 'FIT'),
                ('test_accuracy', '#235C7E', '-', 'TEST')):
            split = traces[metric].reshape(5, 5, count).mean(axis=1) * 100
            y, sd = split.mean(axis=0), split.std(axis=0, ddof=1)
            axes[0, column].plot(rounds, y, color=color, linewidth=1.7,
                                 linestyle=linestyle, label=label)
            axes[0, column].fill_between(rounds, y - sd, y + sd,
                                         color=color, alpha=.10, linewidth=0)
        times = np.cumsum(traces['round_times'], axis=1)
        grid = np.linspace(times[:, 0].max(), times[:, -1].min(), 160)
        interpolated = np.array([np.interp(grid, t, y) for t, y in zip(times, traces['test_accuracy'])])
        split = interpolated.reshape(5, 5, len(grid)).mean(axis=1) * 100
        y, sd = split.mean(axis=0), split.std(axis=0, ddof=1)
        axes[1, column].plot(grid, y, color='#235C7E', linewidth=1.7)
        axes[1, column].fill_between(grid, y - sd, y + sd,
                                     color='#235C7E', alpha=.12, linewidth=0)
        axes[0, column].set_title(f'{name}: first {count} rounds', pad=7)
        axes[0, column].set_xlabel('Communication rounds')
        axes[0, column].set_xlim(0, count)
        axes[0, column].set_xticks(np.arange(0, count + 1, 20 if count == 80 else 10))
        axes[1, column].set_xlabel('Cumulative round compute (s)')
        axes[1, column].set_xlim(left=0)
        for ax in axes[:, column]:
            ax.set_ylim(40, 74)
        print(f'{name}: {count} common rounds; common time window {grid[0]:.3f}--{grid[-1]:.3f}s')
    axes[0, 0].set_ylabel('Clustering ACC (%)')
    axes[1, 0].set_ylabel('TEST ACC (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(.53, 1.01))
    fig.subplots_adjust(left=.075, right=.99, top=.83, bottom=.14, hspace=.92, wspace=.35)
    finish(fig, 'budget_curves.pdf', SOURCES[first:])

    # Solver comparison keeps every native result. Partial solves can change
    # trajectories, so a speed ratio alone does not establish equivalence.
    first = len(SOURCES)
    solver = []
    for dataset in DATASETS:
        clients = REFERENCE_CLIENTS[dataset]
        raw = pd.concat([read_split(dataset, seed, 'he_revision',
            f'grid_ours_solver_ckks_beta0.5_clients{clients}') for seed in range(5)])
        paired = raw.pivot(index=['data_seed', 'init_seed'], columns='condition',
                           values='training_core_seconds')
        solver.append((paired.dense / paired.partial).groupby('data_seed').mean())
    large = pd.concat([read_split('har', seed, 'large_silo',
        'grid_ours_large_silo_ckks_sourcebeta0.5_sourceclients50_viewclients3_rounds3', clients=50)
        for seed in range(5)])
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw={'width_ratios': [1.05, 1.25, 1.25]})
    for position, values in enumerate(solver):
        axes[0].scatter(values, np.full(5, position) + np.linspace(-.1, .1, 5),
                        s=11, color='#AEC1CE', linewidths=0, zorder=2)
        axes[0].errorbar(values.mean(), position, xerr=values.std(ddof=1),
                         fmt='D', color='#235C7E', markersize=4, capsize=2, zorder=3)
    axes[0].axvline(1, color='#8999A6', lw=.8, ls='--')
    axes[0].set_yticks(range(5), NAMES)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Dense / partial core time')
    axes[0].set_title('(a) Canonical solver runs', pad=7)
    for method, label, color, marker in (
            ('ours_dense', 'ACES dense', '#235C7E', 'o'),
            ('ours_partial', 'ACES partial', '#CB7846', '^'),
            ('ffcm_not_applicable', 'FFCM', '#279989', 's')):
        block = large[large.condition.str.startswith(method + '_n')]
        split = block.groupby(['local_samples', 'data_seed'])[
            ['training_core_seconds', 'peak_rss_bytes']].mean()
        stats = split.groupby('local_samples').agg(['mean', 'std'])
        for ax, metric, scale in ((axes[1], 'training_core_seconds', 1),
                                   (axes[2], 'peak_rss_bytes', 1 / 2**20)):
            x = stats.index.to_numpy()
            y = stats[(metric, 'mean')].to_numpy() * scale
            sd = stats[(metric, 'std')].to_numpy() * scale
            ax.plot(x, y, color=color, marker=marker, markersize=4,
                    linewidth=1.6, label=label)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=.10, linewidth=0)
            ax.set_xscale('log', base=2)
            ax.set_xticks([128, 512, 1024, 2048], ['128', '512', '1k', '2k'])
            ax.minorticks_off()
            ax.set_xlabel('Records per silo')
    axes[1].set_ylabel('Training core (s)')
    axes[1].set_ylim(bottom=0)
    axes[1].set_title('(b) HAR: 3 silos, 3 rounds', pad=7)
    axes[2].set_ylabel('Peak RSS (MiB)')
    axes[2].set_title('(c) Worker lifetime memory', pad=7)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.56, 1.01),
               frameon=False, ncol=3, handlelength=1.5, columnspacing=1.4)
    fig.subplots_adjust(left=.10, right=.99, top=.76, bottom=.25, wspace=.65)
    finish(fig, 'solver_large_silo.pdf', SOURCES[first:])

    # Single-column article view; the review copy retains the full solver panel.
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95))
    for method, label, color, marker in (
            ('ours_dense', 'Dense', '#235C7E', 'o'),
            ('ours_partial', 'Partial', '#CB7846', '^'),
            ('ffcm_not_applicable', 'FFCM', '#279989', 's')):
        block = large[large.condition.str.startswith(method + '_n')]
        split = block.groupby(['local_samples', 'data_seed'])[
            ['training_core_seconds', 'peak_rss_bytes']].mean()
        stats = split.groupby('local_samples').agg(['mean', 'std'])
        for ax, metric, scale in ((axes[0], 'training_core_seconds', 1),
                                   (axes[1], 'peak_rss_bytes', 1 / 2**20)):
            x = stats.index.to_numpy()
            y = stats[(metric, 'mean')].to_numpy() * scale
            sd = stats[(metric, 'std')].to_numpy() * scale
            ax.plot(x, y, color=color, marker=marker, markersize=3,
                    linewidth=1.3, label=label)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=.10, linewidth=0)
            ax.set_xscale('log', base=2)
            ax.set_xticks([128, 512, 2048], ['128', '512', '2k'])
            ax.minorticks_off()
            ax.set_xlabel('Records / silo', fontsize=8)
            ax.tick_params(labelsize=8)
    axes[0].set_ylabel('Core time (s)', fontsize=8)
    axes[0].set_ylim(bottom=0)
    axes[0].set_title('(a) Training', fontsize=8.5, pad=6)
    axes[1].set_ylabel('Peak RSS (MiB)', fontsize=8)
    axes[1].set_title('(b) Memory', fontsize=8.5, pad=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, 1.02),
               frameon=False, ncol=3, handlelength=1.1, columnspacing=1.0, fontsize=8)
    fig.subplots_adjust(left=.14, right=.985, top=.72, bottom=.26, wspace=.70)
    finish(fig, 'large_silo_cost.pdf', SOURCES[first:])

    # Full iterative process execution: pair within solver, split and seed.
    # These contrasts include numerical trajectory sensitivity, not just CKKS error.
    first = len(SOURCES)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))
    for position, dataset in enumerate(DATASETS):
        clients = REFERENCE_CLIENTS[dataset]
        raw = pd.concat([read_split(dataset, seed, 'full_roles',
            f'grid_ours_full_roles_beta0.5_clients{clients}') for seed in range(5)])
        for solver, color, marker, offset in (
                ('dense', '#235C7E', 'o', -.13), ('partial', '#CB7846', '^', .13)):
            native = raw[raw.condition.eq('native_' + solver)].set_index(['data_seed', 'init_seed'])
            process = raw[raw.condition.eq('process_' + solver)].set_index(['data_seed', 'init_seed'])
            if (len(native) != 25 or len(process) != 25 or not native.index.equals(process.index)
                    or not process.role_boundary_passed.all()
                    or not process.distinct_role_processes.eq(3).all()):
                raise ValueError('Incomplete full-process contrast: ' + dataset + '/' + solver)
            contrasts = ((process.test_acc - native.test_acc).groupby('data_seed').mean() * 100,
                         (process.wall_seconds / native.wall_seconds).groupby('data_seed').mean())
            for ax, values in zip(axes, contrasts):
                y = position + offset
                ax.scatter(values, y + np.linspace(-.035, .035, 5), color=color,
                           s=11, alpha=.35, linewidths=0, zorder=2)
                ax.errorbar(values.mean(), y, xerr=values.std(ddof=1), fmt=marker,
                            color=color, markersize=4.5, capsize=2.5, elinewidth=1.1,
                            label=solver.capitalize() if position == 0 else None, zorder=3)
    for ax, reference, title, xlabel in (
            (axes[0], 0, '(a) Paired quality change', 'Process - native TEST ACC (pp)'),
            (axes[1], 1, '(b) End-to-end execution', 'Process / native wall time')):
        ax.axvline(reference, color='#8999A6', lw=.8, ls=(0, (3, 3)), zorder=1)
        ax.set_yticks(range(5), NAMES)
        ax.invert_yaxis()
        ax.set_title(title, pad=8)
        ax.set_xlabel(xlabel)
        ax.grid(axis='x', color='#E5E9EE', linewidth=.6)
        ax.margins(x=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.54, 1.01),
               frameon=False, ncol=2, handlelength=1.4, columnspacing=2)
    fig.subplots_adjust(left=.12, right=.985, top=.77, bottom=.23, wspace=.47)
    finish(fig, 'process_training.pdf', SOURCES[first:])

    plot_privacy_execution_cost()

    # Canonical privacy probes, including all exposed/shuffled controls.  These
    # are empirical final-center and byte views, not full-transcript attacks.
    first = len(SOURCES)
    privacy, controls = [], []
    for dataset in DATASETS:
        clients = REFERENCE_CLIENTS[dataset]
        for seed in range(5):
            base = RESULTS / dataset / f'federated_{seed}_clients{clients}_fixed_best'
            for target, study, expected_conditions, metric in (
                    (privacy, 'privacy', ('plain', 'ckks'), 'mia_auc'),
                    (controls, 'privacy_control', ('mia_exposed_fit', 'mia_shuffled_labels',
                                                  'bytes_plaintext', 'bytes_shuffled_labels'), 'auc')):
                path = base / f'grid_ours_{study}_beta0.5_clients{clients}.csv'
                block = pd.read_csv(path)
                expected = {(condition, init_seed) for condition in expected_conditions
                            for init_seed in range(40, 45)}
                if (not block.dataset.eq(dataset).all() or not block.data_seed.eq(seed).all()
                        or not block.clients.eq(clients).all() or len(block) != len(expected)
                        or set(zip(block.condition, block.init_seed)) != expected
                        or not np.isfinite(block[metric].to_numpy(float)).all()
                        or not block[metric].between(0, 1).all()):
                    raise ValueError('Incomplete canonical privacy source: ' + str(path))
                target.append(block)
                SOURCES.append(path)
    privacy = pd.concat(privacy, ignore_index=True)
    controls = pd.concat(controls, ignore_index=True)
    if not np.isfinite(privacy.loc[privacy.condition.eq('ckks'), 'ciphertext_auc']).all():
        raise ValueError('Missing CKKS byte-challenge AUC')
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), sharey=True)
    legends = [
        Line2D([], [], color='#235C7E', marker='o', markerfacecolor='white', lw=0,
               markersize=4.5, label='Plaintext'),
        Line2D([], [], color='#235C7E', marker='o', lw=0,
               markersize=4.5, label='CKKS'),
        Line2D([], [], color='#CB7846', marker='D', lw=0,
               markersize=4.2, label='Exposed control'),
        Line2D([], [], color='#7C818A', marker='x', lw=0,
               markersize=4.5, label='Shuffled labels'),
    ]
    panels = (
        ((privacy, 'plain', 'mia_auc', -.27, '#235C7E', 'o', 'white'),
         (privacy, 'ckks', 'mia_auc', -.09, '#235C7E', 'o', '#235C7E'),
         (controls, 'mia_exposed_fit', 'auc', .09, '#CB7846', 'D', '#CB7846'),
         (controls, 'mia_shuffled_labels', 'auc', .27, '#7C818A', 'x', '#7C818A')),
        ((privacy, 'ckks', 'ciphertext_auc', -.18, '#235C7E', 'o', '#235C7E'),
         (controls, 'bytes_plaintext', 'auc', 0, '#CB7846', 'D', '#CB7846'),
         (controls, 'bytes_shuffled_labels', 'auc', .18, '#7C818A', 'x', '#7C818A')),
    )
    for ax, specifications, title in zip(axes, panels, (
            '(a) Final-center membership inference', '(b) Serialized-byte challenge')):
        for source, condition, metric, offset, color, marker, face in specifications:
            split = (source[source.condition.eq(condition)]
                     .groupby(['dataset', 'data_seed'])[metric].mean().unstack('data_seed')
                     .reindex(DATASETS))
            means, sd = split.mean(axis=1), split.std(axis=1, ddof=1)
            ax.errorbar(means, np.arange(5) + offset, xerr=sd, fmt=marker,
                        color=color, markerfacecolor=face, markeredgewidth=1.0,
                        markersize=4.2, capsize=2, elinewidth=.9, zorder=3)
        ax.axvline(.5, color='#A5AFB8', linewidth=.85, linestyle=(0, (3, 3)), zorder=1)
        ax.set_xlim(.385, 1.04)
        ax.set_ylim(4.48, -.48)
        ax.set_xticks([.4, .5, .6, .8, 1.0], ['.4', '.5', '.6', '.8', '1.0'])
        ax.set_xlabel('Attack AUC')
        ax.set_title(title, fontsize=9.3, pad=9)
        ax.set_yticks(range(5), NAMES)
        ax.grid(axis='x', color='#E5E9EE', linewidth=.6)
    fig.legend(handles=legends, ncol=4, frameon=False, loc='upper center',
               bbox_to_anchor=(.53, 1.01), handlelength=1.0, columnspacing=1.4, fontsize=8.6)
    fig.subplots_adjust(left=.115, right=.985, top=.77, bottom=.20, wspace=.22)
    finish(fig, 'privacy_security.pdf', SOURCES[first:])


if __name__ == '__main__':
    main()
