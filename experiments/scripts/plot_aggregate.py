"""Cross-seed figures: median + IQR bands, method comparison, ablation heatmap.

Stage 3 figures. The convergence panels use the per-round series saved by
``run_single`` (``experiments/results/traces``), aggregated as median with a
25-75% IQR band across seeds -- a single-seed curve cannot show initialization
variance, and the mean alone hides the collapse of the inter-center distance.

``plot_comparison`` from federated/ours/tracker.py is called with duck-typed
tracker shims rebuilt from those traces when at least two methods recorded one.
Baselines (fkm/ffcm/nnfc) keep no per-round history at all, so when they are the
only other methods the comparison falls back to a final-metric bar chart and says
so in the title instead of inventing curves.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from experiments import schema  # noqa: E402
from federated.ours.tracker import plot_comparison  # noqa: E402

BAND_KEYS = (('center_shift_orig', 'Center Shift (original space, log)'),
             ('center_shift_emb', 'Center Shift (embedding space, log)'),
             ('min_distances_orig', 'Min Inter-Center Distance'))


def load_traces(dataset=None, method='ours', data_seed=None, beta=None):
    """Every trace npz matching the filters, as a list of dicts of arrays."""
    traces = []
    for path in sorted(glob.glob(os.path.join(schema.TRACES_DIR, '*', '*.npz'))):
        name = os.path.basename(os.path.dirname(path))
        parts = name.split('_')
        if len(parts) != 4:
            continue
        if dataset is not None and parts[0] != dataset:
            continue
        if data_seed is not None and parts[1] != 'seed%d' % int(data_seed):
            continue
        if beta is not None and parts[2] != 'beta%g' % float(beta):
            continue
        if method is not None and not os.path.basename(path).startswith(method + '_init'):
            continue
        with np.load(path) as stored:
            traces.append({'path': path, 'arrays': {k: stored[k] for k in stored.files}})
    return traces


def _stack(series):
    """Align variable-length round series on their common prefix."""
    length = min(len(values) for values in series)
    return np.vstack([np.asarray(values[:length], dtype=float) for values in series])


def median_band(traces, key):
    """(rounds, median, q25, q75) across seeds; None when the key is absent."""
    series = [trace['arrays'][key] for trace in traces if key in trace['arrays']]
    if not series:
        return None
    stacked = _stack(series)
    q25, median, q75 = np.percentile(stacked, [25, 50, 75], axis=0)
    rounds = np.arange(1, len(median) + 1)
    return rounds, median, q25, q75


def plot_convergence_bands(results_csv=None, out_path=None, dataset=None,
                           method='ours', data_seed=None, beta=None):
    """Three panels, median line plus IQR ribbon; returns the figure path."""
    traces = load_traces(dataset=dataset, method=method, data_seed=data_seed, beta=beta)
    if not traces:
        raise ValueError('No %r traces match dataset=%r data_seed=%r beta=%r under %s'
                         % (method, dataset, data_seed, beta, schema.TRACES_DIR))
    out_path = out_path or os.path.join(schema.FIGURES_DIR, 'convergence_bands_%s.png' % method)
    figure, axes = plt.subplots(1, len(BAND_KEYS), figsize=(16, 4.4))
    for axis, (key, label) in zip(axes, BAND_KEYS):
        band = median_band(traces, key)
        if band is None:
            axis.set_visible(False)
            continue
        rounds, median, q25, q75 = band
        positive = median > 0 if 'log' in label else np.ones_like(median, dtype=bool)
        if 'log' in label:
            axis.set_yscale('log')
        axis.fill_between(rounds, q25, q75, alpha=0.25, color='#2E86AB', label='IQR (25-75%)')
        axis.plot(rounds, median, color='#2E86AB', linewidth=2,
                  marker='o', markersize=4, label='median of %d seed(s)' % len(traces))
        axis.set_xlabel('Round')
        axis.set_ylabel(label)
        axis.set_title('%s (n=%d runs)' % (key, len(traces)))
        axis.grid(True, alpha=0.3, linestyle='--')
        axis.legend(fontsize=8)
        if not positive.all() and 'log' in label:
            axis.text(.02, .02, 'non-positive values clipped by log scale',
                      transform=axis.transAxes, fontsize=7, color='#B00020')
    scope = '%s | beta=%s | data_seed=%s | band across init seeds' % (
        dataset or 'all', 'all' if beta is None else beta,
        'all' if data_seed is None else data_seed)
    figure.suptitle('Convergence: median + IQR (%s)' % scope)
    figure.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    figure.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(figure)
    return out_path


def _shim_trackers(dataset, data_seed, beta):
    """One tracker-like object per method, rebuilt from the saved traces."""
    shims = {}
    for method in ('ours',):
        traces = load_traces(dataset=dataset, method=method, data_seed=data_seed, beta=beta)
        if not traces:
            continue
        first = traces[0]['arrays']
        history = {key: np.asarray(values) for key, values in first.items()
                   if key != 'total_bytes'}
        shims[method] = SimpleNamespace(
            history=history,
            communication={'total_bytes': float(first.get('total_bytes', np.nan))})
    return shims


def _bar_comparison(frame, out_path):
    """Fallback when fewer than two methods kept a per-round history."""
    metrics = ('acc', 'nmi', 'ari', 'test_proxy_acc')
    methods = [m for m in frame['method'].dropna().unique()]
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    width = 0.8 / max(len(methods), 1)
    for position, method in enumerate(methods):
        subset = frame[frame['method'] == method]
        means = [subset[key].mean() for key in metrics]
        stds = [subset[key].std(ddof=1) if subset[key].notna().sum() > 1 else 0 for key in metrics]
        axis.bar(np.arange(len(metrics)) + position * width, means, width,
                 yerr=stds, capsize=3, label=method)
    axis.set_xticks(np.arange(len(metrics)) + width * (len(methods) - 1) / 2)
    axis.set_xticklabels(metrics)
    axis.set_ylabel('score')
    axis.set_title('Final metrics per method (mean, SD across runs)\n'
                   'curves not drawn: baselines keep no per-round history')
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3, linestyle='--', axis='y')
    figure.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    figure.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(figure)
    return out_path


def plot_method_comparison(results_csv, out_path=None, dataset=None, data_seed=None, beta=None):
    """plot_comparison when >=2 methods have traces, else an honest bar chart."""
    frame = schema.read_results(results_csv)
    if dataset is not None:
        frame = frame[frame['dataset'] == dataset]
    out_path = out_path or os.path.join(schema.FIGURES_DIR, 'method_comparison.png')
    shims = _shim_trackers(dataset, data_seed, beta)
    if len(shims) >= 2:
        return plot_comparison(shims, save_path=out_path, show=False)
    return _bar_comparison(frame, out_path)


def plot_ablation_heatmap(results_csv, out_path=None, metric='acc', dataset=None):
    """fuzziness x lambda_space heatmap; skipped (returns None) without an ablation grid."""
    import pandas as pd
    frame = schema.read_results(results_csv)
    if dataset is not None:
        frame = frame[frame['dataset'] == dataset]
    frame = frame[frame['method'] == 'ours']
    if frame.empty:
        print('ablation heatmap skipped: no ours rows in %s' % results_csv)
        return None
    fuzziness = sorted(frame['fuzziness'].dropna().unique())
    lambda_space = sorted(frame['lambda_space'].dropna().unique())
    if len(fuzziness) < 2 or len(lambda_space) < 2:
        print('ablation heatmap skipped: needs a fuzziness x lambda_space grid, found '
              '%d x %d value(s) -- pass a dedicated ablation results.csv'
              % (len(fuzziness), len(lambda_space)))
        return None
    table = frame.pivot_table(index='fuzziness', columns='lambda_space', values=metric,
                              aggfunc='mean')
    figure, axis = plt.subplots(figsize=(1.1 * len(lambda_space) + 3, 0.8 * len(fuzziness) + 2.5))
    image = axis.imshow(table.to_numpy(dtype=float), aspect='auto', cmap='viridis')
    axis.set_xticks(np.arange(len(lambda_space)), ['%.2g' % value for value in lambda_space])
    axis.set_yticks(np.arange(len(fuzziness)), ['%.2g' % value for value in fuzziness])
    axis.set_xlabel('lambda_space'); axis.set_ylabel('fuzziness')
    axis.set_title('Ours: mean %s across seeds' % metric)
    for row in range(len(fuzziness)):
        for column in range(len(lambda_space)):
            value = table.to_numpy(dtype=float)[row, column]
            if np.isfinite(value):
                axis.text(column, row, '%.3f' % value, ha='center', va='center',
                          color='white', fontsize=8)
    figure.colorbar(image, ax=axis, shrink=0.9)
    figure.tight_layout()
    out_path = out_path or os.path.join(schema.FIGURES_DIR, 'ablation_%s.png' % metric)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    figure.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(figure)
    return out_path


def render_all(results_csv, dataset=None, data_seed=None, beta=None):
    """Convenience wrapper used by run_sweep --figures; returns the written paths."""
    written = []
    try:
        written.append(plot_convergence_bands(results_csv, dataset=dataset,
                                              data_seed=data_seed, beta=beta))
    except ValueError as error:
        print('convergence bands skipped: %s' % error)
    written.append(plot_method_comparison(results_csv, dataset=dataset,
                                          data_seed=data_seed, beta=beta))
    heatmap = plot_ablation_heatmap(results_csv, dataset=dataset)
    if heatmap:
        written.append(heatmap)
    print('figures: %s' % ', '.join(written))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results', default=os.path.join(schema.RESULTS_DIR, 'results.csv'))
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--data-seed', type=int, default=None)
    parser.add_argument('--beta', type=float, default=None)
    args = parser.parse_args(argv)
    render_all(args.results, args.dataset, args.data_seed, args.beta)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
