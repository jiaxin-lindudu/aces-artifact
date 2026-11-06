"""Convergence tracking, plotting and empirical-privacy statistics for "Ours".

WHAT THE TRACKER MEASURES
-------------------------
`ConvergenceTracker.update` is called once per federated round with the CURRENT
centers and the labels produced for that round. Everything it stores is either

  * a protocol quantity   : center movement, Procrustes alignment error,
    communication bytes, per-stage timings;
  * an OFFLINE evaluation : inertia, silhouette, Davies-Bouldin,
    Calinski-Harabasz, accuracy/NMI/ARI and the fuzzy indices (PC, PE, XB, FS),
    computed after the round on cached FIT coordinates and on TEST points. None
    of these are uploaded, none are covered by CKKS, and none feed back into the
    training. They must be reported as offline evaluation, not as protocol output.

Every metric exists in a `train_*` and a `test_*` variant and every key is
prefixed accordingly; there are no unprefixed metric keys.

WHAT THE PRIVACY PART IS
------------------------
`membership_inference_on_centers`, `evaluate_attack_features` and
`evaluate_ciphertext_challenge` are EMPIRICAL ATTACK PROBES on released objects.
A result at chance level is an observation, NOT a privacy proof; the dicts they
return say so explicitly and that text must not be watered down.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import cdist
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)

# Set publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


# ============================================================================
# Convergence Tracking Class
# ============================================================================

class ConvergenceTracker:
    """
    Tracks convergence metrics during federated training.
    Tracks metrics on both X_train and X_test separately.
    """

    def __init__(self):
        # Core history
        self.history = {
            'round': [],
            # Center movement
            'center_shift_orig': [],
            'center_shift_emb': [],
            # Procrustes error
            'alignment_error': [],
            # TRAIN set metrics
            'train_inertia': [],
            'train_silhouette': [],
            'train_davies_bouldin': [],
            'train_calinski_harabasz': [],
            'train_accuracy': [],
            # NEW: Fuzzy metrics for TRAIN set
            'train_pc': [],
            'train_pe': [],
            'train_xie_beni': [],
            'train_fukuyama_sugeno': [],
            # TEST set metrics
            'test_inertia': [],
            'test_silhouette': [],
            'test_davies_bouldin': [],
            'test_calinski_harabasz': [],
            'test_accuracy': [],
            # NEW: Fuzzy metrics for TEST set
            'test_pc': [],
            'test_pe': [],
            'test_xie_beni': [],
            'test_fukuyama_sugeno': [],
            # Resources
            'communication_bytes': [],
            'encryption_time': [],
            'aggregation_time': [],
            'round_time': []
        }
        for name in ('train_nmi', 'train_ari', 'test_nmi', 'test_ari', 'embedding_time',
                     'statistics_time', 'decryption_time', 'topology_time', 'uplink_bytes',
                     'downlink_bytes', 'interservice_bytes'):
            self.history[name] = []
        self.communication = {}

    @staticmethod
    def get_labels(X, centers):
        """Assign each point to nearest cluster center.

        OFFLINE EVALUATION helper: an original-space nearest-center proxy, NOT a
        dual-space prediction. Public name kept next to the private alias below
        because plotting code outside this class calls it.
        """
        return np.argmin(cdist(X, centers), axis=1)

    @staticmethod
    def _get_labels(X, centers):
        """Backward-compatible alias of :meth:`get_labels`."""
        return ConvergenceTracker.get_labels(X, centers)

    def _get_clustering_metrics(self, X, centers, labels=None):
        """Return inertia, silhouette, DB, CH metrics for X and given centers."""
        labels = self._get_labels(X, centers) if labels is None else np.asarray(labels)
        inertia = np.sum((X - centers[labels]) ** 2)
        uniq = np.unique(labels)
        # Only compute clustering metrics when more than 1 cluster is detected
        if 1 < len(uniq) < len(X):
            try:
                sil = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=0)
            except ValueError:
                sil = np.nan
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
        else:
            sil, db, ch = np.nan, np.nan, np.nan
        return inertia, sil, db, ch

    # NEW: Helper function to calculate fuzzy metrics
    def _get_fuzzy_metrics(self, X, U, centers, fuzziness_m):
        """
        Calculates Partition Coefficient (PC), Partition Entropy (PE),
        Xie-Beni Index (XB), and Fukuyama-Sugeno Index (FS).

        Args:
            X (np.ndarray): Data points.
            U (np.ndarray): Fuzzy membership matrix (N_samples, N_clusters).
            centers (np.ndarray): Cluster centers (N_clusters, N_features).
            fuzziness_m (float): Fuzziness parameter (m).

        Returns:
            tuple: (pc, pe, xb, fs)
        """
        n_samples, n_clusters = U.shape

        if n_clusters <= 1:  # or any(np.sum(U, axis=0) == 0):
            return np.nan, np.nan, np.nan, np.nan

        # Ensure U sums to 1 across rows
        U = U / np.sum(U, axis=1, keepdims=True)

        # Avoid division by zero if a cluster has no members (or very few)
        # This can happen in early rounds or with very skewed data
        if np.any(np.sum(U, axis=0) < 1e-6):  # If a cluster is essentially empty
            return np.nan, np.nan, np.nan, np.nan

        # Calculate squared distances
        d_sq = cdist(X, centers, metric='sqeuclidean')

        # Partition Coefficient (PC)
        pc = np.sum(U ** 2) / n_samples

        # Partition Entropy (PE)
        # Avoid log(0) for very small U values
        pe = -np.sum(U * np.log(U + 1e-10)) / n_samples

        # Xie-Beni Index (XB)
        # Compactness: sum of (membership^m * distance^2) for each point to its center
        compactness = np.sum((U ** fuzziness_m) * d_sq)

        # Separation: minimum squared distance between any two centers
        if n_clusters > 1:
            center_dist_sq = cdist(centers, centers, metric='sqeuclidean')
            # Replace diagonal with infinity to find min non-self distance
            np.fill_diagonal(center_dist_sq, np.inf)
            separation = np.min(center_dist_sq)
        else:
            separation = 1e-10  # Arbitrary small positive number if only one cluster

        if separation < 1e-10:  # Avoid division by zero
            xb = np.nan
        else:
            xb = compactness / (n_samples * separation)

        # Fukuyama-Sugeno Index (FS)
        # Sum of (membership^m * distance^2) to cluster center
        within_cluster_sum_sq = np.sum((U ** fuzziness_m) * d_sq)

        # Calculate total mean (overall centroid)
        overall_centroid = np.mean(X, axis=0)

        # Sum of (membership^m * distance^2) to overall centroid
        # Re-calculate distances to overall_centroid for all points, then sum with membership
        dist_to_overall_centroid_sq = np.sum((X - overall_centroid) ** 2, axis=1)
        between_cluster_sum_sq = np.sum((U ** fuzziness_m) * dist_to_overall_centroid_sq[:, np.newaxis])

        fs = within_cluster_sum_sq - between_cluster_sum_sq

        return pc, pe, xb, fs

    def update(self, round_idx, server, clients,
               X_train=None, y_train=None, U_train=None,  # NEW: U_train for fuzzy metrics
               X_test=None, y_test=None, U_test=None,  # NEW: U_test for fuzzy metrics
               compute_accuracy_fn=None,
               round_time=0,
               prev_centers_orig=None,
               prev_centers_emb=None,
               train_labels=None, test_labels=None,
               fuzziness_m=2.0,  # NEW: Pass fuzziness parameter for fuzzy metrics
               uplink_bytes=None, downlink_bytes=0, interservice_bytes=0, resource=None):
        """
        Update tracking metrics for current round.

        Args:
            round_idx: Current round number
            server: Server instance with centers_orig and centers_emb
            clients: List of client instances
            X_train: Training data
            y_train: Training labels (for accuracy)
            U_train: Fuzzy membership matrix for training data (N_samples, N_clusters)
            X_test: Test data
            y_test: Test labels (for accuracy)
            U_test: Fuzzy membership matrix for test data (N_samples, N_clusters)
            compute_accuracy_fn: Function(y_true, y_pred) -> accuracy score
            round_time: Time taken for this round
            prev_centers_orig: Previous round centers in original space
            prev_centers_emb: Previous round centers in embedding space
            fuzziness_m: Fuzziness parameter 'm' used in clustering.
        """
        self.history['round'].append(round_idx)

        # Center shifts
        if prev_centers_orig is not None and prev_centers_emb is not None:
            shift_orig = np.linalg.norm(server.centers_orig - prev_centers_orig)
            shift_emb = np.linalg.norm(server.centers_emb - prev_centers_emb)
        else:
            shift_orig, shift_emb = 0.0, 0.0
        self.history['center_shift_orig'].append(shift_orig)
        self.history['center_shift_emb'].append(shift_emb)

        # Alignment error
        if len(server.alignment_errors) > 0:
            self.history['alignment_error'].append(server.alignment_errors[-1])
        else:
            self.history['alignment_error'].append(0.0)

        # Metrics for train and test
        # NEW: Added U_data to the loop for fuzzy metrics
        for X, y, U_data, lbl, pred_labels in [(X_train, y_train, U_train, 'train', train_labels),
                                               (X_test, y_test, U_test, 'test', test_labels)]:
            if X is not None:
                # Clustering quality metrics (crisp)
                if pred_labels is None:
                    pred_labels = self._get_labels(X, server.centers_orig)
                inertia, sil, db, ch = self._get_clustering_metrics(X, server.centers_orig, pred_labels)

                # NEW: Fuzzy metrics
                if U_data is not None and U_data.shape[0] == X.shape[0]:  # Ensure U_data is valid
                    pc, pe, xb, fs = self._get_fuzzy_metrics(X, U_data, server.centers_orig, fuzziness_m)
                else:
                    pc, pe, xb, fs = np.nan, np.nan, np.nan, np.nan

                # Accuracy metric (requires true labels)
                if y is not None and compute_accuracy_fn is not None:
                    acc = compute_accuracy_fn(y, pred_labels)
                    nmi = normalized_mutual_info_score(y, pred_labels)
                    ari = adjusted_rand_score(y, pred_labels)
                else:
                    acc, nmi, ari = np.nan, np.nan, np.nan
            else:
                inertia, sil, db, ch, acc, nmi, ari = (np.nan,) * 7
                pc, pe, xb, fs = (np.nan,) * 4  # NEW: Also set fuzzy metrics to nan

            self.history[f'{lbl}_inertia'].append(inertia)
            self.history[f'{lbl}_silhouette'].append(sil)
            self.history[f'{lbl}_davies_bouldin'].append(db)
            self.history[f'{lbl}_calinski_harabasz'].append(ch)
            self.history[f'{lbl}_accuracy'].append(acc)
            self.history[f'{lbl}_nmi'].append(nmi)
            self.history[f'{lbl}_ari'].append(ari)
            # NEW: Append fuzzy metrics
            self.history[f'{lbl}_pc'].append(pc)
            self.history[f'{lbl}_pe'].append(pe)
            self.history[f'{lbl}_xie_beni'].append(xb)
            self.history[f'{lbl}_fukuyama_sugeno'].append(fs)

        # Communication and resource/time tracking
        uplink_bytes = sum(c.communication_bytes for c in clients) if uplink_bytes is None else uplink_bytes
        total_comm = uplink_bytes + downlink_bytes + interservice_bytes
        resource = resource or {}
        for name in ('embedding', 'statistics', 'decryption', 'topology'):
            self.history[name + '_time'].append(resource.get(name, 0.0))
        self.history['uplink_bytes'].append(uplink_bytes)
        self.history['downlink_bytes'].append(downlink_bytes)
        self.history['interservice_bytes'].append(interservice_bytes)
        total_enc_time = sum(c.encryption_time for c in clients)
        self.history['communication_bytes'].append(total_comm)
        self.history['encryption_time'].append(total_enc_time)
        self.history['aggregation_time'].append(getattr(server, 'aggregation_time', 0))
        self.history['round_time'].append(round_time)


# ============================================================================
# Plotting Functions
# ============================================================================

TRAIN_COLOR = '#006D77'
TEST_COLOR = '#E63946'


def _has_curve(values):
    """Same emptiness test the panels have always used (NaN is not a curve)."""
    return max(values, default=0) > 0


def _plot_train_test(ax, tracker, key, title, ylabel, ylim=None, zero_line=False,
                     legend_loc='upper right'):
    """Draw one train-vs-test panel; return False when there is nothing to draw."""
    rounds = tracker.history['round']
    train, test = tracker.history['train_' + key], tracker.history['test_' + key]
    if not (_has_curve(train) or _has_curve(test)):
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontweight='bold')
        return False
    ax.plot(rounds, train, 'o-', linewidth=2, markersize=6, label='Train', color=TRAIN_COLOR)
    ax.plot(rounds, test, 's-', linewidth=2, markersize=6,
            label='Test (orig proxy)', color=TEST_COLOR)
    ax.set_xlabel('Training Round')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    if ylim is not None:
        ax.set_ylim(ylim)
    if zero_line:
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc=legend_loc)
    ax.grid(True, alpha=0.3, linestyle='--')
    return True


def plot_convergence_summary(tracker, save_path=None, show=True):
    """
    Create a comprehensive convergence summary plot with train/test metrics.

    Every metric panel is OFFLINE evaluation (see the module docstring): the
    values are computed after each round from cached FIT coordinates and from
    TEST points, and they never feed back into training.

    Args:
        tracker: ConvergenceTracker instance
        save_path: Path to save figure (optional)
        show: call plt.show(); pass False in headless/batch use
    """
    if tracker is None:
        print("No convergence tracker available for plotting.")
        return
    if not tracker.history['round']:
        print("No round recorded by the tracker; nothing to plot.")
        return

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

    rounds = tracker.history['round']

    # -------------------------------------------------------------------------
    # (a) Center Shift (Primary Convergence Metric)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.semilogy(rounds, tracker.history['center_shift_orig'],
                 'o-', linewidth=2, markersize=6, label='Original Space', color='#2E86AB')
    ax1.semilogy(rounds, tracker.history['center_shift_emb'],
                 's-', linewidth=2, markersize=6, label='Embedding Space', color='#A23B72')
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Center Shift (log scale)')
    ax1.set_title('(a) Convergence: Center Movement', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add convergence annotation
    if len(rounds) > 5:
        final_shift = tracker.history['center_shift_orig'][-1]
        ax1.axhline(y=final_shift, color='red', linestyle=':', alpha=0.5)
        ax1.text(rounds[-1], final_shift * 1.5, f'Final: {final_shift:.2e}',
                 fontsize=9, color='red')

    # -------------------------------------------------------------------------
    # (b) Alignment Error
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(rounds, tracker.history['alignment_error'],
             'D-', linewidth=2, markersize=5, color='#F18F01')
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Alignment Error')
    ax2.set_title('(b) Procrustes Alignment', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # -------------------------------------------------------------------------
    # (c)-(i) One train/test curve per quality metric
    # -------------------------------------------------------------------------
    _plot_train_test(fig.add_subplot(gs[1, 0]), tracker, 'inertia',
                     '(c) Within-Cluster Variance', 'Inertia')

    ax4 = fig.add_subplot(gs[1, 1])
    if _plot_train_test(ax4, tracker, 'silhouette', '(d) Cluster Separation',
                        'Silhouette Score', ylim=[-0.1, 1.0], zero_line=True,
                        legend_loc='lower right'):
        # Quality indicator for test set
        final_sil = tracker.history['test_silhouette'][-1]
        if final_sil > 0.5:
            quality, color = 'Excellent', 'green'
        elif final_sil > 0.25:
            quality, color = 'Good', 'orange'
        else:
            quality, color = 'Poor', 'red'
        ax4.text(0.05, 0.05, f'Test Quality: {quality}',
                 transform=ax4.transAxes, fontsize=9, color=color,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax5 = fig.add_subplot(gs[1, 2])
    if _plot_train_test(ax5, tracker, 'accuracy', '(e) Clustering Accuracy',
                        'Accuracy', ylim=[0, 1.0], legend_loc='lower right'):
        final_test_acc = tracker.history['test_accuracy'][-1]
        ax5.text(0.05, 0.95, f'Final Test Acc: {final_test_acc:.3f}',
                 transform=ax5.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    _plot_train_test(fig.add_subplot(gs[2, 0]), tracker, 'pc',
                     '(f) Partition Coefficient (PC) [↑]', 'PC',
                     ylim=[0, 1.0], legend_loc='lower right')
    _plot_train_test(fig.add_subplot(gs[2, 1]), tracker, 'pe',
                     '(g) Partition Entropy (PE) [↓]', 'PE')
    _plot_train_test(fig.add_subplot(gs[2, 2]), tracker, 'xie_beni',
                     '(h) Xie-Beni Index (XB) [↓]', 'XB')
    _plot_train_test(fig.add_subplot(gs[3, 0]), tracker, 'fukuyama_sugeno',
                     '(i) Fukuyama-Sugeno Index (FS) [↓]', 'FS')

    # Shift remaining plots to the new 4th row
    # -------------------------------------------------------------------------
    # (j) Communication Overhead
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[3, 1])  # Moved to 4th row, 2nd column
    comm_mb = [c / (1024 ** 2) for c in tracker.history['communication_bytes']]
    ax6.bar(rounds, comm_mb, color='#06A77D', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Training Round')
    ax6.set_ylabel('Communication (MiB)')
    ax6.set_title('(j) Communication Overhead', fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Total communication
    total_comm = tracker.communication.get('total_bytes', sum(tracker.history['communication_bytes'])) / 1024 ** 2
    ax6.text(0.95, 0.95, f'Total: {total_comm:.2f} MiB',
             transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # -------------------------------------------------------------------------
    # (k) Total Round Time
    # -------------------------------------------------------------------------
    ax8 = fig.add_subplot(gs[3, 2])  # Moved to 4th row, 3rd column
    ax8.plot(rounds, tracker.history['round_time'],
             'o-', linewidth=2, markersize=6, color='#FB5607')
    ax8.set_xlabel('Training Round')
    ax8.set_ylabel('Time (seconds)')
    ax8.set_title('(k) Total Round Time', fontweight='bold')
    ax8.grid(True, alpha=0.3, linestyle='--')

    # Average time
    avg_time = np.mean(tracker.history['round_time'])
    ax8.axhline(y=avg_time, color='green', linestyle='--', alpha=0.6)
    ax8.text(rounds[-1], avg_time * 1.1, f'Avg: {avg_time:.2f}s',
             fontsize=9, color='green')

    # Overall title
    fig.suptitle('Federated Anchored Spectral Clustering: Convergence Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    # Layout first: savefig after tight_layout would ignore the layout pass.
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Convergence summary saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_comparison(trackers_dict, save_path=None, show=True):
    """
    Compare multiple methods (e.g., DP vs CKKS vs No Privacy).

    Args:
        trackers_dict: Dictionary of {method_name: ConvergenceTracker}
        save_path: Path to save figure (optional)
        show: call plt.show(); pass False in headless/batch use
    """
    if not trackers_dict:
        raise ValueError("plot_comparison needs at least one tracker")
    empty = [name for name, tracker in trackers_dict.items() if not tracker.history['round']]
    if empty:
        raise ValueError("These trackers recorded no rounds: %s" % ', '.join(empty))
    # Every metric key is train_/test_ prefixed; there is no unprefixed key.
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#006D77', '#E63946', '#9B5DE5']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    def compare_panel(index, key, title, ylabel, ylim=None, log_scale=False,
                      skip_empty=True):
        """One curve per method for a single (prefixed) history key."""
        ax = axes[index]
        for idx, (name, tracker) in enumerate(trackers_dict.items()):
            values = tracker.history[key]
            # Skip methods whose metric is absent or degenerate (all NaN).
            if skip_empty and max(values, default=0) <= 0:
                continue
            draw = ax.semilogy if log_scale else ax.plot
            draw(tracker.history['round'], values,
                 marker=markers[idx % len(markers)], linewidth=2, markersize=5,
                 label=name, color=colors[idx % len(colors)])
        ax.set_xlabel('Training Round')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')

    compare_panel(0, 'center_shift_orig', '(a) Convergence Speed',
                  'Center Shift (log scale)', log_scale=True, skip_empty=False)
    compare_panel(1, 'test_silhouette', '(b) Clustering Quality (Silhouette)',
                  'Silhouette Score', ylim=[-0.1, 1.0])
    compare_panel(2, 'test_inertia', '(c) Within-Cluster Variance (Inertia)', 'Inertia')
    compare_panel(3, 'test_pc', '(d) Fuzzy Quality (PC) [↑]',
                  'Partition Coefficient', ylim=[0, 1.0])
    compare_panel(4, 'test_pe', '(e) Fuzzy Quality (PE) [↓]', 'Partition Entropy')
    compare_panel(5, 'test_xie_beni', '(f) Fuzzy Quality (XB) [↓]', 'Xie-Beni Index')

    # -------------------------------------------------------------------------
    # 7. Communication Comparison
    # -------------------------------------------------------------------------
    method_names = list(trackers_dict.keys())
    def labeled_bars(index, values, ylabel, title, fmt='%.2f'):
        bars = axes[index].bar(method_names, values, color=colors[:len(method_names)],
                               alpha=0.7, edgecolor='black')
        axes[index].set_ylabel(ylabel)
        axes[index].set_title(title, fontweight='bold')
        axes[index].grid(True, alpha=0.3, linestyle='--', axis='y')
        for bar in bars:
            height = bar.get_height()
            axes[index].text(bar.get_x() + bar.get_width() / 2., height, fmt % height,
                             ha='center', va='bottom', fontsize=9)

    labeled_bars(6, [t.communication.get('total_bytes', sum(t.history['communication_bytes'])) / 1024 ** 2
                     for t in trackers_dict.values()],
                 'Total Communication (MiB)', '(g) Communication Overhead')
    labeled_bars(7, [sum(t.history['round_time']) for t in trackers_dict.values()],
                 'Total Time (seconds)', '(h) Computational Cost', fmt='%.1fs')

    # -------------------------------------------------------------------------
    # 9. Final Quality Metrics (Summary Bar Plot) (was 6)
    # -------------------------------------------------------------------------
    metric_labels = ['Silhouette', 'DB Index\\n(inverted)', 'CH Index\\n(×0.001)',
                     'PC', 'PE\\n(inverted)', 'XB\\n(inverted)']

    def inverted_last(history, key):
        """Last value of `key`, inverted so that smaller-is-better becomes larger."""
        value = history[key][-1]
        return 1.0 / (value + 0.1) if value > 0 else 0

    final_metrics = {}
    for name, tracker in trackers_dict.items():
        # "inverted" entries exist so every bar can be read as "higher is better".
        history = tracker.history
        final_metrics[name] = [
            history['test_silhouette'][-1],
            inverted_last(history, 'test_davies_bouldin'),
            history['test_calinski_harabasz'][-1] / 1000.0,
            history['test_pc'][-1],
            inverted_last(history, 'test_pe'),
            inverted_last(history, 'test_xie_beni')]

    x = np.arange(len(metric_labels))
    width = 0.8 / len(method_names)

    for idx, (name, values) in enumerate(final_metrics.items()):
        offset = (idx - len(method_names) / 2 + 0.5) * width
        axes[8].bar(x + offset, values, width, label=name,
                    color=colors[idx % len(colors)], alpha=0.7, edgecolor='black')

    axes[8].set_ylabel('Score')
    axes[8].set_title('(i) Final Quality Metrics', fontweight='bold')
    axes[8].set_xticks(x)
    axes[8].set_xticklabels(metric_labels, fontsize=9)
    axes[8].legend(loc='upper left', fontsize=9)
    axes[8].grid(True, alpha=0.3, linestyle='--', axis='y')

    fig.suptitle('Method Comparison: Convergence, Quality, and Efficiency',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cluster_evolution(X, server, tracker, n_snapshots=4, save_path=None,
                           centers_history=None, show=True):
    """Visualize how clusters evolve over training rounds (2D PCA projection).

    The tracker does NOT store per-round centers, so the snapshots have to come
    from the caller: pass ``history['centers_orig']`` of the trainer as
    ``centers_history``. Its first entry is the initialization, so entry r+1
    belongs to round r+1.

    Without ``centers_history`` the figure is a single panel showing the FINAL
    state only and it says so in its title. Drawing the same final picture
    several times and labelling the copies "Round 1, Round 2, ..." is wrong, and
    that is exactly what this function used to do.

    Args:
        X: Full dataset
        server: Server instance (needs ``centers_orig``)
        tracker: ConvergenceTracker instance
        n_snapshots: Maximum number of snapshots to show
        save_path: Path to save figure
        centers_history: sequence of center matrices, one per round plus the init
        show: call plt.show(); pass False in headless/batch use
    """
    from sklearn.decomposition import PCA

    if n_snapshots < 1:
        raise ValueError("n_snapshots must be >= 1")
    total_rounds = len(tracker.history['round'])
    if total_rounds == 0:
        raise ValueError("The tracker recorded no rounds; nothing to plot")

    if centers_history is None:
        snapshots = [(total_rounds, np.asarray(server.centers_orig))]
        title = 'Final Clusters (no per-round centers supplied)'
    else:
        centers_history = [np.asarray(c, dtype=float) for c in centers_history]
        # Entry 0 is the initialization when one more entry than rounds exists.
        offset = 1 if len(centers_history) > total_rounds else 0
        available = [(round_idx + 1, centers_history[round_idx + offset])
                     for round_idx in range(total_rounds)
                     if round_idx + offset < len(centers_history)]
        if not available:
            raise ValueError("centers_history does not cover any recorded round")
        if len(available) <= n_snapshots:
            snapshots = available
        else:
            indices = np.linspace(0, len(available) - 1, n_snapshots, dtype=int)
            snapshots = [available[i] for i in indices]
        title = 'Cluster Evolution Over Training'

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(5 * len(snapshots), 4))
    if len(snapshots) == 1:
        axes = [axes]

    for ax, (round_num, centers) in zip(axes, snapshots):
        labels = tracker.get_labels(X, centers)
        centers_2d = pca.transform(centers)
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                   cmap='tab10', alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidth=2,
                   label='Centers', zorder=5)
        ax.set_title(f'Round {round_num}', fontweight='bold')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Cluster evolution saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def print_convergence_summary(tracker, test_basis=None):
    """
    Print a text summary of convergence metrics.

    Args:
        tracker: ConvergenceTracker instance
        test_basis: how TEST labels were produced, e.g.
            ``history['prediction']['test']`` of the trainer. The tracker cannot
            know this itself, so the caller has to supply it; without it the
            summary refuses to guess.
    """
    if tracker is None:
        print("No convergence tracker available.")
        return

    print("\n" + "=" * 70)
    print("CONVERGENCE SUMMARY")
    print("=" * 70)

    print(f"\n  Training Rounds: {len(tracker.history['round'])}")
    print(f"  Final Center Shift (Orig): {tracker.history['center_shift_orig'][-1]:.6f}")
    print(f"  Final Center Shift (Emb):  {tracker.history['center_shift_emb'][-1]:.6f}")

    # Alignment error
    if tracker.history['alignment_error'][-1] > 0:
        print(f"  Final Alignment Error:     {tracker.history['alignment_error'][-1]:.6f}")

    # Training set metrics
    if max(tracker.history['train_silhouette'], default=0) > 0:
        print(f"\n  Training Set Quality:")
        print(f"    Accuracy:              {tracker.history['train_accuracy'][-1]:.4f}")
        print(f"    Silhouette Score:      {tracker.history['train_silhouette'][-1]:.4f}")
        print(f"    Davies-Bouldin Index:  {tracker.history['train_davies_bouldin'][-1]:.4f} (↓)")
        print(f"    Calinski-Harabasz:     {tracker.history['train_calinski_harabasz'][-1]:.2f} (↑)")
        print(f"    Inertia:               {tracker.history['train_inertia'][-1]:.2f} (↓)")
        # NEW: Print fuzzy metrics
        print(f"    Partition Coefficient: {tracker.history['train_pc'][-1]:.4f} (↑)")
        print(f"    Partition Entropy:     {tracker.history['train_pe'][-1]:.4f} (↓)")
        print(f"    Xie-Beni Index:        {tracker.history['train_xie_beni'][-1]:.4f} (↓)")
        print(f"    Fukuyama-Sugeno Index: {tracker.history['train_fukuyama_sugeno'][-1]:.4f} (↓)")

    # Test set metrics
    if max(tracker.history['test_silhouette'], default=0) > 0:
        basis = test_basis or ('offline evaluation; the tracker does not know how TEST '
                               'labels were produced, pass test_basis=<history'
                               "['prediction']['test']>")
        print(f"\n  Test Set Quality ({basis}):")
        print(f"    Accuracy:              {tracker.history['test_accuracy'][-1]:.4f}")
        print(f"    Silhouette Score:      {tracker.history['test_silhouette'][-1]:.4f}")
        print(f"    Davies-Bouldin Index:  {tracker.history['test_davies_bouldin'][-1]:.4f} (↓)")
        print(f"    Calinski-Harabasz:     {tracker.history['test_calinski_harabasz'][-1]:.2f} (↑)")
        print(f"    Inertia:               {tracker.history['test_inertia'][-1]:.2f} (↓)")
        # NEW: Print fuzzy metrics
        print(f"    Partition Coefficient: {tracker.history['test_pc'][-1]:.4f} (↑)")
        print(f"    Partition Entropy:     {tracker.history['test_pe'][-1]:.4f} (↓)")
        print(f"    Xie-Beni Index:        {tracker.history['test_xie_beni'][-1]:.4f} (↓)")
        print(f"    Fukuyama-Sugeno Index: {tracker.history['test_fukuyama_sugeno'][-1]:.4f} (↓)")

    # Communication and performance
    total_comm = tracker.communication.get('total_bytes', sum(tracker.history['communication_bytes'])) / (1024 ** 2)
    total_time = sum(tracker.history['round_time'])
    avg_round_time = np.mean(tracker.history['round_time'])

    print(f"\n  Communication & Performance:")
    print(f"    Total Communication:   {total_comm:.2f} MiB")
    print(f"    Total Training Time:   {total_time:.2f} seconds")
    print(f"    Avg. Round Time:       {avg_round_time:.2f} seconds")
    print(f"    Total Encryption Time: {sum(tracker.history['encryption_time']):.2f} seconds")
    print(f"    Total Aggregation Time: {sum(tracker.history['aggregation_time']):.2f} seconds")

    # Convergence rate
    if len(tracker.history['center_shift_orig']) > 1:
        initial_shift = tracker.history['center_shift_orig'][1] if tracker.history['center_shift_orig'][1] > 0 else \
        tracker.history['center_shift_orig'][0]
        final_shift = tracker.history['center_shift_orig'][-1]
        if initial_shift > 0:
            reduction_rate = (1 - final_shift / initial_shift) * 100
            print(f"\n  Convergence Metrics:")
            print(f"    Center Shift Reduction: {reduction_rate:.2f}%")

    # Accuracy improvement (train and test)
    if len(tracker.history['train_accuracy']) > 1:
        train_initial_acc = tracker.history['train_accuracy'][0]
        train_final_acc = tracker.history['train_accuracy'][-1]
        train_acc_improvement = (train_final_acc - train_initial_acc) * 100
        print(f"    Train Accuracy Δ:       {train_acc_improvement:+.2f}%")

    if len(tracker.history['test_accuracy']) > 1:
        test_initial_acc = tracker.history['test_accuracy'][0]
        test_final_acc = tracker.history['test_accuracy'][-1]
        test_acc_improvement = (test_final_acc - test_initial_acc) * 100
        print(f"    Test Accuracy Δ:        {test_acc_improvement:+.2f}%")

    # Inertia reduction
    if len(tracker.history['train_inertia']) > 1 and tracker.history['train_inertia'][0] > 0:
        train_inertia_reduction = (1 - tracker.history['train_inertia'][-1] / tracker.history['train_inertia'][0]) * 100
        print(f"    Train Inertia Reduction: {train_inertia_reduction:.2f}%")

    if len(tracker.history['test_inertia']) > 1 and tracker.history['test_inertia'][0] > 0:
        test_inertia_reduction = (1 - tracker.history['test_inertia'][-1] / tracker.history['test_inertia'][0]) * 100
        print(f"    Test Inertia Reduction:  {test_inertia_reduction:.2f}%")

    print("\n" + "=" * 70 + "\n")


def clustering_quality(X, y, labels, centers=None):
    """Quality for supplied native labels; never replace spectral labels by a proxy."""
    from scipy.optimize import linear_sum_assignment
    labels = np.asarray(labels, dtype=int)
    X = np.asarray(X)
    if labels.shape != (len(X),):
        raise ValueError("Prediction rows do not match evaluation data")
    valid = 1 < len(np.unique(labels)) < len(labels)
    sil = np.nan
    if valid:
        try:
            sil = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=0)
        except ValueError:
            pass
    metrics = {
        'silhouette': sil,
        'davies_bouldin': davies_bouldin_score(X, labels) if valid else np.nan,
        'calinski_harabasz': calinski_harabasz_score(X, labels) if valid else np.nan,
        'acc': np.nan, 'nmi': np.nan, 'ari': np.nan,
    }
    if y is not None:
        _, true = np.unique(y, return_inverse=True)
        _, predicted = np.unique(labels, return_inverse=True)
        matrix = np.zeros((predicted.max() + 1, true.max() + 1), dtype=np.int64)
        np.add.at(matrix, (predicted, true), 1)
        rows, columns = linear_sum_assignment(-matrix)
        metrics.update(acc=float(matrix[rows, columns].sum() / len(labels)),
                       nmi=float(normalized_mutual_info_score(y, labels)),
                       ari=float(adjusted_rand_score(y, labels)))
    return metrics


CKKS_ENTROPY_METRICS = ('total_bytes', 'uplink_bytes', 'downlink_bytes', 'setup_bytes',
                        'interservice_bytes', 'bytes_per_round',
                        'ciphertext_acc', 'ciphertext_auc', 'ciphertext_advantage',
                        'ciphertext_degenerate_folds')


def _group_uses_ckks(data):
    """True when any run in this group took the encrypted path.

    TenSEAL key generation draws on system entropy, so ciphertext sizes differ
    between two runs with identical seeds. That is a property of the library, not
    instability of the algorithm, and readers must be able to tell the two apart.
    """
    if 'use_ckks' not in data:
        return False
    values = data['use_ckks'].dropna()
    if values.empty:
        return False
    if values.dtype == bool:
        return bool(values.any())
    return bool(values.astype(str).str.lower().isin(('true', '1')).any())


def metric_reproducible(metric, ckks_active):
    """Whether a metric is expected to repeat exactly under the same seed."""
    return not (metric in CKKS_ENTROPY_METRICS and ckks_active)


def summarize_results(frame):
    """Mean, sample SD and t-CI across initialization seeds on each fixed split."""
    import pandas as pd
    from scipy.stats import t
    groups = ['dataset', 'method', 'data_seed', 'dirichlet_beta', 'clients',
              'clusters', 'study', 'study_value']
    rows = []
    for keys, data in frame.groupby(groups, dropna=False, sort=False):
        ckks_active = _group_uses_ckks(data)
        for metric in ('acc', 'nmi', 'ari', 'all_fit_acc', 'all_fit_nmi', 'all_fit_ari',
                       'test_proxy_acc', 'test_proxy_nmi', 'test_proxy_ari',
                       'total_bytes', 'uplink_bytes', 'downlink_bytes', 'setup_bytes',
                       'interservice_bytes', 'bytes_per_round', 'rounds', 'wall_seconds', 'embedding_seconds',
                       'statistics_seconds', 'encryption_seconds', 'aggregation_seconds',
                       'decryption_seconds', 'topology_seconds', 'setup_seconds',
                       'evaluation_seconds', 'privacy_seconds', 'peak_python_bytes',
                       'mia_valid', 'mia_acc', 'mia_balanced_accuracy', 'mia_auc',
                       'mia_tpr', 'mia_tnr', 'mia_cohens_d', 'mia_degenerate_folds',
                       'ciphertext_acc', 'ciphertext_auc', 'ciphertext_advantage',
                       'ciphertext_degenerate_folds', 'public_context_has_secret_key',
                       'upload_size_correlation', 'local_time_size_correlation', 'epsilon', 'delta'):
            if metric not in data:
                continue
            values = data[metric].dropna()
            count = len(values)
            # An all-NaN metric column yields n=0; such a row carries no
            # information and must never reach summary.csv or the paper tables.
            if count == 0:
                continue
            std = float(values.std(ddof=1)) if count > 1 else np.nan
            rows.append(dict(zip(groups, keys), metric=metric, n=count,
                             mean=float(values.mean()) if count else np.nan, std=std,
                             ci95=float(t.ppf(.975, count - 1) * std / np.sqrt(count)) if count > 1 else np.nan,
                             reproducible=metric_reproducible(metric, ckks_active)))
    return pd.DataFrame(rows)


def two_factor(matrix):
    """Two-factor statistics for one metric measured on D splits x S init seeds.

    The unit of inference is the DATA SPLIT, never the individual run: no matter
    how many seeds are run inside a split, sigma_split^2/D never averages away.

    One-way random-effects ANOVA on y_ji = mu + a_j + e_ji gives

        MS_b = S * sum_j (ybar_j - ybar)^2 / (D - 1)
        MS_w = sum_ji (y_ji - ybar_j)^2 / (D (S - 1))
        sigma_init^2  = MS_w
        sigma_split^2 = max((MS_b - MS_w) / S, 0)      # negative truncated to 0

    and the standard error of the grand mean collapses to

        SE(ybar) = sqrt(MS_b / (D * S)),   df = D - 1   (NOT D*S - 1)

    Under Protocol A (S = 1) the components are not identifiable, but SE is
    still correct because MS_b is then just the sample variance of the D
    observations, whose expectation is sigma_split^2 + sigma_init^2.
    """
    y = np.asarray(matrix, dtype=float)
    if y.ndim != 2:
        raise ValueError('two_factor expects a (D, S) matrix')
    n_splits, n_seeds = y.shape
    if n_splits < 2:
        raise ValueError('Two-factor analysis needs D >= 2 data splits; '
                         'a single split cannot estimate sigma_split')
    if n_seeds < 1:
        raise ValueError('Two-factor analysis needs at least one run per split')
    grand = float(y.mean())
    split_means = y.mean(axis=1)
    if n_seeds > 1:
        ms_b = n_seeds * ((split_means - grand) ** 2).sum() / (n_splits - 1)
        ms_w = ((y - split_means[:, None]) ** 2).sum() / (n_splits * (n_seeds - 1))
        var_init = float(ms_w)
        var_split = float(max((ms_b - ms_w) / n_seeds, 0.0))
        decomposable = True
    else:
        # S = 1: MS_b degenerates to the sample variance of the D observations.
        ms_b = float(((split_means - grand) ** 2).sum() / (n_splits - 1))
        var_init = np.nan
        var_split = np.nan
        decomposable = False

    se = float(np.sqrt(ms_b / (n_splits * n_seeds)))
    from scipy.stats import t
    tcrit = float(t.ppf(0.975, n_splits - 1))
    return {
        'n_splits': n_splits, 'n_seeds_per_split': n_seeds,
        'n_runs': int(n_splits * n_seeds), 'mean': grand,
        'sd_split': float(np.sqrt(var_split)) if decomposable else np.nan,
        'sd_init': float(np.sqrt(var_init)) if decomposable else np.nan,
        'rho_split': (float(var_split / (var_split + var_init))
                      if decomposable and (var_split + var_init) > 0 else np.nan),
        'se_total': se, 'df': n_splits - 1,
        'ci95_low': grand - tcrit * se, 'ci95_high': grand + tcrit * se,
        'decomposable': decomposable,
        'note': '' if decomposable else 'S=1: variance components not identifiable; '
                                        'SE and CI still valid (they cover both components)',
    }


CROSS_SPLIT_METRICS = ('acc', 'nmi', 'ari', 'all_fit_acc', 'all_fit_nmi', 'all_fit_ari',
                       'test_proxy_acc', 'test_proxy_nmi', 'test_proxy_ari',
                       'wall_seconds', 'rounds', 'total_bytes')


def two_factor_summary(frame):
    """Aggregate a multi-split result frame with the split as the sampling unit.

    This is the AUTHORITATIVE summary whenever more than one data_seed is run.
    The per-split summary.csv produced by summarize_results() answers a different
    question ("how much does the initialization matter on this one split?") and
    its CI must not be quoted as a method-level interval.
    """
    import pandas as pd
    groups = ['dataset', 'method', 'dirichlet_beta', 'clients', 'clusters', 'study', 'study_value']
    rows = []
    if 'data_seed' not in frame:
        return pd.DataFrame(rows)
    for keys, data in frame.groupby(groups, dropna=False, sort=False):
        per_split_by_metric = {}
        for metric in CROSS_SPLIT_METRICS:
            if metric not in data:
                continue
            per_split = [group[metric].dropna().to_numpy(dtype=float)
                         for _, group in data.groupby('data_seed', sort=True)]
            per_split = [values for values in per_split if len(values)]
            if per_split:
                per_split_by_metric[metric] = per_split
        for metric, per_split in per_split_by_metric.items():
            counts = {len(values) for values in per_split}
            if len(counts) != 1:
                raise ValueError(f'Unbalanced two-factor design for {metric}: '
                                 f'splits hold {sorted(counts)} runs')
            pooled = np.concatenate(per_split)
            if len(per_split) < 2:
                rows.append(dict(zip(groups, keys), metric=metric,
                                 n_splits=len(per_split), n_seeds_per_split=len(per_split[0]),
                                 n_runs=int(pooled.size), mean=float(pooled.mean()),
                                 sd_split=np.nan, sd_init=np.nan, rho_split=np.nan,
                                 se_total=np.nan, df=np.nan, ci95_low=np.nan, ci95_high=np.nan,
                                 decomposable=False,
                                 note='single data split: sigma_split not estimable, no CI defined'))
                continue
            rows.append(dict(zip(groups, keys), metric=metric,
                             **two_factor(np.stack(per_split))))
    return pd.DataFrame(rows)


def evaluate_attack_features(features, target, seed=42, folds=5):
    """Out-of-fold empirical attack; constant hard predictions do not invalidate AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    features, target = np.asarray(features, dtype=float), np.asarray(target, dtype=int)
    if (features.ndim != 2 or target.shape != (len(features),)
            or not np.isfinite(features).all() or folds < 2):
        raise ValueError("Finite attack features, matching binary targets and folds >= 2 are required")
    counts = np.bincount(target, minlength=2) if np.isin(target, [0, 1]).all() else []
    if len(counts) != 2 or min(counts) < folds:
        return {'status': 'insufficient_samples', 'class_counts': list(map(int, counts)), 'folds': folds}
    prediction = np.empty(len(target), dtype=int)
    score = np.empty(len(target), dtype=float)
    fold_results = []
    for fit, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(features, target):
        model = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=seed))
        model.fit(features[fit], target[fit])
        prediction[test] = model.predict(features[test])
        score[test] = model.predict_proba(features[test])[:, 1]
        fold_results.append({'fit_samples': len(fit), 'test_samples': len(test),
                             'acc': float(np.mean(prediction[test] == target[test])),
                             'auc': float(roc_auc_score(target[test], score[test])),
                             'constant_prediction': bool(len(np.unique(prediction[test])) == 1)})
    degenerate = sum(row['constant_prediction'] for row in fold_results)
    tpr = float(np.mean(prediction[target == 1] == 1))
    tnr = float(np.mean(prediction[target == 0] == 0))
    return {
        'status': 'evaluated',
        'prediction_status': ('evaluated' if degenerate == 0 else
                              'degenerate_constant_prediction' if degenerate == folds else 'partially_degenerate'),
        'acc': float(np.mean(prediction == target)), 'balanced_accuracy': (tpr + tnr) / 2,
        'tpr': tpr, 'tnr': tnr, 'auc': float(roc_auc_score(target, score)),
        'folds': folds, 'degenerate_folds': degenerate, 'fold_results': fold_results,
        'class_counts': list(map(int, counts)),
        'note': (f'{folds}-fold out-of-fold predictions; scaler fit inside each attack fold; '
                 'constant threshold predictions retained; AUC uses probability scores; NOT a privacy proof'),
    }


def membership_inference_on_centers(X_members, X_nonmembers, centers, seed=42, max_samples=1000,
                                    folds=5):
    """Balanced membership attack on released original-space centers only."""
    rng = np.random.RandomState(seed)
    count = min(len(X_members), len(X_nonmembers), max_samples)
    if count < max(10, folds):
        return {'status': 'insufficient_samples', 'per_class': count}
    members = X_members[rng.choice(len(X_members), count, replace=False)]
    nonmembers = X_nonmembers[rng.choice(len(X_nonmembers), count, replace=False)]
    features = np.sort(cdist(np.vstack((members, nonmembers)), centers), axis=1)
    target = np.r_[np.ones(count, dtype=int), np.zeros(count, dtype=int)]
    result = evaluate_attack_features(features, target, seed, folds)
    nearest_members, nearest_nonmembers = features[:count, 0], features[count:, 0]
    pooled_sd = np.sqrt((nearest_members.var(ddof=1) + nearest_nonmembers.var(ddof=1)) / 2)
    effect = float(nearest_members.mean() - nearest_nonmembers.mean())
    result.update(per_class=count, attack_test_samples=2 * count,
                  cohens_d=effect / pooled_sd if pooled_sd > 0 else None,
                  effect_status='evaluated' if pooled_sd > 0 else 'zero_pooled_variance',
                  attacker_view='released original-space centers + known member/nonmember reference records',
                  nonmember_scope='held-out TEST records; no raw individual uploaded statistics or secret key')
    return result


def size_leakage_diagnostics(client_sizes, upload_totals, local_seconds, observations):
    """Per-client averages; descriptive single-process timing, not a side-channel guarantee."""
    sizes = np.asarray(client_sizes, dtype=float)
    observations = np.asarray(observations)
    active = observations > 0
    result = {'clients_observed': int(active.sum()),
              'scope': 'per-client mean over observed rounds; serial local timings; no p-values or padding claim'}
    for name, totals in (('upload', upload_totals), ('local_time', local_seconds)):
        values = np.asarray(totals, dtype=float)[active] / observations[active]
        x = sizes[active]
        variable = len(x) >= 3 and np.ptp(x) > 0 and np.ptp(values) > 0
        result[name + '_size_correlation'] = float(np.corrcoef(x, values)[0, 1]) if variable else None
        result[name + '_status'] = 'evaluated' if variable else 'insufficient_or_constant'
    return result