import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, normalized_mutual_info_score,
                             adjusted_rand_score)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'axes.labelsize': 12,
    'axes.titlesize': 13, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.titlesize': 14,
    'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
})


def _safe_show():
    if matplotlib.get_backend() not in ('Agg', 'pdf', 'svg', 'ps', 'template'):
        plt.show()


class FedSCConvergenceTracker:
    """Tracks convergence during FedSC two-stage training."""

    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.history = {
            'round': [], 'Z_shift': [], 'reconstruction_error': [],
            'train_silhouette': [], 'train_davies_bouldin': [],
            'train_calinski_harabasz': [], 'train_accuracy': [],
            'test_silhouette': [], 'test_davies_bouldin': [],
            'test_calinski_harabasz': [], 'test_accuracy': [],
            'dictionary_size': [], 'active_clients': [],
            'avg_coefficient_sparsity': [],
            'communication_bytes': [], 'round_time': [],
            'uplink_bytes': [], 'downlink_bytes': [], 'learning_rate': [],
            'train_nmi': [], 'train_ari': [], 'train_inertia': []
        }
        self.Z_history = []
        self.final_labels = None
        self.similarity_matrix = None
        self.communication = {}
        self.performance = {}
        self.spectral_diagnostics = {}
        self.protocol = 'transductive train labels in concatenated client order; no test predictor'

    def update_stage_I(self, round_idx, Z, prev_Z=None, clients=None,
                       round_time=0, alpha=None, communication_bytes=None,
                       C_list=None, reconstruction_error=np.nan,
                       uplink_bytes=None, downlink_bytes=0, learning_rate=np.nan):
        self.history['round'].append(round_idx)
        self.Z_history.append(Z.copy())
        shift = np.linalg.norm(Z - prev_Z, ord='fro') if prev_Z is not None else 0.0
        self.history['Z_shift'].append(shift)
        d = Z.shape[1] if len(Z.shape) > 1 else 0
        self.history['dictionary_size'].append(d)
        if C_list and len(C_list) > 0:
            sp = [np.sum(np.abs(C) < 1e-6) / C.size for C in C_list]
            self.history['avg_coefficient_sparsity'].append(np.mean(sp))
        else:
            self.history['avg_coefficient_sparsity'].append(np.nan)
        self.history['reconstruction_error'].append(float(reconstruction_error))
        for k in ['train_silhouette','train_davies_bouldin','train_calinski_harabasz',
                  'train_accuracy','test_silhouette','test_davies_bouldin',
                  'test_calinski_harabasz','test_accuracy', 'train_nmi',
                  'train_ari', 'train_inertia']:
            self.history[k].append(np.nan)
        self.history['active_clients'].append(alpha if alpha else 0)
        if communication_bytes is not None:
            self.history['communication_bytes'].append(communication_bytes)
        elif clients:
            self.history['communication_bytes'].append(
                sum(getattr(c, 'communication_bytes', 0) for c in clients))
        else:
            self.history['communication_bytes'].append(np.nan)
        self.history['uplink_bytes'].append(float(uplink_bytes) if uplink_bytes is not None else np.nan)
        self.history['downlink_bytes'].append(float(downlink_bytes))
        self.history['learning_rate'].append(float(learning_rate))
        self.history['round_time'].append(round_time)

    def update_stage_II(self, labels, similarity_matrix=None, stage_II_time=0):
        self.final_labels = labels
        self.similarity_matrix = similarity_matrix
        if not self.history['round']:
            return
        li = len(self.history['round']) - 1
        if self.X_train is not None and len(self.X_train) > 0:
            sil, db, ch = self._get_spectral_metrics(self.X_train, labels)
            labels = np.asarray(labels)
            if len(labels) != len(self.X_train):
                raise ValueError('FedSC tracker labels and X_train must be in concatenated client order')
            acc = self._compute_accuracy(self.y_train, labels) if self.y_train is not None else np.nan
            self.history['train_silhouette'][li] = sil
            self.history['train_davies_bouldin'][li] = db
            self.history['train_calinski_harabasz'][li] = ch
            self.history['train_accuracy'][li] = acc
            centers = {value: np.mean(self.X_train[labels == value], axis=0)
                       for value in np.unique(labels)}
            self.history['train_inertia'][li] = float(sum(np.sum(
                (self.X_train[labels == value] - center) ** 2)
                for value, center in centers.items()))
            if self.y_train is not None:
                self.history['train_nmi'][li] = normalized_mutual_info_score(self.y_train, labels)
                self.history['train_ari'][li] = adjusted_rand_score(self.y_train, labels)
        self.performance['stage_II_time'] = float(stage_II_time)

    def _get_spectral_metrics(self, X, labels):
        if X is None or len(X) == 0 or labels is None: return np.nan, np.nan, np.nan
        uniq = np.unique(labels)
        if len(uniq) <= 1 or len(uniq) >= len(X): return np.nan, np.nan, np.nan
        try:
            return (silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=0),
                    davies_bouldin_score(X, labels),
                    calinski_harabasz_score(X, labels))
        except ValueError:
            return np.nan, np.nan, np.nan

    def _compute_accuracy(self, y_true, y_pred):
        if y_true is None or len(y_true) == 0: return np.nan
        from scipy.optimize import linear_sum_assignment
        _, true_indices = np.unique(y_true, return_inverse=True)
        _, predicted_indices = np.unique(y_pred, return_inverse=True)
        if len(true_indices) != len(predicted_indices):
            raise ValueError('labels must have equal length')
        cm = np.zeros((true_indices.max() + 1, predicted_indices.max() + 1), dtype=np.int64)
        np.add.at(cm, (true_indices, predicted_indices), 1)
        r, c = linear_sum_assignment(-cm)
        return float(cm[r, c].sum() / len(y_true))

    def get_final_metrics(self):
        if not self.history['round']: return {}
        return {
            'final_Z_shift': self.history['Z_shift'][-1],
            'train_silhouette': self.history['train_silhouette'][-1],
            'train_accuracy': self.history['train_accuracy'][-1],
            'train_nmi': self.history['train_nmi'][-1],
            'train_ari': self.history['train_ari'][-1],
            'train_davies_bouldin': self.history['train_davies_bouldin'][-1],
            'train_calinski_harabasz': self.history['train_calinski_harabasz'][-1],
            'train_inertia': self.history['train_inertia'][-1],
            'total_comm_mb': self.communication.get('total_bytes',
                np.nansum(self.history['communication_bytes'])) / (1024 ** 2),
            'total_time': self.performance.get('total_training_time',
                np.nansum(self.history['round_time'])),
            'stage_II_time': self.performance.get('stage_II_time', np.nan)
        }


def plot_fedsc_convergence(tracker, method_name='FedSC', save_path=None):
    """Eight measured quality/resource panels, following the Ours tracker layout."""
    if tracker is None or not tracker.history['round']:
        print("No convergence tracker available.")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    rounds = tracker.history['round']

    # (a) Dictionary Shift
    ax1 = fig.add_subplot(gs[0, :2])
    raw = np.nan_to_num(np.array(tracker.history['Z_shift']), nan=0.0, posinf=0.0, neginf=0.0)
    ax1.semilogy(rounds, np.maximum(raw, 1e-15), 'o-', lw=2, ms=6, label='Z Shift', color='#2E86AB')
    ax1.set_xlabel('Training Round'); ax1.set_ylabel('Z Shift (log)')
    ax1.set_title('(a) Stage I: Dictionary Convergence', fontweight='bold')
    ax1.legend(loc='upper right'); ax1.grid(True, alpha=0.3, linestyle='--')

    # (b) Equation (8) objective (mean over participating clients).
    ax2 = fig.add_subplot(gs[0, 2])
    objective = np.asarray(tracker.history['reconstruction_error'], dtype=float)
    if np.isfinite(objective).any():
        ax2.plot(rounds, objective, 'D-', lw=2, ms=5, color='#F18F01')
    else: ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_xlabel('Training Round'); ax2.set_title('(b) Reconstruction Objective', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # (c) Silhouette
    ax3 = fig.add_subplot(gs[1, 0])
    if np.isfinite(tracker.history['train_silhouette']).any():
        ax3.plot(rounds, tracker.history['train_silhouette'], 'o-', lw=2, ms=6, color='#006D77', label='Train')
        ax3.set_ylim([-0.1, 1.0]); ax3.axhline(y=0, color='gray', ls='--', alpha=0.5)
        ax3.legend(loc='lower right')
    else: ax3.text(0.5, 0.5, 'After Stage II', ha='center', va='center', transform=ax3.transAxes, style='italic')
    ax3.set_xlabel('Training Round'); ax3.set_title('(c) Cluster Separation', fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # (d) Davies-Bouldin
    ax4 = fig.add_subplot(gs[1, 1])
    if np.isfinite(tracker.history['train_davies_bouldin']).any():
        ax4.plot(rounds, tracker.history['train_davies_bouldin'], 's-', lw=2, ms=6, color='#A23B72', label='Train')
        ax4.legend(loc='upper right')
    else: ax4.text(0.5, 0.5, 'After Stage II', ha='center', va='center', transform=ax4.transAxes, style='italic')
    ax4.set_xlabel('Training Round'); ax4.set_title('(d) Cluster Compactness (↓)', fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')

    # (e) Accuracy
    ax5 = fig.add_subplot(gs[1, 2])
    if np.isfinite(tracker.history['train_accuracy']).any():
        for key, label, color in [('train_accuracy', 'ACC', '#006D77'),
                                  ('train_nmi', 'NMI', '#2E86AB'),
                                  ('train_ari', 'ARI', '#A23B72')]:
            ax5.plot(rounds, tracker.history[key], 'o-', lw=2, ms=6,
                     color=color, label=label)
        ax5.set_ylim([0, 1.0]); ax5.legend(loc='lower right')
        ax5.text(0.05, 0.95, f"Final: {tracker.history['train_accuracy'][-1]:.3f}",
                 transform=ax5.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    else: ax5.text(0.5, 0.5, 'After Stage II', ha='center', va='center', transform=ax5.transAxes, style='italic')
    ax5.set_xlabel('Training Round'); ax5.set_title('(e) Stage II: ACC / NMI / ARI', fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--')

    # (f) Communication
    ax6 = fig.add_subplot(gs[2, 0])
    comm = [v for v in tracker.history['communication_bytes'] if v > 0]
    if comm:
        comm_mb = [c / (1024 ** 2) for c in tracker.history['communication_bytes']]
        ax6.bar(rounds, comm_mb, color='#06A77D', alpha=0.7, edgecolor='black')
        total_mb = tracker.communication.get('total_bytes', sum(comm) if comm else 0) / (1024 ** 2)
        ax6.text(0.95, 0.95, f"All stages: {total_mb:.2f} MiB",
                 transform=ax6.transAxes, fontsize=9, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    else: ax6.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_xlabel('Training Round'); ax6.set_title('(f) Stage I Logical Payload', fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--', axis='y')

    # (g) Active Clients + Dict Size
    ax7 = fig.add_subplot(gs[2, 1])
    if max(tracker.history['active_clients']) > 0:
        ax7t = ax7.twinx()
        ax7.plot(rounds, tracker.history['active_clients'], 'o-', lw=2, ms=5, color='#FF006E')
        ax7.set_ylabel('Active Clients', color='#FF006E'); ax7.tick_params(axis='y', labelcolor='#FF006E')
        if max(tracker.history['dictionary_size']) > 0:
            ax7t.plot(rounds, tracker.history['dictionary_size'], 's-', lw=2, ms=5, color='#8338EC')
            ax7t.set_ylabel('Dict Size (d)', color='#8338EC'); ax7t.tick_params(axis='y', labelcolor='#8338EC')
    else: ax7.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_xlabel('Training Round'); ax7.set_title('(g) Training Configuration', fontweight='bold')
    ax7.grid(True, alpha=0.3, linestyle='--')

    # (h) Round Time
    ax8 = fig.add_subplot(gs[2, 2])
    if max(tracker.history['round_time']) > 0:
        ax8.plot(rounds, tracker.history['round_time'], 'o-', lw=2, ms=6, color='#FB5607')
        ax8.axhline(y=np.mean(tracker.history['round_time']), color='green', ls='--', alpha=0.6)
        ax8.text(rounds[-1], np.mean(tracker.history['round_time']) * 1.1,
                 f"Avg: {np.mean(tracker.history['round_time']):.2f}s", fontsize=9, color='green')
    else: ax8.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax8.transAxes)
    ax8.set_xlabel('Training Round'); ax8.set_title('(h) Stage I Round Time', fontweight='bold')
    if tracker.performance:
        ax8.text(0.03, 0.97, f"Stage II: {tracker.performance.get('stage_II_time', np.nan):.2f}s\n"
                 f"Training: {tracker.performance.get('total_training_time', np.nan):.2f}s",
                 transform=ax8.transAxes, va='top', fontsize=9)
    ax8.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(f'{method_name}: Convergence Analysis (Two-Stage)',
                 fontsize=16, fontweight='bold', y=0.995)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence summary saved to: {save_path}")
    _safe_show()
    return fig


def print_fedsc_summary(tracker, method_name='FedSC'):
    if tracker is None:
        print("No tracker."); return
    print("\n" + "=" * 70)
    print(f"{method_name.upper()} - CONVERGENCE SUMMARY")
    print("=" * 70)
    print(f"\n  Rounds: {len(tracker.history['round'])}")
    if tracker.history['Z_shift']:
        print(f"  Final Z Shift: {tracker.history['Z_shift'][-1]:.6f}")
        if len(tracker.history['Z_shift']) > 1 and tracker.history['Z_shift'][1] > 0:
            r = (1 - tracker.history['Z_shift'][-1] / tracker.history['Z_shift'][1]) * 100
            print(f"  Shift Reduction: {r:.1f}%")
    if tracker.history['round'] and np.isfinite(tracker.history['train_accuracy'][-1]):
        print(f"\n  Clustering Quality (Stage II):")
        print(f"    Accuracy:  {tracker.history['train_accuracy'][-1]:.4f}")
        print(f"    Silhouette: {tracker.history['train_silhouette'][-1]:.4f}")
        print(f"    DB Index:   {tracker.history['train_davies_bouldin'][-1]:.4f}")
        print(f"    NMI:        {tracker.history['train_nmi'][-1]:.4f}")
        print(f"    ARI:        {tracker.history['train_ari'][-1]:.4f}")
    total_comm = tracker.communication.get('total_bytes',
        np.nansum(tracker.history['communication_bytes'])) / (1024 ** 2)
    total_time = tracker.performance.get('total_training_time', sum(tracker.history['round_time']))
    print(f"\n  Logical Communication: {total_comm:.2f} MiB")
    print(f"  Total Time:          {total_time:.2f}s")
    print(f"  Protocol:            {tracker.protocol}")
    print("  Security:            no encryption; perturbation is not formal DP")
    print("=" * 70 + "\n")
