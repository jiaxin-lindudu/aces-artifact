import warnings
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

warnings.filterwarnings('ignore')

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
# K-Means Convergence Tracking Class
# ============================================================================

class KMeansConvergenceTracker:
    """
    Tracks convergence metrics during federated k-means training.
    Adapted for MA and GS methods - no embedding space or Procrustes alignment.
    """

    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Initialize tracker with train/test data.

        Args:
            X_train: Training data for computing metrics
            y_train: Training labels (for accuracy computation)
            X_test: Test data for computing metrics
            y_test: Test labels (for accuracy computation)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Core history
        self.history = {
            'round': [],
            # Center movement
            'center_shift': [],
            # TRAIN set metrics
            'train_inertia': [],
            'train_silhouette': [],
            'train_davies_bouldin': [],
            'train_calinski_harabasz': [],
            'train_accuracy': [],
            # TEST set metrics
            'test_inertia': [],
            'test_silhouette': [],
            'test_davies_bouldin': [],
            'test_calinski_harabasz': [],
            'test_accuracy': [],
            # Method-specific metrics
            'avg_local_iterations': [],  # Average Q per client
            'active_clients': [],  # Number of active clients
            # Resources
            'communication_bytes': [],
            'round_time': []
        }

        # Store centers history for visualization
        self.centers_history = []

    @staticmethod
    def _get_labels(X, centers):
        """Assign each point to nearest cluster center."""
        if len(centers) == 0 or len(X) == 0:
            return np.array([])
        distances = cdist(X, centers, metric='euclidean')
        return np.argmin(distances, axis=1)

    def _get_clustering_metrics(self, X, centers):
        """Return inertia, silhouette, DB, CH metrics for X and given centers."""
        if X is None or len(X) == 0 or centers is None or len(centers) == 0:
            return 0.0, 0.0, 0.0, 0.0

        labels = self._get_labels(X, centers)

        # Inertia
        inertia = np.sum((X - centers[labels]) ** 2)

        uniq = np.unique(labels)
        # Only compute clustering metrics when more than 1 cluster is detected
        if len(uniq) > 1:
            try:
                sil = silhouette_score(X, labels)
                db = davies_bouldin_score(X, labels)
                ch = calinski_harabasz_score(X, labels)
            except:
                sil, db, ch = 0.0, 0.0, 0.0
        else:
            sil, db, ch = 0.0, 0.0, 0.0

        return inertia, sil, db, ch

    def _compute_accuracy(self, y_true, y_pred):
        """
        Compute clustering accuracy using Hungarian algorithm.

        Args:
            y_true: True labels
            y_pred: Predicted cluster labels

        Returns:
            accuracy: Clustering accuracy score
        """
        if y_true is None or len(y_true) == 0:
            return 0.0

        try:
            from scipy.optimize import linear_sum_assignment

            # Create confusion matrix
            n_clusters = max(max(y_true), max(y_pred)) + 1
            confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)

            for i in range(len(y_true)):
                confusion_matrix[y_true[i], y_pred[i]] += 1

            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

            # Calculate accuracy
            accuracy = confusion_matrix[row_ind, col_ind].sum() / len(y_true)
            return accuracy
        except:
            return 0.0

    def update(self, round_idx, centers, clients=None,
               prev_centers=None, round_time=0,
               Q=None, alpha=None, communication_bytes=None):
        """
        Update tracking metrics for current round.

        Args:
            round_idx: Current round number
            centers: Current cluster centers
            clients: List of client instances (optional, for resource tracking)
            prev_centers: Previous round centers
            round_time: Time taken for this round
            Q: Number of local iterations (optional)
            alpha: Number of active clients (optional)
            communication_bytes: Total communication in bytes (optional)
        """
        self.history['round'].append(round_idx)

        # Store centers for visualization
        self.centers_history.append(centers.copy())

        # Center shift
        if prev_centers is not None:
            shift = np.linalg.norm(centers - prev_centers)
        else:
            shift = 0.0
        self.history['center_shift'].append(shift)

        # Metrics for train and test
        for X, y, lbl in [(self.X_train, self.y_train, 'train'),
                          (self.X_test, self.y_test, 'test')]:
            if X is not None and len(X) > 0:
                # Clustering quality metrics
                inertia, sil, db, ch = self._get_clustering_metrics(X, centers)

                # Accuracy metric (requires true labels)
                if y is not None:
                    pred_labels = self._get_labels(X, centers)
                    acc = self._compute_accuracy(y, pred_labels)
                else:
                    acc = 0.0
            else:
                inertia, sil, db, ch, acc = 0.0, 0.0, 0.0, 0.0, 0.0

            self.history[f'{lbl}_inertia'].append(inertia)
            self.history[f'{lbl}_silhouette'].append(sil)
            self.history[f'{lbl}_davies_bouldin'].append(db)
            self.history[f'{lbl}_calinski_harabasz'].append(ch)
            self.history[f'{lbl}_accuracy'].append(acc)

        # Method-specific metrics
        self.history['avg_local_iterations'].append(Q if Q is not None else 0)
        self.history['active_clients'].append(alpha if alpha is not None else 0)

        # Communication and resource tracking
        if communication_bytes is not None:
            self.history['communication_bytes'].append(communication_bytes)
        elif clients is not None:
            total_comm = sum(getattr(c, 'communication_bytes', 0) for c in clients)
            self.history['communication_bytes'].append(total_comm)
        else:
            # Estimate based on center size (K centers, M dimensions, 8 bytes per float)
            K, M = centers.shape
            estimated_comm = K * M * 8 * (alpha if alpha else 1)
            self.history['communication_bytes'].append(estimated_comm)

        self.history['round_time'].append(round_time)


# ============================================================================
# Plotting Functions for K-Means
# ============================================================================

def plot_kmeans_convergence_summary(tracker, method_name='Federated K-Means', save_path=None):
    """
    Create a comprehensive convergence summary plot for federated k-means.

    Args:
        tracker: KMeansConvergenceTracker instance
        method_name: Name of the method (e.g., 'Model Averaging', 'Gradient Sharing')
        save_path: Path to save figure (optional)
    """
    if tracker is None:
        print("No convergence tracker available for plotting.")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    rounds = tracker.history['round']

    # -------------------------------------------------------------------------
    # (a) Center Shift (Primary Convergence Metric)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.semilogy(rounds, tracker.history['center_shift'],
                 'o-', linewidth=2, markersize=6, label='Center Shift', color='#2E86AB')
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Center Shift (log scale)')
    ax1.set_title('(a) Convergence: Center Movement', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add convergence annotation
    if len(rounds) > 5 and max(tracker.history['center_shift']) > 0:
        final_shift = tracker.history['center_shift'][-1]
        ax1.axhline(y=final_shift, color='red', linestyle=':', alpha=0.5)
        ax1.text(rounds[-1], final_shift * 1.5, f'Final: {final_shift:.2e}',
                 fontsize=9, color='red')

    # -------------------------------------------------------------------------
    # (b) Active Clients & Local Iterations
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    if max(tracker.history['active_clients']) > 0:
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(rounds, tracker.history['active_clients'],
                         'o-', linewidth=2, markersize=5, color='#F18F01', label='Active Clients')
        ax2.set_ylabel('Active Clients', color='#F18F01')
        ax2.tick_params(axis='y', labelcolor='#F18F01')

        if max(tracker.history['avg_local_iterations']) > 0:
            line2 = ax2_twin.plot(rounds, tracker.history['avg_local_iterations'],
                                  's-', linewidth=2, markersize=5, color='#A23B72', label='Local Iterations (Q)')
            ax2_twin.set_ylabel('Local Iterations', color='#A23B72')
            ax2_twin.tick_params(axis='y', labelcolor='#A23B72')

        ax2.set_xlabel('Training Round')
        ax2.set_title('(b) Training Configuration', fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('(b) Training Configuration', fontweight='bold')

    # -------------------------------------------------------------------------
    # (c) Inertia (Train vs Test)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    if max(tracker.history['train_inertia']) > 0 or max(tracker.history['test_inertia']) > 0:
        ax3.plot(rounds, tracker.history['train_inertia'],
                 'o-', linewidth=2, markersize=6, label='Train', color='#006D77')
        ax3.plot(rounds, tracker.history['test_inertia'],
                 's-', linewidth=2, markersize=6, label='Test', color='#E63946')
        ax3.set_xlabel('Training Round')
        ax3.set_ylabel('Inertia')
        ax3.set_title('(c) Within-Cluster Variance', fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, linestyle='--')
    else:
        ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('(c) Within-Cluster Variance', fontweight='bold')

    # -------------------------------------------------------------------------
    # (d) Silhouette Score (Train vs Test)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    if max(tracker.history['train_silhouette']) > 0 or max(tracker.history['test_silhouette']) > 0:
        ax4.plot(rounds, tracker.history['train_silhouette'],
                 'o-', linewidth=2, markersize=6, label='Train', color='#006D77')
        ax4.plot(rounds, tracker.history['test_silhouette'],
                 's-', linewidth=2, markersize=6, label='Test', color='#E63946')
        ax4.set_xlabel('Training Round')
        ax4.set_ylabel('Silhouette Score')
        ax4.set_title('(d) Cluster Separation', fontweight='bold')
        ax4.set_ylim([-0.1, 1.0])
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.legend(loc='lower right')
        ax4.grid(True, alpha=0.3, linestyle='--')

        # Quality indicator for test set
        if len(tracker.history['test_silhouette']) > 0:
            final_sil = tracker.history['test_silhouette'][-1]
            if final_sil > 0.5:
                quality = 'Excellent'
                color = 'green'
            elif final_sil > 0.25:
                quality = 'Good'
                color = 'orange'
            else:
                quality = 'Poor'
                color = 'red'
            ax4.text(0.05, 0.05, f'Test Quality: {quality}',
                     transform=ax4.transAxes, fontsize=9, color=color,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('(d) Cluster Separation', fontweight='bold')

    # -------------------------------------------------------------------------
    # (e) Accuracy (Train vs Test)
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    if max(tracker.history['train_accuracy']) > 0 or max(tracker.history['test_accuracy']) > 0:
        ax5.plot(rounds, tracker.history['train_accuracy'],
                 'o-', linewidth=2, markersize=6, label='Train', color='#006D77')
        ax5.plot(rounds, tracker.history['test_accuracy'],
                 's-', linewidth=2, markersize=6, label='Test', color='#E63946')
        ax5.set_xlabel('Training Round')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('(e) Clustering Accuracy', fontweight='bold')
        ax5.set_ylim([0, 1.0])
        ax5.legend(loc='lower right')
        ax5.grid(True, alpha=0.3, linestyle='--')

        # Add final accuracy annotation
        if len(tracker.history['test_accuracy']) > 0:
            final_test_acc = tracker.history['test_accuracy'][-1]
            ax5.text(0.05, 0.95, f'Final Test Acc: {final_test_acc:.3f}',
                     transform=ax5.transAxes, fontsize=9,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    else:
        ax5.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('(e) Clustering Accuracy', fontweight='bold')

    # -------------------------------------------------------------------------
    # (f) Davies-Bouldin Index (Train vs Test)
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 0])
    if max(tracker.history['train_davies_bouldin']) > 0 or max(tracker.history['test_davies_bouldin']) > 0:
        ax6.plot(rounds, tracker.history['train_davies_bouldin'],
                 'o-', linewidth=2, markersize=6, label='Train', color='#006D77')
        ax6.plot(rounds, tracker.history['test_davies_bouldin'],
                 's-', linewidth=2, markersize=6, label='Test', color='#E63946')
        ax6.set_xlabel('Training Round')
        ax6.set_ylabel('Davies-Bouldin Index')
        ax6.set_title('(f) Cluster Compactness (lower is better)', fontweight='bold')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3, linestyle='--')
    else:
        ax6.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('(f) Davies-Bouldin Index', fontweight='bold')

    # -------------------------------------------------------------------------
    # (g) Communication Overhead
    # -------------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[2, 1])
    if len(tracker.history['communication_bytes']) > 0:
        comm_mb = [c / (1024 ** 2) for c in tracker.history['communication_bytes']]
        ax7.bar(rounds, comm_mb, color='#06A77D', alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Training Round')
        ax7.set_ylabel('Communication (MB)')
        ax7.set_title('(g) Communication Overhead', fontweight='bold')
        ax7.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Total communication
        total_comm = sum(comm_mb)
        ax7.text(0.95, 0.95, f'Total: {total_comm:.2f} MB',
                 transform=ax7.transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    else:
        ax7.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('(g) Communication Overhead', fontweight='bold')

    # -------------------------------------------------------------------------
    # (h) Round Time
    # -------------------------------------------------------------------------
    ax8 = fig.add_subplot(gs[2, 2])
    if len(tracker.history['round_time']) > 0:
        ax8.plot(rounds, tracker.history['round_time'],
                 'o-', linewidth=2, markersize=6, color='#FB5607')
        ax8.set_xlabel('Training Round')
        ax8.set_ylabel('Time (seconds)')
        ax8.set_title('(h) Total Round Time', fontweight='bold')
        ax8.grid(True, alpha=0.3, linestyle='--')

        # Average time
        avg_time = np.mean(tracker.history['round_time'])
        ax8.axhline(y=avg_time, color='green', linestyle='--', alpha=0.6)
        ax8.text(rounds[-1], avg_time * 1.1, f'Avg: {avg_time:.2f}s',
                 fontsize=9, color='green')
    else:
        ax8.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('(h) Total Round Time', fontweight='bold')

    # Overall title
    fig.suptitle(f'{method_name}: Convergence Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Convergence summary saved to: {save_path}")

    plt.tight_layout()
    plt.show()


def print_kmeans_convergence_summary(tracker, method_name='Federated K-Means'):
    """
    Print a text summary of convergence metrics.

    Args:
        tracker: KMeansConvergenceTracker instance
        method_name: Name of the method
    """
    if tracker is None:
        print("No convergence tracker available.")
        return

    print("\n" + "=" * 70)
    print(f"{method_name.upper()} - CONVERGENCE SUMMARY")
    print("=" * 70)

    print(f"\n  Training Rounds: {len(tracker.history['round'])}")
    if len(tracker.history['center_shift']) > 0:
        print(f"  Final Center Shift: {tracker.history['center_shift'][-1]:.6f}")

    # Training set metrics
    if max(tracker.history['train_silhouette']) > 0:
        print(f"\n  Training Set Quality:")
        if max(tracker.history['train_accuracy']) > 0:
            print(f"    Accuracy:              {tracker.history['train_accuracy'][-1]:.4f}")
        print(f"    Silhouette Score:      {tracker.history['train_silhouette'][-1]:.4f}")
        print(f"    Davies-Bouldin Index:  {tracker.history['train_davies_bouldin'][-1]:.4f} (↓)")
        print(f"    Calinski-Harabasz:     {tracker.history['train_calinski_harabasz'][-1]:.2f} (↑)")
        print(f"    Inertia:               {tracker.history['train_inertia'][-1]:.2f} (↓)")

    # Test set metrics
    if max(tracker.history['test_silhouette']) > 0:
        print(f"\n  Test Set Quality:")
        if max(tracker.history['test_accuracy']) > 0:
            print(f"    Accuracy:              {tracker.history['test_accuracy'][-1]:.4f}")
        print(f"    Silhouette Score:      {tracker.history['test_silhouette'][-1]:.4f}")
        print(f"    Davies-Bouldin Index:  {tracker.history['test_davies_bouldin'][-1]:.4f} (↓)")
        print(f"    Calinski-Harabasz:     {tracker.history['test_calinski_harabasz'][-1]:.2f} (↑)")
        print(f"    Inertia:               {tracker.history['test_inertia'][-1]:.2f} (↓)")

    # Communication and performance
    if len(tracker.history['communication_bytes']) > 0:
        total_comm = sum(tracker.history['communication_bytes']) / (1024 ** 2)
        print(f"\n  Communication & Performance:")
        print(f"    Total Communication:   {total_comm:.2f} MB")

    if len(tracker.history['round_time']) > 0:
        total_time = sum(tracker.history['round_time'])
        avg_round_time = np.mean(tracker.history['round_time'])
        print(f"    Total Training Time:   {total_time:.2f} seconds")
        print(f"    Avg. Round Time:       {avg_round_time:.2f} seconds")

    # Method-specific metrics
    if max(tracker.history['active_clients']) > 0:
        avg_active = np.mean(tracker.history['active_clients'])
        print(f"\n  Method-Specific Metrics:")
        print(f"    Avg. Active Clients:   {avg_active:.1f}")

    if max(tracker.history['avg_local_iterations']) > 0:
        avg_Q = np.mean(tracker.history['avg_local_iterations'])
        print(f"    Avg. Local Iterations: {avg_Q:.1f}")

    # Convergence rate
    if len(tracker.history['center_shift']) > 1:
        initial_shift = tracker.history['center_shift'][1] if len(tracker.history['center_shift']) > 1 and \
                                                              tracker.history['center_shift'][1] > 0 else \
        tracker.history['center_shift'][0]
        final_shift = tracker.history['center_shift'][-1]
        if initial_shift > 0:
            reduction_rate = (1 - final_shift / initial_shift) * 100
            print(f"\n  Convergence Metrics:")
            print(f"    Center Shift Reduction: {reduction_rate:.2f}%")

    # Accuracy improvement (train and test)
    if len(tracker.history['train_accuracy']) > 1 and max(tracker.history['train_accuracy']) > 0:
        train_initial_acc = tracker.history['train_accuracy'][0]
        train_final_acc = tracker.history['train_accuracy'][-1]
        train_acc_improvement = (train_final_acc - train_initial_acc) * 100
        print(f"    Train Accuracy Δ:       {train_acc_improvement:+.2f}%")

    if len(tracker.history['test_accuracy']) > 1 and max(tracker.history['test_accuracy']) > 0:
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


def plot_kmeans_comparison(trackers_dict, save_path=None):
    """
    Compare multiple k-means methods (e.g., MA vs GS vs Centralized).

    Args:
        trackers_dict: Dictionary of {method_name: KMeansConvergenceTracker}
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#006D77', '#E63946', '#9B5DE5']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    # -------------------------------------------------------------------------
    # 1. Center Shift Comparison
    # -------------------------------------------------------------------------
    for idx, (name, tracker) in enumerate(trackers_dict.items()):
        if len(tracker.history['center_shift']) > 0:
            axes[0].semilogy(tracker.history['round'],
                             tracker.history['center_shift'],
                             marker=markers[idx], linewidth=2, markersize=5,
                             label=name, color=colors[idx])
    axes[0].set_xlabel('Training Round')
    axes[0].set_ylabel('Center Shift (log scale)')
    axes[0].set_title('(a) Convergence Speed', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, linestyle='--')

    # -------------------------------------------------------------------------
    # 2. Test Silhouette Score Comparison
    # -------------------------------------------------------------------------
    for idx, (name, tracker) in enumerate(trackers_dict.items()):
        if max(tracker.history['test_silhouette']) > 0:
            axes[1].plot(tracker.history['round'],
                         tracker.history['test_silhouette'],
                         marker=markers[idx], linewidth=2, markersize=5,
                         label=name, color=colors[idx])
    axes[1].set_xlabel('Training Round')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('(b) Test Clustering Quality', fontweight='bold')
    axes[1].set_ylim([-0.1, 1.0])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, linestyle='--')

    # -------------------------------------------------------------------------
    # 3. Test Inertia Comparison
    # -------------------------------------------------------------------------
    for idx, (name, tracker) in enumerate(trackers_dict.items()):
        if max(tracker.history['test_inertia']) > 0:
            axes[2].plot(tracker.history['round'],
                         tracker.history['test_inertia'],
                         marker=markers[idx], linewidth=2, markersize=5,
                         label=name, color=colors[idx])
    axes[2].set_xlabel('Training Round')
    axes[2].set_ylabel('Inertia')
    axes[2].set_title('(c) Test Within-Cluster Variance', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, linestyle='--')

    # -------------------------------------------------------------------------
    # 4. Communication Comparison
    # -------------------------------------------------------------------------
    method_names = list(trackers_dict.keys())
    total_comms = [sum(t.history['communication_bytes']) / (1024 ** 2)
                   for t in trackers_dict.values()]
    bars = axes[3].bar(method_names, total_comms, color=colors[:len(method_names)],
                       alpha=0.7, edgecolor='black')
    axes[3].set_ylabel('Total Communication (MB)')
    axes[3].set_title('(d) Communication Overhead', fontweight='bold')
    axes[3].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[3].tick_params(axis='x', rotation=15)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[3].text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontsize=9)

    # -------------------------------------------------------------------------
    # 5. Total Time Comparison
    # -------------------------------------------------------------------------
    total_times = [sum(t.history['round_time']) for t in trackers_dict.values()]
    bars = axes[4].bar(method_names, total_times, color=colors[:len(method_names)],
                       alpha=0.7, edgecolor='black')
    axes[4].set_ylabel('Total Time (seconds)')
    axes[4].set_title('(e) Computational Cost', fontweight='bold')
    axes[4].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[4].tick_params(axis='x', rotation=15)

    for bar in bars:
        height = bar.get_height()
        axes[4].text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}s',
                     ha='center', va='bottom', fontsize=9)

    # -------------------------------------------------------------------------
    # 6. Final Quality Metrics (Test Set)
    # -------------------------------------------------------------------------
    final_metrics = {name: [] for name in method_names}
    metric_labels = ['Silhouette', 'Accuracy', 'DB Index\n(inverted)']

    for name, tracker in trackers_dict.items():
        # Silhouette
        final_metrics[name].append(
            tracker.history['test_silhouette'][-1] if len(tracker.history['test_silhouette']) > 0 else 0
        )
        # Accuracy
        final_metrics[name].append(
            tracker.history['test_accuracy'][-1] if len(tracker.history['test_accuracy']) > 0 else 0
        )
        # DB index (inverted)
        db = tracker.history['test_davies_bouldin'][-1] if len(tracker.history['test_davies_bouldin']) > 0 else 0
        final_metrics[name].append(1.0 / (db + 0.1) if db > 0 else 0)

    x = np.arange(len(metric_labels))
    width = 0.8 / len(method_names)

    for idx, (name, values) in enumerate(final_metrics.items()):
        offset = (idx - len(method_names) / 2 + 0.5) * width
        axes[5].bar(x + offset, values, width, label=name,
                    color=colors[idx], alpha=0.7, edgecolor='black')

    axes[5].set_ylabel('Score')
    axes[5].set_title('(f) Final Test Quality Metrics', fontweight='bold')
    axes[5].set_xticks(x)
    axes[5].set_xticklabels(metric_labels, fontsize=9)
    axes[5].legend(loc='upper left', fontsize=9)
    axes[5].grid(True, alpha=0.3, linestyle='--', axis='y')

    fig.suptitle('Federated K-Means: Method Comparison',
                 fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {save_path}")

    plt.tight_layout()
    plt.show()