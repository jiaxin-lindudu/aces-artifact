"""Ours-style history, quality and resource tracking for one-shot k-FED.

The complete shared tracker provides train/test ACC/NMI/ARI, bounded-sample
silhouette, native labels, timing, communication and center histories. k-FED
adds one real communication-stage record, not a fabricated 50-round curve.
"""
import numpy as np

from baselines.paper_common import PaperTracker


class KFedTracker(PaperTracker):
    def __init__(self):
        super().__init__("k-FED")
        self.method_name = "k-FED"
        self.local_stage_time = 0.0
        self.server_stage_time = 0.0
        self.metrics = {}
        self.privacy = {"status": "no_formal_dp", "cryptographic_protection": False}
        self.privacy_results = self.privacy.copy()

    # ========================================================================
    # Compatibility entry point and native FIT evaluation
    # ========================================================================
    def update(self, round_idx, centers_orig, X_train=None, y_train=None,
               X_test=None, y_test=None, predictor=None, prev_centers=None,
               train_labels=None, stage="one_shot", communication_bytes=None,
               uplink_bytes=None, downlink_bytes=0, statistics_time=0.0,
               aggregation_time=0.0, round_time=0.0, encryption_time=0.0,
               objective=np.nan):
        if uplink_bytes is None:
            uplink_bytes = int(communication_bytes or 0) - int(downlink_bytes)
        if uplink_bytes < 0 or downlink_bytes < 0:
            raise ValueError("communication payloads must be nonnegative")
        self.record(round_idx, centers_orig, prev_centers,
            x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test,
            predictor=predictor, train_labels=train_labels,
            uplink=uplink_bytes, downlink=downlink_bytes,
            logical=uplink_bytes + downlink_bytes,
            times=dict(statistics_time=statistics_time, aggregation_time=aggregation_time,
                encryption_time=encryption_time, round_time=round_time),
            objective=objective, stage=stage)

    def final_metrics(self):
        if not self.history["round"]:
            return {}
        return {split: {("acc" if metric == "accuracy" else metric):
            self.history[f"{split}_{metric}"][-1]
            for metric in ("accuracy", "nmi", "ari", "inertia", "silhouette",
                           "davies_bouldin", "calinski_harabasz")}
            for split in ("train", "test")}


def plot_kfed_convergence_summary(tracker, save_path=None):
    if tracker is None or not tracker.history["round"]:
        print("No k-FED training history available.")
        return None
    if save_path is not None:
        return tracker.plot_convergence_summary(save_path)
    # Preserve the no-path plotting entry without writing temporary artifacts.
    import matplotlib.pyplot as plt
    h = tracker.history
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, metric in zip(axes.flat[:3], ("accuracy", "nmi", "ari")):
        for split, marker in (("train", "o"), ("test", "s")):
            ax.plot(h["round"], h[f"{split}_{metric}"], marker, label=split.title())
        ax.set(title=metric.upper(), xlabel="Communication round")
        ax.legend()
    for ax, key, title in zip(axes.flat[3:],
        ("communication_bytes", "statistics_time", "aggregation_time"),
        ("Payload bytes", "Local training seconds", "Server seconds")):
        ax.bar(h["round"], h[key])
        ax.set(title=title, xlabel="Communication round")
    fig.suptitle("k-FED: One-shot Quality and Resources")
    fig.tight_layout()
    return fig


def print_kfed_convergence_summary(tracker):
    if tracker is None or not tracker.history["round"]:
        print("No k-FED training history available.")
        return
    print("\nk-FED: one-shot training summary")
    print(f"  Communication rounds: {len(tracker.history['round'])}")
    for split, metrics in tracker.final_metrics().items():
        print(f"  {split.title()}: ACC={metrics['acc']:.4f}, "
              f"NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}")
    print(f"  Payload communication: {tracker.communication['total_bytes'] / 2**20:.4f} MiB")
    print(f"  Training time: {sum(tracker.history['round_time']):.4f} seconds")
    print("  Privacy: plain uploaded centers; no formal DP or cryptographic protection")
