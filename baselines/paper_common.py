"""Ours-style records and publication plots shared by the paper baselines.

The common implementation is deliberately complete: thin method-specific
tracker subclasses inherit the same metrics, resource accounting and figures.
No spectral coordinate/alignment metric is invented for single-space methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score)


def validate_clients(x_list, clusters, *, allow_empty=False, require_samples_for_clusters=True):
    if isinstance(clusters, (bool, np.bool_)) or not isinstance(clusters, (int, np.integer)):
        raise ValueError('cluster count must be a positive integer')
    arrays = [np.asarray(x, dtype=np.float64) for x in x_list]
    if not arrays or any(x.ndim != 2 or (not allow_empty and not len(x)) for x in arrays):
        raise ValueError("each client needs a nonempty 2-D array")
    d = arrays[0].shape[1]
    if d < 1 or any(x.shape[1] != d or not np.isfinite(x).all() for x in arrays):
        raise ValueError("inconsistent dimensions or nonfinite client data")
    if clusters < 1 or (require_samples_for_clusters and clusters > sum(map(len, arrays))):
        raise ValueError("invalid cluster count")
    return arrays


def public_centers(clusters, dimensions, seed, bound=3.0):
    """Data-independent initialization; does not pool private samples."""
    if bound <= 0 or not np.isfinite(bound):
        raise ValueError("public bound must be finite and positive")
    return np.random.default_rng(seed).uniform(-bound, bound, (clusters, dimensions))


def labels_from_centers(x, centers):
    return cdist(np.asarray(x), centers).argmin(axis=1)


def clustering_metrics(x, y, centers, labels=None):
    x = np.asarray(x)
    labels = labels_from_centers(x, centers) if labels is None else np.asarray(labels)
    if len(labels) != len(x):
        raise ValueError("labels and samples must have the same order and length")
    valid = 1 < len(np.unique(labels)) < len(labels)
    # A bounded diagnostic sample can occasionally miss a rare cluster.
    sil = np.nan
    if valid:
        try:
            sil = float(silhouette_score(x, labels, sample_size=min(2000, len(x)), random_state=0))
        except ValueError:
            pass
    values = dict(
        inertia=float(np.sum((x - centers[labels]) ** 2)),
        silhouette=sil,
        davies_bouldin=float(davies_bouldin_score(x, labels)) if valid else np.nan,
        calinski_harabasz=float(calinski_harabasz_score(x, labels)) if valid else np.nan)
    if y is not None:
        _, yi = np.unique(y, return_inverse=True)
        _, li = np.unique(labels, return_inverse=True)
        contingency = np.zeros((li.max() + 1, yi.max() + 1), dtype=np.int64)
        np.add.at(contingency, (li, yi), 1)
        a, b = linear_sum_assignment(-contingency)
        values.update(acc=float(contingency[a, b].sum() / len(x)),
                      nmi=float(normalized_mutual_info_score(y, labels)),
                      ari=float(adjusted_rand_score(y, labels)))
    else:
        values.update(acc=np.nan, nmi=np.nan, ari=np.nan)
    return values


class PaperTracker:
    """Train/test quality, native labels, centers and real payload/timing series."""
    def __init__(self, method):
        self.method = method
        self.history = {key: [] for key in (
            "round", "stage", "center_shift_orig", "objective", "round_time",
            "statistics_time", "encryption_time", "aggregation_time", "decryption_time",
            "communication_bytes", "uplink_bytes", "downlink_bytes", "logical_payload_bytes")}
        for split in ("train", "test"):
            for metric in ("inertia", "silhouette", "davies_bouldin", "calinski_harabasz",
                           "accuracy", "nmi", "ari"):
                self.history[f"{split}_{metric}"] = []
        self.centers_history = []
        self.communication = dict(uplink_bytes=0, downlink_bytes=0, total_bytes=0,
            logical_payload_bytes=0, rounds=0,
            measurement_basis="serialized application payload; excludes network framing")
        self.privacy_results = {"status": "no_formal_dp"}
        self.diagnostics = []

    def record(self, round_idx, centers, previous, *, x_train=None, y_train=None,
               x_test=None, y_test=None, predictor=None, train_labels=None,
               uplink=0, downlink=0, logical=0, times=None, objective=np.nan,
               stage="global_update", diagnostics=None):
        times = times or {}
        h = self.history
        h["round"].append(int(round_idx))
        h["stage"].append(stage)
        shift = (float(np.linalg.norm(centers - previous))
                 if previous is not None and centers.shape == previous.shape else np.nan)
        h["center_shift_orig"].append(shift)
        h["objective"].append(float(objective))
        for name in ("round_time", "statistics_time", "encryption_time",
                     "aggregation_time", "decryption_time"):
            h[name].append(float(times.get(name, 0)))
        for key, value in (("uplink_bytes", uplink), ("downlink_bytes", downlink),
                           ("logical_payload_bytes", logical),
                           ("communication_bytes", uplink + downlink)):
            h[key].append(int(value))
        for split, x, y in (("train", x_train, y_train), ("test", x_test, y_test)):
            labels = train_labels if split == "train" else None
            if labels is None and predictor is not None and x is not None:
                labels = predictor(x)
            metrics = clustering_metrics(x, y, centers, labels) if x is not None else {}
            for name in ("inertia", "silhouette", "davies_bouldin", "calinski_harabasz",
                         "accuracy", "nmi", "ari"):
                h[f"{split}_{name}"].append(metrics.get("acc" if name == "accuracy" else name, np.nan))
        self.centers_history.append(centers.copy())
        self.diagnostics.append(diagnostics or {})
        c = self.communication
        c["uplink_bytes"] += int(uplink)
        c["downlink_bytes"] += int(downlink)
        c["total_bytes"] += int(uplink + downlink)
        c["logical_payload_bytes"] += int(logical)
        c["rounds"] = int(round_idx)

    def snapshot(self):
        return dict(completed_rounds=self.communication["rounds"],
                    communication=self.communication.copy(),
                    privacy_results=self.privacy_results.copy(),
                    convergence=self.history, centers=self.centers_history,
                    diagnostics=self.diagnostics)

    def plot_convergence_summary(self, save_path=None):
        """Same 3x3/eight-panel resource layout as Ours, with native objective."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        h = self.history
        if not h["round"]:
            raise ValueError("cannot plot an empty training history")
        plt.rcParams.update({"font.family": "serif", "font.size": 11})
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=.36, wspace=.30)
        axes = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2])]
        axes += [fig.add_subplot(gs[r, c]) for r in (1, 2) for c in range(3)]
        r = h["round"]
        axes[0].plot(r, h["center_shift_orig"], "o-", color="#2E86AB")
        axes[0].set_title("(a) Center Movement")
        axes[0].set_ylabel("Center shift (original space)")
        if np.isfinite(h["objective"]).any():
            axes[1].plot(r, h["objective"], "o-", color="#A23B72")
        else:
            axes[1].text(.5, .5, "Not applicable", transform=axes[1].transAxes, ha="center")
        axes[1].set_title("(b) Native Objective")
        for ax, metric, title in zip(axes[2:5], ("inertia", "silhouette", "accuracy"),
                                    ("(c) Within-Cluster Variance", "(d) Cluster Separation",
                                     "(e) Clustering Accuracy")):
            for split, marker, color in (("train", "o", "#006D77"), ("test", "s", "#E63946")):
                ax.plot(r, h[f"{split}_{metric}"], marker + "-", label=split.title(), color=color)
            ax.set_title(title)
            ax.legend()
        axes[5].bar(r, np.asarray(h["communication_bytes"]) / 1024**2, color="#06A77D")
        axes[5].set_title("(f) Communication Overhead")
        axes[5].set_ylabel("MiB (serialized payload)")
        bottom = np.zeros(len(r))
        for name, label, color in (("statistics_time", "Statistics", "#2E86AB"),
                                  ("encryption_time", "Masking/setup", "#FF006E"),
                                  ("aggregation_time", "Aggregation", "#8338EC"),
                                  ("decryption_time", "Unmasking", "#F18F01")):
            vals = np.asarray(h[name])
            axes[6].bar(r, vals, bottom=bottom, label=label, color=color)
            bottom += vals
        axes[6].set_title("(g) Computation Time")
        axes[6].set_ylabel("Seconds")
        axes[6].legend(fontsize=8)
        axes[7].plot(r, h["round_time"], "o-", color="#FB5607")
        axes[7].set_title("(h) Total Round Time")
        axes[7].set_ylabel("Seconds")
        for ax in axes:
            ax.set_xlabel("Global communication round")
            ax.grid(alpha=.25, linestyle="--")
        fig.suptitle(f"{self.method}: Convergence Analysis", fontsize=16, fontweight="bold")
        fig.text(.5, .015, "Training time excludes metric evaluation and figure generation; communication excludes network framing.",
                 ha="center", fontsize=9)
        path = Path(save_path) if save_path is not None else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
            fig.savefig(path.with_suffix(".png"), dpi=130, bbox_inches="tight")
            plt.close(fig)
        # Keep all quality indices in a companion figure, not just ACC.
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, key in zip(axes.flat, ("nmi", "ari", "davies_bouldin", "calinski_harabasz")):
            ax.plot(r, h[f"train_{key}"], "o-", label="Train")
            ax.plot(r, h[f"test_{key}"], "s-", label="Test")
            ax.set(title=key.upper(), xlabel="Global communication round")
            ax.legend()
            ax.grid(alpha=.25)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path.with_name(path.stem + "_quality.pdf"))
            plt.close(fig)
        else:
            plt.show()
        return fig


@dataclass
class PaperResult:
    centers: np.ndarray
    tracker: PaperTracker
    methodology: dict
    predictor: object = None

    @property
    def history(self):
        return self.tracker.snapshot()

    def predict(self, x):
        return self.predictor(x) if self.predictor is not None else labels_from_centers(x, self.centers)


def timed_record(tracker, round_idx, centers, previous, started, evaluation, **kwargs):
    times = kwargs.pop("times", {})
    times["round_time"] = time.perf_counter() - started
    tracker.record(round_idx, centers, previous, times=times, **evaluation, **kwargs)


def save_model_figures(result, x, stem):
    """Cluster scatter and center trajectories using ONE shared 2-D projection."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    x = np.asarray(x)
    if x.shape[1] > 2:
        projection = PCA(2).fit(x)
        project = projection.transform
        axis_labels = ("PC1 (visualization only)", "PC2 (visualization only)")
    elif x.shape[1] == 1:
        project = lambda v: np.c_[v[:,0], np.zeros(len(v))]
        axis_labels = ("Feature 1", "Display axis")
    else:
        project = lambda v: v
        axis_labels = ("Feature 1", "Feature 2")
    points, centers = project(x), project(result.centers)
    labels = result.predict(x)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(points[:,0], points[:,1], c=labels, cmap=cmap, s=9, alpha=.5,
               vmin=0, vmax=max(1,len(centers)-1))
    ax.scatter(centers[:,0],centers[:,1],marker="X",s=160,c="black",label="Centers")
    ax.set(title=result.tracker.method, xlabel=axis_labels[0], ylabel=axis_labels[1])
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(stem)+"_clusters.pdf")
    plt.close(fig)
    trajectories=np.asarray([project(c) for c in result.tracker.centers_history])
    fig, ax=plt.subplots(figsize=(8,6))
    ax.scatter(points[:,0],points[:,1],s=5,color="gray",alpha=.15)
    for j in range(len(centers)):
        ax.plot(trajectories[:,j,0],trajectories[:,j,1],"o-",color=cmap(j%20),label=f"C{j}")
    ax.set(title=f"{result.tracker.method}: Center Evolution",
           xlabel=axis_labels[0],ylabel=axis_labels[1])
    ax.legend(ncol=2,fontsize=8)
    fig.tight_layout()
    fig.savefig(str(stem)+"_center_evolution.pdf")
    plt.close(fig)


def save_method_comparison(frame, save_path):
    """Ours-style final-quality/cost comparison, with mean/std across seeds."""
    import matplotlib.pyplot as plt
    methods=list(dict.fromkeys(frame["method"]))
    grouped=frame.groupby("method")
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    fields=(("test_acc","(a) Test ACC",1),("test_nmi","(b) Test NMI",1),
            ("test_ari","(c) Test ARI",1),("communication_total_bytes","(d) Payload MiB",1024**2),
            ("wall_seconds","(e) Full Pipeline Seconds",1),("rounds","(f) Global Rounds",1))
    positions=np.arange(len(methods))
    for ax,(key,title,scale) in zip(axes.flat,fields):
        mean=grouped[key].mean().reindex(methods)/scale
        std=grouped[key].std().reindex(methods).fillna(0)/scale
        ax.bar(positions,mean,yerr=std,capsize=3,color="#2E86AB",alpha=.8)
        ax.set_xticks(positions,methods,rotation=45,ha="right",fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y",alpha=.25)
    fig.suptitle("Identical Data and Partition: Quality / Communication / Time (Mean ± SD)",fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(save_path,bbox_inches="tight")
    fig.savefig(Path(save_path).with_suffix(".png"),dpi=130,bbox_inches="tight")
    plt.close(fig)
