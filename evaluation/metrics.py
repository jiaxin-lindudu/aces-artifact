import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)


def compute_center_distances(true_centers, predicted_centers):
    M = np.linalg.norm(true_centers[:, None] - predicted_centers[None, :], axis=2)
    row, col = linear_sum_assignment(M)
    best_distances = M[row, col]
    print('\n=== Center Distance Statistics ===')
    print(f'Mean distance: {best_distances.mean():.4f}')
    print(f'Std distance:  {best_distances.std():.4f}')
    print(f'Min distance:  {best_distances.min():.4f}')
    print(f'Max distance:  {best_distances.max():.4f}')
    print('===================================')
    return {"mean_distance": best_distances.mean(), "std": best_distances.std(),
            "min": best_distances.min(), "max": best_distances.max()}


def compute_clustering_accuracy(true_labels, predicted_clusters):
    """
    计算聚类准确率（Clustering Accuracy, ACC），使得聚类标签最优映射到真实标签后，计算所有匹配结果的准确率。

    Parameters:
    - true_labels: (array-like) 真实标签（1维整数数组）
    - predicted_clusters: (array-like) 聚类结果标签（1维整数数组）

    Returns:
    - acc: (float) 聚类准确率
    """
    true_labels = np.array(true_labels)
    predicted_clusters = np.array(predicted_clusters)
    assert len(true_labels) == len(predicted_clusters)
    D = max(predicted_clusters.max(), true_labels.max()) + 1
    # 匹配矩阵
    w = np.zeros((D, D), dtype=np.int32)
    for i in range(true_labels.size):
        w[predicted_clusters[i], true_labels[i]] += 1
    # 最大化匹配，即最小化负数w的代价
    row_ind, col_ind = linear_sum_assignment(-w)
    acc = w[row_ind, col_ind].sum() / true_labels.size
    return acc


def evaluate_clustering(X, y, centers_orig):
    """Compute clustering metrics."""
    # 验证centers_orig是二维数组
    if centers_orig is None or len(centers_orig) == 0:
        raise ValueError("centers_orig is empty or None")
    
    centers_orig = np.asarray(centers_orig)
    if centers_orig.ndim != 2:
        raise ValueError(f"centers_orig must be 2D array, got shape {centers_orig.shape}")
    
    predictions = np.argmin(cdist(X, centers_orig), axis=1)
    # Internal indices are undefined if every point receives one label (or
    # every point is its own cluster). Keep valid external metrics for this run.
    valid_internal = 1 < len(np.unique(predictions)) < len(predictions)
    metrics = {
        'silhouette': silhouette_score(X, predictions) if valid_internal else np.nan,
        'davies_bouldin': davies_bouldin_score(X, predictions) if valid_internal else np.nan,
        'calinski_harabasz': calinski_harabasz_score(X, predictions) if valid_internal else np.nan,
        'ari': adjusted_rand_score(y, predictions),
        'nmi': normalized_mutual_info_score(y, predictions),
        'acc': compute_clustering_accuracy(y, predictions)
    }
    return predictions, metrics


def print_metrics(metrics):
    """Print evaluation metrics."""
    print("\n" + "=" * 60)
    print("Clustering Quality Evaluation")
    print("=" * 60)
    print(f"Silhouette Score:        {metrics['silhouette']:.4f} (higher is better)")
    print(f"Davies-Bouldin Index:    {metrics['davies_bouldin']:.4f} (lower is better)")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz']:.4f} (higher is better)")
    print(f"Adjusted Rand Index:     {metrics['ari']:.4f} (higher is better)")
    print(f"Normalized Mutual Info:  {metrics['nmi']:.4f} (higher is better)")
    print(f"Clustering Accuracy:     {metrics['acc']:.4f} (higher is better)")
