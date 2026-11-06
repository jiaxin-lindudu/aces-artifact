import os

import matplotlib.pyplot as plt
import numpy as np


def dirichlet_split(labels, n_clients, alpha, random_state=42):
    """
    使用Dirichlet分布划分数据

    参数:
        labels: 样本标签 [n_samples]
        n_clients: 客户端数量
        alpha: Dirichlet参数（控制Non-IID程度）
        random_state: 随机种子

    返回:
        client_indices: 每个客户端的样本索引列表
    """
    rng = np.random.RandomState(random_state)

    labels = np.asarray(labels)
    class_labels = np.unique(labels)
    n_classes = len(class_labels)
    n_samples = len(labels)

    # 为每个类别生成一个Dirichlet分布，决定该类别样本如何分配到各客户端
    # label_distribution[k, j] = 类别k的样本中，分配给客户端j的比例
    label_distribution = rng.dirichlet([alpha] * n_clients, n_classes)

    # 记录每个类别的样本索引
    # Iterate actual labels: USPS uses 1..10, not 0..9. Keep sorted class
    # order and RNG calls unchanged for existing zero-based datasets.
    class_indices = [np.where(labels == label)[0] for label in class_labels]

    # 初始化每个客户端的索引列表
    client_indices = [[] for _ in range(n_clients)]

    # 对每个类别，根据Dirichlet分布将样本分配到客户端
    for k in range(n_classes):
        # 获取当前类别的所有样本索引
        idx_k = class_indices[k]
        rng.shuffle(idx_k)

        # 根据Dirichlet分布计算每个客户端应得到的样本数
        proportions = label_distribution[k]
        proportions = (proportions / proportions.sum() * len(idx_k)).astype(int)

        # 处理舍入误差：确保所有样本都被分配
        proportions[-1] = len(idx_k) - proportions[:-1].sum()

        # 分配样本
        start_idx = 0
        for client_id in range(n_clients):
            end_idx = start_idx + proportions[client_id]
            client_indices[client_id].extend(idx_k[start_idx:end_idx].tolist())
            start_idx = end_idx

    # 转换为numpy数组并打乱每个客户端内部的样本顺序
    client_indices = [np.array(idx, dtype=int) for idx in client_indices]
    for idx in client_indices:
        rng.shuffle(idx)

    return client_indices


def add_quantity_skew(client_indices, skew_factor=3.0, random_state=42):
    """
    添加数据量偏斜：让某些客户端的数据量远多于其他客户端

    参数:
        client_indices: 原始客户端索引列表
        skew_factor: 偏斜因子（最大/最小样本数的比例）
        random_state: 随机种子

    返回:
        skewed_indices: 偏斜后的客户端索引列表
    """
    rng = np.random.RandomState(random_state)

    n_clients = len(client_indices)
    total_samples = sum(len(idx) for idx in client_indices)

    # 生成偏斜的样本数分配（使用指数分布）
    raw_weights = rng.exponential(1.0, n_clients)
    raw_weights = raw_weights / raw_weights.sum()

    # 调整权重以达到目标偏斜因子
    min_weight = 1.0 / (skew_factor + n_clients - 1)
    max_weight = skew_factor / (skew_factor + n_clients - 1)
    raw_weights = min_weight + (max_weight - min_weight) * (raw_weights - raw_weights.min()) / (
                raw_weights.max() - raw_weights.min())
    raw_weights = raw_weights / raw_weights.sum()

    target_sizes = (raw_weights * total_samples).astype(int)
    target_sizes[-1] = total_samples - target_sizes[:-1].sum()  # 修正舍入误差

    # 合并所有样本
    all_indices = np.concatenate(client_indices)
    rng.shuffle(all_indices)

    # 按新的样本数重新分配
    skewed_indices = []
    start = 0
    for size in target_sizes:
        skewed_indices.append(all_indices[start:start + size])
        start += size

    print(f"\nQuantity skew applied:")
    print(f"  Skew factor: {skew_factor:.1f}")
    print(f"  Sample counts: {[len(idx) for idx in skewed_indices]}")
    print(
        f"  Ratio (max/min): {max(len(idx) for idx in skewed_indices) / max(1, min(len(idx) for idx in skewed_indices)):.2f}")

    return skewed_indices


def compute_entropy(class_distribution):
    """计算类别分布的熵（衡量Non-IID程度）"""
    p = class_distribution / class_distribution.sum()
    p = p[p > 0]  # 避免log(0)
    return -np.sum(p * np.log(p))


def visualize_non_iid_distribution(y, client_splits, num_clusters, dataname, title="Non-IID Distribution"):
    """
    可视化Non-IID分布
    """
    n_clients = len(client_splits)

    # 创建堆叠条形图
    fig, ax = plt.subplots(figsize=(12, 6))

    # 统计每个客户端的类别分布
    client_class_counts = np.zeros((n_clients, num_clusters))
    for i, indices in enumerate(client_splits):
        client_class_counts[i] = np.bincount(y[indices], minlength=num_clusters)

    # 归一化为比例
    client_class_proportions = client_class_counts / client_class_counts.sum(axis=1, keepdims=True)

    # 绘制堆叠条形图
    bottom = np.zeros(n_clients)
    colors = plt.cm.Set3(np.linspace(0, 1, num_clusters))

    for k in range(num_clusters):
        ax.bar(range(n_clients), client_class_proportions[:, k],
               bottom=bottom, label=f'Class {k}', color=colors[k])
        bottom += client_class_proportions[:, k]

    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Class Proportion', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_clients))
    ax.set_xticklabels([f'C{i}' for i in range(n_clients)])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(f'figures/{dataname}', exist_ok=True)
    output_path = f'figures/{dataname}/non_iid_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Non-IID distribution plot to '{output_path}'")
    if 'agg' not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close(fig)
