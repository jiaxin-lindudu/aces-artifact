import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_optional_font = _PROJECT_ROOT / 'file-ngwyeoEN29l1M3O1QpdxCwkj-sider-font.ttf'
font_prop = (fm.FontProperties(fname=str(_optional_font))
             if _optional_font.is_file() else fm.FontProperties())

plt.rcParams['axes.unicode_minus'] = False


def prepare_visualization_data(X_train, X_eval, centers, true_centers,
                               center_history, random_state=42):
    """Project data and every center snapshot through one train-fitted PCA."""
    X_train = np.asarray(X_train)
    X_eval = np.asarray(X_eval)
    centers = np.asarray(centers)
    center_history = list(center_history or [centers])
    if X_train.shape[1] <= 2:
        return X_eval, centers, true_centers, center_history
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=random_state).fit(X_train)
    projected_true = (pca.transform(true_centers)
                      if true_centers is not None else None)
    return (pca.transform(X_eval), pca.transform(centers), projected_true,
            [pca.transform(snapshot) for snapshot in center_history])


def visualize_results(X_2d, centers_2d, true_centers_2d, predictions,
                      history_centers_2d, method, cfg, output_dir=None,
                      show=True):
    """Generate the same three publication figures for every method.

    When ``output_dir`` is provided, PDF files are saved and their paths are
    returned. ``show=False`` is appropriate for WSL and batch experiments.
    """
    print("\n" + "=" * 60)
    print("Visualization")
    print("=" * 60)

    paths = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("clusters", plot_2d, (X_2d, predictions), {
            "title": f"Clustering Result - {method.upper()} - {cfg['dataset'].upper()}"}),
        ("center_evolution", plot_centers_evolution,
         (history_centers_2d,), {
             "true_centers": true_centers_2d,
             "title": "Centers Evolution Over Training Rounds"}),
        ("center_comparison", plot_centers,
         (true_centers_2d, centers_2d), {
             "title": f"True vs Learned Centers - {method.upper()}",
             "data_points": X_2d, "labels": predictions}),
    ]
    for name, function, args, kwargs in specs:
        save_path = (output_dir / f"{method}_{name}.pdf"
                     if output_dir is not None else None)
        figure = function(*args, **kwargs, save_path=save_path, show=show)
        if save_path is not None and figure is not None:
            paths[name] = str(save_path)
        if not show and figure is not None:
            plt.close(figure)
    return paths


def plot_2d(X, y, title, save_path=None, show=True):
    fig = plt.figure(figsize=(6, 6))
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        X_ = PCA(n_components=2).fit_transform(X)
    else:
        X_ = X
    plt.scatter(X_[:, 0], X_[:, 1], c=y, s=12, cmap='tab10', alpha=0.60)
    plt.title(title, fontproperties=font_prop)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_centers(true_centers, learned_centers, title, data_points=None,
                 labels=None, save_path=None, show=True):
    fig = plt.figure(figsize=(6, 6))
    if data_points is not None and labels is not None:
        plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='tab10', alpha=0.3, s=12)
    if true_centers is not None:
        plt.scatter(true_centers[:, 0], true_centers[:, 1], marker="X", c='r', s=120, label="True Centers")
    plt.scatter(learned_centers[:, 0], learned_centers[:, 1], marker="o", edgecolor='black', c='lime', s=100,
                label="Learned Centers")
    # for i, tc in enumerate(true_centers):
    #     if i < learned_centers.shape[0]:
    #         plt.plot([tc[0], learned_centers[i, 0]], [tc[1], learned_centers[i, 1]], 'k--', lw=0.7)
    plt.legend(prop=font_prop)
    plt.title(title, fontproperties=font_prop)
    plt.xlabel("x1", fontproperties=font_prop)
    plt.ylabel("x2", fontproperties=font_prop)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_centers_evolution(centers_history, true_centers=None,
                           title="Centers Evolution Over Rounds",
                           annotate_rounds=True, save_path=None, show=True):
    """
    绘制中心点在多轮训练中的演化轨迹 (改进版)

    参数:
        centers_history: 列表，每个元素是一轮的中心点 [K, 2]
        true_centers: 可选，真实中心点 [K, 2]
        title: 图标题
        annotate_rounds: 是否在轨迹点上标注中间轮次的数字
    """
    if not centers_history or centers_history[0].shape[1] != 2:
        print("Warning: Center history is empty or not 2D, skipping plot.")
        return None

    fig = plt.figure(figsize=(12, 10))
    K = centers_history[0].shape[0]
    # Use a color map that provides distinct colors
    colors = plt.get_cmap('tab10', K)

    # Determine the data range to calculate a sensible text offset
    all_points = np.vstack(centers_history)
    y_range = np.max(all_points[:, 1]) - np.min(all_points[:, 1])
    offset = y_range * 0.03  # 3% of the y-axis range as offset

    # Plot the evolution trajectory for each cluster
    for k in range(K):
        trajectory = np.array([centers[k] for centers in centers_history])
        num_rounds = len(trajectory)

        # Plot the trajectory line
        plt.plot(trajectory[:, 0], trajectory[:, 1],
                 color=colors(k), alpha=0.7, linewidth=2,
                 label=f'Cluster {k}')

        # Mark the starting point (Round 0)
        plt.scatter(trajectory[0, 0], trajectory[0, 1],
                    color=colors(k), marker='o', s=200,
                    edgecolors='black', linewidths=1.5, zorder=5)

        # Mark the ending point
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1],
                    color=colors(k), marker='s', s=200,
                    edgecolors='black', linewidths=1.5, zorder=5)

        # Annotate the intermediate rounds
        if annotate_rounds and num_rounds > 2:
            # We only annotate the points between the start and end
            for t in range(1, num_rounds - 1):
                # Alternate the text position to avoid overlap
                pos_offset = offset if t % 2 == 0 else -offset
                plt.text(trajectory[t, 0], trajectory[t, 1] + pos_offset,
                         f'{t}', fontsize=9, ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='none', alpha=0.6))

    # Plot the true centers for reference, if provided
    if true_centers is not None:
        plt.scatter(true_centers[:, 0], true_centers[:, 1],
                    c='black', marker='*', s=600,
                    label='True Centers', zorder=10)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_centers_convergence(centers_history, title="Centers Convergence Over Rounds"):
    """
    绘制中心点随轮次的收敛情况

    参数:
        centers_history: 列表，每个元素是一轮的中心点 [K, 2]
        title: 图标题
    """
    if not centers_history:
        print("Warning: Center history is empty, skipping plot.")
        return

    # 计算每一轮中心点与最终中心点的平均欧氏距离
    final_centers = centers_history[-1]
    distances = []
    for centers in centers_history:
        # 计算每个中心点到最终中心点的最小距离（处理可能的重排序）
        from sklearn.metrics.pairwise import euclidean_distances
        dist_matrix = euclidean_distances(centers, final_centers)
        # 使用匈牙利算法找到最优匹配
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        avg_dist = dist_matrix[row_ind, col_ind].mean()
        distances.append(avg_dist)

    plt.figure(figsize=(10, 6))
    rounds = np.arange(len(centers_history))

    # 绘制收敛曲线
    plt.plot(rounds, distances, 'b-o', linewidth=2, markersize=8,
             markerfacecolor='white', markeredgewidth=1.5)

    # 添加参考线
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 添加收敛阈值线（可选的）
    if len(distances) > 5:  # 如果有多轮数据
        threshold = max(distances[0] * 0.01, 1e-6)  # 1% of initial distance
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7,
                    label=f'Convergence Threshold ({threshold:.2e})')

    plt.title(title, fontsize=14, fontweight='bold', fontproperties=font_prop)
    plt.xlabel('Training Round', fontproperties=font_prop)
    plt.ylabel('Average Distance to Final Centers', fontproperties=font_prop)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 如果数据点较少，显示所有x轴刻度
    if len(rounds) <= 20:
        plt.xticks(rounds)

    # 使用科学计数法显示y轴（如果值很小）
    if max(distances) < 1e-2:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.legend(prop=font_prop)
    plt.tight_layout()
    plt.show()
