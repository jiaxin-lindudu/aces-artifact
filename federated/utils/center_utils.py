import numpy as np

def initialize_centers(num_clusters, num_features, method='random',
                       client_data=None, data_normalized=True, random_state=None):
    """
    初始化全局中心点

    参数:
        num_clusters: 簇数量
        num_features: 特征维度
        method: 初始化方法
            - 'random': 从标准正态分布随机采样 (推荐用于标准化数据)
            - 'uniform': 从均匀分布随机采样
            - 'client': 从客户端数据采样
            - 'kmeans_plusplus': 使用k-means++策略
        client_data: 当method='client'时需要提供
        data_normalized: 数据是否已标准化
        random_state: 随机种子

    返回:
        centers: 初始中心点 [num_clusters, num_features]
    """
    if random_state is not None:
        np.random.seed(random_state)

    if method == 'random':
        if data_normalized:
            centers = np.random.randn(num_clusters, num_features)
            print(f"✓ Initialized centers from N(0, 0.3^2) distribution")
        else:
            raise ValueError("For non-normalized data, use 'client' or provide data statistics")

    elif method == 'uniform':
        if data_normalized:
            centers = np.random.uniform(0.0, 1.0, size=(num_clusters, num_features))
            print(f"✓ Initialized centers from Uniform(0.0, 1.0) distribution")
        else:
            raise ValueError("For non-normalized data, use 'client' method")

    elif method == 'client':
        if client_data is None:
            raise ValueError("client_data must be provided for 'client' method")
        if len(client_data) < num_clusters:
            raise ValueError(f"Client data has only {len(client_data)} samples but need {num_clusters} centers")

        indices = np.random.choice(len(client_data), size=num_clusters, replace=False)
        centers = client_data[indices]
        print(f"✓ Initialized centers by randomly sampling from client data")

    elif method == 'kmeans_plusplus':
        if client_data is None:
            raise ValueError("client_data must be provided for 'kmeans_plusplus' method")
        from sklearn.cluster import kmeans_plusplus
        centers, _ = kmeans_plusplus(client_data, n_clusters=num_clusters, random_state=random_state)
        print(f"✓ Initialized centers using K-means++ algorithm")

    else:
        raise ValueError(f"Unknown initialization method: {method}")

    return centers