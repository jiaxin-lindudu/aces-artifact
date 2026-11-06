import hashlib
import json
import os

import numpy as np
import scipy.io as sio
from sklearn.cluster import SpectralClustering  # Import SpectralClustering
from sklearn.datasets import make_blobs, load_iris, load_wine, load_digits, make_swiss_roll, make_s_curve, make_moons, \
    make_circles, fetch_openml, fetch_olivetti_faces, fetch_covtype
from sklearn.model_selection import train_test_split

# Fixed dataset identity; the newest USPS version is not necessarily ten-class.
DATASET_SOURCES = {
    'covtype_10000': {
        'provider': 'scikit-learn fetch_covtype / UCI Covertype',
        'source_url': 'https://archive.ics.uci.edu/dataset/31/covertype',
        'raw_samples': 581012, 'raw_features': 54, 'classes': 7,
        'subset_size': 10000, 'subset_seed': 42,
        'subset_rule': 'fixed stratified source-row selection; sorted row IDs; same subset for all data_seed values',
        'partition_scope': 'stratified 8000 FIT / 2000 TEST per data_seed; simulated record-wise federation, not natural silos',
        'preprocessing': 'FIT-only StandardScaler; retain all 54 native features, without padding or PCA',
    },
    'mnist_10000': {
        'provider': 'official MNIST IDX files', 'classes': 10,
        'source_url': 'https://ossci-datasets.s3.amazonaws.com/mnist/',
        'raw_samples': 70000, 'subset_size': 10000, 'subset_seed': 42,
        'subset_rule': 'fixed stratified selection from official train+test pool; sorted source row IDs',
        'partition_scope': 'pooled official train/test; fixed subset re-split 80/20 by data_seed; simulated federation',
    },
    'usps': {'provider': 'OpenML', 'data_id': 41082, 'version': 2},
    'cifar10_vit': {'provider': 'local official CIFAR-10 ViT features',
        'official_train_size': 50000, 'official_test_size': 10000,
        'weights': 'ViT_B_16_Weights.IMAGENET1K_V1',
        'feature_metadata': 'data/cifar10_vit_official/metadata.json',
        'preprocessing': 'existing train-fitted StandardScaler + PCA to 128; no TEST fitting',
        'partition_scope': 'official train/test retained; simulated federated training partition',
        'test_evaluation': 'original-space proxy only; primary clustering is transductive FIT'},
    'har': {'provider': 'OpenML', 'data_id': 1478, 'version': 1,
            'source_url': 'https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones',
            'license': 'CC BY 4.0 at UCI source', 'raw_samples': 10299, 'raw_features': 561,
            'classes': 6,
            'partition_scope': 'pooled original train/test; simulated record-wise federation, NOT subject-disjoint',
            'preprocessing': 'train-only scaling and PCA to 64; no activity labels used for features'},
}


def _load_mnist_10000():
    """One fixed raw subset for every data_seed; no processed split is imported."""
    import gzip
    import struct
    import urllib.request
    source = DATASET_SOURCES['mnist_10000']
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist_10000', 'MNIST', 'raw')
    files = (
        ('train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873', (60000, 784), 2051),
        ('train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432', (60000,), 2049),
        ('t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3', (10000, 784), 2051),
        ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c', (10000,), 2049),
    )
    values = []
    os.makedirs(raw_dir, exist_ok=True)
    for filename, expected, shape, magic in files:
        path = os.path.join(raw_dir, filename)
        if not os.path.exists(path):
            urllib.request.urlretrieve(source['source_url'] + filename, path)
        with open(path, 'rb') as stream:
            content = stream.read()
        if hashlib.md5(content).hexdigest() != expected:
            raise ValueError('Official MNIST checksum mismatch: ' + filename)
        raw = gzip.decompress(content)
        actual_magic, count = struct.unpack('>II', raw[:8])
        offset = 16 if magic == 2051 else 8
        if actual_magic != magic or count != shape[0]:
            raise ValueError('Unexpected MNIST IDX header: ' + filename)
        if magic == 2051 and struct.unpack('>II', raw[8:16]) != (28, 28):
            raise ValueError('Expected 28x28 MNIST images')
        values.append(np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(shape))
    X = np.concatenate((values[0], values[2]))
    y = np.concatenate((values[1], values[3])).astype(int)
    selected, _ = train_test_split(np.arange(len(y)), train_size=source['subset_size'],
                                   stratify=y, random_state=source['subset_seed'])
    selected = np.sort(selected)
    # Record membership/order in the existing metadata, without a new sidecar.
    source['subset_indices_sha256'] = hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()
    source['raw_files_md5'] = {name: checksum for name, checksum, _, _ in files}
    return X[selected], y[selected]


def get_dataset(name='blobs', n_samples=1000, n_clusters=4, n_features=3, random_state=42):
    rng = np.random.RandomState(random_state)
    name = 'bank' if name == 'banknote' else name
    data_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sklearn_cache')
    if name == 'blobs':
        X, y, centers = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features,
                                   random_state=random_state,
                                   return_centers=True)
    elif name == 'iris':
        data = load_iris()
        X, y = data.data, data.target
        centers = np.array([X[y == i].mean(axis=0) for i in range(3)])
    elif name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        centers = np.array([X[y == i].mean(axis=0) for i in range(3)])
    elif name == 'digits':
        data = load_digits()
        X, y = data.data, data.target
        centers = np.array([X[y == i].mean(axis=0) for i in range(10)])
    elif name == 'covtype_10000':
        source = DATASET_SOURCES[name]
        data = fetch_covtype(data_home=data_home, as_frame=False)
        if data.data.shape != (source['raw_samples'], source['raw_features']):
            raise ValueError('Unexpected Covertype release: expected 581012 rows and 54 features')
        labels, encoded = np.unique(data.target, return_inverse=True)
        if len(labels) != source['classes']:
            raise ValueError('Unexpected Covertype release: expected seven classes')
        selected, _ = train_test_split(
            np.arange(len(encoded)), train_size=source['subset_size'],
            stratify=encoded, random_state=source['subset_seed'])
        selected = np.sort(selected)
        source['subset_indices_sha256'] = hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()
        source['source_class_labels'] = labels.tolist()
        X, y = np.asarray(data.data[selected], dtype=float), encoded[selected]
        centers = np.array([X[y == i].mean(axis=0) for i in range(source['classes'])])
    elif name == 'coil20':
        # =========================================================
        # 修改：仿照 Bank/Digits 格式，自动从 OpenML 下载，无需手动拖文件
        # =========================================================
        print(f"正在尝试从 OpenML 自动下载 {name} 数据集...")
        try:
            # 尝试下载 COIL-20 (ID: 40683 或名称 'coil-20')
            # version=2 通常是标准版本 (1440 samples, 1024 features)
            coil = fetch_openml(name='coil-20', version=2, as_frame=False,
                                cache=True, data_home=data_home)
            X = coil.data.astype(float)
            y = coil.target

            # OpenML 的标签可能是字符串或从1开始，统一转换为从0开始的整数
            try:
                y = y.astype(int)
            except ValueError:
                # 如果标签是 'class1' 这种字符串，使用 unique 编码
                _, y = np.unique(y, return_inverse=True)

            # 确保标签从 0 开始
            if y.min() == 1:
                y = y - 1

            print(f"成功加载 COIL-20: {X.shape} samples")

            labels = np.unique(y)
            centers = np.array([X[y == i].mean(axis=0) for i in labels])

        except Exception as e:
            print(f"OpenML 下载失败: {e}")
            print("尝试自动下载 .mat 文件作为备选方案...")

            # --- 备选：如果 OpenML 失败，自动下载 .mat 文件 ---
            import urllib.request
            mat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'COIL20')
            os.makedirs(mat_dir, exist_ok=True)
            mat_path = os.path.join(mat_dir, 'COIL20.mat')

            if not os.path.exists(mat_path):
                url = "https://github.com/jundongl/scikit-feature/raw/master/skfeature/data/COIL20.mat"
                print(f"正在从 {url} 下载...")
                try:
                    urllib.request.urlretrieve(url, mat_path)
                    print("下载完成！")
                except Exception as dl_e:
                    raise RuntimeError(f"无法下载数据集，请检查网络: {dl_e}")

            # 加载下载好的 .mat (保持原有逻辑)
            mat = sio.loadmat(mat_path)
            if 'fea' in mat:
                X = mat['fea']
            elif 'X' in mat:
                X = mat['X']

            if 'gnd' in mat:
                y = mat['gnd'].ravel()
            elif 'Y' in mat:
                y = mat['Y'].ravel()
            elif 'labels' in mat:
                y = mat['labels'].ravel()

            if y.min() == 1: y = y - 1
            labels = np.unique(y)
            centers = np.array([X[y == i].mean(axis=0) for i in labels])

    elif name == 'bank':
        # UCI Banknote Authentication: 1372 样本，5 属性，2 类
        bank = fetch_openml(name='banknote-authentication', version=1,
                            as_frame=False, data_home=data_home)
        X = bank.data.astype(float)
        # target 是字符串，需要转成 int
        y = bank.target.astype(int)
        labels = np.unique(y)
        centers = np.array([X[y == i].mean(axis=0) for i in labels])
    elif name == 'usps':
        # USPS 手写数字: 9298 样本，16×16 像素 = 256 特征，10 类
        # Do not select version="active": a different active version is binary.
        usps = fetch_openml(data_id=DATASET_SOURCES['usps']['data_id'],
                            as_frame=False, data_home=data_home)
        X = usps.data.astype(float)
        y = usps.target.astype(int)
        labels = np.unique(y)
        centers = np.array([X[y == i].mean(axis=0) for i in labels])
    elif name == 'har':
        # Fixed OpenML/UCI HAR release; no new loader or experiment script.
        har = fetch_openml(data_id=DATASET_SOURCES['har']['data_id'],
                          as_frame=False, data_home=data_home)
        X = har.data.astype(float)
        _, y = np.unique(har.target, return_inverse=True)
        if X.shape != (10299, 561) or len(np.unique(y)) != 6:
            raise ValueError("Unexpected HAR release: expected 10299 records, 561 features and 6 activities")
        labels = np.unique(y)
        centers = np.array([X[y == i].mean(axis=0) for i in labels])
    elif name == 'orl':
        # --- 修复维度问题：增加转置操作 ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(current_dir, 'orl', 'olivettifaces.mat')

        print(f"检查文件路径: {local_path}")

        if os.path.exists(local_path):
            print(f"成功发现本地文件，正在加载: {local_path}")
            mat = sio.loadmat(local_path)

            if 'faces' in mat:
                faces = mat['faces']
                print(f"原始数据形状: {faces.shape}")  # 应该是 (4096, 400)

                # 关键修复：转置矩阵
                # 从 (特征, 样本) -> (样本, 特征)
                X = faces.T

                print(f"转置后数据形状: {X.shape}")  # 应该是 (400, 4096)

                # 手动生成标签: 40个人，每人10张照片
                y = np.repeat(np.arange(40), 10)
            else:
                raise KeyError(f"在文件 {local_path} 中未找到 'faces' 数据键值")
        else:
            print(f"错误：在路径 {local_path} 未找到文件！")
            faces = fetch_olivetti_faces(data_home=data_home)
            X = faces.data
            y = faces.target

        labels = np.unique(y)
        centers = np.array([X[y == i].mean(axis=0) for i in labels])
    elif name == 'swiss_roll':
        # # 1. 生成数据
        # # n_samples 通常设为 10000 (参照 Blobs)
        # X, color = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=random_state)
        # # 2. 【关键新增】标准化 (Standardization)
        # # 这一步必须加，否则不符合论文 Experimental Setup
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        # # 3. 生成 Ground Truth 标签
        # # n_clusters 建议设为 4 或 5
        # # affinity='nearest_neighbors' 确保了它是基于流形距离而非欧氏距离
        # spec = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=random_state)
        # y = spec.fit_predict(X)
        # # 4. 计算中心点 (Ground Truth Centers)
        # # 注意：在瑞士卷中，这个均值中心点可能实际上位于“空气中”（卷层之间），但在代码逻辑上是没问题的
        # centers = np.array([X[y == i].mean(axis=0) for i in range(n_clusters)])
        
        
        # 1. 生成数据，获取流形坐标 t (就是原来的 color 变量)
        # 注意：noise=0.1 是物理噪声，必须保留以增加难度
        X, t = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=random_state)
        
        # 2. 标准化 (Standardization)
        # 必须加，否则不同维度量纲不一致会影响聚类
        # The shared train-only scaler below standardizes this dataset.
        
        # 3. 【关键调整】基于流形坐标 t 生成 Ground Truth 标签
        # 逻辑：将 t 的范围均匀切分成 n_clusters 段
        # 这样生成的标签就是沿着卷曲方向的“色带”，这才是流形聚类的真正目标
        
        # 计算切分点 (bins)
        bins = np.linspace(t.min(), t.max(), n_clusters + 1)
        
        # 使用 digitize 将 t 映射为 1, 2, 3...
        # 减 1 是为了让标签从 0 开始 (0, 1, 2...)
        y = np.digitize(t, bins) - 1

        # 修正边界：最大值可能会被分到第 n_clusters 类（越界），需归入最后一类
        y[y == n_clusters] = n_clusters - 1
        # 4. 计算中心点 (虽然对于流形数据，欧氏中心意义不大，但代码接口需要)
        centers = np.array([X[y == i].mean(axis=0) for i in range(n_clusters)])

    elif name == 's_curve':
        X, color = make_s_curve(n_samples=n_samples, noise=0.1, random_state=random_state)
        spec = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=random_state)
        y = spec.fit_predict(X)
        centers = np.array([X[y == i].mean(axis=0) for i in range(n_clusters)])
    elif name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
        X = np.column_stack([X, rng.randn(len(X)) * 0.1])  # Add some noise to make it 3D
        centers = np.array([X[y == i].mean(axis=0) for i in np.unique(y)])
    elif name == 'concentric_circles':
        X, y = make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=random_state)
        # X = np.column_stack([X, np.random.randn(len(X)) * 0.1])  # Make it 3D
        # centers = np.array([X[y == i].mean(axis=0) for i in np.unique(y)])
        centers = np.array([np.median(X[y == i], axis=0) for i in np.unique(y)])


    #4.7新增数据ViT
    elif name == 'cifar10_vit':
        # Official train/test only. Never silently reuse the legacy test-only features.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        feature_dir = os.path.join(current_dir, 'cifar10_vit_official')
        with open(os.path.join(feature_dir, 'metadata.json'), encoding='utf-8') as stream:
            metadata = json.load(stream)
        contract = metadata.get('contract', {})
        if (metadata.get('status') != 'completed' or contract.get('split_protocol') != 'official train/test'
                or contract.get('weights') != 'IMAGENET1K_V1' or contract.get('feature_dimension') != 768):
            raise ValueError('Official CIFAR-10 feature extraction is incomplete or has a different contract')
        arrays = []
        for filename, shape in (('X_train.npy', (50000, 768)), ('X_test.npy', (10000, 768)),
                                ('y_train.npy', (50000,)), ('y_test.npy', (10000,))):
            path = os.path.join(feature_dir, filename)
            digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != metadata['files'][filename]['sha256']:
                raise ValueError(f'Official CIFAR-10 feature hash mismatch: {filename}')
            values = np.load(path, mmap_mode='r')
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f'Unexpected official CIFAR-10 array: {filename}')
            arrays.append(values)
        X_train, X_test, y_train, y_test = arrays
        if not np.array_equal(np.unique(y_train), np.arange(10)) or not np.array_equal(np.unique(y_test), np.arange(10)):
            raise ValueError('Official CIFAR-10 must contain labels 0 through 9')
        print('Loaded official CIFAR-10 train/test ViT features: 50000 / 10000 records')
        return X_train, X_test, y_train, y_test, None
      #/////////////////////////////////////////////////////////////////////////////////////

    elif name == 'mnist_10000':
        X, y = _load_mnist_10000()
        centers = np.array([X[y == i].mean(axis=0) for i in range(10)])

    #2026.4.10新增mnist数据集
    elif name == 'mnist':
        print("Fetching MNIST dataset from OpenML (this may take a minute)...")
        # 从 OpenML 下载 MNIST 784 维数据
        data = fetch_openml('mnist_784', version=1, as_frame=False,
                            parser='auto', data_home=data_home)
        X_full, y_full = data.data, data.target.astype(int)
        
        # 为了防止谱聚类在 7 万条数据上内存溢出 (OOM)，默认抽取 10000 条子集
        # np.random.seed(random_state)
        # subset_size = 10000
        # indices = np.random.choice(len(X_full), subset_size, replace=False)
        # X, y = X_full[indices], y_full[indices]
       # 1. 训练集保持全量 70000 条，供客户端进行真实的联邦训练
        X = X_full
        y = y_full
        
        # 2. 🔥 核心修复：单独抽样 10000 条作为测试集 (X_test)
        # 这确保了评估指标（Silhouette Score等）瞬间算完，不炸内存！
        eval_indices = rng.choice(len(X_full), 10000, replace=False)
        X_test = X_full[eval_indices]
        y_test = y_full[eval_indices]
        
        # 计算真实的中心点
        centers = np.array([X[y == i].mean(axis=0) for i in range(10)])

    elif name == 'spiral':
        n = n_samples // 3  # 3 spirals
        theta = np.linspace(0, 4 * np.pi, n)
        r = np.linspace(0.5, 2.5, n)

        # Generate points for each spiral
        x1 = r * np.cos(theta) + rng.normal(0, 0.1, n)
        y1 = r * np.sin(theta) + rng.normal(0, 0.1, n)

        x2 = r * np.cos(theta + 2 * np.pi / 3) + rng.normal(0, 0.1, n)
        y2 = r * np.sin(theta + 2 * np.pi / 3) + rng.normal(0, 0.1, n)

        x3 = r * np.cos(theta + 4 * np.pi / 3) + rng.normal(0, 0.1, n)
        y3 = r * np.sin(theta + 4 * np.pi / 3) + rng.normal(0, 0.1, n)

        # Stack all points
        X = np.vstack([
            np.column_stack([x1, y1, rng.normal(0, 0.1, n)]),
            np.column_stack([x2, y2, rng.normal(0, 0.1, n)]),
            np.column_stack([x3, y3, rng.normal(0, 0.1, n)])
        ])
        y = np.concatenate([np.zeros(n), np.ones(n), np.ones(n) * 2]).astype(int)

        # Calculate centers based on the spiral structure
        centers = np.array([
            [np.mean(x1), np.mean(y1), 0],
            [np.mean(x2), np.mean(y2), 0],
            [np.mean(x3), np.mean(y3), 0]
        ])
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Fit preprocessing on training samples only. Test features must not
    # influence the transformation used for model selection or training.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    # Per-dataset scaling.
    # train-only StandardScaler.
    if name in ['mnist', 'mnist_10000', 'usps', 'orl', 'coil20', 'digits']:
        # 图像像素数据：使用 MinMaxScaler 避免放大黑色背景噪声
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        centers_scaled = scaler.transform(centers)
    else:
        # 深度学习特征 (如 ViT, 各种 embeddings) 或常规表格数据：使用 StandardScaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        centers_scaled = scaler.transform(centers)

    # return X_scaled, y, centers_scaled
    return X_train, X_test, y_train, y_test, centers_scaled


def apply_pca_preprocessing(X_train, X_test, centers, cfg):
    dataset_name = cfg['dataset'].lower()
    
    # 默认开启 PCA，但如果主程序传了 apply_pca: False，就会跳过这段代码
    if cfg.get('apply_pca', True) and dataset_name in ['orl', 'usps', 'coil20', 'cifar10_vit', 'cifar100_vit', 'mnist', 'mnist_10000', 'har'] and X_train.shape[1] > 64:
        print(f"  [Preprocessing] Applying PCA (original dim: {X_train.shape[1]})...")
        target_dim = 128 if dataset_name in ['cifar10_vit', 'cifar100_vit'] else 64 
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_dim, random_state=cfg.get('random_state', 42))
        X_train = pca.fit_transform(X_train)
        if X_test is not None:
            X_test = pca.transform(X_test)
        if centers is not None:
            centers = pca.transform(centers)
            
        print(f"  -> PCA applied. New dimension: {X_train.shape[1]}")
        explained_variance = sum(pca.explained_variance_ratio_)
        print(f"  -> PCA Explained Variance Ratio: {explained_variance:.3f}")
        cfg['num_features'] = target_dim
    else:
        # 如果不执行 PCA，也要把真实的特征维度更新到 cfg 中
        print(f"  [Preprocessing] Skipping PCA. Using original dimension: {X_train.shape[1]}")
        cfg['num_features'] = X_train.shape[1]
        
    return X_train, X_test, centers


"""
===================================================================================
Stage 1 & 2: Centralized and Federated Data Loaders
===================================================================================
These functions load preprocessed data prepared by:
  - prepare_centralized_splits.py (Stage 1)
  - prepare_federated_partitions.py (Stage 2)

They provide a unified interface for both centralized and federated experiments.
"""

from collections import namedtuple
from pathlib import Path

# 定义返回类型
CentralizedData = namedtuple('CentralizedData',
                             ['X_train', 'X_test', 'y_train', 'y_test', 'centers', 'metadata'])

FederatedPartition = namedtuple('FederatedPartition',
                                ['X_train', 'X_test', 'y_train', 'y_test', 'centers',
                                 'client_indices', 'client_info', 'metadata'])


def load_centralized_split(dataset, data_seed=0, base_dir='data', verbose=True):
    """
    Load preprocessed centralized FIT/TEST split from Stage 1.

    This is the first-stage output used for:
      - Centralized clustering baselines (ceiling experiments)
      - Input data for Stage 2 federated partitioning

    Args:
        dataset (str): Dataset name (e.g., 'digits', 'usps', 'cifar10_vit')
        data_seed (int): Data seed used during split (default: 0)
        base_dir (str): Root directory containing splits/ subdirectory
        verbose (bool): Print loading progress

    Returns:
        CentralizedData: Named tuple with:
            - X_train (ndarray): Preprocessed training features [n_train, d]
            - X_test (ndarray): Preprocessed test features [n_test, d]
            - y_train (ndarray): Training labels [n_train]
            - y_test (ndarray): Test labels [n_test]
            - centers (ndarray): Class centers [K, d]
            - metadata (dict): Complete split provenance and validation info

    Raises:
        FileNotFoundError: If split files are missing
        ValueError: If metadata validation fails

    Example:
        >>> split = load_centralized_split('digits', data_seed=0)
        >>> X_train, X_test = split.X_train, split.X_test
        >>> print(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")
    """
    dataset = 'bank' if dataset == 'banknote' else dataset
    base_path = Path(base_dir) / dataset / 'splits'

    # Construct filenames
    seed_suffix = '' if data_seed == 42 else f'_seed_{data_seed}'
    split_filename = f'{dataset}_centralized{seed_suffix}'
    split_file = base_path / f'{split_filename}.npz'
    meta_file = base_path / f'{split_filename}_metadata.json'

    if verbose:
        print(f'\n[Stage 1] Loading centralized split: {dataset} (seed={data_seed})')

    # Check file existence
    if not split_file.exists():
        raise FileNotFoundError(
            f'Centralized split not found: {split_file}\n'
            f'Please run: python data/prepare_centralized_splits.py '
            f'--datasets {dataset} --data-seeds {data_seed}'
        )

    if not meta_file.exists():
        raise FileNotFoundError(f'Split metadata not found: {meta_file}')

    # Load metadata
    with open(meta_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Load arrays
    with np.load(split_file) as archive:
        X_train = np.asarray(archive['X_train'], dtype=np.float32)
        X_test = np.asarray(archive['X_test'], dtype=np.float32)
        y_train = np.asarray(archive['y_train'], dtype=np.int32)
        y_test = np.asarray(archive['y_test'], dtype=np.int32)
        centers = np.asarray(archive['centers'], dtype=np.float32)

    # Validate shapes and metadata consistency
    split_meta = metadata.get('split', {})
    if len(X_train) != split_meta.get('n_train'):
        raise ValueError(f'X_train shape mismatch: got {len(X_train)}, expected {split_meta["n_train"]}')
    if len(X_test) != split_meta.get('n_test'):
        raise ValueError(f'X_test shape mismatch: got {len(X_test)}, expected {split_meta["n_test"]}')

    if verbose:
        print(f'  ✓ Loaded: {len(X_train)} train + {len(X_test)} test samples')
        print(f'  ✓ Classes: {metadata["split"]["n_classes"]}, Features: {X_train.shape[1]}')

    return CentralizedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        centers=centers,
        metadata=metadata
    )


def load_federated_partition(dataset, data_seed, beta, num_clients,
                             base_dir='data', verbose=True):
    """
    Load federated partition from Stage 2, with client-wise data splits.

    This is the second-stage output used for federated clustering experiments.
    FIT samples are partitioned across clients; TEST is shared. Stage 2 requires
    Stage 1 splits to already exist.

    Args:
        dataset (str): Dataset name (e.g., 'digits', 'usps')
        data_seed (int): Data seed (must match Stage 1)
        beta (float or str): Dirichlet concentration, or 'subject' for natural HAR ownership:
                     beta < 1.0 → high Non-IID, beta ≥ 1.0 → more IID
        num_clients (int): Number of clients for partition
        base_dir (str): Root directory containing partitions/ subdirectory
        verbose (bool): Print loading progress

    Returns:
        FederatedPartition: Named tuple with:
            - X_train (ndarray): Full FIT features [n_train, d] (for indexing)
            - X_test (ndarray): Shared TEST features [n_test, d]
            - y_train (ndarray): Full FIT labels [n_train]
            - y_test (ndarray): TEST labels [n_test]
            - centers (ndarray): Shared class centers [K, d]
            - client_indices (list of ndarray): FIT indices assigned to each client
                                               client_indices[i] = indices in X_train
                                               X_client_i = X_train[client_indices[i]]
            - client_info (dict): Metadata per client:
                                 client_info[str(i)] = {
                                     'indices': [...],
                                     'size': int,
                                     'class_distribution': [...]
                                 }
            - metadata (dict): Complete partition provenance

    Raises:
        FileNotFoundError: If partition files are missing
        ValueError: If partition validation fails

    Example:
        >>> partition = load_federated_partition('digits', data_seed=0,
        ...                                      beta=0.5, num_clients=10)
        >>> # Get data for each client
        >>> for cid in range(10):
        ...     X_client_i = partition.X_train[partition.client_indices[cid]]
        ...     print(f'Client {cid}: {len(X_client_i)} samples')
        >>> # Global TEST data (shared by all clients)
        >>> X_test_shared = partition.X_test
    """
    dataset = 'bank' if dataset == 'banknote' else dataset

    # Construct partition directory path
    partition_dir = (Path(base_dir) / dataset / 'partitions' /
                     f'{dataset}_seed_{data_seed}_beta_{beta}_clients_{num_clients}')

    partition_file = partition_dir / 'partition.npz'
    splits_file = partition_dir / 'client_splits.json'
    meta_file = partition_dir / 'partition_metadata.json'

    if verbose:
        print(f'\n[Stage 2] Loading federated partition: {dataset}')
        print(f'  seed={data_seed}, beta={beta}, clients={num_clients}')

    # Check existence
    if not partition_file.exists():
        raise FileNotFoundError(
            f'Federated partition not found: {partition_file}\n'
            f'Please run: python data/prepare_federated_partitions.py '
            f'--datasets {dataset} --data-seeds {data_seed} --betas {beta} --clients {num_clients}'
        )

    if not all(f.exists() for f in [splits_file, meta_file]):
        raise FileNotFoundError(f'Partition metadata files missing in: {partition_dir}')

    # Load metadata
    with open(meta_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Load partition arrays
    with np.load(partition_file) as archive:
        X_train = np.asarray(archive['X_train'], dtype=np.float32)
        X_test = np.asarray(archive['X_test'], dtype=np.float32)
        y_train = np.asarray(archive['y_train'], dtype=np.int32)
        y_test = np.asarray(archive['y_test'], dtype=np.int32)
        centers = np.asarray(archive['centers'], dtype=np.float32)

    # Load client ownership
    with open(splits_file, 'r', encoding='utf-8') as f:
        client_info = json.load(f)

    # Convert client indices to numpy arrays
    client_indices = [
        np.asarray(client_info[str(i)]['indices'], dtype=np.int32)
        for i in range(num_clients)
    ]

    # Validate partition coverage
    flattened = np.sort(np.concatenate(client_indices))
    expected = np.arange(len(y_train), dtype=np.int32)
    if not np.array_equal(flattened, expected):
        raise ValueError('Client partition does not cover all FIT samples exactly once')

    # Validate metadata consistency
    fed_meta = metadata.get('federated', {})
    if fed_meta.get('num_clients') != num_clients:
        raise ValueError(f'Client count mismatch: got {num_clients}, expected {fed_meta["num_clients"]}')
    if beta == 'subject':
        if (fed_meta.get('partition_strategy') != 'subject'
                or fed_meta.get('dirichlet_beta') is not None):
            raise ValueError('Expected natural subject ownership, without a Dirichlet beta')
    elif abs(fed_meta.get('dirichlet_beta', 0) - beta) > 1e-6:
        raise ValueError(f'Beta mismatch: got {beta}, expected {fed_meta["dirichlet_beta"]}')

    if verbose:
        client_sizes = [len(idx) for idx in client_indices]
        print(f'  ✓ Clients: {num_clients}')
        print(f'  ✓ FIT samples: {len(X_train)} → sizes per client: '
              f'[{min(client_sizes)}, {max(client_sizes)}] (avg={np.mean(client_sizes):.1f})')
        print(f'  ✓ TEST samples: {len(X_test)} (shared across all clients)')

    return FederatedPartition(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        centers=centers,
        client_indices=client_indices,
        client_info=client_info,
        metadata=metadata
    )


def extract_client_data(partition, client_id):
    """
    Convenience function: extract local data for a specific client.

    Args:
        partition (FederatedPartition): Output from load_federated_partition()
        client_id (int): Client ID (0-indexed)

    Returns:
        tuple: (X_client, y_client) where:
            - X_client (ndarray): Local training features [local_size, d]
            - y_client (ndarray): Local training labels [local_size]

    Example:
        >>> partition = load_federated_partition('digits', 0, beta=0.5, num_clients=10)
        >>> X_client_0, y_client_0 = extract_client_data(partition, client_id=0)
    """
    if not 0 <= client_id < len(partition.client_indices):
        raise ValueError(f'Invalid client_id {client_id}; expected 0-{len(partition.client_indices) - 1}')

    indices = partition.client_indices[client_id]
    return partition.X_train[indices], partition.y_train[indices]


# ==================================================================================
# Legacy compatibility wrapper (backward compatibility with prepare_experiment_data)
# ==================================================================================

def load_cached_split(dataset, data_seed=0, base_dir='data'):
    """
    Lightweight wrapper for backward compatibility.
    Loads centralized split and returns tuple format expected by older code.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, centers)
    """
    split = load_centralized_split(dataset, data_seed, base_dir, verbose=False)
    return split.X_train, split.X_test, split.y_train, split.y_test, split.centers
