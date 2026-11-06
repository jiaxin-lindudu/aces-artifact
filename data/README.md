# `data` 包说明

`data/` 负责**数据集获取、预处理（缩放/PCA）、集中式 FIT/TEST 划分、联邦 Non-IID 分区**，并把每一步的产物连同元数据（含 SHA-256）落盘，保证实验可复现、可校验。

整个包遵循**两阶段流水线**，两个阶段解耦：

```
原始数据源 (sklearn / OpenML / 本地文件 / ViT 特征)
        │
        │  Stage 1  prepare_centralized_splits.py
        ▼
data/{dataset}/splits/{dataset}_centralized[_seed_{s}].npz        # FIT/TEST 划分
data/{dataset}/splits/{dataset}_centralized[_seed_{s}]_metadata.json
        │
        │  Stage 2  prepare_federated_partitions.py
        ▼
data/{dataset}/partitions/{dataset}_seed_{s}_beta_{β}_clients_{M}/
        ├── partition.npz             # 全量 FIT + 共享 TEST + centers
        ├── client_splits.json        # 每个客户端拥有的 FIT 行索引
        └── partition_metadata.json   # 分区来源与统计
```

关键术语与不变式：

- **FIT**：训练块，Stage 2 中被切分给各客户端；**TEST**：测试块，全局共享，永不参与切分。
- 缩放器与 PCA **只在 FIT 上拟合**，TEST 只做 `transform`（`scaler_fit_on: FIT block only; TEST is transformed, never fitted`）。
- 分区满足：所有客户端索引拼接排序后 `== arange(n_train)`，即每条 FIT 记录**恰好属于一个客户端，不复制、不丢弃**。

---

## 1. 文件与目录职责

| 路径 | 类型 | 职责 |
| --- | --- | --- |
| `__init__.py` | 源码 | 空文件。`data` 是普通包，需 `from data.xxx import ...`；两个 CLI 脚本会自行把仓库根目录插入 `sys.path`，因此可从任意目录直接运行。 |
| `datasets_and_utils.py` | 源码 | ① `get_dataset()` 统一加载 16 个数据集；② `apply_pca_preprocessing()` 降维；③ `DATASET_SOURCES` 固定外部数据版本；④ Stage 1/2 的读取 API（`load_centralized_split` / `load_federated_partition` / `extract_client_data` / `load_cached_split`）。 |
| `non_iid_loader.py` | 源码 | Non-IID 工具：`dirichlet_split()`（按类 Dirichlet 标签偏斜）、`add_quantity_skew()`（数量偏斜）、`compute_entropy()`、`visualize_non_iid_distribution()`。 |
| `prepare_centralized_splits.py` | CLI | Stage 1：批量生成多数据集 × 多种子的集中式划分。 |
| `prepare_federated_partitions.py` | CLI | Stage 2：在 Stage 1 产物上批量生成多 β × 客户端数的联邦分区。 |
| `extract_vit_features.py` | CLI | 离线提取 CIFAR-10 官方 train/test 的 ViT-B/16 特征（带契约校验、SHA-256、断点续跑）。 |
| `sklearn_cache/` | 运行时 | `fetch_openml` / `fetch_olivetti_faces` 的下载缓存（`data_home`）。 |
| `orl/` | 运行时 | ORL 人脸（`olivettifaces.mat`），需自备；缺失时回退到 sklearn 在线下载。 |
| `COIL20/` | 运行时 | COIL-20 的 `.mat` 兜底下载目录（OpenML 失败时才用到）。 |
| `cifar10_vit_official/` | 运行时 | `extract_vit_features.py` 产出的 `X_/y_{train,test}.npy` + `metadata.json`。 |
| `torch_cache/` | 运行时 | `TORCH_HOME`，存放 ViT-B/16 权重（脚本内强制指向此处，不写用户目录）。 |

---

## 2. 第 1 步：准备集中式划分（Stage 1）

```bash
python data/prepare_centralized_splits.py \
    --datasets digits,usps \
    --data-seeds 0,1,2,3,4 \
    --num-clusters 10 \
    --samples-per-cluster 200 \
    --apply-pca
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--datasets` | `digits` | 逗号分隔的数据集名（见 §4 支持矩阵）。`banknote` 会被内部映射为 `bank`。 |
| `--data-seeds` | `0,1,2,3,4` | 逗号分隔的随机种子，每个种子一份独立划分。 |
| `--output-dir` | `<项目根>/data` | 产物根目录。 |
| `--num-clusters` | `10` | 仅对合成数据（`blobs`/`swiss_roll`/…）生效，决定真实簇数。 |
| `--samples-per-cluster` | `200` | 仅对合成数据生效，请求样本数 = `num_clusters × samples_per_cluster`。 |
| `--apply-pca` | 恒为 `True` | 见下方注意事项。 |
| `--force` | `False` | 覆盖已存在的划分。 |
| `--quiet` | `False` | 关闭进度输出。 |

输出：

```
data/digits/splits/
  digits_centralized_seed_0.npz
  digits_centralized_seed_0_metadata.json
  digits_centralized_seed_1.npz
  ...
```

`npz` 内数组：`X_train`(float32)、`X_test`(float32)、`y_train`(int32)、`y_test`(int32)、`centers`(float32，各类原始空间均值经同一缩放/PCA 变换后的结果)。

`metadata.json` 字段：

- `dataset` / `data_seed`
- `split`：`source`、`n_train`、`n_test`、`n_classes`、`test_fraction`（实测值，`get_dataset` 固定 0.2）、`stratified`（分层抽样）
- `preprocessing`：`scaler_fit_on`、`apply_pca`、`original_features`、`final_features`、`source`
- `arrays`：每个数组的 `shape` / `dtype` / `sha256`（用于复现校验）
- `dataset_source`：`usps`、`har`、`cifar10_vit` 等固定来源信息（取自 `DATASET_SOURCES`）

**注意事项**

1. **seed 42 是特殊值**：文件名后缀为空，即 `digits_centralized.npz`；其余种子为 `..._seed_{s}.npz`。Stage 2 与读取 API 使用完全相同的规则。
2. **幂等缓存**：`.npz` 与 `_metadata.json` 同时存在且未加 `--force` 时直接复用并打印 `✓ Split exists`。因此修改 `--num-clusters` / `--samples-per-cluster` / 上游数据后必须加 `--force`。
3. `--apply-pca` 定义为 `store_true` + `default=True`，**命令行无法关闭 PCA**；只有在 Python 中直接调用 `prepare_centralized_split(..., apply_pca=False)` 才能关闭（见 §5）。不过 PCA 只作用于 §4 中列出的高维数据集且要求 `dim > 64`。
4. 划分比例固定为 `test_size=0.2` 且分层；`experiments/scripts/make_partition.py` 的注释说明过：改成 7:3 会重新拟合缩放器、改变所有已发表数值，因此这里如实记录 0.2。

---

## 3. 第 2 步：准备联邦分区（Stage 2）

```bash
python data/prepare_federated_partitions.py \
    --datasets digits,usps \
    --data-seeds 0,1,2,3,4 \
    --betas 0.1,0.5,1.0 \
    --clients 20
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--datasets` | `digits` | 逗号分隔。 |
| `--data-seeds` | `0,1,2,3,4` | 必须与 Stage 1 使用的种子一致，否则找不到输入文件。 |
| `--betas` | `0.1,0.5,1.0` | Dirichlet 浓度参数：β 越小标签偏斜越强（Non-IID 越强），β ≥ 1 趋于 IID。 |
| `--clients` | `10` | **单个整数**，不支持 `10,20` 这种列表写法；需要多个客户端数请多次运行。 |
| `--input-dir` / `--output-dir` | `<项目根>/data` | 分别指定 Stage 1 输入根与分区输出根。 |
| `--force` | `False` | 覆盖已存在的分区。 |
| `--quiet` | `False` | 关闭进度输出。 |

输出：

```
data/digits/partitions/
  digits_seed_0_beta_0.1_clients_20/
    partition.npz            # X_train, y_train, X_test, y_test, centers
    client_splits.json       # {"0": {"indices": [...], "size": 89, "class_distribution": [...]}, ...}
    partition_metadata.json
  digits_seed_0_beta_0.5_clients_20/
  ...
```

`partition_metadata.json` 字段：`dataset`、`data_seed`、`federated`（`num_clients`、`dirichlet_beta`、`client_sizes`、`mean/min/max_size`）、`split`（沿用 Stage 1 统计）、`preprocessing`（从 Stage 1 元数据透传）、`scope`、`client_indices_file`、`partition_arrays_file`。

**分区算法与不变式**

1. `dirichlet_split(labels, n_clients, alpha=β, random_state=data_seed)`：按`np.unique(labels)`的实际标签排序遍历，对每个类别采样一次 `Dirichlet(β,...,β)`，得到该类样本在 M 个客户端间的比例，再按此比例切分该类索引；最后打乱每个客户端内部顺序。这产生的是**标签偏斜（label skew）**，各客户端样本量也会随之波动。标签无需从0开始；对于已有0开始的连续标签，类别顺序和随机数调用保持不变。新Stage 2 metadata的`split.class_labels`对应`client_splits.json`中`class_distribution`各位置的原始类别，向量长度始终等于K。
2. **空客户端修复（min-size = 1 策略）**：若某客户端为空，从当前样本最多的客户端**尾部借 1 条**移入。全程不复制、不删除、不重采样。
3. 落盘前 `assert` 校验：客户端索引拼接排序后等于 `arange(n_train)`。

---

## 4. 数据集支持矩阵

`get_dataset(name, n_samples, n_clusters, n_features, random_state=42)` 是统一入口，返回 `(X_train, X_test, y_train, y_test, centers_scaled)`。

| `name` | 来源 | 规模 / 特征 / 类别 | 缩放 | PCA |
| --- | --- | --- | --- | --- |
| `blobs` | `make_blobs`（合成） | 由 `n_samples`、`n_clusters`、`n_features` 决定 | StandardScaler | 否 |
| `iris` | sklearn | 150 × 4，3 类 | StandardScaler | 否 |
| `wine` | sklearn | 178 × 13，3 类 | StandardScaler | 否 |
| `digits` | sklearn `load_digits` | 1797 × 64，10 类 | **MinMaxScaler** | 否（维度 64，不满足 >64） |
| `mnist` | OpenML `mnist_784` v1 | 70000 × 784，10 类 | MinMaxScaler | → 64 |
| `mnist_10000` | 官方MNIST原始IDX；固定分层子集 | 10000 × 784，10类；8000 FIT / 2000 TEST | MinMaxScaler | → 64 |
| `covtype_10000` | sklearn官方fetch_covtype；固定subset_seed=42分层子集 | 10000 × 54，7类；8000 FIT / 2000 TEST | FIT-only StandardScaler | 保留54维 |
| `usps` | OpenML `data_id=41082` v2 | 9298 × 256，10 类 | MinMaxScaler | → 64 |
| `coil20` | OpenML `coil-20` v2，失败则下载 `COIL20.mat` | 1440 × 1024，20 类 | MinMaxScaler | → 64 |
| `bank`（`banknote`） | OpenML `banknote-authentication` v1 | 1372 × 5，2 类 | StandardScaler | 否 |
| `har` | OpenML `data_id=1478` v1（UCI HAR） | 10299 × 561，6 类 | StandardScaler | → 64 |
| `orl` | 本地 `data/orl/olivettifaces.mat`，缺失则在线下载 | 400 × 4096，40 类 | MinMaxScaler | → 64 |
| `cifar10_vit` | 本地 `data/cifar10_vit_official/` | 官方 train 50000 / test 10000 × 768，10 类 | **无**（已在提取阶段完成） | → 128 |
| `swiss_roll` | `make_swiss_roll`（合成） | 按流形参数 `t` 均匀切成 `n_clusters` 段作真值 | StandardScaler | 否 |
| `s_curve` | `make_s_curve` + `SpectralClustering` | 合成，簇数 = `n_clusters` | StandardScaler | 否 |
| `moons` | `make_moons` + 噪声维 | 合成，2 类 | StandardScaler | 否 |
| `concentric_circles` | `make_circles` | 合成，2 类（中心取中位数） | StandardScaler | 否 |
| `spiral` | 自写 3 螺旋 | 合成，3 类 | StandardScaler | 否 |

通用规则：

- 除 `cifar10_vit` 外，所有数据集都经 `train_test_split(test_size=0.2, stratify=y, random_state=seed)` 切分。
- 缩放器 `fit` 只用 `X_train`；`centers` 用**同一变换**映射，与特征保持同一空间。图像类数据集用 `MinMaxScaler` 以抑制黑色背景噪声，其余用 `StandardScaler`。
- PCA 触发条件：`apply_pca=True` 且 `dataset ∈ {orl, usps, coil20, cifar10_vit, cifar100_vit, mnist, har}` 且 `X_train.shape[1] > 64`；目标维度 cifar* 为 128，其余为 64。PCA 只在 `X_train` 上 `fit`，`X_test` 与 `centers` 只 `transform`。执行后 `cfg['num_features']` 被写回为实际维度（未执行 PCA 时写入原始维度）。

**已知边界**

- `cifar10_vit` 分支**提前 `return`**：不经过 `train_test_split`、不重新缩放，且 `centers=None`。因此该数据集的 Stage 1 产物中 `centers` 无意义，下游不要依赖它；聚类主要以 transductive 的 FIT 为主，TEST 只做原始空间代理评估（`DATASET_SOURCES['cifar10_vit']` 中已声明）。
- `mnist` 分支内部先抽了 10000 条作为评估子集，但函数末尾统一的 `train_test_split` 会覆盖 `X_test`/`y_test`，最终 TEST 仍是全量 70000 的 20%。全量数据进入谱聚类时内存压力较大，需预留资源。
- `s_curve` 内部使用 `SpectralClustering(affinity='nearest_neighbors')`，样本量大时内存/耗时呈平方级增长。
- `har` 是**记录级**模拟联邦，并非受试者不重叠；`DATASET_SOURCES['har']['partition_scope']` 已明确标注。

---

## 5. CIFAR-10 ViT 特征提取（可选前置步骤）

`cifar10_vit` 依赖本地特征目录，必须先生成：

```bash
python data/extract_vit_features.py --batch_size 32 --workers 2 --device auto
# 中断后续跑：
python data/extract_vit_features.py --resume
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--batch_size` | `32` | 写入 `contract`，仅影响推理批次，不改变特征定义。 |
| `--workers` | `2` | DataLoader 进程数。 |
| `--device` | `auto` | `auto`/`cuda`/`cpu`。显式指定 `cuda` 而 CUDA 不可用时会直接报错，**不会静默回退 CPU**。 |
| `--resume` | `False` | 续跑未完成的提取。 |

行为要点：

- 权重固定为 `ViT_B_16_Weights.IMAGENET1K_V1`，去掉分类头，输出 768 维特征；使用官方 train 50000 / test 10000。
- 确定性设置：`seed=0`、`use_deterministic_algorithms(True)`、`cudnn.benchmark=False`、关闭 TF32。
- `metadata.json` 记录完整 `contract`（数据集、切分协议、权重 URL 与 SHA-256、transform、维度、dtype、torch/torchvision/numpy 版本、设备名、原始文件 SHA-256、提取脚本自身 SHA-256 等）。
- 幂等与保护：已 `completed` 时校验四个 `.npy` 的 SHA-256 后直接返回；目录不完整时除非显式 `--resume` 否则拒绝写入；`contract` 与上次不一致时拒绝续跑（例如换了 torch 版本，应重新完整提取）。
- 依赖 `torch` / `torchvision` / `tqdm`（见 `requirements.txt`）。脚本把 `TORCH_HOME` 指向 `data/torch_cache`（ViT 权重下载于此），原始 CIFAR-10 由 torchvision 下载到 `data/`（默认 `cifar-10-batches-py`）。

---

## 6. Python 读取 API

```python
from data.datasets_and_utils import (  # noqa: E402
    get_dataset, apply_pca_preprocessing, DATASET_SOURCES,
    load_centralized_split, load_federated_partition,
    extract_client_data, load_cached_split,
)
```

### 6.1 直接使用原始数据

```python
X_train, X_test, y_train, y_test, centers = get_dataset(
    'digits', n_samples=2000, n_clusters=10, random_state=0)

cfg = {'dataset': 'usps', 'apply_pca': True, 'random_state': 0}
X_train, X_test, centers = apply_pca_preprocessing(X_train, X_test, centers, cfg)
print(cfg['num_features'])   # PCA 后会被写回为 64
```

### 6.2 集中式实验（Stage 1 产物）

```python
split = load_centralized_split('digits', data_seed=0, base_dir='data')
X_train, X_test = split.X_train, split.X_test      # float32
y_train, y_test = split.y_train, split.y_test      # int32
centers, metadata = split.centers, split.metadata
```

返回 `CentralizedData` 具名元组（`X_train, X_test, y_train, y_test, centers, metadata`）。加载时会校验 `n_train` / `n_test` 与元数据一致，不一致抛 `ValueError`。文件缺失时错误信息会直接给出应执行的 Stage 1 命令。`load_cached_split()` 是它的轻量兼容封装，返回 `(X_train, X_test, y_train, y_test, centers)` 五元组。

`centralized/run_clustering_ceiling.py` 就是通过该接口读取集中式上界实验数据的。

### 6.3 联邦实验（Stage 2 产物）

```python
partition = load_federated_partition('digits', data_seed=0, beta=0.5, num_clients=20)

# 每个客户端的本地数据
X_list = [partition.X_train[idx] for idx in partition.client_indices]
X_c0, y_c0 = extract_client_data(partition, 0)

# 共享的 TEST 块与中心
X_test, y_test, centers = partition.X_test, partition.y_test, partition.centers
```

返回 `FederatedPartition` 具名元组（`X_train, X_test, y_train, y_test, centers, client_indices, client_info, metadata`）。其中 `client_info[str(i)] = {'indices': [...], 'size': int, 'class_distribution': [...]}`，`class_distribution` 为 `np.bincount` 结果，可用于统计各客户端的标签偏斜。

加载时校验：① 客户端索引拼接排序后必须等于 `arange(n_train)`；② `num_clients` 与 `dirichlet_beta` 必须与元数据一致。任一不符即抛 `ValueError`。

接入联邦训练器时，务必把 `client_indices` 原样传下去——`federated_anchored_clustering` 会校验 `train_indices` 拼接排序后等于 `arange(len(X_train))`：

```python
from federated.ours.aces_trainer import federated_anchored_clustering

federated_anchored_clustering(
    X_list=X_list, num_clusters=K, d_emb=K,
    X_train=partition.X_train, y_train=partition.y_train,
    X_test=partition.X_test, y_test=partition.y_test,
    train_indices=partition.client_indices,
)
```

### 6.4 Non-IID 工具

```python
from data.non_iid_loader import (
    dirichlet_split, add_quantity_skew, compute_entropy, visualize_non_iid_distribution)

client_indices = dirichlet_split(y_train, n_clients=20, alpha=0.5, random_state=0)
skewed        = add_quantity_skew(client_indices, skew_factor=3.0, random_state=0)
entropy       = compute_entropy(np.bincount(y_train[client_indices[0]]))
visualize_non_iid_distribution(y_train, client_indices, K, 'digits')   # 输出到 figures/digits/
```

`add_quantity_skew` 只重排已有索引的归属，不改变样本总数，也不复制样本。

---

## 7. 复现与自检

- **哈希校验**：Stage 1 元数据记录五个数组的 `sha256`；`extract_vit_features.py` 记录特征文件、权重、原始 CIFAR 文件的哈希；`get_dataset('cifar10_vit')` 读取时会逐一比对 `metadata.json` 中的哈希，并校验 `status == completed`、`split_protocol`、`weights`、`feature_dimension == 768`，任一不符直接报错，避免静默用错版本。
- **来源固定**：`DATASET_SOURCES` 显式固定 `usps`（OpenML `data_id=41082` v2）、`har`（`data_id=1478` v1）与 `cifar10_vit` 的契约；`har` 加载后还会断言形状为 `(10299, 561)` 且类别数为 6。
- **无 TEST 泄漏**：缩放器与 PCA 只在 FIT 上拟合，元数据 `scaler_fit_on` 字段即为该声明的落盘证据。

## 8. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `✗ Split not found: ...`（Stage 2） | 该 `(dataset, seed)` 的 Stage 1 划分不存在或种子不一致，先跑 `prepare_centralized_splits.py`。 |
| `Centralized split not found`（读取 API） | 同上；错误信息里已给出可直接复制的命令。 |
| `Beta mismatch` / `Client count mismatch` | 请求的参数与目录名/元数据不符；目录名即唯一标识，请按 `{dataset}_seed_{s}_beta_{β}_clients_{M}` 取用。 |
| `--clients 10,20` 报 `invalid int value` | `--clients` 只接受单个整数，请分多次运行。 |
| ORL 报“未找到文件” | 把 `olivettifaces.mat` 放到 `data/orl/`；否则会尝试在线下载（需要网络）。 |
| COIL-20 加载慢/失败 | 首选 OpenML `coil-20` v2；失败时自动从 GitHub 下载 `COIL20.mat` 到 `data/COIL20/`。两者都失败会抛 `RuntimeError`，需检查网络。 |
| `cifar10_vit` 报“extraction is incomplete” | 先运行 `python data/extract_vit_features.py`。 |
| 换了 torch 版本后 ViT 续跑被拒 | `contract` 含版本信息，需清空/另存目录后完整重提。 |
| 改了 `--num-clusters` 但数据没变 | Stage 1 存在即复用，需加 `--force`。 |
| `load_*` 找不到文件 | `base_dir` 默认为相对路径 `data`，请从仓库根目录运行或传入绝对路径。 |

## 9. 与仓库其他部分的关系

- `centralized/run_clustering_ceiling.py`：集中式上界实验，依赖 `load_centralized_split`。
- `federated/`：联邦算法，接收 `X_list` 与 `train_indices`（即 `partition.client_indices`）。
- `experiments/scripts/make_partition.py` 等新版三阶段流水线脚本当前在仓库中处于注释状态；其注释明确把切分委托给 `data.datasets_and_utils.get_dataset`，分区语义与本文档一致。`experiments/run_ours_experiments.py` 已标记为 DEPRECATED，仍在引用旧的 `prepare_experiment_data`（已注释掉），不要再作为复现入口。


## 10. 当前已备分区（2026-10-24）

Digits的data_seed=0–4、β={0.1,0.5,1.0}、clients={10,20}共30份Stage 2分区已备齐。10客户端按原有命令单独生成，未改变既有20客户端分区或Stage 1数据：

```powershell
python data/prepare_federated_partitions.py --datasets digits --data-seeds 0,1,2,3,4 --betas 0.1,0.5,1.0 --clients 10
```

所有新增网格入口加载时会核对Stage 1/2四数组完全一致、Stage 1哈希及客户端索引的完整唯一覆盖，并在正常结果CSV记录数据/归属哈希，不额外生成校验报告。这里只代表分区已准备，尚不代表30个条件的所有方法均已训练完成。


## 11. MNIST-10000（用户选择的固定子集）

运行时使用准确名称 `mnist_10000`。从官方MNIST train+test合计70000条原始记录中，以固定subset_seed=42分层抽取同一批10000条记录，再按data_seed=0–4分别做80/20划分；这是重划分的MNIST子集，不是官方60000/10000划分。每份8000训练、2000测试，MinMaxScaler与784→64的PCA均只拟合训练数据。

原始文件位于 `data/mnist_10000/MNIST/raw/`，四份gzip文件逐一校验官方MD5；本机复用了已下载的官方原始文件，没有导入其他项目的预处理数组或客户端划分。缺失时从官方MNIST镜像下载。固定子集索引哈希、选择种子及原始文件校验值记录在现有Stage 1 metadata内，不额外建立诊断文件。

```powershell
python data/prepare_centralized_splits.py --datasets mnist_10000 --data-seeds 0,1,2,3,4 --samples-per-cluster 1000 --apply-pca
python data/prepare_federated_partitions.py --datasets mnist_10000 --data-seeds 0,1,2,3,4 --betas 0.5 --clients 20
```

本机上述两步已经完成并核对：五份划分的固定子集索引哈希相同，Stage 1/2数组完全一致，客户端索引覆盖所有训练行且不重复。其他方法以后也必须加载这同一套 `mnist_10000` 冻结数据。

## 10. eICU：正式数据访问与接入准备

状态：尚未下载正式eICU数据，当前`get_dataset`与Stage 2尚未注册eICU，不能直接运行`--datasets eicu`或报告eICU实验已完成。

为补充老师要求的真实隐私敏感跨机构数据，拟采用eICU-CRD v2.0，并以真实`hospitalid`保留医院归属，在本机模拟联邦；这不等于已经开展实际跨医院部署。现有图像/合成数据继续使用原Dirichlet分区。eICU自然分区的医院数与样本纳入规则须在取得数据后固定，不把医院编号、患者标识或出院结局作为聚类特征。

正式版需要研究者本人完成PhysioNet认证及CITI“Data or Specimens Only Research”培训，并按项目要求取得访问资格、接受数据使用条款：

- 官方申请说明：<https://eicu.mit.edu/gettingstarted/access/>
- 正式项目：<https://physionet.org/content/eicu-crd/2.0/>

建议将获准下载的原始文件保存在`data/eicu/raw/`。拟定的紧凑特征入口优先需要以下四张表（允许CSV或CSV.gz）：

```text
patient.csv.gz
hospital.csv.gz
apacheApsVar.csv.gz
admissionDx.csv.gz
```

`patient`提供患者与医院归属，`apacheApsVar`提供生理测量，`admissionDx`提供待预先确定的外部诊断分组；诊断标签不进入训练特征。患者级FIT/TEST隔离、缺失值处理、特征范围、诊断分组与K需形成统一固定协议，再复用所有方法的冻结数组。不能直接将存活/死亡当作天然聚类真值，或套用现有图像数据的K=10。

公开Demo v2.0.1可用于检查schema与处理流程，但必须标注为Demo子集；官方描述与公开表中医院字段的含义需进一步核验，不能仅凭“来自20家医院”的描述强行构造20个真实医院客户端。此次仅在内存检查公开Demo的表结构，未保存患者记录或生成训练/测试结果。

## 11. USPS：原始标签1–10的分区兼容

本仓库固定OpenML data_id=41082。该版本的原始标签是1–10，保留冻结Stage 1中的标签及所有数组，不做标签重编码或重新拟合预处理。分区函数已改为遍历实际标签值，避免旧`range(K)`逻辑漏掉标签10；客户端类别统计使用临时连续编码，仅用于输出长度为K的统计向量，原始FIT/TEST标签不变。

五份Stage 1及beta=.5、20客户端的Stage 2已准备好，每份7438 FIT / 1860 TEST、64维、10类；Stage 1元数据哈希、Stage 1/2数组一致性及所有FIT行恰好一次归属均已核对。已有Digits、MNIST-10000、COIL20、Swiss Roll的20份冻结分区可由修复后的函数逐索引完全复现，没有重写这些旧分区。

原分区失败留下的空目录可以直接复用；当前已补齐五份标准分区文件。后续直接运行`experiments/scripts/grid_search_ours.py --dataset usps ...`，无需再次生成Stage 1。其他方法同样加载这五份USPS冻结分区。

## 12. HAR自然受试者验证

`prepare_har_subject_splits.py`读取UCI官方ZIP的561维特征、活动标签及受试者ID，
生成独立于原记录级HAR实验的`har_subject`数据。原HAR五份50客户端分区不覆盖。

30名受试者按固定种子20261104分五组，每折24名用于FIT、6名只用于TEST。
一个训练受试者是一个客户端，受试者ID不进入特征；Scaler与PCA64只拟合该折FIT。
五折TEST记录互不重叠，合计10299条，FIT仍有交叠；不是五次独立重复。

Stage 1/2继续使用同一套NPZ及必要metadata文件；自然归属目录使用
`har_subject_seed_{fold}_beta_subject_clients_24`，其中`subject`是字符串标识，
metadata的Dirichlet beta为空。不要使用Dirichlet生成器重分自然归属。

完整模型验证入口见实验README §9.22；官方原始ZIP缓存于
`data/har_subject/source/uci_har.zip`，不是诊断附件。

