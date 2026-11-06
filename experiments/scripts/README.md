# 联邦聚类超参数网格搜索（`grid_search_*.py`）

本目录下的三份脚本在**同一份冻结联邦分区**上分别扫一种方法的超参数：

| 脚本 | 方法 | 训练入口 | GRID 轴数 | 默认 `--tag` |
| --- | --- | --- | --- | --- |
| `grid_search_ours.py` | ours（锚点图谱 + 双空间模糊融合） | `federated.ours.aces_trainer.federated_anchored_clustering` | 27 | `ours` |
| `grid_search_fkm.py` | FKM（联邦 K-Means，MA/GS） | `federated.fkm.fkm_trainer.fkm_clustering` | 8 | `fkm` |
| `grid_search_ffcm.py` | FFCM（联邦模糊 C-Means） | `federated.ffcm.ffcm_trainer.ffcm_clustering` | 5 | `ffcm` |

`grid_search_fkm.py` 是 `grid_search_ours.py` 的镜像，`grid_search_ffcm.py` 又是 `grid_search_fkm.py` 的镜像。三者的**命令行、分区加载、FIT 重建、聚合方式、CSV 命名完全一致**，区别只有三点：GRID 轴、被调用的 trainer、以及 trainer 返回值的解包方式。

公平比较靠三份契约支撑：

1. **同一份分区**：`dataset / --grid-split / --beta / --clients` 唯一定位一份 Stage 2 冻结分区，FIT 行、客户端归属、TEST 块三者同源。
2. **同一选择信号**：只有 FIT 的 `acc/nmi/ari` 用于挑配置（`--select-on`），TEST 指标逐行记录但**绝不参与排序**。
3. **近乎同一的结果列**：fkm 与 ffcm 的 `RESULT_COLUMNS` 逐列完全相同；ours 只多出本方法特有的诊断列（`stop_reason`、`final_center_shift_orig`、`test_basis`），少了 `final_inertia` —— 三份 CSV 可直接并排读（见第 8 节）。

---

## 1. 快速开始

```bash
# 只看网格定义，不训练
python experiments/scripts/grid_search_ffcm.py --dataset digits --dry-run

# 标准跑法：一份分区 x 3 个 init 种子
python experiments/scripts/grid_search_ours.py --dataset digits --beta 0.5 --clients 20 --init-seeds 0,1,2
python experiments/scripts/grid_search_fkm.py  --dataset digits --beta 0.5 --clients 20 --init-seeds 0,1,2
python experiments/scripts/grid_search_ffcm.py --dataset digits --beta 0.5 --clients 20 --init-seeds 0,1,2

# 全笛卡尔积（规模大，需要 --force）
python experiments/scripts/grid_search_ffcm.py --dataset digits --mode cartesian --force

# 想看 trainer 内部 print（默认是吞掉的）
python experiments/scripts/grid_search_ffcm.py --dataset digits --init-seeds 0 --trainer-logs
```

分区文件缺失时脚本会直接给出生成命令（两步：`prepare_centralized_splits.py` -> `prepare_federated_partitions.py`）。

---

## 2. 统一执行流程

单文件线性流水线，无跨脚本抽象：

```text
CLI 固定外部输入 (dataset/grid-split/beta/clients/init-seeds)
  |-- build_grid: oat 或 cartesian
  |-- load_partition_bundle: 加载冻结分区 + 重建自洽 FIT 块
        -> 逐配置 x 逐 init-seed
             -> run_ours / run_fkm / run_ffcm (调用各自的 trainer)
                  -> clustering_quality (FIT/TEST acc-nmi-ari + 派生指标)
                       -> pandas 按 GRID 轴聚合 + std
                            -> 写 *_mean.csv 与 *.csv + 终端汇总表
```

| 阶段 | 主要符号 | 做什么 |
| --- | --- | --- |
| 网格定义 | `GRID`、`ALLOW_POOLED_INIT` | 所有可调超参及其候选值写在脚本里，**没有**对应的命令行开关 |
| 网格展开 | `build_grid(mode)` | `oat` = baseline（每轴取首值）+ 逐轴单值变体；`cartesian` = 全组合 |
| 数据加载 | `load_partition_bundle()` | 读分区、按 `client_indices` 切出 `X_list`、重建 `X_fit/y_fit/train_indices` |
| 单次运行 | `run_ours()` / `run_fkm()` / `run_ffcm()` | 喂一个配置给 trainer，取回中心并算指标 |
| 聚合 | `main()` 内的 `groupby` | 按 GRID 轴求均值，另对 `acc/nmi/ari` 求 `ddof=1` 标准差，排序后取名次第一的配置 |
| 落盘 | `frame.to_csv` / `mean.to_csv` | 逐次运行全量表 + 按配置聚合表 |

规模护栏：`MAX_RUNS_WITHOUT_FORCE = 2000`，即 `配置数 x init-seed 数` 超过它必须先 `--force`。

---

## 3. 命令行

三份完全一致（只有 `--tag` 默认值不同）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--dataset` | `digits` | 数据集名，决定分区目录与输出子目录 |
| `--clients` | `20` | 客户端数，分区身份的一部分 |
| `--beta` | `0.5` | Dirichlet 集中度，**分区身份**，不是网格轴 |
| `--grid-split` | `0` | 分区的 `data_seed`；FIT 与 TEST 都来自它 |
| `--init-seeds` | `0,1,2` | 逗号分隔的初始化种子，每份配置跑一遍取均值 |
| `--data-dir` | `<ROOT>/data` | 分区根目录 |
| `--mode` | `oat` | `oat` / `cartesian` |
| `--select-on` | `acc` | 排序用的 FIT 指标，`acc` / `nmi` / `ari` |
| `--out-dir` | `None` | 默认 `<RESULTS_DIR>/{dataset}/federated_{grid-split}` |
| `--tag` | `ours`/`fkm`/`ffcm` | 只影响输出文件名 |
| `--dry-run` | 关 | 打印完整网格定义后退出，不训练 |
| `--force` | 关 | 越过运行量护栏；同时把单格异常降级为 NaN 而不是中断 |
| `--trainer-logs` | 关 | 放行 trainer 与 `initialize_centers` 的 `print` |

> `grid_search_ours.py` 里 `--out-dir` 的 help 文本写的是 `results/{dataset}/federated_{grid_split}`，实际默认值由 `experiments/schema.py:RESULTS_DIR` 决定，是 `experiments/results/{dataset}/federated_{grid_split}`。三份脚本行为一致，只有 ours 的 help 文案少了一层。

---

## 4. 数据契约：分区加载与 FIT 重建

`load_partition_bundle()` 在三份脚本中逐行相同：

1. `data/datasets_and_utils.py:491:load_federated_partition(dataset, data_seed, beta, num_clients, base_dir)` 读出分区（`metadata / X_train / X_test / y_train / y_test / client_indices`）。
2. `X_list = [X_train[idx] for idx in client_indices]` —— 客户端视角的数据。
3. 重建一块**自洽的 FIT 矩阵**：`X_fit = np.vstack(X_list)`、`y_fit = np.concatenate([y_train[idx] ...])`，并用 `offsets = np.cumsum([0]+sizes)` 生成连续归属 `train_indices`。三条 assert 保证「块数 == 客户端数」「拼接后覆盖 `arange(len(X_fit))`」「`X_fit[train_indices[i]]` 与 `X_list[i]` 逐行相等」。
4. `K` 取 `metadata.split.n_classes`，缺失时退回 `len(np.unique(y_train))`；`median_distance = median(pdist(X_train))`（用全 `X_train`，不是 `X_fit`），只用于 `center_spread_ratio` 的归一化。

bundle 放在模块级 `_PARTITION_CACHE` 里，整轮网格只读一次盘。

指标侧的一致约定：

| 量 | 算法 |
| --- | --- |
| `acc` | 硬指派到最近中心后，`scipy.optimize.linear_sum_assignment` 做 Hungarian 对齐 |
| `nmi` / `ari` | sklearn 同名度量 |
| `final_inertia` | FIT 行到所属中心的均方距离（fkm / ffcm） |
| `center_spread_ratio` | 平均中心间距除以 `median_distance`，趋近 0 即中心坍缩 |
| `rounds_run` | ours 取 `history['completed_rounds']`（有早停）；ffcm 等于 `n_rounds`（无早停） |

---

## 5. 网格定义

### 5.1 `grid_search_ours.py`（27 轴）

范围全部写在 `GRID` 字典里，按类目分组。当前为Swiss Roll的36组OAT邻域网格（每轴首值为共同基准），依据首轮FIT均值，仅调整fuzziness、kernel_sigma、lambda_space、k_neighbors、anchor_weight五项；其余候选保持已完成Swiss Roll首轮的设置，学习率与轮数不扩展。保持原有单文件入口；它会作用于此后调用本脚本的任何数据集。此前Digits/MNIST的36组、COIL20及首轮Swiss Roll的22组网格结果保留，不能用当前候选列表描述旧运行。

**双空间模糊模型（Eq. 11）**

| 轴 | 默认值 | 候选值 |
| --- | --- | --- |
| `fuzziness` | `1.05` | `1.05, 1.01, 1.025, 1.075, 1.10, 1.20, 1.30` |
| `lambda_space` | `0.9` | `0.9, 0.8, 0.85, 0.875, 0.925, 0.95, 0.975, 1.0, 0.75, 0.65, 0.5` |
| `d_emb` | `2` | `2, 3, 4, 6, 8`（恢复原候选） |

**局部锚点图**

| 轴 | 默认值 | 候选值 |
| --- | --- | --- |
| `kernel_sigma` | `None` | `None`（恢复高FIT配置使用的自适应尺度；不在self_tuning中空扫固定sigma） |
| `kernel_scale_mode` | `self_tuning` | `self_tuning`, `global` |
| `k_neighbors` | `5` | `5, 4, 6, 7` |
| `anchor_weight` | `1.0` | `1.0, 0.5, 0.75, 1.25, 1.5, 2.0` |
| `graph_method` | `knn` | `knn` |
| `use_anchors` | `True` | `True`（关掉则无 Nyström OOS，TEST 退回原始空间代理） |

**联邦优化**

| 轴 | 默认值 | 候选值 |
| --- | --- | --- |
| `num_rounds` | `20` | `20, 10, 40`（保持首轮设置） |
| `active_ratio` | `1.0` | `1.0` |
| `init_method` | `random` | `random`, `uniform` |

**拓扑一致性精修**

| 轴 | 默认值 | 候选值 |
| --- | --- | --- |
| `enforce_topology` | `True` | `True` |
| `topology_epochs` | `20` | `20` |
| `learning_rate` | `1e-3` | `1e-3, 5e-3`（保持首轮设置） |
| `topology_fidelity_weight` | `1.0` | `1.0` |
| `topology_topology_weight` | `1.0` | `1.0` |
| `topology_separation_weight` | `1.0` | `1.0` |
| `topology_boundary_mode` | `tanh` | `tanh`, `clamp` |
| `topology_max_displacement` | `1.0` | `1.0`（仅 `clamp` 用得上） |
| `topology_early_stopping` | `True` | `True` |
| `topology_early_stopping_patience` | `5` | `5` |

**融合 / 对齐开关与收敛判据**

| 轴 | 默认值 | 候选值 |
| --- | --- | --- |
| `normalize_spaces` | `True` | `True`, `False` |
| `match_procrustes_scale` | `True` | `True` |
| `use_procrustes` | `True` | `True` |
| `tol_center_shift` | `1e-4` | `1e-4` |
| `early_stopping_patience` | `3` | `3` |

打印时 `kernel_sigma` 的 `None` 会被 `sigma_label()` 渲染成 `adaptive`。

### 5.2 `grid_search_fkm.py`（8 轴）

| 轴 | 默认值 | 候选值 | 备注 |
| --- | --- | --- | --- |
| `method` | `ma` | `ma`, `gs` | Model Averaging / Gradient Sharing |
| `n_rounds` | `50` | `50, 100, 150` | |
| `Q` | `5` | `5, 10, 20` | 每客户端每轮局部迭代数 |
| `B` | `32` | `32, 64` | mini-batch 大小 |
| `alpha` | `None` | `None` | `None` = 全体客户端参与 |
| `learning_rate` | `0.5` | `0.5, 0.8, 1.0` | **映射到 trainer 的 `beta=`** |
| `weighted` | `True` | `True`, `False` | 按簇计数加权平均 |
| `center_init_method` | `random` | `random`, `uniform` | 放开 `ALLOW_POOLED_INIT` 后追加 `federated_kmeans++` |

命名由来：分区身份列已经占用了 `beta`（Dirichlet 集中度），若网格轴也叫 `beta`，结果表会出现重复列。因此轴名取 `learning_rate`，调用时 `kwargs.pop('learning_rate')` 再传 `'beta': learning_rate`。

### 5.3 `grid_search_ffcm.py`（5 轴）

| 轴 | 默认值 | 候选值 | 备注 |
| --- | --- | --- | --- |
| `fuzziness` | `1.1` | `1.1, 1.3, 1.5, 2.0` | **映射到 trainer 的 `m=`**；取值与 `centralized/grids.py:24-29` 的 `FCM_GRID` 一致 |
| `n_rounds` | `10` | `10, 20, 30` | 无早停，全部跑满 |
| `center_init_method` | `random` | `random`, `uniform` | 放开 `ALLOW_POOLED_INIT` 后追加 `client`, `kmeans_plusplus` |
| `check_collapse` | `True` | `True`, `False` | 首值取 `True` 是刻意的，见下 |
| `min_separation` | `0.05` | `0.05, 0.15, 0.30` | **仅在 `check_collapse=True` 时生效** |

两条设计约束：

- **`check_collapse` 的 baseline 必须是 `True`**。trainer 只在开关打开时才用 `min_distance_threshold = min_separation * 0.5` 做坍缩检测、并在坍缩时按 `min_separation` 施加斥力（修复时用的 `data_std=0.3` 在 trainer 内硬编码）。若 baseline 取 `False`，`min_separation` 整条轴都会退化成 no-op 单元格。
- **保留 `fuzziness >= 1.5` 的档位是有意的**。成员度指数是 `2/(m-1)`，`m >= 1.5` 在高维空间基本落在「均匀成员度」不动点上，这些格子是「坍缩是 `m` 的性质而非某次运行的偶然」的证据（与集中式 `FCM_GRID` 同理）。

### 5.4 初始化方法的两条注意事项

- **隐私**：`client` / `kmeans_plusplus`（FFCM）、`federated_kmeans++`（FKM）都需要样本级访问 —— trainer 会自己传 `np.vstack(X_list)`（池化 FIT 块）。故默认排除，只有把 `ALLOW_POOLED_INIT` 改成 `True` 才会追加，且这些行会被标记为 `init_uses_pooled_fit=True`，永不与隐私合规的运行混淆。
- **尺度**：`federated/utils/center_utils.py` 的 `initialize_centers` 硬编码 `data_normalized=True`，`random` 恒从 `N(0,1)` 采样、`uniform` 恒从 `U(0,1)` 采样，**与数据实际尺度无关**。这正是 `experiments/baselines_adapter.py:76-84` 对 FFCM 改用 `kmeans_plusplus` 的原因。两份基线脚本仍保留 `random/uniform`，让网格把这种敏感性显示出来，而不是藏起来。

---

## 6. trainer 返回值契约（最容易踩错的地方）

三份脚本的 trainer **返回形态各不相同**，不能互相照抄解包：

| 脚本 | 返回分支 | 本脚本的钉法 | 解包写法 |
| --- | --- | --- | --- |
| ours | **恒定四元组** `(centers_orig, centers_emb, history or None, tracker)`（`aces_trainer.py:581`） | `track_convergence=True`, `record_history=True` | `centers, _emb, history, tracker = ...`，随后读 `history['train_labels']`、`history['test_labels']`、`history['completed_rounds']`、`history['stopping_configuration']['reason']` |
| fkm | 四分支（`fkm_trainer.py:161-168`）：`(centers, history, tracker)` / `(centers, tracker)` / `(centers, history)` / `centers` | `use_tracker=True`, `record_history=False` | `centers, tracker = result` |
| ffcm | **只取决于 `record_history`**（`ffcm_trainer.py:141-144`）：`True` 返回 `(centers, history)`，`False` 返回裸 `centers` | `record_history=False`, `use_tracker=False` | `centers = result` |

两个必知陷阱：

1. **FFCM 的 `use_tracker=True` 不返回 tracker**。tracker 在内部逐轮构建并 `update`，结束时直接丢弃，返回值仍只由 `record_history` 决定。若把 FKM 的 `centers, tracker = result` 搬过来，会对形状 `(K, M)` 的裸数组做二元组解包，`K != 2` 时抛 `ValueError`。
2. **FKM 那边的 tracker 其实是纯开销**。它开着 `use_tracker=True`，于是每轮都在 FIT 与 TEST 上重算 silhouette / Davies-Bouldin / Calinski-Harabasz，而脚本解包后并未使用 `tracker`。FFCM 脚本刻意把开关关掉，避免了同样的浪费。

---

## 7. FIXED_EVAL：为什么会不一样

`FIXED_EVAL` 会被整体 splat 进 trainer，**因此里面的键必须是 trainer 的真实形参**：

| 脚本 | 键 | 为什么 |
| --- | --- | --- |
| ours | `use_ckks=False, track_convergence=True, dp_sigma=0.0, run_privacy_eval=False, test_embedding_mode='nystrom', test_landmark_mode='stored', protocol='ours', beta=0.5` | `federated_anchored_clustering` 确有 `protocol` 形参（`aces_trainer.py:266`）；`beta` 是 `lambda_space` 的别名，显式传了 `lambda_space` 时惰性 |
| fkm | `dp_sigma=0.0, record_history=False, use_tracker=True` | `'protocol': 'fkm'` **被注释掉**（`grid_search_fkm.py:111`）—— FKM trainer 不接受 `**kwargs` |
| ffcm | `dp_sigma=0.0, record_history=False, use_tracker=False` | 同理，刻意不含 `protocol` 键 |

`dp_sigma = 0.0`（差分隐私噪声）在三份里都归入 `FIXED_EVAL`：它是隐私旋钮，不是效用超参，不参与网格。

其余通用约定：

- **日志**：默认用 `contextlib.redirect_stdout` 把 trainer、`initialize_centers`、坍缩修复的 `print` 收进 `io.StringIO`，`--trainer-logs` 才放行。
- **异常**：默认直接抛出以尽早暴露问题；`--force` 下由 `_failed_metrics()` 把该行指标填 NaN（并保留 `init_uses_pooled_fit=False`）后继续。
- **热路径**：FFCM 的成员度函数（`ffcm_clustering.py:28-38`）是 `O(N*K^2)` 的纯 Python 双层循环，digits 约 0.05 s/轮，是其主要开销来源；这也是不开 `use_tracker` / `record_history` 的原因（避免每轮再叠一层 silhouette 计算与中心拷贝）。

---

## 8. 产物与跨方法 CSV 对比

每次运行产出两份文件，落在 `--out-dir`（默认 `experiments/results/{dataset}/federated_{grid-split}/`）：

| 文件 | 粒度 | 列 |
| --- | --- | --- |
| `grid_{tag}_beta{beta}_clients{N}.csv` | 每个 (配置, init_seed) 一行 | 身份列 + GRID 轴 + 全部结果列 |
| `grid_{tag}_beta{beta}_clients{N}_mean.csv` | 每个配置一行 | GRID 轴 + **数值列的均值** + `acc_std/nmi_std/ari_std` + 可读的 `config` 列，**已按 `--select-on` 降序排好** |

要点：

- `stem` 里 `--beta` 用 `'%s' % float(beta)` 渲染（如 `0.5`），`--dataset` 段经过 `_slug()`。
- 身份列三份一致，只做溯源、不是网格轴：`split, dataset, data_seed, beta, clients, init_seed`。
- **`_mean.csv` 只保留数值列**，`stop_reason` 与 `test_basis`（ours 的字符串诊断列）只出现在逐次运行的 CSV 里。
- fkm 与 ffcm 的 `RESULT_COLUMNS` 逐列相同：`acc, nmi, ari, test_acc, test_nmi, test_ari, n_distinct_labels, center_spread_ratio, rounds_run, final_inertia, init_uses_pooled_fit, wall_seconds`。ours 少了 `final_inertia`，多了 `stop_reason, final_center_shift_orig, test_basis`。
- `--init-seeds` 只给一个种子时，`acc_std/nmi_std/ari_std` 会是 NaN（`std` 用 `ddof=1`，n=1 无定义）。

### 8.1 各方法挑出的最佳配置

```python
import pandas as pd
from pathlib import Path

OUT = Path('experiments/results/digits/federated_0')
CORE = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari']

mean = {tag: pd.read_csv(OUT / f'grid_{tag}_beta0.5_clients20_mean.csv')
        for tag in ('ours', 'fkm', 'ffcm')}

# _mean.csv 已按 --select-on（默认 FIT acc）降序排好，首行就是该脚本挑中的配置
best = pd.concat([frame.head(1).assign(method=tag) for tag, frame in mean.items()])
print(best[['method', 'config'] + CORE].round(4).to_string(index=False))
```

### 8.2 三种方法的分数分布（不只看最优）

```python
pooled = pd.concat([frame.assign(method=tag) for tag, frame in mean.items()])
print(pooled.groupby('method')[CORE].agg(['max', 'mean', 'std']).round(4))
```

`max` 与 `mean` 的差距能直接暴露"某个轴上一两格恰好跑赢"的偶然性 —— 这也是为什么要多个 init 种子取均值。

### 8.3 对齐三份表的列差异

三份表的 **GRID 轴列完全不同**（27 / 8 / 5 轴），因此先只在公共的指标列上比：

```python
METRIC_COLS = ['acc', 'nmi', 'ari', 'test_acc', 'test_nmi', 'test_ari',
               'n_distinct_labels', 'center_spread_ratio', 'rounds_run',
               'final_inertia', 'final_center_shift_orig', 'wall_seconds',
               'acc_std', 'nmi_std', 'ari_std']

for tag, frame in mean.items():
    print(tag, '缺失:', [c for c in METRIC_COLS if c not in frame.columns])
# ours -> 缺 final_inertia
# fkm  -> 缺 final_center_shift_orig
# ffcm -> 缺 final_center_shift_orig
```

### 8.4 逐次运行表的配对读法

fkm 与 ffcm 的逐次运行表**共享身份列与全部结果列**，但各自的 GRID 轴列不同（8 轴 vs 5 轴），纵向拼接后缺失的轴列会被自动填成 NaN —— 这不影响只按结果列做聚合：

```python
runs = pd.concat(
    {tag: pd.read_csv(OUT / f'grid_{tag}_beta0.5_clients20.csv')
     for tag in ('fkm', 'ffcm')},
    names=['method', 'row']).reset_index(level=0)

print(runs.groupby(['method', 'init_seed'])['acc'].agg(['mean', 'max']).round(4))
```

若某行指标全为 NaN，说明该配置在 `--force` 下失败并被 `_failed_metrics()` 填成 NaN，而不是被静默丢弃。




## 9. 新增四种基线（与原三份网格脚本同一路径约定）

| 脚本 | GRID | 固定设置 / 初始化 |
| --- | --- | --- |
| `grid_search_nnfc.py` | k_neighbors={2,5,10,20} | dp_sigma=0；本地KMeans++ / n_init=10；本地锚点数max(1,min(n_i//3,50)) |
| `grid_search_kfed.py` | local_clusters={floor(sqrt(K)),ceil(K/2),K}去重 | author实现；局部迭代上限300；本地谱初始化及单次D平方抽样 |
| `grid_search_fedsc.py` | S={10,20,50}, sigma={1,5,10}, d={3K,5K} | Q=20、eta=.01、lambda_reg=.01；随机公共字典 |
| `grid_search_fastlloyd.py` | num_rounds={2,1,3,5,10,20,50}, radius_alpha={.5,.1,.25,1,2} | ε=3、δ=1e-5；Digits公共域[0,1]，内部offset=.5、bound=.5；其他数据集保持offset=0、bound=3；公共sphere packing；DP随机数不复用初始化种子 |

这四份脚本仍然是独立文件，使用与原三份一致的 `load_partition_bundle / build_grid / run_model / main` 顺序。它们不调用旧 `baselines_adapter.py` 或旧 `run_sweep.py`。k-FED、FedSC、FastLloyd核心来自原仓库，数据准备与实验入口未搬入；数值训练收到的FIT/TEST真实标签均为None。

NNFC保留minimal原核心，关键SNN步骤已对照作者的 `mlyizhang/nnfc`（`codes/utils.py`，SHA ff0daf4e8976da2031222e1da65eccdb0cdde760）。只加通信/计时及回退标记；若旧核心触发随机回退，新网格将该单元格记为失败，不用随机中心冒充NNFC。

### 9.1 运行与路径

在本项目根目录执行；Windows可先设置 `$env:PYTHONIOENCODING='utf-8'`。

```powershell
python experiments/scripts/grid_search_nnfc.py --dataset digits --dry-run
python experiments/scripts/grid_search_kfed.py --dataset digits --mode cartesian --dry-run
python experiments/scripts/grid_search_fedsc.py --dataset digits --dry-run
python experiments/scripts/grid_search_fastlloyd.py --dataset digits --dry-run

# 各配置在同一份冻结分区上运行五个种子；默认是OAT。
python experiments/scripts/grid_search_nnfc.py --dataset digits --grid-split 0 --beta 0.5 --clients 20 --init-seeds 40,41,42,43,44
python experiments/scripts/grid_search_kfed.py --dataset digits --grid-split 0 --beta 0.5 --clients 20 --init-seeds 40,41,42,43,44
python experiments/scripts/grid_search_fedsc.py --dataset digits --grid-split 0 --beta 0.5 --clients 20 --init-seeds 40,41,42,43,44
python experiments/scripts/grid_search_fastlloyd.py --dataset digits --grid-split 0 --beta 0.5 --clients 20 --init-seeds 40,41,42,43,44
```

上述四次运行已完成，默认位置已存在结果。再次执行时请传新 `--tag` 或 `--out-dir`，例如 `--out-dir experiments/results/digits/federated_0_repeat`；新脚本拒绝覆盖。所有脚本的数据默认值都是 `<ROOT>/data`；输出默认值都是 `experiments/results/{dataset}/federated_{grid-split}/`。`--data-dir/--out-dir/--tag` 仍可显式指定。

K相关网格在运行时从冻结元数据解析；无数据的dry-run显示K=10示例并说明表达式。默认初始化种子仍是0,1,2，为论文比较应显式传40–44。当前四方法的默认OAT配置数分别为4、3、6、11，五种子共120次；旧FastLloyd网格为5组，历史90次结果不受影响。

### 9.2 仅输出两份标准网格CSV

仍为 `grid_{tag}_beta{beta}_clients{M}.csv` 与同名 `_mean.csv`，不创建单次JSON、诊断日志或额外报告。

逐次CSV在原身份列/指标列后增加：

- `data_sha256/partition_sha256`：四个规范顺序FIT/TEST数组的组合哈希和客户端归属哈希，加载时同时核对Stage 1元数据。原三份网格脚本以后运行也写这两列，不修改已有结果。
- `fit_basis/test_basis/initialization`：保留k-FED原生FIT映射和FedSC谱标签，不能改为最近中心来提高FIT分数；各TEST代理单独标注。
- `uplink_bytes/downlink_bytes/total_bytes/bytes_per_round/communication_basis`：真实应用层负载，不是网卡流量；每轮均值包含该方法最终发布。NNFC/k-FED恰好一轮。
- `training_seconds/local_seconds/aggregation_seconds`：算法阶段耗时；FedSC单独用 `stage_I_seconds/stage_II_seconds`，不把含服务端工作的Stage I冒称纯本地时间。
- `privacy_status/effective_epsilon/effective_delta`：NNFC的epsilon=0表示关闭噪声，不表示严格DP；FastLloyd仅对固定变换输入的本次拟合计量，不是端到端DP部署。
- `status/error/hit_cap`：失败保留在逐次CSV；`--force`可继续，不能将缺少种子或NaN的配置算作完整。均值表增加 `complete/n_seeds`，完整配置才可选为BEST。

新脚本逐行更新原始CSV以保留进度；中断时只有原始CSV不代表网格完成。FedSC入口使用既有的分块相似度重建、稀疏图求解路径（`return_similarity=False`）。加载时预算对应凝缩pdist向量及数据副本，中位数直接在临时距离向量上计算，避免额外复制；训练时按两块256行重建缓冲、因子、稀疏图和特征求解缓冲估算工作区，不再误用两张完整n×n稠密矩阵的预算。USPS 7438条FIT、d=30、K=10的训练工作区估计约0.064 GiB；这不包括Python及依赖的常驻内存，也不是进程峰值实测。仍要求工作区不超过当时可用物理内存的一半，集中式稠密谱方法的检查保持原口径；不足时明确失败，不改变FIT数据。单线程BLAS用于新脚本的本地计时；多进程同时运行时墙钟耗时仍受资源竞争影响。

### 9.3 统一比较：只产生一个比较CSV

```powershell
python experiments/scripts/compare_grid_results.py --dataset digits --grid-split 0 --beta 0.5 --clients 20 --init-seeds 40,41,42,43,44 --central-dir experiments/results/digits/seed_0_recheck
```

读取同目录各方法的原始/均值CSV，从完整配置中按FIT均值选优，再从原始行重新计算FIT/TEST均值、SD和种子数。默认额外读取 `seed_{data_seed}/` 下的三种集中式结果；本次使用 `--central-dir` 指向保留旧结果后的复查目录。

输出仅 `comparison_beta0.5_clients20.csv`，在联邦结果同目录；重新比较用新 `--out-dir`。`--results-dir` 可指定新的联邦结果目录，`--central-dir` 可指定集中式目录，`--methods` 为逗号分隔的方法名。未知/缺失方法、错分区、错哈希、缺失种子、非有限指标均不会标成完成。旧CSV没有运行哈希时标为 `legacy_unverified`；集中式触顶为 `capped`。`single_cluster_runs`记录单簇运行数，不能把完成运行等同于方法效果可靠。

2026-10-24已生成的十方法比较保留旧Ours/FKM/FFCM结果；新NNFC/k-FED/FedSC/FastLloyd均有实际运行哈希。FastLloyd当前25次均为单簇；需补公共域/初始化对照，不能用这组结果单独支撑“DP必然掉点”。


### 9.4 后续30条件扫描（未自动启动）

数据已准备，下面按原CLI逐条件执行，结果使用独立重复目录以保留已完成的单划分结果。所有可调参数仍在各脚本GRID中定义；没有另一个run_all框架。

```powershell
foreach ($dataSeed in 0..4) {
    foreach ($betaValue in @(0.1, 0.5, 1.0)) {
        foreach ($clientCount in @(10, 20)) {
            foreach ($methodName in @('ours', 'fkm', 'ffcm', 'nnfc', 'kfed', 'fedsc', 'fastlloyd')) {
                python "experiments/scripts/grid_search_$methodName.py" --dataset digits --grid-split $dataSeed --beta $betaValue --clients $clientCount --init-seeds 40,41,42,43,44 --out-dir "experiments/results/digits/federated_${dataSeed}_full"
                if ($LASTEXITCODE -ne 0) { throw "Failed: $methodName / seed=$dataSeed / beta=$betaValue / clients=$clientCount" }
            }
        }
    }
}

# 集中式按同一data_seed各跑一次，不因beta和客户端数重复。
foreach ($dataSeed in 0..4) {
    python centralized/run_clustering_ceiling.py --dataset digits --data-seed $dataSeed --init-seeds 40,41,42,43,44 --out-dir "experiments/results/digits/seed_${dataSeed}_full"
    if ($LASTEXITCODE -ne 0) { throw "Centralized failed: seed=$dataSeed" }
}
```

新网格结果存在时会拒绝覆盖，不能用 `--force` 代替续跑；检查原始CSV后换新的输出目录。比较完整扫描的某一条件时，为比较脚本指定相应的 `--results-dir` 和 `--central-dir`。跨data_seed统计应先逐划分汇总，再比较划分均值，不能把25个运行当作25份独立数据。


### 9.5 只运行 MNIST-10000 的 Ours

本机已按数据README准备好五份8000/2000划分，历史MNIST结果使用当时的36组OAT，每份5个初始化共180次，五份共900次。当前GRID已改为COIL20的22组，以下入口复跑会使用新的候选列表；不会复现旧36组扫描。无需再次运行数据准备。MNIST-10000的样本量更大，耗时明显高于Digits；开始长任务前可先把下方的 `0..4` 改成 `@(0)` 运行第一份划分。

```powershell
$projectPath = (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
Set-Location $projectPath
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
foreach ($dataSeed in 0..4) {
    $outputDir = Join-Path $projectPath "experiments/results/mnist_10000/federated_${dataSeed}_ours"
    & $pythonPath "$projectPath/experiments/scripts/grid_search_ours.py" --dataset mnist_10000 --grid-split $dataSeed --beta 0.5 --clients 20 --init-seeds '40,41,42,43,44' --mode oat --out-dir $outputDir
    if ($LASTEXITCODE -ne 0) { throw "MNIST-10000 Ours failed: data_seed=$dataSeed" }
}
```

每份输出仍只有 `grid_ours_beta0.5_clients20.csv` 与 `_mean.csv`。复跑请换新的输出目录以保留前次结果。

### 9.6 Digits 的 FastLloyd 公共域与轮数复查

Digits冻结数据是MinMax像素特征，FIT位于[0,1]。旧FastLloyd直接使用[-3,3]公共域；当前仅在该方法内部固定平移`X-.5`，用`public_bound=.5`表示同一公开[0,1]范围。该平移不拟合统计量、不使用标签，FIT/TEST均采用同一变换，中心映回原坐标后计算惯性与中心距离；不改Stage 1/2、sphere-packing初始化、DP机制或其他方法。每行记录`public_bound/public_offset`。MNIST保持offset=0、bound=3，其已有结果不重写。

固定epsilon=3、delta=1e-5，搜索轮数{2,1,3,5,10,20,50}与半径系数{.5,.1,.25,1,2}。较少轮数用于检查固定总隐私预算的分摊影响，不以提高epsilon换取质量。OAT为11组，下面Cartesian为35组；每份五个初始化，五份共875次。按FIT平均ACC选参；DP噪声独立于初始化种子，不挑选或重复抽取有利噪声。多个拟合与标签辅助选参没有联合端到端DP保证。

```powershell
$projectPath = (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
Set-Location $projectPath
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
foreach ($dataSeed in 0..4) {
    $outputDir = Join-Path $projectPath "experiments/results/digits/federated_${dataSeed}_fastlloyd_tuned"
    & $pythonPath "$projectPath/experiments/scripts/grid_search_fastlloyd.py" --dataset digits --grid-split $dataSeed --data-dir "$projectPath/data" --beta 0.5 --clients 20 --init-seeds '40,41,42,43,44' --mode cartesian --select-on acc --out-dir $outputDir
    if ($LASTEXITCODE -ne 0) { throw "FastLloyd failed: data_seed=$dataSeed" }
}
```

保留旧`federated_{s}_full`中的结果。新目录仅写该方法原始CSV及`_mean.csv`；双数据集对比使用Digits新FastLloyd、`federated_{s}_ours_refined`中的Ours、其余方法的`federated_{s}_full`，以及MNIST现有`federated_{s}_ours`与两侧`seed_{s}_full`集中式结果。跨数据集不直接平均ACC形成总排名。

本次875次运行已完成，五份划分均按FIT平均ACC选中`num_rounds=3, radius_alpha=1`。Digits的TEST ACC从旧0.121333提高到0.421889，跨五份划分均值的样本标准差为0.005536；TEST NMI=0.496438、ARI=0.285048。选中配置的单次运行仍只有3–7个有效簇，效果尚未恢复到完整10簇，不将剩余差距全部归因于DP。旧、新结果并存，不按TEST挑配置或替换个别种子。

所请求双数据集总表保存为`experiments/results/comparison_digits_mnist_10000.csv`，每个(dataset,method)一行，共20行；含FIT/TEST六项指标的均值及跨五份划分的标准差、网格规模、所选配置、预测规则、FastLloyd公共域与隐私预算及源CSV路径。Digits FastLloyd为新35组网格，MNIST FastLloyd沿用已完成的旧5组网格，搜索预算与公共域差异均在表内记录。

### 9.7 准备 COIL20 并运行22组 Ours

沿用Stage 1冻结80/20划分与FIT端拟合的MinMaxScaler、PCA64，再由Stage 2仅分配FIT。标准COIL20共1440条、20类，预期每份1152 FIT / 288 TEST；实际加载规模以生成的metadata为准，不导入原项目的划分。数据尚未在本checkout准备，先运行以下两条准备命令。每个联邦方法以后均加载这些冻结分区。

历史COIL20运行使用22组OAT：`1 + (3-1)+(5-1)+(5-1)+(2-1)+(3-1)+(3-1)+(3-1)+(2-1)+(2-1)+(2-1)+(2-1) = 22`。共同基准为m=1.05、lambda=.5、d_emb=10、self_tuning、k=5、anchor_weight=1、20轮、random、lr=.001、tanh、normalize=True；只改变一个轴，不额外组合多个轴。保留uniform初始化、锚点、拓扑和Procrustes；当时d_emb候选为10、4、8、16、20，当前已改为Swiss Roll的2、3、4、6、8。下方命令现在复跑会采用新的嵌入候选，不能视为复现历史COIL20扫描。这些是搜索候选，不是已知最优参数。

```powershell
$projectPath = (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
Set-Location $projectPath
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

# Stage 1: 冻结FIT/TEST，预处理仅拟合FIT。
& $pythonPath "$projectPath/data/prepare_centralized_splits.py" --datasets coil20 --data-seeds '0,1,2,3,4' --output-dir "$projectPath/data" --apply-pca
if ($LASTEXITCODE -ne 0) { throw 'COIL20 Stage 1 failed' }

# Stage 2: 同一份FIT分给20客户端，TEST不分配。
& $pythonPath "$projectPath/data/prepare_federated_partitions.py" --datasets coil20 --data-seeds '0,1,2,3,4' --betas 0.5 --clients 20 --input-dir "$projectPath/data" --output-dir "$projectPath/data"
if ($LASTEXITCODE -ne 0) { throw 'COIL20 Stage 2 failed' }

foreach ($dataSeed in 0..4) {
    $outputDir = Join-Path $projectPath "experiments/results/coil20/federated_${dataSeed}_ours"
    & $pythonPath "$projectPath/experiments/scripts/grid_search_ours.py" --dataset coil20 --grid-split $dataSeed --data-dir "$projectPath/data" --beta 0.5 --clients 20 --init-seeds '40,41,42,43,44' --mode oat --select-on acc --out-dir $outputDir
    if ($LASTEXITCODE -ne 0) { throw "COIL20 Ours failed: data_seed=$dataSeed" }
}
```

每份22组×5个初始化=110次，五份共550次；按FIT平均ACC选参。结果每份仅`grid_ours_beta0.5_clients20.csv`与`_mean.csv`。首次数据准备可能需要下载COIL20；已有冻结数据会复用，不传`--force`。重复训练请换输出目录以保留旧结果。

### 9.8 Swiss Roll：集中式、六基线与五项参数调整后的 Ours

五份Swiss Roll冻结数据已准备并核对：每份1600 FIT / 400 TEST、原始3维、10个沿流形参数t均匀划分的标签区间，StandardScaler只拟合FIT，不做PCA。每个data_seed同时控制合成样本生成与训练/测试划分，因此五份结果包含生成随机性；它与固定真实数据的重复划分含义不同。

按首轮五份划分的FIT平均ACC筛选邻域，而不是按TEST或参数数值大小删除：首轮全局最高配置m=1.05、lambda=1、d_emb=2的FIT均值为0.431675；lambda=0和.25配置分别为0.310875和0.356775。当前在m接近1.05、lambda较高端、k接近5、anchor_weight接近1处加密；仍保留m=1.3及lambda=.5等对照，因为不同轴的OAT得分不能当作这些参数在所有组合下的独立优劣证据。

具体五项候选见§5.1；共同基准恢复m=1.05、sigma自适应，lambda改为.9以在较高原始空间权重下仍保留谱空间贡献。kernel_scale_mode保持{self_tuning,global}，固定sigma的示例候选未作为实测低分证据。学习率仍为{.001,.005}、轮数仍为{20,10,40}、d_emb仍为{2,3,4,6,8}；其他候选均保持首轮设置。共36组OAT，每份180次、五份900次。首轮全局最高配置仍包含在新网格，旧结果全部保留；新网格不包含所有旧组合，不保证每个划分都提高。

以下只运行实验，不重新生成数据；旧`federated_{s}_ours`保留。各方法的新结果共同放在`federated_{s}_full`，集中式放在`seed_{s}_full`。仍只有各入口原有的CSV/metadata，无额外诊断报告。

```powershell
$projectPath = (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
Set-Location $projectPath
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

foreach ($dataSeed in 0..4) {
    $centralOutputDir = Join-Path $projectPath "experiments/results/swiss_roll/seed_${dataSeed}_full"
    $federatedOutputDir = Join-Path $projectPath "experiments/results/swiss_roll/federated_${dataSeed}_full"

    & $pythonPath "$projectPath/centralized/run_clustering_ceiling.py" --dataset swiss_roll --data-seed $dataSeed --data-dir "$projectPath/data" --init-seeds '40,41,42,43,44' --select-on acc --recheck-max-iter 1000 --out-dir $centralOutputDir
    if ($LASTEXITCODE -ne 0) { throw "Swiss Roll centralized failed: data_seed=$dataSeed" }

    foreach ($methodName in @('ours', 'ffcm', 'nnfc', 'kfed', 'fedsc', 'fastlloyd', 'fkm')) {
        & $pythonPath "$projectPath/experiments/scripts/grid_search_${methodName}.py" --dataset swiss_roll --grid-split $dataSeed --data-dir "$projectPath/data" --beta 0.5 --clients 20 --init-seeds '40,41,42,43,44' --mode oat --select-on acc --out-dir $federatedOutputDir
        if ($LASTEXITCODE -ne 0) { throw "Swiss Roll failed: method=$methodName / data_seed=$dataSeed" }
    }
}
```

### 9.9 USPS：固定已选参数的四版本机制消融

入口为 `ablate_ours_best.py`。每份划分从已有原始CSV和 `_mean.csv` 按五个初始化种子的平均FIT ACC选出完整模型配置，保留原来的平分顺序，不使用TEST选参。原始CSV的分区身份和哈希与冻结数据核对后，复用完整模型的五条结果，仅训练三个变体：

| variant | 仅修改此参数 |
| --- | --- |
| `full` | 无，复用完整模型选中的原始行 |
| `no_topology` | `enforce_topology=False` |
| `no_procrustes` | `use_procrustes=False` |
| `original_membership` | `lambda_space=1.0` |

`original_membership` 表示仅用原始空间距离计算隶属度，仍保留谱嵌入计算和原来的拓扑项，不能称作纯FCM或完全去除谱空间。三个变体均保留锚点及原有Nyström TEST设置，记录实际 `test_basis`。轮数、学习率、初始化方式等其余参数继承该划分完整模型的已选配置，不再网格搜索，也不为各删减版本重新选参。

固定USPS、β=.5、50客户端，data_seed=0–4、init_seed=40–44；每份复用5次、新训练15次，五份合计新训练75次。`--dry-run`读取并核对已有分区和源结果、展示所选配置，不训练、不写文件。

```powershell
$projectPath = (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
Set-Location $projectPath
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

foreach ($dataSeed in 0..4) {
    $sourceDir = Join-Path $projectPath "experiments/results/usps/federated_${dataSeed}_clients50_heterogeneity"
    $outputDir = Join-Path $projectPath "experiments/results/usps/federated_${dataSeed}_clients50_ablation"

    & $pythonPath "$projectPath/experiments/scripts/ablate_ours_best.py" `
        --dataset usps `
        --grid-split $dataSeed `
        --beta 0.5 `
        --clients 50 `
        --data-dir "$projectPath/data" `
        --init-seeds '40,41,42,43,44' `
        --select-on acc `
        --source-dir $sourceDir `
        --out-dir $outputDir

    if ($LASTEXITCODE -ne 0) { throw "USPS ablation failed: data_seed=$dataSeed" }
}
```

每份只输出 `grid_ours_ablation_beta0.5_clients50.csv`（20条）及同名 `_mean.csv`（4条），均带 `variant`、实际参数和 `result_origin`；原始行记录源CSV，均值文件另含FIT/TEST六项指标的初始化标准差。`wall_seconds`保留各次训练原始值，`result_origin=reused`的完整模型耗时属于历史运行，不应计入本次新增训练时间。已有输出不会覆盖；一个划分中断后需重跑该划分，没有单次运行续跑。

### 9.10 四数据集后续实验：固定最佳配置，分批执行

数据集为Digits、USPS、COIL20、`mnist_10000`（已约定的MNIST固定1万条子集），不包含CIFAR。继续使用现有冻结FIT/TEST和训练端预处理。固定β=.5，Digits/COIL20/MNIST为20客户端，USPS为50客户端；每份data_seed=0–4、init_seed=40–44。每个划分只按源CSV的五种子平均FIT ACC选一次完整模型配置，随后固定参数；新条件不重新选参，不扩大轮数或学习率，TEST不参与选择。

| 批次 | study | 条件 | 四数据集合计新增训练 |
| --- | --- | --- | ---: |
| 第一批 | `mechanism` | 完整模型复用、去拓扑、去Procrustes、λ=1原始距离隶属度 | 300 |
| 第一批 | `participation` | active_ratio=1/.75/.5/.25，明文 | 400 |
| 第一批 | `privacy` | 明文/CKKS，均运行MIA；CKKS另运行密文攻击探针 | 200 |
| 第二批 | `space` | 复用完整模型、λ=0谱距离隶属度；λ=1复用mechanism结果，不重复训练 | 100 |
| 第二批 | `anchors` | 有/无锚点，共同维度和TEST预测口径，两边重训 | 200 |
| 第三批 | `samples` / `clients` / `clusters` | 三类CKKS通信/规模研究，各4条件 | 1200 |
| 第四批 | `quantity` / `feature` | 数量不平衡/特征漂移，各4条件，明文 | 800 |
| 第四批 | `budget` | FastLloyd ε=.5/1/3/10，固定ε=3时选中的轮数/半径 | 400 |

每类至多4个条件，上表是固定参数的对照实验，不是参数网格搜索。第一批共900次新增训练；全部批次合计3600次，分批运行。后续统计先按划分求初始化均值，再统计划分间波动，不把25次运行当作25个独立划分。不同数据集/客户端数之间不预设同一排名。

入口与输出：

- `ablate_ours_best.py --study mechanism|space|anchors`：机制、补充空间和锚点消融。λ只控制隶属度距离权重，谱计算及拓扑其余部分仍保留。锚点组两边使用 `d_emb=min(原选维度, 最小客户端样本数)`，明确记录原选/实际维度；两边TEST都用原始空间最近中心预测。无锚点同时移除锚点对齐，因此是联合移除；不能与原主表Nyström TEST直接混比。
- `run_ours_privacy_participation.py --study privacy|participation`：参与率实际人数为 `max(1,int(M*ratio))`。全部FIT/TEST仍为离线评分，未参加过的客户端可在离线评价时补嵌入，不计入通信。隐私组输出真实序列化CKKS通信、上下行/初始化/解密方传输、阶段时间、打包效率，以及MIA的AUC、TPR/TNR、退化状态和密文探针状态。收敛、对齐误差和逐轮时间存入原始CSV的 `*_trace` 列，不另写轨迹文件。
- `run_ours_scale.py --study samples|clients|clusters`：固定所选配置并开启CKKS。samples在每个原客户端内取嵌套的25%/50%/75%/100%样本，至少1条；clients固定每客户端 `floor(源FIT/100)` 条，对源FIT固定随机排列，M=10/20/50/100，总FIT随M增长；clusters固定源数据，模型K=K/2、K、2K、4K（最少2）。真实标签只用于评分。记录实际FIT/M/K/n_i、采样种子及实际数据/归属哈希。这些是源冻结FIT上的派生规模视图，不改主表分区，不冒充原Dirichlet条件。
- `run_ours_heterogeneity.py --study quantity|feature`：quantity按共同类别比例及客户端权重重新分配全部FIT，目标数量比1/2/5/10，报告实际比值和类别TV偏差；它使用新的受控数量分区，β只表示源参考。feature沿原归属，在训练端每特征标准差单位下加固定客户端方向的偏移，强度0/.25/.5/1，TEST保持公共目标域，属于跨域留出评价。记录源与派生哈希、实际偏斜及数据视图种子，不覆盖任何原分区。
- `run_fastlloyd_budget.py`：ε=3源结果按FIT选轮数/半径，四个预算都重新训练，δ固定1e-5，不对ε选优。Digits公共域为bound=.5/offset=.5，其余保持3/0；MNIST旧CSV缺公共域列，按本会话已记录的3/0协议显式标记来源。初始化种子不控制SystemRandom的DP噪声。

通信总量为up+down+interservice，down已包含setup；迭代轮均字节为 `(total_bytes-setup_bytes)/rounds_run`，setup单独列出。CKKS是单进程中独立密钥方/公开聚合方对象的聚合模拟，不是阈值CKKS或真实部署隔离；不保护最终发布模型的成员关系。MIA常数预测与退化折保留并标记，概率AUC仍可读；接近随机不构成隐私证明。缺失攻击/未运行字段保留NaN，不补零。通信仍依赖K和维度，不能声称与维度无关。

以下完整代码默认只运行第一批；后续批次仅替换 `$studies`，其余不变。所有结果只写各脚本原始CSV和 `_mean.csv`，同一划分集中放在 `federated_{seed}_clients{M}_fixed_best`，不生成报告/诊断文件，不重跑集中式或六基线大网格。

```powershell
$projectPath = (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
Set-Location $projectPath
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

# 第一批。后续批次只替换这一行。
$studies = @('mechanism', 'participation', 'privacy')
# 第二批：@('space', 'anchors')
# 第三批：@('samples', 'clients', 'clusters')
# 第四批：@('quantity', 'feature', 'budget')

$datasets = @(
    @{ Name = 'digits'; Clients = 20; Source = 'federated_{0}_heterogeneity' },
    @{ Name = 'usps'; Clients = 50; Source = 'federated_{0}_clients50_heterogeneity' },
    @{ Name = 'coil20'; Clients = 20; Source = 'federated_{0}_ours' },
    @{ Name = 'mnist_10000'; Clients = 20; Source = 'federated_{0}_ours' }
)

foreach ($dataset in $datasets) {
    $datasetName = $dataset.Name
    $clientCount = $dataset.Clients
    foreach ($dataSeed in 0..4) {
        $sourceName = $dataset.Source -f $dataSeed
        $sourceDir = Join-Path $projectPath "experiments/results/$datasetName/$sourceName"
        $outputDir = Join-Path $projectPath "experiments/results/$datasetName/federated_${dataSeed}_clients${clientCount}_fixed_best"

        foreach ($study in $studies) {
            $extraArgs = @('--study', $study)
            if ($study -in @('mechanism', 'space', 'anchors')) {
                $scriptName = 'ablate_ours_best.py'
                $tag = if ($study -eq 'mechanism') { 'ablation' } else { $study }
                $stem = "grid_ours_${tag}_beta0.5_clients${clientCount}"
            } elseif ($study -in @('participation', 'privacy')) {
                $scriptName = 'run_ours_privacy_participation.py'
                $stem = "grid_ours_${study}_beta0.5_clients${clientCount}"
            } elseif ($study -in @('samples', 'clients', 'clusters')) {
                $scriptName = 'run_ours_scale.py'
                $stem = "grid_ours_scale_${study}_beta0.5_clients${clientCount}"
            } elseif ($study -in @('quantity', 'feature')) {
                $scriptName = 'run_ours_heterogeneity.py'
                $stem = "grid_ours_${study}_sourcebeta0.5_clients${clientCount}"
            } elseif ($study -eq 'budget') {
                $scriptName = 'run_fastlloyd_budget.py'
                $stem = "grid_fastlloyd_budget_beta0.5_clients${clientCount}"
                $extraArgs = @()
            } else {
                throw "Unknown study: $study"
            }

            $rawFile = Join-Path $outputDir "${stem}.csv"
            $meanFile = Join-Path $outputDir "${stem}_mean.csv"
            if ((Test-Path -LiteralPath $rawFile) -and (Test-Path -LiteralPath $meanFile)) {
                Write-Host "跳过已有结果：$datasetName / $study / data_seed=$dataSeed"
                continue
            }

            Write-Host "`n开始：$datasetName / $study / clients=$clientCount / data_seed=$dataSeed"
            & $pythonPath "$projectPath/experiments/scripts/$scriptName" `
                --dataset $datasetName `
                --grid-split $dataSeed `
                --beta 0.5 `
                --clients $clientCount `
                --data-dir "$projectPath/data" `
                --init-seeds '40,41,42,43,44' `
                --select-on acc `
                --source-dir $sourceDir `
                --out-dir $outputDir `
                @extraArgs

            if ($LASTEXITCODE -ne 0) {
                throw "运行失败：$datasetName / $study / data_seed=$dataSeed"
            }
        }
    }
}
Write-Host "`n本批实验完成。"
```

本轮交付的是上述实验入口，正式结果需执行后核查。仍未覆盖真实机构联邦数据（当前只有模拟归属）、独立测量的进程峰值内存、额外CKKS多项式阶数/模数参数扫描；不把已有质量CSV补写成完成了这些实验。原教师文档中的预期领先趋势也必须由新结果验证。

### 9.11 全部剩余挂机实验与联邦对照补充

2026-10-28核对：四数据集第一批900次新增训练已完成，60对raw/mean文件齐全、每条件种子40–44齐全、FIT/TEST指标有限、均值核对通过。Digits20客户端和USPS50客户端已完成三档β的七方法网格比较；COIL20/MNIST主表目前只有β=.5、20客户端。主表无需重跑大网格，但新参与率/数量/特征条件仍需联邦基线对照。

完整挂机入口是 `experiments/scripts/run_remaining_experiments.ps1`，它顺序调用现有Python实验入口，不另建配置框架。已存在raw与mean的整项任务会跳过；任一任务报错则停止并保留已有完整输出，不把末尾提示当作成功。未完成的单个任务仍从头重跑，不支持内部单次运行续跑。

```powershell
& './experiments/scripts/run_remaining_experiments.ps1' `
    -PythonPath (Get-Command python -CommandType Application).Source
```

若当前终端限制执行本地ps1，只在新进程中运行以下命令，不修改系统永久策略：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File './experiments/scripts/run_remaining_experiments.ps1' `
    -PythonPath (Get-Command python -CommandType Application).Source
```

本入口包含以下内容；每方法/划分只取已有参考FIT最优配置，不再网格搜索：

| 顺序 | 内容 | 新训练次数（第一批已完成时） |
| --- | --- | ---: |
| 1 | §9.10剩余空间/锚点/数量/特征消融与FastLloyd预算曲线 | 1500 |
| 1 | 四数据集quantity/feature条件加入FFCM、FedSC，完全共用Ours的派生数据视图 | 1600 |
| 2 | COIL20/MNIST β=.1/1.0，七方法固定β=.5最优配置迁移 | 700 |
| 3 | 四数据集FKM/FedSC的1/.75/.5/.25原生参与率对照 | 800 |
| 4 | 四数据集CKKS samples/clients/clusters规模研究 | 1200 |
| 合计 | 完整计划中的剩余任务，不含已完成第一批 | 5800 |

较大的CKKS规模实验放在最后。结果只写标准raw/mean CSV；主表和第一批结果保留。大多数结果仍在 `federated_{seed}_clients{M}_fixed_best`，COIL20/MNIST固定配置跨β结果独立写入 `federated_{seed}_clients20_fixed_transfer`。新Stage2只生成COIL20/MNIST缺失的β=.1/1.0客户端归属，原FIT/TEST和β=.5分区不改。

联邦补充入口：

- `run_federated_best.py --study partition --method ...`：七方法各自在参考β=.5源结果上按FIT选配置，然后迁移至目标冻结β；源/目标记录各自真实归属哈希，共享相同canonical FIT/TEST。这是**固定参考参数的鲁棒性实验**，不能冒充逐β重新网格选出的最优结果，也不能把不同选参协议混进同一张“最优主表”。
- `run_federated_best.py --study participation --method fkm|fedsc`：FKM使用原生整数alpha，FedSC使用原生active_ratio。FedSC只在Stage I字典学习中部分参与，最后仍广播最终字典、收集全员系数；不把它描述为全流程只有该比例客户端上线。FFCM/FastLloyd没有此原生开关，NNFC/k-FED为单轮方法，不为这些方法伪造多轮参与率。
- `run_ours_heterogeneity.py --method ours|ffcm|fedsc --study quantity|feature`：各方法采用自己的参考FIT最佳配置，但同条件使用完全相同的派生数据及归属。保留FedSC原生FIT谱标签及TEST代理。FFCM当前适配器未记录通信的列保留NaN并标明口径，不编造字节；该补充主要支持质量比较。

规模实验当前用于检验Ours本身的代价变化，未覆盖全七方法在全部规模条件下的排名。真实机构数据、独立峰值内存测量、额外CKKS参数扫描及阈值CKKS仍未完成；跑完本入口也不应写成“老师要求的全部研究问题已解决”。正式结果结束后，还需按数据/归属哈希、原生预测口径、实际偏斜和退化状态做跨划分汇总，不能预设Ours在每个条件下领先。

### 9.12 隐私攻击的正负对照：不重跑模型

入口为 `run_ours_privacy_control.py`，直接读取 §9.10 已完成的 `grid_ours_privacy_beta0.5_clientsM.csv`，沿用四数据集冻结 FIT/TEST、客户端归属和初始化种子。没有参数网格、聚类训练或新一轮 CKKS 加密；只校验原攻击分类器和特征是否能识别刻意泄露的参考输出。

| condition | 对照定义 | 解释 |
| --- | --- | --- |
| mia_exposed_fit | 人为将全部 FIT 记录作为公开参照，提取待测记录到参照中最近 K 个点的排序距离，仍使用原来的标准化+逻辑回归、五折攻击流程 | 泄露型正对照；不是 ACES 的实际输出。与原中心攻击属于相同距离特征族、维度相同，但公开参照对象不同 |
| mia_shuffled_labels | 使用相同参照和特征，随机打乱成员标签 | 检查攻击流程的随机参考表现 |
| bytes_plaintext | 将与实际统计向量等长的全零/全一 float64 明文序列化，提取原密文探针所用的长度、16-bin直方图、字节均值和标准差 | 字节探针的刻意暴露正对照；不是 CKKS 密文 |
| bytes_shuffled_labels | 使用相同明文字节特征，随机打乱挑战标签 | 检查字节攻击流程的随机参考表现 |

每种子平衡抽取最多1000条 FIT 与1000条 TEST（Digits受TEST数量限制），与已完成MIA的抽样规则一致；字节对照固定200条挑战。仅使用成员关系/挑战比特训练攻击器，不使用聚类真标签。原明文/CKKS攻击AUC以 reference 列复用；没有把原结果冒充新测量，也不复制原训练耗时。所有AUC、TPR/TNR、常数预测/退化状态均保留；不按预期结果筛选或删除。正对照识别能力不能证明攻击在原中心输出上已足够强，更不能证明 CKKS 的密码安全。

输出仍放在各数据集现有 `federated_{seed}_clients{M}_fixed_best` 下，仅新增：

- `grid_ours_privacy_control_beta0.5_clientsM.csv`：四条件×五种子，共20行。
- 同名 `_mean.csv`：四条件的初始化均值和样本SD，共4行。

四数据集、五划分合计400条攻击控制记录，不新增模型训练。跨划分汇总时，仍先平均每划分的初始化，再计算五划分间波动。运行前可加 `--dry-run` 核对来源与冻结哈希，不产生文件；已有输出不覆盖。

```powershell
$projectPath = (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
Set-Location $projectPath
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

$datasets = @(
    @{ Name = 'digits'; Clients = 20 },
    @{ Name = 'usps'; Clients = 50 },
    @{ Name = 'coil20'; Clients = 20 },
    @{ Name = 'mnist_10000'; Clients = 20 }
)

foreach ($dataset in $datasets) {
    $datasetName = $dataset.Name
    $clientCount = $dataset.Clients
    foreach ($dataSeed in 0..4) {
        $outputDir = Join-Path $projectPath "experiments/results/$datasetName/federated_${dataSeed}_clients${clientCount}_fixed_best"
        $stem = "grid_ours_privacy_control_beta0.5_clients${clientCount}"
        if ((Test-Path -LiteralPath (Join-Path $outputDir "${stem}.csv")) -and
            (Test-Path -LiteralPath (Join-Path $outputDir "${stem}_mean.csv"))) {
            Write-Host "跳过已有结果：$datasetName / privacy_control / seed=$dataSeed"
            continue
        }
        & $pythonPath "$projectPath/experiments/scripts/run_ours_privacy_control.py" `
            --dataset $datasetName `
            --grid-split $dataSeed `
            --beta 0.5 `
            --clients $clientCount `
            --init-seeds '40,41,42,43,44' `
            --data-dir "$projectPath/data" `
            --source-dir $outputDir `
            --out-dir $outputDir
        if ($LASTEXITCODE -ne 0) {
            throw "隐私对照失败：$datasetName / data_seed=$dataSeed"
        }
    }
}
```
### 9.13 MNIST 按截图固定单组参数补跑 Ours

入口为 `run_mnist_fixed_ours.py`。仅运行现有 `mnist_10000` 的 Ours，保留8000 FIT / 2000 TEST及原Stage 2分区；不重跑集中式、其他基线或其他数据集。

| 全模型参数 | 本批固定值 |
| --- | --- |
| anchor_weight | 10 |
| d_emb | 10 |
| lambda_space（截图 fusion_alpha） | 0.75 |
| fuzziness | 1.1 |
| k_neighbors | 30 |
| kernel_sigma | 6.3170086712 |
| kernel_scale_mode | global，使指定的σ实际参与全局核 |
| num_rounds / learning_rate | 10 / 0.001，沿用MNIST已有预算 |
| init_method / active_ratio | random / 1.0 |
| topology_boundary_mode | tanh，五个划分统一使用当前默认模式 |
| 其余设置 | 当前Ours默认首值，包含空间归一化、Procrustes及尺度匹配 |

截图中的融合权重0.75对应模型 `lambda_space`，不是分区Dirichlet β。主条件β=.5；另迁移到冻结β=.1/1.0。旧MNIST结果中部分划分使用clamp，本批统一tanh，以保证五个划分的全模型参数相同。初始化固定random，不再比较uniform。脚本只在当前进程中将全部网格轴收为单值，不修改原 `grid_search_ours.py` 的网格。

默认依次调用原有入口：主结果 → mechanism / space / anchors → participation → privacy → privacy_control → 跨β → quantity / feature → samples / clients / clusters。消融按定义改变对应开关或λ；规模与参与率按原README改变研究因素，这些均不是重新选参。锚点对照仍使用共同维度及原始空间TEST口径；其余TEST预测口径沿用现有脚本。

范围为data_seed=0–4、init_seed=40–44，共875次新增训练、100条攻击正负对照记录；另有50条完整模型记录由消融复用。每份划分14对CSV，五份合计140个CSV文件。仅补Ours对应的已定义实验，不等于覆盖全部可能的论文实验。

PowerShell完整补跑：

```powershell
Set-Location (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
$env:PYTHONIOENCODING = 'utf-8'

& $pythonPath '.\experiments\scripts\run_mnist_fixed_ours.py'
if ($LASTEXITCODE -ne 0) { throw 'MNIST 单组参数补跑失败，请查看上方 Python 报错。' }
```

若先只看主结果，将调用改为：

```powershell
& $pythonPath '.\experiments\scripts\run_mnist_fixed_ours.py' --studies main
```

检查计划、不训练：

```powershell
& $pythonPath '.\experiments\scripts\run_mnist_fixed_ours.py' --dry-run
```

输出为 `experiments/results/mnist_10000/federated_{seed}_clients20_mnist_fixed_v2/`，仅使用原脚本的raw和 `_mean.csv` 命名；旧结果保留。不创建图、报告、诊断或额外配置文件。相同命令重新执行时跳过已有raw/mean文件对；中断的当前任务需重跑，未实现单次训练续跑。不要混用不同初始化种子集合向同一目录补写。可用 `--data-seeds 2,3,4` 限定剩余划分。非main研究依赖同目录已完成的新主结果；privacy_control另依赖同目录的新privacy结果。

已通过 `--help`、全部冻结分区的dry-run、70次调度的原CLI参数解析及OAT/Cartesian均为单配置的检查；这些检查没有执行聚类训练或隐私攻击，也没有创建测试结果文件。正式效果需本地运行后核查，暂不将该配置写成论文最优结果。

### 9.14 Ours / FFCM 四条件公平成本对照

入口 `run_ckks_cost_comparison.py`；批量运行 `run_ckks_cost_comparison.ps1`。
Digits使用20客户端，USPS使用50客户端；β=.5，data_seed=0–4，init_seed=40–44。
固定四条件：`ours_plain`、`ours_ckks`、`ffcm_plain`、`ffcm_ckks`，共200次运行，不做参数网格。

沿用原Stage 1/2冻结分区，各方法按原完整网格CSV的五初始化平均FIT ACC选定自己的配置；
Digits来源 `federated_{seed}_heterogeneity`，USPS来源 `federated_{seed}_clients50_heterogeneity`。
同一方法的明文/CKKS保持参数、初始化、训练预算和TEST预测规则一致；仅随机/均匀初始化，参与率1。
不同方法保留各自原预算，所以同时报告每轮成本与原预算总成本，不能称为总预算匹配。

FFCM加密分支只传输其原生 `(S_k,n_k)`，打包长度 `K(d+1)`；聚合后仍交给原服务器
`S_sum/(n_sum+1e-10)`，保留原塌缩处理。CKKS复用Ours的N=8192、模数(60,40,40,60)、
scale=2^40和4096槽连续打包；公开聚合器没有私钥，独立密钥权威对象解密总和。
这是单机对象级聚合模拟，不是隔离服务部署，也不提供DP。

计量口径：

- `training_core_seconds = trainer_seconds - evaluation_seconds`；包含初始化、CKKS设置和训练，排除离线评价。
  Ours保留原逐轮评价路径，FFCM保留原最终评价路径；主成本比较使用core，不混用含评价时间。
- `setup_seconds`单列；`bytes_per_round`排除一次性公开上下文分发；总字节包括上下行、设置和聚合器/解密器传输。
  CKKS按实际序列化长度计量，明文统计及中心按NumPy字节计量；各自原生诊断载荷如实保留，不补造相同载荷。
- 每次训练在新进程中执行，`peak_rss_bytes`为该进程整个生命周期的峰值，包含导入、数据和评价，
  不等于训练增量内存；`baseline_rss_bytes`另列。进程启动/模块导入不计入core，内存与时间窗口明确不同。
- 固定BLAS/OMP线程环境为1，TenSEAL上下文线程池沿用库默认；保存系统、Python/NumPy版本及内存信息。
  不运行攻击，逐轮资源轨迹保存在同一raw CSV字段内。

完整PowerShell运行：

```powershell
Set-Location (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
& $pythonPath -m pip install psutil==7.2.2
if ($LASTEXITCODE -ne 0) { throw 'psutil安装失败' }
& '.\experiments\scripts\run_ckks_cost_comparison.ps1' -PythonPath $pythonPath
```

仅检查路径与选参：批量命令末尾加 `-DryRun`。仅运行部分划分可加 `-DataSeeds 2,3,4`。
每次成功训练即更新raw CSV，重跑相同命令跳过已完成条件/种子；中断中的一次训练重跑。
改变来源配置或种子集合时，使用独立输出目录。

结果位于 `experiments/results/{digits,usps}/federated_{seed}_clients{20或50}_ckks_cost/`，
每份划分仅 `grid_ckks_cost_beta0.5_clientsM.csv` 和 `_mean.csv`，总计20个CSV。
raw每份20行，mean每份4行；mean的SD只描述同一划分的初始化波动，论文跨划分统计应先对初始化取均值。
原有结果保留，不生成报告、配置快照或诊断文件。

### 9.15 四数据集挂机补充实验：客户端数、归一化、预算及成本

批量入口 `run_paper_supplement.ps1`，固定数据与参数，不再扩大网格。
四数据集为Digits、USPS、COIL20、MNIST-10k。β=.5，data_seed=0–4，init_seed=40–44。
来源与§9.14一致：Digits `federated_{seed}_heterogeneity`，USPS
`federated_{seed}_clients50_heterogeneity`，COIL20/MNIST `federated_{seed}_ours`。
统一使用当前论文的完整五划分来源；不混入尚未完成的MNIST截图参数v2组。

先前Digits/USPS四臂成本已完成200次，100组明密文配对的六项FIT/TEST指标完全一致。
以下时间为训练core（排除离线评价），通信为各自原预算的总MiB，SD为五划分均值的SD：

| 数据集/方法 | TEST ACC(%) | 明文/CKKS core秒 | 明文/CKKS总MiB |
| --- | --- | --- | --- |
| Digits Ours | 78.322±0.701 | 4.101 / 7.580 | 3.738 / 171.501 |
| Digits FFCM | 75.922±2.252 | 1.096 / 3.349 | 2.756 / 104.580 |
| USPS Ours | 68.086±0.853 | 12.913 / 20.935 | 9.484 / 416.466 |
| USPS FFCM | 64.439±1.155 | 11.196 / 20.593 | 13.779 / 485.100 |

CKKS每轮MiB约为Digits Ours/FFCM 6.754/6.798，USPS16.401/16.508；
总成本受各自选中轮数影响，不据此声称Ours同轮通信更省或全面更快。

批量按以下顺序执行，默认所有研究因素都固定在每个源划分自己的FIT最优配置上：

| 顺序 | 研究 | 设置 | 新训练数 |
| --- | --- | --- | --- |
| 1 | 四条件公平成本 | 四集Ours/FFCM×明文/CKKS；Digits/USPS完整结果跳过训练 | 200 |
| 2 | 归一化 | 四集full/no_normalization；full复用 | 100 |
| 3 | 预算曲线 | MNIST10/20/40；USPS按源T运行0.5T/T/2T/4T；原T复用 | 125 |
| 4 | 客户端数 | 四集完整FIT的M=10/20/30/50，七方法，每方法固定源配置迁移；源M复用 | 2100 |
| 合计 | | 另复用850条源记录，已完成200条成本记录不重训 | 2525 |

七方法为Ours、FFCM、FedSC、k-FED、FKM、NNFC、FastLloyd。
客户端数研究只扩展原Stage2 Dirichlet(.5)归属，FIT总数、TEST及预处理不变，
不是§9.10固定本地样本量、总FIT随M增长的等大小视图。所有方法共享目标冻结分区。
源客户端数为Digits/COIL20/MNIST20、USPS50；不为新M重新调参。
固定总样本时改变M也会改变实际数量/标签偏斜，CSV记录本地样本量min/max/mean、
size ratio及平均标签TV；不能把此结果描述为严格控制异质性的纯M效应。
FKM alpha=None保持全参与；显式源全参与人数适配到目标M，不改其他优化参数。
原生初始化继续遵循各方法原协议；Ours/FKM/FFCM源配置仅random/uniform，不用池化样本初始化。

预算核查只改变轮数，所有预算完整报告，不根据TEST挑选结果。
新运行在同一CSV中保存相对聚合位移、拓扑位移、FIT/TEST及时间轨迹；
旧源行没有记录的轨迹留空。归一化及预算复用行的历史耗时不作为同机成对成本测量。
收敛阈值针对精炼前的相对聚合位移，不能拿后处理后的绝对中心位移直接套阈值。

完整运行（Windows PowerShell兼容，批量脚本为ASCII）：

```powershell
Set-Location (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
& '.\experiments\scripts\run_paper_supplement.ps1' -PythonPath $pythonPath
```

可选：末尾加 `-DryRun` 只检查源文件存在并打印任务计划，不生成分区或训练结果；
它不替代各Python入口的实际分区与哈希校验。
加 `-Stages clients` 只运行客户端比较；加 `-Datasets coil20,mnist_10000` 限定数据集；
加 `-DataSeeds 2,3,4` 限定划分。重复相同命令自动续跑，保留已成功的逐次记录。
每个独立任务失败后继续后面的独立任务，终端末尾列出失败标签并返回非零状态；
Stage2准备失败则停止训练，避免后续全部缺数据。失败原因保留在终端，不另建日志文件。

结果目录：

- 成本：`experiments/results/{dataset}/federated_{seed}_clients{M}_ckks_cost/`，沿用§9.14。
- 归一化/预算：`.../federated_{seed}_clients{M}_fixed_checks/`；
  `grid_ours_normalization_beta0.5_clientsM`、`grid_ours_budget_check_beta0.5_clientsM`。
- 客户端比较：`.../federated_{seed}_clients{10或20或30或50}_fixed_clients/`；
  每方法 `grid_{method}_fixed_beta0.5_clientsM`。

每个stem只写raw和 `_mean.csv`。客户端比较共2800条记录（700复用、2100新训）；
每份数据划分/客户端数有七对CSV。均值文件中的SD只反映该划分的初始化波动；
论文汇总先平均初始化，再汇总五份划分。不得把复用源行作为新增独立重复。

按历史参考配置的时间估计全批预留约6–12小时，USPS10客户端局部谱分解可能明显拖长。
该时间是粗估；程序按实际任务完成结束，不为凑时长延长运行。
未纳入挂机的待办：尚未准备的非图像自然客户端外部验证、真实网络与隔离服务部署。
已有Swiss Roll失败边界、归一化旧OAT配对、成本模型核查及论文图表重排可由现有结果分析，
不因此额外重跑集中式或隐私攻击扫描；不将本批完成解释为保证录用。

### 9.16 本批完成后的结果与论文更新（2026-10-30）

§9.15已完成。核查610对raw/mean、3575条记录及80份冻结分区，未发现缺失种子、
失败状态、非有限质量指标、汇总或数据身份不一致；无须重复运行挂机脚本。
完整记录含400条成本、200条归一化、175条预算、2800条客户端比较。
其中2525次新训练、850条复用记录、200条先前完成的成本记录，复用不是新增独立重复。

| 研究 | 实测结论 | 接下来是否补跑 |
| --- | --- | --- |
| Ours/FFCM × 明文/CKKS | 200组配对的六项FIT/TEST质量指标一致；Ours时间倍率1.85/1.62/1.87/1.12，总通信倍率45.88/43.91/20.77/47.12 | 不需要，成本表和图已更新 |
| 最终配置归一化 | Digits/USPS/COIL20/MNIST配对TEST ACC均值增加2.911/3.591/2.861/2.860个百分点；并非所有运行都改善 | 不需要，已有no_normalization条件，Full复用 |
| 预算 | MNIST10/20/40轮为52.182/52.694/52.116%；USPS0.5T/T/2T/4T为63.551/68.086/69.417/69.824% | 不继续扩大预算；所有运行仍触顶，不宣称优化收敛 |
| 客户端数 | 四集×10/20/30/50客户端×七方法齐全；Ours平均ACC最高9/16条件 | 不需要；保留MNIST50等落后场景 |

上述倍率均按Digits、USPS、COIL20、MNIST-10k顺序；训练core包含setup并排除离线评价。
进程峰值RSS包含导入、加载及评价，不是训练增量；不同方法保留各自预算，
跨方法总成本不是同轮比较。每轮载荷都不随样本数增长，这不是Ours独有的性质。
主表继续保留原定预算，不根据TEST选择更好的新轮数。

已更新当前项目paper/icde2027/main.tex，新增质量—成本表及4张图，重画配对消融，
正文12页；上一级icde2027_draft.pdf用于阅读，icde2027_overleaf.zip用于上传。
完整汇总、数字口径和具体替换文件见paper/icde2027/README.md。

仅重画图可在项目根目录依次运行以下两行，不启动训练、不生成诊断CSV或JSON：

```powershell
& (Get-Command python -CommandType Application).Source '.\experiments\figures\plot_paper_results.py'
& (Get-Command python -CommandType Application).Source '.\experiments\figures\plot_paper_supplement.py'
```

非图像自然客户端外部验证、独立确认及真实网络/密钥隔离部署仍是适用边界；
当前不因这些待研究方向额外启动训练，也不把现有结果解释为录用保证。

### 9.17 HE定位修订后的补充实验（2026-10-31）

本节为SecFDCC_ICDE.md建议补齐运行入口，不代表正式结果已完成。
沿用冻结数据、五划分0–4、初始化40–44；原四集不重新划分或扩大选参网格。
批量入口为`run_he_revision_experiments.ps1`，ASCII编码兼容Windows PowerShell 5.1。

| 阶段 | 补充目的和设置 | 默认工作量 |
| --- | --- | --- |
| roles | 三个实际spawn角色C/A/D，通过认证回环socket验证首次聚合统计；固定旧四集已选配置 | 100条快照 |
| solver | 同配置、同预算上限和停止规则的dense/partial CKKS对照，四集均执行；不重新搜索 | 200次训练 |
| covtype | 原参考流程：六方法每个最多4配置，平均FIT选参 | 575次训练 |
| large-silo | Covtype的3个等大小silo，每silo128/512/1024/2048条真实、互不重复FIT；3轮上限 | 300次训练 |
| central | 只补Covtype的K-means/FCM/Spectral，沿用集中式原入口 | 75条最终记录，另有原预算/参数扫描 |

实际运行：

```powershell
Set-Location (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
& '.\experiments\scripts\run_he_revision_experiments.ps1' -PythonPath $pythonPath
```

重复相同命令自动复用已成功结果并重试失败项；输出仍为raw及`_mean.csv`，
不写额外日志、诊断CSV或JSON。Stage1/2仍按数据README写必要分区与metadata。
末尾列出失败任务并返回错误；原四集集中式和旧参数网格不会重跑。
加`-DryRun`仅打印计划，不下载、不建分区、不训练；它不是完整的数据和模型校验。

可分阶段运行：

```powershell
& '.\experiments\scripts\run_he_revision_experiments.ps1' -PythonPath $pythonPath -Stages roles,solver
& '.\experiments\scripts\run_he_revision_experiments.ps1' -PythonPath $pythonPath -Stages covtype,large-silo,central
```

已有Covtype质量结果后才能单独运行`-Stages large-silo`；它使用已选Ours/FFCM参数。
只执行1个数据划分可加`-DataSeeds 0`，不改变最终论文需要五划分的汇总协议。

具体边界：

- `run_ckks_role_validation.py`：D独占密钥、C/A只持public context，A收到聚合明文更新中心。
  C一个进程复用M个逻辑客户端；这是隔离的聚合传输快照，不是完整训练/每客户端隔离部署。
  记录PID、角色密钥标志、应用层字节、聚合与中心误差、请求/重放检查和数值状态。
  socket握手、TCP封装及离线参考验证不计入应用层正常消息；近似解密输出未实现noise flooding，
  不能把进程分离称为完整协议安全证明。`--smoke-test`为不落盘的合成检查。
- `run_ours_solver_comparison.py --study solver`：原生dense仍为默认；partial保留最低p个特征向量，
  包括最低向量，和旧图、拉普拉斯、行归一化口径相同。图和距离矩阵仍为dense，
  不能写全稀疏、近线性或必然比FFCM更快；每臂在fresh worker运行，保留质量变化和失败。
- Covtype从官方fetch_covtype缓存中用固定seed42分层抽1万条，再按数据种子重分8000/2000，
  共7类、原生54维；不补零成64维。仅FIT拟合StandardScaler。首次会下载官方数据缓存。
  `run_covtype_reference.py`默认候选数为Ours4、FFCM4、FedSC4、FKM4、NNFC4、k-FED3。
  Ours原4组保留；可按§9.18显式追加8组网格邻域，优先使用§9.19的TPE自适应版本。
- `--study large-silo`：Ours dense、Ours partial与FFCM均用CKKS和3轮上限；固定源选参后比较资源。
  从同一8000条FIT的每客户端储备取嵌套前缀，最大实际FIT为6144，TEST不变、不复制样本。
  该成本视图不是原20客户端Dirichlet归属，也不是新的主质量表；实际归属及FIT哈希另列。
- 统计仍先平均初始化，再汇总五划分；不自动把共享样本的划分当作独立重采样或生成p值。

结果目录：

- 原四集：`experiments/results/{dataset}/federated_{seed}_clients{M}_he_revision/`。
- 新质量：`experiments/results/covtype_10000/federated_{seed}_reference/`。
- 大silo：`experiments/results/covtype_10000/federated_{seed}_large_silo/`。
- 新集中式：`experiments/results/covtype_10000/seed_{seed}_full/`。

当前状态：help/dry-run、小合成dense/partial和真实CKKS子进程、三角色真实Digits首次统计快照，
以及Covtype数据/续跑mock检查均通过，临时输出已删除；本节正式全批尚未执行。



### 9.18 Covtype手工邻域网格（备选，默认不开启）

原4组中，candidate_0在五个划分的平均FIT ACC均最高：m=1.1、lambda=.5、p=4、
self_tuning、k=5、anchor_weight=1、random、20轮、学习率.001。因此围绕该配置追加8组，
其余条件不变，不做Cartesian展开，也不根据TEST筛选：

| 追加候选 | 唯一改变 |
| --- | --- |
| candidate_4 | fuzziness=1.05 |
| candidate_5 | fuzziness=1.2 |
| candidate_6 | lambda_space=.2 |
| candidate_7 | lambda_space=.35 |
| candidate_8 | d_emb=2 |
| candidate_9 | d_emb=6 |
| candidate_10 | k_neighbors=10 |
| candidate_11 | anchor_weight=.5 |

candidate_0–3参数、编号均保持原样，random/uniform原候选都保留；不重跑已成功的100条Ours记录。
仅加`--expand-ours`才启用这8组；最多新增200次。默认参考入口仍为原4组，其他基线和原四集不追加搜索。
结果直接追加到各federated_{seed}_reference里的原grid_ours CSV及均值文件，不复制结果或另建诊断文件。

在暂停原批次后，另开PowerShell终端执行：

```powershell
Set-Location (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
foreach ($dataSeed in 0..4) {
    & $pythonPath '.\experiments\scripts\run_covtype_reference.py' `
        --grid-split $dataSeed --clients 20 --beta 0.5 `
        --init-seeds '40,41,42,43,44' --methods ours --expand-ours `
        --eigen-solver dense --data-dir '.\data' `
        --out-dir ".\experiments\results\covtype_10000\federated_${dataSeed}_reference" `
        --resume
    if ($LASTEXITCODE -ne 0) { throw "Covtype Ours failed: data_seed=$dataSeed" }
}
```

本次扩大了Ours的搜索预算；汇报比较时应如实区分它与基线原有的小候选预算。
默认全批次仍使用原4组；手工扩展结果不应用原4组入口混写。恢复旧的内存中批次前，先确认新Ours搜索已结束，
避免同时写同一份结果。暂停不会把当前未完成的单次运行记为完成。


### 9.19 Covtype Ours自适应TPE（推荐）

使用Optuna TPE贝叶斯优化：从已有4组完整结果建立历史，每次根据FIT结果推荐下一组，
不展开Cartesian网格，也不重训已完成的相同配置/种子。主入口`tune_covtype_ours.py`，
批量入口`run_covtype_ours_tpe.ps1`；使用Optuna4.5.0，已加入requirements.txt。

- 目标：五初始化平均FIT ACC；TEST只记录，不参与建议和停止。
- 固定：原数据/客户端归属、dense、20轮、学习率.001、其余原最佳配置字段。
- 搜索：m=1.025–1.2（步长.025）、lambda=.2–.9（步长.05）、p∈{2,4,6}、
  k∈{5,10}、anchor_weight=.5–1.5（步长.25）、init∈{random,uniform}。
- 每组仍完整跑5个初始化；默认不按前几个种子的低分剪枝，以保留原评价协议。
- 每划分累计最多8个新增配置；连续3组未达到.002（0.2个百分点）的有效FIT提升即停。
  若设置`--target-fit-acc`，达到该FIT目标也停止；这不是证明全局最优或模型已收敛。
- 读取旧4组作为warm start，旧源CSV不变；新目录只写通常的逐次CSV和`_mean.csv`。
  不创建SQLite、JSON、额外日志或诊断文件；中断后从CSV恢复，不重训已完成记录。
- 默认全五划分上限200次新训练；实际可能因停滞或目标达成而更早结束。

原批次暂停后，在VSCode新建PowerShell终端运行：

```powershell
Set-Location (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
& '.\experiments\scripts\run_covtype_ours_tpe.ps1' -PythonPath $pythonPath
```

先检查不训练可在末尾加`-DryRun`；可用`-DataSeeds 0`只调一份划分。
结果位于`experiments/results/covtype_10000/federated_{seed}_ours_tpe/`，
其中复用行标明来源，不增加独立重复次数。TPE改变的是外层选参方式，聚类算法内部更新不变。
新的选参结果完成后再决定后续成本实验使用的配置，不在此时自动恢复原批次。

同时修复§9.17大silo入口：FFCM的合法`eigen_solver=not_applicable`不再被误判为非dense；
仅对Ours源配置检查求解器。真实Covtype源的`--study large-silo --dry-run`已通过。


§9.19验证状态：真实五份Covtype源CSV全部读取通过，复用100条历史结果时模型调用为0；
真实TPE提议接五个初始化的合成小数据20轮训练、raw/mean写入及再次续跑零重训通过。
早停、目标停止、失败恢复、重复参数缓存、TEST不影响提议的mock检查通过；
Python/PowerShell语法及ASCII入口检查通过。所有临时测试输出已删除，没有启动正式Covtype自适应调参。

### 9.20 HAR补充验证：Ours自适应TPE、五基线和集中式参考

批量入口为`run_har_experiments.ps1`，沿用现有Stage 1/2、初始化种子40–44和
数据划分种子0–4。默认50客户端、Dirichlet beta=0.5；固定OpenML data_id=1478，
10299条、561个原始特征、6类。每份划分为FIT 8239、TEST 2060，只有FIT拟合
StandardScaler和PCA到64维，所有方法共享冻结数组。当前为记录级模拟联邦，
原始训练/测试合并后重分，不是受试者隔离或自然客户端实验。

**Ours不做网格搜索。** 复用`tune_covtype_ours.py --dataset har`的TPE入口，
HAR自行冷启动，不把Covtype的模型、分数或历史当作HAR结果：

- 起点为m=1.05、lambda=.5、p=4、k=5、anchor_weight=1，先分别评价random和uniform。
  两个启动配置计入默认每划分最多8组的总预算，随后最多6组由TPE逐次推荐。
- 搜索空间为m=1.025–1.2（步长.025）、lambda=.2–.9（步长.05）、p={2,4,6}、
  k={5,10}、anchor_weight=.5–1.5（步长.25）、init={random,uniform}。
- 固定dense、20轮上限、学习率.001和现有停止规则。每组完整评价五个初始化，
  以平均FIT ACC建议参数及选优，TEST只记录，不参与建议、选优和搜索停止。
- 两个启动完成后，连续3组没有达到.002（0.2个百分点）的有效FIT提升即停止；
  也可能先达到总预算。该停止仅表示搜索停止，不代表全局最优或训练已收敛。
- 已完成的参数/种子从CSV复用；部分配置只补缺失或失败种子。输出只有raw和`_mean.csv`，
  不创建Optuna数据库、额外日志或诊断文件。

其余五基线复用`run_covtype_reference.py --dataset har`，该入口在HAR下不运行Ours：

| 方法 | 固定候选 |
| --- | --- |
| FFCM | m={1.1,1.3} × init={random,uniform}，20轮上限，共4组 |
| FedSC | S={10,20} × sigma={1,5}，字典大小3K=18，共4组 |
| k-FED | 本地簇数{2,3,6}，共3组，原生单轮 |
| FKM | method={ma,gs} × init={random,uniform}，50轮上限，共4组 |
| NNFC | k_neighbors={2,5,10,20}，共4组，原生单轮 |

集中式K-means、FCM、Spectral继续使用`centralized/run_clustering_ceiling.py`，
逐初始化预算扫描、FIT选参和最终TEST评价不变；触顶配置按原入口核查至1000。
它们是集中式参考，Ours和各基线搜索预算不同，比较时如实记录。

完整运行：

```powershell
Set-Location (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
& '.\experiments\scripts\run_har_experiments.ps1' `
    -PythonPath $pythonPath `
    -Clients 50 `
    -MaxNewTrials 8 -Patience 3 -MinImprovement 0.002
```

默认顺序为准备数据、联邦方法、集中式。只执行Ours可追加
`-Stages prepare,federated -Methods ours`；已有数据时可去掉prepare。
可用`-DataSeeds 0`只执行一份划分；`-DryRun`只打印绝对路径和任务，不下载或训练。
首次数据信息通过现有OpenML加载器下载并缓存，不需要手动移动数据。

输出位置：

- Ours：`experiments/results/har/federated_{seed}_clients50_ours_tpe/`。
- 五基线：`experiments/results/har/federated_{seed}_clients50_reference/`。
- 集中式：`experiments/results/har/seed_{seed}_full/`，保留原入口3个CSV及必要运行metadata。
- 冻结数据：`data/har/splits/`与`data/har/partitions/`。

`-Clients`统一控制Stage 2及所有联邦方法，结果目录带客户端数；已有20客户端结果保留。
改变客户端数只重分同一份FIT，集中式不依赖客户端数，已完成的集中式划分可以复用。

重复相同命令会复用联邦逐次结果，并跳过协议相同、三方法×五初始化齐全的集中式划分。
集中式原入口不支持单份划分内部续跑，未保存的那份需从头执行。
默认联邦最多200次Ours训练+475次基线训练；集中式产生75条最终记录，
另有预算/参数扫描，默认共975次模型调用（不含触顶追加核查），其中325次谱分解，
所以不能用75条最终记录估计总耗时。

验证状态：HAR完整形状的加载/预处理、FIT-only拟合、Stage 1/2同源和客户端索引检查通过；
五基线入口95条mock、CSV选优和零重训练续跑通过；真实小合成数据上的Ours两初始化、
五基线及集中式三方法共10次模型检查通过；TPE完整入口的两启动及一个自适应提议共15次
真实小合成训练、CSV选优和零重训练续跑通过。TPE增量预算、早停、失败/缺失续跑、
TEST不影响建议及已有Covtype225条记录兼容检查通过；PowerShell解析、ASCII、任意路径及
失败停止检查通过。临时测试产物已清理。

2026-11-03主实验结果核查：HAR 50客户端的六联邦方法共655条逐次记录
（Ours180、FFCM/FedSC/FKM/NNFC各100、k-FED75）和75条集中式最终记录已完成。
每配置五个初始化齐全；无重复、失败、非有限质量指标，raw/mean均值和标准差重算一致，
Stage 1/2和各方法数据/归属哈希一致。集中式75条最终记录均未触及迭代上限。

下表先平均同划分的初始化，再汇总五份划分；数值均乘100，ACC的±为划分均值的标准差。

| 方法 | TEST ACC | TEST NMI | TEST ARI |
| --- | --- | --- | --- |
| Ours（TPE） | 62.84 ± 2.25 | 58.87 | 48.65 |
| Central K-means | 55.19 ± 0.84 | 56.18 | 42.15 |
| FFCM | 53.37 ± 0.79 | 52.34 | 36.89 |
| FKM | 51.10 ± 1.05 | 50.15 | 34.75 |
| FedSC | 50.99 ± 2.29 | 48.82 | 33.46 |
| Central FCM | 50.88 ± 2.23 | 48.21 | 32.48 |
| Central Spectral | 47.34 ± 1.29 | 43.63 | 27.09 |
| k-FED | 43.38 ± 4.40 | 49.92 | 33.19 |
| NNFC | 36.65 ± 0.59 | 48.65 | 30.84 |

Ours候选数为8/8/8/7/5，不能称为与基线相等的搜索预算。选中的25次Ours均为20轮
`round_limit`，TPE停止不代表训练收敛；其中split1/init42、split3/init42/43仅产生5个
预测簇（目标K=6），这些结果按原样计入，没有删除或补造标签。该现象作为算法行为记录，
不是数据或文件损坏。

### 9.21 HAR完整补充批次：固定已选参数，一条指令执行

入口为`run_har_followup_experiments.ps1`。默认使用上述已核查的HAR五划分、
50客户端、beta=.5、初始化40–44；Ours读取`clients50_ours_tpe`，其余五个联邦
方法读取`clients50_reference`，所有选参仍只使用平均FIT指标。主表、TPE和集中式不重跑。

| 阶段 | 实验目的及固定条件 |
| --- | --- |
| ablation | 机制：Full/去拓扑/去Procrustes/原空间成员度/去归一化；空间：Full/纯谱成员度；锚点：matched Full/去锚点 |
| participation | Ours、FKM、FedSC分别比较active ratio={1,.75,.5,.25}；不为单轮方法编造通信轮参与机制 |
| cost | Ours/FFCM × 明文/CKKS，fresh worker重新测量质量、时间、上下行及总字节、峰值RSS |
| privacy | Ours明文/CKKS的现有MIA、字节区分及角色诊断，再运行暴露输出/打乱标签四个攻击对照 |
| heterogeneity | Ours/FFCM/FedSC分别执行数量偏斜和特征漂移四档条件，记录实际偏斜与派生数据哈希 |
| label-skew | 原六联邦方法固定beta=.5所选配置，迁移到beta={.1,1.0}的50客户端分区 |
| clients | 原六联邦方法固定50客户端所选配置，比较M={10,20,30,50}，总FIT和TEST不变 |
| budget | Ours固定其他参数，比较.5T/T/2T/4T，即10/20/40/80轮上限；保留轨迹，不按TEST挑轮数 |
| roles | 三个实际spawn角色的首次CKKS聚合传输快照；不是完整训练或每客户端独立进程部署 |
| solver | 同参数同停止规则的dense/partial CKKS比较；图构造仍为dense |
| large-silo | Ours dense/Ours partial/FFCM，3个silo，各128/512/1024/2048条真实不重复FIT，CKKS、3轮上限 |
| scale | CKKS成本视图分别改变样本量、客户端数和模型簇数，各4档；不替代主表划分 |

去锚点比较保留原入口规则：两臂统一可用维度，均用原空间TEST预测，均重新训练，
不能将无锚点代理与主表Nyström的Full直接混比；该条件也会移除基于锚点的对齐。
机制和空间消融的Full复用原质量结果，不重新训练。
Ours参与率active=1只复用质量；旧TPE未记录的通信/资源/轨迹保持NaN并明确标记
`reused_quality_only`，不当作测量成本。四条件成本则全部在fresh worker重新计量。

主客户端对照10/20/30/50保持总FIT固定；scale中的客户端10/20/50/100保持本地样本量
固定、总FIT随M增长，两者的目的和口径不同。HAR的模型簇数成本视图为3/6/12/24，
真实活动标签仍为6类；这些是资源尺度对照，不是按TEST选择新的真实簇数。
标签偏斜和客户端对照只添加Stage 2分区，不重新拟合预处理或改变TEST。

完整运行：

```powershell
Set-Location (Get-Location).Path
& '.\experiments\scripts\run_har_followup_experiments.ps1' `
    -PythonPath (Get-Command python -CommandType Application).Source `
    -Clients 50
```

默认共294个Python任务（290个研究入口调用和4次Stage 2准备），约2675次新增模型训练，
另复用250条既有质量记录、运行25条角色快照和100条攻击对照；调用数量不等于训练次数。
这是HAR既定补充实验的长批次，不重新运行旧四数据集，也不新增另一套DP选参流程。

可加`-DryRun`打印全部计划，不创建数据或训练；`-DataSeeds 0`只执行一份划分。
可用`-Stages ablation,participation,cost`选择部分阶段；privacy阶段会先生成隐私记录，
再执行依赖它的攻击对照。默认已安排完整依赖顺序，较耗时的规模实验在末尾。
每次成功运行逐条写入raw CSV，中断后重新执行同一指令，按条件和初始化补缺并重建mean；
完成项不重训。失败直接显示Python错误并停止，已保存记录保留，不把缺失条件标为完成。

结果仍保存在`experiments/results/har/`，各划分的目录后缀分别为：
`clients50_fixed_best`、`clients50_ckks_cost`、`clients50_heterogeneity`、
`clients{M}_fixed_transfer`、`clients50_he_revision`、`clients50_large_silo`及`clients50_scale`。
只写各研究通常的raw/mean CSV，以及Stage 2必要分区文件；不创建数据库、诊断报告或独立轨迹文件。
逐轮轨迹留在相应CSV内，复用记录没有的轨迹不补造。

加密质量/成本、隐私诊断和攻击对照不是DP保证或完整安全证明；保留攻击退化/NaN状态。
角色快照、明文聚合结果的可见范围及近似解密输出限制沿用§9.17的说明。

本批交付检查：PowerShell 5.1解析、ASCII及294条计划的真实CLI参数解析通过；
五份实际HAR源的消融/隐私/参与率/预算/角色、solver/large-silo dry-run通过。
临时复制seed0的真实冻结数据后，六种方法在beta=.1/1及客户端10/20/30/50的
36个迁移入口通过预检，随后删除临时分区；正式HAR数据不变。
真实小合成数据的机制/空间/锚点消融、参与率、fresh-worker四条件成本、隐私两臂及
四攻击对照、预算、三角色快照、六种异质性和三种CKKS规模研究均通过功能检查。
中断/缺行补跑和完整零调用续跑通过；privacy→control串行重跑的raw字节hash保持不变，
含高精度浮点及NaN诊断。所有临时源和测试CSV已清理，没有启动正式HAR补充训练。

### 9.22 最后两类验证：自然受试者与完整多进程训练

2026-11-04：§9.21已经完成，290对raw/mean、3050条记录经核查无缺种子、
重复、失败、非有限质量指标或冻结数据身份不一致。以下入口只补当前剩余验证，
不再执行旧的网格、TPE、消融、参与率或规模批次；本轮按用户要求暂停论文修改。

| 阶段 | 运行内容 | 默认新增模型运行 |
| --- | --- | --- |
| subject | HAR五折受试者留出；每折24名训练受试者各为一个自然客户端，6名只进入TEST；Ours dense、partial、三进程CKKS partial，五个联邦基线和三个集中式参考 | 11条件×5折×5初始化=275 |
| transport | 旧四集及HAR，分别运行native/process × dense/partial，均使用CKKS；C/A/D对每一轮实际通过socket传输与更新，沿用原选定轮数及停止规则 | 5数据集×5划分×5初始化×4条件=500 |

共31个Python任务，通常775次固定配置训练；集中式如触顶会按预先规则补到1000迭代，
可有额外重检。四条件是执行方式和求解器的对照，不用于搜索或按TEST挑选最终参数。
普通联邦与集中式参考在自然受试者研究中使用同一份新冻结FIT/TEST。

完整运行：

```powershell
Set-Location (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
& '.\experiments\scripts\run_final_validation_experiments.ps1' `
    -PythonPath $pythonPath
```

原规范客户端数保持Digits/COIL20/MNIST为20，USPS/HAR为50。新自然受试者研究是
另一份名为`har_subject`的数据，真实拥有30名受试者，故不能继续伪装为50个自然客户端。
固定分组种子20261104，五折TEST受试者及记录均互不重叠，覆盖全部10299条记录一次；
每折FIT=8052–8365、TEST=1934–2247。Scaler及PCA64只在该折FIT受试者拟合，
受试者ID只决定归属，不进入特征或模型训练。原HAR分区和预处理不覆盖。

官方ZIP自动缓存于`data/har_subject/source/uci_har.zip`；当前工作区已完成官方数据下载
及完整读取检查，正式冻结文件由本地批次生成。Stage 1沿用`data/har_subject/splits/`；
Stage 2沿用`data/har_subject/partitions/har_subject_seed_{fold}_beta_subject_clients_24/`。
`subject`是自然归属的字符串标识，metadata的`dirichlet_beta=null`，不是某个数值beta。
不要对这份数据调用Dirichlet重分入口；使用`data/prepare_har_subject_splits.py`。

自然研究对所有折固定迁移已有HAR记录级split0的FIT最优配置，不再次搜索；集中式也固定
原split0各初始化的配置与预算，触顶才执行1000迭代重检。新五折FIT仍有重叠，配置来源
也使用过同一HAR队列；这是受试者迁移验证，不能写成五个独立重复或全新盲测确认。

完整进程训练中，C保有逻辑客户端数据与原生谱坐标，A执行加密求和、明文原型更新与拓扑，
D独占secret key且每轮只解规定的聚合批。复用原生Client/Server及FIT/TEST评价规则，
不改聚类公式、CKKS参数或停止条件。一个C进程代表多个逻辑客户端，三角色同主机、
同一OS用户；不声称每客户端独立进程、多机网络、阈值CKKS或完整转录安全证明。
两种执行方式均保留TenSEAL默认上下文线程池，BLAS/OMP为1；每次使用新工作进程。

通信列区分native的逻辑M客户端计量与process的实际socket应用载荷。后者只实际分发两份
公开context，不能将它和M份逻辑分发直接相除声称节省通信；policy probe字节单独记录。
wall含启动、加载及评价；core窗口在行内说明；进程内存总和是各角色生命周期峰值之和，
不是同一时刻的总峰值，也不是训练增量内存。

输出仍只含标准raw与`_mean.csv`：

- 自然受试者：`experiments/results/har_subject/fold_{fold}_clients24_fixed/`。
- 完整多进程：原五数据集的`federated_{seed}_clients{M}_full_roles/`。

每次保存成功或失败记录，完整重复运行会跳过成功项、只补失败或缺失项。算法数值失败
保留真实错误和缺失指标，不用dense冒充partial，也不删除退化簇。独立任务继续执行，
结束时有缺项会明确返回失败，mean中的`n_seeds/n_failed/n_expected/complete`如实标记。
数据准备失败则停止依赖它的训练。

可加`-DryRun`只打印计划，或`-Stages subject` / `-Stages transport`选择阶段。
本批不实现密文内除法/归一化或阈值CKKS；它们属于新的算法/安全机制扩展，不能凭实验
补跑写成已实现。统计论证和正式安全证明同样不能由增加运行次数代替。

开发阶段验证记录（正式本地运行前）：官方ZIP读取与五折真实形状/归属预处理通过；所有原五集25份源配置及五份
正式形状受试者分区均通过入口预检，当时没有训练正式模型。真实小合成数据上，11个自然
验证条件、CKKS native/process × dense/partial四臂、三轮运行及两轮提前停止均通过。
检查了密钥归属、逐轮解密次数、socket计量、成功项零调用续跑、缺一项只补一项，
以及失败保留/其他条件继续/只补失败项。Windows PowerShell5.1的31任务展开、ASCII、
任意工作目录、依赖失败停止及独立任务继续检查通过。临时测试输出不作为实验结果。

正式运行核查（2026-11-04）：本批775条全部完成，30对raw/mean结果的种子、条件、
完成状态、六项质量均值/SD及数据/归属哈希一致；没有失败、重复、缺失或NaN。
自然HAR中dense/partial/process-partial的TEST ACC为62.47/63.87/63.88%，FFCM为54.80%。
五集250次加自然HAR25次完整三角色均通过密钥、轮次和已测请求策略检查；部分求解器
个别配对存在较大轨迹差异，不能将运行完成理解为数值完全一致。详见
[论文与实验总结](../../paper/icde2027/README.md)。原结果保留，无需重跑已完成项。

### 9.23 ACES：固定参数、新初始化配对检验

本批只补统计评估，论文和图表等本地实验完成后更新。不重新网格/TPE搜索，
不重分数据，不重复隐私探针、集中式或历史消融。ACES内部方法名仍为`ours`。

| 数据集 | 客户端 | data_seed | 新init_seed | 方法 | 新增训练 |
| --- | --- | --- | --- | --- | --- |
| Digits | 20 | 0–4 | 100–109 | Ours/ACES、FKM、FFCM、NNFC、k-FED、FedSC | 300 |
| USPS | 50 | 0–4 | 100–109 | 同上 | 300 |
| COIL20 | 20 | 0–4 | 100–109 | 同上 | 300 |
| MNIST-10k | 20 | 0–4 | 100–109 | 同上 | 300 |
| HAR | 50 | 0–4 | 100–109 | 同上 | 300 |

共1500次。beta固定0.5，各方法按原40–44种子的平均FIT ACC选择一份完整配置，
仅更换初始化种子。保持原本轮数、本地迭代、原生FIT标签及TEST预测方式。
ACES保持密集谱求解、明文聚合、关闭隐私攻击计时；FKM/FFCM继续使用已选的
random/uniform，本地初始化不同的基线保留各自实现。

源配置目录与当前主表一致：Digits用`federated_{s}_heterogeneity`；USPS用
`federated_{s}_clients50_heterogeneity`；COIL20/MNIST用`federated_{s}_ours`；
HAR的Ours用`federated_{s}_clients50_ours_tpe`，其他方法用
`federated_{s}_clients50_reference`。不混用MNIST的`mnist_fixed_v2`探索配置。
入口核查冻结FIT/TEST、完整客户端归属、源配置及已有哈希；新记录保存原始/均值
源文件摘要、配置、评价设置及数据身份，不给缺少历史哈希的旧CSV补造元数据。

**完整运行：**

```powershell
Set-Location (Get-Location).Path
$pythonPath = (Get-Command python -CommandType Application).Source
& '.\experiments\scripts\run_paired_wilcoxon.ps1' -PythonPath $pythonPath
```

脚本为ASCII，兼容Windows PowerShell5.1；项目和数据路径由脚本位置确定。
依赖使用现有`requirements.txt`，包括SciPy1.15.3，无需另装统计库。
每个方法/划分使用独立Python进程；每次初始化完成即保存。中断后执行同一条命令，
仅补缺失或失败的初始化。失败与NaN保留在正式CSV，不当成成功；重试追加
`attempt`，不删除前次失败。独立任务继续执行，批末有缺项则返回失败，补齐后才汇总。

输出位置：

- 逐次与均值：`experiments/results/{dataset}/federated_{s}_clients{M}_paired_validation/grid_{method}_paired_validation_beta0.5_clients{M}.csv`及`_mean.csv`。
- 完整统计：`experiments/results/paired_wilcoxon.csv`，25项主检验及对应描述性指标。

预览任务（不训练、不创建输出）：

```powershell
& '.\experiments\scripts\run_paired_wilcoxon.ps1' -PythonPath $pythonPath -DryRun
```

单独汇总已完成结果（不训练，缺少任一成功运行就停止汇总）：

```powershell
& '.\experiments\scripts\run_paired_wilcoxon.ps1' -PythonPath $pythonPath -SummarizeOnly
```

可用`-Datasets`、`-DataSeeds`、`-Methods`或`-InitSeeds`运行子集；全局统计始终
要求全部五个数据集、五份划分、六个方法及100–109全部种子。Python入口另有
`--dry-run`，会读取并核查该任务的源配置和冻结分区，不训练或写文件。

**统计口径：**每个新种子先平均五份固定划分的TEST ACC，再与同种子的基线
配对，共10对，而不是50个独立划分。双侧Wilcoxon枚举完整符号置换，正确处理
并列值和零差值；全零差值报告W=0、p=1。差值按原始0–1尺度保留12位小数，
避免浮点舍入打破实际并列。ACES对五基线×五数据集的25项TEST ACC检验统一
Holm校正，并报告均值/中位数差（百分点）、胜/平/负种子数。NMI/ARI仅作描述。

该检验条件于固定数据、固定划分和固定配置，只评估未参与选参的新初始化波动，
不推断独立人群或机构上的显著性。统计中的`*_seed_std`是10个种子单位的SD，
不同于旧主表五个划分均值的SD。无论是否显著，固定完成这10个新种子，不按p值
追加样本、更换预算、配置或主表。

开发验证（2026-11-05）：150份源配置与25份冻结分区核查通过；六种方法的小合成
数据训练、逐次/均值保存、成功项零调用续跑、失败重试和NaN保留通过。
完整汇总使用临时合成记录检查1500成功项、25项检验及n=10的种子单位；缺项或
归属不符会停止汇总。已知差值、并列/零差值、全零和Holm校正通过。
Windows PowerShell5.1的150任务展开、任意工作目录和失败后继续独立任务通过。
上述开发测试不作为论文数据，临时产物删除；正式1500次训练由用户本地执行。

正式运行核查（2026-11-05）：150对raw/mean、1500次训练全部completed，无失败、
重复、缺失种子或非有限质量值。25份冻结分区、150份源配置/评价设置及来源哈希一致；
独立复算全部配对符号秩和Holm，与保存结果一致。25项比较中16项为校正后正向显著。
对FFCM仅HAR显著（+9.27点，Holm p=.0488）；新种子USPS/COIL20/MNIST的ACES ACC
均值低于FFCM，但未达校正后的负向显著。原40--44质量主表保留，1500次单列为
M4初始化稳健性检验，不替换主表。更新后的正文、中文实验稿及Overleaf包见
[论文说明](../../paper/icde2027/README.md)。
