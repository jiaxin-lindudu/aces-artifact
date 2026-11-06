# centralized — 集中式聚类参考（保留历史 ceiling 文件名）

`centralized/` 提供 **无联邦化的参考基线，不是数学上界**：K-Means、模糊 c-means、谱聚类。它们跑在冻结 split 的
**池化 FIT 块**上（无客户端、无锚点图、无 Nyström 近似误差），并在冻结的 TEST 块上做**样本外**评分。
联邦方法的所有数字都拿它们当参照系：一个只报自己训练块的上界，没法用来评判在留出数据上计分的方法。

设计前提（详见 `centralized/__init__.py` 顶部）：

- **上界必须先调参**。未调参的上界不是界，只是"某个任意设置"。三种方法都在 `centralized/grids.py` 的
  网格上按 FIT 指标选优后才计分。
- **TEST 不得参与选择**。调参阶段（`compute_test=False`）连 TEST 都不打分，网格行的 `test_proxy_*` 全为
  NaN，因此不存在 TEST 信息回漏到选择的通道。
- **集中式数字只依赖 (dataset, data_seed, init_seed)**。客户端数与 Dirichlet beta 只决定 FIT 行如何分发给
  客户端，集中式方法根本看不到这份分配 —— 所以一个上界对所有联邦配置同时有效。

---

## 1. 文件与目录职责

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 包说明（上界语义、样本外规则、依赖分析）与方法注册表 `METHODS = {'central_kmeans', 'central_fcm', 'central_spectral'}`。 |
| `run_clustering_ceiling.py` | **唯一入口脚本**。加载冻结 split → 逐 (方法, init_seed) 测定迭代预算 → FIT 网格调参 → 重训最优配置并对 TEST 计分 → 写 4 个产物。 |
| `grids.py` | 三个方法的超参网格 `GRIDS` 与配置标签函数 `describe()`。每个网格 4 个格子，且刻意包含"确实不同的设置"。 |
| `kmeans.py` | `run(X_train, X_test, K, seed, cfg)` → sklearn KMeans，TEST 用自身 `predict`。 |
| `fuzzy_cmeans.py` | `run(...)` → 教科书 FCM；成员度**直接复用联邦侧的定义** `federated...fuzzy_membership_from_squared_distances`；TEST 用最近中心。 |
| `spectral.py` | `run(...)` → Ng-Jordan-Weiss 谱聚类；TEST 经 **Nyström 扩展**入同一嵌入（与 `ours` 同一机制）。 |
| `README.md` | 本文件。 |

---

## 2. 数据来源

读的是 `data` 包 **Stage 1** 的冻结 split（不是 Stage 2 分区）：

```
data/{dataset}/splits/{dataset}_centralized_seed_{s}.npz      # X_train/X_test/y_train/y_test/centers
data/{dataset}/splits/{dataset}_centralized_seed_{s}_metadata.json
```

经 `data.datasets_and_utils.load_centralized_split(dataset, data_seed)` 读取。脚本内的 `_load_split()`
同时兼容属性式（`split.X_train`）与历史元组式（`(meta, arrays)`）两种返回形态。

不变式：

- FIT / TEST 在**任何客户端分配之前**就已冻结，缩放与 PCA 只在 FIT 上拟合；
- 类别数 `K` = `len(centers)`，取不到时回退 `len(unique(y_train))`；
- 池化块由 `partition_sha256`（`X_train` 的 SHA-256）记录，落到每一行与 metadata，便于核对两次跑的是不是同一份数据。

本 checkout 中 `data/` 下**只有 `digits`** 准备好了 split（seed 0–4），其余数据集需先跑
`python data/prepare_centralized_splits.py --datasets <name> --data-seeds <s>`。
缺失时脚本直接报错退出（`return 1`）并提示先生成 split。

参考值（digits / seed 0）：`FIT=1437 TEST=360 K=10 d=64`，`preprocess=data.datasets_and_utils.apply_pca_preprocessing`。

---

## 3. 三种方法与样本外规则

| 方法 | 网格（`grids.py`） | 可调项 | TEST 样本外规则 | `test_basis` |
| --- | --- | --- | --- | --- |
| `central_kmeans` | `init ∈ {k-means++, random}` × `n_init ∈ {10, 1}` | `init`, `n_init` | sklearn 自带 `predict`（最近中心），精确 | `centers` |
| `central_fcm` | `fuzziness ∈ {1.1, 1.3, 1.5, 2.0}` | `fuzziness` | 原始空间最近中心（与 FIT 成员度同源的中心），按硬标签读数精确 | `centers` |
| `central_spectral` | `gamma_mode ∈ {one_over_d, median}` × `normalize_rows ∈ {True, False}` | `gamma_mode`, `normalize_rows`, `assign`, `n_init` | Nyström 扩展：TEST 点经其对 FIT 点的亲和度入同一嵌入，再取最近嵌入中心 | `nystrom` |

两条刻意的限制（改动前请先读注释，它们不是实现偷懒，是**诚实性约束**）：

- **谱聚类不提供 kNN 亲和图** —— 该图必须用到 TEST 点才能建，得到的测试分是转导式的，会虚高。
- **谱聚类只支持 `assign='kmeans'`** —— `discretize` 没有样本外形式，传入会直接 `ValueError`。

FCM 的两处关键点：

- 中心更新按 `u**m` 加权（不是 `u`），少这个指数是教科书级错误，会把所有中心塌到数据均值；
- 报 `crispness`（平均最大成员度，均匀时为 `1/K`），用来把"塌缩的跑"和"好的跑"区分开：64 维空间里
  `m >= 1.5` 落在均匀成员度不动点上，那时出来的 ACC 根本不是上界。`m=1.5/2.0` 两格是**故意留在网格里**作为证据的。

---

## 4. 工作流：BUDGET → TUNE → SCORE

对**每一个 (方法, init_seed)** 独立走一遍（`--init-seeds` 的每个种子各自调参、各自计分）：

1. **BUDGET（迭代预算）**：在 `DEFAULT_BUDGETS = (1, 2, 5, 10, 30, 100, 300, 1000)` 上扫 FIT，取
   **最小的**满足「FIT acc 与最大预算处严格相等（|Δ| < 1e-12）且未触及迭代上限」的预算；都不满足则回退到最大预算并打印告警。
   传 `--max-iter` 可跳过整段扫描（固定预算，此时不写 convergence CSV）。
2. **TUNE**：在该预算下遍历 4 个网格格子，只按 FIT 的 `--select-on`（acc / nmi / ari）取最大值择优。
   NaN / 缺失被视为 `-inf`；并列时保留先出现者。
3. **SCORE**：用选中配置**重训一次**并对 TEST 计分（这是每个 (方法, init_seed) 唯一一次看 TEST）。

自动预算下的开销：每个 (方法, init_seed) = 8 次预算扫描 + 4 个网格格子 + 1 次计分 = **13 次训练**；
3 个方法 × 5 个 init_seed = 195 次。用 `--max-iter` 固定预算可省掉 8/13。

---

## 5. 命令行

```bash
# 默认：digits / seed 0 / init seeds 0,1,2,3,4 / 自动预算 / 按 FIT acc 选
python centralized/run_clustering_ceiling.py --dataset digits

# 指定 split 与初始化种子，按 NMI 选
python centralized/run_clustering_ceiling.py --dataset digits --data-seed 1 \
    --init-seeds 40,41,42,43,44 --select-on nmi

# 固定迭代预算（跳过 8 档扫描，快得多；代价是不产出 convergence CSV）
python centralized/run_clustering_ceiling.py --dataset digits --max-iter 100

# 自定义预算候选 + 追加配置 + 单元格出错继续
python centralized/run_clustering_ceiling.py --dataset digits \
    --budgets 5,10,30,100 --extra-cfg tol=1e-6 --force
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--dataset` | `digits` | 数据集名；写盘时用 `_slug()` 做文件系统安全化（保留 `cifar10_vit` 这类全名）。 |
| `--data-seed` | `0` | **单个**冻结 split 的种子：调参与计分都在它上面。 |
| `--init-seeds` | `0,1,2,3,4` | 初始化种子，逗号分隔；每个种子独立调参与计分。 |
| `--max-iter` | 无 | 给所有方法/init 固定迭代预算；给定则不做预算扫描。 |
| `--budgets` | `1,2,5,10,30,100,300,1000` | 自动预算的候选档位。 |
| `--select-on` | `acc`（可选 `nmi`/`ari`） | FIT 指标，用于挑配置。 |
| `--extra-cfg` | 空 | `key=value,...` 合并进**每个方法**的配置；数值串会自动转 int/float（如 `tol=1e-6`）。 |
| `--out-dir` | 无 | 默认 `results/{dataset}/seed_{data_seed}`；显式传入优先。 |
| `--force` | 关 | 网格单元格抛异常时继续（打印 `x ...` 并跳过），而不是立刻 `return 1`。 |
| `--quiet` | 关 | 关闭 per-init 进度打印；错误与最终汇总仍打印。 |

注意：**没有** `--tune-seed` / `--data-seeds` / `--skip-convergence-check`（旧 README 里的这几个参数在当前脚本中不存在；
`--data-seed` 是单个整数，不是列表）。

---

## 6. 产物

默认落在 `experiments/results/{dataset}/seed_{data_seed}/`（`schema.RESULTS_DIR` 之下）：

| 文件 | 内容 | 行数（示例） |
| --- | --- | --- |
| `centralized_grid_{dataset}_seed{N}.csv` | 每个 (方法, init_seed, 网格格子) 一行，仅 FIT 指标；`test_proxy_*` 全 NaN | 3 方法 × init 数 × 4 |
| `centralized_ceiling_{dataset}_seed{N}.csv` | 每个 (方法, init_seed) 一行：选中配置的 FIT + TEST | 3 × init 数 |
| `centralized_convergence_{dataset}_seed{N}.csv` | 预算扫描：`method,init_seed,max_iter,fit_acc,n_iter,hit_cap` | 仅自动预算时产出（空则不写） |
| `centralized_run_metadata_{dataset}_seed{N}.json` | 数据规模、preprocess、`arrays_sha256`、`init_seeds`、`select_on`、预算模式，以及 `per_method` 的逐种子预算与胜出配置 | — |

CSV 列即 `experiments/schema.py` 的 `RESULT_COLUMNS`（全列占位、缺失为 NaN），末尾再追加脚本自有的
`config, max_iter, n_iter, hit_cap`。关键列：

- 分组键 `GROUP_KEYS`：`dataset, method, data_seed, dirichlet_beta, clients, clusters, study, study_value`
  （`study` 取值 `ceiling_grid` / `ceiling`）；
- 指标 `acc/nmi/ari/silhouette/davies_bouldin/calinski_harabasz`，以及同名的 `all_fit_*` 与 `test_proxy_*`；
- 溯源 `init_seed, partition_sha256, test_basis, oos_status, use_ckks, ..., preprocess, note`。

`note` 列自带可读摘要，例如
`centralized FCM | cfg: m=1.1 | max_iter=30 iters=30 cap=True | selected on FIT; TEST is held out | mean max membership=0.412 (uniform 0.100)`。

**与联邦结果的位置关系**（两侧 sibling 目录，便于对照）：

```
experiments/results/digits/
├── seed_0/          # 本包：集中式上界（--data-seed 0）
└── federated_0/     # experiments/scripts/grid_search_ours.py：同一 data_seed 的联邦网格
```

---

## 7. Python API

```python
from centralized import METHODS                 # {'central_kmeans', 'central_fcm', 'central_spectral'}
from centralized.grids import GRIDS, describe   # 网格与稳定的单行配置标签
from data.datasets_and_utils import load_centralized_split

split = load_centralized_split('digits', 0)
X_train, y_train = split.X_train, split.y_train
X_test, y_test = split.X_test, split.y_test
K = len(split.centers)

for cfg in GRIDS['central_spectral']:
    out = METHODS['central_spectral'](X_train, X_test, K, seed=0, cfg=cfg)
    print(describe('central_spectral', cfg), out['train_labels'], out['test_labels'],
          out['test_basis'], out['extra'])
```

统一签名：`run(X_train, X_test, K, seed, cfg=None) -> dict`，返回
`{'train_labels', 'test_labels', 'test_basis', 'extra'}`；`extra` 至少含 `n_iter` 与 `hit_cap`
（KMeans 还会给 `inertia`，FCM 给 `crispness`/`uniform_crispness`，谱聚类给 `gamma`/`gamma_mode`/`normalize_rows`/`top_eigenvalues`）。
`cfg` 与模块内 `DEFAULT_CFG` 合并，未给的键取默认值。

评分请统一用 `federated.ours.tracker.clustering_quality(X, y, labels)`（脚本也用它），
以保证与联邦侧的指标定义一致。

---

## 8. 已知边界与一处待办

1. **`centralized/__init__.py` 提到的 `centralized/verify_invariance.py` 在本 checkout 中不存在**
   （全仓搜索 0 个文件）。也就是说"集中式数字与 (clients, beta) 无关"这条目前是**文档声明，尚未有脚本验证**。
   若要落实，写一个比对不同 (clients, beta) 分区池化块 SHA-256 并重跑三方法的小脚本即可。
2. `__init__.py` 说输出行的 `clients` / `dirichlet_beta` 列标明了池化块取自哪个分区 —— 但
   `run_clustering_ceiling.py` 读的是 Stage 1 split，这两列**实际是 NaN**（见产物 CSV）。它们只是
   `schema.RESULT_COLUMNS` 的占位列，别拿去分组。
3. **预算是在方法的默认配置下测定的**：KMeans 用 `k-means++ / n_init=10`，谱聚类用
   `one_over_d / normalize_rows=True`，而 **FCM 用默认 `fuzziness=2.0`（正是塌缩那一格）**。
   塌缩设置的收敛判据失真，所以 FCM 各 init_seed 常落在小预算（digits/seed 0 实测均为 30）。
   想让预算更有意义，可用 `--extra-cfg fuzziness=1.1` 让预算扫描跑在有意义的设置上。
4. `--force` 只包住**网格单元格**的异常；预算扫描与最终计分阶段的异常仍会直接抛出。
5. 脚本 docstring 的 `Run:` 示例写的是 `centralized/run_ceiling_grid.py`，实际文件名是
   `run_clustering_ceiling.py`（docstring 笔误，不影响运行）。
6. 谱聚类的亲和矩阵是 O(n²)（digits 的 1437×1437 尚可，`cifar10_vit` 之类要先估内存）；
   `median` 模式的带宽取自最多 500 点的子样本。

## 9. 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `Cannot load split for X seed=N` | 该 split 未生成；先跑 `python data/prepare_centralized_splits.py --datasets X --data-seeds N`。 |
| `! metadata data_seed=A differs from --data-seed=B` | split 元数据里的种子与 CLI 不一致；核对 `data/{dataset}/splits/*_metadata.json`。 |
| 出现 `! ... even max_iter=N hits the cap` | 最大预算处仍未收敛，预算不可靠；加大 `--budgets` 上限或改用物理上限 `--max-iter`。 |
| 只有 grid CSV、没有 convergence CSV | 用了 `--max-iter`（固定预算），本就没有扫描数据。 |
| `x <method> init=N cfg=...: <error>` 后中途退出 | 某个格子抛异常且未加 `--force`；加 `--force` 跳过该格子继续。 |
| FCM 的 ACC 明显偏低且 `crispness ≈ 1/K` | 成员度塌到均匀分布；检查是否选中了 `m>=1.5`（这正是网格保留这两格要暴露的现象）。 |
| 谱聚类报 `only assign='kmeans' is supported` | 传了 `assign='discretize'`；它没有样本外形式，会破坏 ceiling 的诚实性。 |


## 10. 触顶复查与路径（2026-10-24）

保留 BUDGET → TUNE → SCORE。若FIT选中的配置触及预算，新默认 `--recheck-max-iter 1000` 在最终重训时提高其预算，随后只进行一次TEST计分；不根据TEST改选配置。metadata的 `cap_rechecks` 保存原预算与复查预算，最终CSV的 `hit_cap/n_iter/max_iter` 表示实际最终拟合。设 `--recheck-max-iter 0` 可禁用；仍触顶须报告未收敛，不宣称预算充分。

```powershell
python centralized/run_clustering_ceiling.py --dataset digits --data-seed 0 --init-seeds 40,41,42,43,44 --out-dir experiments/results/digits/seed_0_recheck
```

本命令已完成：新15条结果无触顶，旧 `seed_0` 结果保留。仍只生成本README原有四种产物。当前最优FCM的平均TEST ACC由原先不充分预算的约0.7811变为0.7800；最终数值请读取CSV，不手工拼接不同预算的实验。

数据默认从 `<ROOT>/data` 读取，不再受调用位置影响；可显式 `--data-dir`。默认输出仍是 `experiments/results/{dataset}/seed_{data_seed}`；保留已有结果请指定新的 `--out-dir`。谱方法先检查内存，不足时抛错，不对某一个方法静默抽样。CIFAR分支无有效真实中心时，仅从FIT标签确定全局K，不把NaN占位中心当作模型输入。

集中式谱聚类使用 float64 稠密亲和矩阵，原位归一化，通过 SciPy `eigh(driver='evr', subset_by_index=...)` 只计算最大的 K 个特征对，随后释放亲和矩阵；TEST 的 Nyström 扩展按512行分块计算。保持全部冻结FIT/TEST样本与原有网格，特征求解器变化可能带来数值舍入差异。该实现的内存检查按两份 n×n float64 矩阵估算，仍要求不超过可用物理内存的一半；8000条FIT约需0.95 GiB工作区，检查要求可用内存至少1.91 GiB。FedSC的分块重建、稀疏图路径也使用两份矩阵大小的保守预算，覆盖其数据加载中的pdist/median临时副本；不再套用旧版六份稠密矩阵估算。

集中式参考不需按β和客户端数重复训练，但汇总前必须核对四个规范数组哈希；不同数据划分、预处理或TEST预测规则的数值不能直接合并。五个初始化只衡量同一划分的波动，不能当作五份独立数据。
