# cuLA KDA MTP Verify Kernel 开发计划

## Context

`mtp.md`（1329 行）已完成对 MTP (Multi-Token Prediction) / Speculative Decoding 在 LA kernel 上的完整设计：问题定义、vLLM/SGLang 实现对比、混合 Snapshot+Recompute 策略、CuTe DSL kernel 架构、Conv1d 同步约束等。**本计划不重新设计，只把 mtp.md 落地为可执行的实现路线图。**

### 当前 cuLA 状态（已确认）

- `cula/ops/kda_decode.py` 已实现完整的单 token KDA decode（4 个变体：small/large batch × dense/varlen），1844 行处的 `kda_decode()` 是公开 API。
- 编译缓存机制成熟：`_compiled_kernels` 按 16-tuple 构成 key，constexpr 参数走 `_get_compiled_kernel()`（1537–1667 行）。
- **当前无任何 MTP/verify kernel**。`cula/kda/chunk.py` 中虽有 `return_intermediate_states` 标志，但 kernel 端未实现。
- 测试模板：`tests/test_kda_decode.py` 内 `torch_kda_decode_ref()` 提供 pure-Python ground truth；可直接派生 multi-token 版作为 verify ground truth。
- 学习文档：`docs/learning_kda_fwd_sm90.md` (~2700 行) 已覆盖 forward kernel；verify 主题需新建 `docs/learning_kda_verify.md` 持续沉淀。

### 关键参考实现（已定位）

- **FLA Triton verify reference**：`third_party/flash-linear-attention/fla/ops/kda/fused_recurrent.py` 中 `fused_recurrent_kda_fwd_kernel`。它已包含本计划 Phase 1 需要的全部要素：`T` 维度循环、`IS_SPEC_DECODING` 选取初始 state、KDA gate 形状（`a` 为 `[..., HV, K]` 向量）、`USE_QK_L2NORM_IN_KERNEL`。**Phase 1 直接派生此 kernel**，无需从 SGLang 仓库 vendor。
- **CuTe DSL 派生基础**：`cula/ops/kda_decode.py:296` 起的 `kda_kernel_small_batch`。Phase 2 在其外层加 `for t in range(max_steps)` 循环，state（`sData`）在 SMEM 跨 token 保持。

### 用户决策（已对齐，2026-05-26）

- 覆盖范围：4 个 Phase 全部纳入路线图
- KDA only（GDN 不在本计划范围）
- Snapshot 模式优先实现；recompute_state API 预留但不实现 replay 逻辑
- 线性 spec 优先；`retrieve_parent_token` API 预留但 kernel 内 assert 为 None

---

## 总体架构（一句话回顾 mtp.md）

verify kernel = "在 fused for-loop 内对 `max_steps` 个 draft token 做 sequential recurrent update，state 在 SMEM 跨 token 保持，每步把 state 异步写到 `intermediate_states_buffer`，不写主 state pool"。verify 后由 host 侧根据 `accepted_lens` 做 gather，把选中的 state 写回主 pool。

---

## Phase 0 — 远程 H20 开发环境 setup

**背景**：本地（Mac）无 GPU，所有验证测试与 benchmark 都在远程 H20 八卡机执行。本地仅用于代码编辑（Claude Code）。

### 0.1 远程机器现状（已探测确认）

| 项 | 值 |
|---|---|
| 登录 | `ssh -i ~/sfcloudtest.pem root@60.205.204.134 -p28` |
| GPU | 8× NVIDIA H20（compute capability **sm_90a**，Hopper） |
| CUDA Toolkit | 12.8（`nvcc --version` 输出 `release 12.8`） |
| OS / kernel | Linux 5.10.134 (Alibaba Cloud Linux 8) |
| Conda | miniconda3 装于 `/var/lib/container/miniconda3` |
| 推荐工作目录 | `/var/lib/container/<user>/cuLA-dev`（4.4T 可用，根盘只剩 56G） |

**关键约束**：
- README 写明的"NVCC 12.9+"仅适用于 Blackwell (SM10X)。**H20 是 SM90，可用 NVCC 12.8**——build 时设 `CULA_DISABLE_SM100=1 CULA_DISABLE_SM103=1`。
- pyproject.toml 硬 pin `nvidia-cutlass-dsl==4.4.2` + `apache-tvm-ffi==0.1.9`。
- 现有 conda envs (vllm-backend / inference / train / ms-swift) 都有不同程度的版本冲突或职责冲突 → **推荐新建独立 `cula-dev` env**，不在已有 env 上覆盖装 cuLA 依赖。

### 0.2 一次性 Setup 步骤

```bash
# (本地) 登录远程
ssh -i ~/sfcloudtest.pem root@60.205.204.134 -p28

# (远程) 1. 工作目录
WORKDIR=/var/lib/container/cuLA-dev
mkdir -p $WORKDIR && cd $WORKDIR

# (远程) 2. 创建 conda env (py3.12 匹配 ruff target-version)
source /var/lib/container/miniconda3/etc/profile.d/conda.sh
conda create -n cula-dev python=3.12 -y
conda activate cula-dev

# (远程) 3. 装 torch (cu128 匹配本机 CUDA 12.8 toolkit)
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
# 验证: python -c "import torch; print(torch.__version__, torch.version.cuda)"

# (远程) 4. 装 cuLA 依赖 (硬 pin 版本)
pip install nvidia-cutlass-dsl==4.4.2 apache-tvm-ffi==0.1.9 pytest matplotlib pandas

# (远程) 5. 克隆 cuLA + 子模块 (csrc/cutlass + third_party/flash-linear-attention)
cd $WORKDIR
git clone <仓库地址> cuLA   # 用 user 自己的 fork 或 origin
cd cuLA && git submodule update --init --recursive

# (远程) 6. 装 fla (editable, benchmark 对照需要)
pip install -e third_party/flash-linear-attention

# (远程) 7. 装 cuLA (editable, 仅 SM90 build, 关掉 Blackwell)
CULA_DISABLE_SM100=1 CULA_DISABLE_SM103=1 \
  pip install -e . --no-build-isolation

# (远程) 8. 烟囱测试: 先确认 decode kernel 在 H20 上能跑通
pytest tests/test_kda_decode.py -v -x -k "small_batch and N1 and not varlen" 2>&1 | tail -20
```

### 0.3 验证清单

```bash
# 远程激活 cula-dev env 后:
python -c "import torch; print('cuda cap:', torch.cuda.get_device_capability(0))"
# 期望: (9, 0)

python -c "import cutlass.cute as cute; print('cute OK')"
# 期望: cute OK

python -c "from cula.ops import kda_decode; print('cula OK')"
# 期望: cula OK

pytest tests/test_kda_decode.py -v -x -k "small_batch and N1" 2>&1 | tail -10
# 期望: 至少 1 个 test 通过
```

### 0.4 本地 ↔ 远程代码同步

- **本地编辑**：Claude Code 在 Mac `/Users/zhouaihui/code/cuLA` 修改代码
- **远程执行**：H20 上 `/var/lib/container/cuLA-dev/cuLA` 运行测试/benchmark
- **同步策略**（二选一，建议两者都用）：

#### A. rsync（快速迭代）

新建 `scripts/sync_remote.sh`（仅本计划期间使用，**不提交到 git**——加 `.gitignore`）：

```bash
#!/usr/bin/env bash
# scripts/sync_remote.sh — push working tree to remote H20 dev box
set -e
LOCAL=/Users/zhouaihui/code/cuLA/
REMOTE=root@60.205.204.134:/var/lib/container/cuLA-dev/cuLA/
rsync -avz --delete \
  --exclude='.git/' --exclude='build/' --exclude='*.egg-info/' \
  --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='csrc/cutlass/' --exclude='third_party/' \
  -e "ssh -i ~/sfcloudtest.pem -p 28" \
  "$LOCAL" "$REMOTE"
```

⚠️ 注意：脚本排除了 `csrc/cutlass/` 和 `third_party/`（子模块体积大，且远程已通过 `git submodule update` 拉好）。如修改子模块需单独同步。

#### B. git push/pull（阶段性提交）

每完成一个 sub-phase（如 Phase 1 的 Triton kernel 通过测试），本地 commit + push，远程 pull。这条路径适合需要保留 commit history 的大改动。

### 0.5 远程一键测试脚本

新建 `scripts/run_remote_tests.sh`（不提交到 git）：

```bash
#!/usr/bin/env bash
# scripts/run_remote_tests.sh — sync code then run pytest on remote
set -e
bash "$(dirname "$0")/sync_remote.sh"
ssh -i ~/sfcloudtest.pem root@60.205.204.134 -p28 \
  "source /var/lib/container/miniconda3/etc/profile.d/conda.sh && \
   conda activate cula-dev && \
   cd /var/lib/container/cuLA-dev/cuLA && \
   pytest ${1:-tests/test_kda_verify.py} -v -x"
```

使用：`bash scripts/run_remote_tests.sh tests/test_kda_verify.py::test_phase1`

### 0.6 已知陷阱

1. **OpenBLAS warning**：`OpenBLAS WARNING - could not determine the L2 cache size on this system, assuming 256k` 在远程是常态，不影响功能，可设 `OPENBLAS_NUM_THREADS=1` 抑制。
2. **GPU 资源争抢**：八卡机可能被其它任务占用，跑测试前先 `nvidia-smi` 看 free GPU，必要时 `CUDA_VISIBLE_DEVICES=N` 指定空闲卡。
3. **首次 build 时间长**：cuLA 需 nvcc 编译 SM90 kernel，第一次 `pip install -e .` 大约 5-15 分钟。后续修改 Python 代码无需重 build。
4. **ruff target_version 是 py312**，但 Python 3.10/3.11 也可装（pyproject 说 `>=3.10`）；选 3.12 是为了和 lint 完全对齐。
5. **本机 Mac 不要试图 `pip install cuLA`**：setup.py 强制要求 torch + CUDA，没 GPU 装不上。

---

## Phase 1 — Triton KDA verify baseline（正确性）

**目标**：写出可跑通的 Triton verify kernel，作为 Phase 2 CuTe DSL 实现的参考基线。

### 1.1 新建文件

- `cula/ops/kda_verify_triton.py`（新建）
  - 派生自 `third_party/flash-linear-attention/fla/ops/kda/fused_recurrent.py` 的 `fused_recurrent_kda_fwd_kernel`
  - 改造点（vs FLA 版）：
    - 新增 constexpr `DISABLE_STATE_UPDATE`：True 时跳过最终 `tl.store(p_h0, b_h)`，确保不写主 state pool（决策 1，mtp.md §4.4）
    - 新增 constexpr `CACHE_INTERMEDIATE_STATES` + 入参 `intermediate_states_buffer`、`intermediate_state_indices`：每 token 循环末尾把 `b_h` 异步写出
    - 删除 vLLM 风格的 `ssm_state_indices[i_n, i_t]` 间接寻址，改为连续 buffer 直接 offset（mtp.md §4.3.1）
    - 入参 `retrieve_parent_token` 暂存但 kernel 内不使用（Phase 1 强制线性）

### 1.2 Public API

```python
def kda_verify(
    A_log, dt_bias,
    q, k, v, a, b,
    initial_state_source, initial_state_indices,
    intermediate_states_buffer,        # (N, max_steps, HV, V, K)
    intermediate_state_indices,         # (N,)
    *,
    cache_intermediate_states=True,
    disable_state_update=True,
    recompute_state=False,              # Phase 3 启用；Phase 1 必须为 False
    saved_qkvab=None,                   # Phase 3 启用；Phase 1 必须为 None
    retrieve_parent_token=None,         # 树状预留；Phase 1 必须为 None
    scale=None,
    use_qk_l2norm_in_kernel=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    state_layout="vk",
) -> torch.Tensor                       # output: (1, sum_seqlen, HV, V)
```

Phase 1 在函数顶部对 `recompute_state` / `saved_qkvab` / `retrieve_parent_token` 做 assert，明确"接口已预留，本 Phase 未实现"。

### 1.3 测试

- 新建 `tests/test_kda_verify.py`，新增 `torch_kda_verify_ref()`：复用 `tests/test_kda_decode.py:torch_kda_decode_ref()` 的逻辑做 `max_steps` 次循环，每步保存中间 state 到 list。
- 覆盖 `spec_len ∈ {1, 2, 4, 5, 8}`、`N ∈ {1, 8, 32}`、`HV ∈ {4, 128}`、`use_qk_l2norm ∈ {True, False}`、`state_layout ∈ {"vk", "kv"}`。
- 精度标准（mtp.md §6.10.2）：output bf16 rtol < 1e-4；intermediate state fp32 rtol < 1e-5。

### 1.4 验证

```bash
pytest tests/test_kda_verify.py -v
```

---

## Phase 2 — CuTe DSL KDA verify kernel（性能）

**目标**：在 Phase 1 正确性基础上，用 CuTe DSL 实现 fused verify kernel，利用 SMEM state 跨 token 保持 + cp.async snapshot overlap。

### 2.1 修改文件

- `cula/ops/kda_decode.py`（在现有文件追加，**不替换** decode kernel）
  - 新增 `kda_verify_kernel_small_batch`（派生自 `kda_kernel_small_batch`，see 296–554 行）
    - 在 `pool_idx >= 0` 分支内，外层加 `for t in cutlass.range(max_steps)` 循环（mtp.md §6.3）
    - `t == 0` 时从 `h0_source` 加载初始 state 到 `sData`；`t > 0` 时 `sData` 保持，不重载（关键优化）
    - 每个 token 循环内：q/k/a/b 用 `[i_n, t, ...]` 索引（多了 t 维度）
    - 循环末尾根据 `cache_intermediate_states` 写 `intermediate_states_buffer[i_n, t, i_hv, ...]`
    - 写 snapshot 用 `cp.async`，commit_group 在下一 token 的 q/k/a/b load 之前；wait_group 只需在最终 state 写回前做一次全局 `cp.async.wait_all()`（mtp.md §6.5.2）
    - `max_steps` 必须是 `cutlass.Constexpr[int]`：循环完全展开，每个值编译一个变体（mtp.md §6.5.3）
  - 新增 `kda_verify_kernel_large_batch`（派生自 `kda_kernel_large_batch`，see 796 行起，结构同上）
  - 新增 varlen 版本（派生自 `kda_kernel_small_batch_varlen` / `kda_kernel_large_batch_varlen`）
  - 新增 `kda_verify()` Python 入口，复用 `_get_compiled_kernel()` 缓存（key 中加入 `max_steps`、`cache_intermediate_states`、`disable_state_update`）

### 2.2 SMEM 布局（mtp.md §6.4）

复用现有 `sData / sK / sQ / sG / sGK / smem_o`，不增加新 SMEM 区域。q/k/a/b 每步重新加载到同一 buffer。

### 2.3 关键约束（必须遵守，mtp.md §6.5.2）

- snapshot store 是 `sData(SMEM) → GMEM`，**不会修改 sData**，所以与下一 token 的 sData read 不冲突——可以完全 overlap
- 唯一同步点：`max_steps` 循环结束后、写回最终 state（仅 `disable_state_update=False` 时）前的 `cp.async.wait_all()`
- 不在每个 token 之间插入 wait_group，否则 overlap 失败

### 2.4 测试与性能

- 复用 `tests/test_kda_verify.py`，新增 `backend=["triton", "cute"]` 参数化，两者输出对比 + 与 torch ref 对比
- 新建 `benchmarks/bench_kda_verify.py`（沿用 `benchmarks/bench_kda_decode.py` 风格），3-way 对比：
  - cuLA CuTe verify
  - cuLA Triton verify
  - cuLA decode kernel × spec_len 次串行调用
- 验证 mtp.md §6.5.5 表格中的预期收益（节省 4 次 launch ≈ 32μs + state read traffic）

### 2.5 编译变体爆炸的风险

`max_steps ∈ {1,2,4,5,8}` × 现有 16-tuple key = 编译变体数 ×5。在 `_compiled_kernels` 中新增 `max_steps` 维度，首次调用每个 spec_len 都会触发编译（~秒级）。后续运行命中缓存。**这是预期开销**，与现有 decode kernel 处理 `scale` 等 constexpr 的方式一致。

---

## Phase 3 — KV Cache 模式 + Recovery Kernel

### 设计动机

Full state snapshot 每步写 8MB (HV=128, V=K=128)，是 verify kernel 的主要带宽瓶颈。
社区讨论指出 chunk-based verify 可降低 state I/O，但实现复杂。

更优方案：verify 时只缓存每步的中间变量 (v_new, gate, k_norm)，accept 后用线性扫描恢复 state。

**每步 cache 量**：
- v_new:  (HV, V) fp32 = 64KB
- gate:   (HV, K) fp32 = 64KB
- k_norm: (H, K)  fp32 = 4KB
- Total: **132KB/token** ← vs 8MB full state = **60x 更小**

**恢复公式**（无依赖线性扫描，不需要 delta correction）：
```
for t in range(accepted_len):
    state = gate_t * state + v_new_t[:, :, None] * k_norm_t[None, :, :]
```

### 3.1 Verify kernel KV cache 模式

- 在 `kda_verify_kernel_*` 中新增 constexpr `cache_mode`：
  - `"full_state"`: 当前行为，每步写完整 state snapshot（兼容模式）
  - `"kv"`: 每步只写 (v_new, gate, k_norm) 到紧凑 buffer
- kernel 内部改动极小：v_new 和 gate 已经在寄存器里，只需加几条 store 指令
- 不写 intermediate_states_buffer，改写 kv_cache + gate_cache + knorm_cache

### 3.2 Recovery kernel

新建 `kda_verify_recover()`：接收 initial_state + cached (v_new, gate, k_norm) + accepted_lens → 输出 final state。

- 无 L2norm、无 softplus、无 sigmoid、无 delta correction
- 纯线性扫描，比 kda_decode 还简单
- 可用 Triton 或 CuTe DSL 实现

### 3.3 API 变化

```python
# KV cache buffers（比 intermediate_states_buffer 小 60x）
vnew_cache = torch.empty(N, T, HV, V, dtype=torch.float32, device="cuda")
gate_cache = torch.empty(N, T, HV, K, dtype=torch.float32, device="cuda")
knorm_cache = torch.empty(N, T, H, K, dtype=torch.float32, device="cuda")

# verify 时
o = kda_verify(..., cache_mode="kv",
               vnew_cache=vnew_cache, gate_cache=gate_cache, knorm_cache=knorm_cache)

# accept 后
final_state = kda_verify_recover(
    initial_state, vnew_cache, gate_cache, knorm_cache, accepted_lens)
```

### 3.4 保留的兼容模式

`cache_mode="full_state"` 保留当前完整 snapshot 行为（用于调试、精度验证）。
原 Phase 3 计划的 bf16 snapshot / selective snapshot 不再需要（132KB << 4MB bf16）。

### 3.5 测试

扩展 `tests/test_kda_verify.py`：
- `cache_mode="kv"` 时 verify output 与 torch ref 对比
- `kda_verify_recover` 恢复的 state 与 torch ref 逐步比对
- 覆盖不同 accepted_lens（包括 0, 1, T-1, T）

---

## Phase 4 — 框架集成与 host 侧 state 选取

### 4.1 Host 侧 state 恢复工具

新建 `cula/ops/kda_verify_utils.py`：

```python
def select_accepted_state(
    initial_state: torch.Tensor,      # (N, HV, V, K) 或 (pool_size, HV, V, K)
    vnew_cache: torch.Tensor,          # (N, max_steps, HV, V)
    gate_cache: torch.Tensor,          # (N, max_steps, HV, K)
    knorm_cache: torch.Tensor,         # (N, max_steps, H, K)
    accepted_lens: torch.Tensor,       # (N,) int
    state_pool: torch.Tensor,          # (pool_size, HV, V, K) 写入目标
    state_pool_indices: torch.Tensor,  # (N,) 写入位置
):
    """根据 accepted_lens 做线性扫描恢复 state，写回主 state pool。"""
```

同时保留 full_state 兼容模式的 gather 路径：
```python
def gather_accepted_state(
    intermediate_states_buffer: torch.Tensor,  # (N, max_steps, HV, V, K)
    accepted_lens: torch.Tensor,                # (N,) int
    state_pool: torch.Tensor,                   # (pool_size, HV, V, K)
    state_pool_indices: torch.Tensor,           # (N,) 写入位置
):
    """直接 gather（仅 cache_mode="full_state" 时使用）。"""
```

### 4.2 Varlen 输入/输出支持 ✅ (Phase 2 已完成)

已在 Phase 2 实现。`kda_verify` 支持 dense 和 varlen 两种输入布局，4 个 kernel 变体（small/large × dense/varlen）。

### 4.3 SGLang 集成（参考集成，不一定要合入 SGLang）

- 在 `cula/ops/__init__.py` 暴露 `kda_verify` 与 `select_accepted_state`
- 提供 `docs/integration_sglang.md`：示例代码展示如何在 SGLang `RadixLinearAttention.target_verify()` 中替换现有 chunk_kda 调用为 `kda_verify` + `select_accepted_state`

### 4.4 Conv1d 同步约束文档化（mtp.md §6.8）

在 `kda_verify` docstring 和 `docs/integration_sglang.md` 明确：调用方必须确保 Conv1d state 和 recurrent state 在同一步完成 truncate/gather，否则 state 不一致。cuLA 不在 kernel 内处理 Conv1d。

---

## 关键文件清单

| 路径 | 操作 | Phase |
|---|---|---|
| `scripts/sync_remote.sh` | 新建（本地→远程 rsync，不提交 git） | 0 |
| `scripts/run_remote_tests.sh` | 新建（一键远程测试，不提交 git） | 0 |
| `cula/ops/kda_verify_triton.py` | 新建 | 1 |
| `cula/ops/kda_decode.py` | 追加 `kda_verify_kernel_*` 4 个变体 + `kda_verify()` 入口 | 2 |
| `cula/ops/kda_verify_utils.py` | 新建 `select_accepted_state` + replay 补齐 | 4 |
| `cula/ops/__init__.py` | 导出 `kda_verify`, `select_accepted_state` | 4 |
| `tests/test_kda_verify.py` | 新建 + 增量扩展 | 1/2/3 |
| `benchmarks/bench_kda_verify.py` | 新建 | 2 |
| `docs/learning_kda_verify.md` | 新建 + 每 Phase 增量更新（用户偏好） | 1/2/3/4 |
| `docs/integration_sglang.md` | 新建 | 4 |

## 复用的现有代码

- `cula/ops/kda_decode.py:296` 起 `kda_kernel_small_batch` — Phase 2 派生模板
- `cula/ops/kda_decode.py:796` 起 `kda_kernel_large_batch` — Phase 2 派生模板
- `cula/ops/kda_decode.py:1537` 起 `_get_compiled_kernel` — Phase 2 编译缓存机制，加 `max_steps` 等到 key
- `cula/ops/kda_decode.py:1844` 起 `kda_decode` — Phase 4 replay 路径直接调用
- `tests/test_kda_decode.py:42` 起 `torch_kda_decode_ref` — Phase 1 派生为 `torch_kda_verify_ref`
- `third_party/flash-linear-attention/fla/ops/kda/fused_recurrent.py:31` 起 `fused_recurrent_kda_fwd_kernel` — Phase 1 Triton baseline 直接派生

---

## 端到端验证清单

每完成一个 Phase，按顺序运行：

```bash
# Phase 1
pytest tests/test_kda_verify.py -v -k "triton"

# Phase 2
pytest tests/test_kda_verify.py -v                  # triton + cute 双后端
python benchmarks/bench_kda_verify.py               # 验证性能收益

# Phase 3
pytest tests/test_kda_verify.py -v -k "bf16 or selective"

# Phase 4
pytest tests/test_kda_verify.py -v                  # 全部
# 手动验证: 走通 varlen 路径 + select_accepted_state 端到端
```

每个 Phase 完成后同步更新 `docs/learning_kda_verify.md`（用户偏好：每个 topic 完成后更新学习文档）。

---

## 范围外（明确不做）

- GDN verify（is_kda=False 分支）
- 树状 speculation 的 kernel 内 `retrieve_parent_token` 回滚（仅 API 预留）
- `recompute_state=True` 的 host 侧 replay 编排（API 与 q/k/v/a/b 保存预留，但 replay 不实现）
- Conv1d state 管理（明确归属调用方）
- vLLM 集成（仅做 SGLang 集成示例文档）
