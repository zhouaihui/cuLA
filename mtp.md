# MTP (Multi-Token Prediction) 与 LA Kernel 高效结合方案

## 1. 问题定义

### 1.1 背景

cuLA 仓库实现了高效的 Linear Attention (LA) decode kernel，核心是 recurrent state 的递推更新：

**KDA (Kernel Delta Architecture) Decode:**
```
gate = exp(-A_log * softplus(a + dt_bias))   # 逐元素衰减门控
v_new = sigmoid(b) * (v - H^T @ (gate * k_norm))  # delta rule 修正
H_new = diag(gate) * H + k_norm ⊗ v_new     # 状态更新
output = H_new @ q_scaled                     # 输出计算
```

**LA Decode:**
```
S_new = decay * S + k ⊗ v                    # 简单指数衰减 + 外积
output = q @ S_new
```

当前 kernel 设计为 **单 token decode**：每次 kernel launch 处理 1 个 token，更新 state 后写回 global memory。

### 1.2 MTP/Speculative Decoding 带来的挑战

**符号约定**：
- **N**：同时参与投机验证的 sequence 数量（注意：不是 serving 的总 batch size，而是同一时刻处于 verify phase 的 seq 数量，通常远小于总 batch）
- **L**：模型 Attention 层数（典型 24~64），每层有独立的 recurrent state
- **HV**：value head 数量（支持 GQA/GVA，典型 128）
- **K, V**：key/value head 维度（典型 128）
- **spec_len (k)**：投机生成的 token 数量（典型 4~8）
- **accepted_len (m)**：验证后被接受的 token 数量（1 ≤ m ≤ k）

MTP (Multi-Token Prediction) / Speculative Decoding 在推理时同时产生多个 token，然后由主模型验证。与标准 decoding 有两个根本差异：

1. **每个 sequence 的 token 数不只有 1**：投机模型可能一次生成 k 个 token（典型 k=2~8）
2. **token 可能被拒绝，不能盲目叠加到 KV State 中**：验证后只接受前 m 个 token（m ≤ k），被拒绝的 token 的贡献不能保留

### 1.3 核心矛盾

**LA 的 recurrent state 是累积混合的，不可逆：**

KV Cache 是 append-only 的——拒绝 token 只需截断尾部。但 LA 的 state 中，每个 token 的贡献通过 decay 衰减后与后续 token 的贡献纠缠在一起。一旦被拒绝 token 的 `v⊗k` 混入了 `H`，无法通过减法精确回退，因为后续 token 的衰减和修正已经基于错误的 `H` 计算了。

> **核心问题：如何让 LA kernel 在投机执行多 token 后，以最小代价回退到正确的中间 state？**

---

## 2. 方案对比

### 2.1 三种策略

| 方案 | 思路 | 额外计算 | 额外 kernel launch | 额外显存 | 复杂度 |
|------|------|---------|-------------------|---------|--------|
| **A. Checkpoint + Replay** | 投机前不更新 state pool（verify kernel 用 `DISABLE_STATE_UPDATE`），验证后保存 q/k/v/a/b 并只对 accepted tokens 做 attention replay | O(m) token 的 attention state 重算 + 需保存每层 q/k/v/a/b | L 次 decode kernel launch (replay) | 每层 q/k/v/a/b (~24MB/layer for N=64, spec_len=5) | 低 |
| **B. 逐步快照 (Snapshot)** | 投机过程中每个 token 后保存中间 state 到 `intermediate_states_buffer`，验证后按 accepted_len 选取。vLLM 的 write-all-slots 做法本质上是方案 B 的一种工程实现（用 state pool 内的 slot 而非独立 buffer 存储 snapshot，见第 3 节） | 0 | 0 | L × k 份 state (L×k×HV×V×K) | 中 |
| **C. Lazy commit** | 投机时只计算 output 不更新 state，验证后再 batch update | 需要重新设计 kernel 分离 output 和 state update | 1 次 | 0 | 高 |

### 2.2 端到端性能分析

关键洞察：**验证阶段本身就是一次 prefill**。主模型必须对所有 k 个投机 token 做 forward pass——对 LA kernel 来说，这就是一次 mini-prefill。这次 prefill 内部天然会逐步计算 `S_0 → S_1 → ... → S_k`，这些中间 state 本来就在寄存器/shared memory 里产生了。

| | 方案 A (Checkpoint + Replay) | 方案 B (逐步快照) |
|---|---|---|
| 验证阶段 | prefill k tokens，**不保存**中间 state，**不更新**主 state pool | prefill k tokens，**顺手写出**中间 state |
| 确定 m 后 | 提取被接受的 q/k/v/a/b，**L 次 decode kernel launch** replay state update | **一次 gather**，按 m 索引取对应 state，逐层写回 state pool |
| 额外计算 | L × O(m) attention state 重算 | 0 |
| 额外 kernel launch | L 次 (decode kernel, per layer) | 0 |
| 额外显存 | per-layer q/k/v/a/b (~24MB/layer for N=64, spec_len=5, 见 2.3 节) | L × k × HV × V × K (fp32/bf16) |

> **关于方案 A 为什么需要 L 次 launch 而非 1 次**：replay 不是只跑 attention kernel 就行。完整流程是 `attention_decode(layer_0) → 残差+MLP+Norm → q/k/v/a/b 投影 → attention_decode(layer_1) → ...`。如果 replay 跑完整 model forward，那开销远不止"1 次 kernel launch"。但如果只跑 attention 部分的 state update，就需要在 verify 阶段**保存每层 spec_len 个 token 的 q/k/v/a/b**（约 24MB/layer for N=64, spec_len=5；**注意 verify 时还不知道 m，buffer 必须按 spec_len 分配**），然后在 replay 阶段逐层调用 decode kernel。这 L 次 launch 的开销约为 `L × ~8μs = ~200μs` (L=24)，加上 L 次 state read/write 的 GMEM traffic。

**方案 B 在计算效率上最优，但考虑层数后显存开销不可忽略**，原因：

1. **零额外计算**：中间 state 本来就在计算流中产生，只是多几条 store 指令
2. **零额外 kernel launch**：spec_len 只有 4-8 tokens，kernel launch latency 往往比计算本身还贵
3. **显存开销不可忽略**：由于 LA state 是 V×K 矩阵且无法跨层复用：
   - LA 的 state 是 **V×K 矩阵**（不是向量），单个 (seq, head, layer) 的 state = `128 × 128 × 4B = 64KB`
   - **每个 Attention 层都有独立的 state**，典型模型 L=24 层
   - **无法跨层复用 snapshot buffer**：rejection sampling 必须等所有 L 层 verify 完成后才能进行（需要完整模型 logits），因此每层的中间 state 必须保留到 rejection sampling 完成
   - 单个 seq 全部 head × 全部层 = `64KB × 128 heads × 24 layers = 192MB`
   - N=32 × spec_len=5 = `192MB × 32 × 5 = 30 GB`（fp32 全量 snapshot）
   - 对比 KV Cache：每个 token 的 KV Cache = `2 × K × bf16 = 2 × 128 × 2 = 512B/head/layer`，比 state 小 **128 倍**
   - KV Cache 可以轻松存几十个 spec token 的增量，而 LA state 的 snapshot 显存开销在多层模型下极其严峻
4. **Ragged 天然友好**：batch 内每个 seq 独立索引 `state_snapshots[seq_id, m_seq]`，不需要 padding 或重算

### 2.3 方案 A 的隐藏成本：保存 q/k/v/a/b 用于 Replay

方案 A (Checkpoint + Replay) 如果要避免完整的 model forward replay（即只重算 attention 的 state update），就**必须在 verify 阶段保存每层被接受 token 的 q/k/v/a/b**。这是因为：

- attention decode kernel 需要 q/k/v/a/b 作为输入
- 在 verify 的 forward pass 中，这些值已经被计算过（从输入投影得到）
- 如果不保存，要么重算（需要完整 model forward），要么直接丢弃（无法 replay）

**q/k/v/a/b 保存的显存估算**（N=64, **spec_len=5**, H=16, K=128, HV=128, V=128, bf16）：

> **关键**：verify 时还不知道哪些 token 会被接受，buffer 必须按 spec_len 分配（不能按 m 分配）。Rejection sampling 之后才能选取被接受的 token 子集用于 replay。

| 数据 | Shape | 大小 |
|------|-------|------|
| q | N × spec_len × H × K × sizeof(bf16) | 64 × 5 × 16 × 128 × 2 = ~1.3 MB |
| k | 同上 | ~1.3 MB |
| v | N × spec_len × HV × V × sizeof(bf16) | 64 × 5 × 128 × 128 × 2 = ~10.5 MB |
| a (KDA) | N × spec_len × HV × K × sizeof(bf16) | 64 × 5 × 128 × 128 × 2 = ~10.5 MB |
| b | N × spec_len × HV × sizeof(bf16) | 64 × 5 × 128 × 2 = ~80 KB |
| **单层合计** | | **~24 MB** |
| **L=24 层合计** | | **~570 MB** |

> 这 570MB 是方案 A **在已有 state pool 之外的额外显存**。虽然远小于方案 B 的 snapshot（~30GB for N=64, bf16, L=24），但设计时容易被忽略，需要纳入显存预算。

### 2.4 综合结论：采用混合策略

两种方案各有优势，不存在绝对最优：方案 B 零额外计算但显存压力大，方案 A 显存省但需要 replay compute + L 次 launch + 保存 spec_len × q/k/v/a/b。

**cuLA 的设计选择是两者都支持，通过 `recompute_state` 开关切换**：

| `recompute_state` | 策略 | verify kernel 行为 | verify 后 |
|---|---|---|---|
| `False` (默认) | Snapshot | 每步写 `intermediate_states_buffer` | host 侧 gather + 写回 state pool |
| `True` | Checkpoint + Replay | 保存每层 q/k/v/a/b，不写 intermediate state，不更新 state pool | L 次 decode kernel launch replay |

默认推荐 snapshot 模式（bf16 存储），在显存紧张时切换到 recompute 模式。详细推荐见 4.4 节。

---

## 3. 框架实现分析

### 3.1 vLLM 实现

vLLM 采用了与方案 B 相同的核心思想：**不撤销错误的 state，而是从未使用错误的 state**。通过多 block 分槽 + savepoint 机制，确保下一步的初始 state 始终来自最后一个被接受 token 的位置。

#### 3.1.1 每个 spec token 写入独立的 block/slot

在 `fused_sigmoid_gating_delta_rule_update_kernel`（`fused_sigmoid_gating.py:105-166`）中：

```python
# IS_SPEC_DECODING=True 时，初始状态从第 num_accepted_tokens-1 个 slot 读取
if IS_SPEC_DECODING:
    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
else:
    i_t = 0
state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t)
b_h += tl.load(p_h0, ...)  # 加载"最后一个被接受 token"位置的 state

# 每个 time step 的 state 写入独立的 slot
for i_t in range(0, T):
    b_h *= exp(b_g)
    b_h += v ⊗ k
    # 写入 ssm_state_indices[n, i_t] 对应的 block
    final_state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t)
    tl.store(ht + final_state_idx * ..., b_h)
```

每个 spec token 的 state 写入不同的 block，被拒绝 token 的 block 仅仅是"写了但没人用"。

#### 3.1.2 ssm_state_indices 的布局

`ssm_state_indices` 形状为 `[batch, num_spec+1]`，为每个 spec 位置分配一个 block ID。例如 `num_spec=2`：

```
┌────────┬─────────┬────────────────────────────┐
│  位置  │  block  │            内容            │
├────────┼─────────┼────────────────────────────┤
│ slot 0 │ block_A │ token_5 的 state（已确认） │
├────────┼─────────┼────────────────────────────┤
│ slot 1 │ block_B │ draft_1 的 state           │
├────────┼─────────┼────────────────────────────┤
│ slot 2 │ block_C │ draft_2 的 state           │
└────────┴─────────┴────────────────────────────┘
```

如果 draft_1 被接受但 draft_2 被拒绝，`num_accepted_tokens = 2`，下一步从 `slot[2-1] = slot[1] = block_B` 读取初始 state。block_C 被丢弃，下次覆盖。

#### 3.1.3 Savepoint 机制（mamba_cache_mode = "align"）

`preprocess_mamba()`（`mamba_utils.py:147-219`）在每步 forward 前执行 savepoint：

```python
# 运行态始终保存在倒数第 (1 + num_speculative_blocks) 个 block
curr_state_idx = num_blocks - 1 - num_speculative_blocks
```

以 `block_size=4, num_spec=2` 为例：

```
Block 0: [A, B, C, draft_1]    ← 数据 + 第一个 draft
Block 1: [draft_2, ...]         ← 第二个 draft + savepoint
Block 2: speculative block      ← spec state 写入
Block 3: speculative block      ← spec state 写入
```

Block 1 就是 savepoint block，里面保存的是"只有确认 token 的 state"。

复制时根据 `num_accepted_tokens` 选择源位置：
- **Conv state**（`get_conv_copy_spec`）：`offset = num_accepted_tokens - 1`，从 `state[src_block, offset:]` 复制，截断 draft token 的卷积窗口
- **Temporal state**（`get_temporal_copy_spec`）：`src_block_id = block_ids[cur_block_idx + num_accepted_tokens - 1]`，从最后一个被接受 token 对应的 block 读取

#### 3.1.4 完整流程

```
Step N:
  ┌─ preprocess_mamba()
  │    把上一步 savepoint 的 state 复制到新的 savepoint block
  │    源位置 = block_ids[prev_idx + num_accepted_tokens - 1]
  │    （跳过被拒绝 token 写入的 block）
  │
  ├─ Target model forward → rejection sampling → 确定哪些 token 被接受
  │
  ├─ Draft model forward (fused_sigmoid_gating kernel)
  │    初始 state 从 ssm_state_indices[n, num_accepted_tokens-1] 加载
  │    每个 draft token 写入独立 slot/block
  │
  └─ postprocess_mamba()
       如果 block 边界跨越，把 savepoint state 复制到新填满的 block
```

#### 3.1.5 总结

| 问题 | 答案 |
|------|------|
| KV state 是累加的吗？ | 是，`h += v⊗k`，不可逆 |
| 验证是串行的吗？ | 不是，rejection sampling 是并行的，但 kernel 内部按时间步顺序更新 state |
| 如何删除被拒绝 token 的影响？ | 不删除。每个 spec token 写独立 block，被拒绝 block 直接忽略，下一步从被接受 token 的 block 读取初始 state |
| Conv state 怎么处理？ | 从 `state[block, offset:]` 截断复制，`offset = num_accepted_tokens - 1` |
| Temporal state 怎么处理？ | 从 `block_ids[cur_idx + num_accepted_tokens - 1]` 读取正确的 block |

**核心思想：不是"撤销"错误的 state，而是"从未使用"错误的 state——通过多 block 分槽 + savepoint 机制，确保下一步的初始 state 始终来自最后一个被接受 token 的位置。**

### 3.2 SGLang 实现

#### 3.2.1 架构概览

SGLang 支持 GDN (Qwen3.5) 和 KDA (Kimi Linear) 两种 Linear Attention 变体的 MTP，使用 EAGLE/NEXTN 算法。

**关键发现**：SGLang 的 KDA 后端 **没有 target_verify 实现**，verify 时走标准 extend（chunk prefill）。这恰好是 cuLA 可以填补的空白。

#### 3.2.2 Verify Kernel 设计

SGLang 的 Triton verify kernel（`fused_sigmoid_gating_recurrent.py`）通过 constexpr 开关实现 decode/verify 复用：

| 参数 | Decode | Target Verify |
|---|---|---|
| `T` | 1 | `draft_token_num`（如 5） |
| `DISABLE_STATE_UPDATE` | `False` | `True`（不修改主 SSM pool） |
| `CACHE_INTERMEDIATE_STATES` | `False` | `True`（缓存每步 h 供树回滚） |
| `HAS_EAGLE_TREE_CUSTOM_ATTN_MASK` | `False` | `True`（topk>1 时启用树注意力） |

**核心循环**：

```python
for _ in range(0, T):
    # 树注意力：回滚到父节点状态
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        if step_idx != 0 and cache_idx >= 0:
            parent_step_idx = ...
            b_h = tl.load(intermediate_states_buffer + ...)  # 回滚！

    # 加载 QKV + 计算 gate/beta
    # Delta rule recurrent update
    # 缓存中间状态
    if CACHE_INTERMEDIATE_STATES:
        tl.store(intermediate_states_buffer + ..., b_h)

# 不写回主 state pool
if not DISABLE_STATE_UPDATE:
    tl.store(p_h0, b_h, ...)
```

#### 3.2.3 FlashInfer GDN MTP Kernel

SGLang 还有第三条 verify 路径：FlashInfer 专用 MTP kernel（`gdn_flashinfer.py`）。

```python
output_fi, _ = self._mtp_fn(
    q=query_mtp, k=key_mtp, v=value_mtp,
    initial_state=ssm_states,
    initial_state_indices=cache_indices,
    A_log=A_log.detach(), a=a_mtp, dt_bias=dt_bias.detach(), b=b_mtp,
    intermediate_states_buffer=intermediate_states_buffer,
    disable_state_update=True,
    use_qk_l2norm=True,
)
```

限制：仅支持 topk=1（线性序列），不支持树状 speculation。SM100+ 不支持。

#### 3.2.4 CuTeDSL Kernel 现状

SGLang 的 CuTeDSL kernel（`cutedsl_gdn.py`, `cutedsl_kda.py`）**只实现了 decode**，不支持 extend 和 target_verify：

```python
class CuteDSLGDNKernel(LinearAttnKernelBase):
    def extend(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLGDNKernel only supports decode")
    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLGDNKernel only supports decode")
```

因此 `GDNKernelDispatcher` 中：
- decode 可以走 CuTeDSL
- prefill 和 verify 始终走 Triton 或 FlashInfer

#### 3.2.5 完整调度矩阵

| | Decode | Extend/Prefill | Target Verify (MTP) |
|---|---|---|---|
| **GDN Triton** | `fused_sigmoid_gating_delta_rule_update` (T=1) | `chunk_gated_delta_rule` | `fused_sigmoid_gating_delta_rule_update` (T=draft_len, tree rollback) |
| **GDN FlashInfer** | `gated_delta_rule_decode_pretranspose` | `chunk_gated_delta_rule` | `gated_delta_rule_mtp` (专用 MTP kernel, topk=1 only) |
| **GDN CuTeDSL** | `cutedsl_fused_sigmoid_gating_delta_rule_update` | N/A | N/A |
| **KDA Triton** | `fused_sigmoid_gating_delta_rule_update` (is_kda=True) | `chunk_kda` | **不支持** |
| **KDA CuTeDSL** | `cutedsl_fused_sigmoid_gating_kda_update` | N/A | **不支持** |

**Verify kernel 选择逻辑**：
```python
# GDN: FlashInfer > Triton (CuTeDSL 不参与 verify)
# KDA: 标准 extend 路径（无专用 verify kernel）
```

#### 3.2.6 为什么 Verify 必须用 Recurrent 而非 Prefill

SGLang 的 verify 路径使用的是 **recurrent kernel**（`fused_sigmoid_gating_delta_rule_update` with T=draft_len），而非 prefill kernel（`chunk_gated_delta_rule`）。根本原因在于 speculative decoding 的树状结构：

1. **Prefill kernel** 使用 associative scan / chunk 并行，假设序列的 token 之间是**线性串联**的
2. **Verify 场景**下，draft tokens 构成的是一棵**树**（topk>1 时），而非线性序列。树中同一层级的兄弟节点共享同一个父 SSM 状态，需要：
   - 在处理完一个分支后**回滚**到父节点状态
   - 再从父节点状态出发处理下一个分支
   - 这就是 `retrieve_parent_token` 和 `intermediate_states_buffer` 的作用
3. 这种树状回滚在 parallel scan 中无法表达，只能用 **sequential recurrent** 逐步处理

这对 cuLA 设计的启示：cuLA 的 verify kernel 本质上也是 recurrent 语义（sequential state update），与 SGLang 的 Triton verify kernel 计算模式一致，只是用 CuTe DSL 实现了更高的 SMEM 利用率和 async pipeline。

#### 3.2.7 CuTeDSL GDN vs KDA Kernel 关键差异

SGLang 的 CuTeDSL kernel（仅 decode）揭示了 GDN 和 KDA 在底层实现上的关键差异，这些差异同样影响 verify kernel 设计：

| 特性 | GDN | KDA |
|---|---|---|
| gate `a` 形状 | `[N, 1, HV]` 标量 | `[N, 1, HV, K]` 向量 |
| `dt_bias` 形状 | `[HV]` 标量 | `[HV, K]` 向量 |
| 遗忘门 `g` | 标量，broadcast 到所有 K,V | 向量 `[K]`，需额外 `sG[TILE_K]` 共享内存 |
| `h *= g` | 标量广播 | 逐 K 维度乘 |
| SSM 状态布局 | K-major `(pool, HV, K, V)` | V-major `(pool, HV, V, K)` |
| 额外共享内存 | 无 sG | 需要额外 `sG[TILE_K]` (512B) |
| 额外计算 | 无 | 多一次逐元素乘法（`sData * sG` 和 `sData * sG * sK`） |

这些差异在 verify kernel 中同样存在，需要通过 `IS_KDA` constexpr 分支处理。但 verify kernel 的核心框架（SMEM state 保持、fused loop）对两者完全一致。

#### 3.2.8 Kimi Linear 混合架构

Kimi Linear（`kimi_linear.py`）采用混合层结构：部分层使用 KDA linear attention（`KimiDeltaAttention`），部分层使用标准 MLA attention（`KimiMLAAttention`）。层类型由 `KimiLinearConfig.linear_attn_config` 决定。

模型通过 `HybridLinearAttnBackend` 根据 `layer_id` 分发到不同后端：

```python
class HybridLinearAttnBackend(AttentionBackend):
    def forward_decode(self, layer, forward_batch, ...):
        if layer.layer_id in self.full_attn_layers:
            return self.full_attn_backend.forward_decode(...)  # MLA attention
        return self.linear_attn_backend.forward_decode(...)   # KDA linear attention
```

**KDA 后端的局限**：
- KDA 层**没有 `target_verify` 实现**，verify 时走标准 extend（chunk prefill）
- KDA 的 CuTeDSL kernel **只实现了 decode**，不支持 extend
- 这意味着 Kimi Linear 的 verify 路径有明确的优化空间，cuLA 的 verify kernel 可以直接填补这一空白

### 3.3 vLLM vs SGLang 关键差异

| 维度 | vLLM | SGLang |
|------|------|--------|
| Verify 时主 state | 写入独立 slot（"写但不读"） | **完全不碰**（`DISABLE_STATE_UPDATE=True`） |
| 中间状态存储 | `ssm_state_indices[batch, num_spec+1]` 2D block 索引 | `intermediate_states_buffer` 临时缓冲区 |
| 树状 speculation | 不支持（线性序列） | 支持 topk>1（`retrieve_parent_token`） |
| 被拒绝 slot 写入 | 仍然写入（浪费带宽） | 不写入主 pool，只写 intermediate buffer |
| State 写回时机 | verify kernel 内直接写 | verify 后由框架层决定写回哪个 state |
| Conv1d 处理 | `causal_conv1d_update` with `num_accepted_tokens` | `intermediate_conv_window` + 树回滚 |
| KDA verify | 支持（`IS_KDA=True`） | **不支持** |

### 3.4 对 cuLA 设计的启示

基于 vLLM 和 SGLang 的分析，cuLA 的设计应吸收以下关键洞察：

1. **Verify 时不写主 State Pool**：采用 SGLang 的 `DISABLE_STATE_UPDATE=True` 策略，而非 vLLM 的"写所有 slot"策略。vLLM 写入被拒绝 slot 浪费约 60% 的 state 写带宽，SGLang 的方案更干净：verify 只读初始 state + 写 intermediate buffer，不碰主 pool。verify 后的 state 更新由框架层决定。

2. **Verify kernel 必须是 recurrent 语义**：与 SGLang 的 Triton verify kernel 一致，cuLA 的 verify kernel 核心也是 sequential recurrent update。Prefill kernel 的 parallel scan 无法表达树状回滚。

3. **支持树状 Speculation**：参考 SGLang 的 `retrieve_parent_token` 机制，kernel 内部支持树状回滚。线性序列是树的特例（每个节点的父节点就是前一个节点）。接口层面预留 `retrieve_parent_token` 参数。

4. **KDA target_verify 是市场空白**：SGLang 和 vLLM 的 KDA CuTeDSL 都没有 target_verify，cuLA 实现后可直接填补。

5. **GDN vs KDA 的差异通过 constexpr 分支处理**：标量 gate vs 向量 gate，状态布局 K-major vs V-major。

6. **FlashInfer 的 MTP kernel 有硬件限制**（SM100+ 不支持，仅 topk=1），CuTe DSL 无此限制。

---

## 4. cuLA 设计决策

### 4.1 设计目标

基于 vLLM 和 SGLang 的分析，cuLA 的 MTP verify kernel 应该：

1. **高性能**：利用 CuTe DSL 的 SMEM 保持 + cp.async pipeline，比 Triton 实现更快
2. **灵活的显存策略**：支持 snapshot 和 checkpoint+replay 两种模式，通过 `recompute_state` 开关切换
3. **支持树状 speculation**：兼容 EAGLE topk>1 的树状 draft（通过 `retrieve_parent_token` 参数预留）
4. **KDA + GDN 双支持**：统一接口，内部通过 constexpr 切换
5. **与现有 decode kernel 解耦**：独立实现，避免 decode 路径的性能退化

### 4.2 vLLM 方案与 cuLA 方案的映射

| 方案 B 的概念 | vLLM 的实现 |
|---|---|
| `S_snapshots[t]` | `ssm_state_indices[n, i_t]` → 独立 block |
| `S_snapshots[seq, m]` 选取 | `ssm_state_indices[n, num_accepted-1]` 读取 |
| 验证前保存 init state | savepoint block（`curr_state_idx = num_blocks-1-num_spec`） |
| 一次 gather 选取 final state | `preprocess_mamba()` 中按 accepted 数复制 |

### 4.3 cuLA 的差异化优化机会

#### 4.3.1 去掉 block indirection 开销

vLLM 用 `ssm_state_indices` 做间接寻址是因为它要兼容 PagedAttention 的 block 管理体系。cuLA 作为独立 kernel 库，不需要这层抽象——直接用连续 buffer `state_snapshots[B, spec_len, HV, V, K]` 按 offset 索引即可，省掉一次 `tl.load(ssm_state_indices + ...)` 的间接寻址。

连续布局的优势：
- 地址计算是纯算术：`base + n * spec_len * HV * V * K + t * HV * V * K + hv * V * K + v * K + k`
- 无需额外的 index tensor load
- 对 GPU cache 友好（连续写入）

#### 4.3.2 CuTe kernel 的 async pipeline 利用

vLLM 的 `fused_sigmoid_gating_delta_rule_update_kernel` 是 Triton 写的，加一个 for loop + store 很自然。但 cuLA 的主力是 CuTe DSL kernel（`kda_decode.py`），它的 state 管理有 shared memory staging + cp.async pipeline 的优化。

当前 CuTe kernel 的 state 流：
```
GMEM → (cp.async) → SMEM → compute → SMEM → GMEM
```

加入 snapshot 后：
```
GMEM → (cp.async) → SMEM → compute → SMEM → GMEM (snapshot[t])
                                             ↓
                                      loop next token
                                             ↓
                                      SMEM → GMEM (snapshot[t+1])
```

关键优化：**snapshot 的 store 可以和下一个 token 的 compute overlap**——利用现有的 async pipeline，snapshot store 不需要额外等待。

具体来说，在当前 kernel 的 v_tile 循环末尾，state 写回 GMEM 是同步的。在 spec_decode kernel 中，我们可以：
1. 在 token t 的最后一个 v_tile 完成计算后，用 `cp.async` 将 sData 写到 snapshot[t] 的地址
2. 在 token t+1 开始时，先确保之前的 async store 完成（`cp.async.commit_group + wait`）
3. 然后加载 token t+1 的 q/k/v/a/b 并开始计算

这样 snapshot store 的大部分延迟被下一个 token 的输入加载掩盖了。

#### 4.3.3 Fused loop 省掉多次 kernel launch

spec_len=4-8 是一个尴尬的长度：
- 太短，不值得用 prefill kernel 的 chunk-parallel 策略
- 太长（相对单 token decode），单纯循环 launch 单 token kernel 又浪费 launch overhead

最佳策略是 **一个专门的 speculative decode kernel**，本质上是把当前单 token decode kernel 的主体包进一个 `for t in range(spec_len)` 循环，每步多一个 snapshot store。这比 launch `spec_len` 次单 token kernel 省掉 4-8 次 kernel launch。

#### 4.3.4 q/k/v 的连续布局优化

投机阶段，k 个 token 的 q/k/v 在内存中是连续的：
- q: `[B, spec_len, H, K]`
- k: `[B, spec_len, H, K]`
- v: `[B, spec_len, HV, V]`

在 kernel 循环中，每步只需要 `ptr += stride_t` 即可推进到下一个 token 的数据，无需重新计算基地址。这与 Triton 实现中的指针推进方式一致（`p_q += H * K` 等）。

#### 4.3.5 Shared Memory 复用

当前 CuTe decode kernel 的 SMEM 布局：
```
sData[K, V, NUM_STAGES]  — state tile (fp32)
sK[K]                     — key vector (fp32)
sQ[K]                     — query vector (fp32)
sG[K]                     — gate/decay (fp32)
sGK[K]                    — gate * key (fp32)
smem_o[V]                 — output reduction (fp32)
```

在 spec_decode kernel 中，循环内每步需要：
- 重新加载 q[t], k[t], a[t], b[t] 到 SMEM
- state (sData) 在循环间保持，不需要重新加载
- sG, sGK 每步重新计算（因为 a[t] 不同）

SMEM 总量不变，只是 q/k/a/b 的加载从 "一次" 变成 "循环中每步一次"。

### 4.4 核心设计决策

#### 决策 1：Verify 时不写主 State Pool

采用 SGLang 的 `DISABLE_STATE_UPDATE=True` 策略，而非 vLLM 的"写所有 slot"策略。

**理由**（详见 3.4 节）：
- vLLM 写入所有 slot（包括被拒绝的），浪费 `~60%` 的 state 写带宽
- SGLang 的方案更干净：verify 只读初始 state + 写 intermediate buffer，不碰主 pool
- verify 后的 state 更新由框架层决定（根据 accepted_len 选取 intermediate state 写回主 pool）

#### 决策 2：混合方案 — Snapshot 默认，Recompute 可选

基于第 5.2 节的显存分析（考虑层数 L=24~64），全量 fp32 snapshot 不可行，但 bf16 snapshot 在中小 batch 下可行。

**通过 `recompute_state` 开关支持两种策略**：

| `recompute_state` | 策略 | verify kernel 行为 | verify 后 |
|---|---|---|---|
| `False` (默认) | Snapshot (bf16) | 每步写 `intermediate_states_buffer`，不更新主 state pool | host 侧 gather + 逐层写回 state pool |
| `True` | Checkpoint + Replay | 保存每层 q/k/v/a/b，不写 intermediate state，不更新主 state pool | 逐层用 decode kernel replay accepted tokens |

**策略选择推荐**：

| 场景 | N (verify batch) | 推荐策略 | 额外显存 (L=24) | 额外计算 | 理由 |
|------|-----------------|---------|-------------|---------|------|
| 小 batch | N ≤ 16 | bf16 全量 snapshot | ~7.5 GB | 0 | 显存可控，零额外计算和 kernel launch |
| 中 batch | N = 16~32 | bf16 选择性 snapshot (每2步) | ~9 GB | 最多 replay 1 token | snapshot 减半，replay 开销极小 |
| 大 batch | N = 32~64 | bf16 全量 snapshot | ~30 GB | 0 | 如果显存允许，snapshot 最优；否则切换 recompute |
| | 或 | `recompute_state=True` | ~570MB (qkvab) + state pool | L 次 decode launch | 显存优先模式 |
| 超大 batch | N > 64 | `recompute_state=True` | ~1.1GB+ (qkvab) + state pool | L 次 decode launch | 显存有限时的唯一选择 |

默认推荐 snapshot 模式（bf16，全量或选择性），因为零额外计算、零额外 kernel launch。当显存紧张时，切换到 recompute 模式。

**recompute 模式的额外开销**：
- L 次 decode kernel launch (~200μs, L=24)
- 保存 verify 期间产生的 q/k/v/a/b（~24MB/layer, N=64, spec_len=5；按 spec_len 而非 m 分配）
- 端到端额外延迟约 500-800μs (L=24, N=64)

两种模式的 verify kernel 核心计算完全相同，都用 fused loop + SMEM state 保持，差异仅在 state 输出方式。

#### 决策 3：支持树状 Speculation

参考 SGLang 的 `retrieve_parent_token` 机制，kernel 内部支持树状回滚。

**树状 speculation 的数据结构**：

```
draft_tokens: [t0, t1, t2, t3, t4, t5]
                t0
               /  \
             t1    t2
             |     |
             t3    t4
                   |
                   t5

retrieve_parent_token: [-1, 0, 0, 1, 2, 4]  # 每个节点的父节点索引
```

当处理 t4 时，需要从 t2 的 state 出发（而非 t3），所以需要回滚到 t2 的 intermediate state。

**cuLA 的实现**：在 CuTe kernel 内部，树状回滚需要从 `intermediate_states_buffer` 加载父节点的 state 到 SMEM。这比线性序列多一次 GMEM→SMEM 的 load，但只在分支点发生。

**树状场景的显存影响**：`intermediate_states_buffer` 的维度从 `spec_len` 变为树的总节点数 `total_nodes`。以 EAGLE topk=5, depth=3 为例，`total_nodes = 1 + 5 + 25 = 31`（是线性 spec_len=5 的 6 倍），snapshot 显存需求急剧增加（详见 5.2.4 节）。因此树状 speculation 场景下，`recompute_state=True` 几乎是强制选择。需注意 recompute 模式下树状场景的 qkvab 显存也按 `total_nodes` 膨胀（~6×），但绝对值仍可控（N=32, L=24, total_nodes=31 时约 ~1.8GB，远小于 snapshot 模式的 ~93GB）。

#### 决策 4：Kernel 接口设计

统一命名为 `kda_verify` / `gdn_verify`，接口设计详见 6.9 节。

### 4.5 与 vLLM/SGLang 的关键差异

| 维度 | vLLM | SGLang | cuLA (本方案) |
|------|------|--------|--------------|
| Kernel 框架 | Triton | Triton / FlashInfer / CuTeDSL | CuTe DSL (Cutlass) |
| State 管理 | PagedAttention block + ssm_state_indices 间接寻址 | `intermediate_states_buffer` 临时缓冲区 | 连续 tensor + 直接 offset 索引 |
| Snapshot 存储 | 每个 spec token 写入独立 PagedAttention block | `intermediate_states_buffer` + `CACHE_INTERMEDIATE_STATES` | 写入连续 `intermediate_states_buffer` (snapshot 模式) 或保存 q/k/v/a/b (recompute 模式) |
| Verify 时主 state | 写入所有 slot（含被拒绝的，浪费带宽） | 完全不碰（`DISABLE_STATE_UPDATE=True`） | 完全不碰（同 SGLang） |
| State 精度 | fp32 | fp32 | fp32 (compute) + 可选 bf16 (snapshot) |
| SMEM 复用 | Triton 自动管理 | Triton 自动管理 | 显式管理，state 跨 token 保持 |
| Async overlap | Triton 不支持显式 cp.async | Triton 不支持显式 cp.async | cp.async snapshot store 与输入加载 overlap |
| Kernel launch | 1 次 (Triton for loop) | 1 次 (Triton for loop) | 1 次 (CuTe for loop, 编译时常量展开) |
| 显存优化 | 无（依赖 PagedAttention 的 block 复用） | 无 | 混合方案：bf16 snapshot / 选择性 snapshot / recompute_state |
| 树状 speculation | 不支持 | 支持（`retrieve_parent_token`） | 支持（`retrieve_parent_token`，接口预留） |
| 验证后 state 选取 | preprocess_mamba() 中 block 复制 | 框架层 gather | snapshot: host 侧 gather; recompute: 逐层 decode kernel replay |
| KDA verify | 支持（`IS_KDA=True`） | **不支持** | 支持 |

---

## 5. Kernel 架构与显存

### 5.1 整体架构

```
                    ┌─────────────────────────────────┐
                    │      cuLA public API             │
                    │  kda_verify(                     │
                    │    q, k, v, a, b,                │
                    │    A_log, dt_bias,               │
                    │    initial_state_source,         │
                    │    intermediate_states_buffer,    │
                    │    spec_len,                     │
                    │  ) → (output, state_snapshots)   │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
    ┌─────────▼──────────┐          ┌───────────▼──────────┐
    │ CuTe Verify Kernel │          │  State Select (host) │
    │ (fused loop +      │          │  state_final[n] =    │
    │  per-step snapshot) │          │   snapshots[n, m[n]] │
    └─────────────────────┘          └──────────────────────┘
```

### 5.2 显存分析

**关键认知**：LA 的 recurrent state 是 **V×K 矩阵**（不是长度为 K 的向量），这是显存问题的根源。

**关于 N 的定义**：本文中 N 指 **同时参与投机验证的 sequence 数量**，而非 serving 的总 batch size。实际 serving 中，总 batch 内的 sequence 处于不同阶段：
- 部分 seq 正在做投机生成（draft phase）——这些 seq 不需要 snapshot
- 部分 seq 正在做验证（verify phase）——**这些才是 snapshot 的 N**
- 部分 seq 可能只是普通单 token decode——不需要 snapshot

因此 snapshot 显存计算中的 N 通常 **远小于** 总 batch size。比如总 batch=512，但同一时刻可能只有 32~64 个 seq 进入验证阶段，那 N=32~64。但即使如此，多层模型下显存仍然很严峻（见下方计算）。

#### 5.2.1 逐层拆解

**⚠️ 关键 1：每个 Attention 层都有独立的 recurrent state，层数 L 必须纳入计算。**

**⚠️ 关键 2：没有跨层复用 snapshot 的可能。** rejection sampling 必须在所有 L 层 verify 完成后才能进行（需要完整模型的最后一层 logits 才能判定哪些 token 被接受）。因此在 rejection sampling 之前，无法确定每个 seq 的 `accepted_len`，也就无法判断应该保留哪个中间 state。每层的 `intermediate_states_buffer[layer]` 必须完整保留直到 rejection sampling 完成。

cuLA 的 kernel 是 layer-agnostic 的——每次调用处理一层的 state。但模型有 L 层（典型 L=24~64），每层都有独立的 state tensor。snapshot 需要为每层都存一份。

```
单个 (seq, head, layer) 的 state:
  shape = (V, K) = (128, 128)
  元素数 = 128 × 128 = 16,384
  fp32 大小 = 16,384 × 4B = 64 KB
```

| 层级 | 计算 | 大小 |
|------|------|------|
| 1 个 (seq, head, layer) | V × K × 4B = 128 × 128 × 4 | **64 KB** |
| 1 个 seq, 1 层, 所有 head | × HV = × 128 | **8 MB** |
| 1 个 seq, **L 层**, 所有 head | × L=24 | **192 MB** |
| N=64, 1 份 snapshot | × 64 | **12 GB** |
| N=64, spec_len=5 份 snapshot | × 5 | **60 GB** |

不考虑层数的估算会低估 24~64 倍！

#### 5.2.2 与 KV Cache 的对比

| | LA State Snapshot | KV Cache 增量 |
|---|---|---|
| 每个 (seq, head, layer) | V × K × sizeof(fp32) = 128 × 128 × 4 = **64 KB** | 2 × K × sizeof(bf16) = 2 × 128 × 2 = **512 B** |
| 倍数关系 | **128×** | 基准 |
| spec_len=5, N=64, L=24 | **60 GB** | ~1.2 GB |

KV Cache 可以轻松存几十个 spec token 的增量（每个 token 只增加 512B/head/layer），而 LA state 的 snapshot 每份 64KB/head/layer，差了两个数量级。加上层数因素后，LA 的显存问题比单层估算更加严峻。

#### 5.2.3 不同 batch size 和层数下的显存需求

**单层 (L=1) 的 snapshot 显存**（仅作参考）：

| N (batch) | spec_len | fp32 全量 snapshot | bf16 全量 snapshot | fp32 选择性 (每2步) |
|-----------|----------|-------------------|-------------------|-------------------|
| 8 | 5 | 320 MB | 160 MB | 192 MB |
| 32 | 5 | 1.25 GB | 640 MB | 768 MB |
| 64 | 5 | 2.5 GB | 1.25 GB | 1.5 GB |

**多层 (L=24) 的 snapshot 显存**（实际部署必须看的数字）：

| N (batch) | spec_len | fp32 全量 snapshot | bf16 全量 snapshot | bf16 选择性 (每2步) | bf16 qkvab (方案A) |
|-----------|----------|-------------------|-------------------|-------------------|-------------------|
| 8 | 5 | 7.5 GB | 3.84 GB | 2.3 GB | ~71 MB |
| 32 | 5 | 30 GB | 15 GB | 9 GB | ~285 MB |
| 64 | 5 | 60 GB | 30 GB | 18 GB | ~570 MB |

> **显存估算总结**：
> - fp32 全量 snapshot 在多层模型下显存需求极高（N=32 即 30GB），bf16 可减半
> - 选择性 snapshot (每2步, bf16) 在 N ≤ 32 时约 9GB
> - Checkpoint + Replay 方案（`recompute_state=True`）额外显存 ~570MB (N=64, spec_len=5, bf16)，显存最优
> - 具体策略选择见 4.4 节混合方案推荐

#### 5.2.4 树状 Speculation 的显存影响

树状 speculation 场景下，`intermediate_states_buffer` 的维度不是 `spec_len` 而是 **树的总节点数 `total_nodes`**。以 EAGLE topk=5, tree_depth=3 为例，`total_nodes = 1 + 5 + 25 = 31`（是线性 spec_len=5 的 6 倍），导致 snapshot 显存需求急剧增加：

| 场景 | buffer nodes | L=24 snapshot (bf16, N=32) |
|------|-------------|---------------------------|
| 线性 spec_len=5 | 5 | ~15 GB |
| 树状 topk=5, depth=3 | 31 | ~93 GB |

因此树状 speculation 场景下，`recompute_state=True` 几乎是强制选择。接口层面预留 `retrieve_parent_token` 参数，`intermediate_states_buffer` 的维度按 `total_nodes` 分配（而非 `spec_len`）。

### 5.3 显存优化方案一：State Pool Slot 复用

**关键观察**：验证后我们只需要 `state_snapshots[n, accepted_lens[n]-1]`，即每个 seq 只需要一个 snapshot。但 `accepted_lens` 在 kernel 执行时未知。

**优化：Snapshot 写到 state pool 的不同 slot，而非独立 tensor**

利用 vLLM 的 slot 思想，但去掉 block indirection：

```
initial_state_source: (pool_size, HV, V, K)
  - pool_size = N * (1 + spec_len)
  - slot 0..N-1: 初始 state（已确认的）
  - slot N..N*(1+spec_len)-1: snapshot slots

initial_state_indices: (N,) → 指向初始 state slot
snapshot_indices: (N, spec_len) → 指向每个 seq 的 spec_len 个 snapshot slot
```

验证后选取：
```python
# 为每个 seq 构造最终 state 的 index
# final_idx[n] = snapshot_indices[n, accepted_lens[n] - 1]
final_idx = snapshot_indices.gather(1, (accepted_lens - 1).unsqueeze(1)).squeeze(1)
initial_state_indices.copy_(final_idx)  # 下一步 decode 从这个 slot 读取
```

**In-place 优化**：state pool 中每个 seq 的初始 state 在 spec_decode 开始后就不需要了（因为我们会从 snapshot 中选一个作为新的初始 state）。所以可以 **覆盖初始 state slot** 作为第一个 snapshot 的位置，pool_size 减为 `N * spec_len`。

### 5.4 显存优化方案二：bf16 Snapshot

**核心洞察**：kernel 内部计算用 fp32 保持精度，但 snapshot 存储可以用 bf16 降低一半显存。

```
显存: N * spec_len * HV * V * K * 2B  (bf16)
N=64, spec_len=5, HV=128, V=K=128: 64 × 5 × 128 × 128 × 128 × 2B ≈ 1.25 GB
```

**精度分析**：snapshot 写入时 fp32→bf16 转换，读取时 bf16→fp32。被拒绝的 state 本来就不会使用，只有最终被选中的 state 需要高精度。而选中的 state 会在下一步 decode kernel 中以 fp32 加载和计算，bf16 引入的量化误差会被后续的 decay 逐步稀释，对最终输出影响极小。

**Kernel 实现要点**：snapshot store 时用 `cutlass.BFloat16(sData[...])` 替代 `cutlass.Float32(sData[...])`，其余逻辑不变。

### 5.5 显存优化方案三：选择性 Snapshot

**进一步观察**：在实际 speculative decoding 中，接受长度 `m` 的分布不是均匀的。通常 `m` 的分布集中在 1~3（大部分 spec token 被拒绝）或 `spec_len`（全部接受）。

**方案：只存储关键 checkpoint，而非每步都存**

比如 spec_len=5 时，只存储 t=0（初始 state，已有）和 t=2, t=4 的 snapshot：
- 如果 m=1 或 m=2：使用 t=0 的 state + replay 1~2 tokens
- 如果 m=3 或 m=4：使用 t=2 的 snapshot + replay 1~2 tokens
- 如果 m=5：使用 t=4 的 snapshot

这样最多只需 replay 2 个 tokens，但 snapshot 存储量减半。

**Trade-off**：
- 显存：减少到 `N * ceil(spec_len/2) * HV * V * K * sizeof(dtype)`
- 计算：最坏情况需要 replay `checkpoint_interval - 1` 个 tokens（1 次 kernel launch）
- 这是一个显存和计算之间的平衡

**为什么 replay 开销可接受**：单 token decode kernel 的执行时间约 10~50μs（取决于 batch size），而 20GB 显存的分配和搬运成本远高于此。replay 1~2 个 token 的延迟远小于省掉的显存带来的系统级收益（更大的 batch size → 更高吞吐）。

---

## 6. CuTe Kernel 实现

### 6.1 Kernel 核心循环

```python
@cute.kernel
def kda_verify_kernel(
    ...,
    intermediate_states_buffer,  # (N, max_steps, HV, V, K) 或 None
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    has_tree_attn: cutlass.Constexpr[bool],  # 是否启用树状回滚
    retrieve_parent_token,       # (seq_len,) 或 None
    max_steps: cutlass.Constexpr[int],  # 编译时常量
):
    """MTP Verify kernel: fused loop over max_steps tokens with per-step snapshot and tree rollback."""

    # ... CTA 映射与现有 decode kernel 相同 ...

    # 加载初始 state 到 sData
    for k_iter in range(NUM_K_ITERS):
        sData[...] = h0_source[(pool_idx, i_hv, ...)]

    for t in range(max_steps):
        # --- 树状回滚：从 intermediate buffer 加载父节点 state ---
        if has_tree_attn and t > 0:
            parent_t = retrieve_parent_token[t]
            if parent_t != t - 1:
                # 非连续：需要从 intermediate buffer 重新加载
                for k_iter in range(NUM_K_ITERS):
                    sData[...] = intermediate_states_buffer[(i_n, parent_t, i_hv, ...)]

        # --- 加载当前 token 的 q, k, v, a, b ---
        # (与现有 decode kernel 相同，只是索引多了 t 维度)

        # --- 计算 gate 和 beta ---
        # (与现有 decode kernel 相同)

        # --- Delta rule update ---
        # sum_hk = sum(sData * sGK)
        # v_new = (r_v - sum_hk) * r_beta
        # h_new = sData * sG + sK * v_new
        # sData = h_new
        # output = sum(h_new * sQ)

        # --- 写 output ---
        o[(i_n, t, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

        # --- 缓存中间 state ---
        if cache_intermediate_states:
            for k_iter in range(NUM_K_ITERS):
                intermediate_states_buffer[(i_n, t, i_hv, ...)] = sData[...]

    # --- 写回主 state pool ---
    if not disable_state_update:
        for k_iter in range(NUM_K_ITERS):
            h0_source[(pool_idx, i_hv, ...)] = sData[...]
```

### 6.2 与 Decode Kernel 的关键差异

| 方面 | Decode Kernel | Verify Kernel |
|------|--------------|---------------|
| Token 数 | T=1 | T=1~8 (编译时常量) |
| State 加载 | 每次从 GMEM 加载 | 首次加载，循环内保持 SMEM |
| State 写回 | 每次写回主 pool | 可选：不写 / 写 intermediate / 写主 pool |
| 树状回滚 | 不需要 | 需要 `retrieve_parent_token` 支持 |
| 输出 | (N, 1, HV, V) | (N, T, HV, V) |
| q/k/v/a/b 布局 | (N, 1, H, K) | (1, N*T, H, K) varlen 或 (N, T, H, K) dense |
| Conv1d | 单 token update | 多 token + intermediate conv window |

### 6.3 Small-Batch Verify Kernel 详细设计

以 small batch kernel 为例（N < 1024），展示关键改造点：

```python
@cute.kernel
def kda_verify_small_batch(
    tiled_copy_load, h0_source, smem_layout_staged,
    num_v_tiles, num_blocks_per_state_small,
    q, k, v, a, b, A_log, dt_bias, o,
    h0_indices, intermediate_states_buffer,  # snapshot 输出
    softplus_beta, softplus_threshold, scale,
    H, HV, max_steps,              # max_steps 作为编译时常量
    use_qk_l2norm, state_layout_is_kv,
    precomputed_decay_beta, dense_small_hv_parallel,
    disable_state_update, cache_intermediate_states,
    has_tree_attn, retrieve_parent_token,
):
    # ... CTA 映射与现有 kernel 相同 ...

    pool_idx = h0_indices[i_n]

    if pool_idx >= 0:
        # ... SMEM 分配与现有 kernel 相同 ...
        # sData, sK, sQ, sG, sGK, smem_o

        # ========== max_steps 循环 ==========
        for t in range(max_steps):

            # --- 树状回滚 ---
            if has_tree_attn and t > 0:
                parent_t = retrieve_parent_token[t]
                if parent_t != t - 1:
                    # 从 intermediate buffer 加载父节点 state 到 sData
                    for k_iter in range(NUM_K_ITERS_SMALL):
                        # ... 加载 intermediate_states_buffer[(i_n, parent_t, i_hv, ...)] 到 sData ...

            # --- 1. 加载当前 token 的 q, k ---
            if tidx < TILE_K:
                sK[tidx] = cutlass.Float32(k[i_n, t, i_h, tidx])  # 注意: dim 1 从 0 变为 t
                sQ[tidx] = cutlass.Float32(q[i_n, t, i_h, tidx])

            # --- 2. 计算 gate 和 beta ---
            # (与现有 kernel 相同，只是 a/b 的索引加上 t 维度)
            if precomputed_decay_beta:
                if tidx < TILE_K:
                    sG[tidx] = cutlass.Float32(a[i_n, t, i_hv, tidx])
            else:
                # ... softplus + exp 计算 gate ...
                r_a_k = cutlass.Float32(a[i_n, t, i_hv, tidx])
                # ...

            r_beta = 0.0
            if in_warp_tid == 0:
                r_b = cutlass.Float32(b[i_n, t, i_hv])
                # ... sigmoid ...

            cute.arch.barrier()

            # --- 3. QK L2 normalization (如果启用) ---
            # (与现有 kernel 相同)

            # --- 4. 计算 sGK = sG * sK ---
            if tidx < TILE_K:
                sGK[tidx] = sG[tidx] * sK[tidx]
            cute.arch.barrier()

            # --- 5. 加载 v ---
            v_global = v_tile * TILE_V_SMALL + v_idx
            r_v = 0.0
            if v_global < v.shape[3]:
                r_v = cutlass.Float32(v[i_n, t, i_hv, v_global])  # 注意: t 维度

            # --- 6. V-tile 循环: state update + output ---
            # 注意: t=0 时需要从 GMEM 加载初始 state
            #       t>0 时 sData 已经在 SMEM 中，不需要重新加载
            for v_tile_offset in range(num_v_tiles_per_block):
                stage = v_tile_offset % NUM_STAGES
                v_tile = start_v_tile + v_tile_offset
                v_global_base = v_tile * TILE_V_SMALL

                if t == 0:
                    # 首次: 从 h0_source 加载初始 state
                    for k_iter in range(NUM_K_ITERS_SMALL):
                        # ... 加载 h0_source 到 sData ...
                # t > 0: sData 已在 SMEM 中，无需重新加载

                # --- Delta rule update (与现有 kernel 相同) ---
                # sum_hk = sum(sData * sGK)
                # v_new = (r_v - sum_hk) * r_beta
                # h_new = sData * sG + sK * v_new
                # sData = h_new
                # output = sum(h_new * sQ)

                # --- 写 output ---
                if k_local == 0 and v_global < v.shape[3]:
                    o[(i_n, t, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

                # --- 写 snapshot (如果启用) ---
                if cache_intermediate_states:
                    for k_iter in cutlass.range(NUM_K_ITERS_SMALL, unroll=2):
                        # ... 计算 k_write, v_write, v_global_write ...
                        if v_global_write < v.shape[3]:
                            if state_layout_is_kv:
                                intermediate_states_buffer[(i_n, t, i_hv, k_write, v_global_write)] = \
                                    cutlass.Float32(sData[(k_write, v_write, stage)])
                            else:
                                intermediate_states_buffer[(i_n, t, i_hv, v_global_write, k_write)] = \
                                    cutlass.Float32(sData[(k_write, v_write, stage)])

                # 注意: 最后一个 token (t == max_steps - 1) 的 snapshot
                # 同时也写回 h0_source 作为最终 state (如果 disable_state_update=False)
                if not disable_state_update and t == max_steps - 1:
                    for k_iter in cutlass.range(NUM_K_ITERS_SMALL, unroll=2):
                        # ... 写回 h0_source (与现有 kernel 相同) ...

                cute.arch.barrier()
        # ========== max_steps 循环结束 ==========
```

### 6.4 SMEM 布局

Verify kernel 不需要额外 SMEM——sData 在循环间保持，q/k/a/b 每步重新加载到同一个 buffer。

```
sData:  K × V × NUM_STAGES × 4B = 128 × 16 × 1 × 4 = 8KB (small batch, TILE_V_SMALL=16)
sK:     K × 4B = 512B
sQ:     K × 4B = 512B
sG:     K × 4B = 512B  (KDA only)
sGK:    K × 4B = 512B  (KDA only)
smem_o: V × 4B = 64B
───────────────────
Total:  ~10KB (small batch, KDA)
```

对于 large batch kernel：
```
sData:  K × V × NUM_STAGES × 4B = 128 × 32 × 1 × 4 = 16KB
Total:  ~18KB
```

也在 SMEM 预算内（SM90 每个 SM 有 227KB SMEM）。

### 6.5 性能优化

#### 6.5.1 State 在 SMEM 中跨 token 保持

**这是最重要的优化**：现有 decode kernel 每次调用都要从 GMEM 加载 state 到 SMEM、计算完再写回。在 verify kernel 中，state (sData) 在整个 `max_steps` 循环中保持在 SMEM，省掉了 `2 * (max_steps - 1)` 次 GMEM round-trip。

以 K=128, V=128, fp32, HV=128, N=32 为例，一次 state 的 GMEM 读写：
- 读: 128 × 128 × 4B = 64KB
- 写: 64KB
- 总计: 128KB per state per head per v_tile
- **全部 head 总计**(per layer): 128 heads × 128KB = 16MB read + 16MB write

对于 max_steps=5，省掉 4 次读 + 4 次写 = 512KB 的 GMEM traffic per head per v_tile。

> **L2 Cache 效应**：H100 L2 cache 有 50MB。如果 N 较小（state 总量在 L2 以内），省掉的 GMEM traffic 实际上只是 L2 hits vs misses 的差异（L2 hit latency ~200 cycles vs HBM ~800 cycles）。对于大 N 场景，state 超出 L2 容量时，省掉的才是真正的 HBM bandwidth。实际收益需要根据 N 和 L2 容量评估。

#### 6.5.2 Snapshot Store 与计算 Overlap

Snapshot store 可以用 `cp.async` 实现，与下一个 token 的输入加载 overlap：

```
Timeline (per v_tile):

Token 0:
  [load state] → [compute] → [store output] → [cp.async: store snapshot[0]] ─┐
                                                                               │ overlap
Token 1:                                                                       │
  [load q[1],k[1],a[1],b[1]] ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
  → [compute] → [store output] → [cp.async: store snapshot[1]] ─┐
                                                                  │ overlap
Token 2:                                                          │
  [load q[2],k[2],a[2],b[2]] ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
  → ...
```

这要求 snapshot store 和输入 load 使用不同的 GMEM 地址，天然满足。

**cp.async 同步细节**：

snapshot store 的异步流程需要精确的同步控制：

```python
# Token t 的 snapshot store（在 v_tile 循环末尾）
cp.async.store(sData, intermediate_states_buffer[n, t, hv, ...])

# Token t+1 开始时
cp.async.commit_group()      # 将 snapshot store 提交到一个 group
# ... 加载 q[t+1], k[t+1], a[t+1], b[t+1] 到 SMEM（操作不同 buffer，无冲突）...
cp.async.wait_group(0)       # 确保该 group 内所有异步操作完成
                              # 然后再进入 sData 的 compute（读 sData）
```

**关键技术约束**：snapshot store 是 `sData(SMEM) → GMEM`，而下一个 token 的 compute 是 `sData(SMEM) → SMEM`（delta rule update 会读 sData 并写回同一个 SMEM buffer）。虽然 snapshot store 是异步 copy（GMEM 写入不影响 SMEM），但**下一个 token 的 sData read 和上一个 token 的 snapshot store 不冲突**——因为 snapshot store 是读 sData、写 GMEM，不会修改 sData。

**真正的同步点在**：当前 token 的 v_tile 循环结束后，是否已经有新的 async copy 在修改 sData（如加载新 state）。对于 t > 0，sData 保持在 SMEM 中无需重新加载，所以不存在这个冲突。**唯一的同步需求是**：确保在循环结束后、写回最终 state 之前，所有 pending 的 snapshot store 已经完成。

> 因此，snapshot store 与下一个 token 的 q/k/a/b 加载可以完全 overlap，不需要额外的 wait_group 插入到循环中间。只需要在 max_steps 循环结束后、向 `h0_source` 写回最终 state 前，做一次全局 `cp.async.wait_all()`。

#### 6.5.3 编译时常量 max_steps

`max_steps` 应该作为 `cutlass.Constexpr[int]` 传入，这样：
- 循环可以被编译器完全展开
- `t == 0` 和 `t == max_steps - 1` 的分支可以被优化掉
- 避免了动态循环的 overhead

典型值 `max_steps ∈ {1, 2, 4, 5, 8}`，可以为每个值编译一个 kernel 变体（与现有 compile cache 机制兼容）。

#### 6.5.4 大批量 Kernel 的 CTA 映射

现有 large batch kernel 的 CTA 映射：`1 CTA per (seq, head, v_tile)`。

在 verify kernel 中，有两种选择：

**选择 A: 1 CTA 处理所有 max_steps tokens**
- 优点：state 在 SMEM 中跨 token 保持，省 GMEM traffic
- 缺点：CTA 执行时间变长，可能影响 occupancy
- 适用于 max_steps 较小（≤8）的场景

**选择 B: max_steps 个 CTA 串行处理**
- 优点：occupancy 不受影响
- 缺点：需要 CTA 间同步或 GMEM 中转 state
- 复杂度高，不推荐

**推荐选择 A**，因为 max_steps=4~8 时 CTA 执行时间增加有限（每个 token 的计算量相同，只是循环了多次），而省掉的 GMEM traffic 收益显著。

#### 6.5.5 Verify Kernel vs T 次 Decode Kernel 性能对比

| 指标 | T 次 Decode Kernel | Verify Kernel (fused) |
|------|-------------------|----------------------|
| Kernel launch | T × ~8μs | 1 × ~8μs |
| State 加载 | T × GMEM read | 1 × GMEM read |
| State 写回 | T × GMEM write | 0 (DISABLE_STATE_UPDATE) |
| Intermediate state 写入 | 0 | T × GMEM write (snapshot 模式) / 0 (recompute 模式) |
| 总 GMEM traffic (T=5) | 5 × (64KB read + 64KB write) = 640KB | snapshot: 1 × 64KB read + 5 × 64KB write = 384KB; recompute: 1 × 64KB read |

> 以上为单 head 的 traffic。整个层的 traffic 需乘以 HV (128) 和 v_tile 数。

**Fused verify kernel 节省 (单层, snapshot 模式)**：
- 4 次 kernel launch = ~32μs
- 4 次 state read = 4 × 64KB = 256KB GMEM traffic per head
- 5 次 state write (snapshot) vs 5 次 state write (decode) = 相同

#### 6.5.6 recompute_state=True 的额外开销 (逐层, 完整分析)

**recompute 模式需要 L 次 replay launch，因为 replay 是逐层调用 decode kernel**（这里假设 replay kernel 一次处理一层全部 m 个 token；若用单 token decode kernel，launch 数应乘以 m）：

| accepted_len (m) | 每层 replay tokens | 总 kernel launch | L 层 GMEM traffic (per head) | 总延迟 (L=24) |
|-------------|-------------------|-------------------|---------------------------|--------------|
| 1 | 1 | L × ~8μs = ~192μs | L × ~130KB = ~3.1MB | ~250μs |
| 2 | 2 | L × ~8μs = ~192μs | L × ~160KB = ~3.8MB | ~280μs |
| 3 | 3 | L × ~8μs = ~192μs | L × ~200KB = ~4.7MB | ~310μs |
| 5 | 5 | L × ~8μs = ~192μs | L × ~250KB = ~6MB | ~370μs |

**Host 侧额外开销**（recompute 路径独有）：
- 从 `saved_qkvab` 提取被接受 token 的 q/k/v/a/b：~10-50μs (取决于 N)
- 准备 decode kernel 参数（per layer）：~L × 5μs = ~120μs
- cuLaunchKernel driver overhead：~L × 8μs = ~192μs (包含 CPU-side kernel launch time)
- **Host 总开销**：~300-400μs

> **注意**：recompute 路径的完整延迟分解：(1) replay 需要逐层调用 decode kernel，共 L 次 launch (2) host 侧有 q/k/v/a/b 准备开销和 driver overhead。综合下来，端到端额外延迟约 **500-800μs** (L=24, N=64)。

**Snapshot vs Recompute 对比**（accepted_len=3, spec_len=5, L=24, N=64）：

| 指标 | Snapshot (bf16) | Recompute |
|------|----------------|-----------|
| 额外显存 | ~30 GB | ~570MB (qkvab) |
| 额外计算 | 0 | L × accepted_len = 72 次 attention state update |
| 额外 kernel launch | 0 | L = 24 次 |
| 端到端额外延迟 | ~0 | ~500-800μs |
| 适用 N | N ≤ 32 (显存允许时最优) | N > 32 (显存优先) |

### 6.6 树状 Speculation 支持

#### 6.6.1 树状回滚机制

参考 SGLang 的 `retrieve_parent_token` 机制，kernel 内部支持树状回滚。

当处理树中非连续节点时，需要从 `intermediate_states_buffer` 加载父节点的 state 到 SMEM。这比线性序列多一次 GMEM→SMEM 的 load，但只在分支点发生。

**关键优化**：如果 `retrieve_parent_token[t] == t - 1`（线性序列），sData 已经在 SMEM 中，无需额外 load。只有分支点才需要回滚。

典型 EAGLE topk=5, tree_depth=3 的树结构中，约 60% 的节点是线性连续的（不需要回滚），40% 的节点在分支点（需要回滚）。这个比例基于 depth=3 时路径上每个非叶节点只有一个子节点在"当前路径"上（实际取决于具体树结构和采样策略）。

#### 6.6.2 树状场景的 Buffer 管理

树状 speculation 下 `intermediate_states_buffer` 的维度按 `total_nodes` 分配（而非 `spec_len`）：

```python
# 线性模式: (N, spec_len, HV, V, K)
# 树状模式: (N, total_nodes, HV, V, K)  — total_nodes 可能远大于 spec_len
intermediate_states_buffer: torch.Tensor  # shape 取决于 speculation 模式
retrieve_parent_token: torch.Tensor | None  # (total_nodes,) 父节点索引
```

由于树状场景下 buffer 显存需求急剧增加（见 5.2.4 节），`recompute_state=True` 在树状场景下几乎是强制选择。

### 6.7 Verify 后的 State 更新流程

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Verify Kernel (CuTe, 逐层调用)                            │
│    - 输入: q, k, v, a, b, initial_state (per layer)          │
│    - 输出: output logits                                      │
│    - snapshot 模式: 写出 intermediate_states_buffer[layer]    │
│    - recompute 模式: 保存 per-layer q/k/v/a/b 到 saved_qkvab │
│    - 均不写主 state pool (DISABLE_STATE_UPDATE=True)          │
├──────────────────────────────────────────────────────────────┤
│ 2. Rejection Sampling (GPU, framework 层)                     │
│    - 输入: output logits (最后一层), draft logits            │
│    - 输出: accepted_lens (每个 seq 接受了几个 token)          │
├──────────────────────────────────────────────────────────────┤
│ 3. State 更新 (逐层, 根据 accepted_len 和 recompute_state)    │
│                                                               │
│  ┌─ recompute_state=False (默认):                   ─┐       │
│  │  for layer in range(L):                              │       │
│  │    host 侧 gather:                                   │       │
│  │      state = intermediate_states_buffer[layer]       │       │
│  │              [n, accepted_lens[n]-1, hv, :, :]      │       │
│  │    state_pool[layer][indices] = state                │       │
│  │    # 同时更新 Conv1d state (框架层处理)              │       │
│  └────────────────────────────────────────────────┘       │
│                                                               │
│  ┌─ recompute_state=True:                          ─┐       │
│  │  for layer in range(L):                              │       │
│  │    # 提取该层被接受 token 的 q/k/v/a/b                │       │
│  │    # 对每个 seq, 从 state_pool[layer] 读取初始 state  │       │
│  │    decode_kernel(q_accepted, k_accepted, v_accepted,  │       │
│  │                   a_accepted, b_accepted,             │       │
│  │                   initial_state=state_pool[layer])    │       │
│  │    # decode kernel 将更新后的 state 写回 state_pool   │       │
│  │    # 同时更新 Conv1d state (框架层处理)              │       │
│  └────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

### 6.8 Conv1d 的处理

vLLM 和 SGLang 都需要处理 Conv1d state 的 snapshot。**Conv1d 仅出现在 GDN 路径，KDA 架构本身无 Conv1d**。cuLA 当前的 KDA decode kernel 不涉及 Conv1d（Conv1d 在 kernel 外部处理），但 GDN verify kernel 需要考虑。

**Conv1d state 的特点**：
- Conv1d state 是滑动窗口（如 `[conv_dim, conv_kernel_size-1]`），远小于 recurrent state
- Conv1d 的 snapshot 可以通过截断实现：`state[:, offset:]`，不需要完整复制

**cuLA 的策略**：
- Conv1d 的 snapshot 由框架层处理（不在 verify kernel 内部）
- 框架层在 verify 前保存 Conv1d state，verify 后根据 accepted_len 截断

**⚠️ 同步约束**：Recurrent state 和 Conv1d state 必须在**同一 verify 完成后同步更新**。即：每层 verify 完成后，`intermediate_states_buffer[layer]` 选取的 state 写回 state pool 时，必须同时截断 Conv1d state。如果两个操作在不同步骤完成，可能导致 state 不一致——例如 recurrent state 在 token t2 位置但 Conv1d state 在 token t1 位置。

如果采用 `recompute_state=True`，Conv1d state 需要单独重建或也保存 snapshot。

### 6.9 API 设计

```python
def kda_verify(
    # 模型参数
    A_log: torch.Tensor,          # (HV,)
    dt_bias: torch.Tensor,        # (HV, K) for KDA, (HV,) for GDN
    # 输入
    q: torch.Tensor,              # (1, seq_len, H, K) 或 (batch, draft_len, H, K)
    k: torch.Tensor,              # (1, seq_len, H, K)
    v: torch.Tensor,              # (1, seq_len, HV, V)
    a: torch.Tensor,              # (1, seq_len, HV, K) for KDA, (1, seq_len, HV) for GDN
    b: torch.Tensor,              # (1, seq_len, HV)
    # State
    initial_state_source: torch.Tensor,  # (pool_size, HV, V, K)
    initial_state_indices: torch.Tensor, # (N,)
    # Verify 配置
    cache_intermediate_states: bool = True,  # 是否缓存每步 state
    intermediate_states_buffer: torch.Tensor | None = None,  # (N, max_steps, HV, V, K) — snapshot 模式需要
    intermediate_state_indices: torch.Tensor | None = None,  # (N,)
    disable_state_update: bool = True,  # verify 时不写回主 state pool
    recompute_state: bool = False,  # True: checkpoint+replay; False: snapshot (默认)
    saved_qkvab: dict | None = None,  # per-layer q/k/v/a/b for replay (recompute_state=True 时填充)
    # 树状 speculation
    retrieve_parent_token: torch.Tensor | None = None,  # (seq_len,) 父节点索引
    # 通用参数
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    is_kda: bool = True,  # KDA vs GDN 切换
    state_layout: str = "vk",
) -> torch.Tensor:
    """MTP Verify: process multiple draft tokens with state management for rejection.

    Returns:
        output: (1, seq_len, HV, V) — logits for each draft token
    """
```

验证后选取 state 的操作在 host 侧完成（零 GPU 计算）：

```python
# accepted_lens: (N,) int tensor, 每个 seq 接受了多少个 token
# 注意: accepted_lens 的值域是 [1, spec_len]，至少接受 1 个（验证通过的第一个 token）

N, spec_len = q.shape[0], q.shape[1]
# 构造索引: intermediate_states_buffer[n, accepted_lens[n] - 1]
idx = accepted_lens - 1  # (N,)
idx = idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, HV, V, K)
final_state = intermediate_states_buffer.gather(1, idx.unsqueeze(1)).squeeze(1)  # (N, HV, V, K)

# 更新 state pool
initial_state_source[initial_state_indices] = final_state
```

### 6.10 数值正确性验证

#### 6.10.1 问题

Fused verify kernel 在 SMEM 中跨 token 保持 state，与逐 token decode（每次读写 GMEM）在浮点累加顺序上存在差异：

```
Fused kernel:  S_0 → S_1 → S_2 → ... → S_k  (全部在 SMEM/fp32)
Baseline:      S_0 → S_1 → GMEM(bf16/fp32) → S_1' (load) → S_2 → ...
```

由于 sData(fp32) → GMEM(bf16/fp32) → sData(fp32) 的精度损失，两者的输出可能存在微小差异。对于 KDA kernel，精度的主要来源包括：
1. State 矩阵 (V×K) 的 fp32→bf16→fp32 往返
2. Delta rule 中 `gate * h_old` 的积累顺序不同
3. v_tile 循环中 partial sum 的 reduction 顺序

#### 6.10.2 验证策略

**正确性基线**：逐步单 token decode（复用现有 decode kernel）的输出作为 ground truth。

**测试覆盖**：
- 随机输入，覆盖典型配置：spec_len ∈ {1, 2, 4, 5, 8}, K=128, V=128, HV ∈ {4, 128}
- 对比指标：
  - 输出 logits (bf16): `max(|fused - baseline|) / max(|baseline|)` < 1e-4 (rtol)
  - 最终 state (fp32): `max(|state_fused - state_baseline|) / max(|state_baseline|)` < 1e-5 (rtol)
- 如果启用 bf16 snapshot，state 被选回后经 bf16→fp32 转换的精度损失需单独评估

**Triton 先行验证**：
- Phase 1 用 Triton kernel 实现 fused verify，与逐 token decode 对比
- 确认正确性后再移植到 CuTe DSL，减少 debug 难度

#### 6.10.3 精度容忍

对于 MTP verify 场景，输出 logits 的微小差异（< 1e-4 relative）**不会影响 rejection sampling 的结果**，因为 rejection sampling 是 token-level 的离散决策（max probability token match），而非连续值的比较。但 state 的差异会累积，需要确保在 L 层后总 drift 仍可接受。

---

## 7. 待深入讨论的问题

### 7.1 与现有 decode kernel 的代码复用

verify kernel 与现有 decode kernel 有大量代码重复。结论：**独立实现 verify kernel**，因为：
- verify kernel 有独特的 SMEM 保持策略（state 跨 token 不写回）
- verify kernel 的 q/k/v/a/b 布局不同（多了一个 T 维度）
- 编译时常量 `max_steps` 需要为每个值编译不同变体
- 两个 kernel 的性能调优方向不同

### 7.2 大批量场景的 CTA 映射

现有 large batch kernel 使用 `1 CTA per (seq, head, v_tile)`。verify kernel 中每个 CTA 需要处理 `max_steps` 个 tokens，执行时间变为 `max_steps` 倍。是否需要调整 CTA 映射策略？

对于 SM90 (H100)，每个 SM 最多 2048 threads = 16 warps。当前 large batch kernel 使用 8 warps/CTA，每个 SM 可以跑 2 个 CTA。max_steps=5 时每个 CTA 执行时间约 5x，但 SMEM 占用不变，occupancy 仍然可以维持 2 CTA/SM。

### 7.3 Varlen 输出布局支持

SGLang 的 verify 输入是 varlen 布局（所有 seq 的 tokens 拼成一个大 batch），输出也是 varlen。cuLA 的 verify kernel 当前 API 设计展示了 dense 布局 `(N, T, ...)`，但实际集成时需要支持 varlen 布局。建议 kernel 内部同时支持两种布局，通过 constexpr 开关切换。

### 7.4 树状 Speculation 的进一步优化

当前方案预留了 `retrieve_parent_token` 接口支持树状 speculation，但以下优化待深入：
- 树状场景下 intermediate_states_buffer 的显存管理策略（需要与 `recompute_state=True` 配合）
- 分支节点的 state 回滚延迟（GMEM→SMEM load）是否可以通过预取优化
- EAGLE topk>1 时树的调度顺序对 SMEM 命中率的影响

---

## 8. 实现路线图

### Phase 1: Triton Verify Kernel（正确性验证）
- 基于 SGLang 的 `fused_sigmoid_gating_recurrent.py` 改造
- 增加 KDA 支持（`IS_KDA=True`）
- 增加 `CACHE_INTERMEDIATE_STATES` 和 `DISABLE_STATE_UPDATE`
- 验证正确性：与逐步 decode 的结果对比（见 6.10 节精度标准）
- 支持 `retrieve_parent_token`（树状 speculation，接口预留）

### Phase 2: CuTe Verify Kernel（性能优化）
- 基于 `kda_decode.py` 的 small batch kernel 改造
- 实现 SMEM 中 state 跨 token 保持
- 实现 cp.async snapshot store 与计算 overlap（见 6.5.2 节同步细节）
- 实现树状回滚（从 intermediate_states_buffer 加载父节点 state）
- GDN 标量 gate vs KDA 向量 gate 的 constexpr 分支
- 性能对比：CuTe verify vs Triton verify vs T 次 decode

### Phase 3: 混合方案 — Snapshot + Recompute
- **Snapshot 路径**（默认）：
  - per-layer `intermediate_states_buffer` 分配和管理
  - bf16 snapshot store（见 5.4 节）
  - 选择性 snapshot 策略（见 5.5 节，每 2 步一个 checkpoint）
  - host 侧 gather + state pool 写回
- **Recompute 路径**（`recompute_state=True`）：
  - verify 阶段保存 per-layer q/k/v/a/b
  - verify 后逐层 decode kernel replay
  - Conv1d state 的同步更新和截断（见 6.8 节）
- `recompute_state` 开关的用户接口

### Phase 4: 框架集成
- Python API：`kda_verify()` / `gdn_verify()`（含 `recompute_state` 和 `saved_qkvab` 参数）
- 与 SGLang 的 `RadixLinearAttention` / `GDNKernelDispatcher` 集成
- 与 vLLM 的 `GDNAttentionBackend` 集成
- Varlen 输入/输出布局支持（见 7.3 节）