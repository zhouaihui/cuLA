# vLLM MTP + Linear Attention Speculative Decoding 深度分析

> 本文档整理自一次深度技术对话，涵盖 vLLM 中 MTP（Multi-Token Prediction）验证、Linear Attention 状态管理、kernel 实现与性能优化的完整分析。

---

## 目录

1. [MTP 验证过程中的 State 管理机制](#1-mtp-验证过程中的-state-管理机制)
2. [Linear Attention 层的 KV State 累积与拒绝管理](#2-linear-attention-层的-kv-state-累积与拒绝管理)
3. [Kernel 效率视角的改进空间](#3-kernel-效率视角的改进空间)
4. [Draft Token State 的峰值显存开销](#4-draft-token-state-的峰值显存开销)
5. [Spec Decode 验证是 Prefill 还是 Decode？](#5-spec-decode-验证是-prefill-还是-decode)
6. [GDN Spec Decode Kernel vs Prefill Kernel](#6-gdn-spec-decode-kernel-vs-prefill-kernel)
7. [Spec Decode Kernel vs Packed Decode Kernel](#7-spec-decode-kernel-vs-packed-decode-kernel)

---

## 1. MTP 验证过程中的 State 管理机制

### 问题

对于 KimiLinear 或 Qwen3.5 GDN，在执行 MTP 的时候，验证过程中可能一次验证多个 token，但可能被拒绝，这里是如何保存和管理 state 的？因为拒绝的不能进入 state。

### 核心机制

vLLM 通过**多层协调机制**确保被拒绝的 token 不污染 KV cache 和其他状态：

#### 1. Rejection Sampling 输出

`rejection_sample()` 产出 `[batch, max_spec_len+1]` 张量，被拒绝位置填入 `PLACEHOLDER_TOKEN_ID = -1`（`vllm/v1/sample/rejection_sampler.py`）。

#### 2. Slot Mapping 屏蔽（关键）

`compute_new_slot_mapping()` 将被拒绝 token 的 slot 映射为 `PADDING_SLOT_ID = -1`，**物理阻止** KV cache 写入：

```python
# vllm/v1/spec_decode/utils.py:242
new_slot_mapping.masked_fill_(is_rejected_token_mask, PADDING_SLOT_ID)
```

#### 3. seq_lens 收缩

Proposer 在下一轮 draft 时，将 `seq_lens -= num_rejected_tokens_gpu`，使 attention 层**看不到**被拒绝的 KV 条目（`vllm/v1/spec_decode/llm_base_proposer.py:546-549`）。

#### 4. Scheduler 回滚

调度器乐观地推进 `num_computed_tokens`，验证后按拒绝数回滚：

```python
# vllm/v1/core/sched/scheduler.py:1370-1384
request.num_computed_tokens -= num_rejected
```

#### 5. 混合模型（KimiLinear / BailingMoeLinear）的特殊处理

- **MambaManager** 预分配 `num_speculative_blocks` 个额外块给 speculative tokens
- 验证后，通过 `postprocess_mamba()` 按实际 accepted 数量做状态偏移（shift）
- `remove_skipped_blocks()` 保守地减去 speculative blocks 来调整 `num_computed_tokens`
- KimiLinear 本身跳过 MTP 权重加载（`model.mtp` 前缀返回 None），通过独立路径处理 speculation

#### 总结表

| 层级 | 机制 | 作用 |
|------|------|------|
| GPU kernel | `PADDING_SLOT_ID` 屏蔽 | 阻止被拒 token 写 KV cache |
| Proposer | `seq_lens -= num_rejected` | 让 attention 不看到被拒条目 |
| Scheduler | `num_computed_tokens` 回滚 | 下次调度从最后接受位置重算 |
| 混合模型 | Mamba 状态 shift | 线性注意力状态按接受数对齐 |

核心思想是：**乐观推进 + 验证后回滚**，在 KV cache 层面物理阻止写入，在逻辑层面收缩序列长度和计算进度。

---

## 2. Linear Attention 层的 KV State 累积与拒绝管理

### 问题

在 Linear Attention 层中，KV State 的叠加是如何处理的？验证后的 KV State 是如何管理的，被拒绝的 token 要如何删掉其对 KV State 的作用？

### 核心问题

线性注意力的 recurrent state 是**累加性**的（类似 RNN hidden state），递推公式为：

```
h *= exp(g)              # 指数衰减
v -= (h · k)             # delta rule 修正
v *= sigmoid(b)          # 门控
h += v ⊗ k              # 累加新 token 的贡献
```

一旦被拒绝 token 的 `v⊗k` 混入了 `h`，**无法通过减法精确回退**（因为后续 token 的衰减和修正已经基于错误的 h 计算了）。

### 解决方案：Block-based Savepoint + 多槽位写入

vLLM **不尝试从 state 中"删除"被拒绝 token 的贡献**，而是用 **savepoint/restore** 机制避免被拒绝 token 的 state 被使用。

#### 每个 spec token 写入独立的 block/slot

关键在 kernel `fused_sigmoid_gating_delta_rule_update_kernel`（`fused_sigmoid_gating.py:105-166`）：

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

**每个 spec token 的 state 写入不同的 block**，被拒绝 token 的 block 仅仅是"写了但没人用"。

#### `ssm_state_indices` 的布局

`ssm_state_indices` 形状为 `[batch, num_spec+1]`，为每个 spec 位置分配一个 block ID。例如 `num_spec=2`：

| 位置 | block | 内容 |
|------|-------|------|
| slot 0 | block_A | token_5 的 state（已确认） |
| slot 1 | block_B | draft_1 的 state |
| slot 2 | block_C | draft_2 的 state |

如果 draft_1 被接受但 draft_2 被拒绝，`num_accepted_tokens = 2`，下一步从 `slot[2-1] = slot[1] = block_B` 读取初始 state。block_C 被丢弃，下次覆盖。

#### Savepoint 机制（`mamba_cache_mode = "align"`）

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

- **Conv state**（`get_conv_copy_spec`）：`offset = num_accepted_tokens - 1`，从 `state[src_block, offset:]` 复制，**截断** draft token 的卷积窗口
- **Temporal state**（`get_temporal_copy_spec`）：`src_block_id = block_ids[cur_block_idx + num_accepted_tokens - 1]`，从**最后一个被接受 token 对应的 block** 读取

#### 完整流程

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

#### 总结

| 问题 | 答案 |
|------|------|
| KV state 是累加的吗？ | 是，`h += v⊗k`，不可逆 |
| 验证是串行的吗？ | 不是，rejection sampling 是并行的，但 kernel 内部按时间步顺序更新 state |
| 如何删除被拒绝 token 的影响？ | **不删除**。每个 spec token 写独立 block，被拒绝 block 直接忽略，下一步从被接受 token 的 block 读取初始 state |
| Conv state 怎么处理？ | 从 `state[block, offset:]` 截断复制，offset = num_accepted_tokens - 1 |
| Temporal state 怎么处理？ | 从 `block_ids[cur_idx + num_accepted_tokens - 1]` 读取正确的 block |

核心思想：**不是"撤销"错误的 state，而是"从未使用"错误的 state**——通过多 block 分槽 + savepoint 机制，确保下一步的初始 state 始终来自最后一个被接受 token 的位置。

---

## 3. Kernel 效率视角的改进空间

### P0：消除 `num_accepted_tokens.cpu()` 同步

**现状**：`_update_states_after_model_execute` 中（`gpu_model_runner.py:1441-1447`）：

```python
self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()  # 阻塞！
```

这行代码强制 CPU 等待 GPU 完成 `(output_token_ids != -1).sum(dim=1)`，整个 pipeline 被卡住。

**改进方向**：将 `num_accepted_tokens` 保留在 GPU 上，让 `preprocess_mamba` / `postprocess_mamba` 的 copy 操作也完全在 GPU 端完成（用 GPU kernel 计算 copy 参数代替 Python 循环），消除这步同步。

### P0：`preprocess_mamba` / `postprocess_mamba` 的 Python 循环开销

**现状**（`mamba_utils.py:147-219`）：

- **Python for 循环**遍历所有 request × mamba group × layer × state tensor
- 每次 `collect_mamba_copy_meta` 做 numpy 数组写入（3 次 H2D copy：src_ptrs, dst_ptrs, sizes）
- 每步最多调用 2 次 `do_mamba_copy_block`（pre + post），共 6 次 H2D copy
- 代码里已有 `# TODO(Chen): we need to optimize this function a lot`

**改进方向**：
- 用一个 GPU kernel 从 block IDs 直接计算 copy 参数，替代 Python 循环
- 将 3 个 metadata buffer 合并为一次 H2D copy
- 甚至可以考虑将 savepoint copy 融合到 forward kernel 中，避免单独的 memcpy pass

### P1：`fused_sigmoid_gating` kernel 跳过被拒 slot 写入

**现状**（`fused_sigmoid_gating.py:157-166`）：

```python
for i_t in range(0, T):         # T = 1 + num_speculative_tokens
    b_h += v ⊗ k                # 更新 state
    final_state_idx = ssm_state_indices[n, i_t]
    tl.store(ht + final_state_idx * ..., b_h)  # 每个位置都写！
```

即使只有 2 个 token 被接受、3 个被拒绝，kernel 仍然写入全部 5 个 slot 的 state。被拒绝 slot 的写入在下一步会被覆盖，**白白消耗了 DRAM 带宽**。

**改进方向**：kernel 已知 `num_accepted_tokens`，可以只写 `i_t < num_accepted_tokens` 的位置，跳过被拒绝 slot 的写入。对于 `num_spec=4`、平均接受 2 个 token 的场景，可节省约 60% 的 state 写入带宽。

### P1：Rejection Sampler kernel 融合

**现状**（`rejection_sampler.py:392-503`）：v1 sample 路径每步 launch 6-8 个 kernel：

| 操作 | kernel 数 |
|------|-----------|
| uniform probs 生成 | 1 |
| argmax | 1 |
| greedy rejection | 1 |
| softmax | 1 |
| recovered token sample | ~3 |
| random rejection | 1 |

每次 launch ~5-10μs 开销，累计 ~60μs/step。

**改进方向**：
- 将 softmax + rejection + recovered token sampling 融合为 1-2 个 kernel
- 对 greedy + random 混合 batch，可以统一用 random path（greedy 是 random 的特例）

### P2：GDN Attention Metadata Builder 的 `argsort`

**现状**（`gdn_attn.py:271`）：

```python
index = torch.argsort(spec_token_masks, stable=True)  # O(n log n)
```

**改进方向**：用 exclusive scan（前缀和）+ scatter 实现 O(n) 分区。

### P2：临时 buffer 预分配

| 位置 | 分配 | 改进 |
|------|------|------|
| `prepare_inputs_padded` | `token_indices_to_sample`, `num_rejected_tokens_gpu` | 预分配持久 buffer |
| `_calc_spec_decode_metadata` | 4 个 H2D copy 的 tensor | GPU kernel 直接生成 |
| Rejection sampler | `final_logits = torch.zeros_like(logits)` | 只 gather 需要的 logits |
| GDN metadata builder | 5-6 次 D2D copy 到 cudagraph buffer | 融合为 1 次 |

### P3：MambaManager 精确估算

**现状**（`single_type_kv_cache_manager.py:865`）：

```python
num_computed_tokens = max(0, num_computed_tokens - self.num_speculative_blocks)
```

直接减去全部 speculative blocks，按最坏情况假设所有 draft 都被拒绝。

**改进方向**：用上一步的实际 `num_accepted_tokens` 做更精确的估算，减少 block pool 的 churn。

### 优先级排序

| 优先级 | 改进项 | 预期收益 |
|--------|--------|----------|
| **P0** | 消除 `num_accepted_tokens.cpu()` 同步 | 消除 pipeline stall，对 hybrid 模型影响最大 |
| **P0** | `preprocess_mamba` GPU 化 | 去掉 Python 循环 + 多次 H2D，对多 mamba 层模型至关重要 |
| **P1** | `fused_sigmoid_gating` 跳过被拒 slot 写入 | 节省 ~60% state 写带宽 |
| **P1** | Rejection sampler kernel 融合 | 减少 6-8→2-3 个 kernel，省 ~40μs/step |
| **P2** | `argsort` → partition | O(n log n) → O(n)，大 batch 时明显 |
| **P2** | 临时 buffer 预分配 | 减少分配器开销和内存碎片 |
| **P3** | MambaManager 精确估算 | 减少 block pool churn |
| **P3** | EAGLE parallel drafting 默认化 | 消除串行 forward 循环 |

核心思路：**最大的敌人是 CPU-GPU 同步和 Python 循环**，其次是冗余的内存写入和 kernel launch 开销。对 hybrid 模型（KimiLinear / GDN），P0 的两项改进可能带来 20-30% 的端到端延迟降低。

---

## 4. Draft Token State 的峰值显存开销

### 问题

vLLM 目前无脑把所有 draft token 的 state 都保存，那是不是意味着峰值显存占用要提升几倍？（假设 LA:FullAttention 的比例是 3:1，大部分都是 LA 层）

### 核心机制

vLLM 对每个 LA 层，每个 request 分配 `(1 + num_speculative_blocks)` 个 block（`none` 模式）或 `(2 + num_speculative_blocks)` 个 block（`align` 模式），其中 `num_speculative_blocks = num_speculative_tokens`。

**每个 spec block 保存一份完整的 recurrent state**，即每个 draft token 的 state 独立写入一个 block。

### 定量计算：KimiLinear (KDA) 场景

假设典型配置：

| 参数 | 值 |
|------|-----|
| 总层数 | 48 |
| LA 层数 (KDA) | 36 |
| FA 层数 (MLA) | 12 |
| `num_heads` | 128 |
| `head_dim` | 128 |
| `conv_kernel_size` | 4 |
| `tp_size` | 1 |
| `num_speculative_tokens` | 4 |
| `mamba_block_size` | 1 |
| FA `block_size` | 16 |

**KDA 每 block 每层 state 大小**（`mamba_utils.py:237-267`）：

| State tensor | Shape | 元素数 | dtype | 字节数 |
|---|---|---|---|---|
| `conv_state` | (16384, 3) | 49,152 | BF16 | 96 KB |
| `conv_state_k` | (16384, 3) | 49,152 | BF16 | 96 KB |
| `conv_state_k` (2nd) | (16384, 3) | 49,152 | BF16 | 96 KB |
| `recurrent_state` | (128, 128, 128) | 2,097,152 | **FP32** | **8,192 KB** |
| **合计** | | | | **~8.48 MB** |

> recurrent_state 占 97.5%，是绝对大头。

**每个 request 的 LA 层显存**：

| 场景 | 公式 | 结果 |
|------|------|------|
| 无 spec | `36 × 1 × 8.48 MB` | **305 MB** |
| spec=4 (`none` 模式) | `36 × (1+4) × 8.48 MB` | **1,526 MB** |
| spec=4 (`align` 模式) | `36 × (2+4) × 8.48 MB` | **1,832 MB** |

**每个 request 的 FA 层显存增量**：

FA 层（MLA）的 KV cache 只需多 `ceil(4/16) = 1` 个 block，相比 LA 的额外 4 个 block 几乎可忽略。

### 显存倍率

| `num_spec` | LA 层显存倍率 (vs 无spec) | 总体倍率 (LA:FA=3:1) |
|------------|--------------------------|---------------------|
| 0 | 1× | 1× |
| 1 | 2× | ~1.95× |
| 2 | 3× | ~2.9× |
| 4 | 5× | ~4.8× |
| 8 | 9× | ~8.6× |

**在 LA:FA = 3:1、num_spec=4 的典型配置下，峰值显存约为无 spec 的 5 倍。** 这是因为 LA 层的 recurrent state（FP32, `(H,d,d)` 矩阵）本身就很大，再乘以 `num_spec+1` 个 slot。

### GDN (Qwen3.5) 更严重：conv_state 也会膨胀

GDN 的 `conv_state_shape` 包含 `num_spec` 维度（`mamba_utils.py:226`）：

```python
conv_state_shape = (conv_dim / tp_size, conv_kernel_size - 1 + num_spec)
```

这意味着**不是多分配 block，而是每个 block 的 conv_state 本身就变宽了**。所有 block（包括非 spec 的基础 block）都承担这个膨胀。

假设 `conv_kernel_size=4, num_spec=4`：
- 无 spec：`conv_dim × 3`
- 有 spec：`conv_dim × 7`
- conv_state 膨胀 **2.3 倍**

加上 spec block 数量增加，GDN 的总开销更大。

### 为什么开销这么大？

根本原因是 **recurrent state 不可压缩、不可共享**：

1. **KV cache 可以只存增量**（1 个 block 存 `block_size` 个 token 的 K/V），但 recurrent state 是 `(H, d, d)` 的矩阵，每个位置一整份，无法增量存储
2. **KV cache 可以 PagedAttention 共享 block**（prefix caching），但 recurrent state 每个 request 必须独立持有
3. **recurrent state 用 FP32**（`kda_state_dtype` 第 4 个 tensor 是 `torch.float32`），而 KV cache 通常用 FP8/BF16

### 可能的优化方向

| 方向 | 思路 | 预期收益 |
|------|------|----------|
| **FP32 → BF16 recurrent state** | 将 `recurrent_state` 从 FP32 降精度到 BF16 | 立刻减半 LA 开销，但需验证精度影响 |
| **不存被拒 slot 的 state** | kernel 中只写 `i_t < num_accepted_tokens` 的位置 | 无法减少分配量，但减少写带宽 |
| **共享 spec block pool** | 不同 request 复用同一组 spec block（而非 per-request 分配） | 并发请求多时减少总池大小 |
| **延迟分配 spec block** | 仅在 draft 被接受后才分配对应 block，而非预分配全部 | 减少平均分配量，但增加分配延迟 |
| **Recurrent state checkpoint + recomputation** | 只存 1 个 checkpoint state，被拒时从 checkpoint 重算而非多 slot | 从 `O(num_spec)` 降到 `O(1)` block，但增加重算开销 |
| **GDN conv_state 不膨胀** | conv_state 用 padding/split 方式而非扩展维度 | 避免 conv_state 全局膨胀 |

其中 **FP32 → BF16** 和 **checkpoint + recomputation** 是收益最大的两个方向。前者是"免费"的 2x 减半；后者从根本上改变 `O(num_spec)` 的空间复杂度为 `O(1)`，代价是被拒绝时需要从 checkpoint 重新 forward 被接受的 token 来恢复 state。

---

## 5. Spec Decode 验证是 Prefill 还是 Decode？

### 问题

vLLM scheduler 这里没有区分 prefilling/decoding 吗？

### 答案：区分了，但取决于 Attention Backend

vLLM **确实区分** prefill 和 decode，但对 spec decode request 的处理方式取决于 attention backend 是否支持 **spec-as-decode**。

### 核心分类逻辑

分类入口是 `split_decodes_and_prefills()`（`utils.py:496`），用 `decode_threshold` 判断：

```python
is_prefill = query_lens > decode_threshold
```

- **`decode_threshold = 1`**（默认）：`query_len > 1` 的请求都算 prefill
- **`decode_threshold = 1 + num_spec_tokens`**（spec-as-decode）：`query_len ≤ 1 + num_spec_tokens` 的请求都算 decode

### 各 Backend 的处理方式

| Backend | `supports_spec_as_decode` | Spec request 走哪条路径 |
|---------|--------------------------|----------------------|
| **FlashInfer (TRTLLM)** | `True`（仅 TRTLLM 路径） | **Decode** — `decode_threshold` 被提升到 `1 + num_spec_tokens` |
| **FlashInfer (native)** | `False` | **Prefill** — spec request 的 `query_len > 1`，被归类为 prefill |
| **GDN Attention** | 特殊处理 | **独立第三路径** — spec decode 请求既不是 prefill 也不是 decode，走专用的 spec decode kernel |
| **FlashMLA Sparse** | `True` | **Decode** |
| **ROCm Aiter FA** | `True` | **Decode** |
| **Mamba Attn** | `True` | **Decode**（但 mamba 层本身不走 FA kernel） |
| **CPU Attn** | `False` | **Prefill** |

### 具体路径差异

#### FlashInfer native（不支持 spec-as-decode）

Spec request `query_len = K+1 > 1` → 归类为 **prefill** → 走 `BatchPrefillWithPagedKVCacheWrapper`。

这是 **varlen prefill kernel**，一次处理 K+1 个 token，每个 token 可以 attend 到之前所有 KV cache。计算效率不如 decode kernel，但语义正确。

#### FlashInfer + TRTLLM（支持 spec-as-decode）

`decode_threshold` 被设为 `1 + num_spec_tokens`，所以 spec request `query_len = K+1 ≤ decode_threshold` → 归类为 **decode** → 走 `TRTLLMDecode` 路径。

TRTLLM 的 decode kernel 支持 multi-query decode（每个 request 多个 query token 同时 decode），比 prefill kernel 更高效。

#### GDN Attention（KimiLinear / Qwen3.5 专用）

这是最精细的处理。代码在 `gdn_attn.py:156-340`：

```python
# 用 num_decode_draft_tokens 区分 spec decode 和普通 decode
spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0

# 将 batch 分成三类：
# 1. num_decodes: 普通 decode (query_len=1, 无 spec)
# 2. num_prefills: prefill 请求
# 3. num_spec_decodes: spec decode 请求 (query_len=K+1, 有 draft tokens)
```

关键：**当 spec decode 和普通 decode 同时存在时，普通 decode 会被重新归类为 prefill**（`gdn_attn.py:235-239`）：

```python
if num_decodes > 0 and num_spec_decodes > 0:
    num_prefills += num_decodes    # 普通 decode 归入 prefill
    num_prefill_tokens += num_decode_tokens
    num_decodes = 0
    num_decode_tokens = 0
```

原因是 GDN 的 prefill kernel（FLA chunk ops）可以正确处理 `query_len=1` + `initial_state` 的情况，结果与 decode kernel 一致，这样 batch 中只需要两种 kernel（prefill + spec_decode），而不是三种。

Spec decode 请求走专用路径：
- 使用 `spec_state_indices_tensor` 加载正确的初始 recurrent state
- `spec_query_start_loc` 独立构建
- `fused_sigmoid_gating` kernel 以 `IS_SPEC_DECODING=True` 模式运行

### 还有一个关键区分：`num_decode_draft_tokens` vs `num_draft_tokens`

在 `_prepare_inputs`（`gpu_model_runner.py:2061-2065`）：

```python
if (
    self.input_batch.num_computed_tokens_cpu[req_idx]
    >= self.input_batch.num_prompt_tokens[req_idx]
):
    num_decode_draft_tokens[req_idx] = draft_len  # 只在 decode 阶段才标记
```

**只有当 request 已经完成 prefill（`num_computed_tokens >= num_prompt_tokens`）时，draft tokens 才被标记为 `decode_draft`**。如果 request 还在 prefill 阶段，draft tokens 存在但 `num_decode_draft_tokens = -1`，GDN backend 会将其当作普通 prefill 处理。

### 总结

| 场景 | 分类 | Kernel |
|------|------|--------|
| Spec decode request + FlashInfer native | Prefill | varlen prefill kernel |
| Spec decode request + TRTLLM | Decode | multi-query decode kernel |
| Spec decode request + GDN (纯 spec batch) | Spec Decode | 专用 spec decode kernel（`IS_SPEC_DECODING=True`） |
| Spec decode request + GDN (混合 batch) | Spec Decode + 普通 decode 归入 Prefill | FLA prefill kernel + spec decode kernel |
| Spec request 在 prefill 阶段 | Prefill | 普通 prefill kernel（不使用 spec 路径） |

vLLM 确实区分了 prefill 和 decode，但 spec decode 验证既不是传统 prefill 也不是传统 decode，而是第三种路径。对于 FA 层，它根据 backend 能力选择走 prefill 或 decode kernel；对于 LA 层（GDN），它走专用的 spec decode kernel。

---

## 6. GDN Spec Decode Kernel vs Prefill Kernel

### 问题

GDN 挺有意思，还有专门的 Spec decode kernel，可以探索下实现，和 prefill kernel 有什么差别。

### 架构总览

GDN Linear Attention 有三条完全不同的 kernel 路径：

```
┌─────────────────────────────────────────────────────┐
│             GatedDeltaNetAttention._forward_core      │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Prefill    │  │  Spec Decode │  │  Packed Decode│ │
│  │  chunk_gdn   │  │  sigmoid_    │  │  packed_     │ │
│  │  (6 sub-     │  │  gating_     │  │  decode      │ │
│  │   kernels)   │  │  update      │  │  (1 kernel)  │ │
│  └─────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Prefill Kernel：`chunk_gated_delta_rule`

**核心思想**：将序列分成 `FLA_CHUNK_SIZE=64` 的 chunk，chunk 内并行，chunk 间串行传递 state。

**流水线**（6 个子 kernel 顺序执行）：

```
输入: q[N,T,H,K], k[N,T,H,K], v[N,T,HV,V], g[N,T,HV], beta[N,T,HV]
        │
        ▼
① chunk_local_cumsum      ← chunk 内 g 的累积和
        │
        ▼
② chunk_scaled_dot_kkt    ← A = beta * K @ K^T (intra-chunk, 用 Tensor Core)
        │
        ▼
③ solve_tril              ← (I + A)^{-1} (三角求逆, blocked 16×16)
        │
        ▼
④ recompute_w_u           ← WY 表示: w = (I+A)^{-1}(beta*k*exp(g)), u = (I+A)^{-1}(beta*v)
        │
        ▼
⑤ chunk_gated_delta_rule_fwd_h  ← inter-chunk 递推:
   │                                h *= exp(g_chunk_boundary)
   │                                v_new = v - w @ h^T
   │                                h += v_new^T @ k
   │                                写 final_state
   ▼
⑥ chunk_fwd_o             ← o = (q * exp(g)) @ h + A_intra @ v_new
```

**关键特性**：
- **高度并行**：chunk 之间和 head 之间并行，grid 维度包含 `N * H`
- **Tensor Core 友好**：步骤②④⑤⑥都是矩阵乘法，可以充分利用 Tensor Core
- **Autotuned**：多个子 kernel 对 BK/BV/warps/stages 做自动调优
- **g/beta 预计算**：由 `fused_post_conv_prep` 在 kernel 外计算好，传入时已经是 `float` 值
- **Initial state**：调用方用 `ssm_state[indices].contiguous()` gather 成连续张量传入，kernel 内按 sequence ID 直接索引
- **Final state**：kernel 返回新张量，调用方 scatter 回写 `ssm_state[indices] = final_state`

### Spec Decode Kernel：`fused_sigmoid_gating_delta_rule_update`

**核心思想**：单 kernel 内串行处理 T 个 token（T = 1 + num_spec），每步写一个 state slot。

```
输入: q[1,T,H,K], k[1,T,H,K], v[1,T,HV,V], A_log[HV], a[T,HV], b[T,HV], dt_bias[HV]
      ssm_state_indices[batch, num_spec+1]   ← 2D！
      num_accepted_tokens[batch]             ← spec decode 专属
      initial_state = ssm_state              ← 整个 cache tensor
        │
        ▼
   单个 Triton kernel:
   ┌──────────────────────────────────────────────┐
   │  grid: (1, NV, N*HV)                         │
   │  每个 program 处理 1 个 (sequence, head)       │
   │                                               │
   │  ① 加载 initial state:                        │
   │     slot = ssm_state_indices[n,               │
   │            num_accepted_tokens[n] - 1]         │
   │     h = ssm_state[slot]                       │
   │                                               │
   │  ② for i_t in range(T):                       │
   │     g = -exp(A_log) * softplus(a + dt_bias)  │ ← 融合计算
   │     beta = sigmoid(b)                         │ ← 融合计算
   │     q, k = l2norm(q), l2norm(k)              │ ← 融合计算
   │     h *= exp(g)                               │
   │     v -= (h @ k^T)                            │
   │     v *= beta                                 │
   │     h += v ⊗ k                                │
   │     o = h @ q^T                               │
   │     store(o)                                  │
   │     slot = ssm_state_indices[n, i_t]          │
   │     ssm_state[slot] = h                       │ ← 每步写一个 slot
   └──────────────────────────────────────────────┘
```

### 完整对比表

| 特性 | Prefill (chunk) | Spec Decode (sigmoid gating) | Packed Decode |
|------|----------------|-----------------------------|---------------|
| **Kernel 文件** | `chunk.py` (6 sub-kernels) | `fused_sigmoid_gating.py` | `fused_recurrent.py` (packed decode) |
| **输入格式** | 分开的 q/k/v/g/beta 预计算 | 分开的 q/k/v + raw A_log/a/b/dt_bias | Packed `mixed_qkv` + raw A_log/a/b/dt_bias |
| **g 计算** | 外部预计算 by `fused_post_conv_prep` | kernel 内融合计算 | kernel 内融合计算 |
| **beta 计算** | 外部预计算 (sigmoid'd) | kernel 内融合计算 (sigmoid(b)) | kernel 内融合计算 (sigmoid(b)) |
| **Q/K L2 norm** | 外部预计算 | kernel 内融合计算 | kernel 内融合计算 |
| **Initial state 加载** | gather 成连续 `[N, HV, V, K]`，kernel 直接索引 | 通过 2D `ssm_state_indices[batch, num_accepted[i]-1]` 间接寻址 | 通过 1D `ssm_state_indices[batch]` 间接寻址 |
| **State 写回** | 返回新张量，调用方 scatter | **inplace**，每步写 `ssm_state_indices[n, i_t]` | **inplace**，写 1D `ssm_state_indices[batch]` |
| **`num_accepted_tokens`** | N/A | 必需——决定初始 state 来自哪个 slot | N/A |
| **`ssm_state_indices` 维度** | 1D（调用方用于 gather/scatter） | 2D `[batch, num_spec+1]`（kernel 内直接寻址） | 1D `[batch]` |
| **递推方式** | Chunk 并行 (WY 表示 + inter-chunk carry) | 串行 per-token loop | 单步 (T=1)，无循环 |
| **Autotuning** | 有（多个 sub-kernel 对 BK/BV/warps/stages 调优） | 无（固定 warps=4, stages=3） | 无（固定 warps=1, stages=3） |
| **子 kernel 数** | 6 | 1 | 1 |
| **并行度** | 跨 chunk 和 head；长序列高吞吐 | 仅跨 sequence 和 head | 仅跨 sequence 和 head |
| **内存模式** | Tensor Core 友好（BK/BV 块上的矩阵乘） | 元素级串行递推 | 元素级单步 |
| **Conv1d 路径** | `causal_conv1d_fn`（batch varlen） | `causal_conv1d_update`（multi-token + `num_accepted_tokens`） | `causal_conv1d_update`（single token） |

### Spec Decode Kernel 的独有设计要点

#### a) 2D `ssm_state_indices` 间接寻址

这是 spec decode 最核心的设计。普通 decode 只需 1 个 slot，但 spec decode 需要为每个 draft token 分配独立 slot：

```
ssm_state_indices[batch, num_spec+1]:
  request_0: [block_5, block_10, block_11, block_12, block_13]  ← num_spec=4
  request_1: [block_8, block_14, block_15, block_16, block_17]
```

kernel 内：
- **读初始 state**：`slot = indices[n, num_accepted_tokens[n] - 1]`（跳到被接受的位置）
- **写每步 state**：`slot = indices[n, i_t]`（每步写入不同 slot）

这种设计让被拒绝 token 的 state 虽然被写入了，但**从未被后续步骤读取**，实现了"写而不读"的隐式回滚。

#### b) `num_accepted_tokens` 驱动初始状态选择

```triton
if IS_SPEC_DECODING:
    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
else:
    i_t = 0
```

`IS_SPEC_DECODING` 是 Triton `heuristic`，在编译时决定走哪个分支，**零运行时开销**。`num_accepted_tokens` 是 GPU tensor，由上一步 rejection sampler 写入，无需 CPU-GPU 同步。

#### c) 全融合：gating + norm + recurrence + state write

Prefill 路径需要 6 个子 kernel + 外部预处理，而 spec decode 把所有操作融合进 1 个 kernel：

```
Prefill:  fused_post_conv_prep → cumsum → KKT → solve_tril → WY → h → output
                                                    (6 launches)

Spec Decode:  causal_conv1d_update → fused_sigmoid_gating_delta_rule_update
                                                    (2 launches, 其中第2个是全融合)
```

这种融合在 T 很小（1-5 个 token）时特别有利——kernel launch 开销占比高，融合可以显著减少延迟。

### 性能特征对比

```
Prefill (长序列):
  ████████████████████████████  ← chunk 并行 + Tensor Core，高吞吐
  适合: T >> 64

Spec Decode (短序列, T=1~5):
  ████████                      ← 串行递推，但单 kernel 全融合
  适合: T 很小，减少 launch 开销

Packed Decode (T=1):
  ████                          ← 最极致优化，无循环，packed 输入
  适合: 单 token decode，最高频路径
```

---

## 7. Spec Decode Kernel vs Packed Decode Kernel

### 问题

为什么在 T≤5 的情况下 prefill 的 chunk-wise 不如 spec decode 这种实现高效？以及 spec decode 和 packed decode 的性能差别在哪里？

### 为什么 Chunk Prefill 在 T≤5 时不如 Spec Decode

#### 1. Chunk Pipeline 的中间张量灾难

Prefill 的 6 个子 kernel 之间必须通过**全局显存**传递中间结果：

```
① cumsum → g_cumsum        [1, T, HV]    写出
② KKT    → A               [1, T, T, HV] 读入①, 写出
③ solve  → A_inv           [1, T, T, HV] 读入②, 写出
④ WY     → w, u            [1, T, H, K/V] 读入②③, 写出
⑤ h      → h, v_new        [1, num_chunks, HV, V, K] 读入④, 写出
⑥ output → o               [1, T, HV, V]  读入④⑤, 写出
```

以 `T=5, H=128, K=V=128, HV=128` 为例，6 个 kernel 之间至少 5 次全局显存写 + 5 次读，合计约 ~18 MB 的全局显存流量（per layer per batch）。

而 spec decode 的融合 kernel 的 `b_h` 在寄存器中 `[BV, BK]`，5 个时间步全在寄存器/共享内存完成，0 次中间全局显存读写，只在最后写 output 和 state。

**全局显存带宽是 GPU 上最稀缺的资源**。T=5 时，chunk pipeline 的中间流量远大于实际计算量。

#### 2. Kernel Launch 开销

每个 Triton/CUDA kernel launch 有 ~5-10μs 的固定开销：

```
Chunk pipeline:  6 launches × ~8μs = ~48μs (纯 launch 开销)
Spec decode:     1 launch  × ~8μs = ~8μs
```

T=5 时每个子 kernel 的实际计算量极小，launch 开销占总时间的比例可能超过 50%。

#### 3. WY 修正：O(T²) 的"负优化"

Chunk 并行的核心思想是：把序列分成 chunk，chunk 内用 **WY 表示** 处理依赖关系，chunk 间才串行传递 state。

WY 修正的步骤是：

```
② KKT:    A[i,j] = β_i × k_i × k_j^T × exp(g_i - g_j)   → O(T² × H × K²)
③ solve:   (I + A)^{-1}                                     → O(T³ × H)
④ WY:      w = (I+A)^{-1} (β k exp(g)),  u = (I+A)^{-1} v  → O(T² × H × K × V)
```

当 T=5 时：
- KKT：5×5 矩阵乘法，128 个 head 并行。但每个 head 只做 5×5 = 25 次乘加，**Tensor Core 需要至少 16×16 矩阵才能高效运行**
- solve_tril：5×5 三角求逆，计算量微乎其微
- WY：5×5 矩阵乘以 128×128，还是太小

**串行递推只需要 5 步，每步 O(H×K×V) = O(128×128×128) ≈ 2M FLOPs**。

而 WY 修正引入的额外计算量：

```
KKT:    5×5 × 128 × 128×128 = ~5.2M FLOPs   (但矩阵太小，Tensor Core 严重低效)
solve:  5³ × 128 = 16K FLOPs                  (微不足道但 launch 开销一样)
WY:     5×5 × 128 × 128×128 × 2 = ~10.5M FLOPs
──────
WY 额外: ~15.7M FLOPs

串行递推: 5 × 2M = 10M FLOPs
```

WY 修正**反而比串行递推多 50% 的计算量**，而且这些计算在小矩阵上完全无法利用 Tensor Core。

#### 4. Chunk Size 不匹配

`FLA_CHUNK_SIZE = 64`。T=5 时只有 **1 个 chunk**，chunk 间并行度为 0：

```
Prefill:  grid = (num_chunks=1, NV, N*H)  → 只有 N*H 并行
Spec decode: grid = (1, NV, N*HV)         → N*HV 并行（几乎相同）
```

chunk 并行的核心优势——多 chunk 并行处理——在 T=5 时**完全不存在**。6 个 kernel 的并行度与 1 个融合 kernel 相同，但多了 5 次全局同步。

#### 5. 本质原因

Chunk 并行是 **为长序列设计的算法**。它的复杂度分界点是：

```
串行递推:  O(T × H × K × V)           ← 线性
Chunk WY:  O(T²/CHUNK × H × K² + T × H × K × V)  ← T² 项在 chunk 内, T 项在 chunk 间
```

当 `T >> CHUNK_SIZE` 时，chunk 间并行度 = T/CHUNK，WY 的 T² 项被限制在 chunk 内，整体接近 O(T)。但当 `T < CHUNK_SIZE` 时：

- **chunk 间并行度 = 0**（只有 1 个 chunk）
- **WY 的 T² 项 = T²**（全部在一个 chunk 内）
- **6 个 kernel launch + 中间张量同步** 的固定开销完全无法被并行度摊薄

**简单说：T=5 时 chunk 并行把一个本该 O(T) 的串行问题，用 O(T²) 的 WY 修正 + 6 个 kernel 的方式去"并行"，结果并行度为 1，额外开销 > 收益。**

### Spec Decode vs Packed Decode：逐行对比

#### Grid 和 Warp 配置

```
Spec Decode:  grid = (NK=1, NV, N * HV),  num_warps=4, num_stages=3
Packed Decode: grid = (NV, B * HV),        num_warps=1, num_stages=3
```

| | Spec Decode | Packed Decode |
|---|---|---|
| NK 维度 | 有（但 `NK=1`，所以实际无） | 无 |
| NV 维度 | `cdiv(V, BV)` | `cdiv(V, BV)` |
| 并行度 | `N * HV * NV` | `B * HV * NV` |
| warps | 4 | 1 |

**warps=4 vs warps=1 的含义**：每个 SM 上 4 个 warp 共享同一个 program，可以协作做更大的 BV 块，或者各自做不同工作。对于 spec decode，因为每个 program 要处理 T 个时间步（T=1~5），4 个 warp 可以在时间步之间更好地利用指令级并行和寄存器复用。对于 packed decode（T=1），1 个 warp 就够了，因为计算量极小，多 warp 反而增加调度开销。

#### 输入格式：最大差异

**Packed Decode** — 融合的 `mixed_qkv` 输入：

```triton
// 一次内存读取，在寄存器中拆分
p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
q_off = i_h * K + o_k
k_off = (H * K) + i_h * K + o_k
v_off = (2 * H * K) + i_hv * V + o_v
b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0)
b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0)
b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0)
```

**Spec Decode** — 分离的 q/k/v 张量：

```triton
// 三次独立内存读取
p_q = q + (bos * H + i_h) * K + o_k
p_k = k + (bos * H + i_h) * K + o_k
p_v = v + (bos * HV + i_hv) * V + o_v
// 每个时间步还要递增指针
p_q += H * K; p_k += H * K; p_v += HV * V
```

| 操作 | Packed Decode | Spec Decode |
|------|--------------|-------------|
| QKV 读取 | 1 次合并读取（`mixed_qkv` 连续存放） | 3 次独立读取（q/k/v 分开存放） |
| 内存事务 | 1 次 global load（coalesced） | 3 次 global load（可能不 coalesced） |
| 时间步间指针更新 | 无 | `p_q += H*K; p_k += H*K; p_v += HV*V` 每次 |

Packed 格式让 Q/K/V 在内存中连续排列 `[q | k | v]`，对同一个 head 的同一次访问，GPU 可以用更少的内存事务完成。而 spec decode 的分离张量，q/k/v 分布在不同地址，每次读取是独立的内存事务。

#### Gating 参数读取

**Packed Decode** — 标量参数只读一次：

```triton
a_val = tl.load(a + i_n * stride_a_tok + i_hv)        // 1 次
b_val = tl.load(b + i_n * stride_b_tok + i_hv)        // 1 次
A_log_val = tl.load(A_log + i_hv)                     // 1 次
dt_bias_val = tl.load(dt_bias + i_hv)                  // 1 次
g_val = -exp(A_log_val) * softplus(a_val + dt_bias_val) // 标量
beta_val = sigmoid(b_val)                               // 标量
```

**Spec Decode** — 每个时间步都要重新读取：

```triton
for i_t in range(0, T):
    b_b = tl.load(p_b)                                          // 每步
    x = tl.load(p_a) + tl.load(p_dt_bias)                      // 每步 (2次)
    b_g = -tl.exp(tl.load(p_A_log)) * softplus(x)              // 每步
    b_beta = tl.sigmoid(b_b)                                     // 每步
    // ... 递推 ...
    p_b += HV; p_a += HV                                        // 指针更新
```

| | Packed Decode | Spec Decode (T=5) |
|---|---|---|
| `a` 读取 | 1 次 | 5 次 |
| `b` 读取 | 1 次 | 5 次 |
| `A_log` 读取 | 1 次 | 5 次 |
| `dt_bias` 读取 | 1 次 | 5 次 |
| softplus/sigmoid 计算 | 1 次 | 5 次 |

Packed decode 因为 T=1，所有 gating 参数是标量，只读一次。Spec decode 每步的 gating 参数不同，必须每步重新读取和计算。这是**语义必需的**，不是实现缺陷——spec decode 的每个 token 有不同的 decay 和 beta。

但注意：`A_log` 和 `dt_bias` 是模型参数（不随 token 变化），理论上可以提到循环外只读一次。当前实现中它们在循环内每步读取，这是一个微小的优化空间。

#### State 索引

**Packed Decode** — 1D 直接索引：

```triton
state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq)  // 1 次 load
p_h0 = h0 + state_idx * stride_init_state_token
b_h = tl.load(p_h0, ...)    // 读 state
// ... 递推 ...
p_ht = ht + state_idx * stride_final_state_token
tl.store(p_ht, b_h, ...)    // 写 state
```

**Spec Decode** — 2D 索引 + `num_accepted_tokens`：

```triton
// 初始 state：需要额外的 num_accepted_tokens load
if IS_SPEC_DECODING:
    i_t = tl.load(num_accepted_tokens + i_n) - 1    // 额外 1 次 load
state_idx = tl.load(ssm_state_indices + i_n * stride + i_t)  // 2D 索引
p_h0 = h0 + state_idx * stride_init_state_token
b_h = tl.load(p_h0, ...)

// 每步写 state：需要额外的 ssm_state_indices load
for i_t in range(0, T):
    // ... 递推 ...
    final_state_idx = tl.load(ssm_state_indices + i_n * stride + i_t)  // 每步 1 次
    if final_state_idx > 0:
        p_ht = ht + final_state_idx * stride_final_state_token
        tl.store(p_ht, b_h, ...)    // 写 state（每步一次）
```

| | Packed Decode | Spec Decode (T=5) |
|---|---|---|
| `ssm_state_indices` load | 1 次 | 1 + 5 = 6 次 |
| `num_accepted_tokens` load | 0 | 1 次 |
| State 读取 | 1 次 | 1 次 |
| State 写入 | 1 次 | 5 次 |
| NULL_BLOCK_ID 检查 | 1 次 | 1 + 5 = 6 次 |

Spec decode 的 state 写入是最重的操作——每次写入 `[BV, BK]` 的 FP32 矩阵（若 `BV=32, BK=128`，则 16KB）。T=5 时写入 5 次 = 80KB，而 packed decode 只写 16KB。

#### 递推循环

**Packed Decode** — 无循环，单步展开：

```triton
b_h *= exp(g_val)           // 标量 exp
b_v -= tl.sum(b_h * b_k[None, :], 1)
b_v *= beta_val
b_h += b_v[:, None] * b_k[None, :]
b_o = tl.sum(b_h * b_q[None, :], 1)
```

**Spec Decode** — 循环 T 次：

```triton
for i_t in range(0, T):
    b_q = tl.load(p_q, ...)     // 读 q
    b_k = tl.load(p_k, ...)     // 读 k
    b_v = tl.load(p_v, ...)     // 读 v
    // ... gating 计算 ...
    b_h *= tl.exp(b_g)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o, ...)     // 写 output
    // ... 写 state ...
    p_q += H * K; p_k += H * K; p_v += HV * V; p_o += HV * V; p_b += HV; p_a += HV
```

关键差异：Packed decode 的 `g_val` 和 `beta_val` 是标量，编译器可以更好地优化 `exp(g_val)` 为单次标量运算。而 spec decode 的 `b_g` 虽然在 `IS_KDA=False` 时也是标量，但在循环内每步重新计算，编译器无法做标量提升优化。

#### Conv1d 路径差异

**Packed Decode**：

```python
mixed_qkv = causal_conv1d_update(
    mixed_qkv,          # [B, dim] — 单 token
    conv_state, conv_weights, bias,
    conv_state_indices=indices[:B],  # 1D
    validate_data=False,
)
# 无 num_accepted_tokens, 无 query_start_loc
```

**Spec Decode**：

```python
mixed_qkv = causal_conv1d_update(
    mixed_qkv,          # [num_spec_tokens, dim] — 多 token varlen
    conv_state, conv_weights, bias,
    conv_state_indices=spec_state_indices[:, 0],  # 每个 spec 请求的第 0 个 block
    num_accepted_tokens=num_accepted_tokens,       # ← 额外参数
    query_start_loc=spec_query_start_loc,          # ← 额外参数
    max_query_len=spec_state_indices.size(-1),     # ← 额外参数
)
```

Spec decode 的 conv1d 需要处理变长多 token 输入，从 `num_accepted_tokens` 指定的位置开始滑动窗口更新。Packed decode 只需处理单 token 的简单滑动窗口。

#### 量化对比（单层，N=32 请求，H=HV=128，K=V=128）

| 操作 | Packed Decode | Spec Decode (T=5) | 倍率 |
|------|--------------|-------------------|------|
| **全局显存读** | | | |
| QKV 读取 | 1 merged load × 32 | 3 separate loads × 32 × 5 | ~15× |
| Gating 参数 | 4 scalar loads × 32 | (2+1+1+1) loads × 32 × 5 | ~6× |
| State 读取 | 1 × 32 × 128 | 1 × 32 × 128 | 1× |
| Indices 读取 | 1 × 32 | 6 × 32 | 6× |
| **全局显存写** | | | |
| Output | 1 × 32 × 128 | 5 × 32 × 128 | 5× |
| State | 1 × 32 × 128 | 5 × 32 × 128 | 5× |
| **计算** | | | |
| `h *= exp(g)` | 1 × 32 × 128 | 5 × 32 × 128 | 5× |
| `v -= (h @ k)` | 1 × 32 × 128 | 5 × 32 × 128 | 5× |
| `h += v ⊗ k` | 1 × 32 × 128 | 5 × 32 × 128 | 5× |
| Softplus/Sigmoid | 1 × 32 | 5 × 32 | 5× |
| **调度** | | | |
| Grid | (4, 4096) | (1, 4, 4096) | — |
| Warps/program | 1 | 4 | — |

#### Packed Decode 的优化精髓

Packed decode 的优化可以总结为**把 decode 推到极致的单步极简**：

1. **零冗余 I/O**：QKV 合并为一次读取，gating 参数只读一次，state 读写各一次
2. **零循环开销**：T=1 无循环，无指针递增，无分支
3. **零间接寻址**：1D `ssm_state_indices`，一次 load 直接定位 state
4. **零条件分支**：无 `IS_SPEC_DECODING`/`IS_CONTINUOUS_BATCHING`/`INPLACE_FINAL_STATE` 的 Triton heuristic 分支
5. **最小 warp 配置**：`num_warps=1`，单 warp 独占一个 program，无 warp 间同步开销
6. **Pre-allocated output**：`out` buffer 预分配，直接写入，无 `new_empty`

Spec decode **无法享受这些优化**，不是因为实现不够好，而是语义上它必须：
- 每步读不同的 q/k/v（不同 token）
- 每步读不同的 gating 参数（不同 decay/beta）
- 每步写不同的 state slot（多 slot 保存用于回滚）
- 用 2D 索引 + `num_accepted_tokens`（选择正确的初始 state）

**Spec decode 相比 packed decode 的额外开销，本质上是为"一次 forward 验证多个 draft token"付出的代价。** 如果把 spec decode 拆成 T 次 packed decode，每次 1 个 token，则：
- T 次 kernel launch：T × ~8μs
- T 次 conv1d_update 调用
- T 次 QKV 投影
- T 次 state 读取（从上一步写入的 slot）

而融合的 spec decode kernel 只需 1 次 launch + 1 次 conv1d_update + 1 次 QKV 投影 + 1 次 state 读取。**在 T=5 时，融合 kernel 比 T 次独立 packed decode 节省约 4 × 8μs = 32μs 的 launch 开销 + 4 次 state 读取的显存延迟**。

---

## 关键源码索引

| 组件 | 文件路径 |
|------|----------|
| Rejection Sampler | `vllm/v1/sample/rejection_sampler.py` |
| Spec Decode Utils | `vllm/v1/spec_decode/utils.py` |
| LLM Base Proposer | `vllm/v1/spec_decode/llm_base_proposer.py` |
| Mamba Utils (pre/post process) | `vllm/v1/worker/mamba_utils.py` |
| MambaManager | `vllm/v1/core/single_type_kv_cache_manager.py` |
| GPU Model Runner | `vllm/v1/worker/gpu_model_runner.py` |
| GDN Attention Backend | `vllm/v1/attention/backends/gdn_attn.py` |
| GDN Linear Attention Layer | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` |
| KDA Layer | `vllm/model_executor/layers/kda.py` |
| Mamba State Shapes | `vllm/model_executor/layers/mamba/mamba_utils.py` |
| MambaSpec | `vllm/v1/kv_cache_interface.py` |
| KimiLinear Model | `vllm/model_executor/models/kimi_linear.py` |
| Scheduler | `vllm/v1/core/sched/scheduler.py` |
| Chunk Prefill Kernel | `vllm/model_executor/layers/fla/ops/chunk.py` |
| Spec Decode Kernel | `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py` |
| Packed Decode Kernel | `vllm/model_executor/layers/fla/ops/fused_recurrent.py` |
| Causal Conv1d | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` |
| FlashInfer Backend | `vllm/v1/attention/backends/flashinfer.py` |
| Attention Utils | `vllm/v1/attention/backends/utils.py` |