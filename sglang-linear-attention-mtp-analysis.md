# SGLang Linear Attention MTP 实现深度分析

> 分析日期：2026-05-14
> 涉及模型：Qwen3-Next、Qwen3.5 (GDN)、Kimi Linear (KDA)
> 代码库版本：main 分支，commit 2417a9da5

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [MTP Draft 模型结构](#2-mtp-draft-模型结构)
3. [Linear Attention 统一抽象层：RadixLinearAttention](#3-linear-attention-统一抽象层radixlinearattention)
4. [GDN 后端详解](#4-gdn-后端详解)
5. [KDA 后端详解](#5-kda-后端详解)
6. [MTP Verify 路径深度分析](#6-mtp-verify-路径深度分析)
7. [CuTeDSL Kernel 实现对比](#7-cutedsl-kernel-实现对比)
8. [Triton Verify Kernel 逐行解析](#8-triton-verify-kernel-逐行解析)
9. [FlashInfer GDN MTP Kernel](#9-flashinfer-gdn-mtp-kernel)
10. [完整调度矩阵](#10-完整调度矩阵)
11. [Kimi Linear 混合架构](#11-kimi-linear-混合架构)
12. [HybridLinearAttnBackend 调度逻辑](#12-hybridlinearattnbackend-调度逻辑)
13. [关键文件索引](#13-关键文件索引)

---

## 1. 整体架构概览

SGLang 支持**两种** Linear Attention 模型的 MTP (Multi-Token Prediction)：**Qwen3-Next / Qwen3.5 (GDN)** 和 **Kimi Linear (KDA)**。两者都使用 **EAGLE/NEXTN** speculative decoding 算法，但在 linear attention 的具体变体和 verify 路径上有所不同。

### Speculative 解码算法枚举

文件：`python/sglang/srt/speculative/spec_info.py`

```python
class SpeculativeAlgorithm(Enum):
    DFLASH = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    FROZEN_KV_MTP = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()
```

- GDN/KDA 模型使用 **EAGLE** 算法（通过 `--speculative-algorithm NEXTN` 或自动检测）
- Gemma4 MTP 使用 **FROZEN_KV_MTP** 算法（draft 模型直接读取 target 的 KV cache）

### 核心数据流

```
┌─────────────────────────────────────────────────┐
│                EAGLEWorker                       │
│                                                  │
│  1. Target Forward (decode)                      │
│     → hidden_states + logits                     │
│                                                  │
│  2. Draft Loop (N steps)                         │
│     for step in range(speculative_num_steps):    │
│       input_ids = topk(logits)                   │
│       hidden_states = target_hidden              │
│       → MTP Model Forward:                       │
│         embed = embed_tokens(input_ids)           │
│         h = fc(norm(embed) ⊕ norm(hidden))       │
│         h = 1-layer-decoder(h)  ← Linear Attn!  │
│         logits = lm_head(h)                      │
│       → topk sampling for next step              │
│                                                  │
│  3. Verify (target extend with draft tokens)     │
│     → GDN: target_verify() with state rollback   │
│     → KDA: standard extend path (无专用 verify)   │
│     → Accept/reject draft tokens                 │
└─────────────────────────────────────────────────┘
```

---

## 2. MTP Draft 模型结构

### 2.1 Qwen3.5 MTP

文件：`python/sglang/srt/models/qwen3_5_mtp.py`

```python
class Qwen3_5ForCausalLMMTP(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        # 1. FC 投影层：拼接 [input_embed, hidden_states] → hidden_size
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.pre_fc_norm_embedding = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)

        # 2. 仅 1 层 decoder，强制 full_attention_interval=1
        config.num_hidden_layers = 1
        config.full_attention_interval = 1
        self.model = Qwen3_5ForCausalLM(config, quant_config, ..., is_nextn=True)

        # 3. LM Head（可能与 target 共享）
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, ...)
```

**Forward 流程**（`qwen3_5_mtp.py:132-189`）：

```python
def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
    input_embeds = forward_batch.mm_input_embeds
    # 多模态输入处理（extend + mm_inputs）
    if input_embeds is None:
        input_embeds = self.model.embed_tokens(input_ids)

    hidden_states = forward_batch.spec_info.hidden_states  # 来自 target 模型

    # 双路 RMSNorm → concat → fc 投影
    if not forward_batch.forward_mode.is_idle():
        input_embeds = self.pre_fc_norm_embedding(input_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
    hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
    hidden_states = self.fc(hidden_states)

    # 1 层 decoder forward（包含 linear attention + MLP/MoE）
    with get_global_expert_distribution_recorder().disable_this_region():
        hidden_states = self.model(input_ids, positions, forward_batch, hidden_states)

    return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
```

### 2.2 Qwen3-Next MTP

文件：`python/sglang/srt/models/qwen3_next_mtp.py`

结构与 Qwen3.5 MTP 几乎相同，区别在于继承自 `Qwen3NextForCausalLM` 而非独立模块：

```python
class Qwen3NextForCausalLMMTP(Qwen3NextForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        nn.Module.__init__(self)  # 注意：不调用 super().__init__()
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.pre_fc_norm_embedding = GemmaRMSNorm(...)
        self.pre_fc_norm_hidden = GemmaRMSNorm(...)
        config.num_hidden_layers = 1
        config.full_attention_interval = 1
        self.model = Qwen3NextModel(config, quant_config, ..., is_nextn=True)
        self.lm_head = ParallelLMHead(...)
```

### 2.3 模型架构自动切换

文件：`python/sglang/srt/configs/model_config.py:450-484`

当模型作为 draft model 加载时，架构名自动替换：

```python
if is_draft_model and self.hf_config.architectures[0] == "Qwen3NextForCausalLM":
    self.hf_config.architectures[0] = "Qwen3NextForCausalLMMTP"
    self.hf_config.num_nextn_predict_layers = 1

if is_draft_model and self.hf_config.architectures[0] in [
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "InternS2PreviewForConditionalGeneration",
]:
    self.hf_config.architectures[0] = "Qwen3_5ForCausalLMMTP"
    self.hf_config.num_nextn_predict_layers = 1
```

### 2.4 Kimi Linear — 无独立 MTP 文件

Kimi Linear（`kimi_linear.py`）目前**没有独立的 MTP draft 模型文件**。它通过 `num_nextn_predict_layers` 配置支持 MTP，但实际的 draft 模型加载取决于具体 checkpoint。Kimi K2.5 使用 EAGLE3 架构（`kimi_k25_eagle3.py`）。

---

## 3. Linear Attention 统一抽象层：RadixLinearAttention

文件：`python/sglang/srt/layers/radix_linear_attention.py`

所有 linear attention 变体（GDN、KDA）都通过 `RadixLinearAttention` 统一调度：

```python
class RadixLinearAttention(nn.Module):
    def __init__(self, layer_id, num_q_heads, num_k_heads, num_v_heads,
                 head_q_dim, head_k_dim, head_v_dim,
                 conv_weights=None, bias=None, activation="silu",
                 A_log=None, dt_bias=None):
        ...

    def forward(self, forward_batch, mixed_qkv, a, b):
        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
            # Extend/Prefill 路径
            output = torch.empty(...)
            unified_linear_attention_with_output(mixed_qkv, a, b, output, self.layer_id)
            return output
        else:
            # Decode 路径 — 委托给 attn_backend
            return forward_batch.attn_backend.forward(
                layer=self, forward_batch=forward_batch,
                mixed_qkv=mixed_qkv, a=a, b=b,
            )
```

关键参数：
- `mixed_qkv`：Q、K、V 拼接后的张量
- `a`：forget gate（GDN）或 gate（KDA）
- `b`：beta（插入新信息的权重）

`unified_linear_attention_with_output` 是一个注册了 `@register_custom_op` 和 `@register_split_op` 的函数，内部调用 `forward_batch.attn_backend.forward()`。

---

## 4. GDN 后端详解

文件：`python/sglang/srt/layers/attention/linear/gdn_backend.py`

### 4.1 GDNKernelDispatcher

```python
class GDNKernelDispatcher:
    def __init__(self, decode_backend, prefill_backend):
        # decode kernel 选择
        if decode_backend.is_triton():
            self.decode_kernel = TritonGDNKernel()
        elif decode_backend.is_cutedsl():
            self.decode_kernel = CuteDSLGDNKernel()
        elif decode_backend.is_flashinfer():
            self.decode_kernel = FlashInferGDNKernel()

        # extend/prefill kernel 选择
        if prefill_backend.is_triton():
            self.extend_kernel = TritonGDNKernel()
        elif prefill_backend.is_flashinfer():
            self.extend_kernel = FlashInferGDNKernel()
        # CuTeDSL 不支持 prefill

        # verify kernel：FlashInfer > Triton（CuTeDSL 不参与）
        if decode_backend.is_flashinfer() or prefill_backend.is_flashinfer():
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel
```

### 4.2 Decode 路径 (`forward_decode`)

```python
def forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    conv_states = layer_cache.conv[0]
    ssm_states = layer_cache.temporal
    cache_indices = self.forward_metadata.mamba_cache_indices

    # 1. 1D 因果卷积更新
    mixed_qkv = causal_conv1d_update(mixed_qkv, conv_states, layer.conv_weights, ...)

    # 2a. Packed decode 快速路径（如果 kernel 支持）
    if self.kernel_dispatcher.supports_packed_decode:
        core_attn_out = self.kernel_dispatcher.packed_decode(
            mixed_qkv=mixed_qkv, a=a, b=b,
            A_log=layer.A_log, dt_bias=layer.dt_bias, ...)
        return core_attn_out

    # 2b. 标准 decode 路径：split QKV → reshape → kernel
    query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    query = query.view(1, bs, num_q_heads, head_q_dim)
    key = key.view(1, bs, num_k_heads, head_k_dim)
    value = value.view(1, bs, num_v_heads, head_v_dim)

    core_attn_out = self.kernel_dispatcher.decode(
        q=query, k=key, v=value, a=a, b=b, ...)
    return core_attn_out
```

### 4.3 Extend/Prefill 路径 (`forward_extend`)

```python
def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    is_target_verify = forward_batch.forward_mode.is_target_verify()

    if is_target_verify:
        # === MTP VERIFY 路径 ===
        batch_size = seq_len // forward_batch.spec_info.draft_token_num
        draft_token_num = forward_batch.spec_info.draft_token_num

        # 1. 因果卷积：使用 intermediate_conv_window 支持树状回溯
        mixed_qkv_reshaped = mixed_qkv.view(batch_size, draft_token_num, -1).transpose(1, 2)
        mixed_qkv_processed = causal_conv1d_update(
            mixed_qkv_reshaped, conv_states, ...,
            intermediate_conv_window=intermediate_conv_window_cache,
            intermediate_state_indices=intermediate_state_indices[:batch_size],
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
        )
        mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
    else:
        # === 普通 EXTEND 路径 ===
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1), layer.conv_weights, ...,
            has_initial_state=has_initial_states, ...
        ).transpose(0, 1)[:seq_len]

    # Split QKV
    query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

    if is_target_verify:
        # 2a. Verify: 调用 target_verify kernel
        core_attn_out = self.kernel_dispatcher.target_verify(
            A_log=layer.A_log, dt_bias=layer.dt_bias,
            q=query, k=key, v=value, a=a, b=b,
            ssm_states=ssm_states, cache_indices=cache_indices,
            intermediate_states_buffer=intermediate_state_cache,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=draft_token_num,
            retrieve_parent_token=retrieve_parent_token,
        )
    else:
        # 2b. 普通 Extend: 调用 chunk 并行 prefill kernel
        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
            q=query, k=key, v=value, g=g, beta=beta, ...
        )
```

---

## 5. KDA 后端详解

文件：`python/sglang/srt/layers/attention/linear/kda_backend.py`

### 5.1 KDAKernelDispatcher

```python
class KDAKernelDispatcher:
    def __init__(self, decode_backend, prefill_backend):
        if decode_backend.is_triton():
            self.decode_kernel = TritonKDAKernel()
        elif decode_backend.is_cutedsl():
            self.decode_kernel = CuteDSLKDAKernel()
        # FlashInfer 不支持 KDA

        if prefill_backend.is_triton():
            self.extend_kernel = TritonKDAKernel()
        # CuTeDSL 不支持 prefill
```

**注意：KDA 没有 `target_verify` 方法**，`TritonKDAKernel` 和 `CuteDSLKDAKernel` 都没有实现 `target_verify`。

### 5.2 KDA Decode 路径

```python
def forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    conv_states = layer_cache.conv[0]  # 注意：需要 .transpose(-1, -2)
    ssm_states = layer_cache.temporal

    # 1. 因果卷积更新（注意 transpose 与 GDN 不同）
    qkv = causal_conv1d_update(
        mixed_qkv, conv_states.transpose(-1, -2),
        layer.conv_weights, layer.bias, activation="silu", ...
    )

    # 2. Split + reshape
    q, k, v = qkv.split([q_dim, k_dim, v_dim], dim=-1)
    q = q.unflatten(-1, (-1, head_q_dim)).unsqueeze(0)
    k = k.unflatten(-1, (-1, head_k_dim)).unsqueeze(0)
    v = v.unflatten(-1, (-1, head_v_dim)).unsqueeze(0)

    # 3. 调用 decode kernel（is_kda=True）
    return self.kernel_dispatcher.decode(q=q, k=k, v=v, a=a, b=b, ...)
```

### 5.3 KDA Extend 路径

与 GDN 的关键区别：**Q、K、V 分别做因果卷积**，而非拼接后一起做：

```python
def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    # 注意：没有 is_target_verify 分支！
    splits = [q_dim, k_dim, v_dim]
    q, k, v = mixed_qkv.transpose(0, 1).split(splits, dim=0)
    q_conv_weight, k_conv_weight, v_conv_weight = layer.conv_weights.split(splits, dim=0)
    q_conv_state, k_conv_state, v_conv_state = conv_states.split(splits, dim=-2)

    # Q、K、V 分别做因果卷积
    q = causal_conv1d_fn(q, q_conv_weight, q_bias, ...).transpose(0, 1)
    k = causal_conv1d_fn(k, k_conv_weight, k_bias, ...).transpose(0, 1)
    v = causal_conv1d_fn(v, v_conv_weight, v_bias, ...).transpose(0, 1)

    q = q.unflatten(-1, (-1, head_q_dim)).unsqueeze(0)
    k = k.unflatten(-1, (-1, head_k_dim)).unsqueeze(0)
    v = v.unflatten(-1, (-1, head_v_dim)).unsqueeze(0)

    # 调用 extend kernel（传 a 作为 g, b 作为 beta）
    core_attn_out = self.kernel_dispatcher.extend(
        q=q, k=k, v=v, g=a, beta=b, ...
    )
```

---

## 6. MTP Verify 路径深度分析

### 6.1 核心结论

**GDN verify 走的是 recurrent（decode 语义）kernel，而非 prefill kernel。**

| 路径 | 实际 kernel | 计算语义 |
|---|---|---|
| Decode | `fused_sigmoid_gating_delta_rule_update` (T=1) | 单步 recurrent update，写回 SSM state |
| Extend/Prefill | `chunk_gated_delta_rule` | chunk 并行 prefill，写回 SSM state |
| **Target Verify** | `fused_sigmoid_gating_delta_rule_update` (T=draft_len) | **多步 sequential recurrent**，不写回主 SSM state |

### 6.2 为什么 verify 必须用 recurrent 而非 prefill？

根本原因在于 **SSM 状态的不可并行性**：

1. **Prefill kernel**（`chunk_gated_delta_rule`）使用 **associative scan / chunk 并行**，假设序列的 token 之间是**线性串联**的。它通过并行化 chunk 内的计算来加速。

2. **Verify 场景**下，draft tokens 构成的是一棵**树**（topk>1 时），而非线性序列。树中同一层级的兄弟节点共享同一个父 SSM 状态，需要：
   - 在处理完一个分支后**回滚**到父节点状态
   - 再从父节点状态出发处理下一个分支
   - 这就是 `retrieve_parent_token` 和 `intermediate_states_buffer` 的作用

3. 这种树状回滚在 parallel scan 中无法表达，只能用 **sequential recurrent** 逐步处理。

### 6.3 Verify 的因果卷积处理

Verify 路径中因果卷积也需要特殊处理（`gdn_backend.py:383-402`）：

```python
if is_target_verify:
    batch_size = seq_len // forward_batch.spec_info.draft_token_num
    draft_token_num = forward_batch.spec_info.draft_token_num
    mixed_qkv_reshaped = mixed_qkv.view(batch_size, draft_token_num, -1).transpose(1, 2)
    # 使用 intermediate_conv_window 支持树状回溯
    mixed_qkv_processed = causal_conv1d_update(
        mixed_qkv_reshaped, conv_states, ...,
        intermediate_conv_window=intermediate_conv_window_cache,
        intermediate_state_indices=intermediate_state_indices[:batch_size],
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_parent_token=retrieve_parent_token,
    )
    mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
```

### 6.4 SSM 状态管理

Verify 使用 `MambaPool.SpeculativeState` 提供的中间状态缓存：

```python
if is_target_verify:
    assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
    intermediate_state_cache = mamba_cache_params.intermediate_ssm        # SSM 中间状态缓冲区
    intermediate_conv_window_cache = mamba_cache_params.intermediate_conv_window[0]  # 卷积窗口缓冲区
    intermediate_state_indices = self.verify_intermediate_state_indices
```

---

## 7. CuTeDSL Kernel 实现对比

### 7.1 GDN CuTeDSL Kernel

文件：`python/sglang/jit_kernel/cutedsl_gdn.py`（1495 行）

**架构**：4 个 `@cute.kernel` 变体

| Kernel | 格式 | 线程数 | 分块策略 | 状态加载 |
|---|---|---|---|---|
| `gdn_kernel_small_batch` | `(N, 1, ...)` dense | 128 threads / 4 warps | 8 blocks/state, `TILE_V_SMALL=16` | cpasync 预取 |
| `gdn_kernel_small_batch_varlen` | `(1, N, ...)` varlen | 128 threads / 4 warps | 8 blocks/state, `TILE_V_SMALL=16` | cpasync 预取 |
| `gdn_kernel_large_batch` | `(N, 1, ...)` dense | 256 threads / 8 warps | 1 block/state, `TILE_V=32` | cpasync 预取 |
| `gdn_kernel_large_batch_varlen` | `(1, N, ...)` varlen | 256 threads / 8 warps | 1 block/state, `TILE_V=32` | cpasync 预取 |

Batch size 阈值：`SMALL_BATCH_THRESHOLD = 32`

**GDN 核心计算**（`cutedsl_gdn.py:115-270`）：

```python
# Gate 计算：标量 (per-head)
r_A_log = A_log[i_hv]                    # [HV]
r_dt_bias = dt_bias[i_hv]                # [HV]
r_a = a[i_n, 0, i_hv]                    # [N, 1, HV] — 标量

# g = exp(-exp(A_log) * softplus(a + dt_bias))  — 标量，broadcast 到所有 K, V
r_g = exp(-exp(r_A_log) * softplus(r_a + r_dt_bias))
r_beta = sigmoid(r_b)                    # 标量

# Recurrent step
h_old = sData * r_g                       # r_g 是标量，乘以整个 h[K,V]
sum_hk = reduce(h_old * sK)              # 对 K 维度 reduce
v_new = (v - sum_hk) * r_beta
h_new = h_old + sK * v_new
o = reduce(h_new * sQ)

# 写回状态
h0_source[(pool_idx, i_hv, k_write, v_global_write)] = h_new  # K-major 布局
```

**状态布局**：`(pool_size, HV, K, V)` — **K-major**

**关键优化**：
- cpasync 预取（multi-stage）：在处理当前 v_tile 时预取下一个 v_tile 的 SSM 状态
- L2 norm 的 warp shuffle reduction
- `TILE_K = 128`，`TILE_V_SMALL = 16`，`TILE_V = 32`

### 7.2 KDA CuTeDSL Kernel

文件：`python/sglang/jit_kernel/cutedsl_kda.py`（1518 行）

**架构**：同样 4 个变体，但状态布局和门控方式不同。

**KDA 核心计算**（`cutedsl_kda.py:117-268`）：

```python
# Gate 计算：向量 (per-key-dimension)
r_A_log = A_log[i_hv]                    # [HV]
r_dt_bias_k = dt_bias[i_hv, tidx]        # [HV, K] — 向量
r_a_k = a[i_n, 0, i_hv, tidx]           # [N, 1, HV, K] — 向量

# g[k] = exp(-exp(A_log) * softplus(a[k] + dt_bias[k]))  — 向量，存入共享内存 sG
sG[tidx] = exp(-exp(r_A_log) * softplus(r_a_k + r_dt_bias_k))
r_beta = sigmoid(r_b)                    # 标量

# Recurrent step
sum_hk = reduce(sData * sG * sK)          # sG[K] 逐元素乘，额外一次乘法
v_new = (v - sum_hk) * r_beta
h_old = sData * sG                        # sG[K] 逐元素乘
h_new = h_old + sK * v_new
o = reduce(h_new * sQ)

# 写回状态
h0_source[(pool_idx, i_hv, v_global_write, k_write)] = h_new  # V-major 布局
```

**状态布局**：`(pool_size, HV, V, K)` — **V-major**（与 GDN 的 K-major 相反）

### 7.3 GDN vs KDA 核心差异总结

| 特性 | GDN | KDA |
|---|---|---|
| gate `a` 形状 | `[N, 1, HV]` 标量 | `[N, 1, HV, K]` 向量 |
| `dt_bias` 形状 | `[HV]` 标量 | `[HV, K]` 向量 |
| 遗忘门 `g` | 标量 `exp(-exp(A) * softplus(a+dt))` | 向量 `[K]`，存入 `sG` 共享内存 |
| `h *= g` | 标量广播 | 逐 K 维度乘 |
| SSM 状态布局 | K-major `(pool, HV, K, V)` | V-major `(pool, HV, V, K)` |
| 共享内存 | 无 sG | 需要额外 `sG[TILE_K]` |
| 状态加载方式 | cpasync 预取（multi-stage） | 直接协作加载循环 |
| V 维度对齐 | 无 bounds check | 需要 `if v_global < V` bounds check |
| 性能影响 | 标量门控更高效 | 向量门控多一次乘法 + 额外共享内存 |

### 7.4 CuTeDSL 的支持范围

两个 CuTeDSL kernel **只实现了 decode**，不支持 extend 和 target_verify：

```python
class CuteDSLGDNKernel(LinearAttnKernelBase):
    def extend(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLGDNKernel only supports decode")
    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLGDNKernel only supports decode")

class CuteDSLKDAKernel(LinearAttnKernelBase):
    def extend(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLKDAKernel only supports decode")
    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLKDAKernel only supports decode")
```

因此在 `GDNKernelDispatcher` 中：
- decode 可以走 CuTeDSL
- prefill 和 verify 始终走 Triton 或 FlashInfer

---

## 8. Triton Verify Kernel 逐行解析

文件：`python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`

### 8.1 函数签名

```python
def fused_sigmoid_gating_delta_rule_update(
    A_log, a, dt_bias, softplus_beta, softplus_threshold,
    q, k, v, b,
    initial_state_source, initial_state_indices,
    scale=None, use_qk_l2norm_in_kernel=False, cu_seqlens=None,
    is_kda=False,
    # Verify 专用参数
    disable_state_update=False,
    intermediate_states_buffer=None,
    intermediate_state_indices=None,
    cache_steps=None,
    retrieve_parent_token=None,
):
```

### 8.2 核心 Triton Kernel

```python
@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log, a, dt_bias, softplus_beta, softplus_threshold,
    q, k, v, b, o, h0_source, h0_indices, cu_seqlens,
    # Verify 参数
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    # 常规参数
    scale, T, stride_a, stride_q, stride_k, stride_v, stride_b,
    NP2_T: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    # Verify constexpr 开关
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
):
```

### 8.3 核心循环详解

```python
# 加载初始 SSM 状态
b_h = tl.zeros([BK, BV], dtype=tl.float32)
if USE_INITIAL_STATE:
    idx = tl.load(h0_indices + i_n)
    if idx >= 0:
        p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + ...
        b_h += tl.load(p_h0, mask=mask_h, other=0)

# 加载树注意力数据（topk>1 时）
if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
    token_indices = tl.arange(0, NP2_T)
    parent_idx_tokens = tl.load(retrieve_parent_token_base, mask=mask_retrieve)

# 准备中间状态缓存索引
cache_idx = -1
if CACHE_INTERMEDIATE_STATES:
    cache_idx = tl.load(intermediate_state_indices + i_n)

step_idx = 0
for _ in range(0, T):  # T = draft_token_num (verify) 或 1 (decode)
    # ===== 树注意力：回滚到父节点状态 =====
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        if step_idx != 0 and cache_idx >= 0:
            parent_step_idx = tl.sum(tl.where(token_indices == step_idx, parent_idx_tokens, 0))
            step_offset = parent_step_idx * HV * K * V
            cache_ptr = intermediate_states_buffer + cache_idx * cache_steps * HV * K * V + ...
            b_h = tl.load(cache_ptr, mask=mask_h, other=0)  # 回滚！

    # ===== 加载当前 token 的 QKV =====
    b_q = tl.load(p_q, mask=mask_k, other=0)
    b_k = tl.load(p_k, mask=mask_k, other=0)
    b_v = tl.load(p_v, mask=mask_v, other=0)
    b_b = tl.load(p_b)

    # ===== 计算 gate 和 beta =====
    b_A_log = tl.load(p_A_log)
    if IS_KDA:
        b_a = tl.load(p_a, mask=mask_k, other=0)       # KDA: 向量 gate
        b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0)
    else:
        b_a = tl.load(p_a)                               # GDN: 标量 gate
        b_dt_bias = tl.load(p_dt_bias)

    # g = -exp(A_log) * softplus(a + dt_bias)
    x = b_a + b_dt_bias
    softplus_x = ...
    b_g = -tl.exp(b_A_log) * softplus_x
    b_beta = sigmoid(b_b)

    # ===== L2 归一化 =====
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / sqrt(sum(b_q * b_q) + 1e-6)
        b_k = b_k / sqrt(sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    # ===== Delta Rule Recurrent Update =====
    if IS_KDA:
        b_h *= tl.exp(b_g[:, None])    # KDA: 向量 gate，逐 K 维度
    else:
        b_h *= tl.exp(b_g)             # GDN: 标量 gate，广播
    b_v -= tl.sum(b_h * b_k[:, None], 0)   # v -= h @ k
    b_v *= b_beta                            # v *= beta
    b_h += b_k[:, None] * b_v[None, :]      # h += k * v
    b_o = tl.sum(b_h * b_q[:, None], 0)     # o = h @ q
    tl.store(p_o, b_o, mask=mask_v)

    # ===== 缓存中间状态（verify 专用）=====
    if CACHE_INTERMEDIATE_STATES:
        if cache_idx >= 0:
            step_offset = step_idx * HV * K * V
            cache_ptr = intermediate_states_buffer + cache_idx * cache_steps * HV * K * V + ...
            tl.store(cache_ptr, b_h, mask=mask_h)   # 保存 h 供后续回滚

    step_idx += 1
    # 更新指针
    p_q += stride_q; p_k += stride_k; p_v += stride_v; p_b += stride_b; p_o += HV * V; p_a += stride_a

# ===== 写回最终状态 =====
if not DISABLE_STATE_UPDATE:
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            tl.store(p_h0, b_h, mask=mask_h)   # decode: 写回；verify: 不写回
```

### 8.4 Verify vs Decode 的 Triton Constexpr 差异

| 参数 | Decode | Target Verify |
|---|---|---|
| `T` | 1 | `draft_token_num`（如 5） |
| `DISABLE_STATE_UPDATE` | `False` | `True`（不修改主 SSM pool） |
| `CACHE_INTERMEDIATE_STATES` | `False` | `True`（缓存每步 h 供树回滚） |
| `HAS_EAGLE_TREE_CUSTOM_ATTN_MASK` | `False` | `True`（topk>1 时启用树注意力） |
| `intermediate_states_buffer` | `None` | 非空 tensor |
| `retrieve_parent_token` | `None` | 非空 tensor（topk>1 时） |

---

## 9. FlashInfer GDN MTP Kernel

文件：`python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`

### 9.1 FlashInfer GDN Kernel 概述

```python
class FlashInferGDNKernel(LinearAttnKernelBase):
    """FlashInfer kernel for GDN with K-last SSM state layout.
    SM90 (Hopper): decode, prefill, MTP verify supported.
    SM100+ (Blackwell+): decode-only with bf16 state.
    """
```

### 9.2 MTP Verify 实现

```python
def target_verify(self, A_log, dt_bias, q, k, v, a, b, *,
                  ssm_states, cache_indices, query_start_loc,
                  intermediate_states_buffer, intermediate_state_indices,
                  cache_steps, retrieve_parent_token, **kwargs):
    # SM100+ 不支持
    if self.use_state_pool:
        raise NotImplementedError("FlashInfer GDN MTP verify is not yet supported on SM100+.")

    # topk>1 不支持（retrieve_parent_token 必须为 None）
    if retrieve_parent_token is not None:
        raise RuntimeError("FlashInfer GDN verify kernel only supports topk=1")

    # Reshape 为 [batch, draft_token_num, heads, dim]
    query_mtp = q.view(batch_size, draft_token_num, num_heads, head_k_dim)
    key_mtp = k.view(batch_size, draft_token_num, num_heads, head_k_dim)
    value_mtp = v.view(batch_size, draft_token_num, num_v_heads, head_v_dim)
    a_mtp = a.view(batch_size, draft_token_num, num_v_heads)
    b_mtp = b.view(batch_size, draft_token_num, num_v_heads)

    # 调用 FlashInfer 专用 MTP kernel
    output_fi, _ = self._mtp_fn(
        q=query_mtp, k=key_mtp, v=value_mtp,
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        A_log=A_log.detach(), a=a_mtp, dt_bias=dt_bias.detach(), b=b_mtp,
        intermediate_states_buffer=intermediate_states_buffer,
        disable_state_update=True,       # 不修改主 SSM pool
        use_qk_l2norm=True,
    )
    return output_fi.view(1, seq_len, num_v_heads, head_v_dim)
```

### 9.3 FlashInfer vs Triton Verify 对比

| 特性 | Triton | FlashInfer |
|---|---|---|
| topk>1 树注意力 | 支持（`retrieve_parent_token`） | 不支持（仅 topk=1） |
| SM100+ 支持 | N/A（Triton 无此限制） | 不支持 |
| SSM 状态布局 | K-last `[pool, HV, V, K]` | K-last `[pool, HV, V, K]`（SM90） |
| 输入 reshape | `[1, seq_len, H, D]` | `[batch, draft_len, H, D]` |
| 中间状态管理 | kernel 内部 `tl.store` | `intermediate_states_buffer` 参数 |
| Kernel 来源 | SGLang 自实现 | `flashinfer.gdn_decode.gated_delta_rule_mtp` |

---

## 10. 完整调度矩阵

| | Decode | Extend/Prefill | Target Verify (MTP) |
|---|---|---|---|
| **GDN Triton** | `fused_sigmoid_gating_delta_rule_update` (T=1) | `chunk_gated_delta_rule` | `fused_sigmoid_gating_delta_rule_update` (T=draft_len, tree rollback) |
| **GDN FlashInfer** | `gated_delta_rule_decode_pretranspose` | `chunk_gated_delta_rule` | `gated_delta_rule_mtp` (专用 MTP kernel, topk=1 only) |
| **GDN CuTeDSL** | `cutedsl_fused_sigmoid_gating_delta_rule_update` | N/A | N/A |
| **KDA Triton** | `fused_sigmoid_gating_delta_rule_update` (is_kda=True) | `chunk_kda` | **不支持** |
| **KDA CuTeDSL** | `cutedsl_fused_sigmoid_gating_kda_update` | N/A | **不支持** |

### Verify Kernel 选择逻辑

```
GDNKernelDispatcher.__init__():
    if flashinfer_available:
        self.verify_kernel = FlashInferGDNKernel  # 优先 FlashInfer
    else:
        self.verify_kernel = TritonGDNKernel       # 回退 Triton
    # CuTeDSL 不参与 verify
```

---

## 11. Kimi Linear 混合架构

文件：`python/sglang/srt/models/kimi_linear.py`

### 11.1 混合层结构

```python
class KimiDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, ...):
        if config.is_kda_layer(layer_idx):
            self.self_attn = KimiDeltaAttention(...)   # KDA linear attention
        else:
            self.self_attn = KimiMLAAttention(...)      # DeepSeek-V2 MLA attention
```

层类型由 `KimiLinearConfig.linear_attn_config` 决定：

```python
class KimiLinearConfig(PretrainedConfig):
    linear_attn_config: dict  # 包含 kda_layers, full_attn_layers, num_heads, head_dim, short_conv_kernel_size
```

### 11.2 KimiDeltaAttention 的融合投影

```python
class KimiDeltaAttention(nn.Module):
    def __init__(self, ...):
        if self.do_fuse_qkvbfg:
            # 融合路径：一次性计算 Q, K, V, beta, f_a, g_a
            self.fused_qkvbfg_a_proj = MergedColumnParallelRepeatedLinear(
                hidden_size,
                qkvb_sizes=[projection_size, projection_size, projection_size, num_heads],  # Column parallel
                fg_sizes=[head_dim, head_dim],  # Replicated: f_a, g_a
            )
            self.fused_fg_b_proj = ColumnParallelBatchedLinear(2, head_dim, projection_size)
        else:
            # 非融合路径：分开的投影
            self.qkv_proj = QKVParallelLinear(...)
            self.f_a_proj = ReplicatedLinear(...)
            self.f_b_proj = ColumnParallelLinear(...)
            self.b_proj = ColumnParallelLinear(...)
            self.g_a_proj = ReplicatedLinear(...)
            self.g_b_proj = ColumnParallelLinear(...)
```

### 11.3 Kimi Linear 的 MTP 状态

- Kimi Linear 目前**没有独立的 MTP draft 模型文件**
- `KimiLinearConfig` 有 `num_nextn_predict_layers` 字段，但 SGLang 中未看到 `KimiLinearForCausalLMNextN` 或类似实现
- Kimi K2.5 使用 EAGLE3 架构（`kimi_k25_eagle3.py`），基于 DeepSeek-V2 MLA 注意力
- KDA 后端**没有 `target_verify` 实现**，意味着 Kimi Linear 的 linear attention 层在 verify 时走标准 extend 路径

---

## 12. HybridLinearAttnBackend 调度逻辑

文件：`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

混合模型（如 Qwen3-Next、Qwen3.5、Kimi Linear）同时包含 full attention 层和 linear attention 层。`HybridLinearAttnBackend` 负责根据 `layer_id` 分发到不同的后端：

```python
class HybridLinearAttnBackend(AttentionBackend):
    def __init__(self, full_attn_backend, linear_attn_backend, full_attn_layers):
        self.full_attn_layers = full_attn_layers
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend

    def forward_decode(self, layer, forward_batch, ..., mixed_qkv=None, a=None, b=None, **kwargs):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.forward_decode(q, k, v, layer, forward_batch, ...)
        return self.linear_attn_backend.forward_decode(
            q=q, k=k, v=v, layer=layer, forward_batch=forward_batch,
            mixed_qkv=mixed_qkv, a=a, b=b, **kwargs)

    def forward_extend(self, layer, forward_batch, ..., mixed_qkv=None, a=None, b=None, **kwargs):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.forward_extend(q, k, v, layer, forward_batch, ...)
        return self.linear_attn_backend.forward_extend(
            q=q, k=k, v=v, layer=layer, forward_batch=forward_batch,
            mixed_qkv=mixed_qkv, a=a, b=b, **kwargs)
```

### Verify 时的 metadata 处理

```python
# hybrid_linear_attn_backend.py:180-187
if forward_batch.forward_mode.is_target_verify():
    query_start_loc = torch.arange(
        0, forward_batch.input_ids.shape[0] + 1,
        step=forward_batch.spec_info.draft_token_num,
        dtype=torch.int32, device=forward_batch.input_ids.device,
    )
    if self.topk > 1:
        retrieve_next_token = forward_batch.spec_info.retrieve_next_token
        retrieve_next_sibling = forward_batch.spec_info.retrieve_next_sibling
        if retrieve_next_token is not None:
            retrieve_parent_token = torch.empty_like(retrieve_next_token)
```

---

## 13. 关键文件索引

### Speculative 解码框架

| 文件 | 描述 |
|---|---|
| `python/sglang/srt/speculative/spec_info.py` | 算法枚举、SpecInput 类型定义 |
| `python/sglang/srt/speculative/eagle_worker.py` | EAGLE Worker（GDN/KDA 模型使用） |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | EAGLE Worker V2（overlap 调度） |
| `python/sglang/srt/speculative/frozen_kv_mtp_worker.py` | Frozen-KV MTP Worker（Gemma4 使用） |
| `python/sglang/srt/speculative/eagle_info.py` | EagleDraftInput/EagleVerifyInput 数据结构 |
| `python/sglang/srt/speculative/spec_utils.py` | fast_topk、select_top_k_tokens 等工具函数 |

### MTP Draft 模型

| 文件 | 描述 |
|---|---|
| `python/sglang/srt/models/qwen3_5_mtp.py` | Qwen3.5 MTP draft 模型 |
| `python/sglang/srt/models/qwen3_next_mtp.py` | Qwen3-Next MTP draft 模型 |
| `python/sglang/srt/models/kimi_linear.py` | Kimi Linear 模型（KDA + MLA 混合） |
| `python/sglang/srt/models/kimi_k25_eagle3.py` | Kimi K2.5 EAGLE3 draft 模型 |
| `python/sglang/srt/models/qwen3_5.py` | Qwen3.5 主模型（GDN linear attention） |
| `python/sglang/srt/models/qwen3_next.py` | Qwen3-Next 主模型（GDN linear attention） |

### Linear Attention 核心

| 文件 | 描述 |
|---|---|
| `python/sglang/srt/layers/radix_linear_attention.py` | 统一 Linear Attention 抽象层 |
| `python/sglang/srt/layers/attention/linear/gdn_backend.py` | GDN 后端（decode/extend/verify 调度） |
| `python/sglang/srt/layers/attention/linear/kda_backend.py` | KDA 后端（decode/extend 调度，无 verify） |
| `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | 混合后端（full + linear 分发） |

### Linear Attention Kernel

| 文件 | 描述 |
|---|---|
| `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py` | GDN Triton kernel（decode + extend + target_verify） |
| `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` | GDN FlashInfer kernel（decode + extend + MTP verify） |
| `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py` | GDN CuTeDSL kernel（仅 decode） |
| `python/sglang/srt/layers/attention/linear/kernels/kda_triton.py` | KDA Triton kernel（decode + extend，无 verify） |
| `python/sglang/srt/layers/attention/linear/kernels/kda_cutedsl.py` | KDA CuTeDSL kernel（仅 decode） |
| `python/sglang/srt/layers/attention/linear/kernels/kernel_backend.py` | LinearAttnKernelBase 抽象基类 |

### Triton Kernel 底层实现

| 文件 | 描述 |
|---|---|
| `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` | GDN/KDA 共用的 recurrent kernel（decode + verify） |
| `python/sglang/srt/layers/attention/fla/fused_recurrent.py` | packed decode kernel |
| `python/sglang/srt/layers/attention/fla/kda.py` | KDA chunk prefill kernel |
| `python/sglang/srt/layers/attention/fla/chunk.py` | GDN chunk prefill kernel |
| `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` | GDN gating 融合 kernel |

### CuTeDSL JIT Kernel

| 文件 | 描述 |
|---|---|
| `python/sglang/jit_kernel/cutedsl_gdn.py` | GDN CuTeDSL decode kernel（1495 行） |
| `python/sglang/jit_kernel/cutedsl_kda.py` | KDA CuTeDSL decode kernel（1518 行） |

### 配置与调度

| 文件 | 描述 |
|---|---|
| `python/sglang/srt/configs/kimi_linear.py` | KimiLinearConfig（含 linear_attn_config） |
| `python/sglang/srt/configs/qwen3_next.py` | Qwen3NextConfig（含 HybridLayerType） |
| `python/sglang/srt/configs/qwen3_5.py` | Qwen3_5TextConfig（继承 Qwen3NextConfig） |
| `python/sglang/srt/configs/model_config.py` | 模型架构自动切换逻辑 |
| `python/sglang/srt/server_args.py` | `--speculative-algorithm` 解析 |

### 测试文件

| 文件 | 描述 |
|---|---|
| `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py` | Qwen3-Next MTP 测试 |
| `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py` | Qwen3.5 FP4 MTP V2 测试 |
| `test/registered/models/test_kimi_linear_models.py` | Kimi Linear 测试 |
| `test/registered/8-gpu-models/test_kimi_k25.py` | Kimi K2.5 测试 |