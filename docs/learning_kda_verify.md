# KDA Verify Kernel 学习文档

本文档记录 Phase 2 CuTe DSL KDA verify kernel 的设计思路、实现细节和性能分析。

## 1. Verify vs Decode：问题定义

### 1.1 Decode kernel（已有）

单 token 更新：给定 `(q, k, v, a, b)` 和 `state[N, HV, V, K]`，执行一步 delta-rule recurrent update：

```
gate = exp(-A * softplus(a + dt_bias))        # (K,) per-dim decay
Hk = state @ (gate * k)                        # (V,) 投影
v_new = sigmoid(b) * (v - Hk)                  # (V,) delta
state_new = gate * state + v_new ⊗ k           # (V, K) 更新
o = state_new @ q                               # (V,) 输出
```

每次调用处理 1 个 token，state 从 GMEM 加载并写回。

### 1.2 Verify kernel（Phase 2 新增）

对 T 个 draft token 做 sequential recurrent update，**在同一个 kernel launch 内完成**：

```python
for t in range(T):
    # 每步复用 sData（SMEM 中的 state）
    reload q[t], k[t], a[t], b[t]
    compute gate, delta-rule update on sData
    write output o[n, t, hv, v]
    write snapshot → intermediate_states_buffer[n, t, hv, ...]
```

核心优势：state 在 SMEM 跨 token 保持，省去 T-1 次 GMEM 往返。

## 2. 循环嵌套设计

### 2.1 为什么是 `for v_tile → for t`

sData 只持有一个 V-tile（128×16 for small_batch, 128×32 for large_batch），无法同时容纳全部 V tiles。delta-rule 在 V tiles 间**无依赖**（每个 V-tile 独立更新），所以：

```
for hv_offset:                        # (仅 non-hv-parallel 路径)
    for v_tile in range(num_v_tiles):
        load h0_source[v_tile] → sData   # 仅 t=0
        for t in range(max_steps):       # ← 内层 token 循环
            reload q[t], k[t], a[t], b[t] → sK/sQ/sG
            compute gate, L2-norm
            load v[t], delta-rule update on sData
            write output o[n, t, hv, v]
            write snapshot
        if not disable_state_update:
            write sData → h0_source
```

`for v_tile → for t` 与 `for t → for v_tile` 数学等价，但前者让 sData 跨 token 保持。

### 2.2 Trade-off

- **省**：T-1 次 state GMEM round-trip（每次 ~8KB per V-tile，T=5 省 ~32KB）
- **费**：每 token 重载 q/k/a/b 到 SMEM（~1KB/token），但这远小于 state 流量

## 3. 四个 Kernel 变体

| 变体 | 线程数 | TILE_V | V-tile 循环 | Grid | 适用场景 |
|------|--------|--------|-------------|------|---------|
| small_batch | 128 (4 warps) | 16 | CTA 内循环 | N×(H or HV)×blocks | N < 1024 |
| small_batch_varlen | 128 | 16 | CTA 内循环 | N×HV×blocks | varlen 输入 |
| large_batch | 256 (8 warps) | 32 | 1 CTA per tile | N×HV×num_v_tiles | N ≥ 1024 |
| large_batch_varlen | 256 | 32 | 1 CTA per tile | N×HV×num_v_tiles | N ≥ 1024, varlen |

### 3.1 Small vs Large batch

- **Small batch**: 每 CTA 处理多个 V-tiles（循环），吞吐率更低但 launch 开销小
- **Large batch**: 每 CTA 只处理一个 V-tile，充分利用 SM 并行度
- 阈值 `N = 1024`（`SMALL_BATCH_THRESHOLD`）

### 3.2 Dense vs Varlen

- **Dense**: q shape `(N, T, H, K)`，a shape `(N, T, HV, K)`
- **Varlen**: q shape `(1, N*T, H, K)`，a shape `(N*T, HV, K)`
- 索引差异：`i_n * max_steps + t` vs `(i_n, t, ...)`
- `intermediate_states_buffer` 始终 dense layout `(N, T, HV, V, K)`

## 4. SMEM 布局

复用 decode kernel 的 SMEM 分配，不增加新区域：

### Small batch (128 threads, TILE_V_SMALL=16)
```
sData: (128, 16, 2) float32 = 128 × 18(padded) × 2 = ~18KB
sK:    (128,)       float32 = 512B
sQ:    (128,)       float32 = 512B
sG:    (128,)       float32 = 512B
sGK:   (128,)       float32 = 512B
smem_o:(16,)        float32 = 64B
Total: ~20KB << 228KB SM90 SMEM
```

### Large batch (256 threads, TILE_V=32)
```
sData: (128, 32, 2) float32 = 128 × 36(padded) × 2 = ~37KB
sK:    (128,)       float32 = 512B
sQ:    (128,)       float32 = 512B
sG:    (128,)       float32 = 512B
smem_o:(32,)        float32 = 128B
Total: ~39KB << 228KB SM90 SMEM
```

## 5. Barrier 安排

每个 token step `t` 内的同步点：

1. **q/k load 后** — 确保 sK/sQ 写入完成，才能读取做 L2-norm
2. **L2-norm 后** — 确保 norm 结果写入 smem_o，才能读取做缩放
3. **sGK 计算后**（仅 small_batch）— 确保 sG*sK 写入 sGK，才能读取做 delta-rule
4. **delta-rule 后** — 确保 sData 更新完成，才能写 snapshot
5. **token 结束** — 确保 snapshot 写出完成（或 sData 不再被其它线程读取），才能进入下一 token 重写 sK/sQ/sG

## 6. Snapshot 存储

Phase 2 使用**同步 store**（register → GMEM），与 decode kernel 的 state write-back 模式一致。

每个 token 结束后将 sData 写到 `intermediate_states_buffer[is_idx, t, i_hv, ...]`。写入线程映射复用 state load 的线程分工（确保 coalesced GMEM access）。

`cp.async` (TMA S2G) 异步优化留给 Phase 3。

## 7. 编译变体管理

### 7.1 Cache key

在 decode 的 16-tuple key 基础上追加：`use_small_batch`、`is_varlen`、`max_steps`、`cache_intermediate_states`、`disable_state_update` → 18-tuple key。

### 7.2 变体爆炸

`max_steps ∈ {1,2,4,5,8}` × 其余 key 组合。首次编译每个 `max_steps` 值 ~秒级，后续命中 cache。这与 decode kernel 处理 `scale` 等 constexpr 的方式一致。

## 8. 文件组织

- `cula/ops/kda_verify_cute.py` — 4 个 kernel + 4 个 JIT launcher + 编译缓存 + public API
- `cula/ops/kda_verify_triton.py` — Phase 1 Triton baseline（参考实现）
- `cula/ops/__init__.py` — 导出 `kda_verify`（默认 CuTe DSL 后端）

与 decode 不同的是 verify kernel 独立成文件（而非追加到 `kda_decode.py`），因为：
1. verify 和 decode 的循环结构不同（T token 循环 vs 单 token）
2. 避免 `kda_decode.py` 膨胀到 3000+ 行
3. 独立的编译缓存，不污染 decode 的缓存命中

## 9. Phase 3 设计：KV Cache vs Full State Snapshot

### 9.1 问题

Full state snapshot 每步写 `HV × V × K × 4B` = 8MB (HV=128, V=K=128)。5 个 draft token 就是 40MB。这是 verify kernel 的主要带宽瓶颈。

### 9.2 方案对比

三种 accept 后 state 恢复策略：

| | Full state snapshot | No cache + decode×T | **KV cache + linear scan** |
|---|---|---|---|
| verify 每步额外写入 | 8MB | 0 | **132KB** |
| accept 后恢复延迟 | ~5μs (gather) | ~150μs (T launches) | **~30μs** (1 launch) |
| 实现复杂度 | 已有 | 已有 | 新 recovery kernel |

### 9.3 KV cache 模式原理

verify 每步计算的中间变量中，只有三个需要缓存：

```
v_new_t = sigmoid(b_t) * (v_t - state_t @ (gate_t * k_norm_t))   # (HV, V)
gate_t  = exp(-A * softplus(a_t + dt_bias))                       # (HV, K)
k_norm_t = l2norm(k_t)                                            # (H, K)
```

恢复 state 变成无依赖的线性扫描：
```
state = initial_state
for t in range(accepted_len):
    state = gate_t * state + v_new_t[:, :, None] * k_norm_t[None, :, :]
```

**关键洞察**：v_new 在 verify kernel 里已经在寄存器中算出，写出只需几条 store 指令。恢复时不需要 L2norm / softplus / sigmoid / delta correction — 比 kda_decode 还简单。

### 9.4 与社区 chunk-based 方案的比较

社区讨论中提到的 chunk-based verify（用 chunk_fwd 替代 recurrent）需要实现 intra-chunk quadratic attention（O(T² × HV × K) 的 QK^T），工程量大。KV cache 方案绕过了这个复杂度，同时在带宽上更优。
