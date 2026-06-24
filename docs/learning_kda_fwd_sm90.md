# KDA Forward Kernel SM90 深度学习指南

## 0. GPU 硬件与 CUDA 编程模型基础 (SM90 / Hopper)

### 0.1 硬件层次

```
GPU
 └─ GPC (Graphics Processing Cluster)
     └─ TPC (Texture Processing Cluster)
         └─ SM (Streaming Multiprocessor)  ← 核心执行单元
             ├─ 寄存器文件 (Register File): 每 SM 64K 个 32-bit 寄存器
             ├─ 共享内存 (Shared Memory / SMEM): 每 SM 最大 228 KB
             ├─ L1 缓存
             ├─ Tensor Core (WGMMA 单元): 执行矩阵乘法
             └─ TMA 单元 (Tensor Memory Accelerator): 硬件级异步数据搬运
```

**关键数字 (SM90/Hopper):**
| 资源 | 规格 |
|------|------|
| 寄存器/SM | 65536 个 32-bit |
| 寄存器/线程 | 最大 255 个 |
| 共享内存/SM | 最大 228 KB |
| Warp 大小 | 32 线程 |
| Warp Group | 4 个 Warp = 128 线程 |
| 最大线程数/Block | 1024 |

### 0.2 CUDA 软件层次 → 硬件映射

```
Grid (所有block的集合)        ← 映射到整个 GPU
 └─ Cluster (SM90 新增)       ← 多个 SM 间可协作
     └─ Block (线程块)         ← 映射到一个 SM
         └─ Warp Group (128线程)  ← 4个warp, WGMMA的最小单位
             └─ Warp (32线程)      ← SM 调度的基本单位
                 └─ Thread (线程)   ← 执行的最小单位
```

### 0.3 存储层次

```
Global Memory (GMEM, HBM)  ~80GB, 高延迟(~400 cycles), 高带宽(~3TB/s)
     ↕  TMA (硬件异步搬运)
Shared Memory (SMEM)        ~228KB/SM, 低延迟(~30 cycles)
     ↕  S2R/R2S 指令
Register File               64K/SM, 最低延迟(~1 cycle)
```

### 0.4 关键 SM90 特性

- **TMA (Tensor Memory Accelerator)**: 硬件加速的 GMEM ↔ SMEM 数据搬运，由专用硬件单元执行，不消耗计算线程
- **WGMMA (Warp Group Matrix Multiply-Accumulate)**: 以 Warp Group (128线程) 为粒度的 Tensor Core 矩阵乘
- **异步 Pipeline**: Producer-Consumer 模型，Load 和 Compute 可以流水线重叠
- **Named Barriers**: 细粒度的 warp group 间同步
- **寄存器重分配 (Register Reconfig)**: 运行时动态调整每个 warp group 的寄存器数量

### 0.5 Tiling 策略

KDA kernel 将输入矩阵切分成 Tile 进行计算：

```
TileShape = (BlkSeqQ, BlkSeqKV, HeadSize)
            典型值: (64, 64, 128) 或 (128, 128, 128)

Q: [total_seqlen, d] → 沿 seqlen 方向切分成 BlkSeqQ 大小的 tile
K: [total_seqlen, d] → 沿 seqlen 方向切分成 BlkSeqKV 大小的 tile
V: [total_seqlen, d] → 同 K
O: [total_seqlen, d] → 输出 tile, 同 Q
```

### 0.6 维度体系全景图

以下以 `TileShape = (64, 64, 128)` 即 `BlkSeqQ=64, BlkSeqKV=64, HeadSize=128` 为例，自顶向下建立完整的维度参考。代码中 `HeadSizeQK = HeadSizeV = HeadSize = 128`，`BlkSeqQ == BlkSeqKV == 64`。

#### 0.6.1 层级一: 全局张量维度 (GMEM)

这些是整个 kernel 的输入/输出张量，存储在 HBM 中。TMA descriptor 按此 shape 创建。

| 张量 | Shape | 元素类型 | Layout 类型 | 维度含义 | 代码定义 |
|------|-------|---------|------------|---------|---------|
| Q | (total_seqlen, head_size, num_heads) | bf16 | LayoutQ = (s, d, h) | packed varlen 所有序列拼接 | `mainloop:83` |
| K | (total_seqlen, head_size, num_heads) | bf16 | LayoutK = (s, d, h) | 同 Q，但 TMA 取 tile 时转置为 (d, s) | `mainloop:84` |
| V | (total_seqlen, head_size, num_heads) | bf16 | LayoutV = (s, d, h) | 同 K | `mainloop:85` |
| O | (total_seqlen, head_size, num_heads) | bf16 | LayoutO = (s, d, h) | 输出，TMA store 为 (d, s) 视角 | `mainloop:86` |
| Alpha | (total_seqlen, head_size, num_heads) | float | LayoutAlpha = (s, d, h) | channel-wise gate, 每 token 每维 | `mainloop:87` |
| Beta | (total_seqlen, num_heads) | bf16/float | GmemLayoutBeta = (s, h) | scalar gate, 每 token 一个 | `mainloop:378` |
| State | (dk=128, dv=128, num_heads, num_seqs) | float | LayoutLeft (列优先) | 隐状态矩阵, 全精度 | `mainloop:474` |

**关键**: K/V 在 GMEM 中存储为 (seqlen, d)，但 TMA 取 tile 时使用 `select<1,0,2>(LayoutK{})` 把视角转为 (d, seqlen)。这样 SMEM 中 K/V 是列连续的，满足 WGMMA B 操作数列优先的要求。

**Q 和 K/V 的 TMA tile box 不同**:
```
Q/Alpha TMA box:  (BlkSeqQ, HeadSize) = (64, 128)    — 行方向取 64 token, 取完整 head_size
K/V TMA box:      (HeadSize, BlkSeqKV) = (128, 64)    — 转置视角, 取完整 head_size × 64 token
```

#### 0.6.2 层级二: Tile 级 MMA 维度

每个 TileShape 定义了一次 WGMMA 的 `(M, N, K)` 维度。CUTLASS 的 WGMMA 语义为: **A(M×K) @ B(K×N) → C(M×N)**。

```
代码定义 (mainloop_kda_fwd.hpp L135-152):

BlkSeqQ     = 64       ← Q 的 token 数 (序列方向)
BlkSeqKV    = 64       ← K/V 的 token 数 (序列方向)
HeadSize    = 128      ← head dimension (dk = dv)
HeadSizeQK  = 128      ← 等于 HeadSize
HeadSizeV   = 128      ← 等于 HeadSize
HeadSizeHalf = 64      ← HeadSize / 2
HeadSizeQuar = 32      ← HeadSize / 4
```

| TileShape | 代码定义 | (M, N, K) | A 矩阵 | B 矩阵 | C 累加器 | 使用 Step |
|-----------|---------|-----------|--------|--------|---------|----------|
| TileShapeQK | (64, 64, 128) | BlkSeqQ × BlkSeqKV × HeadSize | Q (64×128) | K^T (128×64) | QK (64×64) | compute_aux_safe |
| TileShapeKK | (64, 64, 128) | BlkSeqKV × BlkSeqKV × HeadSize | K (64×128) | K^T (128×64) | KK (64×64) | compute_aux_safe |
| TileShapeO1 | (128, 64, 128) | HeadSizeV × BlkSeqQ × HeadSizeQK | State (128×128) | Q_scaled (128×64) | O_inter (128×64) | Step B |
| TileShapeO2 | (128, 64, 64) | HeadSizeV × BlkSeqQ × BlkSeqKV | NewV^T (128×64) | QK (64×64) | O_intra (128×64) | Step D |
| TileShapeSK | (128, 64, 128) | HeadSizeV × BlkSeqKV × HeadSizeQK | State (128×128) | K_scaled (128×64) | SK (128×64) | Step C1 |
| TileShapeNewV | (128, 64, 64) | HeadSizeV × BlkSeqKV × BlkSeqKV | V' (128×64) | T^{-T} (64×64) | NewV (128×64) | Step C4 |
| TileShapeKV | (128, 128, 64) | HeadSizeV × HeadSizeQK × BlkSeqKV | NewV^T (128×64) | K_decay^T (64×128) | State (128×128) | Step F |
| TileShapeQK_Half | (64, 64, 64) | 同 QK 但 K 维减半 | — | — | — | (未使用) |
| TileShapeQK_Quar | (64, 64, 32) | 同 QK 但 K 维 1/4 | Q (64×32) | — | — | Step A/E prologue |

**各 Step 的 WGMMA 矩阵乘维度汇总 (代入 d=128, C=64)**:

```
Step B:  O1-MMA:   State(128×128) @ Q_scaled(128×64) → O_inter(128×64)
                   M=128(dv), N=64(BlkSeqQ), K=128(dk)
                   A=寄存器(RS模式), B=SMEM(sQ_K_scaled stage0)

Step C1: SK-MMA:   State(128×128) @ K_scaled(128×64) → SK(128×64)
                   M=128(dv), N=64(BlkSeqKV), K=128(dk)
                   A=寄存器(RS), B=SMEM(sQ_K_scaled stage1)

Step C4: NewV-MMA: V'(128×64)     @ T^{-T}(64×64)   → NewV(128×64)
                   M=128(dv), N=64(BlkSeqKV), K=64(BlkSeqKV)
                   A=寄存器(RS), B=SMEM(sKK_opd)

Step D:  O2-MMA:   NewV^T(128×64) @ QK(64×64)        → O(128×64)
                   M=128(dv), N=64(BlkSeqQ), K=64(BlkSeqKV)
                   A=寄存器(RS), B=SMEM(sQK)

Step F:  KV-MMA:   NewV^T(128×64) @ K_decay^T(64×128) → S_new(128×128)
                   M=128(dv), N=128(dk), K=64(BlkSeqKV)
                   A=寄存器(RS), B=SMEM(sQ_K_scaled_Kt stage0)
```

**注意**: 所有 State 相关的 MMA 都是 RS 模式 (A=Register, B=SMEM)。State 矩阵 (128×128) 始终保存在寄存器中不落地，这是高性能的关键，也是寄存器消耗极高的原因。

#### 0.6.3 层级三: SMEM buffer 维度

SMEM 中每个 buffer 的完整 shape，包含 pipeline stage 维度。stage 维度用于双缓冲 (double buffering): producer 往一个 stage 写，consumer 从另一个 stage 读。

| Buffer | 代码名 | Shape (含 stage) | 维度含义 | 元素类型 | 大小 |
|--------|--------|-----------------|---------|---------|------|
| Q | smem_q | (64, 128, 2) | (BlkSeqQ, HeadSize, stages) | bf16 | 32 KB |
| K | smem_k | (128, 64, 2) | (HeadSize, BlkSeqKV, stages) | bf16 | 32 KB |
| V | smem_v | (128, 64, 1) | (HeadSize, BlkSeqKV, stages) | bf16 | 16 KB |
| Alpha | smem_alpha | (64, 128, 2) | (BlkSeqKV, HeadSize, stages) | float | 64 KB |
| QK 结果 | smem_qk | (64, 64, 2) | (BlkSeqQ, BlkSeqKV, stages) | bf16 | 16 KB |
| KK / T^{-1} | smem_kk | (64, 64, 2) | (BlkSeqKV, BlkSeqKV, stages) | half | 16 KB |
| Q/K scaled | smem_q_k_scaled | (64, 128, 2) | (BlkSeqQ, HeadSize, 2 stages) | bf16 | 32 KB |
| Output | smem_o | (128, 64, 1) | (HeadSize, BlkSeqQ, stages) | bf16 | 16 KB |
| Beta | smem_beta | (64, 2) | (BlkSeqQ, stages) | float | 0.5 KB |
| Alpha last | smem_alpha_last | (128, 2) | (HeadSize, stages) | float | 1 KB |

**同一物理 buffer 的多视图 (不同 layout)**:

```
smem_q 的两种视图:
  sQqk  = QKSmemLayoutQ    = (BlkSeqQ=64, HeadSize=128, stages=2)  — Q@K 计算 (行优先)
  (在 compute_aux_safe 中也用此 layout)

smem_k 的三种视图:
  sKqk  = QKSmemLayoutK    = (HeadSize=128, BlkSeqKV=64, stages=2) 转置为 (64, 128, 2) — QK/KK 计算
  sKkv  = KVSmemLayoutK    = (HeadSize=128, BlkSeqKV=64, stages=2) — KV MMA 的 B 操作数 (列优先)
  sKqk vs sKkv: 同一块内存, 不同 swizzle 和维度排列, 服务不同 MMA 的布局要求

smem_alpha 的两种视图:
  sAqkq = QKQSmemLayoutAlpha = (64, 128, 2)  — Q 端 alpha (行优先)
  sAqkk = QKKSmemLayoutAlpha = 转置           — K 端 alpha (列优先)

smem_q_k_scaled 的两种视图:
  sQ_K_scaled   = QKScaledSmemLayoutQ  = (64, 128, 2)  — 行优先, Q_scaled(stage0)/K_scaled(stage1)
  sQ_K_scaled_Kt = QKScaledSmemLayoutKt = (128, 64, 2) — 列优先, Step F 时 K_decay^T 给 KV-MMA B

smem_kk 的两种视图:
  sKK_inv = SmemLayoutKK 元素类型 InverseType(half) — CollectiveInverse 做 T^{-1}
  sKK_opd = SmemLayoutKK 元素类型 Element(bf16)     — reinterpret 给 NewV-MMA 做 B 操作数
```

**stage 的语义**: `smem_q_k_scaled` 的两个 stage 含义比较特殊 — 不是时间维度的双缓冲，而是空间维度的复用: stage 0 = Q_scaled (Step A 写入)，stage 1 = K_scaled (Step A 写入)。Step E 后 stage 0 被 K_decay 覆写，Step F 读取。

#### 0.6.4 层级四: compute() 中 flat_divide 分片维度

compute() 和 compute_aux_safe() 在使用 SMEM 前，都会用 `flat_divide` 把 2D tensor 按指定 tiler 切成更小的块，产生额外的迭代维度。不同函数使用不同的 tiler。

##### Step A/E prologue 的分片 (TileShapeQK_Quar, 两个 Math WG 协作)

代码 `mainloop_kda_fwd.hpp L797-799`:
```cpp
constexpr auto tiler_alpha = Shape<_64, Shape<_32, _1>>{};
constexpr auto tiler_qk    = Shape<_64, Shape<_32, _1>>{};
constexpr auto tiler_alpha_last = Shape<_32>{};
```

**Q/K 分片 (sQqk, sKqk)**:
```
原始:     sQqk_curr = (64, 128)           — (BlkSeqQ, HeadSize), 去掉 stage 维后的单 stage view
tiler:    Shape<_64, Shape<_32, _1>>       — 行方向不切(64/64=1块), 列方向每 32 一块(128/32=4块)
结果:     sQqk_slice = (64, (32,1), 1, (s_dim, wg_dim))
                        │    │       │    │
                        │    │       │    └── 4 个列块, 被解释为 2×2:
                        │    │       │        make_coord(s, wg_idx) 访问
                        │    │       │        s=0..1 (内循环迭代), wg_idx=0..1 (WG编号)
                        │    │       └── 行方向只有 1 块 (64 行不切)
                        │    └── 32 个 HeadSize 元素
                        └── 64 行 (完整 BlkSeqQ)

循环结构:
  wg_idx = thread_idx / 128    — WG0 (thread 0-127) 处理列块 {0,1}, WG1 (128-255) 处理 {2,3}
  for s in 0..1:                — 内循环, 每次一个 32 元素列块
    alpha_col = wg_idx*2 + s   — alpha 列索引 0,1,2,3
    sQqk_cur = sQqk_slice(_, _, _0{}, make_coord(s, wg_idx))  — 取出 (64, 32) 的子块

合计: 2 WG × 2 iter × 32 elem = 128 = HeadSize ✓
```

**Alpha 分片 (sAqkq)**:
```
原始:     sAqkq_curr = (64, 128)        — (BlkSeqKV, HeadSize)
tiler:    Shape<_64, Shape<_32, _1>>
结果:     sAqkq_slice = (64, (32,1), 1, (0, alpha_col))
                         │    │       │    └── 4 列块, 按 make_coord(0, alpha_col) 访问
                         │    └── 32 个 float 元素
                         └── 64 行

访问方式: sA_cur = sAqkq_slice(_, _, _0{}, make_coord(0, alpha_col))  — (64, 32) 子块
注意第三维始终为 _0{} 因为行方向不切分
```

**Alpha last 分片 (sAlast)**:
```
原始:     sAlast_curr = (128,)            — (HeadSize,) 一维向量
tiler:    Shape<_32>
结果:     sAlast_slice = (32, 4)           — 4 个 32 元素 slice

访问方式: alpha_last_cur = sAlast_slice(_, alast_idx)  — alast_idx = alpha_base + s = 0..3
每次取 32 个 float, 对应 tQcMq_quar 坐标的 head_dim 维 t 来索引
```

**Step A 循环的维度图解**:
```
        ← ─ ─ ─ ─ ─ ─ ─ HeadSize = 128 ─ ─ ─ ─ ─ ─ ─ →
        ┌────────┬────────┬────────┬────────┐
        │  32    │  32    │  32    │  32    │
  64    │ WG0,s0 │ WG0,s1 │ WG1,s0 │ WG1,s1 │    Q / K / Alpha
  行    │ col=0  │ col=1  │ col=2  │ col=3  │
        └────────┴────────┴────────┴────────┘

每个 (64, 32) 子块:
  - S2R Alpha (32 float)
  - exp2f(alpha)
  - S2R Q/K (32 bf16 通过 LDSM)
  - element-wise: Q_scaled = exp(α) * Q, K_scaled = exp(α) * K
  - R2S 写回 sQ_K_scaled 对应 stage
```

##### compute_aux_safe SubChunk 分片 (16×16, MathA WG)

代码 `mainloop_kda_fwd.hpp L1396, L1483-1485`:
```cpp
using TileShape_SubChunk = Shape<_16, _16, _32>;    // MMA atom shape
constexpr int BK = 32;                               // K 维 = 32
constexpr int NK = 128 / BK;                          // = 4, K 维迭代次数

constexpr auto tiler_subchunk_qk    = Shape<_16, Shape<_32, _1>>{};
constexpr auto tiler_subchunk_alpha = Shape<_16, Shape<_32, _1>>{};
constexpr auto tiler_subchunk_beta  = Shape<_16>{};
```

**Q/K SubChunk 分片**:
```
原始:     sQqk_curr = (64, 128)            — (BlkSeqQ, HeadSize)
tiler:    Shape<_16, Shape<_32, _1>>        — 行方向每 16 一块(64/16=4), 列方向每 32 一块(128/32=4)
结果:     sQqk_slice = (16, (32,1), r=4, (s, j))
                        │    │       │    │
                        │    │       │    └── 4 个列块: j=0..3 (BK=32 的 K 维迭代)
                        │    │       │        但代码中用 make_coord(j0, j1) 二级索引
                        │    │       └── r = 0..3: 4 个 16-行的行块 (SubChunk 行索引)
                        │    └── 32 个 HeadSize 元素 (对应 BK=32)
                        └── 16 行 (SubChunk M 维)

访问方式: sQqk_slice(_, _, r_, make_coord(j0, j1))  — 取出 (16, 32) 子块
```

**输出 QK/KK accumulator 分片**:
```
原始:     sQK_curr = (64, 64)               — (BlkSeqQ, BlkSeqKV)
tiler:    Shape<_16, _16>                    — 行列都按 16 切分
结果:     sQK_slice = (16, 16, R=4, C=4)    — R 行块, C 列块

访问方式: sQK_slice(_, _, r_, c_)  — (16, 16) 子块
```

**Beta SubChunk 分片**:
```
原始:     sBeta_curr = (64,)                 — (BlkSeqQ,) 一维向量
tiler:    Shape<_16>
结果:     sBeta_slice = (16, 4)              — 4 个 16 元素 slice

访问方式: sBeta_slice(_, r)  — r = 行块索引, 取对应 16 个 beta 值
```

**SubChunk 循环结构与维度图解**:
```
            QK/KK 输出矩阵 (64×64)               Q 或 K 矩阵 (64×128)
            ← ─ BlkSeqKV = 64 ─ →               ← ─ ─ HeadSize = 128 ─ ─ →
            ┌────┬────┬────┬────┐               ┌────────┬────────┬────────┬────────┐
  行块 r=0  │c=0 │c=1 │c=2 │c=3 │    16×16      │ j=0    │ j=1    │ j=2    │ j=3    │  16×32
  行块 r=1  │    │    │    │    │    SubChunk    │ BK=32  │ BK=32  │ BK=32  │ BK=32  │  (K维迭代)
  行块 r=2  │    │    │    │    │                │        │        │        │        │
  行块 r=3  │    │    │    │    │                │        │        │        │        │
            └────┴────┴────┴────┘               └────────┴────────┴────────┴────────┘

每个 QK[r][c] = Σ_{j=0}^{3} Q_subchunk[r][j] @ K_subchunk[c][j]^T
              = Σ_{j=0}^{3} (16×32) @ (32×16) → (16×16)

特殊: 128 线程分成两组 (local_thread_idx = thread_idx % 64)
  线程 0-63:   一组, 处理 QK 和 KK 各一半 (按行块交错)
  线程 64-127: 另一组, 处理另一半

行块遍历:
  for r in 0..3:
    if (r % 2 != group_idx): skip     — 两组线程交替处理偶/奇行块
    for c (列) ...:                    — 只计算下三角 (c <= r, 因为 QK/KK 是下三角)
      for j in 0..NK-1:               — K 维累加 (32 × 4 = 128 = HeadSize)
        SubChunk MMA: (16, 16, 32)
```

#### 0.6.5 层级五: MMA partition 后的寄存器 fragment 维度

WGMMA 通过 `partition_fragment_C` 将 M×N 的累加器分配到 128 线程上。每线程持有的 float 寄存器数 = `M × N / 128 × (sizeof(float)/4)` (对 Cooperative MMA)。

| 变量 | TileShape 的 (M, N) | 累加器总 floats | 每线程 floats | 每线程 32-bit 寄存器 | 用途 |
|------|---------------------|---------------|-------------|--------------------|----|
| `tKVrKV` | (128, 128) | 16384 | 128 | 128 | State 矩阵 — 最大消耗者 |
| `tOrO` | (128, 64) | 8192 | 64 | 64 | Output 累加器 |
| `tSKrSK` | (128, 64) | 8192 | 64 | 64 | S@K 结果 |
| `tSKrV` | (128, 64) | 8192 | 64 | 64 | V' = V - SK |
| `tNewVrC` | (128, 64) | 8192 | 64 | 64 | NewV 结果 |

State MMA 每线程寄存器 = 168 (代码中的 `mma_registers`):
- tKVrKV 占 128 reg
- B 操作数 fragment + 临时变量约 40 reg
- 合计 ~168 reg/thread

**寄存器复用链**: tSKrV → (make_acc_into_op) → tNewVrA → (MMA) → tNewVrC → (make_acc_into_op) → tOrV_or_tKVrV。物理上是同一组寄存器，通过 layout 重解释 (zero-copy) 在不同 MMA 间传递。

#### 0.6.6 维度速查: 从公式到代码

| 数学量 | 维度 (论文) | 维度 (代码, d=128, C=64) | 所在位置 |
|--------|-----------|------------------------|---------|
| S (State) | dk × dv | 128 × 128 | 寄存器 tKVrKV |
| Q_chunk | C × dk | 64 × 128 | SMEM smem_q |
| K_chunk | C × dk | 64 × 128 | SMEM smem_k (转置视图为 128×64) |
| V_chunk | C × dv | 64 × 128 | SMEM smem_v (转置视图为 128×64) |
| α_chunk | C × dk | 64 × 128 | SMEM smem_alpha |
| β_chunk | C | 64 | SMEM smem_beta |
| Q_scaled | C × dk | 64 × 128 | SMEM smem_q_k_scaled stage0 |
| K_scaled | C × dk | 64 × 128 | SMEM smem_q_k_scaled stage1 |
| K_decay | C × dk | 64 × 128 | SMEM smem_q_k_scaled stage0 (覆写) |
| QK | C × C | 64 × 64 | SMEM smem_qk |
| KK / T^{-1} | C × C | 64 × 64 | SMEM smem_kk |
| SK = S@K | dv × C | 128 × 64 | 寄存器 tSKrSK |
| V' = V - SK | dv × C | 128 × 64 | 寄存器 tSKrV |
| NewV = V'@T^{-T} | dv × C | 128 × 64 | 寄存器 tNewVrC |
| O_inter = S @ Q_scaled | dv × C | 128 × 64 | 寄存器 tOrO |
| O_intra = NewV^T @ QK | dv × C | 128 × 64 | 寄存器 tOrO (累加) |
| α_last | dk | 128 | SMEM smem_alpha_last |

---

## 1. KDA Forward 算法概述

KDA (Kernel Delta-rule Attention) 是一种线性注意力变体，核心特点是维护一个 **隐状态 S**，在处理每个 tile 时更新 S 并产出输出 O。

### 1.1 符号定义 (对齐论文 §2.1 和 §3)

**单步递推 (论文 Eq.1)**:
```
S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
o_t = S_t^T q_t
```
其中:
- `S_t ∈ R^{dk×dv}`: 隐状态矩阵 (dk=dv=128 in KDA)
- `q_t, k_t ∈ R^dk`: query, key 向量
- `v_t ∈ R^dv`: value 向量
- `α_t ∈ (0,1)^dk`: **channel-wise** 遗忘门 (fine-grained gate, KDA 相对 GDN 的核心改进)
- `β_t ∈ (0,1)`: **scalar** 学习率/delta rule 系数
- `Diag(α_t)`: 对角矩阵, 每维独立衰减

**Chunk-wise 符号** (论文 §2.1):
- 序列分成 `L/C` 个 chunk, 每 chunk 长 `C` (代码中 C = BlkSeqQ = 64)
- `Q[t], K[t], V[t] ∈ R^{C×d}`: 第 t 个 chunk 的矩阵 (stack C 个向量)
- `S[t] := S_0[t] = S_C[t-1]`: chunk t 的初始 state = 上一 chunk 的最终 state
- `γ_r[t] := Π_{k=1}^r α_k[t]`: 累积衰减 (cumulative decay), shape `(dk,)`
- `Γ_{i→C}[t] ∈ R^{C×dk}`: 从第 i 行到第 C 行的累积衰减矩阵
- `g = log(α)`: 代码中用 log 域, `γ = exp(g.cumsum)`

### 1.2 论文 Chunk-wise 公式 → 代码映射

**论文的 UT Transform (Eq.6-7)**:
```
# KK 矩阵及其求逆 (forward substitution)
M[t] = (I + StrictTril(diag(β[t]) (Γ⊙K[t]) (K[t]/Γ)^T))^{-1} diag(β[t])

# 辅助矩阵 W 和 U (论文中的 "pseudo value")
W[t] = M[t] (Γ⊙K[t])     ← "decay-scaled K after inverse"
U[t] = M[t] V[t]          ← "corrected V after inverse"
```

**论文的 Output (Eq.9)**:
```
O[t] = (Γ⊙Q[t]) S[t]                                    ... inter-chunk
     + Tril((Γ⊙Q[t]) (K[t]/Γ)^T) (U[t] - W[t] @ S[t])  ... intra-chunk
```

**论文的 State Update (Eq.8)**:
```
S[t+1] = Diag(γ_C[t]) S[t] + (Γ⊙K[t])^T (U[t] - W[t] @ S[t])
```

### 1.3 代码实现中的简化 (safe_gate 模式)

代码中不直接用 `Γ⊙Q` 和 `K/Γ` 的除法形式 (数值不稳定)，而用 **safe gate** 重参数化。
核心思想: 把 `Γ` 的缩放"吸收"进 Q、K 中，并减去一个 anchor (α_first 或 α_last) 防止 exp 溢出。

**Phase 1: compute_aux_safe (MathA WG) — 对应论文 intra-chunk attention matrix**

论文中 intra-chunk 的 attention 矩阵:
```
A_intra = Tril((Γ⊙Q)(K/Γ)^T)                              ... 论文 Eq.9 中间部分
```

代码实现 (safe_gate, 以 α_first = α[0,:] 为 anchor):
```
Q̃[i,:] = Q[i,:] * exp(α[i,:] - α_first)      ← gated Q     ... (1.2)
K̃[j,:] = K[j,:] * exp(α_first - α[j,:])      ← gated K     ... (1.3)
QK = Tril(Q̃ @ K̃^T) * scale                                   ... (1.4)(1.6)
```
可以验证: `Q̃[i,:] * K̃[j,:]^T = Q[i,:] * exp(α[i,:] - α[j,:]) * K[j,:]^T`，等价于论文的 `(Γ⊙Q)(K/Γ)^T` 在下三角区域的值。

**KK 矩阵** (对应论文 Eq.6 中 `StrictTril(diag(β)(Γ⊙K)(K/Γ)^T)`):
```
KK = StrictTril(K̃ @ K̃^T) ⊙ diag(β)                          ... (1.5)(1.7)
T = I + KK                                                     ... (2.6)
T^{-1} = forward_substitution(T)                               ... (2.7)
```

**Phase 2: compute (Math0/1 WG) — 对应论文 Eq.7-9**

论文中 `U = M V`, `W = M (Γ⊙K)` 合并求解:
```
NewV = T^{-1} @ V  (代码中: V' = V - S_old @ K_scaled^T, 然后 NewV = V' @ T^{-T})
```

这里代码和论文有一个微妙差异: 论文公式的 `U - W@S` 被代码拆成了:
- `U = T^{-1} V` → 对应代码中 `NewV = (V - S@K_scaled) @ T^{-T}` (融合了 `W@S` 项)
- 实际上 `U - W@S = T^{-1}(V - (Γ⊙K)@S) = T^{-1}(V - S@K_scaled^T)^T` (转置差异来自代码的列优先布局)

### 1.4 完整代码计算流程 (单 chunk, 对照论文和伪代码)

```
输入: S_old (= S[t]), Q, K, V, α (= log gate), β
输出: O, S_new (= S[t+1])

# 论文变量准备
g = α (已是 log 域)
gc = g.cumsum(dim=seq)            # 累积 log-decay

=== Phase 1: Auxiliary (compute_aux_safe, MathA WG) ===
=== 对应论文伪代码 L22-32 中的 A 计算和 forward substitution ===

  # Gated attention (safe_gate, anchor=α_first=gc[0,:])
  QK[i,j] = Σ_d Q[i,d]*K[j,d]*exp(gc[i,d]-gc[j,d])    ... (1.4) 对应伪代码 L48-51
  QK = Tril(QK) * scale                                  ... (1.6) 对应伪代码 L52

  # KK 矩阵 (对应伪代码 L23-27)
  KK[i,j] = Σ_d K[i,d]*K[j,d]*exp(gc[j,d]-gc[i,d])     ... (1.5)
  KK = -StrictTril(KK) * β                               ... (1.7) 对应伪代码 L27,29

  # Forward substitution (对应伪代码 L30-32, 论文 Eq.6)
  T = I + KK (严格下三角)
  T^{-1} (in-place)                                       ... (2.7)

  # W = T^{-1} @ (exp(gc) * K), U = T^{-1} @ V          ... 对应伪代码 L34-35
  # (代码中 W 和 U 的计算融合到了 Step C 和 Step F 中)

=== Phase 2: State MMA (compute, Math0/1 WG) ===
=== 对应论文伪代码 L43-55 的 chunk 内循环 ===

  --- Step A: Q/K Prologue ---
  Q_scaled = Q ⊙ exp(gc)                                 ... (2.1) 对应伪代码 L54 中 q_i * g_i.exp()
  K_scaled = K ⊙ exp(gc)                                 ... (2.2)

  --- Step B: Inter-chunk output ---                      ... 对应论文 Eq.9 第一项 "(Γ⊙Q) S[t]"
  O_inter = scale * Q_scaled @ S_old                     ... (2.3) 对应伪代码 L54 "(q_i * g_i.exp()) @ S"

  --- Step C: Intra-chunk NewV ---                        ... 对应论文 Eq.9 "U - W@S" 部分
  SK = S_old @ K_scaled^T                                 ... (2.4) 即 W@S = T^{-1}(Γ⊙K) @ S
  V' = V - SK^T                                           ... (2.5) 对应伪代码 L53 "v_i = u_i - w_i @ S"
  NewV = V' @ T^{-T}                                      ... (2.8) 即 T^{-1} @ V'

  --- Step D: Intra-chunk output ---                      ... 对应论文 Eq.9 第二项 "Tril(QK)(U-W@S)"
  O_intra = NewV^T @ QK                                   ... (2.9) 对应伪代码 L54 "A @ v_i"
  O = O_inter + O_intra                                   ... (2.10)

  --- Step E: State decay ---                             ... 对应论文 Eq.8 第一项 "Diag(γ_C) S[t]"
  S_decayed = S_old ⊙ exp(gc[-1,:])                      ... (2.11) 对应伪代码 L55 "S * g_i[:,:,-1:].exp()"
  K_decay = K ⊙ exp(gc[-1,:] - gc)                       ... (2.12) 对应伪代码 L29 "decay" 部分

  --- Step F: State update ---                            ... 对应论文 Eq.8 第二项 "+ (Γ_{i→C}⊙K)^T (U-W@S)"
  S_new = S_decayed + NewV^T @ K_decay^T                  ... (2.13) 对应伪代码 L29 "S += (k_i*decay)^T @ v_i"

  Step E/F 推导 (从论文 Eq.8):
    S[t+1] = Diag(γ_C) · S[t]  +  (Γ_{i→C} ⊙ K)^T · (U - W·S[t])
             ─────────────────     ─────────────────────────────────
             S_decayed              K_decay^T @ NewV

    γ_C = Π_{k=1}^C α_k = exp(gc[-1])
      → 整个 chunk 的总衰减, C 个 token 依次衰减旧 State
      → S_decayed[i,j] = S_old[i,j] × exp(gc[-1, i])

    γ_{i→C} = Π_{k=i+1}^C α_k = exp(gc[-1] - gc[i])
      → 第 i 个 token 到 chunk 末尾的部分衰减
      → K_decay[i,d] = K[i,d] × exp(gc[-1,d] - gc[i,d])
      → 含义: 第 i 个 token 写入 State 后, 被之后的 C-i 个 token 衰减

    K_scaled vs K_decay (互补关系):
      K_scaled[i] = K[i] × exp(gc[i])           ← 从 chunk 开头到第 i 步 (Step A 用)
      K_decay[i]  = K[i] × exp(gc[-1] - gc[i])  ← 从第 i 步到 chunk 末尾 (Step E 用)
      K_scaled × K_decay = K × exp(gc[-1])       ← 两者乘积 = K × 总衰减
```

### 1.5 推导: safe_gate 的 anchor trick

**问题**: 论文公式含 `K/Γ` 除法, 直接计算 `2^{-gc[j]}` 会溢出 (gc[j] 可达 -320, `2^{320}` 超 float32)。

**推导过程**:

```
论文 intra-chunk attention (下三角):
A[i,j] = (Γ⊙Q)[i,:] · (K/Γ)[j,:]^T
        = Σ_d Q[i,d] · 2^{gc[i,d]} · K[j,d] · 2^{-gc[j,d]}
        = Σ_d Q[i,d] · K[j,d] · 2^{gc[i,d] - gc[j,d]}           ... (★) 核心: 乘除→指数加减

选 anchor gc_ref (代码取 gc[0] 即 chunk 第一行):

2^{gc[i] - gc[j]} = 2^{(gc[i] - gc_ref) + (gc_ref - gc[j])}      ... 加减同一个 ref
                   = 2^{gc[i] - gc_ref} · 2^{gc_ref - gc[j]}       ... 指数加法→乘法

代入 (★):
A[i,j] = Σ_d [Q[i,d] · 2^{gc[i]-gc_ref}] · [K[j,d] · 2^{gc_ref-gc[j]}]
        = Q̃[i,:] · K̃[j,:]^T                                      ... 普通矩阵乘!

其中:
  Q̃[i,d] = Q[i,d] · exp2(gc[i,d] - gc_ref[d])     gc[i]-gc_ref ≤ 0 → exp ∈ (0,1] ✓
  K̃[j,d] = K[j,d] · exp2(gc_ref[d] - gc[j,d])     在 16×16 SubChunk 内最大 75 → 3.8e22 ✓
```

**正确性验证**: anchor 在 Q̃·K̃^T 中自动相消:
```
  exp2(gc[i]-ref) · exp2(ref-gc[j]) = exp2(gc[i]-gc[j])   ← ref 消掉了
```

**QK 和 KK 共享 K̃**: 同样的 anchor trick 对 KK = `diag(β)(Γ⊙K)(K/Γ)^T` 也适用, 且 K̃ 完全相同, 所以代码在同一循环里复用。

**为什么 Step A (Q_scaled, K_scaled) 不需要 anchor**: `Γ⊙Q = Q·exp(gc)`, 因为 gc ∈ [-320,0] (chunk 内累积), `exp2(gc) ∈ (0,1]` 不会溢出。只有 `K/Γ = K·exp(-gc)` 的 `-gc` 会正溢出, 才需要 anchor。

### 1.6 论文 vs 代码 的关键差异说明

| 方面 | 论文 | 代码 |
|------|------|------|
| **decay 域** | 乘法域 `Γ = Π α`, 除法 `K/Γ` | log 域 `g = log(α)`, `gc = g.cumsum`, 差值 `exp(gc[i]-gc[j])` |
| **数值稳定** | 论文 Eq.9 有 `K/Γ` 除法 | 代码用 safe_gate: 减去 anchor (α_first 或 α_last) 后再 exp |
| **U,W 计算** | 论文显式 `U = M@V, W = M@(Γ⊙K)` | 代码融合: `V' = V - S@K_scaled^T` 然后 `NewV = V'@T^{-T}` 等价于 `U - W@S` |
| **QK 缩放** | 论文 `Q` 已乘 `d^{-0.5}` (伪代码 L18) | 代码中 `scale` 参数单独乘 |
| **β 的融合** | 论文 `M` 包含 `diag(β)` | 代码中 β 被融合到 KK 矩阵的对角缩放和 T^{-1} 后处理中 |
| **W@S 融合** | 论文先算 `W`, 再算 `W@S` | 代码直接算 `S@K_scaled^T` (= `(K_scaled@S^T)^T`), 不显式构造 W |

### 1.6 公式与代码变量对应

| 公式符号 | 论文符号 | 代码变量 | 存储位置 | 说明 |
|---------|---------|---------|---------|------|
| S_old / S_new | S[t] / S[t+1] | `tKVrKV` | **寄存器** (贯穿全 mainloop) | State 矩阵 (d,d) |
| Q_scaled | Γ⊙Q | `sQ_K_scaled(_, _, _0{})` | SMEM stage 0 | exp(gc)*Q |
| K_scaled | Γ⊙K | `sQ_K_scaled(_, _, _1{})` | SMEM stage 1 | exp(gc)*K |
| K_decay | exp(gc[-1]-gc)⊙K | `sQ_K_scaled(_, _, _0{})` | SMEM stage 0 (复用) | Step E 覆写 |
| QK | Tril((Γ⊙Q)(K/Γ)^T) | `sQK` | SMEM (via qk_pipeline) | MathA 产出 |
| T^{-1} | M / (I+KK)^{-1} | `sKK_inv` | SMEM (via kk_pipeline) | MathA 产出→Step C 求逆 |
| V' | V - W@S = U - W@S 的中间态 | `tSKrV` | 寄存器 | Step C |
| NewV | T^{-1}@V' | `tNewVrC` | 寄存器 | Step C → D,F |
| O | O[t] | `tOrO` | 寄存器→SMEM `sO` | Step B+D → o_pipeline |
| α_last | γ_C = gc[-1,:] | `sAlast` | SMEM (via alpha_last_pipeline) | Step E |
| β | β[t] | `sBeta` | SMEM (via beta_pipeline) | KK缩放 + T^{-1}后处理 |
| scale | d^{-0.5} | `params.scale` | 常量 | QK 和 O_inter 缩放 |

### 1.7 首 tile 特殊处理

当 `S_old = 0` (零初始化, `kInitStateFromInput=false`) 时:
- Step A 跳过 (Q_scaled 无意义, Q@0=0)
- Step B: O_inter = 0
- Step C: SK = 0, V' = V (无残差), NewV = V @ T^{-T}
- Step D: O = O_intra only
- Step E: S_decayed = 0
- Step F: S_new = 0 + NewV^T @ K_decay^T

如果 `kInitStateFromInput=true`, 从 GMEM 加载初始 state, 首 tile 也执行完整流程。

---

## 2. 代码架构总览

### 2.1 文件结构

```
csrc/kda/sm90/
├── kernel/
│   ├── kernel_kda_fwd.hpp       ★ Kernel 入口 (本文学习主体)
│   ├── builder_kda_fwd.hpp      Kernel Builder 工厂模式
│   ├── tile_scheduler.hpp       Tile 调度器 (Grid→Work 映射)
│   └── options.hpp              编译期选项系统
├── collective/
│   ├── mainloop_kda_fwd.hpp     ★ 核心计算 mainloop (~2062行)
│   ├── common.hpp               MMA/转换工具函数
│   ├── load_tma.hpp             TMA 加载
│   ├── load_predicated.hpp      谓词加载 (处理边界)
│   ├── store_tma.hpp            TMA 存储
│   └── named_barriers.hpp       命名 Barrier
├── utils/
│   ├── debug.hpp                调试打印
│   ├── math.hpp                 数学工具
│   ├── math_order_barrier.hpp   有序 Barrier
│   └── type_traits.hpp          类型映射
├── kda_fwd_sm90.cu              Host 端 launcher
└── prefill_kernel_kda_fwd_sm90.cuh  模板实例化
```

### 2.2 调用链

```
launch_kda_fwd_prefill_kernel()           [kda_fwd_sm90.cu]
  → FlatBuilderKdaFwd::Kernel            [builder_kda_fwd.hpp]
    → FlatKernelTmaWarpSpecializedKdaFwd  [kernel_kda_fwd.hpp]  ← operator()
      → CollectiveMainloop 的各方法       [mainloop_kda_fwd.hpp]
        ├── load_qkv()
        ├── load_beta()
        ├── extract_alpha_last()
        ├── store()
        ├── compute()              ← 核心: State MMA WG
        └── compute_aux_safe()     ← 核心: Aux MMA WG
```

### 2.3 Warp Specialization 架构

整个 kernel 使用 **4 个 Warp Group** (512 线程/block)，按角色分工:

```
┌─────────────────────────────────────────────────────────────┐
│  Block (512 threads = 4 Warp Groups)                        │
│                                                             │
│  WG0: LdSt (128 threads)          WG3: MathA (128 threads) │
│  ┌──────────────────────┐          ┌──────────────────────┐ │
│  │ Warp0: LoadQKV       │          │ compute_aux_safe()   │ │
│  │ Warp1: StoreO        │          │ - Q@K 计算           │ │
│  │ Warp2: LoadBeta      │          │ - K@K 计算           │ │
│  │ Warp3: LoadAlpha     │          │ - exp(α) 缩放        │ │
│  │       +ExtractLast   │          │ 产出: QK, KK → SMEM  │ │
│  └──────────────────────┘          └──────────────────────┘ │
│                                                             │
│  WG1: Math0 (128 threads)  WG2: Math1 (128 threads)        │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │ compute()             │  │ compute()             │        │
│  │ - T^{-1} 求逆         │  │ - (协作 128 tile时)   │        │
│  │ - NewV 计算           │  │                       │        │
│  │ - State 更新          │  │                       │        │
│  │ - O 输出              │  │                       │        │
│  └──────────────────────┘  └──────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 数据流 Pipeline

```
GMEM ──TMA──→ SMEM(Q,K,V,α,β) ──Pipeline──→ Register(MMA)
                                                    │
              SMEM(QK,KK) ←──Pipeline──── MathA 产出│
                    │                               │
                    └──Pipeline──→ Math0/1 消费 ────→ SMEM(O)
                                                    │
                                          ──Pipeline──→ StoreO Warp ──TMA──→ GMEM
```

---

## 3. 寄存器分配策略

SM90 允许运行时动态调整寄存器数量。不同角色的 Warp Group 需要不同数量的寄存器:

```c++
// Load/Store WG: 少量寄存器 (主要是地址计算)
cutlass::arch::warpgroup_reg_dealloc<LdStRegisterRequirement>();   // ~24-40 regs

// State MMA WG: 大量寄存器 (accumulator + 中间结果)
cutlass::arch::warpgroup_reg_alloc<StateMmaRegisterRequirement>(); // ~100+ regs

// Aux MMA WG: 中等寄存器
cutlass::arch::warpgroup_reg_alloc<AuxMmaRegisterRequirement>();   // ~100+ regs
```

总寄存器预算: 64K / block, 分配给 4 个 WG 的 512 线程。

---

## 4. 函数级详细教程

### 学习进度

| 步骤 | 状态 | 内容 |
|------|------|------|
| 4.1 get_register_requirements | [x] | 寄存器预算计算 |
| 4.2 IndividualTileScheduler | [x] | Grid→Work 映射 |
| 4.3 Options 系统 | [x] | 编译期选项 |
| 4.4 operator() | [x] | Kernel 入口 |
| 4.5 load_qkv | [x] | TMA 加载 Q/K/V/α |
| 4.6 load_beta | [x] | Predicated 加载 β |
| 4.7 extract_alpha_last | [x] | α 最后一行提取 |
| 4.8 store | [x] | TMA 写回 O |
| 4.9 compute_aux_safe | [x] | 辅助 MMA (QK, KK) |
| 4.10 compute Step A | [x] | Q/K Prologue |
| 4.10 compute Step B | [x] | O_inter |
| 4.10 compute Step C | [ ] | V'→NewV |
| 4.10 compute Step D | [x] | O_intra + O 写出 |
| 4.10 compute Step E | [x] | State 衰减 |
| 4.10 compute Step F | [x] | State 更新 |

---

### 4.1 `get_register_requirements()` — 寄存器预算计算

**文件**: `kernel_kda_fwd.hpp:36-61`

#### 4.1.1 背景: 为什么要手动分配寄存器?

GPU 的每个 SM 上有 **65536 个 32-bit 寄存器**，是所有并发线程共享的固定资源池。编译器通常会均匀分配寄存器给每个线程，但在 warp-specialized kernel 中，不同角色的需求差异很大:

- **Load/Store warp group**: 只做地址计算和 TMA 指令调用，几乎不需要寄存器
- **MMA warp group**: 需要存放 accumulator 矩阵 (比如一个 64×128 的 fp32 accumulator = 64 个寄存器/线程) 和中间结果

如果均匀分配，Load warp 浪费寄存器，MMA warp 又不够用。SM90 提供了运行时指令 `warpgroup_reg_alloc` / `warpgroup_reg_dealloc` 来解决这个问题。

#### 4.1.2 代码逐行解析

```c++
constexpr std::tuple<uint32_t, uint32_t, uint32_t>
get_register_requirements(
    uint32_t max_threads_per_block,          // 512 (= 4 WG × 128)
    uint32_t min_blocks_per_multiprocessor,   // 1
    uint32_t num_state_mma_warp_groups        // 2 (Math0 + Math1)
) {
    uint32_t reg_alloc_granularity = 8;      // SM90 寄存器分配粒度: 8 个一组, 硬件限制
```

**`reg_alloc_granularity = 8`**: 你不能要 37 个寄存器，只能要 32 或 40。

```c++
#if !defined(FLAT_DEBUG_PRINT) || !FLAT_DEBUG_PRINT
    uint32_t load_registers = 40 - 2 * 8;   // = 24
#else
    uint32_t load_registers = 40;
#endif
```

**Load/Store WG 的寄存器**: 非 debug 模式下只要 24 个 (够做地址计算)；debug 模式下 40 个 (printf 需要更多寄存器)。

```c++
    uint32_t total_aux_load_budget = 176;
    uint32_t aux_registers = 176 - load_registers;  // = 152 (非debug) 或 136 (debug)
```

引入"辅助+加载联合预算" 176。Load 少用的寄存器让给 Aux MMA。

```c++
    uint32_t total_registers =
        round_down(64 * 1024 / min_blocks_per_multiprocessor,
                   max_threads_per_block * reg_alloc_granularity)
        / cutlass::NumThreadsPerWarpGroup;
```

这是关键计算。代入实际值:

```
64 * 1024 = 65536                           (SM 上的总寄存器数)
65536 / 1 = 65536                            (只有 1 个 block 占用整个 SM)
max_threads_per_block * reg_alloc_granularity = 512 * 8 = 4096  (对齐粒度)
round_down(65536, 4096) = 65536              (已经对齐)
65536 / 128 = 512                            (÷ 一个 WG 的线程数)
```

**`total_registers = 512`** 的含义: 把整个 SM 的寄存器折算到单 WG 粒度的总预算。这个 512 代表 4 个 WG 每线程寄存器的总和 (因为 `65536 / 512线程 = 128/线程`, `128 × 4个WG = 512`)。

```c++
    uint32_t mma_registers = round_down(
        (total_registers - load_registers - aux_registers) / num_state_mma_warp_groups,
        reg_alloc_granularity);
    // = round_down((512 - 24 - 152) / 2, 8) = round_down(168, 8) = 168
```

State MMA 的预算 = 总预算减去 Load 和 Aux 后，平分给 2 个 State MMA WG。

```c++
    return {cute::min(248, load_registers),    // min(248, 24) = 24
            cute::min(248, mma_registers),     // min(248, 168) = 168
            cute::min(248, aux_registers)};    // min(248, 152) = 152
}
```

`min(248, ...)` 是硬件上限: 每线程最多 255 个寄存器，248 是最大的 8-对齐值。

#### 4.1.3 最终分配结果

| Warp Group | 角色 | 寄存器/线程 | 用途 |
|---|---|---|---|
| WG0 | LdSt | **24** | 地址计算, TMA descriptor |
| WG1 | Math0 (State) | **168** | MMA accumulator, 中间矩阵 |
| WG2 | Math1 (State) | **168** | 同上 |
| WG3 | MathA (Aux) | **152** | QK/KK accumulator |

**验算预算**: `(24 + 168 + 168 + 152) × 128 = 512 × 128 = 65536 ✓ 刚好用满整个 SM。`

#### 4.1.4 Occupancy 和延迟隐藏

用满 65536 寄存器意味着 **occupancy = 1 block/SM**。传统 CUDA 教材说"occupancy 越高越好"，但对于这种 warp-specialized kernel，1 block/SM 反而是最优的:

1. **16 个 warp 全部驻留在 SM 上**，SM 的 4 个 warp scheduler 每 cycle 可从中选 ready 的 warp 发射，并非"一次只能跑一个 WG"
2. **延迟隐藏靠 Pipeline**，而非靠多 block 轮换: Load WG 做 TMA 异步加载时，Math WG 同时做计算
3. **更多寄存器 = 更大 tile = 更高计算密度**: 168 个寄存器/线程已经很紧张了，砍半会导致 tile 缩小或频繁 spill
4. **SMEM 也是瓶颈**: 一个 block 的 SharedStorage 就用掉了大部分 228KB SMEM

#### 4.1.5 运行时如何生效

在 `operator()` 中:

```c++
// LdSt WG: 释放多余寄存器给别人 (PTX setmaxnreg.dec)
cutlass::arch::warpgroup_reg_dealloc<24>();

// State MMA WG: 申请更多寄存器 (PTX setmaxnreg.inc)
cutlass::arch::warpgroup_reg_alloc<168>();

// Aux MMA WG: 申请中等数量
cutlass::arch::warpgroup_reg_alloc<152>();
```

LdSt 释放的寄存器会被 MMA warp group 使用。

---

### 4.2 `IndividualTileScheduler` — Grid 到 Work 的映射

**文件**: `tile_scheduler.hpp:67-137`

#### 4.2.1 解决什么问题

GPU kernel 启动时，host 指定 grid (多少个 block)，每个 block 通过 `blockIdx` 知道自己是谁。Scheduler 的职责: **给定 blockIdx，告诉这个 block 它应该处理哪个 seq、哪个 head**。

#### 4.2.2 WorkDesc — 工作描述符

```c++
struct WorkDesc {
    int32_t seq_idx;      // 第几条序列
    int32_t head_idx;     // 第几个 attention head
    int64_t tok_offset;   // 这条序列在 packed tensor 中的起始 token 位置
    int64_t seq_len;      // 这条序列的长度
    int32_t tile_idx = 0; // 由 mainloop 在迭代过程中更新
};
```

KDA 使用 **variable-length packed** 格式:

```
cu_seqlens = [0, 120, 300, 512]   ← 累积长度，3 条序列

Q/K/V 的 GMEM 布局 (packed):
token: [0 ................. 119|120 .............. 299|300 ....... 511]
        ← seq 0 (len=120) →   ← seq 1 (len=180) →   ← seq 2 (len=212) →

tok_offset 指向每条序列的起始位置 (0, 120, 300)
```

Q/K/V/O head 数在 KDA 中相同 (`q_head_idx() = k_head_idx() = v_head_idx() = o_head_idx() = head_idx`)。

#### 4.2.3 Grid 构建与 Block→Work 映射

```c++
static Params to_underlying_arguments(...) {
    dim3 grid(0, 1, 1);
    grid.x = problem_size.num_seqs * problem_size.num_heads;  // 一维 grid
    return { .grid = grid, .num_seqs = ..., .num_heads = ... };
}
```

最简单的一维 grid: `grid.x = num_seqs × num_heads`。例如 3 条序列 × 8 个 head = 24 个 block。

```c++
CUTE_DEVICE WorkDesc get_next_work(Params params, ProblemSize const& problem_size) {
    int32_t seq_idx  = blockIdx.x / params.num_heads;    // 商 = 第几条序列
    int32_t head_idx = blockIdx.x % params.num_heads;    // 余 = 第几个 head

    int32_t s = problem_size.cu_seqlens[seq_idx];        // 起始 offset
    int32_t e = problem_size.cu_seqlens[seq_idx + 1];    // 结束 offset
    int32_t seq_len = e - s;

    if (scheduled) { seq_idx = -1; }                     // 第二次调用返回无效
    else { scheduled = true; }                           // 标记已调度

    return { .seq_idx = seq_idx, .head_idx = head_idx, .tok_offset = s, .seq_len = seq_len };
}
```

#### 4.2.4 一次性调度器

`scheduled` 是 `bool` 成员变量。第一次调用返回有效 WorkDesc 并设为 true，第二次调用 `seq_idx = -1`，`is_valid()` 返回 false，外层 for 循环终止。

```c++
// kernel 入口中的调用模式:
auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
for (; work_desc.is_valid(params.scheduler);
     work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
    // 实际只执行一次
}
```

写成循环是为兼容未来的 `PersistentTileScheduler` (一个 block 处理多个 work item)。SM100 版本已有 `StaticPersistentTileScheduler` (`csrc/kda/sm100/tile_scheduler.hpp`)。

每个 WG/warp 各自独立构造 scheduler 实例，基于相同的 `blockIdx.x` 计算得到相同结果，无需 WG 间同步。

---

### 4.3 `Options` 系统 — 编译期选项

**文件**: `options.hpp`

#### 4.3.1 解决什么问题

KDA kernel 有多个开关 (是否需要 alpha/beta、是否从 GMEM 初始化 state、SafeGate 模式等)。用运行时 `if` 判断会浪费寄存器和指令。Options 系统把这些开关做成**编译期常量**，通过模板参数传递，编译器彻底消除无关分支。

#### 4.3.2 三个核心组件

**1) `Option` — 单个选项 (key-value pair)**

```c++
template <auto kTag, class Value>
struct Option {
    static constexpr auto tag = kTag;    // key: Tag 枚举值
    using option_value = Value;           // value: 类型 (true_type/false_type/Int<N>/float 等)
};
```

**2) `find_option_t` — 查找**

```c++
find_option_t<Tag::kNeedsAlpha, cute::true_type, Options>
//            ^key              ^默认值           ^在哪个tuple里找
```

递归模板匹配查找，等价于 Python 的 `dict.get(key, default)`。

**3) `add_option` — 追加**

```c++
add_option(Option<Tag::kSafeGate, true_type>{}, existing_options)
// → 新 tuple 末尾多一个 Option
```

#### 4.3.3 Tag 枚举

```c++
enum class Tag {
    kIsDeltaRule,          // delta rule 模式 (KDA 始终 true)
    kIsPersistent,         // persistent scheduler (SM90 未启用)
    kNumMmaWarpGroups,     // MMA warp group 数
    kStagesQ/K/V,          // pipeline 级数
    kNeedsAlpha,           // gated delta rule
    kNeedsBeta,            // delta rule
    kInitStateFromInput,   // GMEM 初始化 state vs 零初始化
    kSafeGate,             // KDA safe gate 模式
    kElementBetaGmem,      // beta GMEM 类型 (float/bf16)
};
```

#### 4.3.4 构建与消费

在 `prefill_kernel_kda_fwd_sm90.cuh:80-90` 中嵌套 `add_option` 构建:

```c++
using Options = decltype(
    add_option(Option<Tag::kElementBetaGmem, TBeta>{},
    add_option(Option<Tag::kSafeGate, SafeGateType>{},
    add_option(Option<Tag::kInitStateFromInput, InitStateType>{},
    add_option(Option<Tag::kNeedsAlpha, NeedsAlphaType>{},
    add_option(Option<Tag::kNeedsBeta, NeedsBetaType>{},
    add_option(Option<Tag::kIsDeltaRule, true_type>{}, DefaultOptions{})))))));
```

在 `mainloop_kda_fwd.hpp` 中消费:

```c++
static constexpr int NeedsAlpha = find_option_t<Tag::kNeedsAlpha, true_type, Options>::value;
// → constexpr, 配合 if constexpr 让编译器彻底消除无关分支, 零运行时开销
```

---

### 4.4 `operator()` — Kernel 入口

**文件**: `kernel_kda_fwd.hpp:229-628`

`operator()` 是每个线程进入 kernel 后执行的入口函数。它**不做计算**，只做编排: 识别身份 → 配置 Pipeline → 分发到对应的 mainloop 函数。

#### 4.4.1 阶段一: 身份识别 (L230-265)

每个线程进入 kernel 后，第一件事是搞清楚"我是谁":

```c++
int lane_idx = cutlass::canonical_lane_idx();           // 线程在 warp 内的编号 (0-31)
int warp_idx = cutlass::canonical_warp_idx_sync();      // warp 在 block 内的编号 (0-15)
int warp_idx_in_wg = warp_idx % cutlass::NumWarpsPerWarpGroup;  // warp 在 WG 内的编号 (0-3)
int warp_group_idx = cutlass::canonical_warp_group_idx();       // WG 编号 (0-3)
```

对于 512 线程的 block:

```
threadIdx:        0 ........... 127 | 128 ......... 255 | 256 ......... 383 | 384 ......... 511
warp_group_idx:          0                   1                   2                   3
warp_group_role:       LdSt               Math0               Math1               MathA

WG0 内部 (4 个 warp):
  warp_idx_in_wg:    0          1          2          3
  ldst_warp_role: LoadQKV    StoreO    LoadBeta   LoadAlpha+ExtractLast
```

`elect_one_sync()` 在每个 warp 内选出唯一 leader 线程 (通常 lane 0)，返回 1/0。TMA 操作只需 leader 发起。

Warp 0 的 leader 负责 `prefetch_tma_descriptors()` 预取 TMA 描述符到 L2 cache，减少后续 TMA 延迟。

#### 4.4.2 阶段二: Pipeline 参数配置 (L270-435)

**两类 Pipeline**:

| 类型 | 同步方式 | 用途 | 典型参数 |
|------|---------|------|---------|
| **TmaPipeline** (Q/K/V/Alpha) | `transaction_bytes` + `is_leader` + `num_consumers` | GMEM→SMEM 异步加载 | Producer=1个leader, Consumer=多个MMA线程 |
| **AsyncPipeline** (O/QK/KK/Beta/AlphaLast) | `producer_arv_count` + `consumer_arv_count` | WG间 SMEM 传递 | 多线程同时 arrive |

TmaPipeline 的特殊之处: TMA 硬件自己搬数据，只需 1 个线程发指令 (`is_leader`)。用 `transaction_bytes` 判断传输是否完成。AsyncPipeline 更朴素: 所有 producer 线程 arrive 后 consumer 才继续。

**9 条 Pipeline 的 Producer→Consumer 关系**:

| Pipeline | Producer | Consumer |
|----------|----------|----------|
| q_pipeline | LoadQKV (1 leader) | Math0+Math1+MathA (384线程) |
| k_pipeline | LoadQKV (1 leader) | Math0+Math1+MathA (384线程) |
| v_pipeline | LoadQKV (1 leader) | Math0+Math1 (256线程, MathA 不需要 V) |
| alpha_pipeline | LoadQKV (1 leader) | Math0+Math1+MathA+LoadAlpha warp (416线程) |
| o_pipeline | Math0+Math1 (256线程) | StoreO (1 warp=32线程) |
| qk_pipeline | MathA (128线程) | Math0+Math1 (256线程) |
| kk_pipeline | MathA (128线程) | Math0+Math1 (256线程) |
| alpha_last_pipeline | LoadAlpha warp (32线程) | Math0+Math1 (256线程) |
| beta_pipeline | LoadBeta warp (32线程) | MathA+Math0+Math1 (384线程) |

**特殊角色**: LoadAlpha warp (WG0.Warp3) 身兼 alpha_pipeline 的 **Consumer** 和 alpha_last_pipeline 的 **Producer**——从 SMEM 读 alpha tile 最后一行，写入单独的 buffer。

代码 L318-377 为每个角色设置 `role = Producer` 或 `role = Consumer`。每条 pipeline 有 read/write 两个游标 (PipelineState)，是环形 buffer 的读写指针。

#### 4.4.3 阶段三: 同步与角色分发 (L437-628)

```c++
__syncthreads();  // 确保所有 pipeline/barrier 初始化对全 block 可见
```

然后进入角色分发——按 WG 角色走不同分支:

```
WG0 (LdSt):
  ├─ warpgroup_reg_dealloc<24>()           ← 释放寄存器
  ├─ Warp0: load_qkv()                    ← TMA 加载 Q/K/V/α
  ├─ Warp1: store()                       ← TMA 写回 O (内部自带 tile 循环)
  ├─ Warp2: load_beta()                   ← 加载 β
  └─ Warp3: extract_alpha_last()          ← 提取 α 最后一行

WG1/WG2 (Math0/Math1):
  ├─ warpgroup_reg_alloc<168>()            ← 获取更多寄存器
  ├─ math_barriers.init(warp_group_idx-1)  ← 有序执行: Math0→init(0), Math1→init(1)
  └─ compute()                             ← 核心状态计算

WG3 (MathA):
  ├─ warpgroup_reg_alloc<152>()            ← 获取寄存器
  └─ compute_aux_safe()                    ← 辅助计算 (Q@K, K@K)
```

寄存器重分配放在 `__syncthreads()` **之后**、各角色分支的**最前面**，确保初始化完成后才改变寄存器容量。

最终 `__syncthreads()` 等所有角色完成。

#### 4.4.4 全局并发时序

```
时间 →

WG0.Warp0 (LoadQKV):  [TMA Q/K/V/α tile0 → SMEM] [TMA tile1 → SMEM] ...
                                │ q/k/v/α_pipeline
WG0.Warp2 (LoadBeta):  [GMEM→reg→SMEM β tile0]    [β tile1]          ...
                                │ beta_pipeline
WG0.Warp3 (LoadAlpha): ─wait α──[SMEM→SMEM α_last0]──wait──[α_last1] ...
                                │ alpha_last_pipeline
WG3 (MathA):           ─wait Q,K,α,β──[Q@K, K@K, exp(α)缩放 tile0]── ...
                                │ qk/kk_pipeline
WG1/2 (Math0/1):      ─wait Q,K,V,QK,KK,α_last,β──[compute tile0]── ...
                                │ o_pipeline
WG0.Warp1 (StoreO):   ─────────wait O──[TMA O tile0 → GMEM]───────── ...
```

---

### 4.5 `load_qkv()` — Q/K/V/Alpha TMA 加载

**文件**: `mainloop_kda_fwd.hpp:596-632` + `load_tma.hpp`

#### 4.5.1 整体结构

`load_qkv` 由 WG0.Warp0 (LoadQKV warp) 执行，负责把 Q/K/V/Alpha 从 GMEM 搬到 SMEM:

```c++
CUTE_DEVICE void load_qkv(...) {
    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));

    // 1. 构建 4 个 CollectiveLoadTma 对象
    auto q_collective_load = LoadQ(params.tma_load_q, q_pipeline, storage.smem_q);
    // ... k, v, alpha 同理

    // 2. GMEM 分区: 建立 src(GMEM) → dst(SMEM) 映射
    auto q_src_dst = q_collective_load.partition_SD(problem_size, load_tile_shape, work_desc);

    // 3. 逐 tile 加载
    for (int blk = 0; blk < num_blocks; ++blk) {
        alpha_collective_load.step(alpha_src_dst, blk, alpha_smem_pipe_write, lane_predicate);
        q_collective_load.step(q_src_dst, blk, q_smem_pipe_write, lane_predicate);
        k_collective_load.step(k_src_dst, blk, k_smem_pipe_write, lane_predicate);
        v_collective_load.step(v_src_dst, blk, v_smem_pipe_write, lane_predicate);
    }
}
```

`num_blocks = ceil_div(seq_len, BlkSeqQ)`: 例如 seq_len=200, BlkSeqQ=64 → 4 个 tile。

Alpha 排在循环最前面加载，因为 LoadAlpha warp 需要从 SMEM 读 alpha 最后一行写入 alpha_last_pipeline，alpha 越早到位这条路径越早启动。

#### 4.5.2 CollectiveLoadTma — 封装 TMA 加载

```c++
template <LoadKind kKind, class Pipeline, class Element, class SmemLayout, class TMA>
struct CollectiveLoadTma {
    TMA const& tma_load;       // TMA 描述符 (tensor 地址/形状/stride)
    Pipeline& pipeline;        // 对应的 TmaPipeline
    SharedStorage& storage;    // SMEM buffer
};
```

模板参数 `LoadKind` 区分 Q/K/V/Alpha，因为 GMEM 布局不同:
- Q/Alpha: `(seqlen, d, h)` — 行优先
- K/V: `(d, seqlen, h)` — 转置存储，原因: WGMMA 计算 Q@K^T 时 K 作 B 矩阵需列连续

#### 4.5.3 partition_SD — GMEM 分区

以 Q 为例 (seq_len=200, BlkSeqQ=64, HeadSize=128):

```c++
// 1. GMEM tensor: (total_seqlen, head_size, num_heads)
Tensor m_varlen_head = tma_load.get_tma_tensor(make_shape(total_seqlen, head_size, num_heads));
// 2. 切到当前 head: (total_seqlen, head_size)
Tensor m_varlen = m_varlen_head(_, _, work_desc.q_head_idx());
// 3. 偏移到当前序列起始 (tok_offset=120)
Tensor m_offset = domain_offset(make_coord(work_desc.tok_offset, _0{}), m_varlen);
// 4. 按 tile 切分: shape = (64, 128, 4) → 4 个 64×128 的 tile
Tensor g_full = local_tile(m_offset, make_tile(BlkSeqQ, HeadSize), make_coord(_, _0{}));
```

K/V 的差异: GMEM 布局是 `(head_size, total_seqlen, num_heads)` (转置), 偏移在第二维, tile 切分为 `(HeadSize, BlkSeqKV, num_tiles)`。

最后建立 TMA 的 src/dst pair:

```c++
Tensor s = make_tensor(make_smem_ptr(storage.data()), SmemLayout{});   // SMEM
auto block_tma = tma_load.get_slice(_0{});                              // 单 block (不支持 cluster)
return make_tuple(block_tma.partition_S(g), block_tma.partition_D(s));  // src=GMEM, dst=SMEM
```

#### 4.5.4 step() — 单次 TMA 加载

```c++
CUTE_DEVICE void step(SrcDst const& src_dst, int src_iter, PipelineState& dst_pipe, uint32_t lane_predicate) {
    if (lane_predicate == 1) {           // 仅 leader 线程执行
        pipeline.producer_acquire(dst_pipe);                 // 1. 等 SMEM stage 可写
        BarrierType* tma_barrier = pipeline.producer_get_barrier(dst_pipe); // 2. 获取 barrier
        copy(tma_load.with(*tma_barrier), src(src_iter), dst(dst_pipe.index())); // 3. 异步 TMA copy
        ++dst_pipe;                                          // 4. 推进写指针
    }
}
```

关键: `copy` 是**异步**的——TMA 硬件开始搬数据，线程**立即返回**。TMA 完成后自动通知 barrier，Consumer 通过 `consumer_wait` 等待。

#### 4.5.5 2-stage Pipeline 时序

```
LoadQKV warp (Producer):               Math WG (Consumer):
─────────────────────────              ───────────────────────
acquire(S0) ← 立即通过
TMA copy(GMEM→SMEM S0) ← 异步
acquire(S1) ← 立即通过                 consumer_wait(S0) ← 等 TMA 完成
TMA copy(GMEM→SMEM S1) ← 异步          读 SMEM S0, 做 MMA
acquire(S0) ← 可能阻塞(S0还没读完)      consumer_release(S0) ← 释放 S0
                                        consumer_wait(S1)
TMA copy(GMEM→SMEM S0) ← S0 已释放      读 SMEM S1, 做 MMA
...                                     consumer_release(S1)
```

#### 4.5.6 alpha_last_pipeline 的设计原因

KDA State 更新需要 `α_last` (当前 tile alpha 的最后一行, shape `(d,)`)，但整个 alpha tile `(64×128)` 和最后一行的消费时机不同: 整个 tile 在计算开始时用于 exp(α) 缩放，最后一行在计算结束时用于 State 衰减。而且 alpha SMEM buffer 会被下一 tile 的 TMA 覆盖，必须及时提取。TMA 硬件只能搬整块矩形，不能只搬一行。

因此设计专门的 LoadAlpha warp (WG0.Warp3) 做中转:

```
LoadQKV warp:        TMA load α tile → SMEM
                            │ alpha_pipeline
                     ┌──────┴──────┐
                     ▼             ▼
              MathA/Math0/1     LoadAlpha warp (WG0.Warp3)
              (消费整个tile)     consumer_wait → 从 SMEM 读最后一行
                                → 写入单独 buffer → producer_commit
                                → consumer_release
                                        │ alpha_last_pipeline
                                        ▼
                                Math0/1: 用 α_last 做 state 衰减
```

---

### 4.6 `load_beta()` — Beta 系数加载

**文件**: `mainloop_kda_fwd.hpp:634-658` + `load_predicated.hpp`

#### 4.6.1 为什么不用 TMA

Beta 是 per-token 标量, shape `(seqlen, num_heads)`。每个 tile 只有 BlkSeqQ 个值 (64 个 float = 256B)。TMA 适合搬 16KB+ 的大块数据，256B 太小反而有额外开销。所以用传统的 **predicated load** (线程直接读 GMEM)。

#### 4.6.2 CollectiveLoadVector vs CollectiveLoadTma

| 方面 | CollectiveLoadTma (Q/K/V/α) | CollectiveLoadVector (β) |
|------|---------------------------|-------------------------|
| 搬运方式 | TMA 硬件异步 | 线程手动 GMEM→reg→SMEM |
| 参与线程 | 仅 1 leader | 32 线程 (整个 warp) |
| 越界处理 | TMA 自动 | 手动 mask + oob_value 填充 |
| barrier | TMA 硬件自动通知 | 手动 fence + producer_commit |
| 构造参数 | TMA descriptor | 原始 GMEM 指针 + layout |
| 数据量/tile | ~16KB (64×128×bf16) | ~256B (64×float) |

为什么 GMEM→reg→SMEM 两跳而非直通? GPU 的普通 load 指令 (`LDG`) 只能 GMEM→寄存器，没有直达 SMEM 的路径。`cp.async` 可以直通但不支持类型转换 (beta GMEM 可能是 bf16, SMEM 要 float) 和细粒度 predicate。

#### 4.6.3 step — 单次加载

```c++
template <bool IsTail>
CUTE_DEVICE void step(SrcDst, int src_iter, PipelineState& dst_pipe, int num_iters) {
    // 1. GMEM → 寄存器 (32 线程协作)
    if constexpr (!IsTail) {
        copy(src(_, _, src_iter), regs);              // 非尾 tile: 直接 copy
    } else {
        fill(regs, src_oob_value_);                   // 尾 tile: 先填 0
        copy_if(mask, src(_, _, src_iter), regs);     // 按 mask 有条件 copy
    }

    // 2. 等 SMEM 槽位 → 写 SMEM → 通知 consumer
    pipeline_.producer_acquire(dst_pipe);
    fence_view_async_shared();                        // 确保后续 SMEM 写可见
    copy(regs, dst(_, _, dst_pipe_idx));              // reg → SMEM
    fence_view_async_shared();
    pipeline_.producer_commit(dst_pipe);              // 通知 consumer
    ++dst_pipe;
}
```

`oob_value = 0`: 越界 token 的 beta=0, 对 State 更新无贡献。

循环结构: 前 `num_blocks-1` 个用 `IsTail=false`, 最后一个用 `IsTail=true`。`IsTail` 是模板参数, 编译器为两种情况生成不同代码。

```
LoadBeta warp (WG0.Warp2, 32 threads) 时序:
────────────────────────────────────────────
GMEM → reg: 32线程各读 BlkSeqQ/32 个元素
  (尾tile: 先 fill(0), 再 copy_if(mask))
producer_acquire(dst_pipe) ← 等 SMEM stage 可写
fence_view_async_shared
reg → SMEM: 写入 beta buffer
fence_view_async_shared
producer_commit(dst_pipe)  ← 通知 MathA+Math0/1 数据就绪
++dst_pipe
```

---

### 4.7 `extract_alpha_last()` — Alpha 最后一行提取

**文件**: `mainloop_kda_fwd.hpp:660-707`

#### 4.7.1 功能与执行者

**WG0.Warp3 (LoadAlpha warp)**, 32 线程。从 alpha tile `(BlkSeqQ, HeadSize, stages)` 中提取最后有效行, 写入 alpha_last buffer `(HeadSize, stages)`。它同时是 alpha_pipeline 的 consumer 和 alpha_last_pipeline 的 producer。

#### 4.7.2 双 Pipeline 协调

```c++
alpha_pipeline.consumer_wait(alpha_smem_pipe_read);          // 等 alpha tile TMA 完成
alpha_last_pipeline.producer_acquire(alpha_last_smem_pipe_write); // 等 alpha_last stage 可写
```

两个条件都满足后才开始工作。

#### 4.7.3 数据拷贝

```c++
int B = is_final_block ? valid_seq_len(work_desc, blk) : BlkSeqKV;  // 有效行数
CUTE_UNROLL
for (int t = thread_idx; t < HeadSize; t += 32) {
    sAlast_out(t) = sAqkq_curr(B - 1, t);   // 取第 B-1 行 (最后有效 token)
}
```

32 线程 stride 循环, 每线程搬 128/32 = 4 个 float。尾 tile 时 B 可能小于 BlkSeqQ (如 seq_len=200, 最后 tile B=8, 取 row 7)。

#### 4.7.4 释放顺序 (关键)

```c++
fence_view_async_shared();
alpha_last_pipeline.producer_commit(alpha_last_smem_pipe_write);  // 先通知下游
++alpha_last_smem_pipe_write;
alpha_pipeline.consumer_release(alpha_smem_pipe_read);             // 再释放上游
++alpha_smem_pipe_read;
```

**顺序不能反**: 如果先 release alpha stage, TMA 可能立即开始加载下一 tile 的 alpha, 覆盖正在读的 SMEM 数据。必须先 commit (确保数据已写完) 再 release。

```
单次迭代时序:
────────────────────────────────────────────
alpha_pipeline.consumer_wait(read)      ← 等 alpha tile TMA 完成
alpha_last_pipeline.producer_acquire(write) ← 等 alpha_last stage 可写
SMEM→SMEM: sAlast_out(t) = sAqkq_curr(B-1, t)  (32 线程 stride)
fence + producer_commit(write)          ← 通知 Math0/1
consumer_release(read)                  ← 释放 alpha stage, TMA 可加载下一 tile
```

---

### 4.8 `store()` — O 矩阵 TMA 写回

**文件**: `mainloop_kda_fwd.hpp:709-735` + `store_tma.hpp`

#### 4.8.1 结构与执行者

**WG0.Warp1 (StoreO warp)**, 32 线程, 仅 leader 发 TMA store 指令。它是 o_pipeline 的 **Consumer** (等 Math WG 把 O 写入 SMEM)。

```c++
CUTE_DEVICE void store(...) {
    for (int blk = 0; blk < num_blocks; ++blk) {
        collective_store.step(problem_size, work_desc, src_dst, smem_pipe_read, blk, num_blocks, lane_predicate);
    }
}
```

O 的 GMEM 布局 `(head_size, total_seqlen, num_heads)` 和 K/V 一样转置。partition_SD 中 src=SMEM, dst=GMEM (与 load 反过来)。

#### 4.8.2 尾 tile 问题

TMA load 越界读到垃圾无害 (会被 mask 掉)。但 TMA **store 越界写是致命的** — 会覆盖下一条序列的数据。

`can_process()` 判断是否安全:
```c++
if (blk < num_blocks - 1) return true;      // 非尾 tile, 安全
if (seq_len % SizeN == 0) return true;       // 尾 tile 恰好满, 安全
if (seq_idx == num_seqs - 1) return true;    // 最后一条序列, 后面无数据, 安全
return false;                                 // 不安全: 不满的尾 tile 且后面还有序列
```

#### 4.8.3 create_tensormap_for_tail — 运行时修改 TMA descriptor

不安全时, 动态修改 TMA descriptor 截断 seqlen 维:

```c++
// 1. Warp 并行复制原始 TMA descriptor 到 per-SM workspace
cute::TmaDescriptor* tensormap = static_cast<cute::TmaDescriptor*>(tensormaps_) + smid();
// ... 多线程并行复制 128B ...

// 2. Leader 用 PTX 指令修改 seqlen 维大小
ptx::tensormap_replace_global_dim(ptx::space_global, tensormap, ptx::n32_t<1>{}, new_total_seqlen);

// 3. Memory fence 确保 TMA 硬件看到修改
ptx::fence_proxy_tensormap_generic(ptx::sem_release, ptx::scope_cta);
```

修改后 TMA 即使写满整个 tile, 也不会超过 `tok_offset + seq_len` 的边界。每 SM 有自己的副本 (`+ smid()` 索引) 避免冲突。

#### 4.8.4 完整时序

```
StoreO warp (WG0.Warp1):
─────────────────────────────────────────────
dst_iter=0:
  can_process(tail)? No → create_tensormap_for_tail()
    warp并行复制TMA desc → 修改seqlen dim → fence release

每个 tile:
  o_pipeline.consumer_wait(read)  ← 等 Math0/1 写完 O
  can_process?
    Yes → copy(tma_store_, SMEM → GMEM)  (leader only)
    No  → acquire tail tensormap → copy with modified desc
  o_pipeline.consumer_release(read) ← 释放 SMEM stage
  ++src_pipe
```

---

### 4.9 `compute_aux_safe()` — 辅助 MMA (QK, KK 计算)

**文件**: `mainloop_kda_fwd.hpp:1376-2052`

#### 4.9.1 数学背景

对应公式 (1.2)-(1.7)。计算 gated 的 QK 和 KK 矩阵 (safe_gate 模式, anchor = gc[0]):

```
Q̃[i,:] = Q[i,:] * exp2(gc[i,:] - gc[0,:])     ← gated Q
K̃[j,:] = K[j,:] * exp2(gc[0,:] - gc[j,:])     ← gated K
QK = Tril(Q̃ @ K̃^T) * scale                     ← 下三角 attention 分数
KK = StrictTril(K̃ @ K̃^T) * diag(β)             ← 严格下三角 + beta 缩放
```

结果写入 SMEM 的 `smem_qk` 和 `smem_kk`，通过 `qk_pipeline` 和 `kk_pipeline` 通知 Math0/1。

#### 4.9.2 执行者与整体结构

**WG3 (MathA)**, 128 线程。3 层嵌套:

```
compute_aux_safe()
  └─ compute_aux_loop_body(blk)              ← 每个 tile
       ├─ qk_kk_subchunk_mma_and_store()     ← 16×16 子块 MMA (核心)
       │    ├─ s2r_compute_subchunk_operandA()  ← 加载 Q/K + α gating + 布局转换
       │    ├─ s2r_compute_subchunk_operandB()  ← 加载 K^T + α gating + 布局转换
       │    ├─ gemm() × 2                       ← QK 和 KK 同时累加
       │    └─ r2s_subchunk_acc()               ← 结果写回 SMEM
       ├─ NamedBarrier(AuxMath)              ← 两组 warp 同步
       ├─ qk_and_kk_epi()                   ← S2R→三角mask+边界mask→R2S
       └─ pipeline commit + release
```

#### 4.9.3 SubChunk 策略 — 为什么用 16×16?

64×64 的 QK/KK 矩阵, 如果一次算完, 每个 accumulator 需要 `64×64/64线程 = 64` 个 fp32 reg，两份 (QK+KK) 就是 128 个, 超出 MathA 的 152 预算 (还需要操作数寄存器)。

切成 **16×16 子块**: 每个只需 `16×16/64线程 = 4` 个 fp32 reg，两份仅 8 个。

```
64×64 矩阵, 4×4 = 16 个 16×16 子块, 只算下三角+对角 = 10 个:

     col0  col1  col2  col3
row0 [D   ] [0   ] [0   ] [0   ]      D = 对角块 (含对角线)
row1 [L   ] [D   ] [0   ] [0   ]      L = 严格下三角块
row2 [L   ] [L   ] [D   ] [0   ]      0 = 上三角 (零填充)
row3 [L   ] [L   ] [L   ] [D   ]
```

#### 4.9.4 Warp 分工

128 线程分两组各 64 线程 (`thread_idx < 64` vs `>= 64`), 并行计算不同子块:

```
Warp 0+1 (thread 0-63):   row0@col0(对角), row3@col0, row3@col1, row3@col2, row3@col3  (5 块)
Warp 2+3 (thread 64-127): row1@col0, row1@col1, row2@col0, row2@col1, row2@col2         (5 块)
```

每组用 `TiledMma_SubChunk` (SM80_16x8x8_F32TF32TF32F32_TN, 64 线程) 做 TF32 MMA。选 TF32 而非 BF16 是因为 gating 后结果是 float, TF32 有 10-bit mantissa (vs BF16 的 7-bit), 精度更高。

#### 4.9.5 S2R 优化: BF16 LDSM + warp shuffle

加载 Q/K/α 的策略: 用 BF16 的 `ldmatrix` (LDSM) 加载到 BF16 MMA 布局 → 在 BF16 布局中做 gating (element-wise exp2 * Q/K) → 通过 **8 次 warp shuffle** 转换到 TF32 MMA 布局。

```c++
// 注释: This replaces the previous AutoVectorizingCopy<16> which caused 50% more smem traffic.
```

BF16 的 `ldmatrix` 是专用指令, 一次搬 16B 且天然匹配 MMA 寄存器布局。虽然最终要做布局转换 (8 次 shuffle), 但比多出 50% 的 SMEM 带宽消耗更划算。

`α_first` 提取也很巧妙: 从 operand A 的 row=0 通过 warp shuffle (`broadcast_row0_operandA_to_operandB_bf16_layout`) 广播到 operand B 布局, 节省额外的 S2R 加载。

#### 4.9.6 qk_and_kk_epi — 三角 mask

子块 MMA 完成后, 128 线程通过 `NamedBarrier::arrive_and_wait(AuxMath)` 同步。然后全 128 线程做 epilogue:

1. S2R: 从 SMEM 读回完整 64×64 的 QK 和 KK
2. 下三角 mask: `s >= t` 保留, `s < t` 填 0
3. 尾 tile 边界 mask: `s < B && t < B` 保留
4. QK 缩放: `QK *= scale`
5. KK 缩放: `KK[i,j] *= beta[i]` (per-row)
6. R2S: 写回 SMEM

#### 4.9.7 Pipeline 操作与时序

```
MathA WG (128 threads) per tile:
────────────────────────────────────
[q/k/α pipeline consumer_wait — 在第一个子块时]
[β pipeline consumer_wait — 在 epilogue 时]
[qk/kk pipeline producer_acquire — 在第一个子块时]

Warp 0+1:                    Warp 2+3:
  row0@col0 (对角)             row1@col0, row1@col1
  row3@col0,col1,col2,col3    row2@col0,col1,col2

── NamedBarrier(AuxMath) ── 两组同步

全128线程: S2R → 三角mask + 边界mask + scale/beta → R2S

fence + qk/kk_pipeline.producer_commit → Math0/1 可以消费
q/k/α/β_pipeline.consumer_release → TMA 可加载下一 tile
```

---

### 4.10 `compute()` — 主状态计算

**文件**: `mainloop_kda_fwd.hpp:737-1375`

`compute()` 是整个 kernel 的核心, 由 Math0 和 Math1 两个 WG (共 256 线程) 执行。对每个 tile 执行 6 个步骤 (Step A-F), 涉及 6 种 MMA 和多个 pipeline 的协调。

#### 4.10.1 算法结构总览

```
对每个 tile (compute_loop_body):
  Step A: Q/K Prologue  → Q_scaled, K_scaled 写入 SMEM      公式 (2.1)(2.2)
  Step B: O_inter       → Q_scaled @ S_old → 寄存器          公式 (2.3)
  Step C: V'→NewV       → S@K, V-SK, KK逆, V'@T^{-T}       公式 (2.4)-(2.8)
  Step D: O_intra       → NewV^T @ QK, O 写入 SMEM          公式 (2.9)(2.10)
  Step E: State 衰减    → S *= exp(α_last), K_decay          公式 (2.11)(2.12)
  Step F: State 更新    → S += NewV^T @ K_decay^T            公式 (2.13)

外层: tile 0 (首tile特殊) → tile 1..N-2 → tile N-1 (尾tile)
最后: kv_store() 把最终 State 写回 GMEM
```

关键数据: **State `tKVrKV`** 始终驻留在寄存器中, shape (d, d) = (128, 128), 是寄存器消耗最大的变量, 贯穿整个 mainloop 不释放。

#### 4.10.2 Step A: Q/K Prologue — 公式 (2.1)(2.2)

**代码**: L1025-1096

**做什么**: 计算 `Q_scaled = Q ⊙ exp(gc)` 和 `K_scaled = K ⊙ exp(gc)`, 写入 SMEM 的 `sQ_K_scaled` (stage 0 = Q_scaled, stage 1 = K_scaled)。

**首 tile 跳过**: State=0 时 Q@0=0, 无需 Q_scaled。

**NamedBarrier 保护**: `sQ_K_scaled` 在上一 tile 的 Step F 中被 WGMMA 读取, 必须等读完才能覆写:

```c++
cutlass::arch::NamedBarrier::arrive_and_wait(NumStateMmaThreads, KdaNamedBarriers::StateMath);
```

**两个 WG 分工处理 HeadSize**:

```c++
int wg_idx = thread_idx / 128;  // WG0=0, WG1=1
int alpha_base = wg_idx * 2;    // WG0 → slice {0,1} = head_dim [0:64), WG1 → slice {2,3} = [64:128)
```

HeadSize=128 切成 4 个 32 元素的 slice (_Quar), 减少寄存器压力。每个 slice 的处理:

```c++
for (int s = 0; s < 2; ++s) {
    // 1. S2R Alpha (32 float) → exp2(α)
    copy(CopyAlphaAtom{}, sA_cur, tArA);
    cute::transform(tArA, [](auto g) { return exp2f(g); });

    // 2. S2R Q (32 bf16) → Q * exp(α) → R2S stage 0
    copy(tiled_load_qk_quar, sQqk_cur, tQKrQ_cv);
    cute::transform(tQKrQ_wg, tArA, tQKrQ_wg, [](auto q, auto a) { return Element(a * float(q)); });
    copy(tiled_store_qk_quar, tQKrQ_out_cv, tQKsQ_out);

    // 3. 复用 α → S2R K → K * exp(α) → R2S stage 1
    // ... 同样流程 ...
}
```

结束时两个 WG 同步 + fence:

```c++
cutlass::arch::NamedBarrier::arrive_and_wait(NumStateMmaThreads, KdaNamedBarriers::StateMath);
cutlass::arch::fence_view_async_shared();  // 确保 WGMMA 异步代理可见
```

```
Step A 时序:
wait α/Q/K pipeline → NamedBarrier(上tile完成)
WG0: slice0,1 (head_dim 0:64)    WG1: slice2,3 (head_dim 64:128)
  S2R α→exp2→S2R Q→Q*exp(α)→R2S    同
  复用α→S2R K→K*exp(α)→R2S          同
NamedBarrier → fence → 进入 Step B
```

#### 4.10.3 Step B: O_inter — 公式 (2.3)

**代码**: L1098-1135

**做什么**: `O_inter = scale * Q_scaled @ S_old` (论文 Eq.9 第一项, 旧 State 对输出的贡献)。

**首 tile 特殊**: State=0, 跳过 WGMMA, 直接 release Q pipeline。

**非首 tile 的 WGMMA 流程**:

```c++
// 1. State 寄存器 → MMA A 操作数 (fp32 accumulator → bf16, 重排布局)
Tensor tOrKV = make_acc_into_op<Element>(tKVrKV, typename TiledMmaO1::LayoutA_TV{});

// 2. Fence: 寄存器对 WGMMA 异步代理可见
warpgroup_fence_operand(tOrKV);
warpgroup_fence_operand(tOrO);

// 3. 有序执行: Math0 先, Math1 等
math_barriers.ordered_or_wait(warpgroup_idx);

// 4. WGMMA RS 模式: A=State(reg), B=Q_scaled(SMEM stage0)
warpgroup_arrive();
gemm_zero_acc(o1_thr_mma, tOrKV, tOrQ(_, _, _, 0), tOrO);  // C 清零后累加
warpgroup_commit_batch();
math_barriers.notify_next_blocked(warpgroup_idx);            // 通知另一个 WG

// 5. 等待 + 后处理
warpgroup_wait<0>();              // 等 WGMMA 完成
q_pipeline.consumer_release(...); // 释放 Q SMEM
o1_epi(tOrO);                     // O_inter *= scale (乘 d^{-0.5})
```

**math_barriers 有序执行**: Math0 和 Math1 按序访问 SMEM 中的 Q_scaled。WGMMA 的 B 操作数是共享的 SMEM, 两个 WG 同时读可能造成 bank conflict。

```
Step B 时序:
Math0: fence → ordered_wait(0) → WGMMA → commit → notify → wait → release Q → *=scale
Math1:                           (等待)           → ordered_wait(1) → WGMMA → ... → *=scale
```

#### 4.10.4 Step C: V'→NewV — 公式 (2.4)-(2.8) (L1137-1218)

  **对应的数学**: 论文 Eq.9 中 "pseudo-value term" `U - W@S`:
  ```
  SK     = S_old @ K_scaled^T              ... (2.4) 对应 W@S (融合形式)
  V'     = V - SK                          ... (2.5) 对应 U - W@S
  T^{-1} = forward_substitution(I + KK)   ... (2.6)(2.7)
  NewV   = V' @ T^{-T}                    ... (2.8) 对应 T^{-1} @ V'
  ```
  直觉: NewV 是"修正后的 V"——从原始 V 中减去旧 State 已经记住的部分 (SK)，再经过下三角求逆 (解耦 token 间依赖)。

  ##### C1: SK = S_old @ K_scaled^T (L1137-1156)

  ```c++
  auto tSKrSK = partition_fragment_C(sk_thr_mma, sVkv(_, _, _0{}));  // accumulator (dv, BlkSeqKV)
  if constexpr (!is_first_block) {
      auto tSKrS = make_acc_into_op<Element>(tKVrKV, typename TiledMmaSK::LayoutA_TV{});
      warpgroup_fence_operand(tSKrSK);
      warpgroup_fence_operand(tSKrS);
      math_barriers.ordered_or_wait(warpgroup_idx);
      warpgroup_arrive();
      gemm_zero_acc(sk_tiled_mma, tSKrS, tSKrK(_, _, _, 1), tSKrSK);
      warpgroup_commit_batch();
      math_barriers.notify_next_blocked(warpgroup_idx);
      warpgroup_wait<0>();
  }
  ```

  - WGMMA RS 模式: A=State(reg), B=K_scaled(**SMEM stage 1**, Step A 写入), C=tSKrSK(reg)
  - `tSKrK(_, _, _, 1)`: B 操作数从 `sQ_K_scaled` 的 stage 1 读取 (K_scaled)
  - 形状: `TileShapeSK = (HeadSizeV=128, BlkSeqKV=64, HeadSizeQK=128)`
    ```
    SK = State(128,128) @ K_scaled(128,64)^T → SK(128,64)
    ```
  - math_barriers 有序执行 (同 Step B: Math0 先, Math1 等)
  - 首 tile (State=0): 跳过 WGMMA, `tSKrSK` 保持未初始化

  ##### C2: V' = V - SK (L1163-1170)

  ```c++
  v_pipeline.consumer_wait(v_smem_pipe_read);                 // 等 V 到 SMEM
  auto tSKrV = sk_load_v(v_smem_pipe_read.index());           // S2R: V 从 SMEM 到寄存器
  if constexpr (!is_first_block) {
      transform(tSKrV, tSKrSK, tSKrV, [](auto v, auto sk) { return v - Element(sk); });
  }
  ```

  `sk_load_v` 使用 `SM75_U16x8_LDSM_T` (LDSM 转置模式) 加载 V。转置 `_T` 是因为 V 在 SMEM 中的布局和 SK MMA 的 C accumulator 布局需要匹配，用转置 LDSM 一步完成布局转换。

  首 tile: V' = V (不做减法, `tSKrSK` 未初始化但不参与运算)。

  ##### C3: KK 矩阵求逆 — T^{-1} (L1172-1180)

  ```c++
  kk_pipeline.consumer_wait(kk_smem_pipe_read);      // 等 MathA 的 KK 到 SMEM
  beta_pipeline.consumer_wait(beta_smem_pipe_read);   // 等 β 到 SMEM
  cutlass::arch::fence_view_async_shared();

  if (warpgroup_idx == 0) {          // 只有 Math0 执行求逆
      kk_inv(kk_smem_pipe_read);
  }
  // 两个 WG 同步, 等求逆完成
  cutlass::arch::NamedBarrier::arrive_and_wait(NumStateMmaThreads, KdaNamedBarriers::StateMath);
  ```

  **为什么只有 Math0 做?** `CollectiveInverse` 是 in-place forward substitution, 直接修改 SMEM。两个 WG 同时写同一块 SMEM 会冲突。128 线程足够完成 64×64 下三角求逆。

  **kk_inv 内部** (L962-995) 分两步:

  **Step 1: Forward substitution**
  ```c++
  auto collective_inverse = CollectiveInverse(KdaNamedBarriers::StateMathWG0);
  collective_inverse.compute(sKK_inv_pipe_slice);  // in-place 修改 SMEM 中的 KK → T^{-1}
  ```

  **Step 2: β 后处理 (如果 NeedsBeta=true)**
  ```c++
  // WG0 内部同步 (等 inverse 写 SMEM 完成)
  NamedBarrier::arrive_and_wait(128, KdaNamedBarriers::StateMathWG0);
  // S2R: 读回 T^{-1}
  copy(tiled_load_kk, ..., tKKrKK_cpy);
  // 逐元素列缩放: T^{-1}[i,j] *= Beta[j]
  cute::transform(tKKrKK_cpy, tKKcMkk_cv, tKKrKK_cvt, [&](auto val, auto coord) {
      auto [_, t] = coord;
      return Element(float(val) * Beta(t, beta_smem_pipe_read.index()));
  });
  // R2S: 写回 SMEM
  copy(tiled_store_kk, tKKrKK_cvt, recast<Element>(tKKsKK));
  ```

  **为什么乘 β?** 论文 Eq.6: `M = (I + KK)^{-1} diag(β)`。`compute_aux_safe` 已经在 KK 中乘了 per-row β, 这里的后处理是 per-column 再乘 β, 对应 `M` 右边的 `diag(β)` 因子。

  ##### C4: NewV = V' @ T^{-T} (L1182-1218)

  ```c++
  auto tNewVrA = make_acc_into_op<Element>(tSKrV, typename TiledMmaNewV::LayoutA_TV{});
  auto tNewVrC = partition_fragment_C(newv_thr_mma, select<0, 1>(TileShapeNewV{}));
  warpgroup_fence_operand(tNewVrA);
  warpgroup_fence_operand(tNewVrC);
  math_barriers.ordered_or_wait(warpgroup_idx);
  warpgroup_arrive();
  gemm_zero_acc(o1_thr_mma, tNewVrA, tNewVrB(_, _, _, kk_smem_pipe_read.index()), tNewVrC);
  warpgroup_commit_batch();
  math_barriers.notify_next_blocked(warpgroup_idx);
  warpgroup_wait<0>();
  ```

  - WGMMA RS: A=V'(reg, via `make_acc_into_op`), B=T^{-1}(SMEM, kk_inv 的结果)
  - `TileShapeNewV = (HeadSizeV=128, BlkSeqKV=64, BlkSeqKV=64)`
    ```
    NewV = V'(128,64) @ T^{-1}(64,64)^T → NewV(128,64)
    ```
  - `tNewVrC` shape (dv=128, BlkSeqKV=64), 和 V' 相同

  **Pipeline 释放** (L1210-1218):
  ```c++
  ++v_smem_pipe_read;  // NOTE: 先 ++ 再 release (注释说反过来会 race condition)
  v_pipeline.consumer_release(v_smem_pipe_read);
  kk_pipeline.consumer_release(kk_smem_pipe_read);
  ++kk_smem_pipe_read;
  beta_pipeline.consumer_release(beta_smem_pipe_read);
  ++beta_smem_pipe_read;
  ```
  V、KK、β 的 SMEM 在此释放, 允许 TMA/MathA 加载/写入下一 tile。

  ##### Step C 完整时序

  ```
  --- C1: SK = State @ K_scaled^T (WGMMA RS) ---
  Math0: fence → ordered_wait(0) → WGMMA(State × K_scaled_stage1) → notify → wait
  Math1:                           (等待)                           → WGMMA → wait

  --- C2: V' = V - SK ---
  wait v_pipeline ← V 到 SMEM
  LDSM_T 加载 V → 寄存器 (tSKrV)
  V' = V - SK (element-wise, 寄存器内)

  --- C3: KK 求逆 ---
  wait kk_pipeline, beta_pipeline ← KK, β 到 SMEM
  fence
  Math0 only: kk_inv()
    → CollectiveInverse.compute() (forward substitution, in-place SMEM)
    → WG0 内 NamedBarrier(StateMathWG0) 同步
    → S2R T^{-1} → *= Beta[col] → R2S
  NamedBarrier(StateMath) ← Math0+Math1 同步, T^{-1} 就绪

  --- C4: NewV = V' @ T^{-T} (WGMMA RS) ---
  Math0: fence → ordered_wait(0) → WGMMA(V' × T^{-1}) → notify → wait
  Math1:                           (等待)                → WGMMA → wait

  release v/kk/beta pipeline → 进入 Step D
  ```
  #### 4.10.5 Step D: O_intra + O 写出 — 公式 (2.9)(2.10)

  **代码:** L1220-1245

  **对应的数学**

  论文 Eq.9 第二项 (intra-chunk output):

  $$O_{\text{intra}} = \text{NewV}^T \cdot QK \quad \text{(2.9)} \quad \text{对应伪代码 L54 "A @ v\_i"}$$

  $$O = O_{\text{inter}} + O_{\text{intra}} \quad \text{(2.10)} \quad \text{首 tile 时 } O = O_{\text{intra}} \text{ only}$$

  **代码解析**

  ```cpp
  // 等 MathA 的 QK 矩阵到 SMEM
  qk_pipeline.consumer_wait(qk_smem_pipe_read);

  // NewV 寄存器 (Step C 产出) → O2 MMA 的 A 操作数
  // 注意变量名 tOrV_or_tKVrV: 这个 tensor 先用于 Step D (作为 O2 的 A), 后续 Step F 复用作为 KV 更新的 A
  auto tOrV_or_tKVrV = make_acc_into_op<Element>(tNewVrC, typename TiledMmaKV::LayoutA_TV{});
  warpgroup_fence_operand(tOrV_or_tKVrV);
  warpgroup_fence_operand(tOrO);

  math_barriers.ordered_or_wait(warpgroup_idx);
  warpgroup_arrive();
  // WGMMA: NewV^T @ QK
  if constexpr (is_first_block) {
      gemm_zero_acc(o2_tiled_mma, tOrV_or_tKVrV, tOrQK(_, _, _, qk_smem_pipe_read.index()), tOrO);
      //                                          ^B = QK (SMEM, MathA 产出)       ^C 清零后累加 (O_inter=0)
  } else {
      gemm(o2_tiled_mma, tOrV_or_tKVrV, tOrQK(_, _, _, qk_smem_pipe_read.index()), tOrO);
      //   ^不清零, 在 O_inter 基础上累加  → O = O_inter + O_intra
  }
  warpgroup_commit_batch();
  math_barriers.notify_next_blocked(warpgroup_idx);
  warpgroup_wait<0>();
  ```

  关键区别: 首 tile 用 `gemm_zero_acc` (清零 C 再累加)，非首 tile 用 `gemm` (在已有 O_inter 上累加)。

  - **首 tile:** `tOrO = 0 + NewV^T @ QK = O_intra` (因为 O_inter=0, Step B 跳过了)
  - **非首 tile:** `tOrO = O_inter + NewV^T @ QK = O_inter + O_intra`

  形状: `TileShapeO2 = (HeadSizeV=128, BlkSeqQ=64, BlkSeqKV=64)`

  ```
  O_intra = NewV^T       @ QK
            (128, 64)^T    (64, 64)
          = (64, 128)^T    @  ...
          → 实际 WGMMA: (128, 64) @ (64, 64)  → (128, 64)
  ```

  `tOrO` 最终形状 `(dv=128, BlkSeqQ=64)`，存放完整的输出 O。

  **Pipeline 释放与 O 写入 SMEM**

  ```cpp
  qk_pipeline.consumer_release(qk_smem_pipe_read);  // 释放 QK SMEM
  ++qk_smem_pipe_read;
  o_store(tOrO);                                      // O 写入 SMEM, 通知 StoreO warp
  ```

  `o_store` lambda (L948-960):

  ```cpp
  auto o_store = [&](auto tOrO) {
      // 1. fp32 → bf16 类型转换
      auto tOrO_cvt = make_fragment_like<ElementO>(tOrO);
      copy(tOrO, tOrO_cvt);

      // 2. 等 O 的 SMEM stage 可写 (StoreO warp 已 TMA 写完上一个)
      o_pipeline.producer_acquire(o_smem_pipe_write);

      // 3. 寄存器 → SMEM (使用 R2S copy)
      Tensor tOrO_cvt_cv = thr_copy_o.retile_S(tOrO_cvt);
      cutlass::arch::fence_view_async_shared();
      copy(tiled_copy_o, tOrO_cvt_cv, tOsO(_, _, _, o_smem_pipe_write.index()));
      cutlass::arch::fence_view_async_shared();

      // 4. 通知 StoreO warp: O 数据就绪
      o_pipeline.producer_commit(o_smem_pipe_write);
      ++o_smem_pipe_write;
  };
  ```

  > **注意:** `o_store` 中 Math0/1 是 `o_pipeline` 的 **Producer** (之前都是 Consumer)。写完后 StoreO warp (consumer) 通过 TMA 把 O 从 SMEM 搬到 GMEM。

  #### 4.10.6 Step E: State 衰减 — 公式 (2.11)(2.12)

  **代码:** L1248-1318

  **对应的数学**

  论文 Eq.8 第一项和 State 更新的准备:

  $$S_{\text{decayed}} = S_{\text{old}} \odot \exp(g_c[-1,:]) \quad \text{(2.11)} \quad \text{旧 State 乘以总衰减}$$

  $$K_{\text{decay}} = K \odot \exp(g_c[-1,:] - g_c) \quad \text{(2.12)} \quad \text{从第 i 步到 chunk 末尾的部分衰减}$$

  **E1: State 衰减 — s_decay (L1252-1256)**

  ```cpp
  if constexpr (NeedsAlpha) {
      alpha_last_pipeline.consumer_wait(alpha_last_smem_pipe_read);  // 等 α_last 到 SMEM
      cutlass::arch::fence_view_async_shared();
  }
  s_decay(tKVrKV, alpha_last_smem_pipe_read);  // State *= exp(α_last)
  ```

  `s_decay` lambda (L932-939):

  ```cpp
  auto s_decay = [&](auto& tKVrKV, auto const& alpha_last_smem_pipe_read) {
      Tensor alpha_last_curr = AlphaLast(_, alpha_last_smem_pipe_read.index());
      for_each(make_int_sequence<size(tKVcS)>{}, [&](auto i) {
          auto coord = tKVcS(i);
          auto [s, t] = coord;             // (head_size_v, head_size_k)
          tKVrKV(i) *= exp2f(alpha_last_curr(t));  // 按 K 维度 (列) 衰减
      });
  };
  ```

  `alpha_last_curr(t)`: 从 SMEM 读取 α_last 的第 t 个元素。State 矩阵 `(dv, dk)` 的每一列乘以 `exp(α_last[t])`。由于 `α_last = gc[-1,:]`，gc 中的值是负的 (log 域, α∈(0,1))，所以 `exp(gc[-1]) ∈ (0,1)`，State 确实在衰减。

  **E2: K_decay 计算 (L1258-1313)**

  和 Step A 的 Q/K Prologue 结构几乎一样，但缩放因子不同:

  - **Step A:** `exp(gc)` — 从 chunk 开头到第 i 步的累积
  - **Step E:** `exp(gc[-1] - gc)` — 从第 i 步到 chunk 末尾的部分衰减

  ```cpp
  // 先同步: 确保上一步的 WGMMA 不再读 sQ_K_scaled (Step B 用了 stage 0 的 Q_scaled)
  cutlass::arch::NamedBarrier::arrive_and_wait(NumStateMmaThreads, KdaNamedBarriers::StateMath);

  // 两个 WG 各处理 HeadSize 的一半
  int wg_idx = thread_idx / 128;   // 0 or 1
  int alpha_base = wg_idx * 2;     // 0 or 2

  for (int s = 0; s < 2; ++s) {
      // S2R Alpha
      copy(CopyAlphaAtom{}, sA_cur, tArA_wg);

      // S2R K
      copy(tiled_load_qk_quar, sKqk_cur, tQKrK_cv);

      // element-wise: K_decay = exp(α_last - α) * K
      for_each(make_int_sequence<size(tQcMq_quar)>{}, [&](auto i) {
          auto coord = tQcMq_quar(i);
          auto [seq, t] = coord;
          auto alpha = tArA_wg(i);                    // gc[i, d] (当前 token 的 alpha)
          auto k = tQKrK_wg(i);
          auto alpha_last = alpha_last_cur(t);         // gc[-1, d] (chunk 最后 token 的 alpha)
          auto k_scaled = Element(exp2f(alpha_last - alpha) * float(k));
          tQKrK_wg(i) = k_scaled;
          if constexpr (is_final_block) {
              if (seq >= B) { tQKrK_wg(i) = Element(0.0f); }  // 尾 tile 越界 token 清零
          }
      });

      // R2S K_decay → sQ_K_scaled stage 0 (覆写之前的 Q_scaled)
      copy(tiled_store_qk_quar, tQKrK_out_cv, tQKsK_out);
  }
  ```

  关键差异 vs Step A:

  | | Step A | Step E |
  |---|---|---|
  | **缩放公式** | `exp(gc[i])` | `exp(gc[-1] - gc[i])` |
  | **含义** | 从开头到第 i 步 | 从第 i 步到末尾 |
  | **写入位置** | stage 0: Q_scaled, stage 1: K_scaled | stage 0: K_decay (覆写 Q_scaled) |
  | **越界处理** | 无 | 尾 tile `seq >= B` 清零 |
  | **α 来源** | gc (SMEM alpha tile) | gc (SMEM alpha tile) + α_last (SMEM alpha_last) |

  写入 stage 0: K_decay 覆写了之前的 Q_scaled。这是安全的，因为 Q_scaled 在 Step B 之后就不再需要了。

  结束后同步 + fence + 释放 alpha_last:

  ```cpp
  cutlass::arch::NamedBarrier::arrive_and_wait(NumStateMmaThreads, KdaNamedBarriers::StateMath);
  cutlass::arch::fence_view_async_shared();   // 确保 WGMMA 可见

  alpha_last_pipeline.consumer_release(alpha_last_smem_pipe_read);
  ++alpha_last_smem_pipe_read;
  ```

  #### 4.10.7 Step F: State 更新 — 公式 (2.13)

  **代码:** L1320-1346

  **对应的数学**

  论文 Eq.8 第二项:

  $$S_{\text{new}} = S_{\text{decayed}} + \text{NewV}^T \cdot K_{\text{decay}}^T \quad \text{(2.13)}$$

  Step E 已经完成了 S_decayed (in-place 修改 tKVrKV) 和 K_decay → SMEM stage 0。

  **代码解析**

  ```cpp
  // NewV 寄存器 (Step C 产出, Step D 已用过但寄存器值仍在) → KV MMA 的 A 操作数
  // 注意: tOrV_or_tKVrV 在 Step D 中已经被 make_acc_into_op 转换过, 这里复用
  warpgroup_fence_operand(tOrV_or_tKVrV);     // A = NewV^T (reg)
  warpgroup_fence_operand(tKVrKV);             // C = S_decayed (reg, accumulator)

  math_barriers.ordered_or_wait(warpgroup_idx);
  warpgroup_arrive();
  // WGMMA: S_new = S_decayed + NewV^T @ K_decay^T
  // 注意: 用 gemm (不是 gemm_zero_acc), 在 S_decayed 基础上累加
  gemm(kv_tiled_mma, tOrV_or_tKVrV, tKVrK(_, _, _, 0), tKVrKV);
  //   ^A=NewV(reg)   ^B=K_decay(SMEM stage0)   ^C=S_decayed(reg, 累加)
  warpgroup_commit_batch();
  math_barriers.notify_next_blocked(warpgroup_idx);
  warpgroup_wait<0>();
  ```

  形状: `TileShapeKV = (HeadSizeV=128, HeadSizeQK=128, BlkSeqKV=64)`

  ```
  S_new  = S_decayed  + NewV^T      @ K_decay^T
          (128, 128)    (128, 64)^T   (128, 64)^T
                      = (64, 128)     @ (64, 128)^T
                      → 实际 WGMMA: (128, 128, 64)
  ```

  结果直接累加到 `tKVrKV`，即 State 更新完毕——`tKVrKV` 现在是 S_new，带入下一个 tile 的 Step A。

  **tOrV_or_tKVrV 复用:** 这个变量在 Step D 中被 `make_acc_into_op` 从 `tNewVrC` 转换而来。Step F 复用同一个寄存器 tensor (值没变)。变量名 `tOrV_or_tKVrV` 暗示了它的双重用途: Step D 中是 "O 的 V 操作数"，Step F 中是 "KV 更新的 V 操作数"。

  **Pipeline 释放 (L1339-1346)**

  ```cpp
  alpha_pipeline.consumer_release(alpha_smem_pipe_read);  // 释放 α SMEM (Step E 中用过)
  ++alpha_smem_pipe_read;
  k_pipeline.consumer_release(k_smem_pipe_read);          // 释放 K SMEM (Step E 中用过)
  ++k_smem_pipe_read;
  ```

  α 和 K 的 SMEM 在这里才释放——它们在 Step A (Q/K Prologue) 和 Step E (K_decay) 中都被使用，直到 Step F 的 WGMMA 完成后才能安全释放给 TMA 加载下一 tile。

  ---

  #### Step D+E+F 完整时序

  **Step D: O = O_inter + NewV^T @ QK**

  1. `wait qk_pipeline` ← QK 到 SMEM (MathA 产出)
  2. `make_acc_into_op(NewV → A)`
  3. Math0: fence → ordered_wait → WGMMA(NewV^T × QK) → notify → wait
  4. Math1: (等待) → WGMMA → wait
    - 首 tile: `gemm_zero_acc` (O = 0 + intra)
    - 非首 tile: `gemm` (O = inter + intra, 在 Step B 结果上累加)
  5. `release qk_pipeline`
  6. `o_store(O)`: fp32→bf16 → acquire o_pipeline → R2S → fence → commit → StoreO warp 可以 TMA 写出

  **Step E: State 衰减 + K_decay**

  1. `wait alpha_last_pipeline` ← α_last 到 SMEM
  2. `s_decay`: `State[i,j] *= exp(α_last[j])` (in-place, 寄存器内, 按列衰减)
  3. `NamedBarrier(StateMath)` ← 等 sQ_K_scaled 不再被 WGMMA 读
  4. 两 WG 各处理 HeadSize 一半 (同 Step A 结构):
    - S2R α → S2R K → K_decay = exp(α_last - α) * K (尾tile清零) → R2S stage 0
  5. NamedBarrier → fence
  6. `release alpha_last_pipeline`

  **Step F: State 更新**

  1. Math0: fence → ordered_wait → WGMMA(NewV^T × K_decay^T, 累加到 S_decayed) → notify → wait
  2. Math1: (等待) → WGMMA → wait
    - State 更新完毕: `tKVrKV = S_new`，带入下一 tile
  3. `release α_pipeline, k_pipeline` → TMA 可加载下一 tile

---

### 4.11 `CollectiveLoadTma` — TMA 加载封装

**文件**: `load_tma.hpp` (171 行)

#### 4.11.1 在 kernel 中的位置

`CollectiveLoadTma` 被 `load_qkv()` (4.5) 调用，封装了 TMA 异步加载的全部逻辑。被实例化为 4 个具体类型:

```c++
using LoadQ     = CollectiveLoadTma<LoadKind::kQ,     MainloopQPipeline,     Element,       QKSmemLayoutQ,       TMA_Q>;
using LoadK     = CollectiveLoadTma<LoadKind::kK,     MainloopKPipeline,     Element,       KVSmemLayoutK,       TMA_K>;
using LoadV     = CollectiveLoadTma<LoadKind::kV,     MainloopVPipeline,     Element,       KVSmemLayoutV,       TMA_V>;
using LoadAlpha = CollectiveLoadTma<LoadKind::kAlpha,  MainloopAlphaPipeline, ElementAlpha,  QKQSmemLayoutAlpha,  TMA_Alpha>;
```

5 个模板参数:

| 参数 | 含义 | Q | K/V | Alpha |
|------|------|---|-----|-------|
| `LoadKind` | 决定 GMEM 布局分支 | `kQ` | `kK`/`kV` | `kAlpha` |
| `Pipeline` | TmaPipeline 类型 | Q pipeline | K/V pipeline | Alpha pipeline |
| `Element` | 数据类型 | bf16 | bf16 | float |
| `SmemLayout` | SMEM swizzled 布局 | `QKSmemLayoutQ` | `KVSmemLayoutK/V` | `QKQSmemLayoutAlpha` |
| `TMA` | TMA 描述符类型 | `TMA_Q` | `TMA_K/V` | `TMA_Alpha` |

#### 4.11.2 类的成员

```c++
struct CollectiveLoadTma {
    using SharedStorage = cute::array_aligned<Element, cute::cosize_v<SmemLayout>>;
    using PipelineState = typename cutlass::PipelineState<Pipeline::Stages>;

    TMA const& tma_load;       // TMA 描述符: GMEM tensor 的地址/形状/stride 元数据
    Pipeline& pipeline;        // TmaPipeline 实例 (operator() 中创建)
    SharedStorage& storage;    // SMEM buffer (SharedStorage 结构体中)
};
```

`SharedStorage` 大小由 `SmemLayout` 的 `cosize_v` 决定，包含 `StageCount` 个 stage 的空间。

`PipelineState` 是 `{ int index; int phase; }`:
- `index`: 当前 stage 编号 (0 或 1, for 2-stage)
- `phase`: barrier 的 phase bit (区分本轮和上轮信号)

#### 4.11.3 partition_SD — GMEM 切分

调用一次，建立 GMEM tile → SMEM stage 的映射。两个分支:

**Q/Alpha 分支: `(seqlen, d)` 布局**

```c++
// 1. 从 TMA 描述符创建逻辑 tensor (不访问 GMEM, 只是地址空间描述)
Tensor m_varlen_head = tma_load.get_tma_tensor(
    make_shape(total_seqlen, head_size, num_heads));    // (total_seqlen, d, h)

// 2. 切到当前 head: (total_seqlen, d)
Tensor m_varlen = m_varlen_head(_, _, work_desc.q_head_idx());

// 3. 偏移到当前序列起始 (不改变 shape/stride, 只改变起始坐标)
Tensor m_offset = domain_offset(make_coord(work_desc.tok_offset, _0{}), m_varlen);

// 4. 按 tile 切分: (BlkSeqQ, HeadSize, num_tiles)
//    make_coord(_, _0{}): seqlen 维迭代(多 tile), head_size 维不迭代(完整)
Tensor g_full = local_tile(m_offset, make_tile(BlkSeqQ, HeadSize), make_coord(_, _0{}));
```

举例: seq_len=200, BlkSeqQ=64 → `g_full shape = (64, 128, 4)`, 4 个 tile, 最后一个只有 8 行有效 (TMA 自动处理越界)。

**K/V 分支: `(d, seqlen)` 转置布局**

```c++
// GMEM: (head_size, total_seqlen, num_heads) — head_size 在前!
Tensor m_varlen_head = tma_load.get_tma_tensor(
    make_shape(head_size, total_seqlen, num_heads));

// 偏移在第二维
Tensor m_offset = domain_offset(make_coord(_0{}, work_desc.tok_offset), m_varlen);

// tile: (HeadSize, BlkSeqKV, num_tiles) — HeadSize 在前
Tensor g_full = local_tile(m_offset, make_tile(HeadSize, BlkSeqKV), make_coord(_0{}, _));
```

K/V 转置存储原因: WGMMA 的 B 矩阵 (从 SMEM 读) 要求**列连续**。`(d, seqlen)` 存储让 TMA 加载后直接满足此要求, 免去 SMEM 内转置。

**最后构建 src/dst pair**:

```c++
Tensor s = make_tensor(make_smem_ptr(storage.data()), SmemLayout{});  // SMEM tensor (含 swizzle 和多 stage)
auto block_tma = tma_load.get_slice(_0{});  // 单 block (不支持 cluster)
return make_tuple(block_tma.partition_S(g),   // src = GMEM 按 TMA box shape 分区
                  block_tma.partition_D(s));   // dst = SMEM 按 TMA box shape 分区
```

`partition_S/D` 按 TMA 描述符中的 box shape 把 tensor 切成多个 chunk, 结果多出一维索引这些 chunk。

#### 4.11.4 step — 发起一次 TMA 异步加载

```c++
template <bool kAcquireBarrier = true, class SrcDst>
CUTE_DEVICE void
step(SrcDst const& src_dst, int src_iter, PipelineState& dst_pipe, uint32_t lane_predicate) {
    if (lane_predicate == 1) {                                       // (A)
        if constexpr (kAcquireBarrier) {
            pipeline.producer_acquire(dst_pipe);                     // (B)
        }
        using BarrierType = typename Pipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(dst_pipe);  // (C)

        auto src = get<0>(src_dst);
        auto dst = get<1>(src_dst);
        copy(tma_load.with(*tma_barrier),                            // (D)
             src(_, _, _, src_iter),
             dst(_, _, _, dst_pipe.index()));
        ++dst_pipe;                                                  // (E)
    }
}
```

逐行:

**(A) `lane_predicate == 1`**: 只有 leader 线程执行。TMA 只需**一个线程发指令**, 硬件自动搬运整块数据。其他 31 个线程跳过。

**(B) `producer_acquire`**: 等 SMEM stage 可写。2-stage 中前两次立即通过, 第三次可能阻塞 (stage 0 还没被 consumer release)。`kAcquireBarrier` 模板参数允许跳过 (调用者已手动 acquire 时)。

**(C) `producer_get_barrier`**: 获取当前 stage 的 mbarrier (SM90 硬件 barrier)。TMA 完成后自动递减此 barrier 的期望计数。

**(D) `copy(tma_load.with(*tma_barrier), src, dst)`**: 核心——发起**异步** TMA:
- `with(*tma_barrier)`: 绑定 barrier, TMA 完成后自动通知 "搬了 transaction_bytes 字节"
- `src(_, _, _, src_iter)`: GMEM 第 src_iter 个 tile
- `dst(_, _, _, dst_pipe.index())`: SMEM 第 index 个 stage
- 编译为 PTX `cp.async.bulk.tensor`, **立即返回**不等搬运完成
- Consumer 通过 `consumer_wait(pipe_read)` 等 barrier 确认数据到位

**(E) `++dst_pipe`**: 写指针前进。`index = (index + 1) % Stages`, 回到 0 时 `phase ^= 1` (区分 barrier 轮次)。

#### 4.11.5 LoadQBytes 等 — TMA 搬运字节数

```c++
static constexpr int LoadQBytes     = size(QKSmemLayoutQ{}(_, _, _0{}))     * sizeof(Element);      // 64×128×2 = 16KB
static constexpr int LoadKBytes     = size(KVSmemLayoutK{}(_, _, _0{}))     * sizeof(Element);      // 128×64×2 = 16KB
static constexpr int LoadVBytes     = size(KVSmemLayoutV{}(_, _, _0{}))     * sizeof(Element);      // 16KB
static constexpr int LoadAlphaBytes = size(QKQSmemLayoutAlpha{}(_, _, _0{})) * sizeof(ElementAlpha); // 64×128×4 = 32KB
```

取 stage 0 的 slice 求元素数, 乘以字节数。设置到 `pipeline_params.transaction_bytes`, 告诉 TmaPipeline 的 barrier "每次搬运到达 N 字节"。

#### 4.11.6 SMEM Swizzle 布局简述

SMEM 布局用 **swizzle** (地址位异或重排) 消除 bank conflict:

```c++
// CUTLASS 从 MMA 配置自动推导 swizzle 模式
using SmemLayoutQ_SD = decltype(unstage_smem_layout(typename CollectiveMmaQK::SmemLayoutA{}, ...));
// Alpha 用 128-byte swizzle
using SmemLayoutAlphaAtom = GMMA::Layout_K_SW128_Atom<ElementAlpha>;
```

```
无 swizzle:                     有 swizzle (SW128):
线程 0 → bank 0                 线程 0 → bank 0
线程 1 → bank 1                 线程 1 → bank 5  (异或)
线程 2 → bank 2                 线程 2 → bank 2
→ 连续访问同一行 → bank conflict  → 分散访问 → 无 conflict
```

`unstage_smem_layout` 把 CUTLASS 多 stage 布局拆成 `(tile_shape..., num_stages)`, 用 `layout(_, _, stage_idx)` 索引特定 stage。

#### 4.11.7 小结

```
CollectiveLoadTma 封装了:
  TMA 描述符 + Pipeline + SMEM buffer → partition_SD (一次) + step (每 tile 一次)

关键设计:
  - Q/Alpha (seqlen, d) vs K/V (d, seqlen) 转置 — 满足 WGMMA B 矩阵列连续
  - Swizzled SMEM 布局 — 消除 bank conflict
  - 2-stage pipeline — load/compute 重叠
  - 只有 leader 线程发 TMA — 不消耗计算线程
  - transaction_bytes — TMA 自动通知 consumer barrier
```

---

### 4.12 `CollectiveStoreTma` — TMA 存储封装

**文件**: `store_tma.hpp` (313 行)

#### 4.12.1 为什么自己实现

文件开头注释: CUTLASS 的 `cutlass::epilogue::collective::CollectiveBuilder` 生成的 store 类型把某些需要的 type alias 设为 private，KDA kernel 无法直接使用，所以自行实现等价的 TMA store。

#### 4.12.2 模板参数

```c++
template <
    typename TileShape_MNK_,    // (HeadSize=128, BlkSeqQ=64, HeadSizeQK=128)
    typename ClusterShape,       // (1, 1, 1)
    typename ElementO,           // bf16 (GMEM 输出类型)
    typename ElementAccumulator, // float (MMA accumulator 类型)
    typename SmemElementO,       // bf16 (SMEM 中存储类型)
    typename StrideO,            // O 的 GMEM stride
    int Stages>                  // 1 (单 stage, 不需要双缓冲)
struct CollectiveStoreTma {
```

`Stages = 1`: 与 Q/K/V 的 2-stage 不同。O 是"写出去"，Math WG 写完一个 tile 的 O，StoreO warp 立即搬走，不需要在 SMEM 中缓存两份。

#### 4.12.3 SMEM 布局与 Copy 类型

```c++
static_assert(sizeof(SmemElementO) == 2);  // bf16
using SmemLayoutAtom = GMMA::Layout_MN_SW32_Atom<SmemElementO>;  // 32-byte swizzle (比 Q/K 的 SW128 更细)

using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtom{},
    make_shape(SizeM{}, SizeN{}, Int<Stages>{}),  // (128, 64, 1)
    cute::conditional_t<is_m_major_O, Step<_2, _1, _3>, Step<_1, _2, _3>>{}));

constexpr static uint32_t TmaTransactionBytes = ...;  // 128×64×2 = 16KB
```

两次数据搬运的 Copy 类型:
```c++
using CopyOpR2S = decltype(sm90_get_smem_store_op_for_accumulator<...>());  // Math WG: reg→SMEM
using CopyOpS2G = SM90_TMA_STORE;                                           // StoreO:  SMEM→GMEM
```

#### 4.12.4 Pipeline: PipelineAsync (不是 PipelineTmaStore)

```c++
using Pipeline = cutlass::PipelineAsync<Stages>;  // NOT PipelineTmaStore!
```

注释特别标注了**不用** `PipelineTmaStore`。原因: o_pipeline 的 producer 是 Math WG (256 线程手动 arrive)，不是 TMA 硬件。`PipelineAsync` 用 producer/consumer arrive 计数模型，`PipelineTmaStore` 会自动跟踪 TMA transaction bytes — 不适用于 R2S 场景。

#### 4.12.5 partition_SD — 与 Load 的对称

```c++
// O 的 GMEM: (head_size, total_seqlen, num_heads) — 和 K/V 一样转置
Tensor m_varlen_head = tma_store_.get_tma_tensor(
    make_shape(head_size, total_seqlen, num_heads));
// ... 切 head, offset, local_tile ...

// 注意 src/dst 与 Load 反过来:
return make_tuple(
    block_tma.partition_S(s),   // src = SMEM
    block_tma.partition_D(g));  // dst = GMEM
//          Load: S=GMEM, D=SMEM
//         Store: S=SMEM, D=GMEM  ← 反过来
```

#### 4.12.6 step — 核心逻辑 (含尾 tile 处理)

```c++
CUTE_DEVICE void step(ProblemSize, WorkDesc, SrcDst, PipelineState& src_pipe,
                      int dst_iter, int num_iters, uint32_t lane_predicate) {
    // (1) 第一次迭代时预创建尾 tile 的修改版 TMA 描述符
    if (dst_iter == 0) {
        if (!can_process(problem_size, work_desc, num_iters - 1, num_iters)) {
            create_tensormap_for_tail(work_desc, lane_predicate);
        }
    }

    // (2) 等 Math WG 把 O 写入 SMEM (consumer 角色)
    pipeline_.consumer_wait(src_pipe);

    // (3) TMA store: SMEM → GMEM
    if (can_process(...)) {
        if (lane_predicate == 1) {
            copy(tma_store_, src(src_pipe.index()), dst(dst_iter));  // 正常 tile
        }
    } else {
        cute::TmaDescriptor* tensormap = acquire_tensormap_for_tail();
        if (lane_predicate == 1) {
            copy(tma_store_.with(tensormap), src(src_pipe.index()), dst(dst_iter));  // 修改版描述符
        }
    }

    // (4) 释放 SMEM stage
    pipeline_.consumer_release(src_pipe);
    ++src_pipe;
}
```

与 Load step 的对比:

| 方面 | Load step | Store step |
|------|-----------|------------|
| Pipeline 角色 | Producer (写 SMEM) | Consumer (读 SMEM) |
| 数据方向 | GMEM → SMEM | SMEM → GMEM |
| Barrier 操作 | `producer_acquire` + TMA 自动通知 | `consumer_wait` + `consumer_release` |
| 越界处理 | TMA 自动 (读垃圾无害) | `create_tensormap_for_tail` (写越界致命) |

#### 4.12.7 can_process — 安全性判断

```c++
static bool can_process(ProblemSize, WorkDesc, int blk, int num_blocks) {
    if (blk < num_blocks - 1) return true;                              // 非尾 tile
    else if (work_desc.seq_len % SizeN{} == 0) return true;             // 尾 tile 恰好满
    else if (work_desc.seq_idx == problem_size.num_seqs - 1) return true; // 最后一条序列
    else return false;                                                    // 不安全
}
```

只有一种不安全: **尾 tile 不满 + 后面还有序列** → TMA 写满会覆盖下一条序列数据。

#### 4.12.8 create_tensormap_for_tail — 运行时修改 TMA 描述符

SM90 高级特性: device 端动态修改 TMA 描述符。

```c++
CUTE_DEVICE void create_tensormap_for_tail(WorkDesc, uint32_t lane_predicate) {
    // (1) 定位 per-SM workspace
    cute::TmaDescriptor* tensormap = static_cast<cute::TmaDescriptor*>(tensormaps_) + smid();
    // smid(): 内联 PTX 读取 %smid 寄存器, 每 SM 唯一编号

    // (2) Warp 并行复制 128B 描述符 (8 线程 × 16B)
    if (lane_idx < num_of_16B) {  // num_of_16B = 128/16 = 8
        dst[lane_idx] = src[lane_idx];  // 128-bit load/store
    }
    __syncwarp();

    // (3) Leader 修改 seqlen 维大小: 截断到当前序列末尾
    if (lane_predicate == 1) {
        uint32_t new_total_seqlen = work_desc.tok_offset + work_desc.seq_len;
        ptx::tensormap_replace_global_dim(ptx::space_global, tensormap,
            /*ord=*/ptx::n32_t<1>{},     // 第 1 维 = seqlen
            new_total_seqlen);            // 截断
    }
    __syncwarp();

    // (4) 刷 TMA 描述符缓存
    ptx::fence_proxy_tensormap_generic(ptx::sem_release, ptx::scope_cta);
}
```

- `tensormap_replace_global_dim`: PTX 指令, 只改描述符中**一个维度**的大小, 其他不变
- 原来 seqlen 维 = total_seqlen (所有序列), 修改后 = tok_offset + seq_len (当前序列末尾)
- `fence_proxy_tensormap_generic(sem_release)`: 专门针对 TMA 描述符缓存的 fence (不是普通 memory fence)

使用时通过 `acquire_tensormap_for_tail()` 获取:
```c++
ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_cta, tensormap, ptx::n32_t<128>{});
```
`sem_acquire` 与 `sem_release` 配对, 标准的 acquire-release 语义。

#### 4.12.9 小结

```
CollectiveStoreTma vs CollectiveLoadTma:

                  Load                           Store
方向:            GMEM → SMEM                    SMEM → GMEM
Pipeline:        TmaPipeline (Producer)         PipelineAsync (Consumer)
src/dst:         S=GMEM, D=SMEM                S=SMEM, D=GMEM
Stage 数:        2 (双缓冲)                     1 (单缓冲)
越界处理:        自动 (读无害)                  create/acquire_tensormap_for_tail
Workspace:       不需要                         sizeof(TmaDescriptor) × sm_count
Swizzle:         SW128                          SW32
自定义原因:      N/A                            CUTLASS epilogue alias 是 private
```

---

### 4.13 `OrderedNamedBarriers` — 有序同步

**文件**: `math_order_barrier.hpp` (116 行)

#### 4.13.1 解决什么问题

`compute()` 的 Step B-F 中，Math0 和 Math1 两个 WG 需要**按顺序**访问 SMEM (如 `sQ_K_scaled`、`sKK_inv`)。如果两个 WG 同时发起 WGMMA 读同一块 SMEM，可能导致 bank conflict。`OrderedNamedBarriers` 保证: **Math0 先做 WGMMA, Math1 等 Math0 做完后再做**。

#### 4.13.2 实例化

```c++
// 两个预留 barrier ID
static constexpr uint32_t OrderedBarrierId0 = uint32_t(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
static constexpr uint32_t OrderedBarrierId1 = uint32_t(cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);

// WG0(Math0) 用 barrier 0, WG1(Math1) 用 barrier 1
using OrderedMathBarriers = OrderedNamedBarriers<true, OrderedBarrierId0, OrderedBarrierId1>;
```

`mapping_[0] = OrderedBarrierId0`, `mapping_[1] = OrderedBarrierId1`。

#### 4.13.3 Named Barrier 基础

SM90 Named Barrier 语义:

```
NamedBarrier::arrive(thread_count, barrier_id):
  当前线程"到达"。到齐 thread_count 后 barrier 释放。
  不阻塞 — 线程立即继续。

NamedBarrier::sync(thread_count, barrier_id):
  = arrive + wait。到齐后继续, barrier 自动重置。
```

本场景: `thread_count = 128 × 2 = 256` (Math0 + Math1 的线程总数)。

#### 4.13.4 init — 初始化偏移

```c++
CUTE_DEVICE void init(int wg_idx) {  // Math0: init(0), Math1: init(1)
    for (int i = wg_idx; i > 0; --i) {
        cutlass::arch::NamedBarrier::arrive(256, mapping_[i - 1]);
    }
}
```

- Math0 `init(0)`: 循环不执行
- Math1 `init(1)`: `arrive(256, B0)` — 128 线程 arrive barrier 0

初始化后状态 (arrived, expected):

```
B0: (128, 256)    ← Math1 init 贡献了 128, 只差 Math0 的 128 即解锁
B1: (0,   256)    ← 无人 arrive, 需 256 才解锁
```

关键: **B0 对 Math0 几乎是"打开"的, B1 对 Math1 是"关闭"的**。

#### 4.13.5 ordered_or_wait — 等轮次

```c++
CUTE_DEVICE void ordered_or_wait(int wg_idx) {
    cutlass::arch::NamedBarrier::sync(256, mapping_[wg_idx]);
}
```

每个 WG 在**自己的 barrier** 上 sync。

第一轮:
```
Math0 sync(B0): (128 → 256, 256) → 解锁! → Math0 立即通过 ✓, B0 重置 (0, 256)
Math1 sync(B1): (0 → 128, 256)   → 还差 128 → Math1 阻塞 ✗
```

#### 4.13.6 notify_next_blocked — 通知

```c++
CUTE_DEVICE void notify_next_blocked(int wg_idx) {
    for (int i = 1; i < NumWG; ++i) {
        cutlass::arch::NamedBarrier::arrive(256, mapping_[(wg_idx + i) % NumWG]);
    }
}
```

2 个 WG 时循环只执行一次: arrive 对方的 barrier。

```
Math0 notify(0): arrive B1 → B1: (128 → 256, 256) → 解锁 → Math1 通过 ✓
Math1 notify(1): arrive B0 → B0: (0 → 128, 256)   → 为下一轮 Math0 准备
```

#### 4.13.7 完整状态转移

```
初始化: B0:(128,256)  B1:(0,256)

═══ 第 1 轮 ═══
Math0 ordered_or_wait(0): B0:(128→256) → 解锁, 重置(0,256). Math0 通过 ✓
Math1 ordered_or_wait(1): B1:(0→128)   → 阻塞 ✗

Math0 做 WGMMA ...
Math0 notify_next_blocked(0): arrive B1 → B1:(128→256) → 解锁. Math1 通过 ✓

Math1 做 WGMMA ...
Math1 notify_next_blocked(1): arrive B0 → B0:(0→128)

═══ 第 2 轮 ═══
Math0 ordered_or_wait(0): B0:(128→256) → 解锁 ✓   ← Math1 上轮 notify 贡献了 128
Math1 ordered_or_wait(1): B1:(0→128)   → 阻塞 ✗   ← 和第 1 轮一样

... 循环往复, 永远 Math0 先通过
```

#### 4.13.8 使用模式

`compute()` 中每次 WGMMA 前后:

```c++
math_barriers.ordered_or_wait(warpgroup_idx);   // 等自己的轮次
warpgroup_arrive();
gemm(...);                                       // WGMMA (读 SMEM)
warpgroup_commit_batch();
math_barriers.notify_next_blocked(warpgroup_idx); // 通知另一个 WG
```

在 Step B/C1/C4/D/F 中各出现一次, 共 **5 次有序 WGMMA**。

#### 4.13.9 为什么不用 __syncthreads

1. `__syncthreads` 同步**所有 512 线程** (含 LdSt 和 MathA), Named Barrier 只同步 Math0+Math1 的 256 线程
2. `__syncthreads` 不保证**顺序**, 只保证"都到齐"; `OrderedNamedBarriers` 保证 Math0 先 Math1 后
3. Named Barrier 是 SM90 的细粒度同步原语, 开销远低于 `__syncthreads`

#### 4.13.10 小结

```
OrderedNamedBarriers 核心机制:
  每个 WG 有自己的 barrier
  init: 让 B0 "半开" (差 Math0 arrive 即解锁), B1 "全关"
  ordered_or_wait: sync 自己的 barrier
  notify_next_blocked: arrive 对方的 barrier

  效果: 每轮 Math0 先 → WGMMA → notify → Math1 → WGMMA → notify → 下一轮
  保证: 两 WG 不同时读 SMEM, 避免 bank conflict
```

---

### 4.14 `common.hpp` — MMA / 类型转换 / 布局工具

**文件**: `collective/common.hpp` (477 行)

提供 compute 路径上的底层工具函数，分三类。

#### 4.14.1 WGMMA 辅助函数

**gemm_zero_acc / gemm_reset_zero_acc**

```c++
CUTE_DEVICE void gemm_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
    atom.accumulate_ = GMMA::ScaleOut::Zero;   // 第一个 k_block: C = A@B (清零)
    gemm_reset_zero_acc(atom, tA, tB, tC);
}

CUTE_DEVICE void gemm_reset_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
    for (int k_block = 0; k_block < size<1>(tA); k_block++) {
        cute::gemm(atom, tA(_, k_block), tB(_, k_block), tC);
        atom.accumulate_ = GMMA::ScaleOut::One;  // 后续 k_block: C += A@B (累加)
    }
}
```

SM90 WGMMA 的 `ScaleOut` 参数: `Zero` = 清零后累加, `One` = 在已有 C 上累加。`gemm_zero_acc` 第一个 k_block 清零, 后续累加, 比先 `clear(tC)` 再 `gemm` 更高效 (省一次显式清零)。

使用处:
- `gemm_zero_acc`: Step B (O = 0 + Q@S), Step C1 (SK = 0 + S@K), Step C4 (NewV = 0 + V'@T^{-1}), Step D 首 tile
- `gemm` (不清零): Step D 非首 tile (O += NewV@QK), Step F (S += NewV@K_decay)

**convert_to_gmma_rs — SS→RS MMA 转换**

```c++
CUTE_DEVICE constexpr auto convert_to_gmma_rs(cute::TiledMMA<Atom, Args...> const& tiled_mma) {
    return cute::TiledMMA<decltype(convert_to_gmma_rs(Atom{})), Args...>{};
}
```

CUTLASS `CollectiveBuilder` 默认生成 SS 模式 MMA (A,B 都在 SMEM)。但 State `tKVrKV` 在寄存器, 需要 RS 模式 (A=寄存器, B=SMEM)。此函数用 `GMMA::rs_op_selector` 找到等价的 RS 指令, 保留 tiling 参数。

```c++
// mainloop 中:
using TiledMmaO1   = decltype(convert_to_gmma_rs(typename CollectiveMmaO1::TiledMma{}));
using TiledMmaSK   = decltype(convert_to_gmma_rs(typename CollectiveMmaSK::TiledMma{}));
using TiledMmaNewV = decltype(convert_to_gmma_rs(typename CollectiveMmaNewV::TiledMma{}));
```

#### 4.14.2 make_acc_into_op — Accumulator→Operand 布局转换

```c++
template <class Element, class Accumulator, class OperandLayout_TV>
CUTE_DEVICE auto make_acc_into_op(Accumulator const& acc, OperandLayout_TV const& operand_layout_tv) {
    // 1. 推导目标布局
    Tensor operand = make_fragment_like<Element>(
        convert_c_layout_to_a_layout(acc.layout(), shape<1>(operand_layout_tv)));
    // 2. 复制数据 (fp32→bf16 类型转换)
    Tensor operand_as_acc = make_tensor(operand.data(), acc.layout());
    cute::copy(acc, operand_as_acc);
    // 3. 对 8-bit 元素做 warp shuffle 重排 (bf16/fp16 不需要)
    if constexpr (sizeof(Element) == 1) { /* shuffle */ }
    return operand;
}
```

**核心问题**: WGMMA 的 accumulator (C) 和 operand (A) 在每个线程中持有的元素位置不同:

```
64×128 矩阵, 128 线程:
Accumulator (C): 线程 0 持有 [r0,c0], [r0,c2], [r1,c0], ...
Operand (A):     线程 0 持有 [r0,c0], [r0,c1], [r0,c4], ...
                 ← 同一个矩阵, 但每线程持有不同元素
```

对于 bf16 (sizeof=2), C 和 A 的寄存器映射恰好兼容, `copy` 实际上是 in-place 的 reinterpret + 类型转换 (fp32→bf16)。不需要 warp shuffle。

在 `compute()` 中每次 WGMMA 切换角色时调用:
```c++
// Step B: State(acc) → O1 MMA 的 A(operand)
Tensor tOrKV = make_acc_into_op<Element>(tKVrKV, typename TiledMmaO1::LayoutA_TV{});
// Step C1: State(acc) → SK MMA 的 A
// Step C4: V'(acc) → NewV MMA 的 A
// Step D/F: NewV(acc) → O2/KV MMA 的 A
```

#### 4.14.3 BF16↔TF32 布局转换 (warp shuffle)

在 `compute_aux_safe()` 的 SubChunk MMA 中使用。先用 BF16 LDSM 高效加载 Q/K, 做完 gating 后转成 TF32 MMA 布局。

**为什么布局不同?** MMA atom shape 相同 (16×8×8), 但 K 维度在线程间的分布不同:

```
BF16 A 布局: 线程 t0 持有 K = {2*t0, 2*t0+1}    ← 连续
TF32 A 布局: 线程 t0 持有 K = {t0, t0+4}         ← 间隔 4
```

`convert_bf16_to_tf32_operandA_layout` 用 warp shuffle 在线程间交换:

```c++
int src_lane_lo = (t0 / 2) + (tid & ~3);      // 源: K=t0 在 t0/2 线程
int src_lane_hi = (t0 / 2 + 2) + (tid & ~3);  // 源: K=t0+4 在 t0/2+2 线程

for (int j = 0; j < NumKAtoms; j++) {
    float in0..in3 = frag_A(4*j+0..3);         // 读 4 个 BF16 布局值
    for (v0_tf32 = 0..1) {
        // 4 次 shuffle: 从 src_lane_lo/hi 取数据
        recv0_lo = __shfl_sync(0xFFFF, val0, src_lane_lo);
        recv1_lo = __shfl_sync(0xFFFF, val1, src_lane_lo);
        recv0_hi = __shfl_sync(0xFFFF, val0, src_lane_hi);
        recv1_hi = __shfl_sync(0xFFFF, val1, src_lane_hi);
        // 根据 t0%2 选择
        out[v0_tf32+0] = sel_odd ? recv1_lo : recv0_lo;
        out[v0_tf32+2] = sel_odd ? recv1_hi : recv0_hi;
    }
    frag_A(4*j+0..3) = out[0..3];               // 写回 TF32 布局 (in-place)
}
```

每个 k-atom 4 次 shuffle, BK=32 (4 atoms) 共 16 次。代码注释: 比 `AutoVectorizingCopy` 直接加载 TF32 布局**节省 50% SMEM 带宽**。

operand B 版本 (`convert_bf16_to_tf32_operandB_layout`) 原理相同, 每 atom 2 个值更简单。

#### 4.14.4 α_first 广播

**broadcast_row0_operandA_to_operandB_bf16_layout**: 从 alpha 的 A 布局中提取 row 0, 直接输出为 B 布局:

```c++
int src_lane = tid % 4;  // row 0 在 t1=0 线程, 保留 t0 设 t1=0
for (int j = 0; j < NumKAtoms; j++) {
    // 从 t1=0 线程 shuffle row 0 的值 (2 次 shuffle)
    auto val0 = __shfl_sync(0xFFFF, frag_A(4*j+0), src_lane);  // K=2*t0
    auto val1 = __shfl_sync(0xFFFF, frag_A(4*j+1), src_lane);  // K=2*t0+1
    frag_B_first(2*j+0) = val0;   // 直接写 B 布局
    frag_B_first(2*j+1) = val1;
}
```

关键: BF16 A 和 B 的 K 维度映射相同 (K={2*t0, 2*t0+1}), row 0 又是 broadcast (所有 M 行相同), 所以 A→B 只需取 v1=0 子集, 无需额外重排。比先广播再提取**省 8 个 float 寄存器**。

还有 `broadcast_row0_operandA_bf16_layout` (广播到所有 M 行, 输出仍是 A 布局) 和 `extract_broadcast_operandA_to_operandB` (broadcast 数据从 A→B, 纯寄存器无 shuffle)。

#### 4.14.5 小结

```
common.hpp 三类工具:

1. WGMMA 控制:
   gemm_zero_acc — ScaleOut::Zero 首 k_block + ScaleOut::One 后续
   convert_to_gmma_rs — SS→RS MMA 转换 (State 在寄存器需要)

2. 布局转换:
   make_acc_into_op — Accumulator C 布局→Operand A 布局 (bf16 无需 shuffle)
   convert_bf16_to_tf32_operandA/B — warp shuffle 转换 (省 50% SMEM 带宽)

3. α_first 广播:
   broadcast_row0_operandA_to_operandB — row0 A→B 直出 (省 8 reg)
```

---

### 4.15 `SharedStorage` — 共享内存布局详解

**文件**: `kernel_kda_fwd.hpp:97-130` (外层) + `mainloop_kda_fwd.hpp:424-446` (内层)

#### 4.15.1 两层结构

**外层 (kernel_kda_fwd.hpp)** — tensor 数据 + pipeline 元数据:

```c++
struct SharedStorage {
    TensorStorage tensors;           // 实际 tensor 数据 (~226 KB)

    // 9 条 Pipeline 的 mbarrier storage (每条 ~32 bytes × stages)
    alignas(16) QPipelineStorage  q_pipeline_storage;
    alignas(16) KPipelineStorage  k_pipeline_storage;
    // ... 另外 7 条 pipeline ...
    alignas(16) cutlass::arch::ClusterBarrier load_warp_barrier;
};
```

**内层 (mainloop SharedStorage)** — 10 个 tensor buffer:

```c++
struct SharedStorage {
    // 输入 (TMA 加载目标)
    alignas(swizzle) array<Element,      cosize<QKSmemLayoutQ>>       smem_q;         // Q
    alignas(swizzle) array<Element,      cosize<KVSmemLayoutK>>       smem_k;         // K
    alignas(swizzle) array<Element,      cosize<KVSmemLayoutV>>       smem_v;         // V
    alignas(swizzle) array<ElementAlpha, cosize<QKQSmemLayoutAlpha>>  smem_alpha;     // α

    // MathA 产出
    alignas(swizzle) array<Element,      cosize<SmemLayoutQK>>   smem_qk;            // Q@K^T
    alignas(swizzle) array<InverseType,  cosize<SmemLayoutKK>>   smem_kk;            // K@K^T→T^{-1}

    // Math0/1 中间
    alignas(swizzle) array<Element,  cosize<QKScaledSmemLayoutQ>>  smem_q_k_scaled;  // exp(α)*Q/K

    // 输出
    SharedStorageO smem_o;                                                            // O

    // 标量/向量
    array<ElementBeta,  cosize<SmemLayoutBeta>>       smem_beta;                      // β
    array<ElementAlpha, cosize<SmemLayoutAlphaLast>>  smem_alpha_last;                // α 最后行
};
```

#### 4.15.2 每个 buffer 的大小 (BlkSeqQ=64, HeadSize=128)

| Buffer | 元素类型 | Shape (含 stages) | 字节数 | 说明 |
|--------|---------|-------------------|--------|------|
| smem_q | bf16 | (64, 128, 2) | **32 KB** | Q, 2-stage pipeline |
| smem_k | bf16 | (128, 64, 2) | **32 KB** | K (转置), 2-stage |
| smem_v | bf16 | (128, 64, 1) | **16 KB** | V (转置), 1-stage |
| smem_alpha | float | (64, 128, 2) | **64 KB** | α, 2-stage |
| smem_qk | bf16 | (64, 64, 2) | **16 KB** | QK 结果, 2-stage |
| smem_kk | half_t | (64, 64, 2) | **16 KB** | KK/T^{-1}, 2-stage |
| smem_q_k_scaled | bf16 | (64, 128, 2) | **32 KB** | S0=Q_scaled, S1=K_scaled |
| smem_o | bf16 | (128, 64, 1) | **16 KB** | O 输出, 1-stage |
| smem_beta | float | (64, 2) | **0.5 KB** | β, 2-stage |
| smem_alpha_last | float | (128, 2) | **1 KB** | α最后行, 2-stage |
| **总计** | | | **~226 KB** | SM90 上限 228 KB |

**~226 KB 接近 228 KB 上限** — 和寄存器一起双重限制 occupancy=1 block/SM。

#### 4.15.3 Stage 数量选择

| Buffer | Stages | 原因 |
|--------|--------|------|
| Q, K, Alpha | 2 | 双缓冲: TMA 加载 tile N+1 时 MMA 计算 tile N |
| V | 1 | V 在 Step C2 才用, 此时 Q/K 已 release, 不需与 Q/K 重叠 |
| QK, KK | 2 | MathA 产出 tile N+1 时 Math0/1 消费 tile N |
| Q_K_scaled | 2 | **不是** pipeline 双缓冲! S0=Q_scaled, S1=K_scaled (同 tile 两个变量) |
| O | 1 | 写完立即 TMA 搬走 |
| Beta, AlphaLast | 2 | 生产者/消费者解耦 |

`smem_q_k_scaled` 的特殊用法:
- Step A: S0 = exp(α)*Q, S1 = exp(α)*K
- Step B: 读 S0 (Q_scaled @ State)
- Step C1: 读 S1 (State @ K_scaled)
- Step E: 覆写 S0 = K_decay = exp(α_last-α)*K (Q_scaled 不再需要)
- Step F: 读 S0 (State += NewV @ K_decay)

#### 4.15.4 Swizzle 对齐

```c++
alignas(alignment_for_swizzle(QKSmemLayoutQ{}))   // 对齐到 swizzle 要求
    cute::array_aligned<Element, cute::cosize_v<QKSmemLayoutQ>> smem_q;
```

`alignment_for_swizzle(Swizzle<B, M, S>)` = `1 << (B + M + |S|)`。

例: SW128 使用 `Swizzle<3, 3, 3>` → `1 << 9 = 512 bytes` 对齐。起始地址不对齐会导致 swizzle 异或产生错误的 bank 映射。

#### 4.15.5 数据流与 buffer 生命周期

```
时间 → (单 tile 内)

              Step A     Step B      Step C       Step D     Step E      Step F
smem_q:      [TMA写] → [读(prologue)] → [读(C)]                → [读(E)] → [release]
smem_k:      [TMA写] →               → [读(C1)]               → [读(E)] → [release]
smem_v:                              → [TMA写+读(C2)] →                  → [release]
smem_alpha:  [TMA写] → [读(prologue)] → [读(C3)]               → [读(E)] → [release]
smem_qk:                             ← [MathA写] → [读(D)]
smem_kk:                             ← [MathA写] → [逆→读写(C3)]
smem_q_k_scaled:
  S0:        [Q_scl写] → [读(B)]                              → [K_dec覆写] → [读(F)]
  S1:        [K_scl写] →           → [读(C1)]
smem_o:                                          → [R2S写(D)] → [StoreO TMA]
smem_beta:   [Load写] →           → [读(C3)]
smem_alpha_last:                                              → [读(E)]    → [release]
             ← LoadAlpha warp 中转 →
```

关键约束: `consumer_release` 必须在最后一次读取后。smem_k 和 smem_alpha 直到 Step F 后才 release, 因为 Step E 还要读。

#### 4.15.6 小结

```
SharedStorage ≈ 226 KB / 228 KB:
  输入 buffer: Q(32K) + K(32K) + V(16K) + α(64K) = 144 KB
  中间 buffer: QK(16K) + KK(16K) + Q_K_scaled(32K) = 64 KB
  输出 buffer: O(16K) = 16 KB
  标量/向量:  β(0.5K) + α_last(1K) ≈ 2 KB
  Pipeline metadata: ~600 bytes

关键: SMEM 和寄存器都接近用满 → 双重限制 occupancy=1 block/SM
```

---

## 5. 概念速查表

### 5.1 GPU 硬件概念

| 术语 | 含义 | 本 kernel 中的典型值/用法 |
|------|------|--------------------------|
| SM (Streaming Multiprocessor) | GPU 的核心执行单元，包含寄存器文件、SMEM、Tensor Core | 本 kernel occupancy=1 block/SM |
| Warp | 32 线程，SM 调度的最小单位 | 16 warps/block |
| Warp Group (WG) | 4 个 Warp = 128 线程，WGMMA 的执行单位 | 4 WG/block: LdSt, Math0, Math1, MathA |
| Lane | 线程在 warp 内的编号 (0-31) | leader = lane 0 (elect_one_sync) |
| Register File | 每 SM 65536 个 32-bit 寄存器，所有线程共享 | LdSt=24, StateMMA=168, AuxMMA=152 reg/thread |
| SMEM (Shared Memory) | 每 SM 最大 228 KB 的片上高速存储 | 本 kernel 用了 ~226 KB |
| GMEM (Global Memory) | HBM 显存，~80GB，高延迟(~400 cycles) | Q/K/V/O/α/β 的存储位置 |
| Bank Conflict | 多个线程同时访问 SMEM 同一 bank 导致串行化 | 用 Swizzle 消除 |
| Occupancy | 每 SM 同时驻留的 block 数 | 1 block/SM (寄存器+SMEM 双重限制) |

### 5.2 SM90 (Hopper) 特有概念

| 术语 | 含义 | 本 kernel 中的用法 |
|------|------|-------------------|
| TMA (Tensor Memory Accelerator) | 硬件异步 GMEM↔SMEM 数据搬运单元，不消耗 CUDA core | load_qkv (GMEM→SMEM), store (SMEM→GMEM) |
| TMA Descriptor | TMA 操作的元数据 (tensor 地址/形状/stride)，128 bytes | operator() 中 prefetch，store 中动态修改尾 tile |
| WGMMA (Warp Group MMA) | 128 线程协作的 Tensor Core 矩阵乘指令 | compute() 中 6 次 WGMMA (Step B/C1/C4/D/F + aux) |
| RS 模式 | WGMMA: A 在寄存器, B 在 SMEM | State @ Q_scaled, State @ K_scaled 等 |
| SS 模式 | WGMMA: A, B 都在 SMEM | compute_aux_safe 中不直接用 (用 SubChunk TF32 MMA) |
| Named Barrier | SM90 细粒度同步原语，支持任意线程子集 | OrderedNamedBarriers, KdaNamedBarriers |
| mbarrier | SM90 硬件 barrier，支持 TMA 自动到达通知 | TmaPipeline 的 ProducerBarrierType |
| Cluster | SM90 新增：多 SM 间协作单位 | 本 kernel ClusterShape=(1,1,1)，不使用 cluster |
| `warpgroup_reg_alloc/dealloc` | 运行时动态调整 WG 的寄存器数量 (PTX setmaxnreg) | LdSt dealloc→24, MMA alloc→168/152 |
| `fence_view_async_shared` | SMEM 写后 fence，确保 WGMMA 异步代理可见 | R2S 写 SMEM 后、producer_commit 前 |
| `fence_proxy_tensormap_generic` | TMA 描述符缓存的 fence (acquire/release 语义) | store 中动态修改 TMA descriptor 后 |
| `tensormap_replace_global_dim` | PTX 指令：运行时修改 TMA 描述符的某一维大小 | store 尾 tile 截断 seqlen 维 |
| `elect_one_sync` | 每 warp 选出唯一 leader 线程 (通常 lane 0) | TMA 操作只需 leader 发起 |
| `cp.async.bulk.tensor` | TMA copy 编译后的 PTX 指令，异步发起，立即返回 | CollectiveLoadTma::step 中的 copy() |

### 5.3 CUTLASS / CuTe 概念

| 术语 | 含义 | 本 kernel 中的用法 |
|------|------|-------------------|
| TiledMMA | CuTe 中描述 MMA 操作的类型，包含 atom + tiling + layout | TiledMmaO1/O2/SK/NewV/KV/QK_RS/KK 等 7 种 |
| MMA Atom | 最小 MMA 指令单位 (如 16×8×8) | SM80_16x8x8_F32TF32TF32F32_TN (SubChunk) |
| partition_fragment_C | 为 MMA 的 C (accumulator) 分配寄存器 fragment | tOrO, tKVrKV, tSKrSK, tNewVrC 等 |
| partition_S / partition_D | TMA 按 box shape 分区 GMEM/SMEM tensor | CollectiveLoadTma::partition_SD |
| make_acc_into_op | accumulator (C 布局) → operand (A 布局) 寄存器重解释 | State→WGMMA A, V'→NewV A 等 |
| Swizzle | SMEM 地址位异或重排，消除 bank conflict | SW32 (O), SW128 (Q/K/V/α) |
| LDSM (ldmatrix) | SM75+ 专用 SMEM→寄存器加载指令，天然匹配 MMA 布局 | SM75_U32x4_LDSM_N, SM75_U16x8_LDSM_T |
| STSM (stmatrix) | SM90 专用 寄存器→SMEM 存储指令 | SM90_U32x2_STSM_N, SM90_U32x4_STSM_N |
| `local_tile` | 按 tile shape 切分 tensor，增加一个迭代维 | partition_SD 中按 BlkSeqQ×HeadSize 切分 |
| `domain_offset` | 偏移 tensor 的起始坐标 (不改变 shape/stride) | 偏移到当前序列的 tok_offset |
| `flat_divide` | 1D tensor 按指定大小切块 | load_beta 中按 BlkSeqQ 切分 β 向量 |
| `tile_to_shape` | 把 atom 布局平铺到指定 shape | SmemLayoutO = tile_to_shape(SW32_Atom, ...) |
| `cosize_v` | 布局的 codomain 大小 (需要的元素总数) | SharedStorage 大小计算 |
| ScaleOut::Zero / One | WGMMA 的 accumulator 行为：清零/累加 | gemm_zero_acc vs gemm |

### 5.4 Pipeline 概念

| 术语 | 含义 | 本 kernel 中的用法 |
|------|------|-------------------|
| Pipeline | Producer-Consumer 异步流水线，用 barrier 同步 | 9 条 pipeline |
| TmaPipeline | TMA 专用 pipeline：`transaction_bytes` + `is_leader` + `num_consumers` | Q/K/V/Alpha pipeline |
| PipelineAsync | 通用 async pipeline：`producer_arv_count` + `consumer_arv_count` | O/QK/KK/Beta/AlphaLast pipeline |
| Stage | Pipeline 的环形 buffer 槽位 | 2-stage (Q/K/α/QK/KK/β/α_last), 1-stage (V/O) |
| PipelineState | `{index, phase}` — 当前 stage 编号 + barrier phase bit | pipe_read / pipe_write |
| producer_acquire | Producer 等待 stage 可写 (consumer 已 release) | TMA load 前 |
| producer_commit | Producer 通知 consumer 数据就绪 | load_beta R2S 后 |
| consumer_wait | Consumer 等待 stage 数据就绪 (producer 已 commit / TMA 完成) | compute 中等 Q/K/V/QK/KK 就绪 |
| consumer_release | Consumer 释放 stage (允许 producer 覆写) | compute 各 step 完成后 |
| transaction_bytes | TMA pipeline 的传输字节数，TMA 完成后自动通知 barrier | LoadQBytes=16KB, LoadAlphaBytes=32KB |

### 5.5 KDA 算法概念

| 术语 | 含义 | 代码变量 |
|------|------|---------|
| State (S) | 隐状态矩阵 (d×d)，贯穿所有 tile 的 RNN 记忆 | `tKVrKV` (寄存器，最大消耗) |
| α (alpha) | channel-wise 遗忘门 (per-token per-dim)，KDA vs GDN 的核心改进 | `smem_alpha`, ElementAlpha=float |
| β (beta) | scalar 学习率/delta rule 系数 (per-token) | `smem_beta` |
| α_last | 当前 tile alpha 最后一行，用于 tile 间 state 衰减 | `smem_alpha_last` |
| γ_C (gamma_C) | chunk 内总累积衰减 = exp(gc[-1]) | s_decay 中使用 |
| γ_{i→C} | 第 i 步到 chunk 末尾的部分衰减 = exp(gc[-1]-gc[i]) | K_decay 的缩放因子 |
| gc (cumulative gate) | g = log₂(α) 的累积和，log 域避免乘法溢出 | 代码中用 exp2f |
| Anchor (gc_ref) | safe_gate 的参考点，减去它防止 exp 溢出 | gc[0] (QK/KK 计算中) |
| Q_scaled | exp(gc) ⊙ Q = Γ⊙Q，用于 inter-chunk output | `sQ_K_scaled` stage 0 |
| K_scaled | exp(gc) ⊙ K = Γ⊙K，用于 S@K 计算 | `sQ_K_scaled` stage 1 |
| K_decay | exp(gc[-1]-gc) ⊙ K = Γ_{i→C}⊙K，用于 state 更新 | `sQ_K_scaled` stage 0 (覆写 Q_scaled) |
| QK | gated Q@K^T，下三角 attention 分数 | `smem_qk` |
| KK / T | I + StrictTril(gated K@K^T ⊙ diag(β))，下三角矩阵 | `smem_kk` |
| T^{-1} | forward substitution 求逆 | `smem_kk` (in-place, CollectiveInverse) |
| V' | V - S@K_scaled = 论文的 U - W@S (pseudo-value) | `tSKrV` 寄存器 |
| NewV | V' @ T^{-T} = 修正后的 value | `tNewVrC` 寄存器 |
| O_inter | Q_scaled @ S_old × scale，旧 state 对输出的贡献 | `tOrO` (Step B 累加) |
| O_intra | NewV^T @ QK，当前 tile 内部的贡献 | `tOrO` (Step D 累加) |
| scale | d^{-0.5}，attention 缩放因子 | `params.scale` |

### 5.6 Warp Specialization 角色

| 角色 | WG/Warp | 线程数 | 寄存器/线程 | 职责 |
|------|---------|--------|------------|------|
| LoadQKV | WG0.Warp0 | 32 (leader only) | 24 | TMA 加载 Q/K/V/α 到 SMEM |
| StoreO | WG0.Warp1 | 32 (leader only) | 24 | TMA 写回 O 到 GMEM |
| LoadBeta | WG0.Warp2 | 32 | 24 | Predicated load β (GMEM→reg→SMEM) |
| LoadAlpha | WG0.Warp3 | 32 | 24 | 提取 α 最后行 (SMEM→SMEM, 双 pipeline 中转) |
| Math0 | WG1 | 128 | 168 | State 计算 (compute Step A-F), 有序执行先手 |
| Math1 | WG2 | 128 | 168 | State 计算 (与 Math0 协作), 有序执行后手 |
| MathA | WG3 | 128 | 152 | 辅助计算 (gated QK/KK, SubChunk 16×16 TF32 MMA) |

### 5.7 代码变量命名规则

KDA kernel 的变量名高度编码化，遵循 CuTe/CUTLASS 的命名惯例。理解这套规则后可以"读名知意"，不用每个变量都查定义。

#### 5.7.1 前缀：存储位置

| 前缀 | 含义 | 示例 |
|------|------|------|
| `t` | Tensor fragment，通常在寄存器中 (Thread-level view) | `tKVrKV`, `tOrO`, `tSKrV` |
| `s` | SMEM tensor (Shared memory) | `sQqk`, `sKqk`, `sQ_K_scaled` |
| `g` | GMEM tensor (Global memory) | `tKVgKV` (通过 tiled_copy 分区后的 GMEM view) |
| `c` | Coordinate tensor (identity tensor，用于坐标映射/predicate) | `tKVcV`, `tOcO` |

#### 5.7.2 MMA 角色编码：`t{MMA名}{位置}{语义}`

这是最核心的规则。变量名 = `t` + **哪个 MMA** + **寄存器/SMEM 角色** + **数据语义**。

**格式: `t{MMA}{r/s}{Operand}`**

- `{MMA}` = 该变量服务于哪个 TiledMma (2-3 个大写字母缩写)
- `{r/s}` = `r` 表示寄存器 fragment，`s` 表示 SMEM 分区
- `{Operand}` = 该 fragment 在 MMA 中的角色 (A/B/C) 或数据语义

**完整解码示例表:**

| 变量名 | MMA | 位置 | 语义 | 完整含义 |
|--------|-----|------|------|---------|
| `tKVrKV` | KV (K@V → State) | r (寄存器) | KV (accumulator C) | KV-MMA 的累加器，即 State S 矩阵 (d×d)，贯穿整个 mainloop |
| `tSKrSK` | SK (State@K) | r | SK (accumulator C) | SK-MMA 的累加结果，即 S@K_scaled |
| `tSKrV` | SK | r | V (fragment，V' = V - SK) | 用 SK-MMA 的 C 布局存放 V'，先 LDSM 加载 V，再减去 SK |
| `tSKsK` | SK | s (SMEM) | K (operand B 的 SMEM 分区) | SK-MMA 的 B 操作数在 SMEM 中的 view |
| `tSKrK` | SK | r | K (operand B 的寄存器 fragment) | SK-MMA 的 B 操作数加载到寄存器 |
| `tNewVrC` | NewV (V'@T^{-T}) | r | C (accumulator) | NewV-MMA 的累加结果 |
| `tNewVrA` | NewV | r | A (operand A) | NewV-MMA 的 A 操作数 (从 tSKrV 通过 make_acc_into_op 转换) |
| `tNewVsB` | NewV | s | B (operand B) | NewV-MMA 的 B 操作数 SMEM 分区 (sKK_opd) |
| `tNewVrB` | NewV | r | B (operand B fragment) | NewV-MMA 的 B 操作数寄存器 fragment |
| `tOrO` | O1/O2 (output MMA) | r | O (accumulator C) | Output 累加器 = O_inter + O_intra |
| `tOrO1` | O1 | r | O1 (O_inter 部分) | O1-MMA 的累加结果 (Q_scaled @ State) |
| `tOrKV` | O1 | r | KV (operand A) | O1-MMA 的 A 操作数 (从 State 转换, make_acc_into_op) |
| `tOsQ` | O1 | s | Q (operand B 的 SMEM 分区) | O1-MMA 的 B 操作数 SMEM 分区 (sQ_K_scaled) |
| `tOrQ` | O1 | r | Q (operand B fragment) | O1-MMA 的 B 操作数寄存器 fragment |
| `tOsQK` | O2 | s | QK (operand B) | O2-MMA 的 B 操作数 SMEM 分区 (smem_qk) |
| `tOrQK` | O2 | r | QK (operand B fragment) | O2-MMA 的 B 操作数寄存器 fragment |
| `tOrV_or_tKVrV` | 双重身份 | r | V (两个 MMA 的 A) | NewV 结果，既是 O2-MMA 的 A (NewV^T@QK→O)，又是 KV-MMA 的 A (NewV^T@K_decay→S_new) |
| `tKVsK` | KV | s | K (operand B) | KV-MMA 的 B 操作数 SMEM 分区 (K_decay 转置布局) |
| `tKVrK` | KV | r | K (operand B fragment) | KV-MMA 的 B 操作数寄存器 fragment |
| `tKKsK` | KK (K@K^T) | s | K (operand B) | KK-MMA 的 B 操作数 SMEM 分区 |
| `tKKrA` | KK | r | A (operand A fragment) | KK-MMA 的 A 操作数寄存器 fragment |
| `tOsO` | O (store) | s | O (SMEM 目标) | R2S copy 的 SMEM 目标分区 |
| `tOrO_cvt` | O | r | O (类型转换后) | float32 → bf16 转换后的 O fragment |
| `tKVgKV` | KV | g (GMEM) | KV (State 的 GMEM view) | State 在 GMEM 中的 view (用于 load/store State) |
| `tKVcV` | KV | c (坐标) | V | KV-MMA 的坐标映射 tensor (用于 predicate) |
| `tKVcS` | KV | c | S (State) | KV-MMA 的 State 坐标映射 tensor |
| `tOcO` | O1 | c | O | O1-MMA 的坐标映射 tensor |

#### 5.7.3 SMEM tensor 命名：`s{数据}{用途后缀}`

SMEM tensor 的命名更直白，但同一块 SMEM 可能有不同的"视图"(不同 layout)：

| 变量名 | 含义 | 说明 |
|--------|------|------|
| `sQqk` | Q 在 SMEM 中，用于 QK 相关 MMA | Q buffer 用 QK 计算需要的 layout (QKSmemLayoutQ) |
| `sKqk` | K 在 SMEM 中，用于 QK 相关 MMA | K buffer 用 QK 计算需要的 layout (QKSmemLayoutK) |
| `sVkv` | V 在 SMEM 中，用于 KV 相关 MMA | V buffer 用 KV 计算需要的 layout (KVSmemLayoutV) |
| `sKkv` | K 在 SMEM 中，用于 KV 相关 MMA | **同一块** smem_k，但换了 KVSmemLayoutK (转置布局) |
| `sAqkq` | Alpha 在 SMEM 中，用于 QK 计算的 Q 端 | Alpha buffer 用 Q 端 layout |
| `sAqkk` | Alpha 在 SMEM 中，用于 QK 计算的 K 端 | **同一块** smem_alpha，但 K 端 layout (compute_aux_safe 中 K 端 alpha) |
| `sAlast` | Alpha last 在 SMEM 中 | α 最后一行的独立 buffer |
| `sQK` | QK 结果在 SMEM 中 | smem_qk buffer |
| `sKK_inv` | KK 在 SMEM 中 (float, 用于 inverse) | smem_kk 以 float 读写 (CollectiveInverse) |
| `sKK_opd` | KK 在 SMEM 中 (bf16, 用于 MMA operand) | **同一块** smem_kk，reinterpret 为 bf16 给 NewV-MMA 做 B 操作数 |
| `sQ_K_scaled` | Q_scaled 和 K_scaled 共享的 2-stage buffer | stage 0 = Q_scaled (后被 K_decay 覆写)，stage 1 = K_scaled |
| `sQ_K_scaled_Kt` | 同上，但用转置 layout | 给 KV-MMA 做 B 操作数时需要列优先布局 |
| `sO` | Output 在 SMEM 中 | smem_o buffer，R2S 写入后 TMA store 到 GMEM |

**关键洞察:** 同一块物理 SMEM (`storage.smem_k`) 被包装成多个不同 layout 的 tensor view：`sKqk` 用行优先给 QK 计算，`sKkv` 用列优先给 KV 计算。代码中变量名后缀准确指示了这个 view 服务于哪个计算。

#### 5.7.4 compute_aux_safe 中的 SubChunk 变量

compute_aux_safe 的变量名包含额外的坐标后缀，因为使用 16×16 SubChunk 策略：

| 变量名 | 含义 |
|--------|------|
| `tQKrQ_1_0` | QK-MMA 的 Q fragment，SubChunk 索引 (1,0) |
| `tQKrKt_1_0` | QK-MMA 的 K^T fragment，SubChunk 索引 (1,0) |
| `sQqk_r_j` | Q 在 SMEM 中，行块 r，列块 j |
| `tArA_r_j` | Alpha fragment，行块 r，列块 j |
| `tArAfirst_r_j_kt` | Alpha first (锚点)，行块 r 列块 j，K^T 端 layout |

这里 `_r_j` 中的 `r` 是 SubChunk 行块索引，`j` 是列块索引，不要和 `r` = register 混淆 (位于变量名不同位置)。

#### 5.7.5 Pipeline 状态命名

Pipeline 读写状态的命名规则：`{数据}_smem_pipe_{read/write}`

| 变量名 | 含义 |
|--------|------|
| `q_smem_pipe_read` / `q_smem_pipe_write` | Q pipeline 的消费端 / 生产端状态 |
| `k_smem_pipe_read` / `k_smem_pipe_write` | K pipeline 的消费端 / 生产端状态 |
| `alpha_smem_pipe_read` / `alpha_smem_pipe_write` | Alpha pipeline 的消费端 / 生产端状态 |
| `o_smem_pipe_read` / `o_smem_pipe_write` | O pipeline 的消费端 / 生产端状态 |
| `qk_smem_pipe_read` / `qk_smem_pipe_write` | QK pipeline 的消费端 / 生产端状态 |
| `kk_smem_pipe_read` / `kk_smem_pipe_write` | KK pipeline 的消费端 / 生产端状态 |
| `beta_smem_pipe_read` / `beta_smem_pipe_write` | Beta pipeline 的消费端 / 生产端状态 |
| `alpha_last_smem_pipe_read` / `alpha_last_smem_pipe_write` | Alpha last pipeline 的消费端 / 生产端状态 |

PipelineState 含两个字段：`.index()` = 当前 stage 编号 (0 或 1)，`.phase()` = barrier phase bit。

#### 5.7.6 SMEM buffer 命名 (SharedStorage 中)

SharedStorage 中的 buffer 命名直接反映存储内容：

| 成员 | 数据 | 元素类型 | 大小 |
|------|------|---------|------|
| `smem_q` | Q 矩阵 | bf16 | 64×128×2stage = 32 KB |
| `smem_k` | K 矩阵 | bf16 | 128×64×2stage = 32 KB |
| `smem_v` | V 矩阵 | bf16 | 128×64×1stage = 16 KB |
| `smem_alpha` | Alpha 矩阵 | float | 64×128×2stage = 64 KB |
| `smem_qk` | QK 结果 | float | 64×64×2stage = 32 KB → 实际 16 KB (占一半) |
| `smem_kk` | KK / T^{-1} | float | 64×64×2stage = 32 KB → 实际 16 KB |
| `smem_q_k_scaled` | Q_scaled + K_scaled | bf16 | 64×128×2stage = 32 KB |
| `smem_o` | Output | bf16 | 64×128×1stage = 16 KB |
| `smem_beta` | Beta 向量 | bf16 | 64×2stage = 0.5 KB |
| `smem_alpha_last` | Alpha 最后一行 | float | 128×2stage = 1 KB |

#### 5.7.7 快速解码口诀

遇到一个陌生变量，按这个顺序拆解：

```
t  SK  r  V
│   │  │  └── 语义: V 数据
│   │  └───── 位置: 寄存器 fragment
│   └──────── MMA: State@K 计算
└──────────── 前缀: Tensor fragment

s  Q  qk
│  │  └──── 用途后缀: 用于 QK 计算
│  └─────── 数据: Q 矩阵
└────────── 前缀: SMEM tensor

s  Q_K_scaled_Kt
│  └────────────── 数据 + 布局变体: Q/K scaled 的 K 转置 layout
└───────────────── 前缀: SMEM tensor
```

**双重身份变量** `tOrV_or_tKVrV` 是最特殊的命名 — 它字面拼出了两个 MMA 角色：O2-MMA 的 A 操作数 (V^T) **或者** KV-MMA 的 A 操作数 (V^T)。这是因为 `make_acc_into_op` 把 NewV 结果 (`tNewVrC`) 重解释为两个不同 MMA 共用的 A 操作数，物理上是同一组寄存器。

### 5.8 compute() 的 6 步流程速查

| Step | 公式 | 代码行 | 关键 MMA | 输入 pipeline | 输出/释放 |
|------|------|--------|----------|--------------|----------|
| A | Q_scaled = Q⊙exp(gc), K_scaled = K⊙exp(gc) | L1025-1096 | 无 (element-wise) | wait α/Q/K | → SMEM sQ_K_scaled |
| B | O_inter = scale × Q_scaled @ S_old | L1098-1135 | TiledMmaO1 (RS) | | release Q |
| C | SK=S@K, V'=V-SK, T^{-1}, NewV=V'@T^{-T} | L1137-1218 | TiledMmaSK + NewV (RS) | wait V/KK/β | release V/KK/β |
| D | O = O_inter + NewV^T @ QK | L1220-1245 | TiledMmaO2 (RS) | wait QK | release QK, o_store |
| E | S *= exp(α_last), K_decay = K⊙exp(α_last-α) | L1248-1318 | 无 (element-wise) | wait α_last | release α_last |
| F | S_new = S_decayed + NewV^T @ K_decay^T | L1320-1346 | TiledMmaKV (RS) | | release α/K |

---

## 6. 学习进度记录

| 日期 | 完成项 | 笔记 |
|------|--------|------|
| 2026-04-20 | 4.1 get_register_requirements | total_registers=65536/128=512 是 4 个 WG per-thread 预算总和; 减去 load(24) 和 aux(152) 后平分给 2 个 State MMA WG 得 168; 合计 (24+168+168+152)×128=65536 用满整个 SM; occupancy=1 但 16 warp 全驻留, 靠 pipeline 隐藏延迟 |
| 2026-04-20 | 4.2 IndividualTileScheduler | grid.x = num_seqs × num_heads; 每 block 通过 blockIdx.x 除余得到 (seq_idx, head_idx); cu_seqlens 查表得 tok_offset/seq_len; scheduled flag 保证只调度一次; for 循环形式兼容未来 PersistentScheduler; SM100 已有 StaticPersistentTileScheduler |
| 2026-04-20 | 4.3 Options 系统 | 编译期 key-value dict: tuple 存储, find_option_t 递归查找(带默认值), add_option 追加; 消费侧 constexpr + if constexpr 消除分支; 构建处在 prefill_kernel_kda_fwd_sm90.cuh:80-90 |
| 2026-04-20 | 4.4 operator() | 三阶段: (1) 身份识别: threadIdx→WG/warp角色+leader选举+TMA描述符预取; (2) 9条Pipeline配置: TmaPipeline(Q/K/V/α)用 transaction_bytes, AsyncPipeline(O/QK/KK/β/α_last)用 arrive_count, LoadAlpha warp 身兼 alpha consumer + alpha_last producer; (3) 全局同步→寄存器重分配→角色分发到 6 个不同 mainloop 函数 |
| 2026-04-21 | 4.5 load_qkv() | CollectiveLoadTma: partition_SD 分区 (Q/α seqlen,d; K/V d,seqlen 转置) → 逐tile step (acquire→TMA async copy with barrier→++pipe); TMA 硬件搬运只需 leader; K/V 转置为满足 WGMMA B 矩阵列连续; alpha_last_pipeline 存在因 state 更新只需最后一行 alpha, 由专门 LoadAlpha warp 提取 |
| 2026-04-21 | 4.6 load_beta() | CollectiveLoadVector: beta 是 per-token 标量 (256B/tile) 太小不适合 TMA; 32线程 warp 手动 GMEM→reg→SMEM; 尾 tile 用 mask+fill(0) 防越界; fence_view_async_shared 保证 SMEM 写可见后再 producer_commit |
| 2026-04-21 | 4.7 extract_alpha_last() | WG0.Warp3 执行 SMEM→SMEM 中转; 双pipeline协调 (consumer_wait alpha + producer_acquire alpha_last); 32线程stride取第B-1行(128 float); 释放顺序: 先commit下游再release上游防覆盖 |
| 2026-04-21 | 4.8 store() | CollectiveStoreTma: o_pipeline consumer; 尾tile不安全时动态修改TMA descriptor裁剪seqlen维; per-SM workspace; acquire-release语义 |
| 2026-04-21 | 4.9 compute_aux_safe() | MathA WG计算gated QK和KK; 16×16 SubChunk减少寄存器压力; 128线程分两组并行; TF32 MMA精度更高; BF16 LDSM+warp shuffle布局转换节省50% SMEM带宽; α_first通过shuffle广播; 下三角+边界mask; QK*=scale, KK*=beta |
| 2026-04-23 | 1.x 公式体系重写 | 对照论文 Eq.1,6-9 和伪代码重建完整公式体系; 补充 safe_gate anchor trick 推导 (指数加减=乘除, 插入 ref 拆分溢出); 补充 S_decayed/K_decay 来源推导 (γ_C 总衰减 vs γ_{i→C} 部分衰减, K_scaled 和 K_decay 互补) |
| 2026-04-23 | 4.10 Step B O_inter | State(reg) @ Q_scaled(SMEM) RS-WGMMA; make_acc_into_op 转换布局; math_barriers 有序执行; 首tile跳过; o1_epi *= scale |
| 2026-04-23 | 4.10 Step C V'→NewV | 4子步: C1 WGMMA SK=S@K_scaled(stage1), C2 V'=V-SK(LDSM_T+element-wise), C3 CollectiveInverse求T^{-1}(Math0 only,in-place)+beta列缩放, C4 WGMMA NewV=V'@T^{-T}; release v/kk/β pipeline |
| 2026-04-27 | 4.10 Step D/E/F | D: NewV^T@QK → O, gemm_zero_acc(首tile) vs gemm(非首, 累加O_inter), o_store写SMEM; E: s_decay(State*=exp(α_last), 按列), K_decay=K*exp(α_last-α) 写stage0覆写Q_scaled; F: WGMMA S_new=S_decayed+NewV^T@K_decay^T 累加到tKVrKV, release α/K |
| 2026-04-29 | 4.11 CollectiveLoadTma | 5个模板参数(Kind/Pipeline/Element/SmemLayout/TMA); partition_SD: Q/α(seqlen,d) vs K/V(d,seqlen)转置→WGMMA B列连续; step: leader-only TMA async copy with barrier→立即返回; LoadQBytes=16KB(bf16), LoadAlphaBytes=32KB(float); SMEM swizzle消除bank conflict |
| 2026-04-29 | 5.7 变量命名规则 | 前缀系统(t/s/g/c), MMA角色编码(tMMA+r/s+Operand), SMEM视图命名(sData+用途后缀), Pipeline状态命名, SubChunk坐标后缀, 双重身份变量tOrV_or_tKVrV, 解码口诀 |
| 2026-04-29 | 0.6 维度体系全景图 | 5层维度: GMEM全局(Q/K/V/O/α/β/State shape+layout), Tile级MMA维度(9种TileShape+M×N×K+Step映射), SMEM buffer(含stage+多视图), flat_divide分片(StepA/E的tiler_qk 2WG×2iter=128, SubChunk 16×16×32的r/j/BK循环), 寄存器fragment(tKVrKV=128 reg/thread), 公式→代码维度速查 |

---

**建议**: 每学完一个函数，在对应 TODO 项打勾 `[x]`，并在"学习进度记录"中添加笔记。
