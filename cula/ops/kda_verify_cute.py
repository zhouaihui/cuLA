"""CuTe DSL Fused KDA Verify Kernels for Multi-Token Prediction.

=============================================================================
KDA (Kernel Delta Attention) Recurrence — Single Token Update
=============================================================================

Dimensions:
    N   — batch size (number of sequences)
    T   — spec_len (number of draft tokens, e.g. 1~8)
    H   — number of QK heads
    HV  — number of value heads (GQA: HV >= H, every HV//H value heads share one QK head)
    K   — key/query head dimension (typically 128)
    V   — value head dimension (typically 128)

State matrix:
    S ∈ R^{K×V}  — per (batch, value_head) hidden state, stored in SMEM

Input vectors (per token, per head):
    q ∈ R^K      — query
    k ∈ R^K      — key
    v ∈ R^V      — value
    g ∈ R^K      — decay gate vector (KDA-specific, derived from a + dt_bias)
    β ∈ R        — sigmoid gate scalar (controls delta update strength)

Update formulas (per token step):

    Step 1 — Decay: apply per-dimension exponential decay to state
        g = -exp(A_log) · softplus(a + dt_bias)     g ∈ R^K, g < 0
        S' = diag(exp(g)) · S                       S'[k,:] = exp(g[k]) · S[k,:]

    Step 2 — Delta rule update:
        S_new = S' + β · k · vᵀ - β · k · (kᵀ · S')

        Equivalent factored form:
        S_new = S' + β · k · (v - kᵀ · S')ᵀ

        where:
            kᵀ · S' ∈ R^V         — what the decayed state "predicts" for key k
            v - kᵀ · S' ∈ R^V     — residual (delta) between actual v and prediction
            β · k · (...)ᵀ        — gated rank-1 outer product update

    Step 3 — Output: query the updated state
        o = S_newᵀ · q  ∈ R^V

    Code implements Step 2 in three in-place sub-steps on SMEM:
        v_delta = v - kᵀ · S'    # residual: what the state didn't predict
        v_delta *= β              # gate: scale the update strength
        S += k ⊗ v_delta         # rank-1 outer product update

=============================================================================
Verify vs Decode
=============================================================================
    Decode kernel:  processes 1 token, writes final state back to state pool.
    Verify kernel:  processes T draft tokens sequentially, writes intermediate
                    state snapshot after each token to a buffer, does NOT modify
                    the state pool (disable_state_update=True by default).

    After rejection sampling, the framework picks intermediate_states[n, accepted_len]
    as the correct state for sequence n.

=============================================================================
CuTe Kernel Architecture
=============================================================================
    Loop nesting: for hv → for v_tile → for t (token).

    sData holds one V-tile in SMEM and persists across tokens; q/k/a/b are
    reloaded each token step. This trades redundant q/k GMEM reads (~1KB/token)
    for saved state GMEM round-trips (~8KB/token per V-tile).

    Two kernel families based on batch size:
      - small_batch: 128 threads, one CTA per (batch, head_group, v_tiles),
        iterates over HV heads serially. Used when N < SMALL_BATCH_THRESHOLD.
      - large_batch: 256 threads, one CTA per (batch, value_head, v_tile).

    Each family has a varlen variant for flattened (1, N*T, H, K) layout.
"""

import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from cula.ops.kda_decode import (
    DENSE_SMALL_HV_PARALLEL_HEAD_THRESHOLD,
    DENSE_SMALL_HV_PARALLEL_MAX_N,
    N4_DENSE_SMALL_HV_PARALLEL_HEAD_THRESHOLD,
    NUM_K_ITERS,
    NUM_STAGES,
    NUM_THREADS,
    NUM_THREADS_LARGE,
    NUM_WARPS_LARGE,
    ROWS_PER_ITER,
    SMALL_BATCH_THRESHOLD,
    TILE_K,
    TILE_V,
    TILE_V_PADDED,
    TILE_V_SMALL,
    TILE_V_SMALL_PADDED,
    V_PER_WARP,
    _canonicalize_state_layout,
    _get_cached_stream,
    _normalize_A_log,
    _normalize_dt_bias,
    _select_small_blocks_per_state,
)

logger = logging.getLogger(__name__)

_compiled_verify_kernels: dict[tuple, object] = {}
_verify_jit_functions = None


def _define_verify_kernels():
    """Define CuTe DSL kernels for KDA verify (multi-token prediction)."""

    # =========================================================================
    # Small-batch kernel tuning constants
    # =========================================================================
    NUM_WARPS_SMALL = 4                              # 128 threads / 32 = 4 warps
    V_PER_WARP_SMALL = TILE_V_SMALL // NUM_WARPS_SMALL  # V elements owned by each warp
    ROWS_PER_ITER_SMALL = 32 // V_PER_WARP_SMALL     # K-rows each thread iterates per k_iter
    NUM_K_ITERS_SMALL = TILE_K // ROWS_PER_ITER_SMALL  # number of iterations to cover full K dim

    @cute.kernel
    def kda_verify_kernel_small_batch(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,             # (pool_size, HV, K, V) or (pool_size, HV, V, K) fp32
        smem_layout_staged: cute.Layout,    # SMEM layout for sData: (TILE_K, TILE_V_SMALL, NUM_STAGES)
        num_v_tiles: cutlass.Constexpr[int],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        q: cute.Tensor,                     # (N, T, H, K) bf16 queries
        k: cute.Tensor,                     # (N, T, H, K) bf16 keys
        v: cute.Tensor,                     # (N, T, HV, V) bf16 values
        a: cute.Tensor,                     # (N, T, HV, K) bf16 KDA vector gate input
        b: cute.Tensor,                     # (N, T, HV) bf16 sigmoid beta gate input
        A_log: cute.Tensor,                 # (HV,) fp32 decay log parameter
        dt_bias: cute.Tensor,               # (HV, K) fp32 gate bias
        o: cute.Tensor,                     # (N, T, HV, V) bf16 output logits
        h0_indices: cute.Tensor,            # (N,) int32 maps batch → state pool index
        intermediate_states_buffer: cute.Tensor,  # (N, T, HV, K, V) or (N, T, HV, V, K) fp32
        is_indices: cute.Tensor,            # (N,) int32 maps batch → intermediate state index
        softplus_beta: cutlass.Constexpr[float],    # softplus sharpness β
        softplus_threshold: cutlass.Constexpr[float],  # softplus overflow threshold
        scale: cutlass.Constexpr[float],    # attention scale, typically K^{-0.5}
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        max_steps: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],  # True → (HV, K, V), False → (HV, V, K)
        dense_small_hv_parallel: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],
        disable_state_update: cutlass.Constexpr[bool],
    ):
        """Small-batch dense KDA verify kernel for q/k/v shaped (N, T, ...).

        For each (hv, v_tile), processes max_steps tokens with state in SMEM.
        128 threads (4 warps) per CTA. Multiple V-tiles may be assigned to one
        CTA when num_blocks_per_state_small > 1.
        """
        del tiled_copy_load
        # =========================================================================
        # Thread and block ID mapping
        # =========================================================================
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32   # lane within warp (0..31)
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)  # 0..NUM_WARPS_SMALL-1
        block_idx, _, _ = cute.arch.block_idx()

        # Block → (batch, v_tile_group) mapping
        batch_idx = block_idx // num_blocks_per_state_small
        batch_inner = block_idx % num_blocks_per_state_small
        num_v_tiles_per_block = num_v_tiles // num_blocks_per_state_small
        start_v_tile = batch_inner * num_v_tiles_per_block

        # =========================================================================
        # GQA head mapping: derive i_n, i_h, i_hv_base from batch_idx
        #
        # Two strategies depending on dense_small_hv_parallel:
        #   True  → each CTA handles exactly one value head (parallel across HV)
        #   False → each CTA handles one QK head, iterates over its value heads
        # =========================================================================
        num_value_heads_per_q = HV // H
        i_n = 0
        i_h = 0
        i_hv_base = 0
        num_hv_iters = 1
        if dense_small_hv_parallel:
            # One CTA per (batch, value_head) — no HV iteration needed
            i_n = batch_idx // HV
            i_hv_base = batch_idx % HV
            i_h = i_hv_base // num_value_heads_per_q
            num_hv_iters = 1
        else:
            # One CTA per (batch, QK head) — iterate over HV//H value heads
            i_n = batch_idx // H
            i_h = batch_idx % H
            i_hv_base = i_h * num_value_heads_per_q
            num_hv_iters = num_value_heads_per_q

        # =========================================================================
        # State pool lookup — indirect index per batch element
        # idx < 0 means no initial state (all zeros)
        # =========================================================================
        pool_idx = h0_indices[i_n]
        is_idx = is_indices[i_n]

        if pool_idx >= 0:
            # =========================================================================
            # Thread-to-element mapping for SMEM state tile sData[K, V_SMALL]
            #
            # Each thread (tidx) owns one (k_local, v_local) element per k_iter.
            # The mapping depends on state layout:
            #   KV: thread strip-mines V dimension  →  v_load = tidx % V, k iterates
            #   VK: thread strip-mines K dimension  →  k_load = tidx % K, v iterates
            # =========================================================================
            k_local = in_warp_tid // V_PER_WARP_SMALL     # K-row index within warp's rows
            v_local = in_warp_tid % V_PER_WARP_SMALL      # V-col index within warp's columns
            v_base = warp_idx * V_PER_WARP_SMALL          # V offset for this warp
            v_idx = v_base + v_local                       # global V index for this thread

            # =========================================================================
            # SMEM allocation
            # sData:  (TILE_K, TILE_V_SMALL, NUM_STAGES) — state tile, persists across tokens
            # sK/sQ:  (TILE_K,) — key/query, reloaded each token
            # sG:     (TILE_K,) — decay gate exp(g), computed each token
            # sGK:    (TILE_K,) — exp(g) * k, precomputed for delta-rule update
            # smem_o: (TILE_V_SMALL,) — scratch for L2 norm reduction
            # =========================================================================
            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V_SMALL,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_q_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_g_layout = cute.make_layout((TILE_K,), stride=(1,))
            smem_gk_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)
            sG = smem.allocate_tensor(cutlass.Float32, smem_g_layout, 128)
            sGK = smem.allocate_tensor(cutlass.Float32, smem_gk_layout, 128)

            # Thread-to-(K,V) mapping for GMEM load/store, depends on state layout
            kv_v_load = 0
            kv_k_load_base = 0
            kv_k_load_step = 0
            vk_k_load = 0
            vk_v_load_base = 0
            vk_v_load_step = 0
            if state_layout_is_kv:
                # KV layout [HV, K, V]: thread strip-mines V, iterates K
                kv_v_load = tidx % TILE_V_SMALL
                kv_k_load_base = tidx // TILE_V_SMALL
                kv_k_load_step = NUM_THREADS // TILE_V_SMALL
            else:
                # VK layout [HV, V, K]: thread strip-mines K, iterates V
                vk_k_load = tidx % TILE_K
                vk_v_load_base = tidx // TILE_K
                vk_v_load_step = NUM_THREADS // TILE_K

            # =========================================================================
            # Outer loops: HV heads → V-tiles → tokens
            # =========================================================================
            for hv_offset in range(num_hv_iters):
                i_hv = i_hv_base + hv_offset

                for v_tile_offset in range(num_v_tiles_per_block):
                    stage = v_tile_offset % NUM_STAGES
                    v_tile = start_v_tile + v_tile_offset
                    v_global_base = v_tile * TILE_V_SMALL

                    # =============================================================
                    # Load initial state h0 into sData (once per v_tile)
                    #
                    # Threads cooperatively load the full [TILE_K, TILE_V_SMALL]
                    # state tile from GMEM. Each thread loads one element per k_iter,
                    # covering all K rows over NUM_K_ITERS_SMALL iterations.
                    # out-of-bounds V elements (v_global_load >= V) are zero-filled.
                    # =============================================================
                    for k_iter in range(NUM_K_ITERS_SMALL):
                        k_load = 0
                        v_load = 0
                        if state_layout_is_kv:
                            k_load = kv_k_load_base + k_iter * kv_k_load_step
                            v_load = kv_v_load
                        else:
                            k_load = vk_k_load
                            v_load = vk_v_load_base + k_iter * vk_v_load_step
                        v_global_load = v_global_base + v_load
                        h_val = 0.0
                        if v_global_load < v.shape[3]:
                            if state_layout_is_kv:
                                h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, k_load, v_global_load)])
                            else:
                                h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, v_global_load, k_load)])
                        sData[(k_load, v_load, stage)] = h_val
                    cute.arch.barrier()

                    # =============================================================
                    # Main token loop: process max_steps draft tokens sequentially
                    # sData persists across iterations; q/k/a/b are reloaded.
                    # =============================================================
                    for t in range(max_steps):

                        # --- Reload q[t], k[t] into SMEM (first TILE_K threads) ---
                        if tidx < TILE_K:
                            sK[tidx] = cutlass.Float32(k[i_n, t, i_h, tidx])
                            sQ[tidx] = cutlass.Float32(q[i_n, t, i_h, tidx])
                        cute.arch.barrier()

                        # =============================================================
                        # L2 normalization of q and k (optional)
                        # Keeps q, k on the unit sphere, preventing unbounded
                        # state/output growth. Reduces within warp 0 using
                        # butterfly shuffle, then broadcasts via smem_o.
                        # =============================================================
                        if use_qk_l2norm:
                            sum_q_partial = 0.0
                            sum_k_partial = 0.0
                            if warp_idx == 0:
                                for norm_iter in range(4):
                                    norm_idx = in_warp_tid + norm_iter * 32
                                    q_val = sQ[norm_idx]
                                    k_val = sK[norm_idx]
                                    sum_q_partial += q_val * q_val
                                    sum_k_partial += k_val * k_val

                                for offset in [16, 8, 4, 2, 1]:
                                    sum_q_partial += cute.arch.shuffle_sync_bfly(sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31)
                                    sum_k_partial += cute.arch.shuffle_sync_bfly(sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31)

                                if in_warp_tid == 0:
                                    smem_o[0] = cute.rsqrt(sum_q_partial + 1e-6)
                                    smem_o[1] = cute.rsqrt(sum_k_partial + 1e-6)
                            cute.arch.barrier()

                            inv_norm_q = smem_o[0]
                            inv_norm_k = smem_o[1]

                            if tidx < TILE_K:
                                sK[tidx] = sK[tidx] * inv_norm_k
                                sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                            cute.arch.barrier()
                        else:
                            if tidx < TILE_K:
                                sQ[tidx] = sQ[tidx] * scale
                            cute.arch.barrier()

                        # --- Apply attention scale (q only; k already scaled by L2 norm if enabled) ---
                        # When L2 norm is off, only q gets scaled here:
                        # sQ = q * scale  (scale = K^{-0.5})
                        # When L2 norm is on, q was already set to q * scale * inv_norm_q above.

                        # =============================================================
                        # Step 1: Gate — compute per-dimension decay factor g ∈ R^K
                        #   g = -exp(A_log) · softplus(a + dt_bias)
                        #
                        # A_log < 0 (initialized as -rand*2) → exp(A_log) ∈ (0, 1)
                        # softplus(x) > 0 for all x → g < 0 → exp(g) ∈ (0, 1) → decay.
                        #
                        # KDA's key difference from standard linear attention: g is a K-dim
                        # vector (not a scalar), so each row of the state matrix decays
                        # independently.
                        #
                        # CuTe implementation: A_log is scalar per HV, so only lane 0 loads
                        # it and broadcasts via shuffle. a + dt_bias is per-K, loaded by the
                        # first TILE_K threads.
                        # =============================================================

                        # Load A_log (scalar per value head) — only lane 0, then broadcast
                        r_exp_A = 0.0
                        if in_warp_tid == 0:
                            r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]))
                        r_exp_A = cute.arch.shuffle_sync(r_exp_A, 0)

                        # Per-K gate computation (first TILE_K threads)
                        # Parameterized softplus: softplus_β(x) = (1/β) · log(1 + e^(βx))
                        # β controls sharpness near zero: β=1 → standard softplus; β→∞ → ReLU.
                        # softplus_beta is a hyperparameter tunable at the model level; the kernel
                        # must adapt to whatever value the caller passes, not assume β=1.
                        if tidx < TILE_K:
                            r_a_k = cutlass.Float32(a[i_n, t, i_hv, tidx])
                            r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                            x = r_a_k + r_dt_bias_k
                            beta_x = softplus_beta * x
                            softplus_x = 0.0
                            if beta_x <= softplus_threshold:
                                exp_beta_x = cute.exp(beta_x)
                                log_input = cutlass.Float32(1.0 + exp_beta_x)
                                log_result = cutlass.Float32(cute.log(log_input))
                                softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
                            else:
                                # large βx: (1/β)·log(1+e^(βx)) ≈ x, avoids exp overflow
                                softplus_x = x
                            sG[tidx] = cute.exp(-r_exp_A * softplus_x)  # decay factor exp(g[k])

                        # =============================================================
                        # β = sigmoid(b) — controls delta update strength
                        # Unlike softplus_beta (a function-shape hyperparameter), this β is
                        # the *output* of sigmoid on a learned scalar b_b — the network
                        # learns b_b directly, so no extra parameterization of sigmoid is needed.
                        # =============================================================
                        r_beta = 0.0
                        if in_warp_tid == 0:
                            r_b = cutlass.Float32(b[i_n, t, i_hv])
                            r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                        r_beta = cute.arch.shuffle_sync(r_beta, 0)

                        # Precompute sGK = exp(g) * k for reuse in decay + delta-rule
                        # This avoids computing exp(g[k]) * k[k] twice (once for kᵀ·S'
                        # and once for k ⊗ v_delta).
                        if tidx < TILE_K:
                            sGK[tidx] = sG[tidx] * sK[tidx]
                        cute.arch.barrier()

                        # =============================================================
                        # Delta-rule update on sData
                        #
                        # Combined decay + update in three sub-steps:
                        #   (a) sum_hk = kᵀ · S' = Σ_k S[k,v] · exp(g[k]) · k[k]
                        #       (S' = diag(exp(g)) · S, so we use sGK = exp(g)*k
                        #        which also applies the decay implicitly)
                        #   (b) v_new = β · (v - sum_hk)         — gated delta
                        #   (c) S_new[k,v] = S'[k,v] + k[k] · v_new[v] — rank-1 update
                        #       where S'[k,v] = exp(g[k]) · S[k,v] = sG[k] * sData[k,v]
                        # =============================================================

                        # Load v[t] from GMEM (each thread loads one V element)
                        v_global = v_tile * TILE_V_SMALL + v_idx
                        r_v = 0.0
                        if v_global < v.shape[3]:
                            r_v = cutlass.Float32(v[i_n, t, i_hv, v_global])

                        # (a) kᵀ · S' — dot product along K dim → prediction for v
                        # Each thread accumulates over K rows in NUM_K_ITERS_SMALL steps.
                        # Each warp covers V_PER_WARP_SMALL V-elements in parallel.
                        sum_hk = 0.0
                        for k_iter in range(NUM_K_ITERS_SMALL):
                            k_base = k_iter * ROWS_PER_ITER_SMALL
                            k_idx = k_base + k_local
                            sum_hk += sData[(k_idx, v_idx, stage)] * sGK[k_idx]

                        # Reduce within warp: sum across K-rows (V_PER_WARP_SMALL threads
                        # hold partial sums for the same V element, at offsets 0..V_PER_WARP_SMALL-1)
                        for offset in [4, 2, 1]:
                            sum_hk += cute.arch.shuffle_sync_bfly(
                                sum_hk,
                                offset=offset * V_PER_WARP_SMALL,
                                mask=-1,
                                mask_and_clamp=31,
                            )

                        # (b) Gate the delta: v_new = β · (v - kᵀ · S')
                        v_new = (r_v - sum_hk) * r_beta
                        # Broadcast v_new to all lanes in the warp that share the same k_local
                        v_new = cute.arch.shuffle_sync(v_new, v_local)

                        # (c) Apply decay + rank-1 update, and compute output simultaneously
                        # S_new[k,v] = exp(g[k]) · S[k,v] + k[k] · v_new[v]
                        # Also accumulate: o[v] = Σ_k S_new[k,v] · q[k]
                        sum_hq = 0.0
                        for k_iter in range(NUM_K_ITERS_SMALL):
                            k_base = k_iter * ROWS_PER_ITER_SMALL
                            k_idx = k_base + k_local
                            h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]   # S'[k,v] = exp(g[k]) · S[k,v]
                            h_new = h_old + sK[k_idx] * v_new                  # S_new[k,v] = S'[k,v] + k[k] · v_new
                            sData[(k_idx, v_idx, stage)] = h_new
                            sum_hq += h_new * sQ[k_idx]                        # accumulate o[v] = Σ_k S_new[k,v] · q[k]

                        # Warp reduce sum_hq (same pattern as sum_hk)
                        for offset in [4, 2, 1]:
                            sum_hq += cute.arch.shuffle_sync_bfly(
                                sum_hq,
                                offset=offset * V_PER_WARP_SMALL,
                                mask=-1,
                                mask_and_clamp=31,
                            )

                        # Write output — only k_local==0 threads hold the final sum
                        if k_local == 0 and v_global < v.shape[3]:
                            o[(i_n, t, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

                        cute.arch.barrier()

                        # =============================================================
                        # Step 5 (verify-only): Snapshot intermediate state to buffer
                        # Save sData → intermediate_states_buffer[is_idx, t, i_hv, :, :]
                        # so the framework can roll back to the correct state after
                        # rejection sampling. Layout-aware: KV or VK indexing.
                        # =============================================================
                        if cache_intermediate_states:
                            for k_iter in cutlass.range(NUM_K_ITERS_SMALL, unroll=2):
                                k_write = 0
                                v_write = 0
                                if state_layout_is_kv:
                                    k_write = kv_k_load_base + k_iter * kv_k_load_step
                                    v_write = kv_v_load
                                else:
                                    k_write = vk_k_load
                                    v_write = vk_v_load_base + k_iter * vk_v_load_step
                                v_global_write = v_global_base + v_write
                                if v_global_write < v.shape[3]:
                                    if state_layout_is_kv:
                                        intermediate_states_buffer[(is_idx, t, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                                    else:
                                        intermediate_states_buffer[(is_idx, t, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]

                    # =============================================================
                    # Optional: write final state back to state pool (decode mode)
                    # Verify mode (disable_state_update=True) skips this — draft
                    # tokens may be rejected, so the framework selects the correct
                    # intermediate state from the snapshot buffer instead.
                    # =============================================================
                    # Write final state back to pool (decode mode)
                    if not disable_state_update:
                        cute.arch.barrier()
                        for k_iter in cutlass.range(NUM_K_ITERS_SMALL, unroll=2):
                            k_write = 0
                            v_write = 0
                            if state_layout_is_kv:
                                k_write = kv_k_load_base + k_iter * kv_k_load_step
                                v_write = kv_v_load
                            else:
                                k_write = vk_k_load
                                v_write = vk_v_load_base + k_iter * vk_v_load_step
                            v_global_write = v_global_base + v_write
                            if v_global_write < v.shape[3]:
                                if state_layout_is_kv:
                                    h0_source[(pool_idx, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                                else:
                                    h0_source[(pool_idx, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]

                    cute.arch.barrier()

    @cute.kernel
    def kda_verify_kernel_small_batch_varlen(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        intermediate_states_buffer: cute.Tensor,
        is_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        max_steps: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],
        disable_state_update: cutlass.Constexpr[bool],
    ):
        """Small-batch varlen KDA verify kernel. q/k: (1, N*T, H, K), a: (N*T, HV, K).

        Same logic as kda_verify_kernel_small_batch but for varlen layout where
        the batch and token dims are flattened: q/k shaped (1, N*T, H, K) instead
        of (N, T, H, K), and a shaped (N*T, HV, K) instead of (N, T, HV, K).
        Always parallelizes across HV (no hv iteration loop).
        """
        del tiled_copy_load
        # =========================================================================
        # Thread and block ID mapping (same structure as small_batch)
        # =========================================================================
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()

        batch_idx = block_idx // num_blocks_per_state_small
        batch_inner = block_idx % num_blocks_per_state_small
        num_v_tiles_per_block = num_v_tiles // num_blocks_per_state_small
        start_v_tile = batch_inner * num_v_tiles_per_block

        # Varlen always parallelizes across HV — one CTA per (batch, value_head)
        i_n = batch_idx // HV
        i_hv = batch_idx % HV
        i_h = i_hv // (HV // H)

        pool_idx = h0_indices[i_n]
        is_idx = is_indices[i_n]

        if pool_idx >= 0:
            # Thread-to-element mapping and SMEM allocation (same as small_batch)
            k_local = in_warp_tid // V_PER_WARP_SMALL
            v_local = in_warp_tid % V_PER_WARP_SMALL
            v_base = warp_idx * V_PER_WARP_SMALL
            v_idx = v_base + v_local

            # SMEM allocation (same as small_batch)
            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V_SMALL,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            smem_k_layout = cute.make_layout((TILE_K,), stride=(1,))
            sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
            sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((TILE_K,), stride=(1,)), 128)
            sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((TILE_K,), stride=(1,)), 128)
            sGK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((TILE_K,), stride=(1,)), 128)

            # State layout thread mapping (same as small_batch)
            kv_v_load = 0
            kv_k_load_base = 0
            kv_k_load_step = 0
            vk_k_load = 0
            vk_v_load_base = 0
            vk_v_load_step = 0
            if state_layout_is_kv:
                kv_v_load = tidx % TILE_V_SMALL
                kv_k_load_base = tidx // TILE_V_SMALL
                kv_k_load_step = NUM_THREADS // TILE_V_SMALL
            else:
                vk_k_load = tidx % TILE_K
                vk_v_load_base = tidx // TILE_K
                vk_v_load_step = NUM_THREADS // TILE_K

            # V-tile loop (no HV iteration loop — varlen always parallelizes across HV)
            for v_tile_offset in range(num_v_tiles_per_block):
                stage = v_tile_offset % NUM_STAGES
                v_tile = start_v_tile + v_tile_offset
                v_global_base = v_tile * TILE_V_SMALL

                # Load initial state into sData (same as small_batch)
                for k_iter in range(NUM_K_ITERS_SMALL):
                    k_load = 0
                    v_load = 0
                    if state_layout_is_kv:
                        k_load = kv_k_load_base + k_iter * kv_k_load_step
                        v_load = kv_v_load
                    else:
                        k_load = vk_k_load
                        v_load = vk_v_load_base + k_iter * vk_v_load_step
                    v_global_load = v_global_base + v_load
                    h_val = 0.0
                    if v_global_load < v.shape[3]:
                        if state_layout_is_kv:
                            h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, k_load, v_global_load)])
                        else:
                            h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, v_global_load, k_load)])
                    sData[(k_load, v_load, stage)] = h_val
                cute.arch.barrier()

                # Token loop: state persists in sData across steps
                for t in range(max_steps):
                    # Varlen uses flattened index: seq_idx = n * T + t
                    seq_idx = i_n * max_steps + t

                    # Reload q[t], k[t] into SMEM (note varlen indexing: dim0=1, dim1=seq_idx)
                    if tidx < TILE_K:
                        sK[tidx] = cutlass.Float32(k[0, seq_idx, i_h, tidx])
                        sQ[tidx] = cutlass.Float32(q[0, seq_idx, i_h, tidx])
                    cute.arch.barrier()

                    # L2 normalization (same as small_batch — see small_batch for detailed comments)
                    if use_qk_l2norm:
                        sum_q_partial = 0.0
                        sum_k_partial = 0.0
                        if warp_idx == 0:
                            for norm_iter in range(4):
                                norm_idx = in_warp_tid + norm_iter * 32
                                q_val = sQ[norm_idx]
                                k_val = sK[norm_idx]
                                sum_q_partial += q_val * q_val
                                sum_k_partial += k_val * k_val
                            for offset in [16, 8, 4, 2, 1]:
                                sum_q_partial += cute.arch.shuffle_sync_bfly(sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31)
                                sum_k_partial += cute.arch.shuffle_sync_bfly(sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31)
                            if in_warp_tid == 0:
                                smem_o[0] = cute.rsqrt(sum_q_partial + 1e-6)
                                smem_o[1] = cute.rsqrt(sum_k_partial + 1e-6)
                        cute.arch.barrier()
                        inv_norm_q = smem_o[0]
                        inv_norm_k = smem_o[1]
                        if tidx < TILE_K:
                            sK[tidx] = sK[tidx] * inv_norm_k
                            sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                        cute.arch.barrier()
                    else:
                        if tidx < TILE_K:
                            sQ[tidx] = sQ[tidx] * scale
                        cute.arch.barrier()

                    # Step 1: Gate — compute per-dimension decay factor (same as small_batch)
                    r_exp_A = 0.0
                    if in_warp_tid == 0:
                        r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]))
                    r_exp_A = cute.arch.shuffle_sync(r_exp_A, 0)
                    if tidx < TILE_K:
                        r_a_k = cutlass.Float32(a[seq_idx, i_hv, tidx])
                        r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                        x = r_a_k + r_dt_bias_k
                        beta_x = softplus_beta * x
                        softplus_x = 0.0
                        if beta_x <= softplus_threshold:
                            exp_beta_x = cute.exp(beta_x)
                            log_input = cutlass.Float32(1.0 + exp_beta_x)
                            log_result = cutlass.Float32(cute.log(log_input))
                            softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
                        else:
                            softplus_x = x
                        sG[tidx] = cute.exp(-r_exp_A * softplus_x)

                    r_beta = 0.0
                    if in_warp_tid == 0:
                        r_b = cutlass.Float32(b[seq_idx, i_hv])
                        r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                    r_beta = cute.arch.shuffle_sync(r_beta, 0)

                    # Precompute sGK = exp(g) * k (same as small_batch)
                    if tidx < TILE_K:
                        sGK[tidx] = sG[tidx] * sK[tidx]
                    cute.arch.barrier()

                    # Delta-rule update on sData (same as small_batch — see small_batch for detailed comments)
                    v_global = v_tile * TILE_V_SMALL + v_idx
                    r_v = 0.0
                    if v_global < v.shape[3]:
                        r_v = cutlass.Float32(v[0, seq_idx, i_hv, v_global])

                    sum_hk = 0.0
                    for k_iter in range(NUM_K_ITERS_SMALL):
                        k_base = k_iter * ROWS_PER_ITER_SMALL
                        k_idx = k_base + k_local
                        sum_hk += sData[(k_idx, v_idx, stage)] * sGK[k_idx]
                    for offset in [4, 2, 1]:
                        sum_hk += cute.arch.shuffle_sync_bfly(sum_hk, offset=offset * V_PER_WARP_SMALL, mask=-1, mask_and_clamp=31)

                    v_new = (r_v - sum_hk) * r_beta
                    v_new = cute.arch.shuffle_sync(v_new, v_local)

                    sum_hq = 0.0
                    for k_iter in range(NUM_K_ITERS_SMALL):
                        k_base = k_iter * ROWS_PER_ITER_SMALL
                        k_idx = k_base + k_local
                        h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]
                        h_new = h_old + sK[k_idx] * v_new
                        sData[(k_idx, v_idx, stage)] = h_new
                        sum_hq += h_new * sQ[k_idx]
                    for offset in [4, 2, 1]:
                        sum_hq += cute.arch.shuffle_sync_bfly(sum_hq, offset=offset * V_PER_WARP_SMALL, mask=-1, mask_and_clamp=31)

                    if k_local == 0 and v_global < v.shape[3]:
                        o[(0, seq_idx, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

                    cute.arch.barrier()

                    # Step 5: Snapshot intermediate state (same as small_batch)
                    if cache_intermediate_states:
                        for k_iter in cutlass.range(NUM_K_ITERS_SMALL, unroll=2):
                            k_write = 0
                            v_write = 0
                            if state_layout_is_kv:
                                k_write = kv_k_load_base + k_iter * kv_k_load_step
                                v_write = kv_v_load
                            else:
                                k_write = vk_k_load
                                v_write = vk_v_load_base + k_iter * vk_v_load_step
                            v_global_write = v_global_base + v_write
                            if v_global_write < v.shape[3]:
                                if state_layout_is_kv:
                                    intermediate_states_buffer[(is_idx, t, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                                else:
                                    intermediate_states_buffer[(is_idx, t, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]
                # Write final state back to pool (decode mode, same as small_batch)
                if not disable_state_update:
                    cute.arch.barrier()
                    for k_iter in cutlass.range(NUM_K_ITERS_SMALL, unroll=2):
                        k_write = 0
                        v_write = 0
                        if state_layout_is_kv:
                            k_write = kv_k_load_base + k_iter * kv_k_load_step
                            v_write = kv_v_load
                        else:
                            k_write = vk_k_load
                            v_write = vk_v_load_base + k_iter * vk_v_load_step
                        v_global_write = v_global_base + v_write
                        if v_global_write < v.shape[3]:
                            if state_layout_is_kv:
                                h0_source[(pool_idx, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                            else:
                                h0_source[(pool_idx, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]

                cute.arch.barrier()

    @cute.kernel
    def kda_verify_kernel_large_batch(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        intermediate_states_buffer: cute.Tensor,
        is_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        max_steps: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],
        disable_state_update: cutlass.Constexpr[bool],
    ):
        """Large-batch dense KDA verify kernel. 256 threads, one CTA per v_tile.

        Key difference from small_batch:
          - 256 threads (NUM_THREADS_LARGE) instead of 128
          - One CTA per (batch, value_head, v_tile), no HV iteration loop
          - Uses TILE_V / TILE_V_PADDED instead of TILE_V_SMALL / TILE_V_SMALL_PADDED
          - Single stage (no multi-stage buffering) since one CTA = one v_tile
          - All warps participate in L2 norm reduction (two-level: warp → smem → warp0)
          - sGK is not stored separately; sG * sK is fused into the delta-rule loop
        """
        del tiled_copy_load
        # =========================================================================
        # Thread and block ID mapping
        # Grid: (N*HV*num_v_tiles,) — one CTA per (batch, value_head, v_tile)
        # =========================================================================
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        batch_idx, _, _ = cute.arch.block_idx()

        i_nhv = batch_idx // num_v_tiles
        v_tile = batch_idx % num_v_tiles
        i_n = i_nhv // HV                     # batch index
        i_hv = i_nhv % HV                     # value head index
        i_h = i_hv // (HV // H)              # GQA: which QK head this value head belongs to

        pool_idx = h0_indices[i_n]
        is_idx = is_indices[i_n]

        if pool_idx >= 0:
            # Thread-to-element mapping (uses V_PER_WARP = TILE_V/NUM_WARPS_LARGE)
            k_local = in_warp_tid // V_PER_WARP
            v_local = in_warp_tid % V_PER_WARP
            v_base = warp_idx * V_PER_WARP
            v_idx = v_base + v_local

            # =========================================================================
            # SMEM allocation
            # Note: no sGK buffer — large_batch fuses sG[k] * sK[k] inline
            # =========================================================================
            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((TILE_K,), stride=(1,)), 128)
            sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((TILE_K,), stride=(1,)), 128)
            sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((TILE_K,), stride=(1,)), 128)

            stage = 0  # single stage for large_batch

            # Load initial state into sData
            # Uses flat_idx = tidx + k_iter * NUM_THREADS_LARGE for cooperative loading
            for k_iter in range(NUM_K_ITERS):
                flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                k_load = 0
                v_load = 0
                if state_layout_is_kv:
                    k_load = flat_idx // TILE_V
                    v_load = flat_idx % TILE_V
                else:
                    k_load = flat_idx % TILE_K
                    v_load = flat_idx // TILE_K
                v_global_load = v_tile * TILE_V + v_load
                h_val = 0.0
                if v_global_load < v.shape[3]:
                    if state_layout_is_kv:
                        h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, k_load, v_global_load)])
                    else:
                        h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, v_global_load, k_load)])
                sData[(k_load, v_load, stage)] = h_val
            cute.arch.barrier()

            # =========================================================================
            # Main token loop (same recurrence as small_batch, different thread mapping)
            # =========================================================================
            for t in range(max_steps):

                # Reload q[t], k[t] into SMEM
                if tidx < TILE_K:
                    sK[tidx] = cutlass.Float32(k[i_n, t, i_h, tidx])
                    sQ[tidx] = cutlass.Float32(q[i_n, t, i_h, tidx])

                # Step 1: Gate — compute per-dimension decay factor (same as small_batch)
                r_exp_A = 0.0
                if in_warp_tid == 0:
                    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]))
                r_exp_A = cute.arch.shuffle_sync(r_exp_A, 0)
                if tidx < TILE_K:
                    r_a_k = cutlass.Float32(a[i_n, t, i_hv, tidx])
                    r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                    x = r_a_k + r_dt_bias_k
                    beta_x = softplus_beta * x
                    softplus_x = 0.0
                    if beta_x <= softplus_threshold:
                        exp_beta_x = cute.exp(beta_x)
                        log_input = cutlass.Float32(1.0 + exp_beta_x)
                        log_result = cutlass.Float32(cute.log(log_input))
                        softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
                    else:
                        softplus_x = x
                    sG[tidx] = cute.exp(-r_exp_A * softplus_x)

                r_beta = 0.0
                if in_warp_tid == 0:
                    r_b = cutlass.Float32(b[i_n, t, i_hv])
                    r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                r_beta = cute.arch.shuffle_sync(r_beta, 0)

                # L2 normalization — two-level reduction for large_batch (8 warps)
                # Level 1: each warp reduces its partial sum via butterfly shuffle
                # Level 2: warp 0 reads all warp sums from smem_o, reduces again,
                #          and writes final inv_norm back to smem_o
                cute.arch.barrier()

                if use_qk_l2norm:
                    sum_q_partial = 0.0
                    sum_k_partial = 0.0
                    if tidx < TILE_K:
                        q_val = sQ[tidx]
                        k_val = sK[tidx]
                        sum_q_partial = q_val * q_val
                        sum_k_partial = k_val * k_val
                    for offset in [16, 8, 4, 2, 1]:
                        sum_q_partial += cute.arch.shuffle_sync_bfly(sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31)
                        sum_k_partial += cute.arch.shuffle_sync_bfly(sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31)
                    if in_warp_tid == 0:
                        smem_o[warp_idx] = sum_q_partial
                        smem_o[warp_idx + 8] = sum_k_partial
                    cute.arch.barrier()
                    if warp_idx == 0:
                        local_sum_q = 0.0
                        local_sum_k = 0.0
                        if in_warp_tid < NUM_WARPS_LARGE:
                            local_sum_q = smem_o[in_warp_tid]
                            local_sum_k = smem_o[in_warp_tid + 8]
                        for offset in [4, 2, 1]:
                            local_sum_q += cute.arch.shuffle_sync_bfly(local_sum_q, offset=offset, mask=-1, mask_and_clamp=31)
                            local_sum_k += cute.arch.shuffle_sync_bfly(local_sum_k, offset=offset, mask=-1, mask_and_clamp=31)
                        if in_warp_tid == 0:
                            smem_o[0] = cute.rsqrt(local_sum_q + 1e-6)
                            smem_o[1] = cute.rsqrt(local_sum_k + 1e-6)
                    cute.arch.barrier()
                    inv_norm_q = smem_o[0]
                    inv_norm_k = smem_o[1]
                    if tidx < TILE_K:
                        sK[tidx] = sK[tidx] * inv_norm_k
                        sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                    cute.arch.barrier()
                else:
                    if tidx < TILE_K:
                        sQ[tidx] = sQ[tidx] * scale
                    cute.arch.barrier()

                # Delta-rule update on sData (same logic as small_batch, but large_batch
                # fuses sG * sK inline instead of using a separate sGK buffer)
                v_global = v_tile * TILE_V + v_idx
                r_v = 0.0
                if v_global < v.shape[3]:
                    r_v = cutlass.Float32(v[i_n, t, i_hv, v_global])

                # (a) kᵀ · S' = Σ_k S[k,v] · exp(g[k]) · k[k]  (fused: sG[k] * sK[k])
                sum_hk = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    sum_hk += sData[(k_idx, v_idx, stage)] * sG[k_idx] * sK[k_idx]
                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(sum_hk, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31)

                # (b) Gate the delta: v_new = β · (v - kᵀ · S')
                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                # (c) Apply decay + rank-1 update + accumulate output
                # S_new[k,v] = exp(g[k]) · S[k,v] + k[k] · v_new[v]
                # o[v] = Σ_k S_new[k,v] · q[k]
                sum_hq = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]   # S'[k,v] = exp(g[k]) · S[k,v]
                    h_new = h_old + sK[k_idx] * v_new                  # S_new[k,v] = S'[k,v] + k[k] · v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * sQ[k_idx]                        # o[v] = Σ_k S_new[k,v] · q[k]
                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(sum_hq, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31)

                if k_local == 0 and v_global < v.shape[3]:
                    o[(i_n, t, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

                cute.arch.barrier()

                # Snapshot intermediate state (same as small_batch)
                if cache_intermediate_states:
                    for k_iter in cutlass.range(NUM_K_ITERS, unroll=2):
                        flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                        k_write = 0
                        v_write = 0
                        if state_layout_is_kv:
                            k_write = flat_idx // TILE_V
                            v_write = flat_idx % TILE_V
                        else:
                            k_write = flat_idx % TILE_K
                            v_write = flat_idx // TILE_K
                        v_global_write = v_tile * TILE_V + v_write
                        if v_global_write < v.shape[3]:
                            if state_layout_is_kv:
                                intermediate_states_buffer[(is_idx, t, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                            else:
                                intermediate_states_buffer[(is_idx, t, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]

            # Write final state back to pool (decode mode)
            if not disable_state_update:
                cute.arch.barrier()
                for k_iter in cutlass.range(NUM_K_ITERS, unroll=2):
                    flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                    k_write = 0
                    v_write = 0
                    if state_layout_is_kv:
                        k_write = flat_idx // TILE_V
                        v_write = flat_idx % TILE_V
                    else:
                        k_write = flat_idx % TILE_K
                        v_write = flat_idx // TILE_K
                    v_global_write = v_tile * TILE_V + v_write
                    if v_global_write < v.shape[3]:
                        if state_layout_is_kv:
                            h0_source[(pool_idx, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                        else:
                            h0_source[(pool_idx, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]

    @cute.kernel
    def kda_verify_kernel_large_batch_varlen(
        tiled_copy_load: cute.TiledCopy,
        h0_source: cute.Tensor,
        smem_layout_staged: cute.Layout,
        num_v_tiles: cutlass.Constexpr[int],
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        o: cute.Tensor,
        h0_indices: cute.Tensor,
        intermediate_states_buffer: cute.Tensor,
        is_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        max_steps: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],
        disable_state_update: cutlass.Constexpr[bool],
    ):
        """Large-batch varlen KDA verify kernel. q/k: (1, N*T, H, K), a: (N*T, HV, K).

        Same logic as kda_verify_kernel_large_batch but for varlen layout.
        Uses flattened seq_idx = i_n * max_steps + t for GMEM addressing.
        """
        del tiled_copy_load
        # Thread and block ID mapping (same as large_batch)
        tidx, _, _ = cute.arch.thread_idx()
        in_warp_tid = tidx % 32
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        batch_idx, _, _ = cute.arch.block_idx()

        i_nhv = batch_idx // num_v_tiles
        v_tile = batch_idx % num_v_tiles
        i_n = i_nhv // HV
        i_hv = i_nhv % HV
        i_h = i_hv // (HV // H)

        pool_idx = h0_indices[i_n]
        is_idx = is_indices[i_n]

        if pool_idx >= 0:
            k_local = in_warp_tid // V_PER_WARP
            v_local = in_warp_tid % V_PER_WARP
            v_base = warp_idx * V_PER_WARP
            v_idx = v_base + v_local

            smem = cutlass.utils.SmemAllocator()
            sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
            smem_o_layout = cute.make_layout((TILE_V,), stride=(1,))
            smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
            sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((TILE_K,), stride=(1,)), 128)
            sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((TILE_K,), stride=(1,)), 128)
            sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((TILE_K,), stride=(1,)), 128)

            stage = 0

            for k_iter in range(NUM_K_ITERS):
                flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                k_load = 0
                v_load = 0
                if state_layout_is_kv:
                    k_load = flat_idx // TILE_V
                    v_load = flat_idx % TILE_V
                else:
                    k_load = flat_idx % TILE_K
                    v_load = flat_idx // TILE_K
                v_global_load = v_tile * TILE_V + v_load
                h_val = 0.0
                if v_global_load < v.shape[3]:
                    if state_layout_is_kv:
                        h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, k_load, v_global_load)])
                    else:
                        h_val = cutlass.Float32(h0_source[(pool_idx, i_hv, v_global_load, k_load)])
                sData[(k_load, v_load, stage)] = h_val
            cute.arch.barrier()

            # Main token loop (same recurrence as large_batch, varlen indexing)
            for t in range(max_steps):
                seq_idx = i_n * max_steps + t  # flattened varlen index

                # Reload q[t], k[t] into SMEM (varlen: dim0=1, dim1=seq_idx)
                if tidx < TILE_K:
                    sK[tidx] = cutlass.Float32(k[0, seq_idx, i_h, tidx])
                    sQ[tidx] = cutlass.Float32(q[0, seq_idx, i_h, tidx])

                # Step 1: Gate (same as large_batch — see small_batch for detailed comments)
                r_exp_A = 0.0
                if in_warp_tid == 0:
                    r_exp_A = cute.exp(cutlass.Float32(A_log[i_hv]))
                r_exp_A = cute.arch.shuffle_sync(r_exp_A, 0)
                if tidx < TILE_K:
                    r_a_k = cutlass.Float32(a[seq_idx, i_hv, tidx])
                    r_dt_bias_k = cutlass.Float32(dt_bias[i_hv, tidx])
                    x = r_a_k + r_dt_bias_k
                    beta_x = softplus_beta * x
                    softplus_x = 0.0
                    if beta_x <= softplus_threshold:
                        exp_beta_x = cute.exp(beta_x)
                        log_input = cutlass.Float32(1.0 + exp_beta_x)
                        log_result = cutlass.Float32(cute.log(log_input))
                        softplus_x = cutlass.Float32((cutlass.Float32(1.0) / softplus_beta) * log_result)
                    else:
                        softplus_x = x
                    sG[tidx] = cute.exp(-r_exp_A * softplus_x)

                r_beta = 0.0
                if in_warp_tid == 0:
                    r_b = cutlass.Float32(b[seq_idx, i_hv])
                    r_beta = 1.0 / (1.0 + cute.exp(-r_b))
                r_beta = cute.arch.shuffle_sync(r_beta, 0)

                # L2 normalization — two-level reduction for large_batch (same as dense large_batch)
                cute.arch.barrier()

                if use_qk_l2norm:
                    sum_q_partial = 0.0
                    sum_k_partial = 0.0
                    if tidx < TILE_K:
                        q_val = sQ[tidx]
                        k_val = sK[tidx]
                        sum_q_partial = q_val * q_val
                        sum_k_partial = k_val * k_val
                    for offset in [16, 8, 4, 2, 1]:
                        sum_q_partial += cute.arch.shuffle_sync_bfly(sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31)
                        sum_k_partial += cute.arch.shuffle_sync_bfly(sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31)
                    if in_warp_tid == 0:
                        smem_o[warp_idx] = sum_q_partial
                        smem_o[warp_idx + 8] = sum_k_partial
                    cute.arch.barrier()
                    if warp_idx == 0:
                        local_sum_q = 0.0
                        local_sum_k = 0.0
                        if in_warp_tid < NUM_WARPS_LARGE:
                            local_sum_q = smem_o[in_warp_tid]
                            local_sum_k = smem_o[in_warp_tid + 8]
                        for offset in [4, 2, 1]:
                            local_sum_q += cute.arch.shuffle_sync_bfly(local_sum_q, offset=offset, mask=-1, mask_and_clamp=31)
                            local_sum_k += cute.arch.shuffle_sync_bfly(local_sum_k, offset=offset, mask=-1, mask_and_clamp=31)
                        if in_warp_tid == 0:
                            smem_o[0] = cute.rsqrt(local_sum_q + 1e-6)
                            smem_o[1] = cute.rsqrt(local_sum_k + 1e-6)
                    cute.arch.barrier()
                    inv_norm_q = smem_o[0]
                    inv_norm_k = smem_o[1]
                    if tidx < TILE_K:
                        sK[tidx] = sK[tidx] * inv_norm_k
                        sQ[tidx] = sQ[tidx] * scale * inv_norm_q
                    cute.arch.barrier()
                else:
                    if tidx < TILE_K:
                        sQ[tidx] = sQ[tidx] * scale
                    cute.arch.barrier()

                # Delta-rule update on sData (same as large_batch — see small_batch for detailed comments)
                v_global = v_tile * TILE_V + v_idx
                r_v = 0.0
                if v_global < v.shape[3]:
                    r_v = cutlass.Float32(v[0, seq_idx, i_hv, v_global])

                # (a) kᵀ · S' = Σ_k S[k,v] · exp(g[k]) · k[k]
                sum_hk = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    sum_hk += sData[(k_idx, v_idx, stage)] * sG[k_idx] * sK[k_idx]
                for offset in [4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(sum_hk, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31)

                # (b) Gate the delta: v_new = β · (v - kᵀ · S')
                v_new = (r_v - sum_hk) * r_beta
                v_new = cute.arch.shuffle_sync(v_new, v_local)

                # (c) Apply decay + rank-1 update + accumulate output
                sum_hq = 0.0
                for k_iter in range(NUM_K_ITERS):
                    k_base = k_iter * ROWS_PER_ITER
                    k_idx = k_base + k_local
                    h_old = sData[(k_idx, v_idx, stage)] * sG[k_idx]
                    h_new = h_old + sK[k_idx] * v_new
                    sData[(k_idx, v_idx, stage)] = h_new
                    sum_hq += h_new * sQ[k_idx]
                for offset in [4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(sum_hq, offset=offset * V_PER_WARP, mask=-1, mask_and_clamp=31)

                if k_local == 0 and v_global < v.shape[3]:
                    o[(0, seq_idx, i_hv, v_global)] = cutlass.BFloat16(sum_hq)

                cute.arch.barrier()

                # Snapshot intermediate state (verify-only)
                if cache_intermediate_states:
                    for k_iter in cutlass.range(NUM_K_ITERS, unroll=2):
                        flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                        k_write = 0
                        v_write = 0
                        if state_layout_is_kv:
                            k_write = flat_idx // TILE_V
                            v_write = flat_idx % TILE_V
                        else:
                            k_write = flat_idx % TILE_K
                            v_write = flat_idx // TILE_K
                        v_global_write = v_tile * TILE_V + v_write
                        if v_global_write < v.shape[3]:
                            if state_layout_is_kv:
                                intermediate_states_buffer[(is_idx, t, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                            else:
                                intermediate_states_buffer[(is_idx, t, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]

            # Write final state back to pool (decode mode)
            if not disable_state_update:
                cute.arch.barrier()
                for k_iter in cutlass.range(NUM_K_ITERS, unroll=2):
                    flat_idx = tidx + k_iter * NUM_THREADS_LARGE
                    k_write = 0
                    v_write = 0
                    if state_layout_is_kv:
                        k_write = flat_idx // TILE_V
                        v_write = flat_idx % TILE_V
                    else:
                        k_write = flat_idx % TILE_K
                        v_write = flat_idx // TILE_K
                    v_global_write = v_tile * TILE_V + v_write
                    if v_global_write < v.shape[3]:
                        if state_layout_is_kv:
                            h0_source[(pool_idx, i_hv, k_write, v_global_write)] = sData[(k_write, v_write, stage)]
                        else:
                            h0_source[(pool_idx, i_hv, v_global_write, k_write)] = sData[(k_write, v_write, stage)]

    return (
        kda_verify_kernel_small_batch,
        kda_verify_kernel_small_batch_varlen,
        kda_verify_kernel_large_batch,
        kda_verify_kernel_large_batch_varlen,
    )


# ---------------------------------------------------------------------------
# JIT launcher
# ---------------------------------------------------------------------------


def _create_verify_jit_functions():
    """Create JIT-compiled launchers for all KDA verify kernel variants."""

    (kda_verify_small, kda_verify_small_varlen,
     kda_verify_large, kda_verify_large_varlen) = _define_verify_kernels()

    @cute.jit
    def run_verify_small_batch(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        intermediate_states_buffer: cute.Tensor,
        is_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        max_steps: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        dense_small_hv_parallel: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],
        disable_state_update: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        del K
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * (HV if dense_small_hv_parallel else H)

        num_v_tiles_small = cute.ceil_div(V, TILE_V_SMALL)
        smem_layout_small = cute.make_layout(
            (TILE_K, TILE_V_SMALL, NUM_STAGES),
            stride=(TILE_V_SMALL_PADDED, 1, TILE_K * TILE_V_SMALL_PADDED),
        )
        smem_bytes_small = 4 * TILE_K * TILE_V_SMALL_PADDED * NUM_STAGES + 4 * TILE_V_SMALL + 4 * TILE_K * 4 + 64

        kda_verify_small(
            None, h0_source, smem_layout_small,
            num_v_tiles_small, num_blocks_per_state_small,
            q, k, v, a, b, A_log, dt_bias, o, h0_indices,
            intermediate_states_buffer, is_indices,
            softplus_beta, softplus_threshold, scale,
            H, HV, max_steps,
            use_qk_l2norm, state_layout_is_kv,
            dense_small_hv_parallel,
            cache_intermediate_states, disable_state_update,
        ).launch(
            grid=(batch_size * num_blocks_per_state_small, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes_small,
            stream=stream,
        )

    @cute.jit
    def run_verify_small_batch_varlen(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        intermediate_states_buffer: cute.Tensor,
        is_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        max_steps: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        dense_small_hv_parallel: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],
        disable_state_update: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        del K, dense_small_hv_parallel
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * HV

        num_v_tiles_small = cute.ceil_div(V, TILE_V_SMALL)
        smem_layout_small = cute.make_layout(
            (TILE_K, TILE_V_SMALL, NUM_STAGES),
            stride=(TILE_V_SMALL_PADDED, 1, TILE_K * TILE_V_SMALL_PADDED),
        )
        smem_bytes_small = 4 * TILE_K * TILE_V_SMALL_PADDED * NUM_STAGES + 4 * TILE_V_SMALL + 4 * TILE_K * 4 + 64

        kda_verify_small_varlen(
            None, h0_source, smem_layout_small,
            num_v_tiles_small, num_blocks_per_state_small,
            q, k, v, a, b, A_log, dt_bias, o, h0_indices,
            intermediate_states_buffer, is_indices,
            softplus_beta, softplus_threshold, scale,
            H, HV, max_steps,
            use_qk_l2norm, state_layout_is_kv,
            cache_intermediate_states, disable_state_update,
        ).launch(
            grid=(batch_size * num_blocks_per_state_small, 1, 1),
            block=[NUM_THREADS, 1, 1],
            smem=smem_bytes_small,
            stream=stream,
        )

    @cute.jit
    def run_verify_large_batch(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        intermediate_states_buffer: cute.Tensor,
        is_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        max_steps: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        dense_small_hv_parallel: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],
        disable_state_update: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        del K, num_blocks_per_state_small, dense_small_hv_parallel
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * HV

        num_v_tiles = cute.ceil_div(V, TILE_V)
        smem_layout = cute.make_layout(
            (TILE_K, TILE_V, NUM_STAGES),
            stride=(TILE_V_PADDED, 1, TILE_K * TILE_V_PADDED),
        )
        smem_bytes = 4 * TILE_K * TILE_V_PADDED * NUM_STAGES + 4 * TILE_V + 4 * TILE_K * 3 + 64

        kda_verify_large(
            None, h0_source, smem_layout, num_v_tiles,
            q, k, v, a, b, A_log, dt_bias, o, h0_indices,
            intermediate_states_buffer, is_indices,
            softplus_beta, softplus_threshold, scale,
            H, HV, max_steps,
            use_qk_l2norm, state_layout_is_kv,
            cache_intermediate_states, disable_state_update,
        ).launch(
            grid=(batch_size * num_v_tiles, 1, 1),
            block=[NUM_THREADS_LARGE, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    @cute.jit
    def run_verify_large_batch_varlen(
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        h0_source: cute.Tensor,
        h0_indices: cute.Tensor,
        o: cute.Tensor,
        intermediate_states_buffer: cute.Tensor,
        is_indices: cute.Tensor,
        softplus_beta: cutlass.Constexpr[float],
        softplus_threshold: cutlass.Constexpr[float],
        scale: cutlass.Constexpr[float],
        H: cutlass.Constexpr[int],
        HV: cutlass.Constexpr[int],
        K: cutlass.Constexpr[int],
        V: cutlass.Constexpr[int],
        max_steps: cutlass.Constexpr[int],
        use_qk_l2norm: cutlass.Constexpr[bool],
        state_layout_is_kv: cutlass.Constexpr[bool],
        num_blocks_per_state_small: cutlass.Constexpr[int],
        dense_small_hv_parallel: cutlass.Constexpr[bool],
        cache_intermediate_states: cutlass.Constexpr[bool],
        disable_state_update: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
    ):
        del K, num_blocks_per_state_small, dense_small_hv_parallel
        n_indices = h0_indices.layout.shape[0]
        batch_size = n_indices * HV

        num_v_tiles = cute.ceil_div(V, TILE_V)
        smem_layout = cute.make_layout(
            (TILE_K, TILE_V, NUM_STAGES),
            stride=(TILE_V_PADDED, 1, TILE_K * TILE_V_PADDED),
        )
        smem_bytes = 4 * TILE_K * TILE_V_PADDED * NUM_STAGES + 4 * TILE_V + 4 * TILE_K * 3 + 64

        kda_verify_large_varlen(
            None, h0_source, smem_layout, num_v_tiles,
            q, k, v, a, b, A_log, dt_bias, o, h0_indices,
            intermediate_states_buffer, is_indices,
            softplus_beta, softplus_threshold, scale,
            H, HV, max_steps,
            use_qk_l2norm, state_layout_is_kv,
            cache_intermediate_states, disable_state_update,
        ).launch(
            grid=(batch_size * num_v_tiles, 1, 1),
            block=[NUM_THREADS_LARGE, 1, 1],
            smem=smem_bytes,
            stream=stream,
        )

    return (
        run_verify_small_batch,
        run_verify_small_batch_varlen,
        run_verify_large_batch,
        run_verify_large_batch_varlen,
    )


def _get_verify_jit_functions():
    global _verify_jit_functions
    if _verify_jit_functions is None:
        _verify_jit_functions = _create_verify_jit_functions()
    return _verify_jit_functions


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------


def _get_compiled_verify_kernel(
    N,
    H,
    HV,
    K,
    V,
    pool_size,
    use_small_batch,
    is_varlen,
    max_steps,
    scale,
    use_qk_l2norm,
    state_layout_is_kv,
    num_blocks_per_state_small,
    dense_small_hv_parallel,
    softplus_beta,
    softplus_threshold,
    cache_intermediate_states,
    disable_state_update,
):
    global _compiled_verify_kernels

    key = (
        N, H, HV, K, V, pool_size,
        use_small_batch, is_varlen, max_steps,
        scale, use_qk_l2norm, state_layout_is_kv,
        num_blocks_per_state_small, dense_small_hv_parallel,
        softplus_beta, softplus_threshold,
        cache_intermediate_states, disable_state_update,
    )
    if key in _compiled_verify_kernels:
        return _compiled_verify_kernels[key]

    total_tokens = N * max_steps
    if is_varlen:
        q = torch.zeros(1, total_tokens, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(1, total_tokens, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros(1, total_tokens, HV, V, dtype=torch.bfloat16, device="cuda")
        a = torch.zeros(total_tokens, HV, K, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(total_tokens, HV, dtype=torch.bfloat16, device="cuda")
        o = torch.zeros(1, total_tokens, HV, V, dtype=torch.bfloat16, device="cuda")
    else:
        q = torch.zeros(N, max_steps, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(N, max_steps, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros(N, max_steps, HV, V, dtype=torch.bfloat16, device="cuda")
        a = torch.zeros(N, max_steps, HV, K, dtype=torch.bfloat16, device="cuda")
        b = torch.zeros(N, max_steps, HV, dtype=torch.bfloat16, device="cuda")
        o = torch.zeros(N, max_steps, HV, V, dtype=torch.bfloat16, device="cuda")

    A_log = torch.zeros(HV, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(HV, K, dtype=torch.float32, device="cuda")
    if state_layout_is_kv:
        h0_source = torch.zeros(pool_size, HV, K, V, dtype=torch.float32, device="cuda")
        is_buf = torch.zeros(N, max_steps, HV, K, V, dtype=torch.float32, device="cuda")
    else:
        h0_source = torch.zeros(pool_size, HV, V, K, dtype=torch.float32, device="cuda")
        is_buf = torch.zeros(N, max_steps, HV, V, K, dtype=torch.float32, device="cuda")
    h0_indices = torch.zeros(N, dtype=torch.int32, device="cuda")
    is_indices = torch.zeros(N, dtype=torch.int32, device="cuda")

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    run_small, run_small_varlen, run_large, run_large_varlen = _get_verify_jit_functions()
    if use_small_batch:
        kernel_func = run_small_varlen if is_varlen else run_small
    else:
        kernel_func = run_large_varlen if is_varlen else run_large

    compiled_kernel = cute.compile(
        kernel_func,
        from_dlpack(q, assumed_align=16),
        from_dlpack(k, assumed_align=16),
        from_dlpack(v, assumed_align=16),
        from_dlpack(a, assumed_align=16),
        from_dlpack(b, assumed_align=16),
        from_dlpack(A_log, assumed_align=16),
        from_dlpack(dt_bias, assumed_align=16),
        from_dlpack(h0_source, assumed_align=16),
        from_dlpack(h0_indices, assumed_align=16),
        from_dlpack(o, assumed_align=16),
        from_dlpack(is_buf, assumed_align=16),
        from_dlpack(is_indices, assumed_align=16),
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        scale=scale,
        H=H,
        HV=HV,
        K=K,
        V=V,
        max_steps=max_steps,
        use_qk_l2norm=use_qk_l2norm,
        state_layout_is_kv=state_layout_is_kv,
        num_blocks_per_state_small=num_blocks_per_state_small,
        dense_small_hv_parallel=dense_small_hv_parallel,
        cache_intermediate_states=cache_intermediate_states,
        disable_state_update=disable_state_update,
        stream=stream,
        options="--enable-tvm-ffi --opt-level 1",
    )

    _compiled_verify_kernels[key] = compiled_kernel
    logger.info(
        "CuTe DSL KDA verify kernel compiled: "
        f"N={N}, H={H}, HV={HV}, K={K}, V={V}, max_steps={max_steps}, "
        f"pool_size={pool_size}, small_batch={use_small_batch}, varlen={is_varlen}"
    )
    return compiled_kernel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def kda_verify(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    *,
    cache_intermediate_states: bool = True,
    disable_state_update: bool = True,
    recompute_state: bool = False,
    saved_qkvab: dict | None = None,
    retrieve_parent_token: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    state_layout: str = "vk",
) -> torch.Tensor:
    """CuTe DSL KDA verify: fused multi-token decode for speculative decoding.

    Dense inputs:
        q/k: (N, T, H, K)   v: (N, T, HV, V)   a: (N, T, HV, K)   b: (N, T, HV)

    Varlen inputs:
        q/k: (1, N*T, H, K)   v: (1, N*T, HV, V)   a: (N*T, HV, K)   b: (N*T, HV)

    Returns output: (N, T, HV, V) bfloat16 (dense) or (1, N*T, HV, V) bfloat16 (varlen).
    """
    assert not recompute_state, "recompute_state not implemented (Phase 3)"
    assert saved_qkvab is None, "saved_qkvab not implemented (Phase 3)"
    assert retrieve_parent_token is None, "tree speculation not implemented"

    B_q, T_q, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    if initial_state_indices is not None:
        N = initial_state_indices.shape[0]
    else:
        N = T_q if B_q == 1 and T_q > 1 else B_q

    is_varlen = B_q == 1 and N > 1 and T_q == N * (T_q // N)
    if is_varlen:
        T = T_q // N
    else:
        N = B_q
        T = T_q

    if scale is None:
        scale = K**-0.5

    state_layout = _canonicalize_state_layout(state_layout)
    state_layout_is_kv = state_layout == "kv"

    assert K == TILE_K, f"CuTe verify requires K={TILE_K}, got {K}"
    assert V % TILE_V_SMALL == 0, f"CuTe verify requires V % {TILE_V_SMALL} == 0, got V={V}"

    use_small_batch = N < SMALL_BATCH_THRESHOLD
    if not use_small_batch:
        assert V % TILE_V == 0, f"CuTe verify large batch requires V % {TILE_V} == 0, got V={V}"

    A_log = _normalize_A_log(A_log, HV)
    dt_bias = _normalize_dt_bias(dt_bias, HV, K)

    pool_size = initial_state_source.shape[0]
    if state_layout_is_kv:
        assert initial_state_source.shape == (pool_size, HV, K, V), (
            f"State shape mismatch: {initial_state_source.shape}"
        )
    else:
        assert initial_state_source.shape == (pool_size, HV, V, K), (
            f"State shape mismatch: {initial_state_source.shape}"
        )

    if initial_state_indices is None:
        initial_state_indices = torch.arange(N, device=q.device, dtype=torch.int32)
    if intermediate_state_indices is None:
        intermediate_state_indices = torch.arange(N, device=q.device, dtype=torch.int32)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if is_varlen:
        if a.dim() == 4:
            a = a.reshape(T_q, HV, K)
        if b.dim() == 3 and b.shape[0] == 1:
            b = b.reshape(T_q, HV)
        o = torch.empty(1, T_q, HV, V, dtype=torch.bfloat16, device=q.device)
    else:
        o = torch.empty(N, T, HV, V, dtype=torch.bfloat16, device=q.device)

    a = a.contiguous()
    b = b.contiguous()

    num_blocks_per_state_small = _select_small_blocks_per_state(N, H, HV, V)
    dense_small_hv_parallel_head_threshold = (
        N4_DENSE_SMALL_HV_PARALLEL_HEAD_THRESHOLD if N <= 4 else DENSE_SMALL_HV_PARALLEL_HEAD_THRESHOLD
    )
    dense_small_hv_parallel = (
        use_small_batch
        and (not is_varlen)
        and dense_small_hv_parallel_head_threshold >= H
        and N <= DENSE_SMALL_HV_PARALLEL_MAX_N
    )

    stream = _get_cached_stream(q.device)

    compiled_kernel = _get_compiled_verify_kernel(
        N, H, HV, K, V, pool_size,
        use_small_batch=use_small_batch,
        is_varlen=is_varlen,
        max_steps=T,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        state_layout_is_kv=state_layout_is_kv,
        num_blocks_per_state_small=num_blocks_per_state_small,
        dense_small_hv_parallel=dense_small_hv_parallel,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        cache_intermediate_states=cache_intermediate_states,
        disable_state_update=disable_state_update,
    )

    compiled_kernel(
        q, k, v, a, b,
        A_log, dt_bias,
        initial_state_source,
        initial_state_indices,
        o,
        intermediate_states_buffer,
        intermediate_state_indices,
        stream,
    )

    return o
