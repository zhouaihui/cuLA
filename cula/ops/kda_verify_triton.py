# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Triton KDA verify kernel for MTP (Multi-Token Prediction) / Speculative Decoding.
Derived from cula/ops/kda_decode_fla.py's fused_sigmoid_gating_delta_rule_update_kernel.

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
    S ∈ R^{K×V}  — per (batch, value_head) hidden state, stored in KV layout inside kernel

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
        o = S_new^T · q  ∈ R^V

    Code implements Step 2 in three in-place sub-steps on registers:
        b_v -= kᵀ · S'        # delta: v_delta = v - kᵀ · S'
        b_v *= β               # gate:  v_delta = β · (v - kᵀ · S')
        S   += k · b_vᵀ       # update: S_new = S' + k ⊗ v_delta

=============================================================================
Verify vs Decode
=============================================================================
    Decode kernel:  processes 1 token, writes final state back to state pool.
    Verify kernel:  processes T draft tokens sequentially, writes intermediate
                    state snapshot after each token to a buffer, does NOT modify
                    the state pool (DISABLE_STATE_UPDATE=True by default).

    After rejection sampling, the framework picks intermediate_states[n, accepted_len]
    as the correct state for sequence n.

=============================================================================
Tiling Strategy
=============================================================================
    State S ∈ R^{K×V} is tiled as [BK, BV] register tiles.
        BK = next_power_of_2(K) = 128  (one tile covers full K dimension, NK=1)
        BV = min(next_power_of_2(V), 32) = 32  (V=128 split into NV=4 tiles)

    Grid: (NK=1, NV=4, N*HV)
        - Dim 2: each program handles one (batch, value_head) pair
        - Dim 1: each program handles one V-tile (32 elements of V dimension)
        - Dim 0: always 1 (full K dimension in one tile)

    So 4 programs cooperate to compute the full (K=128, V=128) state for each
    (batch, value_head) pair, each handling a [128, 32] slice.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def kda_verify_triton_kernel(
    # --- Model parameters ---
    A_log,          # (HV,)       fp32  decay log parameter, one scalar per value head
    a,              # (N*T, HV*K) bf16  KDA vector gate input (flattened as [N*T, HV, K])
    dt_bias,        # (HV, K)     fp32  gate bias, added to `a` before softplus
    softplus_beta,  # float             softplus β parameter
    softplus_threshold,  # float        softplus threshold for numerical stability
    # --- Token inputs (all have T tokens per batch element, flattened as N*T) ---
    q,              # (N*T, H, K)  bf16  queries
    k,              # (N*T, H, K)  bf16  keys
    v,              # (N*T, HV, V) bf16  values
    b,              # (N*T, HV)    bf16  sigmoid beta gate input
    # --- Output ---
    o,              # (NK, N*T, HV, V)   output logits, NK=1 so squeezed later
    # --- State pool ---
    h0_source,      # (pool_size, HV, K, V) fp32  state pool (always KV layout in kernel)
    h0_indices,     # (N,)         int32  maps batch element → state pool index
    # --- Intermediate state buffer (verify-specific) ---
    intermediate_states_buffer,  # (N, T, HV, K, V) fp32  state snapshot per token step
    # --- Scalars ---
    scale,          # float  attention scale, typically K^{-0.5}
    T,              # int    number of draft tokens (spec_len)
    # --- Compile-time constants ---
    B: tl.constexpr,    # batch size N
    H: tl.constexpr,    # number of QK heads
    HV: tl.constexpr,   # number of value heads
    K: tl.constexpr,     # key/query dimension
    V: tl.constexpr,     # value dimension
    BK: tl.constexpr,    # K-tile size (= next_power_of_2(K), typically 128)
    BV: tl.constexpr,    # V-tile size (= min(next_power_of_2(V), 32), typically 32)
    USE_INITIAL_STATE: tl.constexpr,          # load initial state from h0_source
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,    # L2-normalize q and k
    CACHE_INTERMEDIATE_STATES: tl.constexpr,  # save state snapshot after each token
    DISABLE_STATE_UPDATE: tl.constexpr,       # skip final state writeback to pool
):
    # =========================================================================
    # Program ID → (batch, value_head, K-tile, V-tile) mapping
    # Grid: (NK, NV, N * HV)
    # =========================================================================
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = i_nh // HV     # batch index ∈ [0, N)
    i_hv = i_nh % HV     # value head index ∈ [0, HV)
    i_h = i_hv // (HV // H)  # GQA: which QK head this value head belongs to

    # Token range for this batch element in the flattened N*T dimension
    bos = i_n * T   # beginning-of-sequence offset
    all = B * T     # total tokens across all batch elements

    # =========================================================================
    # Tile offsets within K and V dimensions
    # o_k: [0, 1, ..., BK-1] — which K elements this program handles
    # o_v: [i_v*BV, ..., i_v*BV+BV-1] — which V elements this program handles
    # =========================================================================
    o_k = i_k * BK + tl.arange(0, BK)  # (BK,) e.g. [0..127]
    o_v = i_v * BV + tl.arange(0, BV)  # (BV,) e.g. [0..31], [32..63], ...

    # =========================================================================
    # Pointer initialization — all tensors are contiguous in memory
    #
    # q/k layout: [N*T, H, K] contiguous
    #   stride: token = H*K, head = K, element = 1
    #   p_q → q[bos, i_h, o_k]
    #
    # v layout: [N*T, HV, V] contiguous
    #   stride: token = HV*V, head = V, element = 1
    #
    # b layout: [N*T, HV] contiguous
    #   stride: token = HV, element = 1
    #
    # o layout: [NK, N*T, HV, V]
    #   stride: nk = all*HV*V, token = HV*V, head = V, element = 1
    #
    # a layout: [N*T, HV, K] contiguous (KDA vector gate)
    #   stride: token = HV*K, head = K, element = 1
    # =========================================================================
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    p_A_log = A_log + i_hv                    # scalar per value head
    p_a = a + bos * HV * K + i_hv * K + o_k   # (BK,) vector gate for this head
    p_dt_bias = dt_bias + i_hv * K + o_k       # (BK,) gate bias for this head

    # =========================================================================
    # Boundary masks — handle cases where BK > K or BV > V (power-of-2 padding)
    # =========================================================================
    mask_k = o_k < K                            # (BK,)
    mask_v = o_v < V                            # (BV,)
    mask_h = mask_k[:, None] & mask_v[None, :]  # (BK, BV)

    # =========================================================================
    # Initialize state tile: b_h ∈ R^{BK × BV} (fp32)
    #
    # This is the core register tile representing a [K-tile, V-tile] slice of
    # the state matrix S ∈ R^{K×V}. Four programs (i_v=0..3) each hold a
    # [128, 32] slice, together covering the full [128, 128] state.
    # =========================================================================
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)  # indirect index into state pool
        if idx >= 0:  # idx < 0 → no initial state, use zero-initialized b_h
            # h0_source layout: row-major [pool_size, HV, K, V]
            #
            # We need a (BK, BV) pointer grid p_h0[i,j] → offset = base + o_k[i]*V + o_v[j].
            # The `[:, None]` / `[None, :]` broadcast produces exactly this:
            #   o_k[:, None] * V  has shape (BK, 1) — row offsets replicated across columns
            #   o_v[None, :]      has shape (1, BV) — column offsets replicated across rows
            #   Adding them broadcasts both to (BK, BV), each element = row_off + col_off.
            p_h0 = (
                h0_source
                + idx * HV * K * V    # skip to pool slot `idx`
                + i_hv * K * V        # skip to value head `i_hv`
                + o_k[:, None] * V    # (BK,1) row stride — each K-row is V elements apart
                + o_v[None, :]        # (1,BV) col offset — column index within a row
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # =========================================================================
    # Main loop: process T draft tokens sequentially
    #
    # Each iteration performs the full KDA recurrence for one token:
    #   1. Compute decay gate:  g = -exp(A_log) · softplus(a + dt_bias)
    #   2. Decay state:         S' = diag(exp(g)) · S
    #   3. Delta rule update:   S_new = S' + β·k·vᵀ - β·k·(kᵀ·S')
    #   4. Compute output:      o = S_newᵀ · q
    #   5. (Verify only) Cache intermediate state snapshot
    # =========================================================================
    for i_t in range(0, T):
        # --- Load token inputs (bf16 → fp32) ---
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)  # (BK,) query
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)  # (BK,) key
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)  # (BV,) value
        b_b = tl.load(p_b).to(tl.float32)                         # scalar, sigmoid gate input

        # --- Load gate parameters ---
        b_A_log = tl.load(p_A_log).to(tl.float32)      # scalar, decay log parameter
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)      # (BK,) vector gate input
        b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)  # (BK,) gate bias

        # =================================================================
        # Step 1: Gate — compute per-dimension decay factor g ∈ R^K
        #   g = -exp(A_log) · softplus(a + dt_bias)
        #
        # A_log < 0 (initialized as -rand*2) → exp(A_log) ∈ (0, 1)
        # softplus(x) > 0 for all x → g < 0 → exp(g) ∈ (0, 1) → decay.
        #
        # KDA's key difference from standard linear attention: g is a K-dim vector
        # (not a scalar), so each row of the state matrix decays independently.
        # =================================================================
        x = b_a + b_dt_bias
        # Parameterized softplus: softplus_β(x) = (1/β) · log(1 + e^(βx))
        # β controls sharpness near zero: β=1 → standard softplus; β→∞ → ReLU.
        # softplus_beta is a hyperparameter tunable at the model level; the kernel
        # must adapt to whatever value the caller passes, not assume β=1.
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,  # large βx: (1/β)·log(1+e^(βx)) ≈ x, avoids exp overflow
        )
        b_g = -tl.exp(b_A_log) * softplus_x  # (BK,), < 0

        # --- β = sigmoid(b), controls delta update strength ---
        # Unlike softplus_beta (a function-shape hyperparameter), this β is the
        # *output* of sigmoid on a learned scalar b_b — the network learns b_b
        # directly, so no extra parameterization of sigmoid is needed.
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))  # scalar ∈ (0, 1)

        # --- Optional L2 normalization of q and k ---
        # Keeps q, k on the unit sphere, preventing unbounded state/output growth.
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        # --- Apply attention scale ---
        b_q = b_q * scale  # scale = K^{-0.5}

        # =================================================================
        # Step 2: Decay — S' = diag(exp(g)) · S
        #   b_h[k, v] *= exp(g[k])
        # b_g is (BK,); [:, None] broadcasts to (BK, BV) so each row k is
        # scaled by its own exp(g[k]) — per-dimension row-wise decay.
        # =================================================================
        b_h *= tl.exp(b_g[:, None])

        # =================================================================
        # Step 3: Delta-rule update — S_new = S' + β · k · (v - kᵀ · S')ᵀ
        #
        # Factored into three in-place register operations:
        #   (a) v_delta = v - kᵀ · S'       residual: what the state didn't predict
        #   (b) v_delta *= β                 gate: scale the update strength
        #   (c) S += k ⊗ v_delta            rank-1 outer product update
        # =================================================================

        # (a) kᵀ · S' — dot product along K dim → prediction for each V element
        # b_h: (BK, BV), b_k[:, None]: (BK, 1) → elementwise mul → (BK, BV)
        # tl.sum(..., 0) reduces K dim → (BV,) = Σ_k S'[k, v] · k[k]
        b_v -= tl.sum(b_h * b_k[:, None], 0)   # v_delta = v - kᵀ · S'

        # (b) Gate the delta
        b_v *= b_beta                            # v_delta = β · (v - kᵀ · S')

        # (c) Rank-1 outer product update: S += k ⊗ v_delta
        # b_k[:, None]: (BK, 1), b_v[None, :]: (1, BV) → (BK, BV)
        b_h += b_k[:, None] * b_v[None, :]

        # =================================================================
        # Step 4: Output — o = S_newᵀ · q ∈ R^V
        # Similar to kᵀ · S' in Step 3a but with q instead of k.
        # Sum over K dim: (BK, BV) · (BK, 1) → sum dim 0 → (BV,)
        # =================================================================
        b_o = tl.sum(b_h * b_q[:, None], 0)    # o[v] = Σ_k S[k, v] · q[k]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # =================================================================
        # Step 5 (verify-only): Snapshot intermediate state
        # Save b_h → intermediate_states_buffer[n, t, hv, :, :] so the
        # framework can roll back to the correct state after rejection sampling.
        # Buffer layout: [N, T, HV, K, V] row-major (KV layout in kernel).
        # =================================================================
        if CACHE_INTERMEDIATE_STATES:
            p_is = (
                intermediate_states_buffer
                + (i_n * T + i_t) * HV * K * V  # → [n, t]
                + i_hv * K * V                    # → [hv]
                + o_k[:, None] * V                # K-dim row offset (BK,1) broadcast
                + o_v[None, :]                    # V-dim col offset (1,BV) broadcast
            )
            tl.store(p_is, b_h.to(tl.float32), mask=mask_h)

        # --- Advance pointers to the next token ---
        p_q += H * K      # q stride per token = H * K
        p_k += H * K      # k stride per token = H * K
        p_o += HV * V     # o stride per token = HV * V
        p_v += HV * V     # v stride per token = HV * V
        p_b += HV         # b stride per token = HV
        p_a += HV * K     # a stride per token = HV * K (KDA vector gate)

    # =========================================================================
    # Optional: write final state back to state pool (decode mode only)
    # Verify mode (DISABLE_STATE_UPDATE=True) skips this — draft tokens may be
    # rejected, so the framework selects the correct intermediate state from
    # the snapshot buffer instead of overwriting the pool.
    # =========================================================================
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n)
            if idx >= 0:
                # Same row-major pointer arithmetic as initial-state load above
                p_h0 = (
                    h0_source
                    + idx * HV * K * V            # skip to pool slot `idx`
                    + i_hv * K * V                # skip to value head `i_hv`
                    + o_k[:, None] * V            # (BK,1) K-dim row stride
                    + o_v[None, :]                # (1,BV) V-dim col offset
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


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
    saved_qkvab: Optional[dict] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    state_layout: str = "vk",
) -> torch.Tensor:
    """
    KDA verify kernel for MTP / Speculative Decoding (Triton baseline).

    Processes T draft tokens per sequence in a fused loop. Writes intermediate
    states to a buffer for post-rejection-sampling selection.

    The kernel works internally in KV layout (state shape K×V). When the caller
    uses VK layout (state shape V×K, cuLA's default), this wrapper transposes
    before launching the kernel and transposes back after.

    Args:
        A_log:            (HV,) fp32            decay log parameter
        dt_bias:          (HV, K) fp32          gate bias
        q:                (N, T, H, K) bf16     queries for T draft tokens
        k:                (N, T, H, K) bf16     keys
        v:                (N, T, HV, V) bf16    values
        a:                (N, T, HV, K) bf16    KDA vector gate input
        b:                (N, T, HV) bf16       sigmoid beta gate input
        initial_state_source:
            (pool_size, HV, V, K) fp32  if state_layout="vk"
            (pool_size, HV, K, V) fp32  if state_layout="kv"
        initial_state_indices:  (N,) int32  maps batch → state pool slot
        intermediate_states_buffer:
            (N, T, HV, V, K) fp32  if state_layout="vk"
            (N, T, HV, K, V) fp32  if state_layout="kv"
        intermediate_state_indices: (N,) int32  reserved for Phase 4
        cache_intermediate_states:  write intermediate state per token step
        disable_state_update:       do not write final state back to pool
        scale:            attention scale, default K^{-0.5}
        use_qk_l2norm_in_kernel: L2-normalize Q and K before computing
        state_layout:     "vk" (state is V×K) or "kv" (state is K×V)

    Returns:
        output: (N, T, HV, V) logits for each draft token
    """
    # Phase 1 guards — these features are not yet implemented
    assert not recompute_state, "recompute_state not implemented (Phase 3)"
    assert saved_qkvab is None, "saved_qkvab not implemented (Phase 3)"
    assert retrieve_parent_token is None, "tree speculation not implemented (Phase 1)"

    # --- Extract dimensions ---
    B, T, H, K = q.shape  # B=N (batch), T=spec_len, H=num QK heads, K=head dim
    HV = v.shape[2]       # num value heads (HV >= H for GQA)
    V = v.shape[3]        # value head dimension
    N = initial_state_indices.shape[0]
    assert B == N, "Dense layout required: B must equal N"

    # --- Tile sizes (power-of-2 for Triton vectorization) ---
    # BK covers the full K dimension in one tile (NK=1 required by kernel)
    # BV capped at 32 to limit register pressure; V=128 → NV=4 tiles of 32
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NK = triton.cdiv(K, BK)    # always 1 for K=128
    NV = triton.cdiv(V, BV)    # e.g. V=128, BV=32 → NV=4
    assert NK == 1, "NK > 1 is not supported"

    if scale is None:
        scale = K ** -0.5

    # --- State layout conversion ---
    # The kernel always works in KV layout: state shape (pool, HV, K, V).
    # If the caller uses VK layout (pool, HV, V, K), transpose before kernel,
    # and transpose the results back after.
    is_vk = state_layout.lower() == "vk"

    if is_vk:
        # VK → KV: (pool, HV, V, K) → (pool, HV, K, V)
        h0_kv = initial_state_source.permute(0, 1, 3, 2).contiguous()
        if cache_intermediate_states:
            # Allocate internal KV-layout buffer: (N, T, HV, K, V)
            is_buf_kv = torch.empty(
                N, T, HV, K, V, dtype=torch.float32, device=q.device
            )
        else:
            is_buf_kv = intermediate_states_buffer
    else:
        # Already KV layout, use directly
        h0_kv = initial_state_source
        is_buf_kv = intermediate_states_buffer

    if is_buf_kv is None:
        is_buf_kv = torch.empty(0, device=q.device, dtype=torch.float32)

    # --- Allocate output: (NK, N, T, HV, V), NK=1 squeezed after kernel ---
    o = q.new_empty(NK, B, T, HV, V)

    # --- Launch kernel ---
    # Grid dims: (NK=1, NV, N*HV)
    #   x: K-tile (always 1 since BK=K)
    #   y: V-tile (NV tiles to cover full V, e.g. 4 for V=128)
    #   z: (batch, value_head) pairs — each program handles one (n, hv)
    grid = (NK, NV, N * HV)

    kda_verify_triton_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=h0_kv,
        h0_indices=initial_state_indices,
        intermediate_states_buffer=is_buf_kv,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        CACHE_INTERMEDIATE_STATES=cache_intermediate_states,
        DISABLE_STATE_UPDATE=disable_state_update,
        num_warps=1,
        num_stages=3,
    )

    # --- Post-kernel: convert KV-layout results back to VK if needed ---
    if is_vk and cache_intermediate_states and intermediate_states_buffer is not None:
        # intermediate_states: KV → VK: (N, T, HV, K, V) → (N, T, HV, V, K)
        intermediate_states_buffer.copy_(is_buf_kv.permute(0, 1, 2, 4, 3))

    if not disable_state_update and is_vk:
        # h0_source: KV → VK: (pool, HV, K, V) → (pool, HV, V, K)
        initial_state_source.copy_(h0_kv.permute(0, 1, 3, 2))

    # Squeeze NK dimension: (1, N, T, HV, V) → (N, T, HV, V)
    return o.squeeze(0)
