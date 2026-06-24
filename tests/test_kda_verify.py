#!/usr/bin/env python3
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
Unit tests for kda_verify (Triton KDA verify kernel for MTP / Speculative Decoding).

Compares against a pure PyTorch reference that loops single-token decode over
max_steps tokens, accumulating intermediate states.
"""

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cula.ops.kda_verify_triton import kda_verify


# ---------------------------------------------------------------------------
# PyTorch reference: single-token decode (copied from test_kda_decode.py)
# ---------------------------------------------------------------------------
def torch_kda_decode_ref(
    q,  # (N, H, K) float32
    k,  # (N, H, K) float32
    v,  # (N, HV, V) float32
    a,  # (N, HV, K) float32
    b,  # (N, HV) float32
    A_log,  # (HV,) float32
    dt_bias,  # (HV, K) float32
    state,  # (N, HV, V, K) float32
    scale,  # float
    use_l2norm=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
):
    """
    Pure PyTorch reference for single-token KDA decode.

    Returns:
        o:         (N, HV, V) float32
        state_new: (N, HV, V, K) float32
    """
    N, HV, V, K = state.shape
    H = q.shape[1]
    heads_per_group = HV // H

    A = torch.exp(A_log)  # (HV,)

    state_new = state.clone()
    o = torch.zeros(N, HV, V, dtype=torch.float32, device=q.device)

    for n in range(N):
        for hv in range(HV):
            i_h = hv // heads_per_group

            # Gate: exp(-A * softplus(a + dt_bias))
            x = a[n, hv, :] + dt_bias[hv, :]  # (K,)
            sp = F.softplus(x, beta=softplus_beta, threshold=softplus_threshold)
            gate = torch.exp(-A[hv] * sp)  # (K,)

            if use_l2norm:
                q_vec = F.normalize(q[n, i_h, :], dim=0) * scale
                k_vec = F.normalize(k[n, i_h, :], dim=0)
            else:
                q_vec = q[n, i_h, :] * scale
                k_vec = k[n, i_h, :]

            Hk = state[n, hv] @ (gate * k_vec)  # (V,)

            beta_val = torch.sigmoid(b[n, hv])
            v_new = beta_val * (v[n, hv, :] - Hk)  # (V,)

            state_new[n, hv] = gate[None, :] * state[n, hv] + v_new[:, None] * k_vec[None, :]

            o[n, hv, :] = state_new[n, hv] @ q_vec  # (V,)

    return o, state_new


# ---------------------------------------------------------------------------
# PyTorch reference: multi-token verify
# ---------------------------------------------------------------------------
def torch_kda_verify_ref(
    q,  # (N, T, H, K) float32
    k,  # (N, T, H, K) float32
    v,  # (N, T, HV, V) float32
    a,  # (N, T, HV, K) float32
    b,  # (N, T, HV) float32
    A_log,  # (HV,) float32
    dt_bias,  # (HV, K) float32
    state,  # (N, HV, V, K) float32
    scale,  # float
    use_l2norm=True,
    softplus_beta=1.0,
    softplus_threshold=20.0,
):
    """
    Multi-token verify reference: loop over T tokens using single-token decode.

    Returns:
        output:              (N, T, HV, V) float32
        intermediate_states: (N, T, HV, V, K) float32
    """
    T = q.shape[1]
    outputs = []
    intermediate_states = []

    for t in range(T):
        o_t, state = torch_kda_decode_ref(
            q[:, t],
            k[:, t],
            v[:, t],
            a[:, t],
            b[:, t],
            A_log,
            dt_bias,
            state,
            scale,
            use_l2norm=use_l2norm,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
        )
        outputs.append(o_t)
        intermediate_states.append(state.clone())

    return torch.stack(outputs, dim=1), torch.stack(intermediate_states, dim=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_verify_inputs(N, T, H, HV, K, V, device="cuda", seed=42):
    """Generate random inputs for KDA verify."""
    torch.manual_seed(seed)
    q = torch.randn(N, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(N, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(N, T, HV, V, device=device, dtype=torch.bfloat16)
    a = (torch.randn(N, T, HV, K, device=device, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    b = torch.randn(N, T, HV, device=device, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32) * 2
    dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32) * 0.01
    return q, k, v, a, b, A_log, dt_bias, state


def _assert_close(name, ref, actual, atol=3e-2, rtol=2e-2):
    """Assert tensors are close with informative error message."""
    diff = (ref.float() - actual.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = torch.allclose(ref.float(), actual.float(), atol=atol, rtol=rtol)
    assert ok, (
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
        f"atol={atol}, rtol={rtol}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec_len", [1, 2, 4, 5, 8])
@pytest.mark.parametrize("N", [1, 8, 32])
@pytest.mark.parametrize("HV", [4, 128])
@pytest.mark.parametrize("use_qk_l2norm", [True, False])
@pytest.mark.parametrize("state_layout", ["vk", "kv"])
def test_kda_verify_triton(spec_len, N, HV, use_qk_l2norm, state_layout):
    H = max(1, HV // 2)
    K, V = 128, 128
    scale = K**-0.5

    q, k, v, a, b, A_log, dt_bias, state = make_verify_inputs(
        N, spec_len, H, HV, K, V
    )

    # --- Reference (fp32, always in VK layout) ---
    o_ref, states_ref = torch_kda_verify_ref(
        q.float(),
        k.float(),
        v.float(),
        a.float(),
        b.float(),
        A_log,
        dt_bias,
        state.clone(),
        scale,
        use_l2norm=use_qk_l2norm,
    )

    # --- Kernel ---
    if state_layout == "vk":
        state_source = state.clone()  # (N, HV, V, K)
        is_buf = torch.zeros(
            N, spec_len, HV, V, K, dtype=torch.float32, device="cuda"
        )
    else:
        state_source = state.permute(0, 1, 3, 2).contiguous()  # (N, HV, K, V)
        is_buf = torch.zeros(
            N, spec_len, HV, K, V, dtype=torch.float32, device="cuda"
        )

    indices = torch.arange(N, device="cuda", dtype=torch.int32)

    o_kernel = kda_verify(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=state_source,
        initial_state_indices=indices,
        intermediate_states_buffer=is_buf,
        intermediate_state_indices=indices,
        cache_intermediate_states=True,
        disable_state_update=True,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        state_layout=state_layout,
    )

    # --- Assert output ---
    _assert_close("output", o_ref, o_kernel.float(), atol=3e-2, rtol=2e-2)

    # --- Assert intermediate states ---
    if state_layout == "kv":
        # Transpose kernel states from (N,T,HV,K,V) to (N,T,HV,V,K) for comparison
        states_kernel = is_buf.permute(0, 1, 2, 4, 3).contiguous()
    else:
        states_kernel = is_buf

    if use_qk_l2norm:
        _assert_close(
            "intermediate_states", states_ref, states_kernel, atol=1e-4, rtol=1e-5
        )
    else:
        _assert_close(
            "intermediate_states", states_ref, states_kernel, atol=1e-1, rtol=5e-2
        )


def test_kda_verify_disable_state_update():
    """Verify initial_state_source is NOT modified when disable_state_update=True."""
    N, T, H, HV, K, V = 4, 3, 4, 8, 128, 128
    q, k, v, a, b, A_log, dt_bias, state = make_verify_inputs(N, T, H, HV, K, V)

    state_original = state.clone()
    is_buf = torch.zeros(N, T, HV, V, K, dtype=torch.float32, device="cuda")
    indices = torch.arange(N, device="cuda", dtype=torch.int32)

    kda_verify(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=state,
        initial_state_indices=indices,
        intermediate_states_buffer=is_buf,
        intermediate_state_indices=indices,
        disable_state_update=True,
    )

    assert torch.equal(
        state, state_original
    ), "initial_state_source was modified despite disable_state_update=True"


def test_kda_verify_single_token_matches_decode():
    """Verify spec_len=1 produces same output as single-token decode ref."""
    N, H, HV, K, V = 8, 4, 8, 128, 128
    scale = K**-0.5

    q, k, v, a, b, A_log, dt_bias, state = make_verify_inputs(
        N, 1, H, HV, K, V
    )

    # Single-token decode reference
    o_decode, state_decode = torch_kda_decode_ref(
        q[:, 0].float(),
        k[:, 0].float(),
        v[:, 0].float(),
        a[:, 0].float(),
        b[:, 0].float(),
        A_log,
        dt_bias,
        state.clone(),
        scale,
    )

    # Verify kernel with spec_len=1
    is_buf = torch.zeros(N, 1, HV, V, K, dtype=torch.float32, device="cuda")
    indices = torch.arange(N, device="cuda", dtype=torch.int32)

    o_verify = kda_verify(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=state.clone(),
        initial_state_indices=indices,
        intermediate_states_buffer=is_buf,
        intermediate_state_indices=indices,
        disable_state_update=True,
        scale=scale,
    )

    _assert_close("single_token_output", o_decode.unsqueeze(1), o_verify.float(), atol=5e-2, rtol=1e-4)
    _assert_close("single_token_state", state_decode.unsqueeze(1), is_buf, atol=1e-4, rtol=1e-5)


# ---------------------------------------------------------------------------
# CuTe DSL backend tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec_len", [1, 2, 4, 5, 8])
@pytest.mark.parametrize("N", [1, 8])
@pytest.mark.parametrize("HV", [4, 128])
@pytest.mark.parametrize("use_qk_l2norm", [True, False])
@pytest.mark.parametrize("state_layout", ["vk", "kv"])
def test_kda_verify_cute(spec_len, N, HV, use_qk_l2norm, state_layout):
    from cula.ops.kda_verify_cute import kda_verify as kda_verify_cute

    H = max(1, HV // 2)
    K, V = 128, 128
    scale = K**-0.5

    q, k, v, a, b, A_log, dt_bias, state = make_verify_inputs(
        N, spec_len, H, HV, K, V
    )

    # --- Reference ---
    o_ref, states_ref = torch_kda_verify_ref(
        q.float(), k.float(), v.float(), a.float(), b.float(),
        A_log, dt_bias, state.clone(), scale, use_l2norm=use_qk_l2norm,
    )

    # --- CuTe kernel ---
    if state_layout == "vk":
        state_source = state.clone()
        is_buf = torch.zeros(N, spec_len, HV, V, K, dtype=torch.float32, device="cuda")
    else:
        state_source = state.permute(0, 1, 3, 2).contiguous()
        is_buf = torch.zeros(N, spec_len, HV, K, V, dtype=torch.float32, device="cuda")

    indices = torch.arange(N, device="cuda", dtype=torch.int32)

    o_kernel = kda_verify_cute(
        A_log=A_log, dt_bias=dt_bias,
        q=q, k=k, v=v, a=a, b=b,
        initial_state_source=state_source,
        initial_state_indices=indices,
        intermediate_states_buffer=is_buf,
        intermediate_state_indices=indices,
        cache_intermediate_states=True,
        disable_state_update=True,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        state_layout=state_layout,
    )

    _assert_close("cute_output", o_ref, o_kernel.float(), atol=3e-2, rtol=2e-2)

    if state_layout == "kv":
        states_kernel = is_buf.permute(0, 1, 2, 4, 3).contiguous()
    else:
        states_kernel = is_buf

    if use_qk_l2norm:
        _assert_close("cute_intermediate_states", states_ref, states_kernel, atol=1e-4, rtol=1e-5)
    else:
        _assert_close("cute_intermediate_states", states_ref, states_kernel, atol=1e-1, rtol=5e-2)


@pytest.mark.parametrize("spec_len", [1, 2, 4])
@pytest.mark.parametrize("state_layout", ["vk", "kv"])
def test_kda_verify_cute_large_batch(spec_len, state_layout):
    """Test CuTe verify with N >= SMALL_BATCH_THRESHOLD (large_batch kernel path)."""
    from cula.ops.kda_verify_cute import kda_verify as kda_verify_cute

    N = 1024
    HV = 4
    H = 2
    K, V = 128, 128
    scale = K**-0.5

    q, k, v, a, b, A_log, dt_bias, state = make_verify_inputs(
        N, spec_len, H, HV, K, V
    )

    o_ref, states_ref = torch_kda_verify_ref(
        q.float(), k.float(), v.float(), a.float(), b.float(),
        A_log, dt_bias, state.clone(), scale,
    )

    if state_layout == "vk":
        state_source = state.clone()
        is_buf = torch.zeros(N, spec_len, HV, V, K, dtype=torch.float32, device="cuda")
    else:
        state_source = state.permute(0, 1, 3, 2).contiguous()
        is_buf = torch.zeros(N, spec_len, HV, K, V, dtype=torch.float32, device="cuda")

    indices = torch.arange(N, device="cuda", dtype=torch.int32)

    o_kernel = kda_verify_cute(
        A_log=A_log, dt_bias=dt_bias,
        q=q, k=k, v=v, a=a, b=b,
        initial_state_source=state_source,
        initial_state_indices=indices,
        intermediate_states_buffer=is_buf,
        intermediate_state_indices=indices,
        cache_intermediate_states=True,
        disable_state_update=True,
        scale=scale,
        state_layout=state_layout,
    )

    _assert_close("large_batch_output", o_ref, o_kernel.float(), atol=3e-2, rtol=2e-2)

    if state_layout == "kv":
        states_kernel = is_buf.permute(0, 1, 2, 4, 3).contiguous()
    else:
        states_kernel = is_buf
    _assert_close("large_batch_states", states_ref, states_kernel, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("spec_len", [1, 2, 4])
@pytest.mark.parametrize("N", [2, 8])
@pytest.mark.parametrize("state_layout", ["vk", "kv"])
def test_kda_verify_cute_varlen(spec_len, N, state_layout):
    """Test CuTe verify with varlen inputs (q shape (1, N*T, H, K))."""
    from cula.ops.kda_verify_cute import kda_verify as kda_verify_cute

    HV = 4
    H = 2
    K, V = 128, 128
    scale = K**-0.5

    q, k, v, a, b, A_log, dt_bias, state = make_verify_inputs(
        N, spec_len, H, HV, K, V
    )

    o_ref, states_ref = torch_kda_verify_ref(
        q.float(), k.float(), v.float(), a.float(), b.float(),
        A_log, dt_bias, state.clone(), scale,
    )

    # Reshape to varlen format
    q_vl = q.reshape(1, N * spec_len, H, K)
    k_vl = k.reshape(1, N * spec_len, H, K)
    v_vl = v.reshape(1, N * spec_len, HV, V)
    a_vl = a.reshape(N * spec_len, HV, K)
    b_vl = b.reshape(N * spec_len, HV)

    if state_layout == "vk":
        state_source = state.clone()
        is_buf = torch.zeros(N, spec_len, HV, V, K, dtype=torch.float32, device="cuda")
    else:
        state_source = state.permute(0, 1, 3, 2).contiguous()
        is_buf = torch.zeros(N, spec_len, HV, K, V, dtype=torch.float32, device="cuda")

    indices = torch.arange(N, device="cuda", dtype=torch.int32)

    o_kernel = kda_verify_cute(
        A_log=A_log, dt_bias=dt_bias,
        q=q_vl, k=k_vl, v=v_vl, a=a_vl, b=b_vl,
        initial_state_source=state_source,
        initial_state_indices=indices,
        intermediate_states_buffer=is_buf,
        intermediate_state_indices=indices,
        cache_intermediate_states=True,
        disable_state_update=True,
        scale=scale,
        state_layout=state_layout,
    )

    # Reshape output back to dense for comparison
    o_dense = o_kernel.reshape(N, spec_len, HV, V)
    _assert_close("varlen_output", o_ref, o_dense.float(), atol=3e-2, rtol=2e-2)

    if state_layout == "kv":
        states_kernel = is_buf.permute(0, 1, 2, 4, 3).contiguous()
    else:
        states_kernel = is_buf
    _assert_close("varlen_states", states_ref, states_kernel, atol=1e-4, rtol=1e-5)
