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
bench_kda_verify.py — 3-way benchmark for KDA verify (multi-token prediction)

Compares:
  1. cuLA CuTe DSL verify (fused T-token kernel)
  2. cuLA Triton verify (fused T-token kernel)
  3. cuLA CuTe DSL decode × T (serial single-token launches)

Usage:
    python benchmarks/bench_kda_verify.py
    python benchmarks/bench_kda_verify.py --batch-sizes 1 4 8 32
    python benchmarks/bench_kda_verify.py --spec-lens 2 4 5 8
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmarks.utils import benchmark_cuda_fn


def make_verify_inputs(N, T, H, HV, K, V, device="cuda", seed=42):
    torch.manual_seed(seed)
    q = torch.randn(N, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(N, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(N, T, HV, V, device=device, dtype=torch.bfloat16)
    a = (torch.randn(N, T, HV, K, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    b = torch.randn(N, T, HV, device=device, dtype=torch.bfloat16)
    A_log = -torch.rand(HV, device=device, dtype=torch.float32) * 2
    dt_bias = torch.randn(HV, K, device=device, dtype=torch.float32) * 0.1
    state = torch.randn(N, HV, V, K, device=device, dtype=torch.float32) * 0.01
    return q, k, v, a, b, A_log, dt_bias, state


def bench_one(N, T, H, HV, K, V, warmup=20, rep=100):
    from cula.ops.kda_decode import kda_decode
    from cula.ops.kda_verify_cute import kda_verify as kda_verify_cute
    from cula.ops.kda_verify_triton import kda_verify as kda_verify_triton

    q, k, v, a, b, A_log, dt_bias, state = make_verify_inputs(N, T, H, HV, K, V)
    scale = K**-0.5
    indices = torch.arange(N, device="cuda", dtype=torch.int32)
    is_buf = torch.zeros(N, T, HV, V, K, dtype=torch.float32, device="cuda")

    # Warmup + benchmark CuTe verify
    def run_cute():
        is_buf.zero_()
        kda_verify_cute(
            A_log=A_log, dt_bias=dt_bias,
            q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state.clone(),
            initial_state_indices=indices,
            intermediate_states_buffer=is_buf,
            intermediate_state_indices=indices,
            scale=scale,
        )

    ms_cute = benchmark_cuda_fn(run_cute, warmup=warmup, rep=rep)

    # Warmup + benchmark Triton verify
    def run_triton():
        is_buf.zero_()
        kda_verify_triton(
            A_log=A_log, dt_bias=dt_bias,
            q=q, k=k, v=v, a=a, b=b,
            initial_state_source=state.clone(),
            initial_state_indices=indices,
            intermediate_states_buffer=is_buf,
            intermediate_state_indices=indices,
            scale=scale,
        )

    ms_triton = benchmark_cuda_fn(run_triton, warmup=warmup, rep=rep)

    # Warmup + benchmark decode × T (serial)
    q_dec = q.reshape(N * T, 1, H, K)[:N]  # just first token shape
    k_dec = k.reshape(N * T, 1, H, K)[:N]
    v_dec = v.reshape(N * T, 1, HV, V)[:N]
    a_dec = a.reshape(N * T, 1, HV, K)[:N]
    b_dec = b.reshape(N * T, 1, HV)[:N]
    dec_indices = torch.arange(N, device="cuda", dtype=torch.int32)

    def run_decode_serial():
        s = state.clone()
        for t in range(T):
            kda_decode(
                A_log=A_log, dt_bias=dt_bias,
                q=q[:, t:t+1], k=k[:, t:t+1], v=v[:, t:t+1],
                a=a[:, t:t+1], b=b[:, t:t+1],
                initial_state_source=s,
                initial_state_indices=dec_indices,
                scale=scale,
            )

    ms_serial = benchmark_cuda_fn(run_decode_serial, warmup=warmup, rep=rep)

    return ms_cute, ms_triton, ms_serial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 32])
    parser.add_argument("--spec-lens", nargs="+", type=int, default=[2, 4, 5, 8])
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--HV", type=int, default=128)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"H={args.H}, HV={args.HV}, K={args.K}, V={args.V}")
    print()

    header = f"{'N':>5} {'T':>3} | {'CuTe(ms)':>10} {'Triton(ms)':>10} {'Decode×T(ms)':>12} | {'CuTe speedup':>12}"
    print(header)
    print("-" * len(header))

    for N in args.batch_sizes:
        for T in args.spec_lens:
            ms_cute, ms_triton, ms_serial = bench_one(
                N, T, args.H, args.HV, args.K, args.V
            )
            speedup = ms_serial / ms_cute if ms_cute > 0 else float("inf")
            print(
                f"{N:5d} {T:3d} | {ms_cute:10.3f} {ms_triton:10.3f} {ms_serial:12.3f} | {speedup:11.2f}x"
            )


if __name__ == "__main__":
    main()
