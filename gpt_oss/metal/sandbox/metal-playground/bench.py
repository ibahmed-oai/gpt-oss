import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from metal_runner import MetalRunner

ROOT = Path(__file__).parent
KDIR = ROOT / "metal_kernels"
WARMUP_ITERS = 10
AUTOTUNE_ITERS = 10


def ceil_div(a, b):
    return (a + b - 1) // b


def mps_available():
    try:
        import torch

        return torch.backends.mps.is_available()
    except Exception:
        return False


def time_mps_op(fn, iters: int):
    times = []
    assert torch.backends.mps.is_available()
    REPEAT = 10
    for _ in range(iters):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            fn()
        torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0 / REPEAT)
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
    }


def read_buffer(buf, shape, dtype):
    """
    Read a Metal MTLBuffer into a NumPy array.

    Args:
        buf   : MTLBuffer
        shape : tuple[int, ...]  — desired array shape (e.g., (n,), (m, n))
        dtype : np.dtype or any dtype-like (e.g., np.float32, np.int32, "float16")

    Returns:
        np.ndarray with given dtype and shape, copied to host memory.
    """
    dtype = np.dtype(dtype)
    n_elems = int(np.prod(shape))
    nbytes_needed = n_elems * dtype.itemsize
    nbytes_avail = buf.length()

    if nbytes_needed > nbytes_avail:
        raise ValueError(
            f"read_buffer: need {nbytes_needed} bytes for shape {shape} and dtype {dtype}, "
            f"but buffer only has {nbytes_avail} bytes."
        )

    mv = buf.contents().as_buffer(nbytes_avail)
    arr = np.frombuffer(mv, dtype=dtype, count=n_elems).reshape(shape).copy()
    return arr


def _validate_matmul_simdgroup_block_cfg(
    cfg: dict[str, int], M: int, K: int, N: int, runner: MetalRunner = None
) -> tuple[bool, str]:
    """
    Validate the matmul_simdgroup_block kernel configuration.
    """
    for key, val in cfg.items():
        if key == "RHS_TRANSPOSE":
            continue
        if val <= 0:
            return False, f"{key} must be greater than 0"
        if val % 8 != 0:
            return False, f"{key} must be divisible by 8"
    if M % cfg["Bm"] != 0:
        return False, "Bm must be divisible by M"
    if N % cfg["Bn"] != 0:
        return False, "Bn must be divisible by N"
    if K % cfg["Bk"] != 0:
        return False, "Bk must be divisible by K"
    if cfg["Bm"] % cfg["Sg_Bm"] != 0:
        return False, "Bm must be divisible by Sg_Bm"
    if cfg["Bn"] % cfg["Sg_Bn"] != 0:
        return False, "Bn must be divisible by Sg_Bn"
    if runner is not None:
        max_threads_per_threadgroup = runner.max_threads_per_threadgroup()
        threads_per_tg = (
            ceil_div(cfg["Bm"], cfg["Sg_Bm"]) * ceil_div(cfg["Bn"], cfg["Sg_Bn"]) * 32
        )
        if threads_per_tg > max_threads_per_threadgroup:
            return (
                False,
                f"cfg {cfg} results in {threads_per_tg} threads per threadgroup, which is greater than max_threads_per_threadgroup {max_threads_per_threadgroup}",
            )
    return True, ""


def _autotune_matmul_simdgroup_block(
    runner: MetalRunner,
    buffers: dict[int, object],
    iters: int,
    M: int,
    K: int,
    N: int,
    cfgs: list[dict[str, int]],
    kernel: str,
    log_path: Path | None = None,
    rhs_transpose: bool = False,
):
    """
    Autotune the matmul_simdgroup_block kernel.
    cfgs is a list of dicts of the following keys:
    - Bm
    - Bn
    - Bk
    - Sg_Bm
    - Sg_Bn
    """
    total_configs = len(cfgs)
    configs_successfully_compiled = 0
    best = None
    best_time = float("inf")
    records = []
    for cfg in cfgs:
        record = {
            "config": dict(cfg),
            "valid": False,
            "reason": "",
            "threads_per_tg": None,
            "tgs_per_grid": None,
            "compiled": False,
            "timings_ms": None,
            "p50_ms": None,
            "mean_ms": None,
            "p90_ms": None,
            "gflops": None,
        }
        valid, err = _validate_matmul_simdgroup_block_cfg(cfg, M, K, N, runner)
        if not valid:
            print(f"Skipping invalid configuration: {err}")
            record["valid"] = False
            record["reason"] = err
            records.append(record)
            continue
        grid, thread_group = _get_grid_and_thread_group_matmul_simdgroup_block(
            cfg, M, N
        )
        cfg["THREADS_PER_TG"] = int(np.prod(thread_group))
        cfg["RHS_TRANSPOSE"] = 1 if rhs_transpose else 0
        lib = runner.compile_library_from_file(str(KDIR / f"{kernel}.metal"), cfg)
        try:
            pipe = runner.pipeline(lib, kernel)
        except Exception as e:
            print(f"Skipping invalid configuration: {e}")
            record["valid"] = False
            record["reason"] = str(e)
            records.append(record)
            continue
        for _ in range(WARMUP_ITERS):
            runner.run(pipe, buffers, grid, thread_group)
        times = []
        for _ in range(iters):
            info = runner.run(pipe, buffers, grid, thread_group)
            ms = info.get("gpu_ms", float("nan"))
            assert math.isfinite(ms)
            times.append(ms)
        p50 = float(np.percentile(times, 50))
        print(f"Configuration {cfg} took {p50:.3f} ms")
        if p50 < best_time:
            best = cfg
            best_time = p50
        record["valid"] = True
        record["compiled"] = True
        record["threads_per_tg"] = int(np.prod(thread_group))
        record["tgs_per_grid"] = {
            "x": grid[0],
            "y": grid[1],
            "z": grid[2],
        }
        record["timings_ms"] = times
        record["p50_ms"] = p50
        record["mean_ms"] = float(np.mean(times))
        record["p90_ms"] = float(np.percentile(times, 90))
        record["gflops"] = 2 * M * K * N / p50 / 1e6
        records.append(record)
        configs_successfully_compiled += 1
    print(
        f"Out of {total_configs} configurations, {configs_successfully_compiled} configurations were successfully compiled."
    )
    print(f"Best configuration: {best}")
    if log_path:
        with open(log_path, "w") as f:
            json.dump(records, f, indent=2)
    return best


def _get_cfgs_matmul_simdgroup_block(
    M: int,
    K: int,
    N: int,
) -> list[dict[str, int]]:
    """
    Get the configurations for the matmul_simdgroup_block kernel.
    """
    assert M % 8 == 0
    assert N % 8 == 0
    assert K % 8 == 0
    # Bm, Bk, Bn are multiples of 8.
    # Sg_Bm, Sg_Bn are multiples of 8.
    cfgs = []
    for Bm_exp in range(1, 4):
        for Bk_exp in range(1, 4):
            for Bn_exp in range(1, 4):
                Bm_multiplier = 2**Bm_exp
                Bk_multiplier = 2**Bk_exp
                Bn_multiplier = 2**Bn_exp
                if Bm_multiplier * 8 > M:
                    continue
                if Bn_multiplier * 8 > N:
                    continue
                if Bk_multiplier * 8 > K:
                    continue
                for Sg_Bm_exp in range(Bm_exp + 1):
                    for Sg_Bn_exp in range(Bn_exp + 1):
                        Sg_Bm_factor = 2**Sg_Bm_exp
                        Sg_Bn_factor = 2**Sg_Bn_exp
                        cfgs.append(
                            {
                                "Bm": Bm_multiplier * 8,
                                "Bn": Bn_multiplier * 8,
                                "Bk": Bk_multiplier * 8,
                                "Sg_Bm": Sg_Bm_factor * 8,
                                "Sg_Bn": Sg_Bn_factor * 8,
                            }
                        )
    print(f"Generated {len(cfgs)} configurations")
    return cfgs


def _get_grid_and_thread_group_matmul_simdgroup_block(
    cfg: dict[str, int],
    M: int,
    N: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Get the grid and thread group for the matmul_simdgroup_block kernel.
    """
    Bm = cfg["Bm"]
    Bn = cfg["Bn"]
    Sg_Bm = cfg["Sg_Bm"]
    Sg_Bn = cfg["Sg_Bn"]
    sg_count = ceil_div(Bm, Sg_Bm) * ceil_div(Bn, Sg_Bn)
    threads_per_sg = 32
    thread_group = (sg_count * threads_per_sg, 1, 1)
    grid = (ceil_div(M, Bm), ceil_div(N, Bn), 1)
    grid = (grid[1], grid[0], 1)
    return grid, thread_group


def run_matmul(
    runner: MetalRunner,
    a: torch.Tensor,
    b: torch.Tensor,
    iters: int,
    kernel: str,
    autotune: bool = False,
    autotune_log_path: Path | None = None,
    c: torch.Tensor | None = None,
    rhs_transpose: bool = False,
    bias: torch.Tensor | None = None,
):
    if kernel == "matmul_f32_bf16w":
        b = b.to(torch.bfloat16)
        if bias is not None:
            bias = bias.to(torch.bfloat16)
    input_bytes_accessed = a.element_size() * a.numel() + b.element_size() * b.numel()
    if c is not None:
        input_bytes_accessed += c.element_size() * c.numel()
        ref = torch.matmul(a.float(), b.float()) + c
    else:
        ref = torch.matmul(a.float(), b.float())
    if bias is not None:
        input_bytes_accessed += bias.element_size() * bias.numel()
        ref += bias
    ref = ref.numpy()
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    # Host data
    a = a.numpy()
    if rhs_transpose:
        b = b.T.contiguous()
    b = b.numpy() if kernel != "matmul_f32_bf16w" else b.view(torch.uint16).numpy()
    if bias is not None:
        bias = (
            bias.numpy()
            if kernel != "matmul_f32_bf16w"
            else bias.view(torch.uint16).numpy()
        )
    # Buffers
    buf_a = runner.make_buffer(a, label="a")
    buf_b = runner.make_buffer(b, label="b")
    out = np.empty((M, N), dtype=np.float32) if c is None else c.numpy()
    total_bytes_accessed = input_bytes_accessed + out.itemsize * np.prod(out.shape)
    buf_c = runner.make_buffer(out, label="c")
    buf_bias = runner.make_buffer(bias, label="bias") if bias is not None else None
    # Params
    buffers = {}
    if kernel == "matmul_f32_bf16w":
        add = 0 if c is None else 1
        rhs_transpose = 1 if rhs_transpose else 0
        params = np.array(
            [(M, K, N, add, rhs_transpose)],
            dtype=[
                ("M", np.uint32),
                ("K", np.uint32),
                ("N", np.uint32),
                ("add", np.uint32),
                ("rhs_transpose", np.uint32),
            ],
        )
        buf_p = runner.make_buffer(params.tobytes(), label="params")
        buffers = {0: buf_a, 1: buf_b, 2: buf_c, 3: buf_p}
        if bias is not None and kernel == "matmul_f32_bf16w":
            print("Adding bias to buffer")
            buffers = {0: buf_p, 1: buf_a, 2: buf_b, 3: buf_bias, 4: buf_c}
        Bm = 8 * 8
        Bn = 8 * 8
        Bk = 8 * 4
        Sg_Bm = int(Bm / 4)
        Sg_Bn = int(Bn / 4)

        Bm = 8 * 4
        Bn = 8 * 8
        Bk = 8 * 4
        Sg_Bm = int(16)
        Sg_Bn = int(32)
        if kernel == "matmul_simdgroup_block" or kernel == "matmul_f32_bf16w":
            if rhs_transpose:
                if M == 1024:
                    Bm = 8 * 8
                    Bn = 8 * 2
                    Bk = 8 * 8
                    Sg_Bm = int(8 * 2)
                    Sg_Bn = int(8 * 2)
            else:
                Bm = 8 * 4
                Bn = 8 * 8
                Bk = 8 * 4
                Sg_Bm = int(8 * 4)
                Sg_Bn = int(8 * 4)
            # Bm = 8 * 1
            # Bn = 8 * 1
            # Bk = 8 * 1
            # Sg_Bm = int(8 * 1)
            # Sg_Bn = int(8 * 1)
        elif kernel == "matmul_simdgroup_tg_mem":
            Bm = 8 * 4
            Bn = 8 * 8
            Bk = 8 * 2
            Sg_Bm = int(8 * 1)
            Sg_Bn = int(8 * 8)
            Bm = 8 * 4
            Bn = 8 * 8
            Bk = 8 * 8
            Sg_Bm = int(8 * 1)
            Sg_Bn = int(8 * 2)
        func_constants = {
            "Bm": Bm,
            "Bn": Bn,
            "Bk": Bk,
            "Sg_Bm": Sg_Bm,
            "Sg_Bn": Sg_Bn,
        }
        if autotune:
            cfgs = _get_cfgs_matmul_simdgroup_block(M, K, N)
            best = _autotune_matmul_simdgroup_block(
                runner,
                buffers,
                AUTOTUNE_ITERS,
                M,
                K,
                N,
                cfgs,
                kernel,
                autotune_log_path,
                rhs_transpose,
            )
            func_constants = best
        assert func_constants is not None
        valid, err = _validate_matmul_simdgroup_block_cfg(
            func_constants, M, K, N, runner
        )
        assert valid, err
        grid, thread_group = _get_grid_and_thread_group_matmul_simdgroup_block(
            func_constants, M, N
        )
        func_constants["THREADS_PER_TG"] = int(np.prod(thread_group))
        func_constants["RHS_TRANSPOSE"] = 1 if rhs_transpose else 0
        lib = runner.compile_library_from_file(
            str(KDIR / f"{kernel}.metal"), func_constants
        )
        pipe = runner.pipeline(lib, kernel)
        # Each thread group is formed of a single simdgroup for now. Each simdgroup performs an 8x8 matrix multiplication.
        assert M % Bm == 0
        assert N % Bn == 0
        assert K % Bk == 0
        print(f"sg per tg {np.prod(thread_group) / 32}")
        print(
            f"Bm:{func_constants['Bm']}, Bn:{func_constants['Bn']}, Bk:{func_constants['Bk']}"
        )
        print(f"Sg_Bm:{func_constants['Sg_Bm']}, Sg_Bn:{func_constants['Sg_Bn']}")
        print(f"THREADS_PER_TG: {func_constants['THREADS_PER_TG']}")
        print(
            f"{thread_group[0]} * {thread_group[1]} * {thread_group[2]} = {np.prod(thread_group)}"
        )
    else:
        raise ValueError(f"Kernel {kernel} not supported")
    # Warmup
    for i in range(WARMUP_ITERS):
        # runner.run(pipe, {0: buf_a, 1: buf_b, 2: buf_c, 3: buf_p}, grid, thread_group)
        runner.run(pipe, buffers, grid, thread_group)
        if i == 0:
            c_host = read_buffer(buf_c, (M, N), np.float32)

    # Timed
    times = []
    for _ in range(iters):
        info = runner.run(pipe, buffers, grid, thread_group)
        ms = info.get("gpu_ms", float("nan"))
        assert math.isfinite(ms)
        times.append(ms)

    # Copy back
    # c_host = read_buffer(buf_c, (M, N), np.float32)
    print(c_host.shape)
    print(ref.shape)
    diff_max = float(np.max(np.abs(c_host - ref)))
    # with np.printoptions(threshold=np.inf):
    #     print(ref)
    #     print(c_host)
    print("-" * 80)
    print(f"matmul m={M}, n={N}, k ={K}, kernel={kernel}, iters={iters}")
    print(f"max|diff|: {diff_max:.3e}")
    p50 = np.percentile(times, 50)
    print(
        f"mean: {np.mean(times):.3f} ms, p50: {p50:.3f} ms, p90: {np.percentile(times, 90):.3f} ms"
    )
    print(f"GFLOPS: {2 * M * K * N / p50 / 1e6:.3f}")
    print(f"GB/s: {total_bytes_accessed / p50 / 1e6:.3f}")
    np.testing.assert_allclose(c_host, ref, rtol=1e-4, atol=1e-4)


def vadd_mps(a: torch.Tensor, b: torch.Tensor, iters: int):
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS backend not available. (macOS + Apple Silicon + PyTorch MPS build required)"
        )

    # Function we time on MPS
    out = torch.empty_like(a, device="mps")

    def op():
        torch.add(a, b, out=out)

    for _ in range(WARMUP_ITERS):
        op()

    stats = time_mps_op(op, iters)
    c_host = out.cpu().numpy().copy()
    ref = (a + b).cpu().numpy()
    diff_max = float(np.max(np.abs(c_host - ref)))

    print("-" * 80)
    print(f"vadd_mps n={a.shape[0]}, iters={iters}")
    print(f"max|diff|: {diff_max:.3e}")
    print(
        f"mean: {stats['mean_ms']:.3f} ms, p50: {stats['p50_ms']:.3f} ms, p90: {stats['p90_ms']:.3f} ms"
    )
    # Assumes float32
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32
    assert c_host.dtype == np.float32
    total_bytes_accessed = 4 * (
        np.prod(a.shape) + np.prod(b.shape) + np.prod(c_host.shape)
    )
    print(f"GB/s: {total_bytes_accessed / stats['p50_ms'] / 1e6:.3f}")
    np.testing.assert_allclose(c_host, ref)


def matmul_mps(a: torch.Tensor, b: torch.Tensor, iters: int):
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS backend not available. (macOS + Apple Silicon + PyTorch MPS build required)"
        )

    out = torch.empty(a.shape[0], b.shape[1], device="mps")
    total_bytes_accessed = (
        a.element_size() * a.numel()
        + b.element_size() * b.numel()
        + out.element_size() * out.numel()
    )

    def op():
        torch.matmul(a, b, out=out)

    for _ in range(WARMUP_ITERS):
        op()

    stats = time_mps_op(op, iters)
    c_host = out.cpu().numpy().copy()
    ref = torch.matmul(a, b).cpu().numpy()
    diff_max = float(np.max(np.abs(c_host - ref)))

    print("-" * 80)
    print(f"matmul_mps m={a.shape[0]}, n={b.shape[1]}, iters={iters}")
    print(f"max|diff|: {diff_max:.3e}")
    print(
        f"mean: {stats['mean_ms']:.3f} ms, p50: {stats['p50_ms']:.3f} ms, p90: {stats['p90_ms']:.3f} ms"
    )
    print(
        f"GFLOPS: {2 * a.shape[0] * a.shape[1] * b.shape[1] / stats['p50_ms'] / 1e6:.3f}"
    )
    print(f"GB/s: {total_bytes_accessed / stats['p50_ms'] / 1e6:.3f}")
    np.testing.assert_allclose(c_host, ref)


def main():
    ap = argparse.ArgumentParser(description="Metal kernel playground & benchmarks")
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--mps", action="store_true")
    parent_parser.add_argument("--autotune", action="store_true")
    parent_parser.add_argument("--autotune-log-path", type=str, default="autotune.json")

    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_mm = sub.add_parser("matmul", help="Matmul", parents=[parent_parser])
    ap_mm.add_argument("--m", type=int, default=1024)
    ap_mm.add_argument("--n", type=int, default=1024)
    ap_mm.add_argument("--k", type=int, default=1024)
    ap_mm.add_argument("--iters", type=int, default=10)
    ap_mm.add_argument("--kernel", type=str, default="matmul_naive")
    ap_mm.add_argument("--add", action="store_true")
    ap_mm.add_argument("--rhs-transpose", action="store_true")
    ap_mm.add_argument("--bias", action="store_true")

    args = ap.parse_args()

    runner = MetalRunner()

    if args.cmd == "matmul":
        a = torch.randn(args.m, args.k, dtype=torch.float32)
        b = torch.randn(args.k, args.n, dtype=torch.float32)
        # fill c with 1s
        c = (
            # torch.zeros(args.m * args.n, dtype=torch.float32).reshape(args.m, args.n)
            torch.randn(args.m * args.n, dtype=torch.float32).reshape(args.m, args.n)
            if args.add
            else None
        )
        # a = torch.arange(args.m * args.k, dtype=torch.float32).reshape(args.m, args.k)
        # a = torch.eye(args.m, dtype=torch.float32).reshape(args.m, args.k)
        # b = torch.eye(args.n, dtype=torch.float32).reshape(args.k, args.n)
        # bias = torch.ones(args.n, dtype=torch.float32) if args.bias else None
        bias = torch.randn(args.n, dtype=torch.float32) if args.bias else None
        run_matmul(
            runner,
            a,
            b,
            args.iters,
            args.kernel,
            args.autotune,
            Path(args.autotune_log_path) if args.autotune else None,
            c,
            args.rhs_transpose,
            bias,
        )
        if args.mps:
            a_mps = a.to(device="mps")
            b_mps = b.to(device="mps")
            matmul_mps(a_mps, b_mps, args.iters)
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
