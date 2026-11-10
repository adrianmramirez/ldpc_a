
import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numba import (
    get_num_threads,
    set_num_threads,
    config
)

from alist import LDPCCode
from bp_decoder_numba import BPDecoderNumba
from simulation import gf2_rank


# ============================================================
# GENERATE IDENTICAL CHANNEL OBSERVATIONS
# ============================================================

def generate_llrs(code, ebn0_db, frames, seed):

    rank = gf2_rank(code.H)
    K = code.N - rank

    if K <= 0:
        raise ValueError("Code dimension must be positive")

    rate = K / code.N

    ebn0_linear = 10.0 ** (ebn0_db / 10.0)

    sigma2 = 1.0 / (2.0 * rate * ebn0_linear)

    rng = np.random.default_rng(seed)

    # All-zero codeword, BPSK mapped to +1.
    received = 1.0 + rng.normal(
        0.0,
        np.sqrt(sigma2),
        size=(frames, code.N)
    )

    llrs = 2.0 * received / sigma2

    return np.ascontiguousarray(llrs), K


# ============================================================
# SEQUENTIAL NUMBA BASELINE
# ============================================================

def decode_sequential(decoder, channel_llrs):

    frames, N = channel_llrs.shape

    decoded = np.empty(
        (frames, N),
        dtype=np.uint8
    )

    successes = np.empty(
        frames,
        dtype=np.bool_
    )

    iterations = np.empty(
        frames,
        dtype=np.int64
    )

    for frame in range(frames):

        bits, success, count = decoder.decode(
            channel_llrs[frame]
        )

        decoded[frame] = bits
        successes[frame] = success
        iterations[frame] = count

    return decoded, successes, iterations


# ============================================================
# TIMING UTILITY
# ============================================================

def benchmark_function(func, repeats):

    runtimes = []
    result = None

    for _ in range(repeats):

        start = time.perf_counter()

        result = func()

        elapsed = time.perf_counter() - start

        runtimes.append(elapsed)

    return result, float(np.median(runtimes)), runtimes


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(
    code,
    frames,
    ebn0_db,
    iteration_limit,
    thread_counts,
    repeats,
    seed
):

    channel_llrs, K = generate_llrs(
        code,
        ebn0_db,
        frames,
        seed
    )

    decoder = BPDecoderNumba(
        code,
        max_iterations=iteration_limit
    )

    print("\n=== BENCHMARK PARAMETERS ===")
    print(f"Codeword length N: {code.N}")
    print(f"Information bits K: {K}")
    print(f"Frames: {frames:,}")
    print(f"Eb/N0: {ebn0_db:.2f} dB")
    print(f"Maximum BP iterations: {iteration_limit}")
    print(f"Repeats: {repeats}")
    print(f"Available Numba threads: {config.NUMBA_NUM_THREADS}")
    print(f"Thread counts: {thread_counts}")

    # --------------------------------------------------------
    # Warm up JIT compilation
    # --------------------------------------------------------

    print("\nWarming up Numba kernels...")

    decoder.decode(channel_llrs[0])

    # Compile the parallel kernel before timing.
    set_num_threads(1)
    decoder.decode_batch(channel_llrs[:2])

    # --------------------------------------------------------
    # Sequential baseline
    # --------------------------------------------------------

    print("\n=== SEQUENTIAL NUMBA BASELINE ===")

    sequential_result, sequential_time, seq_times = (
        benchmark_function(
            lambda: decode_sequential(
                decoder,
                channel_llrs
            ),
            repeats
        )
    )

    seq_decoded, seq_success, seq_iterations = (
        sequential_result
    )

    print(f"Median runtime: {sequential_time:.6f} s")
    print(
        f"Average iterations: "
        f"{np.mean(seq_iterations):.3f}"
    )

    results = [{
        "configuration": "sequential",
        "threads": 1,
        "median_runtime": sequential_time,
        "speedup_vs_sequential": 1.0,
        "speedup_vs_parallel_1": float("nan"),
        "parallel_efficiency": float("nan"),
        "throughput_info_bps": (
            frames * K / sequential_time
        ),
        "mismatched_frames": 0,
        "mismatched_iterations": 0,
        "mismatched_successes": 0
    }]

    # --------------------------------------------------------
    # Parallel thread scaling
    # --------------------------------------------------------

    parallel_one_time = None

    for threads in thread_counts:

        set_num_threads(threads)

        print(
            f"\n=== PARALLEL NUMBA: "
            f"{get_num_threads()} THREAD(S) ==="
        )

        parallel_result, runtime, times = (
            benchmark_function(
                lambda: decoder.decode_batch(
                    channel_llrs
                ),
                repeats
            )
        )

        par_decoded, par_success, par_iterations = (
            parallel_result
        )

        # --------------------------------------------
        # Correctness verification
        # --------------------------------------------

        mismatched_frames = int(
            np.count_nonzero(
                np.any(
                    seq_decoded != par_decoded,
                    axis=1
                )
            )
        )

        mismatched_iterations = int(
            np.count_nonzero(
                seq_iterations != par_iterations
            )
        )

        mismatched_successes = int(
            np.count_nonzero(
                seq_success != par_success
            )
        )

        if parallel_one_time is None and threads == 1:
            parallel_one_time = runtime

        speedup = sequential_time / runtime

        if parallel_one_time is not None:
            parallel_speedup = parallel_one_time / runtime
            efficiency = parallel_speedup / threads
        else:
            parallel_speedup = float("nan")
            efficiency = float("nan")

        throughput = frames * K / runtime

        print(f"Median runtime: {runtime:.6f} s")
        print(f"Speedup vs sequential: {speedup:.3f}x")
        print(f"Speedup vs parallel 1-thread: {parallel_speedup:.3f}x")
        print(f"Parallel efficiency: {efficiency * 100:.2f}%")
        print(f"Information throughput: {throughput:,.0f} bits/s")

        print(f"Mismatched frames: {mismatched_frames}")
        print(f"Mismatched iterations: {mismatched_iterations}")
        print(f"Mismatched success flags: {mismatched_successes}")

        results.append({
            "configuration": "parallel",
            "threads": threads,
            "median_runtime": runtime,
            "speedup_vs_sequential": speedup,
            "speedup_vs_parallel_1": parallel_speedup,
            "parallel_efficiency": efficiency,
            "throughput_info_bps": throughput,
            "mismatched_frames": mismatched_frames,
            "mismatched_iterations": mismatched_iterations,
            "mismatched_successes": mismatched_successes
        })

    return results


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(results, filename):

    with open(filename, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {filename}")


# ============================================================
# PLOT PERFORMANCE
# ============================================================

def plot_results(results, prefix):

    parallel = [
        r for r in results
        if r["configuration"] == "parallel"
    ]

    threads = np.array([
        r["threads"] for r in parallel
    ])

    runtimes = np.array([
        r["median_runtime"] for r in parallel
    ])

    speedups = np.array([
        r["speedup_vs_parallel_1"] for r in parallel
    ])

    efficiencies = np.array([
        r["parallel_efficiency"] * 100
        for r in parallel
    ])

    throughput = np.array([
        r["throughput_info_bps"] / 1e6
        for r in parallel
    ])

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        threads, runtimes,
        "o-", label="Parallel Numba"
    )

    plt.axhline(
        results[0]["median_runtime"],
        linestyle="--",
        color="black",
        label="Sequential Numba"
    )

    plt.xlabel("Number of Threads")
    plt.ylabel("Runtime (seconds)")
    plt.title("LDPC Parallel Decoding Runtime")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{prefix}_runtime.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Speedup
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        threads, speedups,
        "o-", label="Measured speedup"
    )

    plt.plot(
        threads, threads,
        "k--", label="Ideal linear speedup"
    )

    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.title("LDPC Parallel Speedup")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{prefix}_speedup.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Parallel efficiency
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        threads, efficiencies,
        "o-"
    )

    plt.xlabel("Number of Threads")
    plt.ylabel("Parallel Efficiency (%)")
    plt.title("LDPC Parallel Efficiency")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    plt.savefig(f"{prefix}_efficiency.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        threads, throughput,
        "o-"
    )

    plt.xlabel("Number of Threads")
    plt.ylabel("Information Throughput (Mbit/s)")
    plt.title("LDPC Decoding Throughput")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    plt.savefig(f"{prefix}_throughput.png", dpi=300)
    plt.close()

    print(f"Saved: {prefix}_runtime.png")
    print(f"Saved: {prefix}_speedup.png")
    print(f"Saved: {prefix}_efficiency.png")
    print(f"Saved: {prefix}_throughput.png")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Benchmark parallel Numba LDPC decoding"
    )

    parser.add_argument("alist_file")

    parser.add_argument(
        "--frames", type=int, default=1000
    )

    parser.add_argument(
        "--ebn0", type=float, default=1.5
    )

    parser.add_argument(
        "--iterations", type=int, default=50
    )

    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8]
    )

    parser.add_argument(
        "--repeats", type=int, default=5
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--output",
        default="results/parallel_benchmark"
    )

    args = parser.parse_args()

    if args.frames <= 0:
        parser.error("--frames must be positive")

    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    if any(t < 1 for t in args.threads):
        parser.error("Thread counts must be positive")

    if max(args.threads) > config.NUMBA_NUM_THREADS:
        parser.error(
            "Requested threads exceed Numba's configured "
            f"maximum of {config.NUMBA_NUM_THREADS}"
        )

    # Ensure a one-thread parallel baseline.
    thread_counts = sorted(
        set([1] + args.threads)
    )

    code = LDPCCode(args.alist_file)

    results = benchmark(
        code,
        frames=args.frames,
        ebn0_db=args.ebn0,
        iteration_limit=args.iterations,
        thread_counts=thread_counts,
        repeats=args.repeats,
        seed=args.seed
    )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_csv(results, f"{output}.csv")
    plot_results(results, str(output))

    print("\nParallel benchmark complete.")


if __name__ == "__main__":
    main()
