
import argparse
import csv
import time
from math import erfc, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import config, set_num_threads, get_num_threads

from alist import LDPCCode
from bp_decoder_numba import BPDecoderNumba
from simulation import gf2_rank


# ============================================================
# 1. STATISTICAL FUNCTIONS
# ============================================================

def theoretical_bpsk_ber(ebn0_db):
    linear = 10.0 ** (ebn0_db / 10.0)
    return 0.5 * erfc(sqrt(linear))


def wilson_interval(errors, trials, z=1.9599639845):
    """Approximate two-sided 95% FER confidence interval."""
    if trials == 0:
        return 0.0, 1.0

    p = errors / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials

    center = (p + z2 / (2 * trials)) / denominator

    margin = z / denominator * sqrt(
        p * (1.0 - p) / trials
        + z2 / (4.0 * trials * trials)
    )

    return (
        max(0.0, center - margin),
        min(1.0, center + margin)
    )


def zero_error_upper_bound(frames, confidence=0.95):
    """
    Exact one-sided upper bound for zero frame errors.

    Also conservatively upper-bounds BER.
    """
    return 1.0 - (1.0 - confidence) ** (1.0 / frames)


# ============================================================
# 2. PARALLEL MONTE CARLO SIMULATION
# ============================================================

def simulate_parallel(
    code,
    decoder,
    ebn0_values,
    max_frames=20000,
    min_frame_errors=100,
    batch_size=1000,
    seed=42,
    progress_interval=5000
):
    """
    Adaptive Monte Carlo simulation using Numba
    parallel batch decoding.

    The all-zero codeword is transmitted through
    a BPSK-AWGN channel.

    Each batch is decoded in parallel. If the target
    number of frame errors is reached inside a batch,
    only the prefix through the target error is counted.

    Runtime includes channel generation, decoding,
    and statistics, but excludes initial JIT compilation.
    """

    rank = gf2_rank(code.H)
    K = code.N - rank
    R = K / code.N

    if K <= 0:
        raise ValueError("Code dimension must be positive")

    if max_frames <= 0 or batch_size <= 0:
        raise ValueError("Frame and batch counts must be positive")

    print("\n=== LDPC CODE PARAMETERS ===")
    print(f"Codeword length N: {code.N}")
    print(f"Parity checks M: {code.M}")
    print(f"GF(2) rank: {rank}")
    print(f"Information bits K: {K}")
    print(f"Code rate R: {R:.6f}")
    print(f"Maximum BP iterations: {decoder.max_iterations}")
    print(f"Numba threads: {get_num_threads()}")
    print(f"Batch size: {batch_size:,}")
    print(f"Maximum frames: {max_frames:,}")
    print(f"Target frame errors: {min_frame_errors}")

    # Independent, reproducible RNG streams for SNR points.
    seeds = np.random.SeedSequence(seed).spawn(
        len(ebn0_values)
    )

    results = []

    for point_index, ebn0_db in enumerate(ebn0_values):

        rng = np.random.default_rng(seeds[point_index])

        ebn0_linear = 10.0 ** (ebn0_db / 10.0)
        sigma2 = 1.0 / (2.0 * R * ebn0_linear)
        sigma = sqrt(sigma2)

        frames = 0
        processed_frames = 0
        bit_errors = 0
        frame_errors = 0
        raw_bit_errors = 0
        syndrome_failures = 0
        total_iterations = 0

        print(
            f"\n=== Eb/N0 = {ebn0_db:.2f} dB ===",
            flush=True
        )

        start_time = time.perf_counter()

        while frames < max_frames:

            current_batch = min(
                batch_size,
                max_frames - frames
            )

            # ----------------------------------------
            # A. Generate noisy BPSK codewords
            # ----------------------------------------

            received = 1.0 + rng.normal(
                0.0,
                sigma,
                size=(current_batch, code.N)
            )

            channel_llrs = np.ascontiguousarray(
                2.0 * received / sigma2
            )

            # ----------------------------------------
            # B. Decode entire batch in parallel
            # ----------------------------------------

            decoded, successes, iterations = (
                decoder.decode_batch(channel_llrs)
            )

            processed_frames += current_batch

            # ----------------------------------------
            # C. Calculate per-frame error counts
            # ----------------------------------------

            # All-zero codeword: decoded ones are errors.
            frame_bit_errors = np.count_nonzero(
                decoded,
                axis=1
            )

            frame_error_flags = frame_bit_errors > 0

            raw_errors = np.count_nonzero(
                received < 0,
                axis=1
            )

            # ----------------------------------------
            # D. Adaptive stopping inside the batch
            # ----------------------------------------

            used = current_batch

            if min_frame_errors > 0:

                remaining_errors = (
                    min_frame_errors - frame_errors
                )

                error_positions = np.flatnonzero(
                    frame_error_flags
                )

                if len(error_positions) >= remaining_errors:
                    used = int(
                        error_positions[remaining_errors - 1] + 1
                    )

            # Count only the selected prefix.
            frames += used

            bit_errors += int(
                np.sum(frame_bit_errors[:used])
            )

            frame_errors += int(
                np.count_nonzero(frame_error_flags[:used])
            )

            raw_bit_errors += int(
                np.sum(raw_errors[:used])
            )

            syndrome_failures += int(
                np.count_nonzero(~successes[:used])
            )

            total_iterations += int(
                np.sum(iterations[:used])
            )

            # ----------------------------------------
            # E. Progress reporting
            # ----------------------------------------

            if progress_interval > 0:
                if (
                    frames // progress_interval
                    != (frames - used) // progress_interval
                ):
                    print(
                        f"  {frames:,} counted frames | "
                        f"{frame_errors} frame errors | "
                        f"{processed_frames:,} processed",
                        flush=True
                    )

            if (
                min_frame_errors > 0
                and frame_errors >= min_frame_errors
            ):
                break

        elapsed = time.perf_counter() - start_time

        # --------------------------------------------
        # F. Final statistics
        # --------------------------------------------

        total_bits = frames * code.N

        ber = bit_errors / total_bits
        fer = frame_errors / frames
        raw_ber = raw_bit_errors / total_bits

        avg_iterations = total_iterations / frames

        fer_low, fer_high = wilson_interval(
            frame_errors, frames
        )

        if frame_errors == 0:
            upper_bound = zero_error_upper_bound(frames)
        else:
            upper_bound = float("nan")

        result = {
            "ebn0_db": float(ebn0_db),
            "ber": float(ber),
            "fer": float(fer),
            "raw_ber": float(raw_ber),
            "theoretical_ber": theoretical_bpsk_ber(ebn0_db),
            "frames": frames,
            "processed_frames": processed_frames,
            "bits": total_bits,
            "bit_errors": bit_errors,
            "frame_errors": frame_errors,
            "syndrome_failures": syndrome_failures,
            "fer_ci_low": fer_low,
            "fer_ci_high": fer_high,
            "zero_error_upper": upper_bound,
            "avg_iterations": float(avg_iterations),
            "runtime_seconds": elapsed,
            "throughput_info_bps": (
                processed_frames * K / elapsed
            ),
            "threads": get_num_threads(),
            "batch_size": batch_size
        }

        results.append(result)

        print(
            f"Eb/N0={ebn0_db:5.2f} dB | "
            f"BER={ber:.6e} | "
            f"FER={fer:.6e} | "
            f"Frames={frames:7d} | "
            f"Errors={frame_errors:5d} | "
            f"AvgIter={avg_iterations:5.2f} | "
            f"Time={elapsed:7.2f}s",
            flush=True
        )

    return results


# ============================================================
# 3. SAVE RESULTS
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
# 4. GENERATE PLOTS
# ============================================================

def plot_results(results, prefix):

    x = np.array([r["ebn0_db"] for r in results])

    ber = np.array([r["ber"] for r in results])
    fer = np.array([r["fer"] for r in results])
    raw = np.array([r["raw_ber"] for r in results])
    theory = np.array([
        r["theoretical_ber"] for r in results
    ])

    iterations = np.array([
        r["avg_iterations"] for r in results
    ])

    frames = np.array([
        r["frames"] for r in results
    ])

    runtimes = np.array([
        r["runtime_seconds"] for r in results
    ])

    zero_bounds = np.array([
        r["zero_error_upper"] for r in results
    ])

    # --------------------------------------------
    # BER
    # --------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.semilogy(
        x, theory, "k--",
        label="Theoretical uncoded BPSK"
    )

    plt.semilogy(
        x,
        np.where(raw > 0, raw, np.nan),
        "s-",
        label="Hard decisions before BP"
    )

    plt.semilogy(
        x,
        np.where(ber > 0, ber, np.nan),
        "o-",
        label="Parallel Numba BP"
    )

    zero_mask = (ber == 0)

    if np.any(zero_mask):
        plt.semilogy(
            x[zero_mask],
            zero_bounds[zero_mask],
            "v",
            label="Zero-error upper bound"
        )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("Parallel LDPC BER vs Eb/N0")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{prefix}_ber.png", dpi=300)
    plt.close()

    # --------------------------------------------
    # FER
    # --------------------------------------------

    plt.figure(figsize=(9, 6))

    low = np.array([
        r["fer_ci_low"] for r in results
    ])

    high = np.array([
        r["fer_ci_high"] for r in results
    ])

    positive = fer > 0

    plt.semilogy(
        x,
        np.where(positive, fer, np.nan),
        "o-",
        label="Measured FER"
    )

    if np.any(positive):
        plt.fill_between(
            x,
            np.maximum(low, 1e-12),
            np.maximum(high, 1e-12),
            where=positive,
            alpha=0.2,
            label="Approx. 95% Wilson interval"
        )

    if np.any(~positive):
        plt.semilogy(
            x[~positive],
            zero_bounds[~positive],
            "v",
            label="Zero-error 95% upper bound"
        )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Frame Error Rate")
    plt.title("Parallel LDPC FER vs Eb/N0")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{prefix}_fer.png", dpi=300)
    plt.close()

    # --------------------------------------------
    # Average iterations
    # --------------------------------------------

    plt.figure(figsize=(9, 6))
    plt.plot(x, iterations, "o-")

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Average BP Iterations")
    plt.title("Parallel BP Decoder Convergence")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    plt.savefig(f"{prefix}_iterations.png", dpi=300)
    plt.close()

    # --------------------------------------------
    # Number of frames
    # --------------------------------------------

    plt.figure(figsize=(9, 6))
    plt.plot(x, frames, "s-")

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Counted Frames")
    plt.title("Adaptive Monte Carlo Frame Counts")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    plt.savefig(f"{prefix}_frames.png", dpi=300)
    plt.close()

    # --------------------------------------------
    # Runtime
    # --------------------------------------------

    plt.figure(figsize=(9, 6))
    plt.plot(x, runtimes, "o-")

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Parallel LDPC Simulation Runtime")
    plt.grid(True, linestyle=":")
    plt.tight_layout()

    plt.savefig(f"{prefix}_runtime.png", dpi=300)
    plt.close()

    print("\nPlots saved:")
    for suffix in [
        "ber", "fer", "iterations", "frames", "runtime"
    ]:
        print(f"  {prefix}_{suffix}.png")


# ============================================================
# 5. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Parallel Numba LDPC Monte Carlo simulation"
    )

    parser.add_argument("alist_file")

    parser.add_argument("--ebn0-start", type=float, default=1.0)
    parser.add_argument("--ebn0-stop", type=float, default=3.0)
    parser.add_argument("--ebn0-step", type=float, default=0.25)

    parser.add_argument("--max-frames", type=int, default=20000)
    parser.add_argument("--min-frame-errors", type=int, default=100)

    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-interval", type=int, default=5000)

    parser.add_argument(
        "--output",
        default="results/parallel_adaptive"
    )

    args = parser.parse_args()

    if args.ebn0_step <= 0:
        parser.error("--ebn0-step must be positive")

    if args.ebn0_stop < args.ebn0_start:
        parser.error("--ebn0-stop must be >= --ebn0-start")

    if args.max_frames <= 0 or args.batch_size <= 0:
        parser.error("Frame and batch counts must be positive")

    if args.min_frame_errors < 0:
        parser.error("--min-frame-errors cannot be negative")

    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    if not 1 <= args.threads <= config.NUMBA_NUM_THREADS:
        parser.error(
            f"--threads must be between 1 and "
            f"{config.NUMBA_NUM_THREADS}"
        )

    code = LDPCCode(args.alist_file)

    decoder = BPDecoderNumba(
        code,
        max_iterations=args.iterations
    )

    set_num_threads(args.threads)

    # Warm up JIT compilation outside simulation timing.
    print("\nWarming up Numba kernels...")

    warmup_llrs = np.ones(
        (max(2, args.threads), code.N),
        dtype=np.float64
    )

    # Include a nonzero syndrome to exercise BP updates.
    warmup_llrs[:, 3] = -0.5

    decoder.decode(warmup_llrs[0])
    decoder.decode_batch(warmup_llrs)

    ebn0_values = np.arange(
        args.ebn0_start,
        args.ebn0_stop + 0.5 * args.ebn0_step,
        args.ebn0_step
    )

    results = simulate_parallel(
        code=code,
        decoder=decoder,
        ebn0_values=ebn0_values,
        max_frames=args.max_frames,
        min_frame_errors=args.min_frame_errors,
        batch_size=args.batch_size,
        seed=args.seed,
        progress_interval=args.progress_interval
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    save_csv(results, f"{output}.csv")
    plot_results(results, str(output))

    print("\nParallel simulation complete.")


if __name__ == "__main__":
    main()
