
import argparse
import csv
import time
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from alist import LDPCCode
from bp_decoder import BPDecoder


# ============================================================
# GF(2) MATRIX RANK
# ============================================================

def gf2_rank(H):
    A = np.asarray(H, dtype=np.uint8).copy() % 2
    rows, cols = A.shape
    rank = 0

    for col in range(cols):
        if rank == rows:
            break

        candidates = np.flatnonzero(A[rank:, col])

        if candidates.size == 0:
            continue

        pivot = rank + int(candidates[0])

        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]

        for row in range(rank + 1, rows):
            if A[row, col]:
                A[row, col:] ^= A[rank, col:]

        rank += 1

    return rank


# ============================================================
# STATISTICS
# ============================================================

def theoretical_bpsk_ber(ebn0_db):
    linear = 10.0 ** (ebn0_db / 10.0)
    return 0.5 * erfc(sqrt(linear))


def wilson_interval(errors, trials, z=1.9599639845):
    """Two-sided 95% Wilson interval for frame error rate."""
    if trials == 0:
        return 0.0, 1.0

    p = errors / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials

    center = (p + z2 / (2 * trials)) / denominator

    margin = (
        z / denominator
        * sqrt(p * (1 - p) / trials
               + z2 / (4 * trials * trials))
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def zero_error_upper_bound(frames, confidence=0.95):
    """One-sided upper bound for zero frame errors."""
    return 1.0 - (1.0 - confidence) ** (1.0 / frames)


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def simulate(
    code,
    decoder,
    ebn0_values,
    max_frames,
    min_frame_errors,
    seed,
    progress_interval
):
    rank = gf2_rank(code.H)
    K = code.N - rank
    R = K / code.N

    if K <= 0:
        raise ValueError("Code dimension must be positive.")

    print("\n=== CODE PARAMETERS ===")
    print(f"N = {code.N}")
    print(f"M = {code.M}")
    print(f"K = {K}")
    print(f"Rate = {R:.6f}")
    print(f"GF(2) rank = {rank}")
    print(f"Maximum iterations = {decoder.max_iterations}")

    rng = np.random.default_rng(seed)

    # All-zero codeword maps to all +1 BPSK symbols.
    transmitted = np.zeros(code.N, dtype=np.uint8)
    symbols = np.ones(code.N, dtype=np.float64)

    results = []

    for ebn0_db in ebn0_values:
        start = time.perf_counter()

        ebn0_linear = 10.0 ** (ebn0_db / 10.0)
        sigma2 = 1.0 / (2.0 * R * ebn0_linear)
        sigma = sqrt(sigma2)

        frames = 0
        bit_errors = 0
        frame_errors = 0
        raw_bit_errors = 0
        syndrome_failures = 0
        total_iterations = 0

        for _ in range(max_frames):
            noise = rng.normal(0.0, sigma, code.N)
            received = symbols + noise

            # Hard decisions before decoding.
            raw_decoded = (received < 0).astype(np.uint8)
            raw_bit_errors += int(
                np.count_nonzero(raw_decoded != transmitted)
            )

            # Decode.
            llr = 2.0 * received / sigma2
            decoded, success, iterations = decoder.decode(llr)

            errors = int(np.count_nonzero(decoded != transmitted))

            frames += 1
            bit_errors += errors
            frame_errors += int(errors > 0)
            syndrome_failures += int(not success)
            total_iterations += iterations

            if progress_interval > 0:
                if frames % progress_interval == 0:
                    print(
                        f"  {ebn0_db:.2f} dB: "
                        f"{frames:,} frames, "
                        f"{frame_errors} frame errors",
                        flush=True
                    )

            # Adaptive stopping.
            if (
                min_frame_errors > 0
                and frame_errors >= min_frame_errors
            ):
                break

        elapsed = time.perf_counter() - start

        total_bits = frames * code.N

        ber = bit_errors / total_bits
        fer = frame_errors / frames
        raw_ber = raw_bit_errors / total_bits

        fer_low, fer_high = wilson_interval(
            frame_errors, frames
        )

        # For zero observed frame errors, report a
        # one-sided 95% bound. This also upper-bounds BER.
        if frame_errors == 0:
            zero_bound = zero_error_upper_bound(frames)
        else:
            zero_bound = float("nan")

        result = {
            "ebn0_db": float(ebn0_db),
            "ber": ber,
            "fer": fer,
            "raw_ber": raw_ber,
            "theoretical_ber": theoretical_bpsk_ber(ebn0_db),
            "frames": frames,
            "bits": total_bits,
            "bit_errors": bit_errors,
            "frame_errors": frame_errors,
            "syndrome_failures": syndrome_failures,
            "fer_ci_low": fer_low,
            "fer_ci_high": fer_high,
            "zero_error_upper": zero_bound,
            "avg_iterations": total_iterations / frames,
            "runtime_seconds": elapsed
        }

        results.append(result)

        print(
            f"Eb/N0={ebn0_db:5.2f} dB | "
            f"BER={ber:.6e} | "
            f"FER={fer:.6e} | "
            f"Frames={frames:7d} | "
            f"FrameErrors={frame_errors:5d} | "
            f"AvgIter={total_iterations / frames:5.2f} | "
            f"Time={elapsed:7.2f}s",
            flush=True
        )

    return results


# ============================================================
# CSV EXPORT
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
# PLOTTING
# ============================================================

def plot_results(results, prefix):
    x = np.array([r["ebn0_db"] for r in results])

    ber = np.array([r["ber"] for r in results])
    fer = np.array([r["fer"] for r in results])
    raw = np.array([r["raw_ber"] for r in results])
    theory = np.array([r["theoretical_ber"] for r in results])

    iterations = np.array([
        r["avg_iterations"] for r in results
    ])

    frames = np.array([r["frames"] for r in results])

    zero_bounds = np.array([
        r["zero_error_upper"] for r in results
    ])

    # --------------------------------------------------------
    # BER
    # --------------------------------------------------------

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
        label="LDPC BP decoded BER"
    )

    # Conservative upper bounds for zero-error points.
    zero_mask = (ber == 0)

    if np.any(zero_mask):
        plt.semilogy(
            x[zero_mask],
            zero_bounds[zero_mask],
            "v",
            color="tab:orange",
            label="Zero-error upper bound (conservative)"
        )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("LDPC BER vs Eb/N0")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_ber.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # FER WITH CONFIDENCE INTERVAL
    # --------------------------------------------------------

    low = np.array([r["fer_ci_low"] for r in results])
    high = np.array([r["fer_ci_high"] for r in results])

    plt.figure(figsize=(9, 6))

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
            label="95% Wilson interval"
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
    plt.title("LDPC FER vs Eb/N0")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_fer.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # AVERAGE ITERATIONS
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(x, iterations, "o-")

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Average BP Iterations")
    plt.title("BP Decoder Convergence")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(f"{prefix}_iterations.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # SIMULATED FRAMES
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(x, frames, "s-")

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Frames Simulated")
    plt.title("Adaptive Monte Carlo Sample Size")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(f"{prefix}_frames.png", dpi=300)
    plt.close()

    print(f"Saved: {prefix}_ber.png")
    print(f"Saved: {prefix}_fer.png")
    print(f"Saved: {prefix}_iterations.png")
    print(f"Saved: {prefix}_frames.png")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive LDPC BER/FER simulation"
    )

    parser.add_argument("alist_file")

    parser.add_argument("--ebn0-start", type=float, default=0.0)
    parser.add_argument("--ebn0-stop", type=float, default=3.0)
    parser.add_argument("--ebn0-step", type=float, default=0.25)

    parser.add_argument("--max-frames", type=int, default=10000)
    parser.add_argument("--min-frame-errors", type=int, default=100)

    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--progress-interval", type=int, default=1000)

    parser.add_argument(
        "--output",
        default="ldpc_adaptive"
    )

    args = parser.parse_args()

    if args.ebn0_step <= 0:
        parser.error("--ebn0-step must be positive")

    if args.ebn0_stop < args.ebn0_start:
        parser.error("--ebn0-stop must be >= --ebn0-start")

    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    if args.min_frame_errors < 0:
        parser.error("--min-frame-errors cannot be negative")

    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    code = LDPCCode(args.alist_file)

    decoder = BPDecoder(
        code,
        max_iterations=args.iterations
    )

    ebn0_values = np.arange(
        args.ebn0_start,
        args.ebn0_stop + args.ebn0_step * 0.5,
        args.ebn0_step
    )

    results = simulate(
        code,
        decoder,
        ebn0_values,
        max_frames=args.max_frames,
        min_frame_errors=args.min_frame_errors,
        seed=args.seed,
        progress_interval=args.progress_interval
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    save_csv(results, f"{output}.csv")
    plot_results(results, str(output))

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
