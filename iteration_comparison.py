
import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alist import LDPCCode
from bp_decoder import BPDecoder
from simulation import gf2_rank


def compare_iterations(
    code,
    ebn0_values,
    iteration_limits,
    frames_per_point,
    seed=42,
    progress_interval=1000
):
    """
    Compare BP decoders with different maximum iteration counts.

    Every decoder receives identical channel observations.
    """

    rank = gf2_rank(code.H)
    K = code.N - rank
    rate = K / code.N

    if K <= 0:
        raise ValueError("Code must have a positive dimension.")

    iteration_limits = sorted(set(iteration_limits))

    decoders = {
        limit: BPDecoder(code, max_iterations=limit)
        for limit in iteration_limits
    }

    rng = np.random.default_rng(seed)

    print("\n=== LDPC CODE ===")
    print(f"N = {code.N}")
    print(f"K = {K}")
    print(f"Rate = {rate:.6f}")
    print(f"GF(2) rank = {rank}")
    print(f"Iteration limits = {iteration_limits}")
    print(f"Frames per SNR = {frames_per_point}")

    results = []

    for ebn0_db in ebn0_values:

        ebn0_linear = 10.0 ** (ebn0_db / 10.0)
        sigma2 = 1.0 / (2.0 * rate * ebn0_linear)
        sigma = np.sqrt(sigma2)

        stats = {}

        for limit in iteration_limits:
            stats[limit] = {
                "bit_errors": 0,
                "frame_errors": 0,
                "syndrome_failures": 0,
                "total_iterations": 0,
                "runtime_seconds": 0.0
            }

        print(f"\n=== Eb/N0 = {ebn0_db:.2f} dB ===")

        # All-zero codeword maps to all +1 symbols.
        for frame in range(frames_per_point):

            # Generate one shared noisy codeword.
            received = 1.0 + rng.normal(
                0.0, sigma, size=code.N
            )

            channel_llr = 2.0 * received / sigma2

            # Decode the SAME LLRs with every limit.
            for limit in iteration_limits:

                start = time.perf_counter()

                decoded, success, iterations = (
                    decoders[limit].decode(channel_llr)
                )

                elapsed = time.perf_counter() - start

                # All-zero transmitted codeword:
                # each decoded 1 is a bit error.
                errors = int(np.count_nonzero(decoded))

                stats[limit]["bit_errors"] += errors
                stats[limit]["frame_errors"] += int(errors > 0)
                stats[limit]["syndrome_failures"] += int(not success)
                stats[limit]["total_iterations"] += iterations
                stats[limit]["runtime_seconds"] += elapsed

            if (
                progress_interval > 0
                and (frame + 1) % progress_interval == 0
            ):
                print(
                    f"  Completed {frame + 1:,} / "
                    f"{frames_per_point:,} frames",
                    flush=True
                )

        total_bits = frames_per_point * code.N

        for limit in iteration_limits:

            s = stats[limit]

            result = {
                "ebn0_db": float(ebn0_db),
                "max_iterations": int(limit),
                "frames": frames_per_point,
                "bit_errors": s["bit_errors"],
                "frame_errors": s["frame_errors"],
                "syndrome_failures": s["syndrome_failures"],
                "ber": s["bit_errors"] / total_bits,
                "fer": s["frame_errors"] / frames_per_point,
                "avg_iterations": (
                    s["total_iterations"] / frames_per_point
                ),
                "runtime_seconds": s["runtime_seconds"],
                "throughput_bits_per_second": (
                    total_bits / s["runtime_seconds"]
                    if s["runtime_seconds"] > 0 else float("nan")
                )
            }

            results.append(result)

            print(
                f"MaxIter={limit:3d} | "
                f"BER={result['ber']:.6e} | "
                f"FER={result['fer']:.6e} | "
                f"AvgIter={result['avg_iterations']:6.2f} | "
                f"Time={result['runtime_seconds']:7.2f}s"
            )

    return results


def save_csv(results, filename):

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {filename}")


def plot_comparison(results, iteration_limits, prefix):

    limits = sorted(set(iteration_limits))

    metrics = [
        ("ber", "Bit Error Rate", True),
        ("fer", "Frame Error Rate", True),
        ("avg_iterations", "Average BP Iterations", False),
        ("runtime_seconds", "Decoding Runtime (seconds)", False)
    ]

    for metric, ylabel, logarithmic in metrics:

        plt.figure(figsize=(9, 6))

        for limit in limits:

            subset = sorted(
                [
                    r for r in results
                    if r["max_iterations"] == limit
                ],
                key=lambda r: r["ebn0_db"]
            )

            x = np.array([
                r["ebn0_db"] for r in subset
            ])

            y = np.array([
                r[metric] for r in subset
            ])

            if logarithmic:
                y = np.where(y > 0, y, np.nan)
                plt.semilogy(
                    x, y, "o-",
                    label=f"{limit} iterations"
                )
            else:
                plt.plot(
                    x, y, "o-",
                    label=f"{limit} iterations"
                )

        plt.xlabel("Eb/N0 (dB)")
        plt.ylabel(ylabel)

        plt.title(
            f"LDPC BP Iteration Comparison: {ylabel}"
        )

        plt.grid(True, which="both", linestyle=":")
        plt.legend()
        plt.tight_layout()

        filename = f"{prefix}_{metric}.png"

        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Saved: {filename}")


def main():

    parser = argparse.ArgumentParser(
        description="Compare LDPC BP iteration limits"
    )

    parser.add_argument("alist_file")

    parser.add_argument(
        "--limits",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100]
    )

    parser.add_argument(
        "--ebn0-start",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--ebn0-stop",
        type=float,
        default=3.0
    )

    parser.add_argument(
        "--ebn0-step",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--output",
        default="results/iteration_comparison"
    )

    args = parser.parse_args()

    if args.frames <= 0:
        parser.error("--frames must be positive")

    if args.ebn0_step <= 0:
        parser.error("--ebn0-step must be positive")

    if args.ebn0_stop < args.ebn0_start:
        parser.error("--ebn0-stop must be >= --ebn0-start")

    if any(limit <= 0 for limit in args.limits):
        parser.error("All iteration limits must be positive")

    code = LDPCCode(args.alist_file)

    ebn0_values = np.arange(
        args.ebn0_start,
        args.ebn0_stop + 0.5 * args.ebn0_step,
        args.ebn0_step
    )

    results = compare_iterations(
        code=code,
        ebn0_values=ebn0_values,
        iteration_limits=args.limits,
        frames_per_point=args.frames,
        seed=args.seed,
        progress_interval=args.progress_interval
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    save_csv(results, f"{output}.csv")

    plot_comparison(
        results,
        args.limits,
        str(output)
    )

    print("\nIteration comparison complete.")


if __name__ == "__main__":
    main()
