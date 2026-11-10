
import argparse
import csv
import time

import numpy as np
import matplotlib.pyplot as plt

from alist import LDPCCode
from bp_decoder import BPDecoder


# ============================================================
# 1. BINARY MATRIX RANK OVER GF(2)
# ============================================================

def gf2_rank(H):
    """
    Calculate the rank of a binary matrix over GF(2).

    Uses Gaussian elimination with XOR operations.
    """
    A = np.asarray(H, dtype=np.uint8).copy() % 2

    rows, cols = A.shape
    rank = 0

    for col in range(cols):

        # Find a pivot in this column.
        candidates = np.flatnonzero(A[rank:, col])

        if candidates.size == 0:
            continue

        pivot = rank + candidates[0]

        # Swap pivot row into position.
        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]

        # Eliminate below pivot using XOR.
        for row in range(rank + 1, rows):
            if A[row, col]:
                A[row, col:] ^= A[rank, col:]

        rank += 1

        if rank == rows:
            break

    return rank


# ============================================================
# 2. THEORETICAL UNCODED BPSK BER
# ============================================================

def theoretical_bpsk_ber(ebn0_db):
    """
    Theoretical uncoded BPSK BER over AWGN.

    Pb = Q(sqrt(2 * Eb/N0))
       = 0.5 * erfc(sqrt(Eb/N0))
    """
    from math import erfc, sqrt

    ebn0_linear = 10.0 ** (ebn0_db / 10.0)

    return 0.5 * erfc(sqrt(ebn0_linear))


# ============================================================
# 3. MONTE CARLO LDPC SIMULATION
# ============================================================

def simulate_ldpc(
    code,
    decoder,
    ebn0_values,
    frames_per_point=1000,
    min_frame_errors=50,
    seed=42
):
    """
    Simulate LDPC decoding over a BPSK-AWGN channel.

    Parameters
    ----------
    code:
        LDPCCode object.

    decoder:
        BPDecoder object.

    ebn0_values:
        Array of Eb/N0 values in dB.

    frames_per_point:
        Maximum frames simulated at each Eb/N0.

    min_frame_errors:
        Stop early after observing this many frame errors.
        Set to 0 to disable early stopping.

    seed:
        Random seed for reproducibility.
    """

    # Compute actual code dimension and rate.
    rank = gf2_rank(code.H)

    K = code.N - rank
    R = K / code.N

    if K <= 0:
        raise ValueError(
            "The parity-check matrix has zero code dimension. "
            "Choose a positive-rate LDPC code for BER simulation."
        )

    print("\n=== LDPC CODE PARAMETERS ===")
    print(f"Codeword length N: {code.N}")
    print(f"Parity-check rows M: {code.M}")
    print(f"GF(2) rank: {rank}")
    print(f"Information bits K: {K}")
    print(f"Code rate R: {R:.6f}")
    print(f"BP maximum iterations: {decoder.max_iterations}")

    rng = np.random.default_rng(seed)

    # Transmit the all-zero codeword.
    transmitted = np.zeros(code.N, dtype=np.uint8)

    # BPSK: zero maps to +1.
    symbols = np.ones(code.N, dtype=np.float64)

    results = []

    for ebn0_db in ebn0_values:

        start_time = time.perf_counter()

        ebn0_linear = 10.0 ** (ebn0_db / 10.0)

        # AWGN noise variance accounting for code rate.
        sigma2 = 1.0 / (2.0 * R * ebn0_linear)
        sigma = np.sqrt(sigma2)

        total_bit_errors = 0
        total_uncoded_errors = 0
        total_frame_errors = 0
        total_iterations = 0
        total_frames = 0
        syndrome_failures = 0

        for frame in range(frames_per_point):

            # --------------------------------------------
            # Generate AWGN channel observations
            # --------------------------------------------

            noise = rng.normal(
                loc=0.0,
                scale=sigma,
                size=code.N
            )

            received = symbols + noise

            # --------------------------------------------
            # Hard decisions before decoding
            # --------------------------------------------

            uncoded_bits = (
                received < 0
            ).astype(np.uint8)

            uncoded_errors = int(
                np.count_nonzero(
                    uncoded_bits != transmitted
                )
            )

            total_uncoded_errors += uncoded_errors

            # --------------------------------------------
            # Channel LLR calculation
            # --------------------------------------------

            channel_llr = 2.0 * received / sigma2

            # --------------------------------------------
            # Belief propagation decoding
            # --------------------------------------------

            decoded, success, iterations = decoder.decode(
                channel_llr
            )

            # --------------------------------------------
            # Count decoding errors
            # --------------------------------------------

            bit_errors = int(
                np.count_nonzero(
                    decoded != transmitted
                )
            )

            total_bit_errors += bit_errors
            total_iterations += iterations
            total_frames += 1

            if bit_errors > 0:
                total_frame_errors += 1

            if not success:
                syndrome_failures += 1

            # --------------------------------------------
            # Optional early stopping
            # --------------------------------------------

            if (
                min_frame_errors > 0
                and total_frame_errors >= min_frame_errors
            ):
                break

        # --------------------------------------------
        # Compute statistics
        # --------------------------------------------

        total_bits = total_frames * code.N

        ber = total_bit_errors / total_bits

        raw_ber = total_uncoded_errors / total_bits

        fer = total_frame_errors / total_frames

        avg_iterations = (
            total_iterations / total_frames
        )

        elapsed = time.perf_counter() - start_time

        theoretical_ber = theoretical_bpsk_ber(
            ebn0_db
        )

        result = {
            "ebn0_db": float(ebn0_db),
            "ber": float(ber),
            "raw_ber": float(raw_ber),
            "fer": float(fer),
            "theoretical_ber": float(theoretical_ber),
            "frames": total_frames,
            "bit_errors": total_bit_errors,
            "frame_errors": total_frame_errors,
            "syndrome_failures": syndrome_failures,
            "avg_iterations": float(avg_iterations),
            "runtime_seconds": float(elapsed)
        }

        results.append(result)

        print(
            f"Eb/N0={ebn0_db:5.2f} dB | "
            f"BER={ber:.6e} | "
            f"FER={fer:.6e} | "
            f"Frames={total_frames:6d} | "
            f"BitErrors={total_bit_errors:6d} | "
            f"AvgIter={avg_iterations:6.2f} | "
            f"Time={elapsed:7.2f}s"
        )

    return results


# ============================================================
# 4. SAVE RESULTS TO CSV
# ============================================================

def save_results(results, filename):

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(filename, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {filename}")


# ============================================================
# 5. PLOT BER AND FER
# ============================================================

def plot_results(results, output_prefix="ldpc"):

    ebn0 = np.array([
        r["ebn0_db"] for r in results
    ])

    ber = np.array([
        r["ber"] for r in results
    ])

    raw_ber = np.array([
        r["raw_ber"] for r in results
    ])

    fer = np.array([
        r["fer"] for r in results
    ])

    theoretical = np.array([
        r["theoretical_ber"] for r in results
    ])

    # --------------------------------------------
    # BER plot
    # --------------------------------------------

    plt.figure(figsize=(9, 6))

    # Zero-error observations are not plotted on a
    # logarithmic axis. Their upper-bound markers
    # are shown separately below.
    positive_ber = np.where(ber > 0, ber, np.nan)
    positive_raw = np.where(raw_ber > 0, raw_ber, np.nan)

    plt.semilogy(
        ebn0,
        theoretical,
        "k--",
        label="Theoretical uncoded BPSK"
    )

    plt.semilogy(
        ebn0,
        positive_raw,
        "s-",
        label="Hard decisions before BP"
    )

    plt.semilogy(
        ebn0,
        positive_ber,
        "o-",
        label="LDPC BP decoded BER"
    )

    # If no errors were observed, plot an approximate
    # 95% upper confidence bound (rule of three).
    for i, r in enumerate(results):

        if r["bit_errors"] == 0:
            upper = min(
                1.0,
                3.0 / (r["frames"] * code_length_for_plot)
            )

            plt.semilogy(
                ebn0[i],
                upper,
                "v",
                color="tab:blue",
                markersize=7
            )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("LDPC Belief Propagation: BER vs Eb/N0")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()

    ber_file = f"{output_prefix}_ber.png"
    plt.savefig(ber_file, dpi=300)
    plt.close()

    # --------------------------------------------
    # FER plot
    # --------------------------------------------

    plt.figure(figsize=(9, 6))

    positive_fer = np.where(fer > 0, fer, np.nan)

    plt.semilogy(
        ebn0,
        positive_fer,
        "o-",
        label="LDPC frame error rate"
    )

    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Frame Error Rate (FER)")
    plt.title("LDPC Belief Propagation: FER vs Eb/N0")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()

    fer_file = f"{output_prefix}_fer.png"
    plt.savefig(fer_file, dpi=300)
    plt.close()

    print(f"BER plot saved to: {ber_file}")
    print(f"FER plot saved to: {fer_file}")


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="LDPC BER simulation using BP decoding"
    )

    parser.add_argument(
        "alist_file",
        help="Path to LDPC AList file"
    )

    parser.add_argument(
        "--ebn0-start",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--ebn0-stop",
        type=float,
        default=5.0
    )

    parser.add_argument(
        "--ebn0-step",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--min-frame-errors",
        type=int,
        default=50
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--output",
        type=str,
        default="ldpc_results"
    )

    args = parser.parse_args()

    if args.ebn0_step <= 0:
        parser.error("--ebn0-step must be positive")

    if args.ebn0_stop < args.ebn0_start:
        parser.error("--ebn0-stop must be >= --ebn0-start")

    if args.frames <= 0:
        parser.error("--frames must be positive")

    if args.min_frame_errors < 0:
        parser.error("--min-frame-errors cannot be negative")

    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    # Load code
    code = LDPCCode(args.alist_file)

    # Create BP decoder
    decoder = BPDecoder(
        code,
        max_iterations=args.iterations
    )

    # Eb/N0 sweep
    ebn0_values = np.arange(
        args.ebn0_start,
        args.ebn0_stop + args.ebn0_step * 0.5,
        args.ebn0_step
    )

    # Run Monte Carlo simulation
    results = simulate_ldpc(
        code=code,
        decoder=decoder,
        ebn0_values=ebn0_values,
        frames_per_point=args.frames,
        min_frame_errors=args.min_frame_errors,
        seed=args.seed
    )

    # Save results
    save_results(
        results,
        f"{args.output}.csv"
    )

    # Provide code length for zero-error bounds
    code_length_for_plot = code.N

    # Generate figures
    plot_results(
        results,
        output_prefix=args.output
    )

    print("\nSimulation complete.")
