
import argparse
import time

import numpy as np

from alist import LDPCCode
from bp_decoder import BPDecoder
from bp_decoder_numba import BPDecoderNumba
from simulation import gf2_rank


def benchmark(code, frames, ebn0_db, iterations, seed):

    rate = (code.N - gf2_rank(code.H)) / code.N

    if rate <= 0:
        raise ValueError("Code rate must be positive")

    sigma2 = 1.0 / (
        2.0 * rate * 10.0 ** (ebn0_db / 10.0)
    )

    rng = np.random.default_rng(seed)

    # Generate all test samples once.
    received = (
        1.0
        + rng.normal(
            0.0,
            np.sqrt(sigma2),
            size=(frames, code.N)
        )
    )

    channel_llrs = 2.0 * received / sigma2

    numpy_decoder = BPDecoder(
        code, max_iterations=iterations
    )

    numba_decoder = BPDecoderNumba(
        code, max_iterations=iterations
    )

    # Warm up JIT compilation outside timing.
    numba_decoder.decode(channel_llrs[0])

    print("\n=== BENCHMARK PARAMETERS ===")
    print(f"Frames: {frames}")
    print(f"Eb/N0: {ebn0_db} dB")
    print(f"Maximum iterations: {iterations}")

    results = {}

    for name, decoder in [
        ("NumPy", numpy_decoder),
        ("Numba", numba_decoder)
    ]:

        start = time.perf_counter()

        decoded_frames = []
        iteration_counts = []
        successes = []

        for llr in channel_llrs:

            decoded, success, count = decoder.decode(llr)

            decoded_frames.append(decoded.copy())
            iteration_counts.append(count)
            successes.append(success)

        elapsed = time.perf_counter() - start

        decoded_frames = np.asarray(decoded_frames)

        bit_errors = int(
            np.count_nonzero(decoded_frames)
        )

        results[name] = {
            "time": elapsed,
            "ber": bit_errors / (frames * code.N),
            "iterations": np.asarray(iteration_counts),
            "successes": np.asarray(successes),
            "decoded": decoded_frames
        }

        print(f"\n{name}:")
        print(f"Runtime: {elapsed:.4f} s")
        print(f"BER: {results[name]['ber']:.8e}")
        print(
            "Average iterations:",
            np.mean(iteration_counts)
        )

    # Compare decoded results.
    mismatched_frames = int(
        np.count_nonzero(
            np.any(
                results["NumPy"]["decoded"]
                != results["Numba"]["decoded"],
                axis=1
            )
        )
    )

    mismatched_iterations = int(
        np.count_nonzero(
            results["NumPy"]["iterations"]
            != results["Numba"]["iterations"]
        )
    )

    mismatched_successes = int(
        np.count_nonzero(
            results["NumPy"]["successes"]
            != results["Numba"]["successes"]
        )
    )

    speedup = (
        results["NumPy"]["time"]
        / results["Numba"]["time"]
    )

    print("\n=== COMPARISON ===")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Mismatched decoded frames: {mismatched_frames}")
    print(f"Mismatched iteration counts: {mismatched_iterations}")
    print(f"Mismatched success flags: {mismatched_successes}")

    if (
        mismatched_frames == 0
        and mismatched_iterations == 0
        and mismatched_successes == 0
    ):
        print("All tested decoder outputs match.")
    else:
        print(
            "Some results differ. Inspect numerical "
            "behavior before using the new decoder."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("alist_file")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--ebn0", type=float, default=1.5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.frames <= 0 or args.iterations <= 0:
        parser.error(
            "--frames and --iterations must be positive"
        )

    code = LDPCCode(args.alist_file)

    benchmark(
        code,
        frames=args.frames,
        ebn0_db=args.ebn0,
        iterations=args.iterations,
        seed=args.seed
    )
