
import math
import sys

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads


# ============================================================
# 1. NUMERICALLY STABLE PHI FUNCTION
# ============================================================

@njit(cache=True)
def phi(x):
    """
    Phi function used by the log-domain sum-product algorithm.

        phi(x) = -log(tanh(x / 2))

    The equivalent expression below is more numerically stable.
    """

    x = max(x, 1e-12)

    return (
        math.log1p(math.exp(-x))
        - math.log(-math.expm1(-x))
    )


# ============================================================
# 2. SEQUENTIAL JIT-COMPILED BP DECODER
# ============================================================

@njit(cache=True)
def bp_decode_kernel(
    channel_llr,
    check_ptr,
    edge_variables,
    max_iterations,
    llr_clip
):
    """
    Decode one binary LDPC codeword using the
    log-domain sum-product algorithm.

    Uses a flooding message-passing schedule.

    Returns:
        decoded
        success
        iterations
        unsatisfied_history
        mean_llr_history
    """

    N = channel_llr.size
    M = check_ptr.size - 1
    E = edge_variables.size

    # --------------------------------------------
    # Initialize channel LLRs
    # --------------------------------------------

    llr = np.empty(N, dtype=np.float64)

    for i in range(N):
        llr[i] = min(
            max(channel_llr[i], -llr_clip),
            llr_clip
        )

    # --------------------------------------------
    # Initialize messages
    # --------------------------------------------

    v_to_c = np.empty(E, dtype=np.float64)
    c_to_v = np.zeros(E, dtype=np.float64)

    posterior = np.empty(N, dtype=np.float64)
    decoded = np.zeros(N, dtype=np.uint8)

    for e in range(E):
        v_to_c[e] = llr[edge_variables[e]]

    for i in range(N):
        posterior[i] = llr[i]
        decoded[i] = 1 if llr[i] < 0 else 0

    unsatisfied_history = np.zeros(
        max_iterations, dtype=np.int64
    )

    mean_llr_history = np.zeros(
        max_iterations, dtype=np.float64
    )

    # --------------------------------------------
    # Initial syndrome check
    # --------------------------------------------

    initial_valid = True

    for check in range(M):

        parity = 0

        for e in range(
            check_ptr[check],
            check_ptr[check + 1]
        ):
            parity ^= int(decoded[edge_variables[e]])

        if parity != 0:
            initial_valid = False
            break

    if initial_valid:
        return (
            decoded,
            True,
            0,
            unsatisfied_history,
            mean_llr_history
        )

    # ============================================
    # BELIEF PROPAGATION ITERATIONS
    # ============================================

    for iteration in range(max_iterations):

        # ========================================
        # A. CHECK-NODE UPDATE
        # ========================================

        for check in range(M):

            start = check_ptr[check]
            end = check_ptr[check + 1]

            degree = end - start

            if degree == 0:
                continue

            if degree == 1:
                c_to_v[start] = llr_clip
                continue

            total_phi = 0.0
            negative_count = 0
            zero_count = 0
            zero_edge = -1

            # Calculate shared check-node information.
            for e in range(start, end):

                value = v_to_c[e]

                if value < 0:
                    negative_count += 1

                if value == 0:
                    zero_count += 1
                    zero_edge = e
                else:
                    total_phi += phi(abs(value))

            # Two or more zero-valued inputs.
            if zero_count >= 2:

                for e in range(start, end):
                    c_to_v[e] = 0.0

                continue

            # Exactly one zero-valued input.
            if zero_count == 1:

                for e in range(start, end):
                    c_to_v[e] = 0.0

                magnitude = phi(
                    max(total_phi, 1e-12)
                )

                sign = (
                    -1.0
                    if negative_count % 2
                    else 1.0
                )

                message = sign * magnitude

                c_to_v[zero_edge] = min(
                    max(message, -llr_clip),
                    llr_clip
                )

                continue

            # No zero-valued inputs.
            for e in range(start, end):

                value = v_to_c[e]

                outgoing_phi = max(
                    total_phi - phi(abs(value)),
                    1e-12
                )

                magnitude = phi(outgoing_phi)

                negatives = negative_count

                if value < 0:
                    negatives -= 1

                if negatives % 2:
                    magnitude = -magnitude

                c_to_v[e] = min(
                    max(magnitude, -llr_clip),
                    llr_clip
                )

        # ========================================
        # B. POSTERIOR LLR UPDATE
        # ========================================

        for i in range(N):
            posterior[i] = llr[i]

        for e in range(E):
            variable = edge_variables[e]
            posterior[variable] += c_to_v[e]

        # ========================================
        # C. HARD DECISION
        # ========================================

        abs_llr_sum = 0.0

        for i in range(N):

            decoded[i] = (
                1 if posterior[i] < 0 else 0
            )

            abs_llr_sum += abs(posterior[i])

        # ========================================
        # D. SYNDROME CHECK
        # ========================================

        unsatisfied = 0

        for check in range(M):

            parity = 0

            for e in range(
                check_ptr[check],
                check_ptr[check + 1]
            ):
                parity ^= int(
                    decoded[edge_variables[e]]
                )

            unsatisfied += parity

        unsatisfied_history[iteration] = unsatisfied

        mean_llr_history[iteration] = (
            abs_llr_sum / N
        )

        # Early stopping.
        if unsatisfied == 0:

            return (
                decoded,
                True,
                iteration + 1,
                unsatisfied_history,
                mean_llr_history
            )

        # ========================================
        # E. VARIABLE-NODE UPDATE
        # ========================================

        for e in range(E):

            variable = edge_variables[e]

            message = (
                posterior[variable] - c_to_v[e]
            )

            v_to_c[e] = min(
                max(message, -llr_clip),
                llr_clip
            )

    return (
        decoded,
        False,
        max_iterations,
        unsatisfied_history,
        mean_llr_history
    )


# ============================================================
# 3. PARALLEL BATCH DECODER
# ============================================================

@njit(parallel=True, cache=True)
def decode_batch_kernel(
    channel_llrs,
    check_ptr,
    edge_variables,
    max_iterations,
    llr_clip
):
    """
    Decode multiple LDPC frames in parallel.

    Each frame is independent and processed using
    the sequential JIT-compiled BP kernel.

    Numba distributes frames across CPU threads.

    Parameters
    ----------
    channel_llrs : ndarray
        Shape (num_frames, N).

    Returns
    -------
    decoded : ndarray
        Shape (num_frames, N).

    successes : ndarray
        Boolean success flag for each frame.

    iterations : ndarray
        Number of iterations performed for each frame.
    """

    num_frames, N = channel_llrs.shape

    decoded = np.empty(
        (num_frames, N),
        dtype=np.uint8
    )

    successes = np.empty(
        num_frames,
        dtype=np.bool_
    )

    iterations = np.empty(
        num_frames,
        dtype=np.int64
    )

    # Parallel execution across independent frames.
    for frame in prange(num_frames):

        bits, success, count, _, _ = bp_decode_kernel(
            channel_llrs[frame],
            check_ptr,
            edge_variables,
            max_iterations,
            llr_clip
        )

        for i in range(N):
            decoded[frame, i] = bits[i]

        successes[frame] = success
        iterations[frame] = count

    return decoded, successes, iterations


# ============================================================
# 4. BP DECODER CLASS
# ============================================================

class BPDecoderNumba:
    """
    Numba-accelerated binary LDPC BP decoder.

    Supports:
        - Sequential JIT decoding
        - Parallel batch decoding
        - Arbitrary binary LDPC parity-check matrices
        - Regular and irregular LDPC codes
        - Early stopping
        - Iteration history
    """

    def __init__(
        self,
        code,
        max_iterations=50,
        llr_clip=30.0
    ):

        self.code = code
        self.N = code.N
        self.M = code.M

        self.max_iterations = int(max_iterations)
        self.llr_clip = float(llr_clip)

        if self.max_iterations < 1:
            raise ValueError(
                "max_iterations must be positive"
            )

        if (
            not np.isfinite(self.llr_clip)
            or self.llr_clip <= 0
        ):
            raise ValueError(
                "llr_clip must be finite and positive"
            )

        self._build_graph()

    def _build_graph(self):
        """
        Build CSR-style Tanner graph adjacency.

        check_ptr[j] gives the starting edge index
        for check node j.
        """

        check_ptr = [0]
        edge_variables = []

        for check in range(self.M):

            for variable in self.code.row_neighbors[check]:
                edge_variables.append(int(variable))

            check_ptr.append(len(edge_variables))

        self.check_ptr = np.asarray(
            check_ptr,
            dtype=np.int64
        )

        self.edge_variables = np.asarray(
            edge_variables,
            dtype=np.int64
        )

        self.E = len(edge_variables)

    # --------------------------------------------------------
    # Sequential decoding
    # --------------------------------------------------------

    def decode(self, channel_llr, return_history=False):
        """
        Decode one LDPC codeword sequentially.
        """

        channel_llr = np.asarray(
            channel_llr,
            dtype=np.float64
        )

        if channel_llr.shape != (self.N,):
            raise ValueError(
                f"Expected {self.N} channel LLR values"
            )

        if not np.all(np.isfinite(channel_llr)):
            raise ValueError(
                "Channel LLRs must be finite"
            )

        (
            decoded,
            success,
            iterations,
            unsatisfied,
            mean_llr
        ) = bp_decode_kernel(
            channel_llr,
            self.check_ptr,
            self.edge_variables,
            self.max_iterations,
            self.llr_clip
        )

        if return_history:

            history = []

            for i in range(iterations):

                history.append({
                    "iteration": i + 1,
                    "unsatisfied_checks": int(
                        unsatisfied[i]
                    ),
                    "mean_abs_llr": float(
                        mean_llr[i]
                    )
                })

            return (
                decoded,
                bool(success),
                int(iterations),
                history
            )

        return (
            decoded,
            bool(success),
            int(iterations)
        )

    # --------------------------------------------------------
    # Parallel batch decoding
    # --------------------------------------------------------

    def decode_batch(self, channel_llrs):
        """
        Decode multiple frames using Numba parallelism.

        Input:
            channel_llrs: shape (num_frames, N)

        Returns:
            decoded, successes, iterations
        """

        channel_llrs = np.asarray(
            channel_llrs,
            dtype=np.float64
        )

        if (
            channel_llrs.ndim != 2
            or channel_llrs.shape[1] != self.N
        ):
            raise ValueError(
                f"Expected shape (frames, {self.N})"
            )

        if not np.all(np.isfinite(channel_llrs)):
            raise ValueError(
                "Channel LLRs must be finite"
            )

        return decode_batch_kernel(
            channel_llrs,
            self.check_ptr,
            self.edge_variables,
            self.max_iterations,
            self.llr_clip
        )

    # --------------------------------------------------------
    # Configure number of Numba threads
    # --------------------------------------------------------

    @staticmethod
    def set_threads(num_threads):
        """
        Set the number of Numba worker threads.

        Must not exceed the maximum available
        Numba thread count.
        """
        set_num_threads(num_threads)

    @staticmethod
    def get_threads():
        """Return the current Numba thread count."""
        return get_num_threads()


# ============================================================
# 5. STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    from alist import LDPCCode

    if len(sys.argv) != 2:
        print(
            "Usage: python bp_decoder_numba.py <alist_file>"
        )
        sys.exit(1)

    code = LDPCCode(sys.argv[1])

    decoder = BPDecoderNumba(
        code,
        max_iterations=50
    )

    print("\n=== NUMBA LDPC BP DECODER ===")
    print(f"Codeword length N: {code.N}")
    print(f"Parity checks M: {code.M}")
    print(f"Tanner graph edges: {decoder.E}")

    # ========================================================
    # TEST 1: SEQUENTIAL SINGLE-BIT ERROR
    # ========================================================

    print("\n=== TEST 1: SEQUENTIAL DECODING ===")

    transmitted = np.zeros(
        code.N,
        dtype=np.uint8
    )

    received = np.ones(
        code.N,
        dtype=np.float64
    )

    error_position = 3
    received[error_position] = -0.5

    sigma = 0.7

    channel_llr = 2.0 * received / sigma**2

    initial = (channel_llr < 0).astype(np.uint8)

    print(
        "Initial bit errors:",
        np.count_nonzero(initial != transmitted)
    )

    decoded, success, iterations, history = decoder.decode(
        channel_llr,
        return_history=True
    )

    print("Decoding success:", success)
    print("Iterations:", iterations)

    print(
        "Final bit errors:",
        np.count_nonzero(decoded != transmitted)
    )

    print("\nIteration history:")

    for entry in history:
        print(entry)

    # ========================================================
    # TEST 2: PARALLEL BATCH DECODING
    # ========================================================

    print("\n=== TEST 2: PARALLEL BATCH DECODING ===")

    rng = np.random.default_rng(42)

    num_frames = 100

    received_batch = (
        1.0
        + rng.normal(
            0.0,
            sigma,
            size=(num_frames, code.N)
        )
    )

    channel_llrs = (
        2.0 * received_batch / sigma**2
    )

    # Use four CPU threads.
    decoder.set_threads(4)

    print(
        "Numba threads:",
        decoder.get_threads()
    )

    decoded_batch, successes, batch_iterations = (
        decoder.decode_batch(channel_llrs)
    )

    print(
        "Decoded batch shape:",
        decoded_batch.shape
    )

    print(
        "Successful decodings:",
        np.count_nonzero(successes)
    )

    print(
        "Average iterations:",
        np.mean(batch_iterations)
    )

    print(
        "Total bit errors:",
        np.count_nonzero(decoded_batch)
    )

    # ========================================================
    # TEST 3: SEQUENTIAL VS PARALLEL CORRECTNESS
    # ========================================================

    print("\n=== TEST 3: CORRECTNESS COMPARISON ===")

    mismatched_frames = 0
    mismatched_iterations = 0
    mismatched_successes = 0

    for frame in range(num_frames):

        bits, ok, count = decoder.decode(
            channel_llrs[frame]
        )

        if not np.array_equal(
            bits,
            decoded_batch[frame]
        ):
            mismatched_frames += 1

        if count != batch_iterations[frame]:
            mismatched_iterations += 1

        if ok != successes[frame]:
            mismatched_successes += 1

    print(
        "Mismatched decoded frames:",
        mismatched_frames
    )

    print(
        "Mismatched iteration counts:",
        mismatched_iterations
    )

    print(
        "Mismatched success flags:",
        mismatched_successes
    )

    if (
        mismatched_frames == 0
        and mismatched_iterations == 0
        and mismatched_successes == 0
    ):
        print(
            "\nSequential and parallel results match!"
        )
    else:
        print(
            "\nWARNING: Sequential and parallel "
            "results differ."
        )
