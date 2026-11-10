
import numpy as np


class BPDecoder:
    """
    Optimized binary LDPC belief propagation decoder.

    Algorithm:
        Log-domain sum-product with flooding schedule.

    Improvements:
        - O(E) check-node updates per iteration
        - Vectorized variable-node updates
        - Sparse Tanner graph representation
        - Stable phi-domain calculations
        - Early stopping using parity checks

    Compatible with the existing simulation.py.
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
            raise ValueError("max_iterations must be positive")

        if not np.isfinite(self.llr_clip) or self.llr_clip <= 0:
            raise ValueError("llr_clip must be finite and positive")

        self._build_graph()

    def _build_graph(self):
        """
        Build an edge-based Tanner graph.

        Every nonzero H[j, i] becomes one edge.
        """

        edge_variables = []
        edge_checks = []

        self.check_edges = []
        self.variable_edges = [[] for _ in range(self.N)]

        edge_id = 0

        for check in range(self.M):

            current_edges = []

            for variable in self.code.row_neighbors[check]:

                variable = int(variable)

                edge_variables.append(variable)
                edge_checks.append(check)

                current_edges.append(edge_id)
                self.variable_edges[variable].append(edge_id)

                edge_id += 1

            self.check_edges.append(
                np.asarray(current_edges, dtype=np.int64)
            )

        self.edge_variables = np.asarray(
            edge_variables, dtype=np.int64
        )

        self.edge_checks = np.asarray(
            edge_checks, dtype=np.int64
        )

        self.variable_edges = [
            np.asarray(edges, dtype=np.int64)
            for edges in self.variable_edges
        ]

        self.E = edge_id

        # Precompute the check-node degree.
        self.check_degrees = np.array(
            [len(edges) for edges in self.check_edges],
            dtype=np.int64
        )

    @staticmethod
    def _phi(x):
        """
        Stable phi function:

            phi(x) = -log(tanh(x/2))

        Equivalent form:

            phi(x) = log(1 + exp(-x))
                   - log(1 - exp(-x))

        for x > 0.
        """

        x = np.asarray(x, dtype=np.float64)

        # Avoid phi(0) = infinity.
        x = np.maximum(x, 1e-12)

        return (
            np.log1p(np.exp(-x))
            - np.log(-np.expm1(-x))
        )

    def decode(self, channel_llr, return_history=False):
        """
        Decode one binary LDPC codeword.

        Parameters
        ----------
        channel_llr : array_like
            Channel LLR values of length N.

        return_history : bool
            Return per-iteration diagnostics.

        Returns
        -------
        decoded : ndarray
            Estimated codeword.

        success : bool
            True if syndrome is zero.

        iterations : int
            Number of iterations performed.

        history : list, optional
            Per-iteration diagnostic data.
        """

        channel_llr = np.asarray(
            channel_llr, dtype=np.float64
        )

        if channel_llr.shape != (self.N,):
            raise ValueError(
                f"Expected {self.N} channel LLR values"
            )

        if not np.all(np.isfinite(channel_llr)):
            raise ValueError("Channel LLRs must be finite")

        channel_llr = np.clip(
            channel_llr,
            -self.llr_clip,
            self.llr_clip
        )

        # Initialize variable-to-check messages.
        v_to_c = channel_llr[self.edge_variables].copy()

        # Initialize check-to-variable messages.
        c_to_v = np.zeros(self.E, dtype=np.float64)

        history = []

        decoded = (channel_llr < 0).astype(np.uint8)

        # Early stopping before the first iteration.
        if self.code.is_codeword(decoded):

            if return_history:
                return decoded, True, 0, history

            return decoded, True, 0

        for iteration in range(1, self.max_iterations + 1):

            # =================================================
            # 1. CHECK-NODE UPDATE
            # =================================================

            for check in range(self.M):

                edges = self.check_edges[check]
                degree = len(edges)

                if degree == 0:
                    continue

                if degree == 1:
                    # A degree-one parity check requires
                    # the connected variable to be zero.
                    c_to_v[edges[0]] = self.llr_clip
                    continue

                incoming = v_to_c[edges]

                magnitudes = np.abs(incoming)

                # Sign product excluding the target edge.
                negative = incoming < 0

                total_negative = np.count_nonzero(negative)

                outgoing_negative = (
                    (total_negative - negative.astype(int)) % 2
                ).astype(bool)

                # Handle zero-valued messages explicitly.
                zero_mask = incoming == 0
                zero_count = int(np.count_nonzero(zero_mask))

                if zero_count >= 2:
                    # Every outgoing product contains
                    # at least one zero.
                    c_to_v[edges] = 0.0
                    continue

                if zero_count == 1:
                    # Only the message directed toward
                    # the zero-valued input can be nonzero.
                    c_to_v[edges] = 0.0

                    zero_index = int(np.flatnonzero(zero_mask)[0])

                    others = magnitudes[~zero_mask]

                    magnitude = self._phi(
                        np.sum(self._phi(others))
                    )

                    sign = (
                        -1.0
                        if outgoing_negative[zero_index]
                        else 1.0
                    )

                    c_to_v[edges[zero_index]] = np.clip(
                        sign * magnitude,
                        -self.llr_clip,
                        self.llr_clip
                    )

                    continue

                # All incoming messages are nonzero.
                phi_values = self._phi(magnitudes)

                total_phi = np.sum(phi_values)

                # Exclude the target edge's contribution.
                outgoing_phi = np.maximum(
                    total_phi - phi_values,
                    1e-12
                )

                outgoing_magnitudes = self._phi(
                    outgoing_phi
                )

                outgoing_signs = np.where(
                    outgoing_negative,
                    -1.0,
                    1.0
                )

                c_to_v[edges] = np.clip(
                    outgoing_signs * outgoing_magnitudes,
                    -self.llr_clip,
                    self.llr_clip
                )

            # =================================================
            # 2. POSTERIOR LLR UPDATE
            # =================================================

            posterior_llr = channel_llr.copy()

            np.add.at(
                posterior_llr,
                self.edge_variables,
                c_to_v
            )

            # =================================================
            # 3. HARD DECISION
            # =================================================

            decoded = (
                posterior_llr < 0
            ).astype(np.uint8)

            # =================================================
            # 4. SYNDROME CHECK
            # =================================================

            syndrome = self.code.syndrome(decoded)

            unsatisfied_checks = int(
                np.count_nonzero(syndrome)
            )

            if return_history:
                history.append({
                    "iteration": iteration,
                    "unsatisfied_checks": unsatisfied_checks,
                    "mean_abs_llr": float(
                        np.mean(np.abs(posterior_llr))
                    )
                })

            if unsatisfied_checks == 0:

                if return_history:
                    return decoded, True, iteration, history

                return decoded, True, iteration

            # =================================================
            # 5. VARIABLE-NODE UPDATE
            # =================================================

            v_to_c = (
                posterior_llr[self.edge_variables]
                - c_to_v
            )

            v_to_c = np.clip(
                v_to_c,
                -self.llr_clip,
                self.llr_clip
            )

        if return_history:
            return decoded, False, self.max_iterations, history

        return decoded, False, self.max_iterations


# ============================================================
# CONTROLLED ERROR TEST
# ============================================================

if __name__ == "__main__":

    import sys
    from alist import LDPCCode

    if len(sys.argv) != 2:
        print("Usage: python bp_decoder.py <alist_file>")
        sys.exit(1)

    code = LDPCCode(sys.argv[1])

    decoder = BPDecoder(
        code,
        max_iterations=50
    )

    print("\n=== OPTIMIZED LDPC BP DECODER ===")
    print(f"Codeword length: {code.N}")
    print(f"Parity checks: {code.M}")
    print(f"Tanner graph edges: {decoder.E}")

    # All-zero transmitted codeword.
    transmitted = np.zeros(code.N, dtype=np.uint8)

    # BPSK modulation.
    symbols = np.ones(code.N, dtype=np.float64)

    # Introduce one controlled error.
    received = symbols.copy()

    error_position = 3
    received[error_position] = -0.5

    sigma = 0.7

    channel_llr = 2.0 * received / sigma**2

    initial = (channel_llr < 0).astype(np.uint8)

    initial_errors = int(
        np.count_nonzero(initial != transmitted)
    )

    print("\n=== BEFORE DECODING ===")
    print(f"Initial bit errors: {initial_errors}")
    print(
        "Unsatisfied checks:",
        np.count_nonzero(code.syndrome(initial))
    )

    decoded, success, iterations, history = decoder.decode(
        channel_llr,
        return_history=True
    )

    final_errors = int(
        np.count_nonzero(decoded != transmitted)
    )

    print("\n=== AFTER DECODING ===")
    print(f"Decoding success: {success}")
    print(f"Iterations: {iterations}")
    print(f"Final bit errors: {final_errors}")

    print("\n=== ITERATION HISTORY ===")

    for entry in history:
        print(
            f"Iteration {entry['iteration']:2d} | "
            f"Unsatisfied checks: "
            f"{entry['unsatisfied_checks']:3d} | "
            f"Mean |LLR|: "
            f"{entry['mean_abs_llr']:.3f}"
        )

    if final_errors == 0:
        print("\nAll bit errors successfully corrected!")
    else:
        print("\nSome bit errors remain.")
