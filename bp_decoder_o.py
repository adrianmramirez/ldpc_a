
import numpy as np


class BPDecoder:
    """
    Binary LDPC belief propagation decoder.

    Algorithm:
        Log-domain sum-product (flooding schedule)

    Supports:
        - Arbitrary binary LDPC parity-check matrices
        - Regular and irregular LDPC codes
        - Early stopping using syndrome checks
        - Configurable maximum iterations
    """

    def __init__(self, code, max_iterations=50, llr_clip=30.0):

        self.code = code
        self.N = code.N
        self.M = code.M

        self.max_iterations = max_iterations
        self.llr_clip = llr_clip

        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        if llr_clip <= 0:
            raise ValueError("llr_clip must be positive")

        self._build_graph()

    def _build_graph(self):
        """
        Assign one edge index to every nonzero entry of H.
        """

        edge_variables = []
        edge_checks = []

        self.check_edges = [[] for _ in range(self.M)]
        self.variable_edges = [[] for _ in range(self.N)]

        edge_id = 0

        for check in range(self.M):

            for variable in self.code.row_neighbors[check]:

                variable = int(variable)

                edge_variables.append(variable)
                edge_checks.append(check)

                self.check_edges[check].append(edge_id)
                self.variable_edges[variable].append(edge_id)

                edge_id += 1

        self.edge_variables = np.asarray(
            edge_variables, dtype=np.int64
        )

        self.edge_checks = np.asarray(
            edge_checks, dtype=np.int64
        )

        self.check_edges = [
            np.asarray(edges, dtype=np.int64)
            for edges in self.check_edges
        ]

        self.variable_edges = [
            np.asarray(edges, dtype=np.int64)
            for edges in self.variable_edges
        ]

        self.E = edge_id

    def decode(self, channel_llr, return_history=False):
        """
        Decode a received LDPC codeword.

        Parameters
        ----------
        channel_llr : ndarray
            Channel LLRs of length N.

        return_history : bool
            Return per-iteration diagnostics when True.

        Returns
        -------
        decoded : ndarray
            Estimated binary codeword.

        success : bool
            True if all parity checks are satisfied.

        iterations : int
            Number of iterations performed.

        history : list, optional
            Per-iteration diagnostic information.
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

        # Initialize messages
        v_to_c = channel_llr[self.edge_variables].copy()
        c_to_v = np.zeros(self.E, dtype=np.float64)

        posterior_llr = channel_llr.copy()

        decoded = (posterior_llr < 0).astype(np.uint8)

        history = []

        # Initial syndrome check
        syndrome = self.code.syndrome(decoded)

        if np.all(syndrome == 0):

            if return_history:
                return decoded, True, 0, history

            return decoded, True, 0

        # ==========================================
        # BELIEF PROPAGATION ITERATIONS
        # ==========================================

        for iteration in range(1, self.max_iterations + 1):

            # --------------------------------------
            # 1. CHECK NODE UPDATE
            # --------------------------------------

            for check in range(self.M):

                edges = self.check_edges[check]
                degree = len(edges)

                if degree == 0:
                    continue

                if degree == 1:
                    c_to_v[edges[0]] = self.llr_clip
                    continue

                incoming = v_to_c[edges]

                tanh_values = np.tanh(incoming / 2.0)

                for local_index, edge in enumerate(edges):

                    other_values = np.delete(
                        tanh_values, local_index
                    )

                    product = np.prod(other_values)

                    product = np.clip(
                        product,
                        -1.0 + 1e-12,
                        1.0 - 1e-12
                    )

                    message = 2.0 * np.arctanh(product)

                    c_to_v[edge] = np.clip(
                        message,
                        -self.llr_clip,
                        self.llr_clip
                    )

            # --------------------------------------
            # 2. POSTERIOR LLR UPDATE
            # --------------------------------------

            posterior_llr = channel_llr.copy()

            np.add.at(
                posterior_llr,
                self.edge_variables,
                c_to_v
            )

            # --------------------------------------
            # 3. HARD DECISIONS
            # --------------------------------------

            decoded = (
                posterior_llr < 0
            ).astype(np.uint8)

            # --------------------------------------
            # 4. SYNDROME CHECK
            # --------------------------------------

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

            # --------------------------------------
            # 5. VARIABLE NODE UPDATE
            # --------------------------------------

            v_to_c = (
                posterior_llr[self.edge_variables] - c_to_v
            )

            v_to_c = np.clip(
                v_to_c,
                -self.llr_clip,
                self.llr_clip
            )

        if return_history:
            return decoded, False, self.max_iterations, history

        return decoded, False, self.max_iterations


# ==================================================
# CONTROLLED SINGLE-BIT ERROR TEST
# ==================================================

if __name__ == "__main__":

    from alist import LDPCCode
    import sys

    if len(sys.argv) != 2:
        print("Usage: python bp_decoder.py <alist_file>")
        sys.exit(1)

    # ==============================================
    # 1. LOAD LDPC CODE
    # ==============================================

    code = LDPCCode(sys.argv[1])

    decoder = BPDecoder(
        code,
        max_iterations=50
    )

    print("\n=== LDPC BP Decoder ===")
    print(f"Codeword length N: {code.N}")
    print(f"Parity checks M: {code.M}")
    print(f"Tanner graph edges: {decoder.E}")

    # ==============================================
    # 2. GENERATE TRANSMITTED CODEWORD
    # ==============================================

    transmitted = np.zeros(code.N, dtype=np.uint8)

    # BPSK mapping:
    # 0 -> +1
    # 1 -> -1

    symbols = 1.0 - 2.0 * transmitted.astype(float)

    print("\n=== TRANSMISSION ===")

    print("\nTransmitted codeword:")
    print(transmitted)

    print("\nBPSK symbols:")
    print(symbols)

    # ==============================================
    # 3. INTRODUCE CONTROLLED ERROR
    # ==============================================

    received = symbols.copy()

    # Deliberately introduce one incorrect bit.
    error_position = 3

    received[error_position] = -0.5

    sigma = 0.7

    print("\n=== CHANNEL ===")

    print(f"\nControlled error at bit index: {error_position}")
    print(f"Noise standard deviation: {sigma}")

    print("\nReceived symbols:")
    print(np.round(received, 3))

    # ==============================================
    # 4. COMPUTE CHANNEL LLRs
    # ==============================================

    channel_llr = 2.0 * received / (sigma ** 2)

    print("\nChannel LLRs:")
    print(np.round(channel_llr, 3))

    # ==============================================
    # 5. INITIAL HARD DECISIONS
    # ==============================================

    initial_decisions = (
        channel_llr < 0
    ).astype(np.uint8)

    initial_errors = int(
        np.count_nonzero(initial_decisions != transmitted)
    )

    initial_syndrome = code.syndrome(initial_decisions)

    print("\n=== BEFORE DECODING ===")

    print("\nInitial hard decisions:")
    print(initial_decisions)

    print(f"\nInitial bit errors: {initial_errors}")

    print("\nInitial syndrome:")
    print(initial_syndrome)

    print(
        "\nUnsatisfied parity checks:",
        np.count_nonzero(initial_syndrome)
    )

    # ==============================================
    # 6. RUN BELIEF PROPAGATION
    # ==============================================

    decoded, success, iterations, history = decoder.decode(
        channel_llr,
        return_history=True
    )

    # ==============================================
    # 7. FINAL RESULTS
    # ==============================================

    final_errors = int(
        np.count_nonzero(decoded != transmitted)
    )

    final_syndrome = code.syndrome(decoded)

    print("\n=== AFTER DECODING ===")

    print("\nDecoded codeword:")
    print(decoded)

    print(f"\nDecoding success: {success}")
    print(f"Iterations performed: {iterations}")
    print(f"Final bit errors: {final_errors}")

    print("\nFinal syndrome:")
    print(final_syndrome)

    # ==============================================
    # 8. ITERATION HISTORY
    # ==============================================

    print("\n=== ITERATION HISTORY ===")

    for entry in history:

        print(
            f"Iteration {entry['iteration']:2d} | "
            f"Unsatisfied checks: "
            f"{entry['unsatisfied_checks']:3d} | "
            f"Mean |LLR|: {entry['mean_abs_llr']:.3f}"
        )

    # ==============================================
    # 9. SUMMARY
    # ==============================================

    print("\n=== SUMMARY ===")

    print(f"Initial bit errors: {initial_errors}")
    print(f"Final bit errors:   {final_errors}")

    if final_errors == 0:
        print("\nAll bit errors successfully corrected!")

    elif final_errors < initial_errors:
        print("\nDecoder reduced the number of bit errors.")

    else:
        print("\nDecoder did not correct the introduced error.")
