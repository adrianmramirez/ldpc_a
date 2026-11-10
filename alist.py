
import numpy as np
from pathlib import Path


class LDPCCode:
    """
    Binary LDPC code represented by a parity-check matrix H.

    Supports regular and irregular AList files using
    MacKay's standard AList format.
    """

    def __init__(self, filename):
        self.filename = str(filename)
        self.H = None
        self.N = None
        self.M = None
        self.column_weights = None
        self.row_weights = None
        self.column_neighbors = None
        self.row_neighbors = None

        self.load_alist(filename)

    def load_alist(self, filename):
        # Read all integer entries, ignoring blank lines
        # and optional comment lines.
        lines = Path(filename).read_text().splitlines()

        data = []
        for line in lines:
            line = line.split("#")[0].strip()
            if line:
                data.extend(map(int, line.split()))

        pos = 0

        def take(count):
            nonlocal pos
            if pos + count > len(data):
                raise ValueError("Incomplete AList file")
            values = data[pos:pos + count]
            pos += count
            return values

        # 1. Matrix dimensions
        self.N, self.M = take(2)

        if self.N <= 0 or self.M <= 0:
            raise ValueError("Invalid matrix dimensions")

        # 2. Maximum column and row weights
        max_col_weight, max_row_weight = take(2)

        # 3. Individual column and row weights
        self.column_weights = np.array(take(self.N))
        self.row_weights = np.array(take(self.M))

        if (
            np.any(self.column_weights < 0)
            or np.any(self.row_weights < 0)
            or np.any(self.column_weights > max_col_weight)
            or np.any(self.row_weights > max_row_weight)
        ):
            raise ValueError("Invalid row or column weights")

        if sum(self.column_weights) != sum(self.row_weights):
            raise ValueError("Row and column weight totals disagree")

        # 4. Read column adjacency lists
        self.column_neighbors = []

        for col in range(self.N):
            entries = take(max_col_weight)
            weight = int(self.column_weights[col])

            active = entries[:weight]
            padding = entries[weight:]

            if any(x < 1 or x > self.M for x in active):
                raise ValueError(f"Invalid row index in column {col + 1}")

            if len(set(active)) != len(active):
                raise ValueError(f"Duplicate row index in column {col + 1}")

            if any(x != 0 for x in padding):
                raise ValueError(f"Invalid padding in column {col + 1}")

            self.column_neighbors.append(
                np.array([x - 1 for x in active], dtype=int)
            )

        # 5. Read row adjacency lists
        self.row_neighbors = []

        for row in range(self.M):
            entries = take(max_row_weight)
            weight = int(self.row_weights[row])

            active = entries[:weight]
            padding = entries[weight:]

            if any(x < 1 or x > self.N for x in active):
                raise ValueError(f"Invalid column index in row {row + 1}")

            if len(set(active)) != len(active):
                raise ValueError(f"Duplicate column index in row {row + 1}")

            if any(x != 0 for x in padding):
                raise ValueError(f"Invalid padding in row {row + 1}")

            self.row_neighbors.append(
                np.array([x - 1 for x in active], dtype=int)
            )

        if pos != len(data):
            raise ValueError("Unexpected extra data in AList file")

        # 6. Construct binary parity-check matrix
        self.H = np.zeros((self.M, self.N), dtype=np.uint8)

        for col, neighbors in enumerate(self.column_neighbors):
            self.H[neighbors, col] = 1

        # 7. Validate row adjacency lists
        for row, neighbors in enumerate(self.row_neighbors):
            actual = np.flatnonzero(self.H[row])

            if not np.array_equal(np.sort(actual), np.sort(neighbors)):
                raise ValueError(
                    f"Row adjacency mismatch at row {row + 1}"
                )

        print(f"Successfully loaded: {filename}")
        print(f"Matrix dimensions: {self.M} x {self.N}")
        print(f"Number of nonzero entries: {int(self.H.sum())}")
        print(f"Maximum column weight: {max_col_weight}")
        print(f"Maximum row weight: {max_row_weight}")

    def syndrome(self, codeword):
        """Compute H @ codeword modulo 2."""
        codeword = np.asarray(codeword, dtype=np.uint8)

        if codeword.shape != (self.N,):
            raise ValueError(f"Codeword must contain {self.N} bits")

        if np.any(codeword > 1):
            raise ValueError("Codeword must be binary")

        return (self.H.astype(np.int64) @ codeword.astype(np.int64)) % 2

    def is_codeword(self, codeword):
        """Check whether all parity-check equations are satisfied."""
        return bool(np.all(self.syndrome(codeword) == 0))

    def display(self):
        """Print the parity-check matrix."""
        print("\nParity-check matrix H:")
        print(self.H)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python alist.py <alist_file>")
        sys.exit(1)

    code = LDPCCode(sys.argv[1])
    code.display()

    zero_codeword = np.zeros(code.N, dtype=np.uint8)

    print("\nAll-zero codeword:")
    print(zero_codeword)
    print("Syndrome:", code.syndrome(zero_codeword))
    print("Valid codeword:", code.is_codeword(zero_codeword))

