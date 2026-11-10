import numpy as np

from numba import set_num_threads, get_num_threads

from alist import LDPCCode
from bp_decoder_numba import BPDecoderNumba


code = LDPCCode("codes/wimax_576_0.5.alist")

decoder = BPDecoderNumba(
    code,
    max_iterations=50
)

# Generate 100 received frames.
rng = np.random.default_rng(42)

sigma = 0.7

received = 1.0 + rng.normal(
    0.0,
    sigma,
    size=(100, code.N)
)

channel_llrs = 2.0 * received / sigma**2

# Configure parallel threads.
set_num_threads(4)

print("Numba threads:", get_num_threads())

# Decode all frames in parallel.
decoded, successes, iterations = decoder.decode_batch(
    channel_llrs
)

print("Decoded shape:", decoded.shape)
print("Successful decodings:", np.count_nonzero(successes))
print("Average iterations:", np.mean(iterations))