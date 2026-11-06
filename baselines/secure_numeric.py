"""Fixed-point ring encoding and cryptographic PRG for protocol simulations."""
import hashlib
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def encode(vector, fractional_bits=16):
    vector = np.asarray(vector, dtype=np.float64)
    rounded = np.rint(vector * 2**fractional_bits)
    if not np.isfinite(rounded).all() or np.any(np.abs(rounded) >= 2**62):
        raise OverflowError("fixed-point statistic exceeds the supported ring range")
    return rounded.astype(np.int64).view(np.uint64)


def decode(vector, fractional_bits=16):
    return np.asarray(vector, dtype=np.uint64).view(np.int64).astype(float) / 2**fractional_bits


def prg(seed, size, domain):
    key = hashlib.sha256(bytes(seed) + bytes(domain)).digest()
    encryptor = Cipher(algorithms.AES(key), modes.CTR(bytes(16))).encryptor()
    stream = encryptor.update(bytes(size * 8)) + encryptor.finalize()
    return np.frombuffer(stream, dtype="<u8").astype(np.uint64, copy=True)


def ring_sum(vectors):
    vectors = list(vectors)
    if not vectors:
        raise ValueError("cannot aggregate an empty set")
    total = np.zeros_like(vectors[0], dtype=np.uint64)
    for vector in vectors:
        np.add(total, vector, out=total)
    return total
