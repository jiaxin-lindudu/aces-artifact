"""FastLloyd MSA: a server adds noise without learning the clients' masks.

Same-process, all-client/no-dropout/non-colluding simulation of Figure 1.
Secret seed and unmasked values are never passed to the MSA server object.
"""
import secrets
import numpy as np
from baselines.secure_numeric import encode, decode, prg, ring_sum


class MSAServer:
    def aggregate(self, masked_messages, noise, bits):
        return ring_sum([*masked_messages, encode(noise, bits)])


class MSAClients:
    def __init__(self, participants, bits=16):
        self.participants, self.bits = participants, bits
        # Simulates a confidential client-group setup, NOT a public experiment seed.
        self._shared_seed = secrets.token_bytes(32)
        self._round = 0

    def mask(self, vectors):
        if len(vectors) != self.participants:
            raise ValueError("FastLloyd MSA requires all configured participants")
        self._round += 1
        size = len(vectors[0])
        masks = [prg(self._shared_seed, size,
                     f"fastlloyd:{self._round}:{i}".encode()) for i in range(self.participants)]
        messages = [encode(v, self.bits) + mask for v, mask in zip(vectors, masks)]
        return messages, ring_sum(masks)

    def reveal_to_clients(self, masked_aggregate, mask_sum):
        return decode(masked_aggregate - mask_sum, self.bits)
