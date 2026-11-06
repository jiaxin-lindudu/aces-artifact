"""Reusable client/server contracts for the federated clustering experiments.

WHY THIS EXISTS
---------------
The "Ours" implementation (`federated/ours/`) hard-codes its own Client/Server
classes inside the training loop. A baseline method (FFCM, FKM, NNFC, or a new
one) can only be run by copying that loop. This module states the *minimum*
contract a training loop needs, so a loop can be written once and the method
selected by name.

WHAT THIS IS NOT
----------------
It is not an algorithm, not a privacy boundary and not a distributed runtime.
Everything here is in-process and describes plain Python objects only.

HOW A NEW METHOD JOINS
----------------------
1. Implement the client/server classes (duck-typed is enough).
2. Either subclass :class:`ClientProtocol` / :class:`ServerProtocol`, or keep
   your own classes untouched and call ``ClientProtocol.register(MyClient)``
   (``ABCMeta.register`` creates a virtual subclass: ``isinstance`` works, no
   code change inside your class).
3. ``register_protocol('my_method', MyClient, MyServer)``.

:class:`AnchoredSpectralClient` / :class:`AnchoredSpectralServer` already satisfy
the contract and are registered under the name ``'ours'``. Registered
structurally (not by inheritance) on purpose: inheritance would force a change
inside the numerically validated Ours classes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple


class RoundPayload(NamedTuple):
    """One client's upload for one round.

    S_orig/S_emb : weighted sums of the local points in the two spaces,
                   shape (K, d_orig) / (K, d_emb)
    n_k          : soft cluster masses, shape (K,)
    diagnostics  : JSON-serialisable scalars reported in the clear (never a
                   per-sample quantity)
    """

    S_orig: Any
    S_emb: Any
    n_k: Any
    diagnostics: Dict[str, Any]


class ClientProtocol(ABC):
    """What the trainer needs from a client, and nothing more.

    Attributes (documented, not enforced, because ours sets them in __init__):
        id                    : stable client index
        communication_bytes   : bytes of the last upload
        embedding_time        : seconds spent on the local embedding
        statistics_time       : seconds spent on the sufficient statistics
        encryption_time       : seconds spent encrypting (0 in plaintext)
        training_embedding    : cached local spectral coordinates, or None
        landmark_orig         : landmarks of the Nyström extension, or None
        landmark_embedding    : landmark spectral coordinates, or None
    """

    @abstractmethod
    def get_noisy_statistics(self, centers_orig, centers_emb, fuzziness,
                             use_procrustes=True, add_noise=True):
        """Return the four fields of :class:`RoundPayload` as a tuple.

        `add_noise` is the caller's switch for the (uncalibrated) perturbation;
        clients must still honour their own dp_sigma == 0 short circuit.
        """

    @abstractmethod
    def get_encrypted_statistics(self, centers_orig, centers_emb, fuzziness,
                                 ckks_manager, use_procrustes=True):
        """Return (ciphertext chunks, S_orig.shape, S_emb.shape, diagnostics)."""

    @abstractmethod
    def embed_out_of_sample(self, X_new, centers_orig=None):
        """Map new points into the client's spectral embedding, or raise.

        OFFLINE EVALUATION ONLY in the Ours implementation: it adds no
        communication and is not covered by CKKS.
        """


class ServerProtocol(ABC):
    """What the trainer needs from a server, and nothing more."""

    @abstractmethod
    def initialize_centers(self, method, init_data=None, random_state=42):
        """Return (centers_orig, centers_emb)."""

    @abstractmethod
    def aggregate_statistics(self, stats_list):
        """Fold the clients' payloads into new centers; return (orig, emb)."""

    @abstractmethod
    def aggregate_encrypted_statistics(self, encrypted_stats_list):
        """Sum ciphertexts without holding the secret key."""


class ProtocolSpec(NamedTuple):
    name: str
    client_cls: type
    server_cls: type
    trainer: Optional[Callable]


#: name -> ProtocolSpec. Populated by ``register_protocol``.
PROTOCOLS: Dict[str, ProtocolSpec] = {}


def register_protocol(name, client_cls, server_cls, trainer=None):
    """Register (or re-register) a method implementation by name."""
    if not isinstance(name, str) or not name:
        raise ValueError("protocol name must be a nonempty string")
    ClientProtocol.register(client_cls)
    ServerProtocol.register(server_cls)
    previous = PROTOCOLS.get(name)
    PROTOCOLS[name] = ProtocolSpec(
        name, client_cls, server_cls,
        trainer if trainer is not None else (previous.trainer if previous else None))
    return PROTOCOLS[name]


def register_trainer(name, trainer):
    """Attach the training entry point to an already registered protocol."""
    if name not in PROTOCOLS:
        raise KeyError("register_protocol('%s', ...) first; known: %s"
                       % (name, sorted(PROTOCOLS)))
    spec = PROTOCOLS[name]
    PROTOCOLS[name] = ProtocolSpec(spec.name, spec.client_cls, spec.server_cls, trainer)
    return PROTOCOLS[name]


def get_protocol(name='ours') -> ProtocolSpec:
    if name not in PROTOCOLS:
        raise KeyError("unknown protocol %r; registered: %s" % (name, sorted(PROTOCOLS)))
    return PROTOCOLS[name]


def available_protocols() -> Tuple[str, ...]:
    return tuple(sorted(PROTOCOLS))


def get_trainer(name='ours') -> Callable:
    """Training entry point of a protocol; Ours is ``federated_anchored_clustering``."""
    spec = get_protocol(name)
    if spec.trainer is None:
        raise KeyError("protocol %r has no registered trainer" % name)
    return spec.trainer
