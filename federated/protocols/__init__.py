"""Method-agnostic contracts and the protocol registry.

See :mod:`federated.protocols.protocol` for the contract and for how to plug a
new baseline method in. Importing this package never imports an implementation,
so registration cannot create an import cycle.
"""
from federated.protocols.protocol import (
    PROTOCOLS, ClientProtocol, ProtocolSpec, RoundPayload, ServerProtocol,
    available_protocols, get_protocol, get_trainer, register_protocol, register_trainer,
)

__all__ = ['PROTOCOLS', 'ClientProtocol', 'ProtocolSpec', 'RoundPayload', 'ServerProtocol',
           'available_protocols', 'get_protocol', 'get_trainer', 'register_protocol',
           'register_trainer']
