"""Public PACE-VIO factor-packet API.

The implementation remains importable through ``Utility.T2FactorPacket`` so
archived experiments continue to run, but new code should use this module.
"""

from Utility.T2FactorPacket import PACEFactorPacket

__all__ = ["PACEFactorPacket"]
