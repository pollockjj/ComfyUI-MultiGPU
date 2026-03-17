"""P2P accessibility registry for multi-GPU DLPack operations.

Caches cudaDeviceCanAccessPeer results per GPU pair to avoid
repeated CUDA runtime API calls.
"""

import ctypes
import logging
import torch

logger = logging.getLogger("MultiGPU")

_libcudart = None


def _get_libcudart():
    """Load libcudart.so once and cache the handle."""
    global _libcudart
    if _libcudart is None:
        _libcudart = ctypes.CDLL("libcudart.so")
    return _libcudart


class MultiGPUP2PRegistry:
    """Cached registry for CUDA peer-to-peer accessibility between GPU pairs.

    Uses the CUDA runtime cudaDeviceCanAccessPeer API directly via ctypes
    because torch.cuda.can_access_peer does not exist in PyTorch 2.10.0+.
    Results are cached per (src, dst) pair for the lifetime of the registry.
    """

    def __init__(self):
        self._cache = {}

    @staticmethod
    def _raw_can_access_peer(device_a: int, device_b: int) -> bool:
        """Call cudaDeviceCanAccessPeer via ctypes. Returns True if P2P is available."""
        lib = _get_libcudart()
        can_access = ctypes.c_int(0)
        result = lib.cudaDeviceCanAccessPeer(ctypes.byref(can_access), device_a, device_b)
        if result != 0:
            logger.warning(
                f"[MultiGPU P2P] cudaDeviceCanAccessPeer({device_a}, {device_b}) "
                f"returned error code {result}, assuming no P2P"
            )
            return False
        return bool(can_access.value)

    def can_access_peer(self, src_device: int, dst_device: int) -> bool:
        """Check if src_device can access dst_device memory via P2P.

        Results are cached per (src, dst) pair.
        """
        if src_device == dst_device:
            return True

        key = (src_device, dst_device)
        if key not in self._cache:
            if not torch.cuda.is_available():
                self._cache[key] = False
            else:
                result = self._raw_can_access_peer(src_device, dst_device)
                self._cache[key] = result
                logger.info(
                    f"[MultiGPU P2P] can_access_peer({src_device}, {dst_device}) = {result}"
                )
        return self._cache[key]

    def clear_cache(self):
        """Clear the P2P cache (useful for testing)."""
        self._cache.clear()


# Module-level singleton
p2p_registry = MultiGPUP2PRegistry()
