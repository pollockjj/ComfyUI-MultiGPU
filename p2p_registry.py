"""P2P accessibility registry for multi-GPU DLPack operations.

Caches cudaDeviceCanAccessPeer results per GPU pair to avoid
repeated CUDA runtime API calls.
"""

import ctypes
import glob
import logging
import os
import torch

logger = logging.getLogger("MultiGPU")

_libcudart = None


def _cudart_candidates():
    """CUDA runtime library names to try, most-specific first.

    On Windows the runtime is cudart64_<major>.dll, not libcudart.so. Prefer the
    copy PyTorch ships in torch/lib so the version always matches the build in
    use; then PATH-resolved names; then the POSIX sonames.
    """
    names = []
    if os.name == "nt":
        try:
            libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
            names.extend(sorted(glob.glob(os.path.join(libdir, "cudart64_*.dll")), reverse=True))
        except Exception:
            pass
        names.extend(["cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll", "cudart64_101.dll"])
    names.extend(["libcudart.so", "libcudart.dylib"])
    return names


def _get_libcudart():
    """Load the CUDA runtime once and cache the handle."""
    global _libcudart
    if _libcudart is None:
        last_err = None
        for name in _cudart_candidates():
            try:
                _libcudart = ctypes.CDLL(name)
                logger.debug(f"[MultiGPU P2P] loaded CUDA runtime: {name}")
                break
            except OSError as e:
                last_err = e
        if _libcudart is None:
            raise OSError(f"could not load CUDA runtime (tried {_cudart_candidates()})") from last_err
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
        """Return True if P2P is available between the two devices.

        Prefers torch.cuda.can_device_access_peer, which does exist in torch 2.x
        (the previous comment referred to "can_access_peer", which is not the
        API name -- so this always fell through to ctypes).

        This is an OPTIMIZATION probe: a False answer only means transfers get
        staged through host memory. Every failure mode therefore degrades to
        False rather than propagating -- an unloadable CUDA runtime must never
        abort a render.
        """
        fn = getattr(torch.cuda, "can_device_access_peer", None)
        if fn is not None:
            try:
                return bool(fn(device_a, device_b))
            except Exception as e:
                logger.warning(
                    f"[MultiGPU P2P] torch.cuda.can_device_access_peer({device_a}, {device_b}) "
                    f"failed ({e}); falling back to the CUDA runtime"
                )

        try:
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
        except Exception as e:
            logger.warning(
                f"[MultiGPU P2P] could not probe P2P for ({device_a}, {device_b}): {e}. "
                f"Assuming no P2P; transfers will be staged through host memory."
            )
            return False

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
