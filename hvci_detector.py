"""
HVCI (Hypervisor-enforced Code Integrity) Detection Module

Detects if Windows Memory Integrity (HVCI) is enabled, which causes
mmap + multi-threading conflicts in MultiGPU checkpoint loading.

This allows automatic selection of appropriate loading strategy:
- HVCI enabled: Use tensor copying workaround
- HVCI disabled: Use original mmap-based loading (faster)
"""

import platform
import logging

logger = logging.getLogger("MultiGPU.HVCI")

_hvci_status_cache = None
_hvci_check_attempted = False


def is_windows():
    """Check if running on Windows."""
    return platform.system() == "Windows"


def check_hvci_enabled():
    """
    Check if Windows HVCI (Memory Integrity) is enabled.
    
    Returns:
        bool: True if HVCI is enabled, False if disabled or cannot determine.
        None: If not on Windows or check failed.
    """
    global _hvci_status_cache, _hvci_check_attempted
    
    # Return cached result if already checked
    if _hvci_check_attempted:
        return _hvci_status_cache
    
    _hvci_check_attempted = True
    
    # Only check on Windows
    if not is_windows():
        logger.debug("[HVCI] Not running on Windows, HVCI check skipped.")
        _hvci_status_cache = False
        return False
    
    try:
        import subprocess
        
        # Method 1: Check via WMI Win32_DeviceGuard
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "Get-CimInstance -ClassName Win32_DeviceGuard | Select-Object -ExpandProperty SecurityServicesRunning"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        
        if result.returncode == 0 and result.stdout:
            output = result.stdout.strip()
            # SecurityServicesRunning values:
            # 0 = None
            # 1 = Credential Guard
            # 2 = HVCI (Hypervisor-enforced Code Integrity)
            # {1, 2} = Both enabled
            
            if '2' in output:
                logger.info("[HVCI] ✅ HVCI (Memory Integrity) is ENABLED")
                logger.info("[HVCI] → Will use tensor copying workaround for stability")
                _hvci_status_cache = True
                return True
            else:
                logger.info("[HVCI] ❌ HVCI (Memory Integrity) is DISABLED")
                logger.info("[HVCI] → Will use original mmap loading for best performance")
                _hvci_status_cache = False
                return False
        
        # Method 2: Fallback - Check registry
        logger.debug("[HVCI] WMI check failed, trying registry method...")
        cmd_reg = [
            "reg", "query",
            "HKLM\\SYSTEM\\CurrentControlSet\\Control\\DeviceGuard\\Scenarios\\HypervisorEnforcedCodeIntegrity",
            "/v", "Enabled"
        ]
        
        result_reg = subprocess.run(
            cmd_reg,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        
        if result_reg.returncode == 0 and "0x1" in result_reg.stdout:
            logger.info("[HVCI] ✅ HVCI is ENABLED (detected via registry)")
            _hvci_status_cache = True
            return True
        elif result_reg.returncode == 0 and "0x0" in result_reg.stdout:
            logger.info("[HVCI] ❌ HVCI is DISABLED (detected via registry)")
            _hvci_status_cache = False
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning("[HVCI] ⚠️ HVCI check timed out, assuming disabled")
    except Exception as e:
        logger.warning(f"[HVCI] ⚠️ HVCI check failed: {e}, assuming disabled")
    
    # Default to False (disabled) if check fails
    # This is the safe default - uses workaround if we can't determine
    logger.info("[HVCI] ⚠️ Cannot determine HVCI status, defaulting to workaround mode")
    _hvci_status_cache = None  # Unknown state
    return None


def should_use_mmap_workaround():
    """
    Determine if mmap workaround should be used based on HVCI status.
    
    Returns:
        bool: True if workaround should be used (HVCI enabled or unknown on Windows),
              False if safe to use original mmap loading.
    """
    if not is_windows():
        # Linux/Mac can safely use mmap with multi-threading
        return False
    
    hvci_status = check_hvci_enabled()
    
    if hvci_status is True:
        # HVCI is enabled - must use workaround
        return True
    elif hvci_status is False:
        # HVCI is disabled - safe to use mmap
        return False
    else:
        # Unknown status on Windows - use workaround to be safe
        logger.warning("[HVCI] ⚠️ HVCI status unknown, using workaround for safety")
        return True


def get_hvci_status_string():
    """Get human-readable HVCI status string."""
    if not is_windows():
        return "N/A (not Windows)"
    
    status = check_hvci_enabled()
    if status is True:
        return "Enabled (using workaround)"
    elif status is False:
        return "Disabled (using mmap)"
    else:
        return "Unknown (using workaround)"


def force_recheck():
    """Force re-checking HVCI status (useful for testing)."""
    global _hvci_status_cache, _hvci_check_attempted
    _hvci_status_cache = None
    _hvci_check_attempted = False
    return check_hvci_enabled()


if __name__ == "__main__":
    # Test the detector
    logging.basicConfig(level=logging.INFO)
    print(f"Platform: {platform.system()}")
    print(f"HVCI Enabled: {check_hvci_enabled()}")
    print(f"Should use workaround: {should_use_mmap_workaround()}")
    print(f"Status: {get_hvci_status_string()}")
