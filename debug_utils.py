import logging
# Utility to get memory stats, using self-contained hardware_info module.

def log_memory_usage(label=""):
    try:
        from .hardware_info import CHardwareInfo

        hardware_info = CHardwareInfo(switchRAM=True, switchVRAM=True)
        status = hardware_info.getStatus()
        
        ram_used_gb = status.get('ram_used', 0) / (1024**3)
        ram_total_gb = status.get('ram_total', 0) / (1024**3)
        
        log_message = f"[MEM_DEBUG] {label} | RAM Used: {ram_used_gb:.2f}/{ram_total_gb:.2f} GB"
        
        if 'gpus' in status:
            for i, gpu in enumerate(status['gpus']):
                vram_used_gb = gpu.get('vram_used', 0) / (1024**3)
                vram_total_gb = gpu.get('vram_total', 0) / (1024**3)
                log_message += f" | VRAM cuda:{i}: {vram_used_gb:.2f}/{vram_total_gb:.2f} GB"
        
        logging.info(log_message)

    except ImportError:
        logging.warning("[MEM_DEBUG] Could not import local CHardwareInfo. Cannot log memory usage.")
    except Exception as e:
        logging.error(f"[MEM_DEBUG] Error getting memory usage: {e}")
