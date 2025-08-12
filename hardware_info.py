import torch
import logging
import psutil
import comfy.model_management as mm

# --- Surgically lifted from ComfyUI-Crystools by Crystian ---
# This is a self-contained version of the necessary hardware monitoring classes
# to avoid cross-node import issues related to load order.

# --- From gpu.py ---
class CGPUInfo:
    def __init__(self):
        self.pynvmlLoaded = False
        self.cudaDevicesFound = 0
        self.gpus = []

        try:
            import pynvml
            self.pynvml = pynvml
            self.pynvml.nvmlInit()
            self.pynvmlLoaded = True
            self.cudaDevicesFound = self.pynvml.nvmlDeviceGetCount()
            
            for i in range(self.cudaDevicesFound):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_name = self.pynvml.nvmlDeviceGetName(handle)
                self.gpus.append({'index': i, 'name': gpu_name})

        except ImportError:
            logging.warning("[MultiGPU Hardware] pynvml not installed, VRAM monitoring disabled.")
        except Exception as e:
            logging.error(f"[MultiGPU Hardware] Could not initialize pynvml: {e}")

    def getStatus(self):
        gpus_status = []
        if self.pynvmlLoaded:
            for i in range(self.cudaDevicesFound):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    mem = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpus_status.append({
                        'vram_total': mem.total,
                        'vram_used': mem.used,
                        'vram_used_percent': mem.used / mem.total * 100 if mem.total > 0 else 0,
                    })
                except Exception as e:
                    logging.error(f"Could not get VRAM info for GPU {i}: {e}")
                    gpus_status.append({'vram_total': 0, 'vram_used': 0, 'vram_used_percent': 0})
        
        return {
            'device_type': 'cuda' if self.pynvmlLoaded else 'cpu',
            'gpus': gpus_status,
        }

# --- From hardware.py ---
class CHardwareInfo:
    def __init__(self, switchRAM=False, switchVRAM=False):
        self.switchRAM = switchRAM
        self.GPUInfo = CGPUInfo()
        self.switchVRAM = switchVRAM

    def getStatus(self):
        ramTotal = -1
        ramUsed = -1
        ramUsedPercent = -1

        if self.switchRAM:
            ram = psutil.virtual_memory()
            ramTotal = ram.total
            ramUsed = ram.used
            ramUsedPercent = ram.percent

        gpu_status = {'device_type': 'cpu', 'gpus': []}
        if self.switchVRAM:
            gpu_status = self.GPUInfo.getStatus()

        return {
            'ram_total': ramTotal,
            'ram_used': ramUsed,
            'ram_used_percent': ramUsedPercent,
            'device_type': gpu_status['device_type'],
            'gpus': gpu_status['gpus'],
        }
