import torch
import pynvml
from typing import List, Dict, Optional, Any
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUManager:
    def __init__(self):
        """Initialize GPU manager with NVML for monitoring."""
        self.device_count = torch.cuda.device_count()
        pynvml.nvmlInit()
        self.handles = []
        self._initialize_gpu_handles()
        
    def _initialize_gpu_handles(self):
        """Initialize NVML handles for all available GPUs."""
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            self.handles.append(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            logger.info(f"Initialized GPU {i}: {name.encode()}")
            
    @property
    def nvlink_status(self) -> Dict[str, Dict[str, Any]]:
        """Get NVLink status between all available GPUs"""
        status = {}
        
        try:
            for i in range(len(self.handles)):
                for j in range(i + 1, len(self.handles)):
                    link_name = f"{i}-{j}"
                    
                    # Check NVLink version and status
                    try:
                        version = pynvml.nvmlDeviceGetNvLinkVersion(self.handles[i], j)
                        # NVLink is active if we can get utilization counters
                        try:
                            pynvml.nvmlDeviceGetNvLinkUtilizationCounter(self.handles[i], j, 0)
                            active = True
                        except:
                            active = False
                            
                        status[link_name] = {
                            'version': version,
                            'active': active
                        }
                    except pynvml.NVMLError:
                        continue
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get NVLink status: {str(e)}")
            return {}
    
    def get_gpu_utilization(self) -> List[float]:
        """Get current GPU utilization for all devices."""
        utilization = []
        for handle in self.handles:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization.append(util.gpu)
        return utilization
    
    def get_memory_info(self) -> List[Dict[str, int]]:
        """Get memory information for all GPUs."""
        memory_info = []
        for handle in self.handles:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_info.append({
                'total': info.total,
                'used': info.used,
                'free': info.free
            })
        return memory_info
    
    def get_nvlink_throughput(self, gpu1: int, gpu2: int) -> Optional[float]:
        """Get NVLink throughput between two GPUs in GB/s"""
        key = f"{min(gpu1, gpu2)}-{max(gpu1, gpu2)}"
        if key not in self.nvlink_status or not self.nvlink_status[key]['active']:
            return None
            
        try:
            # Get TX and RX counters
            tx = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(self.handles[gpu1], gpu2, 0)
            rx = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(self.handles[gpu1], gpu2, 1)
            
            # Convert to GB/s (assuming max NVLink bandwidth of 50 GB/s)
            throughput = (tx + rx) * 50.0 / 100.0
            return throughput
        except pynvml.NVMLError:
            return None

    def get_nvlink_throughput(self, link: str) -> float:
        """Get the current throughput for a specific NVLink connection in GB/s"""
        try:
            # Parse GPU indices from link (format: "0-1")
            gpu1, gpu2 = map(int, link.split('-'))
            
            # Get NVLink throughput for both directions
            handle1 = self.handles[gpu1]
            handle2 = self.handles[gpu2]
            
            # Get NVLink utilization counters
            counter_tx = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(handle1, gpu2, 0)  # 0 for TX
            counter_rx = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(handle1, gpu2, 1)  # 1 for RX
            
            # Convert to GB/s (NVLink 3.0 max bandwidth is 50 GB/s per link)
            max_bandwidth = 50.0  # GB/s for NVLink 3.0
            tx_throughput = (counter_tx * max_bandwidth) / 100.0
            rx_throughput = (counter_rx * max_bandwidth) / 100.0
            
            # Return total bidirectional throughput
            return tx_throughput + rx_throughput
            
        except Exception as e:
            logger.error(f"Failed to get NVLink throughput for link {link}: {str(e)}")
            return 0.0
    
    def __del__(self):
        """Cleanup NVML on deletion."""
        try:
            pynvml.nvmlShutdown()
        except:
            pass
