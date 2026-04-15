import os
import subprocess
import threading
from contextlib import contextmanager


class GPUManager:
    def __init__(self, gpus):
        self.gpus = gpus
        self.lock = threading.Lock()
        self.available = list(gpus)

    def acquire(self):
        with self.lock:
            if not self.available:
                return None
            return self.available.pop(0)

    def release(self, gpu):
        if gpu is None:
            return
        with self.lock:
            # 增加安全检查，防止同一个 GPU 被意外重复释放导致可用列表越来越长
            if gpu not in self.available:
                self.available.append(gpu)
            else:
                print(f"Warning: GPU {gpu} is already in the available list.")

    @contextmanager
    def allocate(self):
        """
        使用上下文管理器安全地分配和释放 GPU
        """
        gpu = self.acquire()
        try:
            yield gpu
        finally:
            if gpu is not None:
                self.release(gpu)


def _get_available_gpus():
    """
    自动检测可用 GPU 列表，优先级：
    1. 环境变量 GPU_IDS（逗号分隔，如 "0,1,2"）
    2. nvidia-smi 自动检测
    3. 回退到 [0, 1, 2, 3]
    """
    gpu_env = os.environ.get("GPU_IDS")
    if gpu_env:
        return [int(g.strip()) for g in gpu_env.split(",") if g.strip().isdigit()]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip().isdigit()]
    except Exception:
        pass

    return [0, 1, 2, 3]  # fallback


gpu_manager = GPUManager(_get_available_gpus())
