import psutil
import os

def size_str(size_in_bytes: int) -> str:
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < (1024**2):
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < (1024**3):
        return f"{size_in_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_in_bytes / (1024 ** 3):.2f} GB"

def memory_usage() -> dict:
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {"rss": size_str(mem_info.rss), "vms": size_str(mem_info.vms)}