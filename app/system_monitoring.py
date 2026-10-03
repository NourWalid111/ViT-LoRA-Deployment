import os
import time

import psutil


# ============================================================
# PROCESS START TIME
# ============================================================

PROCESS_START_TIME = time.time()


# ============================================================
# SYSTEM METRICS
# ============================================================

def get_system_metrics():
    """
    Collect infrastructure metrics for the API process
    and the host system.
    """

    process = psutil.Process(
        os.getpid()
    )

    memory = psutil.virtual_memory()

    disk = psutil.disk_usage(
        "/"
    )

    uptime_seconds = (
        time.time()
        - PROCESS_START_TIME
    )

    return {
        "cpu_percent": psutil.cpu_percent(
            interval=0.1
        ),

        "memory_percent": memory.percent,

        "memory_used_mb": round(
            memory.used / (1024 * 1024),
            2,
        ),

        "memory_available_mb": round(
            memory.available / (1024 * 1024),
            2,
        ),

        "process_memory_mb": round(
            process.memory_info().rss
            / (1024 * 1024),
            2,
        ),

        "disk_percent": disk.percent,

        "disk_free_gb": round(
            disk.free
            / (1024 * 1024 * 1024),
            2,
        ),

        "uptime_seconds": round(
            uptime_seconds,
            2,
        ),
    }