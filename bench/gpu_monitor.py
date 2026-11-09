import subprocess, threading, time

def gpu_now():
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu",
        "--format=csv,noheader,nounits"]).decode().strip()
    mem, util, sm, temp = (int(float(x)) for x in out.split(","))
    return mem, util, sm, temp

def host_mem():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.split()[0]) // 1024      # kB -> MiB
    return info["MemAvailable"], info["SwapTotal"] - info["SwapFree"]

class GpuMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append(gpu_now() + host_mem())
            except Exception:
                pass                                  # one failed sample shouldn't kill a run
            time.sleep(self.interval)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join()
        if not self.samples:
            return {}
        mem, util, sm, temp, avail, swap = zip(*self.samples)
        return {
            "vram_peak_mib": max(mem),
            "gpu_util_avg": round(sum(util) / len(util), 1),
            "sm_clock_min_mhz": min(sm),
            "temp_max_c": max(temp),
            "host_avail_min_mib": min(avail),
            "swap_max_mib": max(swap),
        }
