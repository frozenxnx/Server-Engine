import argparse, asyncio, itertools, json, os, re, signal, subprocess, sys, time
import httpx
sys.path.insert(0, os.path.dirname(__file__))
from gpu_monitor import GpuMonitor, gpu_now, host_mem
from load_gen import run

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8000
MAX_NUM_SEQS = [8, 32, 64]
MAX_BATCHED_TOKENS = [256, 2048]
CONCURRENCIES = [8, 32, 64]

def server_up():
    try:
        return httpx.get(f"http://localhost:{PORT}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False

def start_server(seqs, batched, mode):
    cmd = ["vllm", "serve", MODEL, "--dtype", "float32",
           "--max-model-len", "2048",
           "--gpu-memory-utilization", "0.70",
           "--max-num-seqs", str(seqs),
           "--max-num-batched-tokens", str(batched),
           "--port", str(PORT)]
    if mode == "eager":
        cmd.append("--enforce-eager")
    logpath = f"results/server_{mode}_{seqs}_{batched}.log"
    log = open(logpath, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc, logpath

def wait_ready(proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup, check its log")
        if server_up():
            return
        time.sleep(2)
    raise TimeoutError("server never became ready")

def kv_tokens(logpath):
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", open(logpath).read())
    return int(m.group(1).replace(",", "")) if m else None

def stop_server(proc):
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    for _ in range(30):                      # wait until VRAM is really released
        if gpu_now()[0] < 600:
            return
        time.sleep(1)

def wait_cool(max_temp=60, timeout=240):
    t0 = time.time()
    while gpu_now()[3] > max_temp and time.time() - t0 < timeout:
        time.sleep(5)

def bench(concurrency):
    ns = argparse.Namespace(model=MODEL, port=PORT, concurrency=concurrency,
                            num_requests=max(16, concurrency * 3),
                            prompt_words=512, max_tokens=128)
    return asyncio.run(run(ns))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eager", "graphs"], default="eager")
    mode = ap.parse_args().mode

    os.makedirs("results", exist_ok=True)
    if server_up():
        sys.exit("A server is already running on this port. Stop it first.")

    for seqs, batched in itertools.product(MAX_NUM_SEQS, MAX_BATCHED_TOKENS):
        print(f"\n=== {mode} seqs={seqs} batched={batched} ===", flush=True)
        wait_cool()
        proc, logpath = start_server(seqs, batched, mode)
        try:
            wait_ready(proc)
            kv = kv_tokens(logpath)
            print(f"KV cache: {kv} tokens", flush=True)
            bench(4)                         # warmup, discarded
            for c in CONCURRENCIES:
                wait_cool()
                mon = GpuMonitor()
                mon.start()
                s = bench(c)
                s.update(mon.stop())
                s.update({"mode": mode, "dtype": "float32",
                          "max_num_seqs": seqs, "max_num_batched_tokens": batched,
                          "kv_cache_tokens": kv})
                with open("results/results.jsonl", "a") as f:
                    f.write(json.dumps(s) + "\n")
                print(s, flush=True)
        except Exception as e:
            print(f"CONFIG FAILED: {e}", flush=True)
        finally:
            stop_server(proc)
