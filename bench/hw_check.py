import time, torch

def timeit(fn, n=20):
    fn(); torch.cuda.synchronize()          # warmup, then wait for GPU
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()                # GPU work is async: sync before stopping the clock
    return (time.perf_counter() - t) / n

# 1. Device memory bandwidth: copy 256 MiB (reads 256, writes 256)
x = torch.empty(2**27, dtype=torch.float16, device="cuda")
y = torch.empty_like(x)
t = timeit(lambda: y.copy_(x))
print(f"GPU memory bandwidth : {2 * x.numel() * 2 / t / 1e9:6.1f} GB/s  (spec ~192)")

# 2. fp16 matmul throughput: 4096x4096 @ 4096x4096
a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
b = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
t = timeit(lambda: a @ b)
print(f"fp16 matmul          : {2 * 4096**3 / t / 1e12:6.2f} TFLOPS (peak ~5)")

# 3. Kernel launch overhead: a tiny op on a 1-element tensor
s = torch.zeros(1, device="cuda")
t = timeit(lambda: s.add_(1), n=2000)
print(f"Launch overhead      : {t * 1e6:6.1f} us per tiny kernel")

# 4. Host -> GPU transfer speed (PCIe)
h = torch.empty(2**27, dtype=torch.float16).pin_memory()
d = torch.empty(2**27, dtype=torch.float16, device="cuda")
t = timeit(lambda: d.copy_(h, non_blocking=True))
print(f"Host->GPU (PCIe)     : {h.numel() * 2 / t / 1e9:6.1f} GB/s")
