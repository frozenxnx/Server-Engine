import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
free, total = torch.cuda.mem_get_info()
print(f"VRAM free/total: {free/2**30:.2f} / {total/2**30:.2f} GiB")
