import torch

print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)

    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(gpu.total_memory / 1024**3, 2), "GB")