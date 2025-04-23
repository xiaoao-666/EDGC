import os
import glob
import time
import numpy as np
import torch

# 设置梯度路径
gradient_dir = r"C:\Users\16663\Desktop\fsdownload\gradients\1000\1000"

# 匹配的文件模式
patterns = [
    "layer_0_module_language_model_encoder_layers_0_*.npy",
    "layer_0_module_language_model_encoder_layers_1_*.npy",
    "layer_0_module_language_model_encoder_layers_2_*.npy",
    "layer_0_module_language_model_encoder_layers_3_*.npy",
    "layer_0_module_language_model_encoder_layers_4_*.npy",
    "layer_0_module_language_model_encoder_layers_5_*.npy",
]

# ✅ 设置计算设备（"cuda" or "cpu"）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# ✅ 熵计算函数（支持设备切换）
def calculate_entropy(grad, bins=1000, sample_ratio=1):
    grad = grad.to(device)
    stride = int(1 / sample_ratio)
    flat_grad = grad.flatten()
    mask = torch.isfinite(flat_grad)
    if not mask.any():
        return torch.tensor(0.0, device=device)
    valid_grad = flat_grad[mask]
    #sample_grad = valid_grad[torch.arange(0, valid_grad.numel(), stride)]
    sample_grad = valid_grad[::stride]
    min_val, max_val = torch.min(sample_grad), torch.max(sample_grad)
    hist = torch.histc(sample_grad, bins=bins, min=min_val.item(), max=max_val.item())
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    diff_entropy = -torch.sum(hist * torch.log(hist))
    return diff_entropy

# ✅ 遍历每组 pattern 文件，计算总熵与总耗时
for pattern in patterns:
    file_pattern = os.path.join(gradient_dir, pattern)
    file_list = glob.glob(file_pattern)

    total_entropy = 0.0
    total_time = 0.0

    for file_path in file_list:
        grad_np = np.load(file_path)
        grad_tensor = torch.from_numpy(grad_np).float().to(device)

        start_time = time.time()
        entropy = calculate_entropy(grad_tensor)
        end_time = time.time()
        total_entropy += entropy.item()
        #print((end_time - start_time) * 1000)
        total_time = total_time + (end_time - start_time) * 1000  # 毫秒

        print(f"[{os.path.basename(file_path)}] Entropy: {entropy.item():.6f}")


    print(f"\nPattern: {pattern}")
    print(f"Total Entropy: {total_entropy:.6f}")
    print(f"Total Time: {total_time:.2f} ms\n")
