import torch

ckpt = torch.load("./experiments/ITS-L/models/net_g_2784010.pth", map_location="cpu")

# 兼容不同保存格式
if isinstance(ckpt, dict) and "params" in ckpt:
    sd = ckpt["params"]
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    sd = ckpt["state_dict"]
elif isinstance(ckpt, dict):
    sd = ckpt
else:
    sd = ckpt.state_dict()

bad = []
max_abs = 0.0
max_name = None

for k, v in sd.items():
    if not torch.is_tensor(v):
        continue
    v = v.float()
    if torch.isnan(v).any() or torch.isinf(v).any():
        bad.append(k)
    m = v.abs().max().item() if v.numel() else 0.0
    if m > max_abs:
        max_abs = m
        max_name = k

print("NaN/Inf tensors:", len(bad))
for k in bad[:20]:
    print("  ", k)
print("max(abs(w)):", max_abs, "at", max_name)