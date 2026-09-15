import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import numpy as np
from PIL import Image


def load_img(filepath):
    """模拟原始代码中的utils.load_img"""
    img = Image.open(filepath).convert('RGB')
    return np.array(img)


def calculate_psnr_ssim(gt_path, pred_path):
    """
    完全按照原始代码方式计算PSNR和SSIM
    """
    # 使用与原始代码完全相同的方式加载图片
    gt_img = np.float32(load_img(gt_path)) / 255.0
    pred_img = np.float32(load_img(pred_path)) / 255.0

    # 转换为tensor，与原始代码完全相同
    gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1).unsqueeze(0)

    # 计算PSNR（与原始代码完全相同）
    psnr = 10 * torch.log10(1 / F.mse_loss(pred_tensor, gt_tensor)).item()

    # 计算SSIM（与原始代码完全相同）
    # data_range=1 因为图片已经归一化到[0,1]
    ssim_val = ssim(pred_tensor, gt_tensor, data_range=1, size_average=False).item()

    return psnr, ssim_val


if __name__ == "__main__":
    # 替换成您的图片路径
    gt_path = "/home/ubuntu/data/ITS/test/GT/1439_1.png"
    pred_path = "/home/ubuntu/zengmaowei/MB-TaylorFormerV2-main/MB-TaylorFormerV2-main/results/ITS/1439_1_PSNR_42p71_SSIM_0p9978.png"

    psnr, ssim_val = calculate_psnr_ssim(gt_path, pred_path)

    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim_val:.4f}")