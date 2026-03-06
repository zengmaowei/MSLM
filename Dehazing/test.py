import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils
from pytorch_msssim import ssim
from natsort import natsorted
from glob import glob
from basicsr.models.archs.MSSM import MSSM
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deraining using MB_TaylorFormer')
parser.add_argument('--size', default='L', type=str,choices=['B','L'], help='Path to weights')
parser.add_argument('--input_dir', default="Datasets/ITS_SOTS/input/", type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--target_dir', default="Datasets/ITS_SOTS/gt/", type=str, help='Directory for results')
parser.add_argument('--dataset', default='ITS', type=str, help='Test Dataset') # ['ITS, OTS, Haze4K, densehaze, nhhaze, ohaze']
parser.add_argument('--weights', default='../experiments/ITS-L/models/net_g_latest.pth'
                                         , type=str, help='Path to weights')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 例如 '0' 或 '1'

####### Load yaml #######
if args.size=='B':
    yaml_file = '../Dehazing/Options/MB-TaylorFormerV2-ITS-B.yml'
elif args.size=='L':
    yaml_file = '../Dehazing/Options/MB-TaylorFormerV2-ITS-L.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = MSSM(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint["params"], strict=False)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()



factor = 8
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)

target_dir = "/home/ubuntu/data/ITS/test/GT"
inp_dir = "/home/ubuntu/data/ITS/test/hazy"
files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
SSIM = []
PSNR=[]
with torch.no_grad():
    for file_ in tqdm(files):

        img = np.float32(utils.load_img(file_))/255.

        target=np.float32(utils.load_img(os.path.join(target_dir,file_.split('/')[-1])))/255.

        img = torch.from_numpy(img).permute(2,0,1)
        target=torch.from_numpy(target).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()
        target_ = target.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)
        # 模型返回的是多尺度输出列表，取最后一个（全分辨率）
        if isinstance(restored, list):
            print(f"Model returned list with {len(restored)} outputs")
            # 最后一个应该是全分辨率输出（1200×1600）
            restored = restored[-1]  # 取最后一个元素
            print(f"Using last output (full resolution): shape {restored.shape}")
                # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]
        
        output=restored.clamp_(0, 1)
        psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target_)).item()

        #down_ratio = max(1, round(min(H, W) / 256))  # Zhou Wang
        ssim_val = ssim(output,
                        target_,
                        data_range=1, size_average=False).item()
        base_filename = os.path.splitext(os.path.split(file_)[-1])[0]
         # ====== 打印每张图的指标 ======
        print(f"\nImage: {base_filename}")
        print(f"  PSNR: {psnr_val:.4f} dB")
        print(f"  SSIM: {ssim_val:.4f}")
        # =============================
        
        
        # 格式化数值：PSNR保留2位小数，SSIM保留4位小数
        psnr_str = f"{psnr_val:.2f}".replace('.', 'p')  # 例如: 25.67 -> 25p67
        ssim_str = f"{ssim_val:.4f}".replace('.', 'p')  # 例如: 0.8765 -> 0p8765
        
        # 构建新文件名
        new_filename = f"{base_filename}_PSNR_{psnr_str}_SSIM_{ssim_str}.png"
    
        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        #utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))
        utils.save_img((os.path.join(result_dir, new_filename)), img_as_ubyte(restored))

        PSNR.append(psnr_val)
        SSIM.append(ssim_val)



    print('final PSNR:',np.mean(PSNR),'final SSIM:',np.mean(SSIM))
            