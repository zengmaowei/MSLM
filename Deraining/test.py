import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.MSSM import MSSM
from skimage import img_as_ubyte
from pdb import set_trace as stx
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser(description='Image Deraining using MB_TaylorFormer')

parser.add_argument('--input_dir', default='/home/ubuntu/data/Deraining/deraining_test/deraining_testset/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/home/ubuntu/zengmaowei/MB-TaylorFormerV2-main/MB-TaylorFormerV2-main/experiments/Rain100H-L/models/net_g_1295595.pth', type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
yaml_file = '/home/ubuntu/zengmaowei/MB-TaylorFormerV2-main/MB-TaylorFormerV2-main/Dehazing/Options/MB-TaylorFormerV2-Rain100H-L.yml'
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
model_restoration.load_state_dict(checkpoint['params'], strict=False)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']

for dataset in datasets:
    result_dir  = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    inp_dir = os.path.join(args.input_dir, dataset, 'input')
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_))/255.
            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            restored = model_restoration(input_)
            
            # Handle model output format - take the last scale (original resolution)
            if isinstance(restored, list):
                restored = restored[-1]  # Take the last scale (original resolution)

            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]

            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))
