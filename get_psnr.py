import argparse
import random
import os, os.path as osp

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

import time
import argparse
from matplotlib import pyplot as plt

from Util.network_util import Build_Generator_From_Dict, Convert_Tensor_To_Image
from Evaluation.image_projection.image_projector import Image_Projector, im2tensor, Get_LPIPS_Model_Image, Get_PSNR_Model_Image
import lpips
from Util.content_aware_pruning import Get_Parsing_Net, Batch_Img_Parsing, Get_Masked_Tensor

from torchvision import transforms, utils

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str)
parser.add_argument('--size', type=int)
parser.add_argument('--kernel_size', type=int)
args = parser.parse_args()

device = 'cuda:0'
gpu_device_ids = [0]

def fuck(args):

    args.num_iters = 800
    args.info_print = False
    args.generated_img_size = args.size
    path = '/home/xuguodong/DATA/FFHQ/images1024x1024/'

    transform = transforms.Compose([
                transforms.Resize(args.generated_img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])

    model_dict = torch.load(args.ckpt, map_location=device)
    g_ema = Build_Generator_From_Dict(model_dict['g_ema'], size=args.generated_img_size, kernel_size=args.kernel_size).to(device)
    g_ema = nn.DataParallel(g_ema, device_ids=gpu_device_ids)
    g_ema.eval()

    parsing_net, _ = Get_Parsing_Net(device)

    files = os.listdir('Helen/')
    lpips_percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=[int(device[-1])])
    lpips_percept.eval()
    LPIPS, ca_lpips, PSNR, ca_psnr = [],[],[],[]
    result, masked_result = [],[]
    target, masked_target = [],[]
    for file in files:
        image_file = osp.join('Helen', file)
        image = Image.open(image_file).convert("RGB")
        transformed_image = transform(image).to(device)
        target_img = image.resize((args.generated_img_size, args.generated_img_size))

        if args.info_print:
            print_iters = 100
        else:
            print_iters = np.inf
        input_img_tensor, output_img_tensor = Image_Projector(generator = g_ema,
                        device = device,
                        per_layer_W = True,
                        target_im = target_img,
                        opt = 'LBFGS',
                        num_iters = args.num_iters,
                        print_iters = print_iters)

        output_img_tensor = output_img_tensor.to(device)
        with torch.no_grad():
            mask = Batch_Img_Parsing(transformed_image[None,...], parsing_net, device)

        target_img = np.array(target_img)
        target_img_tensor = im2tensor(target_img)
        target_img_tensor = target_img_tensor.to(device)
        lpips_score = Get_LPIPS_Model_Image([output_img_tensor], [target_img_tensor], lpips_percept)[0][0]
        LPIPS.append(round(lpips_score, 4))

        masked_output_img_tensor = Get_Masked_Tensor(output_img_tensor, mask, device, mask_grad=False)
        masked_target_img_tensor = Get_Masked_Tensor(target_img_tensor, mask, device, mask_grad=False)
        lpips_score = Get_LPIPS_Model_Image([masked_output_img_tensor], [masked_target_img_tensor], lpips_percept)[0][0]
        ca_lpips.append(round(lpips_score, 4))

        output_img = np.array(Convert_Tensor_To_Image(output_img_tensor))
        result.append(output_img)
        target.append(target_img)
        psnr_score = Get_PSNR_Model_Image([output_img], [target_img])[0][0]
        PSNR.append(round(psnr_score, 4))

        masked_output_img = np.array(Convert_Tensor_To_Image(masked_output_img_tensor))
        masked_target_img = np.array(Convert_Tensor_To_Image(masked_target_img_tensor))
        masked_result.append(masked_output_img)
        masked_target.append(masked_target_img)
        psnr_score = Get_PSNR_Model_Image([masked_output_img], [masked_target_img])[0][0]
        ca_psnr.append(round(psnr_score, 4))
    return  np.array(LPIPS), np.array(ca_lpips), np.array(PSNR), np.array(ca_psnr)

a,b,c,d = fuck(args)
#a,b,c,d = np.array([1.1111]),np.array([1.1111]),np.array([1.1111]),np.array([1.1111])
tmp = osp.abspath(args.ckpt).split('content-aware-gan-compression')[1][1:-3].replace('/', '-')
os.makedirs(f'psnr/{tmp}', exist_ok=True)
np.save(f'psnr/{tmp}/lpips.npy', a)
np.save(f'psnr/{tmp}/ca_lpips.npy', b)
np.save(f'psnr/{tmp}/psnr.npy', c)
np.save(f'psnr/{tmp}/ca_psnr.npy', d)
with open('psnr.txt', 'a') as f:
    line = f'{tmp}:  {a.mean():.4f}, {b.mean():.4f}, {c.mean():.4f}, {d.mean():.4f}\n'
    f.writelines(line)
