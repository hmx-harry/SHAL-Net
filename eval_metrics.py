import os
import sys
import warnings
from skimage.color import deltaE_ciede2000
import lpips
import skimage
import torch
from tabulate import tabulate
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
import glob
import decimal
from PIL import Image
import numpy as np

warnings.filterwarnings('ignore')


def PSNR(im1, im2):
    im1 = np.asarray(im1) if im1 is not np.ndarray else im1
    im2 = np.asarray(im2) if im2 is not np.ndarray else im2
    psnr = skimage.metrics.peak_signal_noise_ratio(im1, im2)
    return psnr


def SSIM(im1, im2):
    im1 = np.asarray(im1) if im1 is not np.ndarray else im1
    im2 = np.asarray(im2) if im2 is not np.ndarray else im2
    ssim = skimage.metrics.structural_similarity(im1, im2, multichannel=True, channel_axis=2)
    return ssim


def LPIPS(im1, im2):
    lpips_model = lpips.LPIPS(net='alex', verbose=False)
    im1_tensor = torch.tensor((np.array(im1).transpose(2, 0, 1).astype(np.float32)) / 255.0).unsqueeze(0)
    im2_tensor = torch.tensor((np.array(im2).transpose(2, 0, 1).astype(np.float32)) / 255.0).unsqueeze(0)
    return lpips_model(im1_tensor, im2_tensor).item()


def MAE(im1, im2):
    im1 = transforms.ToTensor()(im1)
    im2 = transforms.ToTensor()(im2)
    return nn.L1Loss()(im1, im2).item()


def DeltaE(im1, im2):
    im1 = np.asarray(im1) if im1 is not np.ndarray else im1
    im2 = np.asarray(im2) if im2 is not np.ndarray else im2
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2LAB).astype(np.float32)
    im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2LAB).astype(np.float32)
    return deltaE_ciede2000(im1, im2).mean()


def cal_score(GT_dir, img_dir):
    def num_round(num):
        decimal.getcontext().rounding = "ROUND_HALF_UP"
        try:
            num_r = decimal.Decimal(str(num)).quantize(decimal.Decimal("0.0000"))
            return str(num_r)
        except:
            return str(num)

    GT_list = glob.glob(os.path.join(GT_dir, '*.*'))
    img_list = glob.glob(os.path.join(img_dir, '*.*'))

    GT_list.sort()
    img_list.sort()
    psnr_l, ssim_l, lpips_l, deltae_l, mae_l = [], [], [], [], []
    for i in tqdm(range(len(GT_list)), file=sys.stdout):
        if os.path.basename(GT_list[i]) != os.path.basename(img_list[i]):
            return 'GT does not match input list'

        GT = Image.open(GT_list[i])
        img = Image.open(img_list[i])
        psnr_l.append(PSNR(GT, img))
        ssim_l.append(SSIM(GT, img))
        lpips_l.append(LPIPS(GT, img))
        deltae_l.append(DeltaE(GT, img))
        mae_l.append(MAE(GT, img))

    headers = ['PSNR↑', 'SSIM↑', 'LPIPS↓', 'MAE↓', 'DeltaE↓']
    data = [[num_round(np.mean(psnr_l)),
             num_round(np.mean(ssim_l)),
             num_round(np.mean(lpips_l)),
             num_round(np.mean(mae_l)),
             num_round(np.mean(deltae_l))]]
    print(tabulate(data, headers, tablefmt='grid'))


if __name__ == '__main__':
    gt_dir = 'datasets/test_data/reference/LOL_15'
    evl_dir = 'results/LOL_15/Enh'
    cal_score(gt_dir, evl_dir)
