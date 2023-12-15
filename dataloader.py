import os
import torch.utils.data as data
import cv2
import numpy as np
from PIL import Image
import glob
from torchvision import transforms


class LoadDataset(data.Dataset):
    def __init__(self, data_dir, tr, img_size=None):
        self.data_dir = data_dir
        self.transform = tr
        self.img_size = img_size
        self.img_list = self.load_imglist()
        self.img_list = self.load_imglist()

    def __getitem__(self, idx):
        img, img_clahe, name = self.load_img(self.img_list[idx])
        return img, img_clahe, name

    def __len__(self):
        return len(self.img_list)

    def load_img(self, img_path):
        img = Image.open(img_path)
        if self.img_size != None:
            img_norm = img.resize((self.img_size[0], self.img_size[1]), Image.ANTIALIAS)
        else:
            img_norm = img

        img_norm = self.transform(img_norm)
        img_clahe_norm = self.img_HE(img_norm)  # obtain CLAHE image
        img_name = os.path.basename(img_path)
        return img_norm, img_clahe_norm, img_name

    def load_imglist(self):
        img_list = glob.glob(os.path.join(self.data_dir, '*/*.*'))
        if len(img_list) == 0:
            img_list = glob.glob(os.path.join(self.data_dir, '*.*'))
        return img_list

    def img_HE(self, tensor):
        def clahe(im):
            ycrcb = cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB)
            channels = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1, 1))
            clahe.apply(channels[0], channels[0])
            cv2.merge(channels, ycrcb)
            cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB, im)
            return im

        im_pros = tensor.mul(255).byte()
        im_pros = im_pros.detach().cpu().numpy().transpose(1, 2, 0)

        im_pros = clahe(im_pros.copy())
        im_pros = cv2.bilateralFilter(im_pros, 5, 10, 10)
        im_pros = cv2.cvtColor(im_pros, cv2.COLOR_RGB2GRAY)
        im_pros = im_pros.astype(np.uint8)
        im_clahe = transforms.ToTensor()(im_pros)
        return im_clahe
