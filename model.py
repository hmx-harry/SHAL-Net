import cv2
import torch
import torch.nn as nn
import math
import numpy as np
from PIL import Image
from torchvision import transforms


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class DepScp_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepScp_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channel
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        x1 = self.depth_conv(x)
        out = self.point_conv(x1)
        return out


class SCR(nn.Module):
    """
    learnable branch for recovering degraded structure and texture
    features of low-light image itself
    """
    def __init__(self):
        super(SCR, self).__init__()
        filters = 16
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.ds_conv1 = DepScp_Conv(3, filters)
        self.ds_conv2 = DepScp_Conv(filters, filters)
        self.ds_conv3 = DepScp_Conv(filters, filters)
        self.ds_conv4 = DepScp_Conv(filters, 1)

    def forward(self, x, x_h):
        """
        @param x: low-light input
        @param x_h: x after histogram equalization
        @return: recovered x_h
        """
        x1 = self.relu(self.ds_conv1(x))
        x2 = self.relu(self.ds_conv2(x1))
        x3 = self.relu(self.ds_conv3(x2))
        alpha = self.sigmoid(self.ds_conv4(x3)) + 1

        gam_batch = self.cal_gam_val(x.detach())    # adaptive gamma correction
        x_scr = torch.pow(x, gam_batch) + (alpha * x)
        x_h_scr = torch.pow(x_h, gam_batch) + (alpha * x_h)
        return x_scr, x_h_scr

    def cal_gam_val(self, im):
        gam_batch = torch.tensor([])
        for i in range(im.shape[0]):
            im_np = im[i].mul(255).byte()
            im_np = im_np.detach().cpu().numpy().transpose(1, 2, 0)
            mean_gray = np.mean(np.max(im_np, axis=2))

            if mean_gray < 255 / 2:
                if mean_gray == 0:  # set default value, avoid divide by zero
                    gam_batch = torch.cat([gam_batch, torch.tensor([0.25])], dim=0)
                else:  # cal adaptive gamma value
                    gam = math.log10(0.5) / math.log10(mean_gray / 255.0)
                    gam_batch = torch.cat([gam_batch, torch.tensor([min(1, gam)])], dim=0)
            else:  # skip if input image is bright enough
                gam_batch = torch.cat([gam_batch, torch.tensor([1])], dim=0)
        gam_batch = gam_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(im.device)
        return gam_batch


class DecomNet(nn.Module):
    def __init__(self):
        super(DecomNet, self).__init__()
        filters = 64
        self.SCR = SCR()
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = Conv(4, filters)
        self.conv2 = Conv(filters, filters)
        self.conv3 = Conv(filters, filters)
        self.conv4 = Conv(filters, filters)
        self.conv5 = Conv(filters * 2, filters)
        self.conv6 = Conv(filters * 2, filters)
        self.conv7 = Conv(filters * 2, 4)

    def forward(self, x, x_h):
        x_scr, x_h_scr = self.SCR(x, x_h)
        # concat inputs
        x_out = torch.cat([x, x_h_scr], dim=1)
        x1 = self.relu(self.conv1(x_out))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        x7 = self.sigmoid(self.conv7(torch.cat([x1, x6], 1)))
        R, L = torch.split(x7, 3, 1)

        # extract hue information of images from HSV color space
        R_hue = self.rgb2hsv(R)[:, 0, :, :] / 360.0
        R_hue = transforms.GaussianBlur(kernel_size=7, sigma=(3, 3))(R_hue)
        x_hue = self.rgb2hsv(x)[:, 0, :, :] / 360.0
        x_hue = transforms.GaussianBlur(kernel_size=7, sigma=(3, 3))(x_hue)
        return x_scr, x_h_scr, R, L, R_hue, x_hue

    def rgb2hsv(self, input, epsilon=1e-10):
        assert (input.shape[1] == 3)
        r, g, b = input[:, 0], input[:, 1], input[:, 2]
        max_rgb, argmax_rgb = input.max(1)
        min_rgb, argmin_rgb = input.min(1)
        max_min = max_rgb - min_rgb + epsilon

        h1 = 60.0 * (g - r) / max_min + 60.0
        h2 = 60.0 * (b - g) / max_min + 180.0
        h3 = 60.0 * (r - b) / max_min + 300.0

        h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
        s = max_min / (max_rgb + epsilon)
        v = max_rgb
        return torch.stack((h, s, v), dim=1)


class EnhStage():
    def __init__(self):
        super().__init__()

    def enh(self, im, R, L, im_h_scr, scale=0.85):
        mean = torch.mean(im)
        gam = math.log10(scale) / math.log10(mean)
        out = self.guided_filter(R, im_h_scr) * torch.pow(L, gam)
        return out

    def guided_filter(self, im, guide):
        im_np = im.mul(255).squeeze(0).byte()
        im_np = im_np.detach().cpu().numpy().transpose(1, 2, 0)
        guide_np = guide.squeeze(0).mul(255).byte()
        guide_np = guide_np.detach().cpu().numpy().transpose(1, 2, 0)
        im_dns = cv2.ximgproc.guidedFilter(guide_np, im_np, 1, 50, -1)
        im_dns = transforms.ToTensor()(im_dns).to(im.device)
        return im_dns
