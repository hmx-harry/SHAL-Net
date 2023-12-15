import torch.nn as nn
import torchvision.transforms as transforms
import torch


class LRetinex(nn.Module):
    def __init__(self):
        super(LRetinex, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, im, R, L):
        return self.mse(L * R, im) + self.mse(R, im / (L.detach()))


class LIlluination(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(LIlluination, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.mse = torch.nn.MSELoss()

    def forward(self, im, L):
        return self.cal_l_tv(L) + self.cal_l_init(im, L)

    def cal_l_init(self, im, L):
        im_max, _ = torch.max(im, dim=1, keepdim=True)
        return self.mse(im_max, L)

    def cal_l_tv(self, L):
        batch_size = L.size()[0]
        h_x = L.size()[2]
        w_x = L.size()[3]
        count_h = (L.size()[2] - 1) * L.size()[3]
        count_w = L.size()[2] * (L.size()[3] - 1)
        h_tv = torch.pow((L[:, :, 1:, :] - L[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((L[:, :, :, 1:] - L[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class LExp(nn.Module):
    def __init__(self):
        super(LExp, self).__init__()

    def forward(self, Iscb):
        mean = torch.nn.AvgPool2d(64)
        return torch.mean(torch.pow(mean(Iscb) - torch.FloatTensor([0.6]).to(Iscb.device), 2))


class LSC(nn.Module):
    def __init__(self):
        super(LSC, self).__init__()

    def forward(self, Iscb, R):
        R_gray = transforms.Grayscale()(R)
        return 1 - self.ssim_torch(Iscb, R_gray)

    def ssim_torch(self, im1, im2, L=1):
        K2 = 0.03
        C2 = (K2 * L) ** 2
        C3 = C2 / 2
        ux = torch.mean(im1)
        uy = torch.mean(im2)
        ox_sq = torch.var(im1)
        oy_sq = torch.var(im2)
        ox = torch.sqrt(ox_sq)
        oy = torch.sqrt(oy_sq)
        oxy = torch.mean((im1 - ux) * (im2 - uy))
        oxoy = ox * oy
        C = (2 * ox * oy + C2) / (ox_sq + oy_sq + C2)
        S = (oxy + C3) / (oxoy + C3)
        return S * C


class LHue(nn.Module):
    def __init__(self):
        super(LHue, self).__init__()
        self.mse = torch.nn.SmoothL1Loss()

    def forward(self, im_h, R_h):
        return self.mse(im_h, R_h)
