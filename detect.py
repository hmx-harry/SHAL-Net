import argparse
import os
import time
from torch.utils.data import DataLoader
from model import DecomNet, EnhStage
import torch
from dataloader import LoadDataset
import torchvision.transforms as transforms
import util
import warnings

warnings.filterwarnings('ignore')


def detect(opt, device, test_loader, model):
    model.eval()
    timelist = []
    enh_module = EnhStage()
    with torch.no_grad():
        for _, (input, input_h, name) in enumerate(test_loader):
            input = input.to(device)
            input_h = input_h.to(device)

            start = time.time()
            _, input_h_scr, R, L, _, _ = model(input, input_h)
            timelist.append(time.time() - start)
            out = enh_module.enh(input, R, L, input_h_scr)
            print('Img: {name}\t\t\tInference time:{time}'.format(name=name[0], time=time.time() - start))

            I_folder = os.path.join(opt.result_folder, 'Enh')  # enhanced iamge
            R_folder = os.path.join(opt.result_folder, 'R')  # reflectance map R
            L_folder = os.path.join(opt.result_folder, 'L')  # illumination map L
            if not os.path.exists(I_folder):
                os.makedirs(I_folder)
                os.makedirs(R_folder)
                os.makedirs(L_folder)
            util.save_img(out, I_folder, name[0])
            util.save_img(R, R_folder, name[0])
            util.save_img(L, L_folder, name[0])


def main(opt):
    # check input dir
    assert os.path.exists(opt.test_data), 'test_data folder {dir} does not exist'.format(dir=opt.test_data)

    # initialize device
    device_id = 'cuda:' + opt.device
    device = torch.device(device_id if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')

    # load datasets
    trans_compose = transforms.Compose([transforms.ToTensor()])
    test_set = LoadDataset(opt.test_data, trans_compose)

    test_loader = DataLoader(dataset=test_set, num_workers=opt.num_workers,
                             batch_size=1, shuffle=False)
    print('Num of test set: {num}'.format(num=len(test_set)))

    # load model weights
    model = DecomNet().to(device)
    model.load_state_dict(torch.load(opt.weights))

    # start inference
    detect(opt, device, test_loader, model)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default='datasets/test_data/low-light/LOL_15', help='train set path')
    parser.add_argument('--weights', type=str, default='weights/SHAL-Net.pth', help='location of model weights')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--device', default='0', help='use cuda device; 0, 1, 2 or cpu')
    parser.add_argument('--result_folder', type=str, default='results/LOL_15', help='location for saving results')
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
