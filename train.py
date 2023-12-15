import argparse
import torch
import sys
import os

import torchvision
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import DecomNet
from dataloader import LoadDataset
import loss
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


def load_lossfn():
    L_r = loss.LRetinex()
    L_l = loss.LIlluination()
    L_exp = loss.LExp()
    L_sc = loss.LSC()
    L_hue = loss.LHue()
    return L_r, L_l, L_exp, L_sc, L_hue


def train(opt, device, train_loader, val_loader, model, optimizer):
    summary_writer = SummaryWriter(os.path.join(opt.ckpt_folder, 'logs')) if opt.save_graph else ''

    L_r, L_l, L_exp, L_sc, L_hue = load_lossfn()

    for epoch in range(opt.epochs):
        print('\n||||===================== epoch {num} =====================||||'.format(num=epoch + 1))
        epoch_loss = []
        model.train()
        for _, (input, input_h, name) in enumerate(tqdm(train_loader, desc='Training', file=sys.stdout)):
            input = input.to(device)
            input_h = input_h.to(device)
            input_scr, input_h_scr, R, L, R_hue, input_hue = model(input, input_h)

            l_r = L_r(input, R, L)
            l_l = L_l(input, L)
            l_exp = L_exp(input_scr)
            l_sc = L_sc(input_h_scr, R)
            l_hue = L_hue(input_hue, R_hue)
            loss = 15 * l_r + l_l + l_sc + 2 * l_exp + 2 * l_hue
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip_norm)
            optimizer.step()
        print('Train loss: ', sum(epoch_loss) / len(epoch_loss))

        if opt.save_graph:
            summary_writer.add_scalar('train', sum(epoch_loss) / len(epoch_loss), epoch)

        if ((epoch + 1) % opt.save_period) == 0:
            torch.save(model.state_dict(),
                       os.path.join(opt.ckpt_folder, 'epoch_{num}.pth'.format(num=epoch + 1)))

        if ((epoch + 1) % opt.val_period) == 0:
            epoch_loss = []
            with torch.no_grad():
                for _, (input, input_clahe, name) in enumerate(tqdm(val_loader, desc='Validating', file=sys.stdout)):
                    input = input.to(device)
                    input_clahe = input_clahe.to(device)
                    input_scr, input_h_scr, R, L, R_hue, input_hue = model(input, input_clahe)

                    l_r = L_r(input, R, L)
                    l_l = L_l(input, L)
                    l_exp = L_exp(input_scr)
                    l_sc = L_sc(input_h_scr, R)
                    l_hue = L_hue(input_hue, R_hue)
                    loss = 15 * l_r + l_l + l_sc + 2 * l_exp + 2 * l_hue
                    epoch_loss.append(loss.item())

                print('Val loss: '.format(num=epoch + 1), sum(epoch_loss) / len(epoch_loss))

                if opt.save_graph:
                    summary_writer.add_scalar('val_set', sum(epoch_loss) / len(epoch_loss), epoch)
    if opt.save_graph:
        summary_writer.close()


def main(opt):
    # check input dir
    assert os.path.exists(opt.train_data), 'train_data folder {dir} does not exist'.format(dir=opt.train_data)
    assert os.path.exists(opt.val_data), 'val_data folder {dir} does not exist'.format(dir=opt.val_data)
    if not os.path.exists(opt.ckpt_folder):
        os.makedirs(opt.ckpt_folder)
    device_id = 'cuda:' + opt.device
    device = torch.device(device_id if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')

    # load datasets
    train_transforms = transforms.Compose([transforms.RandomCrop((opt.patch_size, opt.patch_size)),
                                           transforms.ToTensor()])
    train_set = LoadDataset(opt.train_data, train_transforms, img_size=(opt.imgsz, opt.imgsz))
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers,
                              batch_size=opt.batch_size, shuffle=True)
    val_transforms = transforms.Compose([transforms.ToTensor()])
    val_set = LoadDataset(opt.val_data, val_transforms, img_size=(opt.imgsz, opt.imgsz))
    val_loader = DataLoader(dataset=val_set, num_workers=opt.num_workers,
                            batch_size=opt.batch_size, shuffle=True)

    print('Num of train set: {num}'.format(num=len(train_set)))
    print('Num of val_set set: {num}'.format(num=len(val_set)))

    model = DecomNet().to(device)
    model.apply(init_weights)

    # init optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                           betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)

    # strat training
    train(opt, device, train_loader, val_loader, model, optimizer)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='datasets/train_set', help='train set path')
    parser.add_argument('--val_data', type=str, default='datasets/val_set', help='val_set set path')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--patch_size', type=int, default=128, help='random crop imgsz during training')
    parser.add_argument('--num_workers', type=int, default=12, help='dataloader workers')
    parser.add_argument('--imgsz', type=int, default=512, help='input image size')
    parser.add_argument('--epochs', type=int, default=200, help='total epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--device', default='0', help='use cuda device; 0, 1, 2 or cpu')
    parser.add_argument('--val_period', type=int, default=20, help='perform validation every x epoch')
    parser.add_argument('--save_period', type=int, default=10, help='save checkpoint every x epoch')
    parser.add_argument('--ckpt_folder', type=str, default='ckpts/exp1', help='location for saving ckpts')
    parser.add_argument('--save_graph', action='store_true', default=True,
                        help='generate graph of training updating process')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
