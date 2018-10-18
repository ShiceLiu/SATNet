import argparse
import sys, os, gc, resource
import numpy as np
from PIL import Image
import cv2
import random

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as torch_data

from engine import Engine


parser = argparse.ArgumentParser(description='seg2D Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias = False)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class ASPP(nn.Module):
    def __init__(self, inplanes, planes, conv_list):
        super(ASPP, self).__init__()
        self.conv_list = conv_list
        self.conv = nn.ModuleList([nn.Conv2d(inplanes, planes, kernel_size=3, padding=dil, dilation=dil, bias = False) for dil in conv_list])
        self.bn = nn.ModuleList([nn.BatchNorm2d(planes) for dil in conv_list])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.bn[0](self.conv[0](x))
        for i in range(1, len(self.conv_list)):
            y += self.bn[i](self.conv[i](x))
        x = self.relu(y)

        return x

class Seg2DNet(nn.Module):

    def __init__(self, model, num_classes):
        super(Seg2DNet, self).__init__()

        self.num_classes = num_classes

        self.conv1 = model.conv1
        self.bn0 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.duc1 = DUC(2048, 2048*2)
        self.duc2 = DUC(1024, 1024*2)
        self.duc3 = DUC(512, 512*2)
        self.duc4 = DUC(128, 128*2)
        self.duc5 = DUC(64, 64*2)

        self.ASPP = ASPP(32, 64, [1, 3, 5, 7])
        self.ASPPout = nn.Conv2d(64, 12, 1)

        self.transformer = nn.Conv2d(320, 128, kernel_size=1)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes/2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes/2, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes/2, self.num_classes, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        dfm1 = fm3 + self.duc1(fm4)

        dfm2 = fm2 + self.duc2(dfm1)

        dfm3 = fm1 + self.duc3(dfm2)

        dfm3_t = self.transformer(torch.cat((dfm3, pool_x), 1))

        dfm4 = conv_x + self.duc4(dfm3_t)

        dfm5 = self.duc5(dfm4)

        out = self.ASPP(dfm5)

        out = self.ASPPout(out)

        return out,


class TrainDataLoader(torch_data.Dataset):
    def __init__(self, path, train, img_transform = None, label_transform = None, num_classes = 12):

        super(TrainDataLoader, self).__init__()

        if train == True:
            HHA_path = '/media/jason/JetsonSSD/myselect_suncg_HHA'
        else:
            HHA_path = '/media/jason/JetsonSSD/myselect_suncg_val_HHA'

        fid = open(path, "r")
        self.colorlist = []
        self.categlist = []
        ind = 1 # HHA's index begins with 1, instead of 0.
        for line in fid.readlines():
            line = line.rstrip("\n")
            line1 = line[0:len(line)-9] + 'category_uint8.png'
            if os.path.exists(line) and os.path.exists(line1):
                self.colorlist.append(HHA_path+'/%06d.png'%ind)
                self.categlist.append(line1)
            ind = ind + 1
        fid.close()

        self.num_classes = num_classes
        self.img_transform = img_transform
        self.label_transform = label_transform

        self.resize_size = (384, 288) # 12:9

    def __len__(self):

        return len(self.colorlist)

    def __getitem__(self, index):

        color = Image.open(self.colorlist[index]).convert('RGB')
        color = color.resize(self.resize_size, Image.ANTIALIAS)
        color1 = color.transpose(Image.FLIP_LEFT_RIGHT)
        color = self.img_transform(color)
        color1 = self.img_transform(color1)
        img = torch.cat((color, color1), 0)

        categ1 = cv2.imread(self.categlist[index], -1).astype(int)
        categ1 = cv2.resize(categ1, dsize=self.resize_size, interpolation=cv2.INTER_NEAREST)
        categ11 = np.fliplr(categ1)
        categ2 = cv2.resize(categ1, dsize=(self.resize_size[0]/2,self.resize_size[1]/2), interpolation=cv2.INTER_NEAREST)
        categ21 = np.fliplr(categ2)
        categ4 = cv2.resize(categ1, dsize=(self.resize_size[0]/4,self.resize_size[1]/4), interpolation=cv2.INTER_NEAREST)
        categ41 = np.fliplr(categ4)
        categ8 = cv2.resize(categ1, dsize=(self.resize_size[0]/8,self.resize_size[1]/8), interpolation=cv2.INTER_NEAREST)
        categ81 = np.fliplr(categ8)
        categ16 = cv2.resize(categ1, dsize=(self.resize_size[0]/16,self.resize_size[1]/16), interpolation=cv2.INTER_NEAREST)
        categ161 = np.fliplr(categ16)
        categ1 = torch.from_numpy(categ1.astype(np.int64))
        categ11 = torch.from_numpy(categ11.astype(np.int64))
        categ2 = torch.from_numpy(categ2.astype(np.int64))
        categ21 = torch.from_numpy(categ21.astype(np.int64))
        categ4 = torch.from_numpy(categ4.astype(np.int64))
        categ41 = torch.from_numpy(categ41.astype(np.int64))
        categ8 = torch.from_numpy(categ8.astype(np.int64))
        categ81 = torch.from_numpy(categ81.astype(np.int64))
        categ16 = torch.from_numpy(categ16.astype(np.int64))
        categ161 = torch.from_numpy(categ161.astype(np.int64))
        label1 = torch.cat((categ1, categ11), 0)
        label2 = torch.cat((categ2, categ21), 0)
        label4 = torch.cat((categ4, categ41), 0)
        label8 = torch.cat((categ8, categ81), 0)
        label16 = torch.cat((categ16, categ161), 0)

        return img, label1, label2, label4, label8, label16



# CUDA_VISIBLE_DEVICES=0 python seg_depth_suncg.py /home/jason/lscCcode/selectImageFromRawSUNCG/selectedImage 2>&1 | tee ./logs/seg_depth_suncg.log
def main():
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # define dataset
    input_transform = Compose([
        ToTensor(),
        Normalize([.5282, .3914, .4266], [.1945, .2480, .1506])
    ])
    train_dataset = TrainDataLoader(os.path.join(args.data, 'image_list_train.txt'), train = True, img_transform = input_transform)
    val_dataset = TrainDataLoader(os.path.join(args.data, 'image_list_val.txt'), train = False, img_transform = input_transform)

    # load model
    model = Seg2DNet(model=models.resnet101(True), num_classes=12)

    # define loss function (criterion)
    cri_weights = torch.FloatTensor([0.0001, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    criterion = nn.CrossEntropyLoss(weight = cri_weights/torch.sum(cri_weights))

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'workers': args.workers, 'start_epoch': args.start_epoch,
             'max_epochs': args.epochs, 'evaluate': args.evaluate, 'resume': args.resume,
             'multi_gpu': False, 'device_ids': [0, 1], 'use_gpu': use_gpu,
             'save_iter': 1200, 'print_freq': args.print_freq, 'epoch_step': [20, 40, 60]}
    state['save_model_path'] = './save_models/seg_depth_suncg'

    engine = Engine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':

    resource.setrlimit(resource.RLIMIT_STACK, (-1,-1))
    sys.setrecursionlimit(100000)

    main()

