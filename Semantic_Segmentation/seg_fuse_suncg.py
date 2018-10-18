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

from engine_fuse import Engine


parser = argparse.ArgumentParser(description='seg2D Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
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

class DepthSeg(nn.Module):

    def __init__(self, model, num_classes):
        super(DepthSeg, self).__init__()

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

        return out,


class ColorSeg(nn.Module):

    def __init__(self, model, num_classes):
        super(ColorSeg, self).__init__()

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

        return out,

class Seg2DNet(nn.Module):

    def __init__(self, cs_path, ds_path, model, num_classes):
        super(Seg2DNet, self).__init__()

        self.num_classes = num_classes

        self.cs = ColorSeg(model = models.resnet101(False), num_classes = num_classes)
        chpo = torch.load(cs_path)
        self.cs.load_state_dict(chpo['state_dict'], strict = False)
        print "=> ColorSeg loaded checkpoint '{}'".format(cs_path)
        
        self.ds = DepthSeg(model = models.resnet101(False), num_classes = num_classes)
        chpo = torch.load(ds_path)
        self.ds.load_state_dict(chpo['state_dict'], strict = False)
        print "=> DepthSeg loaded checkpoint '{}'".format(ds_path)

        self.fuse = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 12, 1)
        )


    def forward(self, color, depth):
        color = self.cs(color)
        depth = self.ds(depth)
        x = torch.cat((color[0], depth[0]), dim = 1)
        out = self.fuse(x)

        return out


def PCA_Jittering(img):

    img = np.asanyarray(img, dtype = 'float32')

    img = img / 255.0
    img_size = img.size / 3
    img1 = img.reshape(img_size, 3)
    img1 = np.transpose(img1)
    img_cov = np.cov([img1[0], img1[1], img1[2]])
    lamda, p = np.linalg.eig(img_cov)

    p = np.transpose(p)
    alpha1 = random.normalvariate(0,1)
    alpha2 = random.normalvariate(0,1)
    alpha3 = random.normalvariate(0,1)
    v = np.transpose((alpha1*lamda[0], alpha2*lamda[1], alpha3*lamda[2]))
    add_num = np.dot(p,v)

    img2 = np.array([img[:,:,0]+add_num[0], img[:,:,1]+add_num[1], img[:,:,2]+add_num[2]])
    img2 = img2.reshape(3, img_size)
    img2 = np.transpose(img2)
    img2 = img2.reshape(img.shape)
    img2 = img2 * 255.0
    img2[img2<0] = 0
    img2[img2>255] = 255
    img2 = img2.astype(np.uint8)

    return Image.fromarray(img2)


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
        self.depthlist = []
        ind = 1 # HHA's index begins with 1, instead of 0.
        for line in fid.readlines():
            line = line.rstrip("\n")
            line1 = line[0:len(line)-9] + 'category_uint8.png'
            if os.path.exists(line) and os.path.exists(line1):
                self.depthlist.append(HHA_path+'/%06d.png'%ind)
                self.colorlist.append(line)
                self.categlist.append(line1)
            ind = ind + 1
        fid.close()

        self.num_classes = num_classes
        self.color_transform = Compose([
            ToTensor(),
            Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        self.depth_transform = Compose([
            ToTensor(),
            Normalize([.5282, .3914, .4266], [.1945, .2480, .1506])
        ])
        self.label_transform = label_transform

        self.resize_size = (384, 288) # 12:9

    def __len__(self):

        return len(self.colorlist)

    def __getitem__(self, index):

        depth = Image.open(self.depthlist[index]).convert('RGB')
        depth = depth.resize(self.resize_size, Image.ANTIALIAS)
        depth1 = depth.transpose(Image.FLIP_LEFT_RIGHT)
        depth = self.depth_transform(depth)
        depth1 = self.depth_transform(depth1)
        depth = torch.cat((depth, depth1), 0)

        color = Image.open(self.colorlist[index]).convert('RGB')
        color = color.resize(self.resize_size, Image.ANTIALIAS)
        color1 = color.transpose(Image.FLIP_LEFT_RIGHT)
        color = PCA_Jittering(color)
        color1 = PCA_Jittering(color1)
        color = self.color_transform(color)
        color1 = self.color_transform(color1)
        color = torch.cat((color, color1), 0)

        categ1 = cv2.imread(self.categlist[index], -1).astype(int)
        categ1 = cv2.resize(categ1, dsize=self.resize_size, interpolation=cv2.INTER_NEAREST)
        categ11 = np.fliplr(categ1)
        categ1 = torch.from_numpy(categ1.astype(np.int64))
        categ11 = torch.from_numpy(categ11.astype(np.int64))
        label1 = torch.cat((categ1, categ11), 0)

        return color, depth, label1



# CUDA_VISIBLE_DEVICES=1 python seg_fuse_suncg.py /home/jason/lscCcode/selectImageFromRawSUNCG/selectedImage 2>&1 | tee logs/seg_fuse_suncg.log
def main():
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = TrainDataLoader(os.path.join(args.data, 'image_list_train.txt'), train = True)
    val_dataset = TrainDataLoader(os.path.join(args.data, 'image_list_val.txt'), train = False)

    # load model
    model = Seg2DNet('./save_models/seg_RGB_suncg/checkpoint.pth.tar',
        './save_models/seg_depth_suncg/checkpoint.pth.tar',
        model=models.resnet101(False), num_classes=12)

    # define loss function (criterion)
    cri_weights = torch.FloatTensor([0.0001, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    criterion = nn.CrossEntropyLoss(weight = cri_weights/torch.sum(cri_weights))

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'workers': args.workers, 'start_epoch': args.start_epoch,
             'max_epochs': args.epochs, 'evaluate': args.evaluate, 'resume': args.resume,
             'multi_gpu': False, 'device_ids': [0, 1], 'use_gpu': use_gpu,
             'save_iter': 1200, 'print_freq': args.print_freq, 'epoch_step': [20, 40, 60]}
    state['save_model_path'] = './save_models/seg_fuse_suncg'

    engine = Engine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':

    resource.setrlimit(resource.RLIMIT_STACK, (-1,-1))
    sys.setrecursionlimit(100000)

    main()

