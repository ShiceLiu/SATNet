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
from torch.nn import Parameter
import torch.optim as optim
import torch.utils.data as torch_data

from engine_depth import Engine
sys.path.append('../../')
import configs


parser = argparse.ArgumentParser(description='SUNCGRGBD Depth Training')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=3, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
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


class ASPP3D(nn.Module):
    def __init__(self, inplanes, planes, conv_list):
        super(ASPP3D, self).__init__()
        self.conv_list = conv_list
        self.conv1 = nn.ModuleList([nn.Conv3d(inplanes, planes, kernel_size=3, padding=dil, dilation=dil, bias = False) for dil in conv_list])
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in conv_list])
        self.conv2 = nn.ModuleList([nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias = False) for dil in conv_list])
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in conv_list])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x)))))
        x = self.relu(y+x) # modified

        return x


class ImageGen3DNet(nn.Module):
    def __init__(self, seg2d_path, img_size):
        super(ImageGen3DNet, self).__init__()

        self.seg2d = Seg2DNet(model = models.resnet101(False), num_classes = 12)
        
        chpo = torch.load(seg2d_path)
        self.seg2d.load_state_dict(chpo['state_dict'], strict = False)
        print "=> seg2d loaded checkpoint '{}'".format(seg2d_path)

        self.seq1 = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding = 1, bias = False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True),
            nn.Conv3d(64, 64, 3, padding = 1, bias = False),
            nn.BatchNorm3d(64)
        )
        self.seq2 = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding = 1, bias = False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True),
            nn.Conv3d(64, 64, 3, padding = 1, bias = False),
            nn.BatchNorm3d(64)
        )
        self.relu = nn.ReLU(inplace = True)
        self.ASPP3D1 = ASPP3D(64, 64, [1, 2, 3])
        self.ASPP3D2 = ASPP3D(64, 64, [1, 2, 3])
        self.ASPP3Dout = nn.Sequential(
            nn.Conv3d(256, 128, 1, bias = False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace = True),
            nn.Conv3d(128, 128, 1, bias = False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace = True),
            nn.Conv3d(128, 12, 1),
            nn.Conv3d(12, 12, 3, padding = 1)
        )

        self.img_required_size = (640, 480)
        self.img_size = img_size
        if cmp(self.img_required_size, self.img_size) != 0:
            x = np.array(range(self.img_required_size[0]), dtype = np.float32)
            y = np.array(range(self.img_required_size[1]), dtype = np.float32)
            scale = 1.0 * self.img_size[0] / self.img_required_size[0]
            x = x * scale + 0.5
            y = y * scale + 0.5
            x = x.astype(np.int64)
            y = y.astype(np.int64)
            if x[self.img_required_size[0]-1] >= self.img_size[0]:
                x[self.img_required_size[0]-1] = self.img_size[0] - 1
            if y[self.img_required_size[1]-1] >= self.img_size[1]:
                y[self.img_required_size[1]-1] = self.img_size[1] - 1
            xx = np.ones((self.img_required_size[1], self.img_required_size[0]), dtype = np.int64)
            yy = np.ones((self.img_required_size[1], self.img_required_size[0]), dtype = np.int64)
            xx[:] = x
            yy[:] = y.reshape((self.img_required_size[1], 1)) * self.img_size[0]
            image_mapping1 = (xx + yy).reshape(-1)
        else:
            image_mapping1 = np.array(range(self.img_required_size[0]*self.img_required_size[1]), dtype = np.int64)
        self.register_buffer('image_mapping', torch.autograd.Variable(torch.LongTensor(image_mapping1), requires_grad=False))

        self.dim_inc_dim = 64

    def forward(self, img, depth_mapping_3d):
        
        bs, ch, hi, wi = img.size()

        tempseg = self.seg2d(img)[0]
        
        segres = tempseg.contiguous().view(bs*self.dim_inc_dim, hi*wi)
        segres = torch.index_select(segres, 1, self.image_mapping).contiguous().view(
            bs, self.dim_inc_dim, self.img_required_size[0]*self.img_required_size[1]).permute(0, 2, 1)
        zerosVec = Variable(torch.zeros(bs, 1, self.dim_inc_dim), requires_grad = False).cuda(img.get_device())
        segVec = torch.cat((segres, zerosVec), 1)
        # confirm: 'depth_mapping_3d' is 640*480 when the value is less than 0
        segres = [torch.index_select(segVec[i], 0, depth_mapping_3d[i]) for i in xrange(bs)]
        segres = torch.stack(segres).permute(0, 2, 1).contiguous().view(bs, self.dim_inc_dim, 60, 36, 60)

        x1 = self.relu(self.seq1(segres) + segres) # different from before
        x2 = self.relu(self.seq2(x1) + x1)
        x3 = self.ASPP3D1(x2)
        x4 = self.ASPP3D2(x3)
        x = torch.cat((x1,x2,x3,x4), dim = 1)
        x = self.ASPP3Dout(x)

        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.seg2d.parameters(), 'lr': lr * lrp},
                {'params': self.seq1.parameters()},
                {'params': self.seq2.parameters()},
                {'params': self.ASPP3D1.parameters()},
                {'params': self.ASPP3D2.parameters()},
                {'params': self.ASPP3Dout.parameters()}
                ]


# use my_selected dataset to train, not shurans
class TrainDataLoader(torch_data.Dataset):
    def __init__(self, path, npz_path, train_or_test, img_transform = None, label_transform = None, num_classes = 12):

        super(TrainDataLoader, self).__init__()

        if train_or_test == "train":
            HHA_path = SUNCGRGBD_HHA_PATH_TRAIN
        else:
            HHA_path = SUNCGRGBD_HHA_PATH_TEST

        self.path = path
        self.npz_path = npz_path

        fid = open(path, "r")
        self.colorlist = []
        self.depthlist = []
        ind = 1
        for line in fid.readlines():
            line = line.rstrip("\n")
            if os.path.exists(line):
                self.colorlist.append(line)
                self.depthlist.append(HHA_path+'/%06d.png'%ind)
            ind = ind + 1
        fid.close()

        self.filelist = range(len(self.colorlist))

        dontexist = []
        for i in xrange(len(self.colorlist)):
            if not os.path.isfile(os.path.join(self.npz_path, '%06d.npz'%self.filelist[i])):
                dontexist.append(i)
        for i in dontexist[::-1]:
            del self.filelist[i]
            del self.colorlist[i]
            del self.depthlist[i]

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

        self.train_or_test = train_or_test

    def __len__(self):

        return len(self.filelist)

    def __getitem__(self, index):

        depth = Image.open(self.depthlist[index]).convert('RGB')
        depth = depth.resize(self.resize_size, Image.ANTIALIAS)
        depth = self.depth_transform(depth)

        loaddata = np.load(os.path.join(self.npz_path,'%06d.npz'%self.filelist[index]))

        # shurans
        label = torch.LongTensor(loaddata['arr_1'].astype(np.int64))
        label_weight = torch.FloatTensor(loaddata['arr_2'].astype(np.float32))
        mapping = loaddata['arr_3'].astype(np.int64)
        mapping1 = np.ones((8294400), dtype = np.int64)
        mapping1[:] = -1
        ind, = np.where(mapping>=0)
        mapping1[mapping[ind]] = ind
        mapping2 = torch.autograd.Variable(torch.FloatTensor(mapping1.reshape((1,240,144,240)).astype(np.float32)))
        mapping2 = torch.nn.MaxPool3d(4,4)(mapping2).data.view(-1).numpy()
        mapping2[mapping2<0] = 307200
        depth_mapping_3d = torch.LongTensor(mapping2.astype(np.int64))

        return depth, label, label_weight, depth_mapping_3d



# CUDA_VISIBLE_DEVICES=0 python SATNet_Depth.py 2>&1 | tee logs/SATNet_Depth.log
def main():
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = TrainDataLoader(SUNCGRGBD_SAMPLE_TXT_TRAIN, SUNCGRGBD_NPZ_PATH_TRAIN, "train")
    val_dataset = TrainDataLoader(SUNCGRGBD_SAMPLE_TXT_TEST, SUNCGRGBD_NPZ_PATH_TEST, "test")

    # load model
    model = ImageGen3DNet('../../Semantic_Segmentation/save_models/seg_depth_suncg/checkpoint.pth.tar', (384, 288))

    # define loss function (criterion)
    cri_weights = torch.FloatTensor([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    criterion = nn.CrossEntropyLoss(weight = cri_weights/torch.sum(cri_weights))

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, 0.1), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'workers': args.workers, 'start_epoch': args.start_epoch,
             'max_epochs': args.epochs, 'evaluate': args.evaluate, 'resume': args.resume,
             'multi_gpu': False, 'device_ids': [0, 1], 'use_gpu': use_gpu,
             'save_iter': 1200, 'print_freq': args.print_freq, 'epoch_step': []}
    state['save_model_path'] = './save_models/SATNet_Depth'

    engine = Engine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':

    resource.setrlimit(resource.RLIMIT_STACK, (-1,-1))
    sys.setrecursionlimit(100000)

    main()

