import argparse
import sys, os, gc, resource
import struct
import random
import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Compose, Normalize
from torch.autograd import Variable
from torch.nn import Parameter
import torch.optim as optim
import torch.utils.data as torch_data

from engine import Engine
from reprojection import Reprojection, getLabelWeight
sys.path.append('./build/lib.linux-x86_64-2.7/')
import DataProcess as dp


parser = argparse.ArgumentParser(description='MergeFineTune Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=2e-5, type=float,
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
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class Seg2DNet(nn.Module):

    def __init__(self, model, num_classes):
        super(Seg2DNet, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(4, 64, 7, stride = 2, padding = 3, bias = False)
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

        self.out5 = self._classifier(32)

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
        out = self.out5(dfm5)

        return out

class TSDFpreprocess(nn.Module):
    
    def __init__(self):
        super(TSDFpreprocess, self).__init__()

        self.gen_conv1 = nn.Conv3d(1, 8, 7, stride = 2, padding = 3)
        self.gen_relu1 = nn.ReLU(inplace = True)
        self.gen_conv2_1 = nn.Conv3d(8, 16, 3, padding = 1)
        self.gen_relu2_1 = nn.ReLU(inplace = True)
        self.gen_conv2_2 = nn.Conv3d(16, 16, 3, padding = 1)
        self.gen_conv2 = nn.Conv3d(8, 16, 1)
        self.gen_relu2 = nn.ReLU(inplace = True)
        self.gen_pool2 = nn.MaxPool3d(2, stride = 2)
        self.gen_reduction = nn.Conv3d(16, 16, 1)

    def forward(self, x):
        x = self.gen_relu1(self.gen_conv1(x))
        x = self.gen_pool2(self.gen_relu2(self.gen_conv2_2(self.gen_relu2_1(self.gen_conv2_1(x))) + self.gen_conv2(x)))
        x = self.gen_reduction(x)

        return x

class Gen3DNet(nn.Module):

    def __init__(self):
        super(Gen3DNet, self).__init__()

        self.gen_conv3_pre = nn.Conv3d(17, 16, 3, padding = 1)
        self.gen_relu3_pre = nn.ReLU(inplace = True)

        self.gen_conv3_1 = nn.Conv3d(16, 32, 3, padding = 1) # 8->16
        self.gen_relu3_1 = nn.ReLU(inplace = True)
        self.gen_conv3_2 = nn.Conv3d(32, 32, 3, padding = 1)
        self.gen_conv3 = nn.Conv3d(16, 32, 1)
        self.gen_relu3 = nn.ReLU(inplace = True)

        self.gen_conv4_1 = nn.Conv3d(32, 32, 3, padding = 1)
        self.gen_relu4_1 = nn.ReLU(inplace = True)
        self.gen_conv4_2 = nn.Conv3d(32, 32, 3, padding = 1)
        self.gen_relu4 = nn.ReLU(inplace = True)

        self.gen_conv5_1 = nn.Conv3d(32, 32, 3, padding = 2, dilation = 2)
        self.gen_relu5_1 = nn.ReLU(inplace = True)
        self.gen_conv5_2 = nn.Conv3d(32, 32, 3, padding = 2, dilation = 2)
        self.gen_relu5 = nn.ReLU(inplace = True)

        self.gen_conv6_1 = nn.Conv3d(32, 32, 3, padding = 2, dilation = 2)
        self.gen_relu6_1 = nn.ReLU(inplace = True)
        self.gen_conv6_2 = nn.Conv3d(32, 32, 3, padding = 2, dilation = 2)
        self.gen_relu6 = nn.ReLU(inplace = True)

        self.gen_conv7 = nn.Conv3d(128, 32, 1)
        self.gen_relu7 = nn.ReLU(inplace = True)
        self.gen_conv8 = nn.Conv3d(32, 32, 1)
        self.gen_relu8 = nn.ReLU(inplace = True)
        self.gen_conv9 = nn.Conv3d(32, 1, 1)

    def forward(self, x):
        x = self.gen_relu3_pre(self.gen_conv3_pre(x))
        x1 = self.gen_relu3(self.gen_conv3_2(self.gen_relu3_1(self.gen_conv3_1(x))) + self.gen_conv3(x))
        x2 = self.gen_relu4(self.gen_conv4_2(self.gen_relu4_1(self.gen_conv4_1(x1))) + x1)
        x3 = self.gen_relu5(self.gen_conv5_2(self.gen_relu5_1(self.gen_conv5_1(x2))) + x2)
        x4 = self.gen_relu6(self.gen_conv6_2(self.gen_relu6_1(self.gen_conv6_1(x3))) + x3)
        x = torch.cat((x1,x2,x3,x4), 1)
        x = self.gen_conv9(self.gen_relu8(self.gen_conv8(self.gen_relu7(self.gen_conv7(x)))))

        return x

class SceneCompletionRGBD(nn.Module):
    
    def __init__(self, vox_size = np.array([240,144,240])):
        
        super(SceneCompletionRGBD, self).__init__()

        self.vox_size = vox_size

        self.Seg2DNet = Seg2DNet(model = models.resnet50(True), num_classes = 12)
        self.Reprojection = Reprojection(vox_size)
        self.TSDFpreprocess = TSDFpreprocess()
        self.MaxPool3d = nn.MaxPool3d(4, stride = 4)
        self.Gen3DNet0 = Gen3DNet()
        self.Gen3DNet1 = Gen3DNet()
        self.Gen3DNet2 = Gen3DNet()
        self.Gen3DNet3 = Gen3DNet()
        self.Gen3DNet4 = Gen3DNet()
        self.Gen3DNet5 = Gen3DNet()
        self.Gen3DNet6 = Gen3DNet()
        self.Gen3DNet7 = Gen3DNet()
        self.Gen3DNet8 = Gen3DNet()
        self.Gen3DNet9 = Gen3DNet()
        self.Gen3DNet10 = Gen3DNet()
        self.genvoid = nn.Conv3d(11, 1, 1)

        checkpoint = torch.load('../segmentImage/10.3.0.135/save_models/resnet50/model_best_0.8118.pth.tar')
        own_state = self.Seg2DNet.state_dict()
        for name, param in checkpoint['state_dict'].items():
            if name[7:] not in own_state:
                print name[7:] + ' is not in own_state.'
                continue
            if isinstance(param, Parameter):
                param = param.data
            print name[7:] + ' is copying.'
            own_state[name[7:]].copy_(param)

    def forward(self, x, tsdf, depth_mapping_3d):

        imagecat = self.Seg2DNet(x) # channel 12
        tsdfcat = self.Reprojection(imagecat, depth_mapping_3d) # channel 11
        tsdfcat = self.MaxPool3d(tsdfcat)
        tsdf_pre = self.TSDFpreprocess(tsdf)
        
        gpu_index = x.get_device()
        x0 = self.Gen3DNet0(torch.cat((tsdf_pre, torch.index_select(tsdfcat, 1, torch.autograd.Variable(torch.LongTensor([0]).cuda(gpu_index)))), dim = 1))
        x1 = self.Gen3DNet1(torch.cat((tsdf_pre, torch.index_select(tsdfcat, 1, torch.autograd.Variable(torch.LongTensor([1]).cuda(gpu_index)))), dim = 1))
        x2 = self.Gen3DNet2(torch.cat((tsdf_pre, torch.index_select(tsdfcat, 1, torch.autograd.Variable(torch.LongTensor([2]).cuda(gpu_index)))), dim = 1))
        x3 = self.Gen3DNet3(torch.cat((tsdf_pre, torch.index_select(tsdfcat, 1, torch.autograd.Variable(torch.LongTensor([3]).cuda(gpu_index)))), dim = 1))
        x4 = self.Gen3DNet4(torch.cat((tsdf_pre, torch.index_select(tsdfcat, 1, torch.autograd.Variable(torch.LongTensor([4]).cuda(gpu_index)))), dim = 1))
        x5 = self.Gen3DNet5(torch.cat((tsdf_pre, torch.index_select(tsdfcat, 1, torch.autograd.Variable(torch.LongTensor([5]).cuda(gpu_index)))), dim = 1))
        x6 = self.Gen3DNet6(torch.cat((tsdf_pre, torch.index_select(tsdfcat, 1, torch.autograd.Variable(torch.LongTensor([6]).cuda(gpu_index)))), dim = 1))
        x7 = self.Gen3DNet7(torch.cat((tsdf_pre, torch.index_select(tsdfcat, 1, torch.autograd.Variable(torch.LongTensor([7]).cuda(gpu_index)))), dim = 1))
        x8 = self.Gen3DNet8(torch.cat((tsdf_pre, torch.index_select(tsdfcat, 1, torch.autograd.Variable(torch.LongTensor([8]).cuda(gpu_index)))), dim = 1))
        x9 = self.Gen3DNet9(torch.cat((tsdf_pre, torch.index_select(tsdfcat, 1, torch.autograd.Variable(torch.LongTensor([9]).cuda(gpu_index)))), dim = 1))
        x10 = self.Gen3DNet10(torch.cat((tsdf_pre, torch.index_select(tsdfcat, 1, torch.autograd.Variable(torch.LongTensor([10]).cuda(gpu_index)))), dim = 1))
        x = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10), dim = 1)

        x11 = self.genvoid(x) # void

        return torch.cat((x11, x), dim = 1)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.Seg2DNet.parameters(), 'lr': lr * lrp},
                {'params': self.TSDFpreprocess.parameters()},
                {'params': self.Gen3DNet0.parameters()},
                {'params': self.Gen3DNet1.parameters()},
                {'params': self.Gen3DNet2.parameters()},
                {'params': self.Gen3DNet3.parameters()},
                {'params': self.Gen3DNet4.parameters()},
                {'params': self.Gen3DNet5.parameters()},
                {'params': self.Gen3DNet6.parameters()},
                {'params': self.Gen3DNet7.parameters()},
                {'params': self.Gen3DNet8.parameters()},
                {'params': self.Gen3DNet9.parameters()},
                {'params': self.Gen3DNet10.parameters()},
                {'params': self.genvoid.parameters()}]


class TrainDataLoader(torch_data.Dataset):
    def __init__(self, path, train_or_test, img_transform = None, label_transform = None, num_classes = 12):

        super(TrainDataLoader, self).__init__()

        # SUNCG
        fid = open(path, "r")
        self.colorlist = []
        self.depthlist = []
        self.binlist = []
        for line in fid.readlines():
            line = line.rstrip("\n")
            # line = '/home/zhidao/LSC2' + line[11:]
            line1 = line[0:len(line)-23] + 'depthVox/'
            for binname in os.listdir(line1):
                binn = os.path.splitext(binname)
                if binn[1] == '.bin' and binname[2:8] == line[len(line)-16:len(line)-10]:
                    line1 = line1 + binname
                    break
            line2 = line[0:len(line)-9] + 'depth.png'
            if os.path.isfile(line) and os.path.isfile(line1) and os.path.isfile(line2):
                self.colorlist.append(line)
                self.binlist.append(line1)
                self.depthlist.append(line2)
        fid.close()
        self.hardexamples = []
        if train_or_test == 'test':
            self.hardexamples = range(len(self.colorlist))
        else:
            fid = open('/home/jason/lscCcode/select_difficult_examples/results/hard_examples.txt', "r")
            for line in fid.readlines():
                line = line.rstrip("\n")
                if line in self.colorlist:
                    self.hardexamples.append(self.colorlist.index(line))
            fid.close()

        # NYUv2
        # fid = open(path, "r")
        # self.colorlist = []
        # self.depthlist = []
        # self.binlist = []
        # for line in fid.readlines():
        #     line = line.rstrip("\n")
        #     line1 = '/home/jason/sscnet-master/data/depthbin/NYU' + train_or_test + '/NYU' + line[len(line)-14:len(line)-9] + '0000.bin'
        #     line2 = line[0:len(line)-9] + 'depth.png'
        #     if os.path.isfile(line) and os.path.isfile(line1) and os.path.isfile(line2):
        #         self.colorlist.append(line)
        #         self.binlist.append(line1)
        #         self.depthlist.append(line2)
        # fid.close()

        self.num_classes = num_classes
        self.img_transform = img_transform
        self.label_transform = label_transform

        self.vox_size = np.array([240,144,240],dtype=np.int64)
        self.vox_unit = 0.02
        self.vox_margin = 0.24
        self.sampleRatio = 4
        self.cam_K = np.array([518.8579,0,320,
                               0,518.8579,240,
                               0, 0, 1], dtype=np.float32)
        # self.cam_K = np.array([518.8579,0,325.58,
        #                        0,519.4696,253.74,
        #                        0, 0, 1], dtype=np.float32)
        self.segmentation_class_map = np.array([0,1,2,3,4,
            11,5,6,7,8,8,10,10,10,11,11,9,8,11,11,11,11,11,
            11,11,11,11,10,10,11,8,10,11,9,11,11,11])

    def __len__(self):

        return len(self.colorlist)

    def __getitem__(self, index):

        while True:
            
            binfile = open(self.binlist[index], 'rb')
            vox_origin = np.array(struct.unpack('3f', binfile.read(12)),dtype=np.float32)
            cam_pose = np.array(struct.unpack('16f', binfile.read(64)),dtype=np.float32)
            readin_bin = np.zeros(self.vox_size[0]*self.vox_size[1]*self.vox_size[2], dtype=np.float32)
            all_the_byte = binfile.read()
            binfile.close()
            all_the_val = struct.unpack("%dI"%(len(all_the_byte)/4), all_the_byte)
            last_index = 0
            count_bad_data = 0
            for i in xrange(len(all_the_val)/2):
                if last_index + all_the_val[i * 2 + 1] > self.vox_size[0] * self.vox_size[1] * self.vox_size[2]:
                    count_bad_data = 0
                    break;
                if all_the_val[i * 2] == 255:
                    readin_bin[last_index:last_index+all_the_val[i * 2 + 1]] = 255
                else:
                    readin_bin[last_index:last_index+all_the_val[i * 2 + 1]] = self.segmentation_class_map[all_the_val[i * 2]%37]
                    if all_the_val[i * 2] != 0:
                        count_bad_data += all_the_val[i * 2 + 1]
                last_index += all_the_val[i * 2 + 1]
            if count_bad_data < 10 or last_index != self.vox_size[0] * self.vox_size[1] * self.vox_size[2]:
                index = self.hardexamples[random.randint(0, len(self.hardexamples)-1)]
                continue

            color = cv2.imread(self.colorlist[index], -1)
            depth = cv2.imread(self.depthlist[index], -1).astype(np.float32)

            tsdf = np.zeros(self.vox_size[2]*self.vox_size[1]*self.vox_size[0], dtype=np.float32)
            depth_mapping_3d = np.ones(self.vox_size[2]*self.vox_size[1]*self.vox_size[0], dtype=np.float32) * (-1)
            dp.TSDF((depth/1000.0).reshape(-1), self.cam_K, cam_pose, vox_origin,
                    self.vox_unit, self.vox_size.astype(np.float32), self.vox_margin, depth.shape[0], depth.shape[1],
                    tsdf, depth_mapping_3d)
            depth_mapping_3d = depth_mapping_3d.astype(np.int64)
            label = np.zeros(self.vox_size[2]*self.vox_size[1]*self.vox_size[0]
                    /(self.sampleRatio*self.sampleRatio*self.sampleRatio), dtype=np.float32)
            tsdf_downsample = np.zeros(self.vox_size[2]*self.vox_size[1]*self.vox_size[0]
                    /(self.sampleRatio*self.sampleRatio*self.sampleRatio), dtype=np.float32)
            dp.DownSampleLabel(readin_bin, self.vox_size.astype(np.float32), self.sampleRatio, tsdf, label, tsdf_downsample)
            label = label.astype(np.int32)
            label_weight = getLabelWeight(label, tsdf_downsample)
            label[np.where(label==255)] = 0
            dp.TSDFTransform(tsdf, self.vox_size.astype(np.float32))

            color = np.transpose(color, (2,0,1)).astype(np.float32) / 255.0
            depth1 = torch.FloatTensor(depth / 10000.0).view(1,depth.shape[0],depth.shape[1])
            color = torch.FloatTensor(color)
            color = self.img_transform(color)
            img = torch.cat((color,depth1), dim=0).contiguous()
            tsdf = torch.FloatTensor(tsdf).view((1,self.vox_size[2],self.vox_size[1],self.vox_size[0]))
            label = torch.LongTensor(label).view(self.vox_size[2]/self.sampleRatio,
                    self.vox_size[1]/self.sampleRatio, self.vox_size[0]/self.sampleRatio)
            label_weight = torch.FloatTensor(label_weight).view(self.vox_size[2]/self.sampleRatio,
                    self.vox_size[1]/self.sampleRatio, self.vox_size[0]/self.sampleRatio)
            depth_mapping_3d = torch.LongTensor(depth_mapping_3d)

            return img, tsdf, label, label_weight, depth_mapping_3d


# CUDA_VISIBLE_DEVICES=1 python nyu.py /home/jason/lscCcode/selectImageFromRawSUNCG/selectedImage
# CUDA_VISIBLE_DEVICES=1 python nyu.py /home/jason/NYUv2/NYU_images
def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # define dataset
    input_transform = Compose([Normalize([.485, .456, .406], [.229, .224, .225])])
    train_dataset = TrainDataLoader(os.path.join(args.data, 'image_list_train.txt'), 'train', img_transform = input_transform)
    val_dataset = TrainDataLoader(os.path.join(args.data, 'image_list_val.txt'), 'test', img_transform = input_transform)
    # train_dataset = TrainDataLoader(os.path.join(args.data, 'nyu_train.txt'), 'train', img_transform = input_transform)
    # val_dataset = TrainDataLoader(os.path.join(args.data, 'nyu_test.txt'), 'test', img_transform = input_transform)

    # load model
    model = SceneCompletionRGBD()
    # chpo = torch.load('./save_models/preweights.pth.tar')
    # model.load_state_dict(chpo['state_dict'])
    # print "=> loaded checkpoint '{}'".format('./save_models/preweights.pth.tar')

    # define loss function (criterion)
    # criterion = nn.CrossEntropyLoss()
    cri_weights = torch.FloatTensor([1, 10, 10, 10, 20, 10, 10, 10, 10, 20, 10, 10])
    criterion = nn.CrossEntropyLoss(weight = cri_weights/torch.sum(cri_weights))

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, 1),
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'workers': args.workers, 'max_epochs': args.epochs,
                'evaluate': args.evaluate, 'resume': args.resume}
    state['save_model_path'] = './save_models/suncg_new_loss1'

    engine = Engine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':

    resource.setrlimit(resource.RLIMIT_STACK, (-1,-1))
    sys.setrecursionlimit(100000)

    main()
