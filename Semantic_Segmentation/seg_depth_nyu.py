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
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=5, type=int,
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


from seg_depth_suncg import Seg2DNet


class TrainDataLoader(torch_data.Dataset):
    def __init__(self, path, img_transform = None, label_transform = None, num_classes = 12, train = True):

        super(TrainDataLoader, self).__init__()

        fid = open(path, "r")
        self.colorlist = []
        self.categlist = []
        for line in fid.readlines():
            line = line.rstrip("\n")
            line1 = line[0:len(line)-9] + 'category_suncg.png'
            if os.path.exists(line) and os.path.exists(line1):
                self.colorlist.append(line)
                self.categlist.append(line1)
        fid.close()

        self.num_classes = num_classes
        self.img_transform = img_transform
        self.label_transform = label_transform

        self.train = train

        if train == True:
            self.filelist = np.arange(795)
        else:
            self.filelist = np.arange(654)

        self.resize_size = (384, 288) # 12:9

    def __len__(self):

        return len(self.colorlist)

    def __getitem__(self, index):

        if self.train == True:

            color = Image.open('%s/%06d.png'%('/media/jason/JetsonSSD/nyu_selected_HHA', self.filelist[index]+1)).convert('RGB') # HHA begins with 1, not 0.
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
        else:
            color = Image.open('%s/%06d.png'%('/media/jason/JetsonSSD/nyu_selected_val_HHA', self.filelist[index]+1)).convert('RGB') # HHA begins with 1, not 0.
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



# CUDA_VISIBLE_DEVICES=0 python seg_depth_nyu.py /home/jason/NYUv2/NYU_images 2>&1 | tee logs/seg_depth_nyu.log
def main():
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # define dataset
    input_transform = Compose([
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    train_dataset = TrainDataLoader(os.path.join(args.data, 'nyu_train.txt'), img_transform = input_transform)
    val_dataset = TrainDataLoader(os.path.join(args.data, 'nyu_test.txt'), img_transform = input_transform, train = False)

    # load model
    model = Seg2DNet(model=models.resnet101(True), num_classes=12)
    chpo = torch.load('./save_models/seg_depth_suncg/checkpoint.pth.tar')
    model.load_state_dict(chpo['state_dict'])
    print "=> loaded checkpoint '{}'".format('./save_models/seg_depth_suncg/checkpoint.pth.tar')

    # define loss function (criterion)
    cri_weights = torch.FloatTensor([0.0001, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5])
    criterion = nn.CrossEntropyLoss(weight = cri_weights/torch.sum(cri_weights))

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'workers': args.workers, 'start_epoch': args.start_epoch,
             'max_epochs': args.epochs, 'evaluate': args.evaluate, 'resume': args.resume,
             'multi_gpu': False, 'device_ids': [0, 1], 'use_gpu': use_gpu,
             'save_iter': 0, 'print_freq': args.print_freq, 'epoch_step': [20,40,60], 'image_visdom_iters': 0}
    state['save_model_path'] = './save_models/seg_depth_nyu'

    engine = Engine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':

    resource.setrlimit(resource.RLIMIT_STACK, (-1,-1))
    sys.setrecursionlimit(100000)

    main()

