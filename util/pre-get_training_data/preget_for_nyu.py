# coding=utf-8

import sys, os
import struct
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose, Normalize
from reprojection import getLabelWeight
sys.path.append('./build/lib.linux-x86_64-2.7/')
import DataProcess as dp

# NYU train
# path = os.path.join('/home/jason/NYUv2/NYU_images', 'nyu_train.txt')
# outputpath = '/media/jason/新加卷/nyu_selected'
# train_or_test = 'train'
# hardpath = os.path.join(outputpath, 'hard_examples.npy')

# NYU test
path = os.path.join('/home/jason/NYUv2/NYU_images', 'nyu_test.txt')
outputpath = '/media/jason/新加卷/nyu_selected_val'
train_or_test = 'test'
# hardpath = os.path.join(outputpath, 'hard_examples.npy')

colorlist = []
depthlist = []
binlist = []
# hardexamples = []
fid = open(path, "r")
colorlist = []
depthlist = []
binlist = []
for line in fid.readlines():
    line = line.rstrip("\n")
    line1 = '/home/jason/sscnet-master/data/depthbin/NYU' + train_or_test + '/NYU' + line[len(line)-14:len(line)-9] + '0000.bin'
    line2 = line[0:len(line)-9] + 'depth.png'
    if os.path.isfile(line) and os.path.isfile(line1) and os.path.isfile(line2):
        colorlist.append(line)
        binlist.append(line1)
        depthlist.append(line2)
fid.close()
# fid = open('/home/jason/lscCcode/select_difficult_examples/results/hard_examples.txt', "r")
# for line in fid.readlines():
#     line = line.rstrip("\n")
#     if line in colorlist:
#         hardexamples.append(colorlist.index(line))
# fid.close()

num_classes = 12
img_transform = Compose([Normalize([.485, .456, .406], [.229, .224, .225])])
vox_size = np.array([240,144,240],dtype=np.int64)
vox_unit = 0.02
vox_margin = 0.24
sampleRatio = 4
# cam_K = np.array([518.8579,0,320,
#                0,518.8579,240,
#                0, 0, 1], dtype=np.float32)
cam_K = np.array([518.8579,0,325.58,
               0,519.4696,253.74,
               0, 0, 1], dtype=np.float32)
segmentation_class_map = np.array([0,1,2,3,4,
    11,5,6,7,8,8,10,10,10,11,11,9,8,11,11,11,11,11,
    11,11,11,11,10,10,11,8,10,11,9,11,11,11])

fileindex = 0
# hardlist = []

for index in xrange(len(colorlist)):

    print index

    binfile = open(binlist[index], 'rb')
    vox_origin = np.array(struct.unpack('3f', binfile.read(12)),dtype=np.float32)
    cam_pose = np.array(struct.unpack('16f', binfile.read(64)),dtype=np.float32)
    readin_bin = np.zeros(vox_size[0]*vox_size[1]*vox_size[2], dtype=np.float32)
    all_the_byte = binfile.read()
    binfile.close()
    all_the_val = struct.unpack("%dI"%(len(all_the_byte)/4), all_the_byte)
    last_index = 0
    count_bad_data = 0
    for i in xrange(len(all_the_val)/2):
        if last_index + all_the_val[i * 2 + 1] > vox_size[0] * vox_size[1] * vox_size[2]:
            count_bad_data = 0
            break;
        if all_the_val[i * 2] == 255:
            readin_bin[last_index:last_index+all_the_val[i * 2 + 1]] = 255
        else:
            readin_bin[last_index:last_index+all_the_val[i * 2 + 1]] = segmentation_class_map[all_the_val[i * 2]%37]
            if all_the_val[i * 2] != 0:
                count_bad_data += all_the_val[i * 2 + 1]
        last_index += all_the_val[i * 2 + 1]
    if count_bad_data < 10 or last_index != vox_size[0] * vox_size[1] * vox_size[2]:
        continue

    color = cv2.imread(colorlist[index], -1)
    depth = cv2.imread(depthlist[index], -1).astype(np.float32)

    tsdf = np.zeros(vox_size[2]*vox_size[1]*vox_size[0], dtype=np.float32)
    depth_mapping_3d = np.ones(vox_size[2]*vox_size[1]*vox_size[0], dtype=np.float32) * (-1)
    dp.TSDF((depth/1000.0).reshape(-1), cam_K, cam_pose, vox_origin,
            vox_unit, vox_size.astype(np.float32), vox_margin, depth.shape[0], depth.shape[1],
            tsdf, depth_mapping_3d)
    depth_mapping_3d = depth_mapping_3d.astype(np.int64)
    label = np.zeros(vox_size[2]*vox_size[1]*vox_size[0]
            /(sampleRatio*sampleRatio*sampleRatio), dtype=np.float32)
    tsdf_downsample = np.zeros(vox_size[2]*vox_size[1]*vox_size[0]
            /(sampleRatio*sampleRatio*sampleRatio), dtype=np.float32)
    dp.DownSampleLabel(readin_bin, vox_size.astype(np.float32), sampleRatio, tsdf, label, tsdf_downsample)
    label = label.astype(np.int32)
    label_weight = getLabelWeight(label, tsdf_downsample)
    label[np.where(label==255)] = 0
    dp.TSDFTransform(tsdf, vox_size.astype(np.float32))

    color = np.transpose(color, (2,0,1)).astype(np.float32) / 255.0
    depth1 = torch.FloatTensor(depth / 10000.0).view(1,depth.shape[0],depth.shape[1])
    color = torch.FloatTensor(color)
    color = img_transform(color)
    img = torch.cat((color,depth1), dim=0).contiguous()
    tsdf = torch.FloatTensor(tsdf).view((1,vox_size[2],vox_size[1],vox_size[0]))
    label = torch.LongTensor(label).view(vox_size[2]/sampleRatio,
            vox_size[1]/sampleRatio, vox_size[0]/sampleRatio)
    label_weight = torch.FloatTensor(label_weight).view(vox_size[2]/sampleRatio,
            vox_size[1]/sampleRatio, vox_size[0]/sampleRatio)
    depth_mapping_3d = torch.LongTensor(depth_mapping_3d)

    np.savez_compressed(os.path.join(outputpath,'%06d.npz'%fileindex), img.numpy(), tsdf.numpy(),
            label.numpy(), label_weight.numpy(), depth_mapping_3d.numpy())
    # if index in hardexamples:
    #     hardlist.append(fileindex)
    fileindex += 1

# np.save(hardpath, np.array(hardlist))