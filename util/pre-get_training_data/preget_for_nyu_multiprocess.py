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
from multiprocessing import Pool

# data_dir = '/home/jason/sscnet-master/data/depthbin/NYUtrain'
# outputpath = '/media/jason/JetsonSSD/nyu_selected'

# data_dir = '/home/jason/sscnet-master/data/depthbin/NYUtest'
# outputpath = '/media/jason/JetsonSSD/nyu_selected_val'

# data_dir = '/home/jason/sscnet-master/data/depthbin/NYUCADtrain'
# outputpath = '/media/jason/JetsonSSD/nyucad_selected'

# data_dir = '/home/jason/sscnet-master/data/depthbin/NYUCADtest'
# outputpath = '/media/jason/JetsonSSD/nyucad_selected_val'

## for nyu camera intrinsic
# data_dir = '/home/jason/sscnet-master/data/depthbin/NYUtrain'
# outputpath = '/media/jason/JetsonSSD/nyu_selected1'

## for nyu camera intrinsic
data_dir = '/home/jason/sscnet-master/data/depthbin/NYUtest'
outputpath = '/media/jason/JetsonSSD/nyu_selected1_val'

depthlist = []
binlist = []

list_file = os.listdir(data_dir)
for file in list_file:
    if '.bin' in file:
        binlist.append(data_dir + '/' + file)
    elif '.png' in file:
        depthlist.append(data_dir + '/' + file)

depthlist.sort()
binlist.sort()

num_classes = 12
vox_size = np.array([240,144,240],dtype=np.int64)
vox_unit = 0.02
vox_margin = 0.24
sampleRatio = 4
# cam_K = np.array([518.8579,0,320,
#                   0,518.8579,240,
#                   0, 0, 1], dtype=np.float32)
cam_K = np.array([518.8579,0,325.58,
                  0,519.4696,253.74,
                  0, 0, 1], dtype=np.float32)
segmentation_class_map = np.array([0,1,2,3,4,
    11,5,6,7,8,8,10,10,10,11,11,9,8,11,11,11,11,11,
    11,11,11,11,10,10,11,8,10,11,9,11,11,11])

def multi_process(index, lenlist, startindex):

    print str(index) + '/' + str(lenlist)

    binfile = open(binlist[index-startindex], 'rb')
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
            readin_bin[last_index:last_index+all_the_val[i * 2 + 1]] = segmentation_class_map[all_the_val[i * 2]]
            if all_the_val[i * 2] != 0:
                count_bad_data += all_the_val[i * 2 + 1]
        last_index += all_the_val[i * 2 + 1]
    if count_bad_data < 10 or last_index != vox_size[0] * vox_size[1] * vox_size[2]:
        print str(index) + ' is awful!'
        return

    depth = cv2.imread(depthlist[index-startindex], -1)
    depth = ((depth << 13) | (depth >> 3)).astype(np.float32)

    tsdf = np.ones(vox_size[2]*vox_size[1]*vox_size[0], dtype=np.float32) # modified: zeros to ones
    depth_mapping_3d = np.ones(640*480, dtype=np.float32) * (-1) # vox_size[2]*vox_size[1]*vox_size[0] seems to be wrong, it should be 640x480
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

    tsdf = torch.FloatTensor(tsdf).view((1,vox_size[2],vox_size[1],vox_size[0]))
    label = torch.LongTensor(label).view(vox_size[2]/sampleRatio,
            vox_size[1]/sampleRatio, vox_size[0]/sampleRatio)
    label_weight = torch.FloatTensor(label_weight).view(vox_size[2]/sampleRatio,
            vox_size[1]/sampleRatio, vox_size[0]/sampleRatio)
    depth_mapping_3d = torch.LongTensor(depth_mapping_3d)

    np.savez_compressed(os.path.join(outputpath,'%06d.npz'%index), tsdf.numpy(),
            label.numpy(), label_weight.numpy(), depth_mapping_3d.numpy())

pool = Pool(processes = 8)

startindex = 0
lenlist = len(depthlist)
for index in range(startindex, startindex + lenlist):
    result = pool.apply_async(multi_process, (index, startindex + lenlist, startindex))
pool.close()
pool.join()
if result.successful():
    print 'successful'
