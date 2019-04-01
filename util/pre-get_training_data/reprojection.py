import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function, Variable

class ReprojectionFunction(Function):
    def __init__(self, vox_size):
        super(ReprojectionFunction, self).__init__()
        self.vox_size = vox_size

    def forward(self, input, depth_mapping_3d):

        self.save_for_backward(input, depth_mapping_3d)
        
        batch_size, num_channels, h, w = input.size()
        num_classes = num_channels - 1

        depth_mapping_3dnp = depth_mapping_3d
        inputnp = input.view(batch_size, num_channels, -1)
        if depth_mapping_3dnp.is_cuda == True:
            depth_mapping_3dnp = depth_mapping_3dnp.cpu()
            inputnp = inputnp.cpu()
        depth_mapping_3dnp = depth_mapping_3dnp.numpy()
        inputnp = inputnp.numpy()

        self.depth_mapping_3dnp = depth_mapping_3dnp

        output = np.zeros((batch_size, num_classes, self.vox_size[2]*self.vox_size[1]*self.vox_size[0]), dtype=float)
        indexo, indexi = np.where(depth_mapping_3dnp > 0)
        output[indexo, :, depth_mapping_3dnp[indexo, indexi]] = inputnp[indexo, 1:, indexi]

        return torch.FloatTensor(output).cuda(input.get_device()).view(
                batch_size, num_classes, self.vox_size[2], self.vox_size[1], self.vox_size[0])

    def backward(self, grad_output):

        input, depth_mapping_3d = self.saved_tensors
        batch_size, num_channels, h, w = input.size()

        grad_outputnp = grad_output.view(batch_size, num_channels-1, -1)
        if grad_outputnp.is_cuda == True:
            grad_outputnp = grad_outputnp.cpu()
        grad_outputnp = grad_outputnp.numpy()
        grad_input = np.zeros((batch_size, num_channels, h*w), dtype=float)
        indexo, indexi = np.where(self.depth_mapping_3dnp > 0)
        grad_input[indexo, 1:, indexi] = grad_outputnp[indexo, :, self.depth_mapping_3dnp[indexo, indexi]]

        return torch.FloatTensor(grad_input).cuda(input.get_device()).view(
                batch_size, num_channels, h, w), torch.zeros_like(depth_mapping_3d)

class Reprojection(nn.Module):
    def __init__(self, vox_size = [240, 144, 240]):
        super(Reprojection, self).__init__()
        self.vox_size = vox_size

    def forward(self, input, depth_mapping_3d):
        return ReprojectionFunction(self.vox_size)(input, depth_mapping_3d)

    def __repr__(self):
        return self.__class__.__name__ + ' (vox_size=' + str(self.vox_size) + ')'

def getLabelWeight(label, tsdf_downsample):
    label_weight = np.zeros(label.shape, dtype=float)
    foreindex, = np.where(abs(127 - label) < 127)
    label_weight[foreindex] = 1.0
    foresize = foreindex.size
    bgindex, = np.where(tsdf_downsample < -0.5)
    bgsize = bgindex.size
    if bgsize <= foresize * 2:
        label_weight[bgindex] = 1.0
    else:
        rand = random.sample(range(0,bgsize), foresize*2)
        label_weight[bgindex[rand]] = 1.0

    return label_weight
