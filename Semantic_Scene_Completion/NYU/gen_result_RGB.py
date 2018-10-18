from SATNet_RGB import ImageGen3DNet, TrainDataLoader # HERE
import sys, resource, os
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
import torch.backends.cudnn as cudnn
import h5py
sys.path.append('../../')
import configs


# CUDA_VISIBLE_DEVICES=0 python gen_result_RGB.py
def main():
    cudnn.benchmark = True
    resume_path = './save_models/SATNet_RGB/checkpoint.pth.tar' # HERE
    output_path = './results/nyu_pred.hdf5'

    input_transform = Compose([
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225])
    ]) # HERE
    dataset = TrainDataLoader(NYU_SAMPLE_TXT_TEST, NYU_NPZ_PATH_TEST, 'test', img_transform = input_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = 4,
                                              shuffle = False, num_workers = 1)
    data_loader.pin_memory = True

    # model = SceneCompletionRGBD()
    model = ImageGen3DNet((384, 288)) # HERE
    if not os.path.isfile(resume_path):
        print "=> no checkpoint found at '{}'".format(resume_path)
        exit()
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = torch.nn.DataParallel(model).cuda(0)

    softmax_layer = torch.nn.Softmax(dim = 1).cuda(0)

    predictions = []
    with torch.no_grad():
        for i, (color, label, label_weight, depth_mapping_3d) in enumerate(data_loader):
            print '{0}/{1}'.format(i, len(data_loader))

            input_var = torch.autograd.Variable(color.cuda(async=True))
            depth_mapping_3d_var = torch.autograd.Variable(depth_mapping_3d.cuda(async=True))

            output = model(input_var, depth_mapping_3d_var)
            output = softmax_layer(output) # HERE
            predictions.append(output.cpu().data.numpy())
        predictions = np.vstack(predictions)

    fp = h5py.File(output_path, 'w')
    result = fp.create_dataset('result', predictions.shape, dtype='f')
    result[...] = predictions
    fp.close()

if __name__ == '__main__':

    resource.setrlimit(resource.RLIMIT_STACK, (-1,-1))
    sys.setrecursionlimit(100000)

    main()