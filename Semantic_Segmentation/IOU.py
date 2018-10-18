import torch
import numpy as np

# tensor size is (batchsize, ...)
# channel0 is void and isn't counted.
def computeIOU(output, label, num_classes = 12):
    output1, output2 = torch.max(output, 1)

    batchsize = output2.size(0)
    output2 = output2.view(batchsize, -1)
    label2 = label.view(batchsize, -1)
    
    if output2.size(1) != label2.size(1):
        print "Tensors' sizes don't equal. Tensor1 is (%d,%d), but tensor2 is (%d,%d)."%(
            batchsize, output2.size(1), batchsize, label2.size(1))
        return -1

    result = np.ones((batchsize, num_classes-1), dtype = np.float32)
    for i in range(1, num_classes):
        a = torch.eq(output2, i)
        b = torch.eq(label2, i)
        c = torch.add(a, 1, b)
        d = torch.mul(a, b)
        for j in xrange(batchsize):
            union = torch.nonzero(c[j]).nelement()
            intersection = torch.nonzero(d[j]).nelement()
            if union == 0:
                result[j, i-1] = -1
            else:
                result[j, i-1] = float(intersection) / float(union)

    return result