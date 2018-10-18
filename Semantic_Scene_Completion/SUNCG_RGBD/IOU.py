import torch
import numpy as np

### This version is same as Shurans'.

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

    # tp fp fn
    result = np.zeros((3, num_classes-1), dtype = np.float32)
    for i in range(1, num_classes):
        a = torch.eq(output2, i)
        b = torch.ne(output2, i)
        c = torch.eq(label2, i)
        d = torch.ne(label2, i)
        tp = torch.mul(a, c)
        fp = torch.mul(a, d)
        fn = torch.mul(b, c)
        result[0, i-1] += torch.nonzero(tp).nelement()
        result[1, i-1] += torch.nonzero(fp).nelement()
        result[2, i-1] += torch.nonzero(fn).nelement()

    return result