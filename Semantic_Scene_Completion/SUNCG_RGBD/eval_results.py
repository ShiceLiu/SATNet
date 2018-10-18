import numpy as np
import os, sys
import h5py
import cv2

# inputs have been filtered by nonfree.
def eval_prediction(label, pred):
    res = np.zeros((11, 3), dtype = np.float32)
    for i in xrange(1, 12):
        labeli = (label == i).astype(np.int32)
        label_i = (label != i).astype(np.int32)
        predi = (pred == i).astype(np.int32)
        pred_i = (pred != i).astype(np.int32)
        tp = np.sum(labeli*predi)
        fp = np.sum(label_i*predi)
        fn = np.sum(labeli*pred_i)
        res[i-1, :] = np.array([tp, fp, fn])

    return res

def eval_completion(label, pred):
    labeli = (label > 0).astype(np.int32)
    label_i = (label == 0).astype(np.int32)
    predi = (pred > 0).astype(np.int32)
    pred_i = (pred == 0).astype(np.int32)
    intersection = np.sum(labeli*predi)
    union = np.sum((labeli+predi!=0).astype(np.int32))

    res = np.zeros((3), dtype = np.float32)
    res[0] = float(intersection)/float(union) # iou
    tp = intersection
    fp = np.sum(label_i*predi)
    fn = np.sum(labeli*pred_i)
    if tp + fp > 0:
        res[1] = float(tp)/float(tp+fp) # prec
    else:
        res[1] = 0
    if tp + fn > 0:
        res[2] = float(tp)/float(tp+fn) # recall
    else:
        res[2] = 0

    return res


# python eval_results.py >> results/SeeNetFuse.txt
filename = './results/result_suncg.hdf5' # HERE

classname = ['ceiling','floor','wall','window','chair','bed','sofa','table','tvs','furn','objs']

labelname_list = range(500)
labelname_list.remove(9)
label_list = []
for i in xrange(len(labelname_list)):
    label_file = cv2.imread('./labels/%06d.png'%labelname_list[i], -1).reshape((60,36,60))
    label_list.append(label_file)
labels = np.stack(label_list, axis = 0).reshape((499, -1))

f = h5py.File(filename, 'r')
preds = f['result'][:]
preds = np.argmax(preds, axis = 1).reshape((499, -1))

seg = np.zeros((11, 3), dtype = np.int32)
com = np.zeros((3), dtype = np.float32)
for i in xrange(preds.shape[0]):
    label = labels[i]
    pred = preds[i]
    mask = label != 255
    label = label[mask]
    pred = pred[mask]
    ep = eval_prediction(label, pred)
    ec = eval_completion(label, pred)
    seg = seg + ep
    com = com + ec

com = com/preds.shape[0]
seg = seg.astype(np.float32)
precision = seg[:, 0]/(seg[:, 0] + seg[:, 1])
recall = seg[:, 0]/(seg[:, 0] + seg[:, 2])
iou = seg[:, 0]/np.sum(seg, axis = 1)

# print
print 'Semantic Scene Completion:\nprec ,recall ,IoU\n mean: %f,%f,%f'%(np.mean(precision), np.mean(recall), np.mean(iou))
for i in xrange(11):
    print ' %s %f %f %f'%(classname[i], precision[i], recall[i], iou[i])
print 'Scene Completion:\nprec ,recall , IoU\n %f,%f,%f'%(com[1], com[2], com[0])
