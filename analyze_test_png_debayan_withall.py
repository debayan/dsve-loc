import numpy as np
from scipy import misc, linalg
import argparse
import os,sys
import torch
import math
from PIL import Image

cocolabels = ['bg','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

parser = argparse.ArgumentParser(description="DVSE Engilberge IoU evaluation by johann/debayan")
parser.add_argument('--thresh', type=float, default=0.5)
args = parser.parse_args()

img_gt_name_list = open('val5k.txt').read().splitlines()
img_name_list = [img[:-4] for img in img_gt_name_list]

totalintersection = np.zeros((81))
totalunion = np.zeros((81))
for line_counter, img_name in enumerate(img_name_list):
    gt = misc.imread('./data/ClassSegmentation/val2014/'+img_name+'.png')
    pred_ = np.load('./numpyserialised13/'+img_name+'.npz')['arr_0'] #This is 81 channels thick, each channel 13x13. channel 0 = 0, other channels have values from -inf to +inf, each channel belonging to a coco class
    pred = np.zeros((81, gt.shape[0], gt.shape[1]))
    for i in range(81):
        pred[i] = misc.imresize(pred_[i],(gt.shape[0], gt.shape[1]), mode='F')
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            torchx = torch.from_numpy(pred[:,i,j])
            m = torch.nn.Sigmoid()
            sigmoidtorchx = m(torchx)
            pred[:,i,j] = sigmoidtorchx.cpu().numpy()
    pred[0] = args.thresh
    pred1c = np.argmax(pred,0) #pred is 81 channels, pred1c is 1 channel
    for i in range(81):
        totalintersection[i] += np.sum(np.logical_and(np.equal(gt, i), np.equal(pred1c, i)))
        totalunion[i] += np.sum(np.logical_or(np.equal(gt, i), np.equal(pred1c, i)))
    if line_counter%100 == 99:
        iou = np.sum(np.divide(totalintersection, totalunion+0.000001))/81.0
        for l in range(81):
            print("%s = %f"%(cocolabels[l], totalintersection[l]/float(totalunion[l])),end=' ')
        print("%d images, IoU: %f"%(line_counter, iou))
        print("%d classes still not seen"%(81-np.count_nonzero(totalintersection)))

iou = np.sum(np.divide(totalintersection, totalunion+0.000001))/81.0
for l in range(81):
    print("%s = %f"%(cocolabels[l], totalintersection[l]/float(totalunion[l])),end=' ')
print("%d images, IoU: %f"%(line_counter, iou))
print("%d classes still not seen"%(81-np.count_nonzero(totalintersection)))
