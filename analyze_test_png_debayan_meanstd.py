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

img_gt_name_list = open('smallval.txt').read().splitlines()
img_name_list = [img[:-4] for img in img_gt_name_list]

totalintersection = np.zeros((81))
totalunion = np.zeros((81))
presentmean = 0.0
presentstd = 0.0
absentmean = 0.0
absentstd = 0.0
presentcount = 0
absentcount = 0
for line_counter, img_name in enumerate(img_name_list):
    gt = misc.imread('./data/ClassSegmentation/val2014/'+img_name+'.png')
    pred = np.load('./numpyserialised13/'+img_name+'.npz')['arr_0'] #This is 81 channels thick, each channel 13x13. channel 0 = 0, other channels have values from -inf to +inf, each channel belonging to a coco class 
    for i in range(13):
        for j in range(13):
            torchx = torch.from_numpy(pred[:,i,j])
            m = torch.nn.Sigmoid()
            sigmoidtorchx = m(torchx)
            pred[:,i,j] = sigmoidtorchx.cpu().numpy()
    pred[0] = args.thresh
    for i in range(1,81):
        if i in np.unique(gt):
            presentmean += np.mean(pred[i])
            presentstd += np.std(pred[i])
            presentcount += 1
        else:
            absentmean += np.mean(pred[i])
            absentstd += np.std(pred[i])
            absentcount += 1
        
    #pred1c = np.argmax(pred,0) #pred is 81 channels, pred1c is 1 channel
    #print(pred1c)
    #pred1cresized = np.resize(pred1c,gt.shape)
#    for i in range(81):
#        totalintersection[i] += np.sum(np.logical_and(np.equal(gt, i), np.equal(pred1cresized, i)))
#        totalunion[i] += np.sum(np.logical_or(np.equal(gt, i), np.equal(pred1cresized, i)))
    if line_counter%1000 == 999:
        print("%d images, present mean = %f, present std = %f, absent mean = %f, absent std = %f"%(line_counter,presentmean/float(presentcount), presentstd/float(presentcount),absentmean/float(absentcount),absentstd/float(absentcount)))
#
#iou = np.sum(np.divide(totalintersection, totalunion+0.000001))/81.0
#for l in range(81):
#    print("%s = %f"%(cocolabels[l], totalintersection[l]/float(totalunion[l])),end=' ')
#print("%d images, IoU: %f"%(line_counter, iou))
#print("%d classes still not seen"%(81-np.count_nonzero(totalintersection)))
