import numpy as np
from scipy import misc, linalg
import argparse
import os,sys
import torch
import math
from PIL import Image

parser = argparse.ArgumentParser(description="Discovering Class Specific Pixels (DCSP) Network.")
parser.add_argument('--source', type=str, default='out_la_crf_aff')
args = parser.parse_args()

img_gt_name_list = open('smallval.txt').read().splitlines()
img_name_list = [img[:-4] for img in img_gt_name_list]

totalintersection = 0
totalunion = 0
catg_seen = set()
for line_counter, img_name in enumerate(img_name_list):
    gt1 = misc.imread('./data/ClassSegmentation/val2014/'+img_name+'.png')
    gt = misc.imresize(gt1,(13,13))
    for val in np.unique(gt):
        if val not in gt1:
            gt[gt==val] = 0
    pred = np.load('./numpyserialised13/'+img_name+'.npz')['arr_0']
    for i in range(13):
        for j in range(13):
            torchx = torch.from_numpy(pred[:,i,j])
            m = torch.nn.Softmax()
            sigmoidtorchx = m(torchx)
            pred[:,i,j] = sigmoidtorchx.cpu().numpy()
    pred[0] = 0.25
    pred = np.argmax(pred,0)
    for i in range(81):
        intersection = np.sum(np.logical_and(np.equal(gt, i), np.equal(pred, i)))
        union = np.sum(np.logical_or(np.equal(gt, i), np.equal(pred, i)))
        totalintersection += intersection
        totalunion += union
    if line_counter%100 == 99:
        print("%d images, IoU: %f"%(line_counter,(totalintersection/float(totalunion))))
print("%d images, IoU: %f"%(line_counter,(totalintersection/float(totalunion))))
