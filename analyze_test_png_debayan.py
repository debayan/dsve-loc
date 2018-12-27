import numpy as np
from scipy import misc, linalg
import argparse
import os,sys
import math
from PIL import Image

parser = argparse.ArgumentParser(description="Discovering Class Specific Pixels (DCSP) Network.")
parser.add_argument('--source', type=str, default='out_la_crf_aff')
args = parser.parse_args()

img_gt_name_list = open('smallval.txt').read().splitlines()
img_name_list = [img[:-4] for img in img_gt_name_list]

def sigmoid(x):
  return 1 / float(1 + math.exp(-x))
sigmoid_v = np.vectorize(sigmoid)

TP = np.zeros((81))
FP = np.zeros((81))
FN = np.zeros((81))

count = 0
totaliou = 0.0
catg_seen = set()
for line_counter, img_name in enumerate(img_name_list):
    gt1 = misc.imread('./data/ClassSegmentation/val2014/'+img_name+'.png')
    gt = misc.imresize(gt1,(200,200))
    for val in np.unique(gt):
        if val not in gt1:
            gt[gt==val] = 0
    pred = np.load('./numpyserialised/'+img_name+'.npz')['arr_0']
 
    catg_seen = catg_seen | set(np.unique(gt))
    for i in range(0,81):
        if i in gt:
            predslice = pred[i]
            predslice[predslice < 0.45*255] = 0
            predslice[predslice >= 0.45*255] = i
            count += 1
            intersection = np.sum(np.logical_and(np.equal(gt, i), np.equal(predslice, i)))
            union = np.sum(np.logical_or(np.equal(gt, i), np.equal(predslice, i)))
            iou = intersection/float(union)
            totaliou += iou
    if line_counter%100 == 99:
        print("%d images, IoU: %f"%(line_counter,totaliou/float(count)))
print("%d images, IoU: %f"%(line_counter,totaliou/float(count)))
