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
    pred = np.argmax(pred,axis=0)
    pred = pred.astype(np.int8)
    #print(np.amax(gt),np.amax(pred))
 
    catg_seen = catg_seen | set(np.unique(gt))

    for i in range(81):
        TP[i] += np.sum(np.multiply(np.equal(gt, i).astype(float), np.equal(pred, i).astype(float)))
        FP[i] += np.sum(np.equal(pred, i).astype(float)) - np.sum(np.multiply(np.equal(gt, i).astype(float), np.equal(pred, i).astype(float)))
        FN[i] += np.sum(np.equal(gt, i).astype(float)) - np.sum(np.multiply(np.equal(gt, i).astype(float), np.equal(pred, i).astype(float)))

    if line_counter % 100 == 99:
        IoU = np.divide(TP, FP + TP + FN + 0.0000001)
        iou = np.sum(TP)/(200*200*100)
        recall = np.divide(TP, TP + FN + 0.0000001)
        precision = np.divide(TP, FP + TP + 0.0000001)
        print('processed ' + str(line_counter + 1) + 'images')
        if len(catg_seen) == 81:
            for i in range(81):
                print('catg: ', i, 'recall: ', recall[i], 'precision: ', precision[i], 'IoU: ', IoU[i])
            print('mean IoU: ', np.sum(IoU)/81.0)
            print('mean iou:', iou/81.0)
            print('mean recall: ', np.sum(recall)/81.0)
            print('mean precision: ', np.sum(precision)/81.0)
        else:
            print('still have not seen' + str(81 - len(catg_seen)) + ' categories')
        np.savetxt(args.source + '_results.txt', np.concatenate((np.expand_dims(IoU, axis=1), np.expand_dims(recall, axis=1), np.expand_dims(precision, axis=1)), axis=1))
    
