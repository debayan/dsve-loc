"""
****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
Copyright (c) 2018 [Thomson Licensing]
All Rights Reserved
This program contains proprietary information which is a trade secret/business \
secret of [Thomson Licensing] and is protected, even if unpublished, under \
applicable Copyright laws (including French droit d'auteur) and/or may be \
subject to one or more patent(s).
Recipient is to retain this program in confidence and is not permitted to use \
or make copies thereof other than as permitted in a written agreement with \
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or \
by [Thomson Licensing] under express agreement.
Thomson Licensing is a company of the group TECHNICOLOR
*******************************************************************************
This scripts permits one to reproduce training and experiments of:
    Engilberge, M., Chevallier, L., Pérez, P., & Cord, M. (2018, April).
    Finding beans in burgers: Deep semantic-visual embedding with localization.
    In Proceedings of CVPR (pp. 3984-3993)

Author: Martin Engilberge
"""

import numpy as np
import cv2,os,sys

from PIL import Image
from scipy.misc import imresize
from pycocotools import mask as maskUtils


# ################### Functions for the pointing game evaluation ################### #

def regions_scale(x, y, rw, rh, h, w, org_dim, cc=None):
    if cc is None:
        fx = x * org_dim[0] / w
        fy = y * org_dim[1] / h
        srw = rw * org_dim[0] / w
        srh = rh * org_dim[1] / h
    else:
        if (h > w):
            r = float(h) / float(w)

            sx = x * cc / w
            sy = y * cc / w

            srw = rw * cc / w
            srh = rh * cc / w

            fx = sx - (cc - org_dim[0]) / 2
            fy = sy - (cc * r - org_dim[1]) / 2
        else:
            r = float(w) / float(h)

            sx = x * cc / h
            sy = y * cc / h

            srw = rw * cc / h
            srh = rh * cc / h

            fy = sy - (cc - org_dim[1]) / 2
            fx = sx - (cc * r - org_dim[0]) / 2

    return fx, fy, srw, srh


def is_in_region(x, y, bx, by, w, h):
    return (x > bx and x < (bx + w) and y > by and y < (by + h))


def one_img_process(act_map, caps_enc, fc_w, regions, h, w, org_dim, nmax=180, bilinear=False, cc=None):
    size = act_map.shape[1:]
    act_map = act_map.reshape(act_map.shape[0], -1)
    prod = np.dot(fc_w, act_map)

    total = 0
    correct = 0
    for i, cap in enumerate(caps_enc):
        order = np.argsort(cap)[::-1]

        heat_map = np.reshape(
            np.dot(np.abs(cap[order[:nmax]]), prod[order[:nmax]]), size)

        if bilinear:
            heat_map = imresize(heat_map, (org_dim[0], org_dim[1]))
            x, y = np.unravel_index(heat_map.T.argmax(), heat_map.T.shape)
        else:
            x, y = np.unravel_index(heat_map.T.argmax(), heat_map.T.shape)
            if cc is None:
                x = (org_dim[0] / size[0]) * x
                y = (org_dim[1] / size[1]) * y
            else:
                if (h > w):
                    r = float(h) / float(w)
                    x = (org_dim[0] / size[0]) * x + (cc - org_dim[0]) / 2
                    y = (org_dim[1] / size[1]) * y + (cc * r - org_dim[1]) / 2
                else:
                    r = float(w) / float(h)
                    x = (org_dim[0] / size[0]) * x + (cc * r - org_dim[0]) / 2
                    y = (org_dim[1] / size[1]) * y + (cc - org_dim[1]) / 2

        r = regions[i]
        fx, fy, srw, srh = regions_scale(
            r.x, r.y, r.width, r.height, h, w, org_dim, cc)

        if is_in_region(x, y, fx, fy, srw, srh):
            correct += 1
        total += 1

    return correct, total


def compute_pointing_game_acc(imgs_stack, caps_stack, nb_regions, regions, fc_w, org_dim, cc=None, nmax=180):
    correct = 0
    total = 0

    for i, act_map in enumerate(imgs_stack):
        seen_region = sum(nb_regions[:i])
        caps_enc = caps_stack[seen_region:seen_region + nb_regions[i]]
        region = regions[i][1]
        h = regions[i][0].height
        w = regions[i][0].width

        c, t = one_img_process(act_map, caps_enc, fc_w,
                               region, h, w, org_dim, nmax=nmax, cc=cc)
        correct += c
        total += t

    return float(correct) / float(total)


# ################### Functions for the semantic segmentation evaluation ################### #


def generate_heat_map(act_map, caps_enc, fc_w, nmax=180, in_dim=(224, 224)):
    size = act_map.shape[1:]
    act_map = act_map.reshape(act_map.shape[0], -1)
    prod = np.dot(fc_w, act_map)

    order = np.argsort(caps_enc)[::-1]
    # print order
    heat_map = np.reshape(
        np.dot(np.abs(caps_enc[order[:nmax]]), prod[order[:nmax]]), size)
    # print heat_map

    heat_map = np.reshape(heat_map, in_dim)
    return heat_map

def generate_heat_map1(act_map, caps_enc, fc_w, nmax=180):
    size = act_map.shape[1:]
    act_map = act_map.reshape(act_map.shape[0], -1)
    prod = np.dot(fc_w, act_map)

    order = np.argsort(caps_enc)[::-1]
    # print order
    heat_map = np.reshape(
        np.dot(np.abs(caps_enc[order[:nmax]]), prod[order[:nmax]]), size)
    # print heat_map

    return heat_map

def gen_binary_heat_map(maps, concept, fc_w, c_thresh, in_dim=(400, 400)):
    hm = generate_heat_map(maps, concept, fc_w, nmax=10, in_dim=in_dim)

    # hm += abs(np.min(hm))

    def thresh(a, coef):
        return coef * (np.max(a) - np.min(a))

    return np.int32(hm > thresh(hm, c_thresh))

def gen_heat_maps1(maps, concept, fc_w):
    hm = generate_heat_map1(maps, concept, fc_w, nmax=10)

    return hm


def compute_iou(hm, target_mask):
    return np.sum(hm * target_mask) / (np.sum(target_mask) + np.sum(hm) - np.sum(hm * target_mask))


def mask_from_poly(polygons, org_size, in_dim):
    mask_poli = np.zeros((org_size[1], org_size[0]))

    for i in range(len(polygons)):
        if polygons[i][0] == "rle":
            m = maskUtils.decode(polygons[i][1])
            mask_poli += m.squeeze()
        else:
            poly = np.int32(np.array(polygons[i]).reshape(
                (int(len(polygons[i]) / 2), 2)))
            cv2.fillPoly(mask_poli, [poly], [1])

    mask_poli = imresize(mask_poli, in_dim, interp="nearest")

    return np.float32(mask_poli > 0)


def compute_semantic_seg(batch_number, imgs_stack, sizes_list, target_ann, cats_stack, fc_w, c_thresh, in_dim=(200, 200)):

    imgcount = 0
    mAp = 0
    IoUs = dict()
    for k in cats_stack.keys():
        IoUs[k] = list()
        for i in range(imgs_stack.shape[0]):
            imgcount += 1
            if k in target_ann[i]:
                target_mask = mask_from_poly(target_ann[i][k], sizes_list[i], in_dim)
                heat_map = gen_binary_heat_map(imgs_stack[i], cats_stack[k], fc_w, c_thresh, in_dim=in_dim)
                
                iou = compute_iou(heat_map, target_mask)
                target_mask *= 255
                heat_map *= 255
 
#                if not os.path.exists("output/%s"%k):
#                    os.makedirs("output/%s"%k)
#                im = Image.fromarray(target_mask)
#                im = im.convert('L')
#                im.save("output/%s/%d_%d_tgt.png"%(k,batch_number,imgcount))
#                im = Image.fromarray(heat_map)
#                im = im.convert('L')
#                im.save("output/%s/%d_%d_heatmap.png"%(k,batch_number,imgcount))

                # last element of tuple is groundtruth target
                IoUs[k] += [(iou, 1)]
            else:
                # if categorie k is not present in grountruth set iou at 0
                IoUs[k] += [(0, 0)]

    return IoUs

def generate_semantic_seg(batch_number, image_paths, imgs_stack, sizes_list, target_ann, cats_stack, fc_w, c_thresh, in_dim=(200, 200)):
    cocolabels = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
    if not os.path.exists("./numpyserialised13/"):
        os.makedirs("./numpyserialised13/")
    for i in range(imgs_stack.shape[0]):
        image_path = image_paths[i]
        hms = np.zeros((81,13,13))
        for index,k in enumerate(cocolabels):
            hm = gen_heat_maps1(imgs_stack[i], cats_stack[k], fc_w)
            hms[index+1] = hm
        image_path = image_path.replace('jpg','npz')
        np.savez_compressed("./numpyserialised13/%s"%image_path,hms)
