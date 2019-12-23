#!/usr/bin/python3
# encoding: utf-8

import numpy as np
import json

from spm_config import spm_config as params
from pycocotools.coco import COCO

def draw_ttfnet_gaussian(heatmap, center, sigmax, sigmay, mask=None):
    # print (sigmax, sigmay)
    center_x, center_y = int(center[0]), int(center[1])
    th = 4.6052
    delta = np.sqrt(th * 2)

    height = heatmap.shape[0]
    width = heatmap.shape[1]

    x0 = int(max(0, center_x - delta * sigmax + 0.5))
    y0 = int(max(0, center_y - delta * sigmay + 0.5))
    x1 = int(min(width, center_x + delta * sigmax + 0.5))
    y1 = int(min(height, center_y + delta * sigmay + 0.5))

    ## fast way
    arr_heat = heatmap[y0:y1, x0:x1]
    exp_factorx = 1 / 2.0 / sigmax / sigmax
    exp_factory = 1 / 2.0 / sigmay / sigmay
    x_vec = (np.arange(x0, x1) - center_x) ** 2
    y_vec = (np.arange(y0, y1) - center_y) ** 2
    arr_sumx = exp_factorx * x_vec
    arr_sumy = exp_factory * y_vec
    xv, yv = np.meshgrid(arr_sumx, arr_sumy)
    arr_sum = xv + yv
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0

    heatmap[y0:y1, x0:x1] = np.maximum(arr_heat, arr_exp)
    if mask is not None:
        mask[y0:y1, x0:x1] = 1
        return heatmap, mask
    return heatmap

def draw_gaussian(heatmap, center, sigma, mask=None):

    if type(sigma) == type([]):
        sigma = sigma[0]
    # If not turn center into int, the gaussian kernel will not 
    # include position with value 1, beacue center may be a float number.
    center_x, center_y = int(center[0]), int(center[1])
    th = 4.6052
    delta = np.sqrt(th * 2)

    height = heatmap.shape[0]
    width = heatmap.shape[1]

    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))

    x1 = int(min(width, center_x + delta * sigma + 0.5))
    y1 = int(min(height, center_y + delta * sigma + 0.5))

    ## fast way
    arr_heat = heatmap[y0:y1, x0:x1]
    exp_factor = 1 / 2.0 / sigma / sigma
    x_vec = (np.arange(x0, x1) - center_x) ** 2
    y_vec = (np.arange(y0, y1) - center_y) ** 2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0

    heatmap[y0:y1, x0:x1] = np.maximum(arr_heat, arr_exp)
    if mask is not None:
        mask[y0:y1, x0:x1] = 1
        return heatmap, mask
    return heatmap

def gaussian_radius(det_size, min_overlap=0.7):
    width,height  = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_msra_gaussian(heatmap, center, sigma, mask=None):
    tmp_size = sigma * 3
    mu_x = int(center[0])
    mu_y = int(center[1])
    h, w = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    mask[img_y[0]:img_y[1], img_x[0]:img_x[1]] = 1
    return heatmap, mask

def draw_wh(wh_map, center, wh):
    center_x, center_y = int(center[0]), int(center[1])

    height = wh_map.shape[0]
    width = wh_map.shape[1]
    
    x0 = max(0, center_x - 1)
    x1 = min(center_x + 2, width)
    y0 = max(0, center_y - 1)
    y1 = min(center_y + 2, height)

    w, h = wh[0], wh[1]
    wh_map[y0:y1, x0:x1, 0] = w
    wh_map[y0:y1, x0:x1, 1] = h

    return wh_map

def draw_center_reg(center_reg_map, center, mask):
    center_x, center_y = int(center[0]), int(center[1])

    height = center_reg_map.shape[0]
    width = center_reg_map.shape[1]

    x0 = max(0, center_x - 1)
    x1 = min(center_x + 2, width)
    y0 = max(0, center_y - 1)
    y1 = min(center_y + 2, height)

    x_int, y_int = int(center[0]), int(center[1])
    center_reg_map[y0:y1, x0:x1, 0] = center[0] - x_int
    center_reg_map[y0:y1, x0:x1, 1] = center[1] - y_int

    mask[y0:y1, x0:x1] = 1
    return center_reg_map, mask

def draw_center_kps_offset(center_kps_offset, center, kp, mask):
    center_x, center_y = int(center[0]), int(center[1])
    height = center_kps_offset.shape[0]
    width = center_kps_offset.shape[1]
    x0 = max(0, center_x - 1)
    x1 = min(center_x + 2, width)
    y0 = max(0, center_y - 1)
    y1 = min(center_y + 2, height)
    center_kps_offset[y0:y1, x0:x1, :] = np.asarray(kp[:2]) - np.asarray([center_x, center_y])
    mask[y0:y1, x0:x1, :] = 1
    return center_kps_offset, mask

def nms(inputs, size=3):
    assert len(inputs.shape) == 3
    padding = (size - 1)
    pad_zeros = np.zeros(shape=(inputs.shape[0]+padding, inputs.shape[1]+padding, inputs.shape[2]), dtype=inputs.dtype)
    pad_zeros[:-padding, :-padding,:] = inputs
    out = np.zeros_like(inputs)
    for c in range(inputs.shape[2]):
        for row in range(inputs.shape[0]):
            for col in range(inputs.shape[1]):
                left = col
                top = row
                right = col + size
                bot = row + size
                out[row, col, c] = np.max(pad_zeros[top:bot, left:right, c])
    return out

def point_nms(inputs, score=0.1, dis=10):
    '''
    NMS function based on point value and point's euclidean distance
    Note that returns points cooridinat is (x, y) for numpy format, so actually it is (y, x) on image
    :param inputs:
    :param score:
    :param dis:
    :return:
        kept coors: [ [x,y], [x,y], ..., [x,y] ]
    '''
    inputs = np.asarray(inputs)
    assert len(inputs.shape) == 3
    kept_coors = []
    for c in range(inputs.shape[2]):
        heatmap = inputs[...,c]
        x, y = np.where(heatmap > score)
        coors = list(zip(x, y))
        scores = []
        for coor in coors:
            coor_score = heatmap[coor]
            scores.append(coor_score)
        scores_index = np.asarray(scores).argsort()[::-1]
        kept = []
        kept_coor = []
        while scores_index.size > 0:
            kept.append(scores_index[0])
            coors_score = list(coors[kept[-1]])
            coors_score.append(scores[scores_index[0]])
            kept_coor.append(coors_score)
            scores_index = scores_index[1:]
            last_index = []
            for index in scores_index:
                distance = np.sqrt(np.sum(np.square(
                    np.asarray(coors[kept[-1]]) - np.asarray(coors[index])
                )))
                if distance > dis:
                    last_index.append(index)
            scores_index = np.asarray(last_index)

        kept_coors.append(kept_coor)
    return kept_coors
    

def clip(num, minx, maxx):
    num = max(minx, num)
    num = min(maxx, num)
    return num

'''
About the center points:
Here just treat the center of one person's box as root joint (center point) of one person, 
so there is always has root joint if one person can been seen in image. Ignore its completeness. 
I dose not 100% sure if I am right or not, but the author didn't make clear whcih root joint is, 
so if someone has interest with it, you can email to the author and make root joint clear.
'''
def prepare_bbox(bboxs, orih, oriw, outh, outw):
    '''
    :param bboxs: lists of list, [ [x1,y1,x2,y2], [x1,y1,x2,y2], ..., [xn1,yn1,xn2,yn2] ]
    :param orih: original img height
    :param oriw: original img width
    :param outh: network output height
    :param outw: network output width
    :return:
        centers: [ [ centerx, centery], ..., [centerx, centery] ]
        sigmas:  [ [sigmax, sigmay], [sigmax, sigmay], ..., [sigmax, sigmay]]
        whs:     [ [w1, h1], [w2, h2], ..., [wn, hn] ]
        they all have been rescale to network output size.
    '''
    centers = []
    sigmas = []
    whs = []
    factory = orih / outh
    factorx = oriw / outw
    alpha = 1.5
    for box in bboxs:
        x1 = clip(box[0], 0, oriw) / factorx
        y1 = clip(box[1], 0, orih) / factory
        x2 = clip(box[2], 0, oriw) / factorx
        y2 = clip(box[3], 0, orih) / factory
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        center[0] = clip(center[0], 0, outw - 1)
        center[1] = clip(center[1], 0, outh - 1)
        # In ttfnet, they set two sigmas, sigma_x = box_w / (6 * 0.54), sigma_y = box_h / ( 6 * 0.54)
        # sigma = gaussian_radius((x2 - x1, y2 - y1))
        # sigma = clip(sigma, 4, 7.)
        sigmax = (x2-x1) / (6*alpha) + 0.8
        sigmay = (y2-y1) / (6*alpha) + 0.8
        # print (sigmax, sigmay)
        wh = [x2 - x1, y2 - y1]
        # sigmas.append(sigma)
        sigmas.append([sigmax, sigmay])
        whs.append(wh)
        centers.append(center)
    return centers, sigmas, whs

def prepare_kps(kps, orih, oriw, outh, outw):
    '''
    :param kps: list of list, [ [n x 3], [n x 3], ..., ]
    :param orih: original img height
    :param oriw: original img width
    :param outh: network output height
    :param outw: network output width
    :return:
        keypoints: [ [n x 3], [n x 3], ..., ]
        sigmas:    [ sigma1, sigma2, ...., ]
        they all have been rescale to network output size.
    '''
    keypoints = []
    sigmas = []
    factory = orih / outh
    factorx = oriw / outw
    for kp in kps:
        kp = np.reshape(np.asarray(kp, dtype=np.float), (-1,3))
        kp[:,0] /= factorx
        kp[:,1] /= factory
        for i in range (kp.shape[0]):
            if kp[i, 0] == 0 or kp[i, 1] == 0 or kp[i, 2] == -1:
                kp[i, :] = 0
        kp = list(np.reshape(kp, (-1,)))
        keypoints.append(kp)
        # If This sigma can be adjusted according to the size of the object, it is better.
        sigmas.append(params['kps_sigma'])
    return keypoints, sigmas
  
  
    
def prepare_bbox_kps(bboxs, kps, orih, oriw, outh, outw):
    centers = []
    sigmas = []
    keypoints = []
    kps_sigmas = []
    factory = orih / outh
    factorx = oriw / outw
    for box, kp in zip(bboxs, kps):
        kp = np.reshape(np.asarray(kp, dtype=np.float), (-1,3))
        kp[:,0] /= factorx
        kp[:,1] /= factory
        for i in range (kp.shape[0]):
            if kp[i, 0] == 0 or kp[i, 1] == 0 or kp[i, 2] == -1:
                kp[i, :] = 0
        x1 = clip(box[0], 0, oriw) / factorx
        y1 = clip(box[1], 0, orih) / factory
        x2 = clip(box[2], 0, oriw) / factorx
        y2 = clip(box[3], 0, orih) / factory
        sigmax = (1 + (x2-x1)/oriw) * params['sigma']
        sigmay = (1 + (y2-y1)/orih) * params['sigma']
        sigmas.append([sigmax, sigmay])
        if kp[5,0]==0 or kp[6,0]==0 or kp[11,0]==0 or kp[12,0]==0:
            center_x = clip((x1 + x2)/2, 0, outw - 1)
            center_y = clip((y1 + y2)/2, 0, outh - 1)
        else:
            center_x = clip((kp[5,0] + kp[6,0] + kp[11,0] + kp[12,0])/4, 0, outw - 1)
            center_y = clip((kp[5,1] + kp[6,1] + kp[11,1] + kp[12,1])/4, 0, outh - 1)
        centers.append([center_x, center_y])
        kp = list(np.reshape(kp, (-1,)))
        keypoints.append(kp)
        kps_sigma = (np.sqrt((x2-x1)*(y2-y1)/(oriw*orih)) + 1) * params['sigma']
        kps_sigmas.append(kps_sigma)
    return centers, sigmas, keypoints, kps_sigmas 
        
    

'''
read ai-format json files, returns:
1. img_ids, a list, contains all img id
2. id_bboxs_dict, a dictory, kye is img_id, value is corresponding bboxes as format [ [x1, y1, x2, y2], ..., [x1, y1, x2, y2] ]
3. id_kps_dict, a dictory, key is img_id, value is corresponding keypoints as format [ [ x1, y1, v1, x2, y2, v2, ..., x14, y14, v14] ... ]
:param json_file:
:return:
'''
def read_json_AI_challenge(json_file):
    img_ids = []
    id_bboxs_dict = {}
    id_kps_dict = {}
    f = open(json_file, encoding='utf-8')
    annos = json.load(f)
    for anno in annos:
        img_id = anno['image_id']
        img_ids.append(img_id)
        bboxs = []
        for human, box in anno['human_annotations'].items():
            bboxs.append(box)
        id_bboxs_dict[img_id] = bboxs

        kps = []
        for human, kp in anno['keypoint_annotations'].items():
            '''set v=-1 for point is (0, 0) or (0, x) or (x, 0)'''
            for i in range(14):
                if kp[3*i+0] == 0 or kp[3*i+1] == 0:
                    kp[3*i+2] = -1
            kps.append(kp)
        id_kps_dict[img_id] = kps

    return img_ids, id_bboxs_dict, id_kps_dict


''' COCO humans keypoints json '''
def read_json_COCO(json_file):
    img_ids = []
    id_bboxs_dict = {}
    id_kps_dict = {}
    f = open(json_file, encoding='utf-8')
    annos = json.load(f)
    for anno in annos['annotations']:
        if anno['num_keypoints']==0: continue
        img_id = anno['image_id']
        
        if img_id in id_bboxs_dict.keys():
            [x, y, w, h] = anno['bbox']
            id_bboxs_dict[img_id].append([x, y, x+w, y+h])
        else:
            [x, y, w, h] = anno['bbox']
            bboxs = [[x, y, x+w, y+h]]
            id_bboxs_dict[img_id] = bboxs

        kp = anno['keypoints']
        ''' set v=-1 for point is (0, 0) or (0, x) or (x, 0) '''
        for i in range(params['joints']):
            if kp[3*i+0] == 0 or kp[3*i+1] == 0:
                kp[3*i+2] = -1
        if img_id in id_kps_dict.keys():
            id_kps_dict[img_id].append(kp)
        else:
            kps = [kp]
            id_kps_dict[img_id] = kps
    img_ids = list(id_bboxs_dict.keys())
    return img_ids, id_bboxs_dict, id_kps_dict

'''Some tiny persons in COCO are not annotated with keypoints.
So we have to remove them out to prevent polluting the training.'''
def read_json_COCO_v2(json_file):
    coco = COCO(json_file)
    img_ids = list(coco.imgs.keys())
    
    img_ids_left = []
    id_bboxs_kps_masks_dict = {}
    id_img_names_dict = {}
    
    print("Original Images Number: ", len(img_ids))
    for img_id in img_ids:
        height, width = coco.imgs[img_id]['height'], coco.imgs[img_id]['width']
        removed_mask = np.zeros((height, width), dtype='bool')
        bboxes = []
        keypoints = []
        
        img_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        if len(img_anns) == 0: continue
        
        for anno in img_anns:
            if anno['area']==0: continue
            mask = coco.annToMask(anno)
            if anno['iscrowd'] == 1:
                removed_mask = np.logical_or(removed_mask, mask) # crowd_mask
            elif anno['num_keypoints'] == 0:
                removed_mask = np.logical_or(removed_mask, mask) # unannotated_mask
                # keypoints.append(anno['keypoints'])
            else:
                [x, y, w, h] = anno['bbox']
                bboxes.append([x, y, x+w, y+h])
                keypoints.append(anno['keypoints'])    
        
        if len(bboxes)!=0 and len(keypoints)!=0:
            img_ids_left.append(img_id)
            id_bboxs_kps_masks_dict[img_id] = [bboxes, keypoints, np.logical_not(removed_mask)]
            id_img_names_dict[img_id] = coco.imgs[img_id]['file_name']
    print("Left Images Number: ", len(img_ids_left))
    return img_ids_left, id_bboxs_kps_masks_dict, id_img_names_dict
