#!/usr/bin/python3
# encoding: utf-8

import tensorflow as tf
import numpy as np
import random
import json
import cv2
import os
import math
from spm_config import spm_config as params
# from utils import clip, gaussian_radius, prepare_bbox, read_json, prepare_kps
from utils import prepare_bbox_kps, read_json_COCO_v2
from data_aug import data_aug
from spm_encoder import SingleStageLabel

id_bboxs_kps_masks_dict = None
id_img_names_dict = None
img_path = None

def get_dataset(num_gpus = 1, mode = 'train'):
    assert mode in ['train', 'val']
    global img_path, params, id_bboxs_kps_masks_dict, id_img_names_dict

    if mode == 'train':
        json_file = params['train_json_file']
        img_path  = params['train_img_path']
    else:
        json_file = params['val_json_file']
        img_path  = params['val_img_path']
    
    img_ids, id_bboxs_kps_masks_dict, id_img_names_dict = read_json_COCO_v2(json_file)
    if mode == 'train':
        random.shuffle(img_ids)
        dataset = tf.data.Dataset.from_tensor_slices(img_ids)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(img_ids)

    dataset = dataset.shuffle(buffer_size=1000).repeat(1)

    if mode == 'train':
        dataset = dataset.map(tf_parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(tf_parse_func_for_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(params['batch_size'] * num_gpus, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def tf_parse_func_for_val(img_id):
    [img_id, height, width, img] = tf.py_function(paser_func_for_val, [img_id], [tf.string, tf.float32, tf.float32, tf.float32])
    img.set_shape([params['height'], params['width'], 3])
    return img_id, height, width, img

def paser_func_for_val(img_id):
    global params, img_path, id_img_names_dict

    # img = cv2.imread(os.path.join(img_path, str(img_id) + '.jpg'))
    img = cv2.imread(os.path.join(img_path, id_img_names_dict[img_id.numpy()]))
    orih, oriw, oric = img.shape

    # padding img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    temp_scale_h, temp_scale_w = int(w*neth/netw), int(h*netw/neth)
    if w > temp_scale_w:
        img = cv2.copyMakeBorder(img, 0, temp_scale_h-h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        img = cv2.copyMakeBorder(img, 0, 0, 0, temp_scale_w-w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # create img input
    img = cv2.resize(img, (params['width'], params['height']), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.  # conver to 0~1 tools if focal loss is right

    return img_id, orih, oriw, img
    

def tf_parse_func(img_id):
    [img, center_map, kps_map] = tf.py_function(paser_func, [img_id], [tf.float32, tf.float32, tf.float32])
    # [img, center_map, kps_map, kps_map_weight] = tf.py_function(paser_func, [img_id], [tf.float32, tf.float32, tf.float32, tf.float32])

    img.set_shape([params['height'], params['width'], 3])
    center_map.set_shape([params['height'], params['width'], 1])
    kps_map.set_shape([params['height'], params['width'], params['joints']*2])
    # kps_map_weight.set_shape([params['height'], params['width'], params['joints']*2])
    
    return img, center_map, kps_map
    # return img, center_map, kps_map, kps_map_weight

def paser_func(img_id):
    global params, img_path, id_bboxs_kps_masks_dict, id_img_names_dict
    
    neth, netw = params['height'], params['width']
    outh, outw = neth.copy(), netw.copy()
    
    [bboxs, kps, mask] = id_bboxs_kps_masks_dict[img_id.numpy()]
    # img = cv2.imread(os.path.join(img_path, str(img_id)+'.jpg'))
    img = cv2.imread(os.path.join(img_path, id_img_names_dict[img_id.numpy()]))
    # unannotated and crowd masks
    for c_id in range(3):
        img[:,:,c_id] = img[:,:,c_id] * mask
    # data aug
    img, bboxs, kps = data_aug(img, bboxs, kps)
    
    # padding image, keep the width/height ratio stable
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    temp_scale_h, temp_scale_w = int(w*neth/netw), int(h*netw/neth)
    if w > temp_scale_w:
        img = cv2.copyMakeBorder(img, 0, temp_scale_h-h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        img = cv2.copyMakeBorder(img, 0, 0, 0, temp_scale_w-w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # create center label
    orih, oriw, oric = img.shape
    # centers, sigmas, whs = prepare_bbox(bboxs, orih, oriw, outh, outw)
    # keypoints, kps_sigmas = prepare_kps(kps, orih, oriw, outh, outw)
    centers, sigmas, keypoints, kps_sigmas = prepare_bbox_kps(bboxs, kps, orih, oriw, outh, outw)
    spm_label = SingleStageLabel(outh, outw, centers, sigmas, keypoints, kps_sigmas)
    # center_map, kps_map, kps_map_weight = spm_label()
    center_map, kps_map = spm_label()

    # create img input
    img = cv2.resize(img, (netw, neth), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.
    
    return img, center_map, kps_map
    # return img, center_map, kps_map, kps_map_weight
