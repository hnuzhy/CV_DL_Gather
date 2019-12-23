#!/usr/bin/python3
# encoding: utf-8

import tensorflow as tf
import os
import cv2
import argparse
import time

import numpy as np
from src.spm_model import SpmModel
from src.spm_decoder import SpmDecoder
from src.spm_config import spm_config as params

parser = argparse.ArgumentParser()
parser.add_argument('--video', default=None, type=str)
parser.add_argument('--imgs', default='./imgs/021_ch38_1354.jpg', type=str)
# parser.add_argument('--imgs', default='./imgs/000000013291.jpg', type=str)
# parser.add_argument('--imgs', default='./imgs/0100.jpg', type=str)
parser.add_argument('--ckpt', default='./ckpt/2019-12-18-22-51/ckpt-2', type=str)
parser.add_argument('--use_gpu', default=False, type=bool)
args = parser.parse_args()


netH, netW = params['height'], params['width']
score = params['nms_score']
dist = params['nms_dist']
scales = params['scale_search']


@tf.function
def tf_model_infer(model, inputs):
    center_map, kps_reg_map = model(inputs)
    return center_map, kps_reg_map

def inference_run(model, img):
    img_show = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    
    '''Only padding below or right in the original image. 
    This will not influence the position values of boxes or points.
    We need not to change their coordinate values after prediction.'''
    temp_scale_h, temp_scale_w = int(w*netH/netW), int(h*netW/netH)
    if w > temp_scale_w:
        img = cv2.copyMakeBorder(img, 0, temp_scale_h-h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        img = cv2.copyMakeBorder(img, 0, 0, 0, temp_scale_w-w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    factor_x = img.shape[1] / netW
    factor_y = img.shape[0] / netH
    spm_decoder = SpmDecoder(factor_x, factor_y, netH, netW)

    center_map_avg = tf.zeros(shape=(netH, netW, 1), dtype=np.float32)
    kps_reg_map_avg = tf.zeros(shape=(netH, netW, params['joints']*2), dtype=np.float32)
    for i, scale in enumerate(scales):
        print("scale:", scale)
        img = cv2.resize(img, (int(netH*scale), int(netW*scale)), interpolation=cv2.INTER_CUBIC)
        img_input = tf.expand_dims(img.astype(np.float32) / 255., axis=0)
        center_map, kps_reg_map = tf_model_infer(model, img_input)
        center_map_avg = tf.add(center_map_avg, tf.image.resize(center_map, [netH, netW])[0])
        kps_reg_map_avg = tf.add(kps_reg_map_avg, tf.image.resize(kps_reg_map/scale, [netH, netW])[0])
    center_map = tf.divide(center_map_avg, len(scales))
    kps_reg_map = tf.divide(kps_reg_map_avg, len(scales))
    cv2.imwrite(args.imgs[:-4]+'_center_map.jpg', np.array(center_map*255))
    img_heatmap = spm_decoder.visualization_heatmap(img_show.copy(), center_map)
    cv2.imwrite(args.imgs[:-4]+'_img_heatmap.jpg', img_heatmap)
    
    joints, centers = spm_decoder([center_map, kps_reg_map], score_thres=score, dis_thres=dist)
    print("PoseNum:", len(centers), "\n", centers)
    img_keypoints = spm_decoder.visualization_skeleton_coco(img_show.copy(), centers, joints)
    cv2.imwrite(args.imgs[:-4]+'_result.jpg', img_keypoints)
    

if __name__ == '__main__':
    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    inputs = tf.keras.Input(shape=(netH, netW, 3), name='modelInput')
    outputs = SpmModel(inputs, params['joints'], is_training=False)
    model = tf.keras.Model(inputs, outputs)
    
    # tf.keras.utils.plot_model(model, 'SPM_network.png', show_shapes=True)

    assert args.ckpt is not None
    ckpt_object = tf.train.Checkpoint(net=model)
    ckpt_object.restore(args.ckpt)
    # ckpt.restore(args.ckpt).assert_existing_objects_matched()  # error occured
    # model.load_weights(args.ckpt)  # This is for Keras file .h5
    print('Model has been loaded successfully!')

    # tf.keras.utils.plot_model(model, 'hrnet.png', show_shapes=True)

    if args.video is not None:
        cap = cv2.VideoCapture(0)
        ret, img = cap.read()
        while ret:
            tic = time.time()
            inference_run(model, img)
            print ('Time taking: ', time.time()-tic)
            ret, img = cap.read()
    elif os.path.isdir(args.imgs):
        for img_name in os.listdir(args.imgs):
            tic = time.time()
            inference_run(model, cv2.imread(os.path.join(args.imgs, img_name)))
            print ('Time taking: ', time.time()-tic)
    elif os.path.isfile(args.imgs):
        tic = time.time()
        inference_run(model, cv2.imread(args.imgs))
        print ('Time taking: ', time.time()-tic)
    else:
        print ('You Must Provide one video or imgs/img_path')


