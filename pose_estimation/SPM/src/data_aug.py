#!/usr/bin/python3
# encoding: utf-8

import cv2
import numpy as np
import random
import copy
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from spm_config import spm_config as params

def data_aug(img, bboxs=None, keypoints=None, test_flag=False):
    '''
    :param img: The image should be made augmentation
    :param bboxs: list, [ [x1, y1, x2, y2], ..., [xn1, yn1, xn2, yn2] ]
    :param keypoints: COCO format or Ai-challenger format, list of list, 
            [ [num_joints x 3], [num_joints x 3], ..., ]
    :return:
    '''
    # is_flip = [random.randint(0, 1), random.randint(0, 1)]
    # seq = iaa.Sequential([
        # iaa.Affine(rotate=(-15, 15), scale=(0.8, 1.2), mode='constant'),
        # iaa.Multiply((0.7, 1.5)),
        # iaa.Grayscale(iap.Choice(a=[0, 1], p=[0.8, 0.2]), from_colorspace='BGR'),
        # iaa.Fliplr(is_flip[0]),
        # iaa.Flipud(is_flip[1]),
    # ])
    
    is_flip = [random.randint(0, 1)]
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-10, 10), scale=(0.5, 1.5),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, mode='constant'),
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.Multiply((0.7, 1.5)),
        iaa.Fliplr(is_flip[0])
    ])

    seq_det = seq.to_deterministic()
    
    bbs = None
    kps = None
    joint_nums = params['joints']
    new_bboxs = []
    new_keypoints = []
    kps_ori = copy.copy(keypoints)
    kps_ori = np.reshape(np.asarray(kps_ori), newshape=(-1, joint_nums, 3)) if kps_ori is not None else None

    if bboxs is not None:
        assert type(bboxs) == type([])
        bbs = ia.BoundingBoxesOnImage([], shape=img.shape)
        for box in bboxs:
            bbs.bounding_boxes.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
    if keypoints is not None:
        kps = ia.KeypointsOnImage([], shape=img.shape)
        assert type(keypoints) == type([])
        for sp_keypoints in keypoints:
            for i in range(joint_nums):
                joint = sp_keypoints[i*3:i*3+3]
                kps.keypoints.append(ia.Keypoint(x=joint[0], y=joint[1]))

    bbs_aug = None
    kps_aug = None
    img_aug = seq_det.augment_image(img)
    if bbs is not None:
        bbs_aug = seq_det.augment_bounding_boxes(bbs)
        for i in range(len(bbs_aug.bounding_boxes)):
            box_aug = bbs_aug.bounding_boxes[i]
            box = [box_aug.x1, box_aug.y1, box_aug.x2, box_aug.y2]
            new_bboxs.append(box)
    if kps is not None:
        kps_aug = seq_det.augment_keypoints(kps)
        for i in range(len(kps_aug.keypoints)):  # Is this right???
            point = kps_aug.keypoints[i]
            new_keypoints.append([point.x, point.y, 1])
        new_keypoints = np.reshape(np.asarray(new_keypoints), newshape=(-1, joint_nums, 3))

        # keep ori keypoint visiable attribute
        for i in range(kps_ori.shape[0]):
            for joint in range(kps_ori.shape[1]):
                new_keypoints[i][joint][2] = kps_ori[i][joint][2]
                if kps_ori[i][joint][0] == 0 or kps_ori[i][joint][1] == 0:
                    new_keypoints[i][joint] = np.asarray([0, 0, 0])

        ''' if flip, change keypoint order (left <-> right)
        coco-format: TODO add coco-foramt change index '''
        change_index = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        for flip in is_flip:
            if flip:
                for i in range(kps_ori.shape[0]):
                    for index in change_index:
                        right_point = copy.copy(new_keypoints[i][index[0]])
                        new_keypoints[i][index[0]] = new_keypoints[i][index[1]]
                        new_keypoints[i][index[1]] = right_point
        new_keypoints = [list(np.reshape(sp_keypoints,(-1,))) for sp_keypoints in new_keypoints]

    '''aug reslut test'''
    if test_flag:
        if bbs is not None:
            img_before = bbs.draw_on_image(img, color=(0, 255, 0), thickness=2)
            img_after = bbs_aug.draw_on_image(img_aug, color=(0,0,255), thickness=2)
            cv2.imshow('box_ori', img_before)
            cv2.imshow('box_aug', img_after)
            cv2.waitKey(0)
        if kps is not None:
            img_before = kps.draw_on_image(img, color=(0, 255, 0), size=5)
            img_after = kps_aug.draw_on_image(img_aug, color=(0, 0, 255), size=5)
            for i in range(kps_ori.shape[0]):
                for joint in range(kps_ori.shape[1]):
                    point = kps_ori[i][joint]
                    cv2.putText(img_before, str(joint), (int(point[0]), int(point[1])), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 1)
                    point = new_keypoints[i][3*joint:3*joint+3]
                    cv2.putText(img_after, str(joint), (int(point[0]), int(point[1])), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 1)
            cv2.imshow('kps_ori', img_before)
            cv2.imshow('kps_aug', img_after)
            cv2.waitKey(0)

    return img_aug, new_bboxs, new_keypoints

