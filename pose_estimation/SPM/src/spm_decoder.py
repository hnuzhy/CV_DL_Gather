#!/usr/bin/python3
# encoding: utf-8

import numpy as np
import math
from utils import point_nms
from src.spm_config import spm_config as params
import cv2

class SpmDecoder():
    def __init__(self, factor_x, factor_y, outw, outh):
        self.factor_x = factor_x
        self.factor_y = factor_y
                      
        ''' COCO: hierarchical SPR '''
        self.level = [[0, 1, 3],
                      [0, 2, 4],
                      [5, 7, 9],
                      [6, 8, 10],
                      [11, 13, 15],
                      [12, 14, 16]]
                      
        self.limb_connection = [[0,5], [0,6], [0,1], [0,2], [1,3], [2,4], 
                                [5,6], [5,7], [7,9], [5,11], [6,8], [8,10], [6,12], 
                                [11,12], [11,13], [13,15], [12,14], [14,16]]
        self.person_colors = [[0,128,255], [0,255,128], [128,0,255], [128,255,0], [255,0,128], [255,128,0],
                              [0,0,255], [0,255,0], [255,0,0], [0,255,255], [255,0,255], [255,255,0], 
                              [0,0,128], [0,128,0], [128,0,0], [0,128,128], [128,0,128], [128,128,0],
                              [128,128,255], [128,255,128], [255,128,128], [128,255,255], [255,128,255], [255,255,128]]
        self.colors_num = len(self.person_colors)
                      
        # self.Z = math.sqrt(outw*outw + outh*outh)
        # print ('decoder self.z', self.Z)
        self.Z = 1
        self.outw = outw
        self.outh = outh

    def __call__(self, spm_label, score_thres=0.9, dis_thres=5):
        center_map, kps_map = np.array(spm_label[0]), np.array(spm_label[1])
        keep_coors = point_nms(center_map, score_thres, dis_thres)
        centers = keep_coors[0]
        all_joints, ret_centers = [], []
        for center in centers:
            single_person_joints = np.zeros(params['joints']*2)
            ret_centers.append([center[1]*self.factor_x, center[0]*self.factor_y, center[2]])
            for single_path in self.level:
                for i, index in enumerate(single_path):
                    if i == 0:
                        start_joint = [int(center[1]), int(center[0])]
                    if start_joint[0] >= self.outw or start_joint[1] >= self.outh \
                            or start_joint[0] < 0 or start_joint[1] < 0:
                        break
                    offset = kps_map[start_joint[1], start_joint[0], 2*index:2*index+2] * self.Z
                    joint = [start_joint[0]+offset[0], start_joint[1]+offset[1]]
                    single_person_joints[2*index:2*index+2] = joint
                    start_joint = [int(x) for x in joint]
            
            for i in range(params['joints']):
                single_person_joints[2*i] *= self.factor_x
                single_person_joints[2*i+1] *= self.factor_y
            all_joints.append(single_person_joints)

        return all_joints, ret_centers
        
    def visualization_skeleton_coco(self, img_show, centers, keypoints):
        for j, single_person_joints in enumerate(keypoints):
            color = self.person_colors[j%self.colors_num]
            cx, cy = int(centers[j][0]), int(centers[j][1])
            # cv2.circle(img_show, (cx, cy), 4, color, thickness=-1)
            '''Paint keypoints'''
            joints_mark = np.zeros(params['joints'])
            for i in range(params['joints']):
                x = int(single_person_joints[2*i])
                y = int(single_person_joints[2*i+1])
                if x!=0 or y!=0:
                    joints_mark[i] = 1
                    # cv2.circle(img_show, (x, y), 4, (255,255,255), thickness=-1)
            '''Paint limbs connection'''
            for id, [p1, p2] in enumerate(self.limb_connection):
                if joints_mark[p1] and joints_mark[p2]:
                    p1_x, p1_y = int(single_person_joints[2*p1]), int(single_person_joints[2*p1+1])
                    p2_x, p2_y = int(single_person_joints[2*p2]), int(single_person_joints[2*p2+1])
                    cv2.line(img_show, (p1_x, p1_y), (p2_x, p2_y), color, 2)
                    cv2.circle(img_show, (p1_x, p1_y), 2, (255,255,255), thickness=-1)
                    cv2.circle(img_show, (p2_x, p2_y), 2, (255,255,255), thickness=-1)
            cv2.circle(img_show, (cx, cy), 4, (0,0,0), thickness=-1)                    
        return img_show
        
    def visualization_skeleton_spm(self, img_show, centers, keypoints):
        for j, single_person_joints in enumerate(keypoints):
            color = self.person_colors[j%self.colors_num]
            cx, cy = int(centers[j][0]), int(centers[j][1])
            # cv2.circle(img_show, (cx, cy), 4, color, thickness=-1)
            '''Paint keypoints'''
            joints_mark = np.zeros(params['joints'])
            for i in range(params['joints']):
                x = int(single_person_joints[2*i])
                y = int(single_person_joints[2*i+1])
                if x!=0 or y!=0:
                    joints_mark[i] = 1
                    # cv2.circle(img_show, (x, y), 4, (255,255,255), thickness=-1)
            '''Paint limbs connection'''
            for single_path in self.level:
                for i, index in enumerate(single_path):
                    if i==0:
                        start_joint = [cx, cy]
                    '''Under this connection, some keypoints may be lost and influence the pose'''
                    if joints_mark[index]:
                        end_x = int(single_person_joints[2*index])
                        end_y = int(single_person_joints[2*index+1])
                        if start_joint is not None:
                            cv2.line(img_show, (start_joint[0], start_joint[1]), (end_x, end_y), color, 2)
                            cv2.circle(img_show, (end_x, end_y), 2, (255,255,255), thickness=-1)
                        start_joint = [end_x, end_y]
                    else:
                        start_joint = None
            cv2.circle(img_show, (cx, cy), 4, (0,0,0), thickness=-1)
        return img_show
        
    def visualization_heatmap(self, img, center_map):
        '''Crop and resize the output center_map into the same shape with original img'''
        h, w, c = img.shape
        temp_scale_h, temp_scale_w = int(w*self.outh/self.outw), int(h*self.outw/self.outh)
        if w > temp_scale_w:
            heatmap = cv2.resize(np.array(center_map), (w, w))
            heatmap = heatmap[0:h, :]
        else:
            heatmap = cv2.resize(np.array(center_map), (h, h))
            heatmap = heatmap[:, 0:w]
            
        heatmapshow = None
        heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        img_heatmap = heatmapshow * 0.5 + img
        return img_heatmap