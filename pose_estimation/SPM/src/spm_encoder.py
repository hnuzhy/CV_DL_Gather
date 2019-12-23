#!/usr/bin/python3
# encoding: utf-8

import numpy as np
import math
from utils import draw_gaussian, clip, draw_ttfnet_gaussian
from spm_config import spm_config as params

class SingleStageLabel():
    def __init__(self, height, width, centers, sigmas, kps, kps_sigmas):
        self.centers = centers
        self.sigmas = sigmas
        self.kps = kps
        self.kps_sigmas = kps_sigmas
        self.height = height
        self.width = width
        # self.Z = math.sqrt(height*height+width*width)
        # print ('encoder: self.Z', self.Z)
        self.Z = 1

        self.center_map = np.zeros(shape=(height, width, 1), dtype=np.float32)
        self.kps_map = np.zeros(shape=(height, width, params['joints']*2), dtype=np.float32)
        self.kps_count = np.zeros(shape=(height, width, params['joints']*2), dtype=np.uint)
        # self.kps_map_weight = np.zeros(shape=(height, width, params['joints']*2), dtype=np.float32)
  
        ''' COCO: hierarchical SPR '''
        self.level = [[0, 1, 3],
                      [0, 2, 4],
                      [5, 7, 9],
                      [6, 8, 10],
                      [11, 13, 15],
                      [12, 14, 16]]

    def __call__(self):
        for i, center in enumerate(self.centers):
            sigma = self.sigmas[i]
            kps_sigma = self.kps_sigmas[i]
            kps = self.kps[i]
            if center[0] == 0 and center[1] == 0:
                continue
            # self.center_map[..., 0] = draw_gaussian(self.center_map[...,0], center, sigma, mask=None)
            self.center_map[..., 0] = draw_ttfnet_gaussian(self.center_map[...,0], center, sigma[0], sigma[1])
            self.body_joint_displacement(center, kps, kps_sigma)

        # print(np.where(self.kps_count > 2))
        self.kps_count[self.kps_count == 0] += 1
        self.kps_map = np.divide(self.kps_map, self.kps_count)
        return self.center_map, self.kps_map
        # return self.center_map, self.kps_map, self.kps_map_weight

    def body_joint_displacement(self, center, kps, sigma):
        for single_path in self.level:
            # print('encoder single path: ', single_path)
            for i, index in enumerate(single_path):
                if i == 0:
                    start_joint = center
                end_joint = kps[3*index:3*index+3]
                if start_joint[0] == 0 or start_joint[1] == 0:
                    continue
                if end_joint[0] == 0 or end_joint[1] == 0:
                    continue
                self.create_dense_displacement_map(index, start_joint, end_joint, sigma)
                start_joint = end_joint

    def create_dense_displacement_map(self, index, start_joint, end_joint, sigma=2):
        # print('start joint {} -> end joint {}'.format(start_joint, end_joint))
        center_x, center_y = int(start_joint[0]), int(start_joint[1])
        x0 = int(max(0, center_x - sigma))
        y0 = int(max(0, center_y - sigma))
        x1 = int(min(self.width, center_x + sigma))
        y1 = int(min(self.height, center_y + sigma))
        for x in range(x0, x1):
            for y in range(y0, y1):
                x_offset = (end_joint[0] - x) / self.Z
                y_offset = (end_joint[1] - y) / self.Z
                self.kps_map[y, x, 2*index] += x_offset
                self.kps_map[y, x, 2*index+1] += y_offset
                # self.kps_map_weight[y, x, 2*index:2*index+2] = 1.
                if end_joint[0] != x or end_joint[1] != y:
                    self.kps_count[y, x, 2*index:2*index+2] += 1

