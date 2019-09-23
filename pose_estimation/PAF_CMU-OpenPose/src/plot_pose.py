import os
import math
import cv2
import numpy as np
import src.util as util
from src.config_reader import config_reader

# import matplotlib
# import pylab as plt

param_, model_ = config_reader()
PKlist, limbSeq, mapIdx, colors = util.get_connection_info(param_['pk_mode'])

def plot_pose(poses, canvas, img_name, save_path, save_joints_info=False):
    stickwidth = int(4 * param_['scale_ratio'])
    
    imageSavePath = os.path.join(save_path, img_name[:-4]+'_pose.jpg')
    if save_joints_info:
        txtNamePath = os.path.join(save_path, img_name[:-4]+'_pose.txt')
        txtWrite = open(txtNamePath , 'w')
     
    # limb_num = len(limbSeq)-2 if param_['pk_mode'] == 'fullKP' else len(limbSeq)  # fullKP is 19-2=17; PK12 is 11    
    limb_num = len(limbSeq)-2   
    for pose in poses:
        for i in range(limb_num):
            point1, point2 = np.array(limbSeq[i])-1
            x1, y1 = pose[point1*3+1], pose[point1*3+2]
            x2, y2 = pose[point2*3+1], pose[point2*3+2]
            if (x1==0 and y1==0) or (x2==0 and y2==0):
                continue
            cur_canvas = canvas.copy()
            # cv2.circle(canvas, (int(x1), int(y1)), 4, colors[0], thickness=-1)
            cv2.circle(canvas, (int(x2), int(y2)), 4, colors[0], thickness=-1)
            
            mX, mY = int((x1+x2)/2), int((y1+y2)/2)
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y1-y2, x1-x2))
            polygon = cv2.ellipse2Poly((mX, mY), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
        if save_joints_info:
            for index in PKlist:
                txtWrite.writelines('{:d},{:s},{:s},'.format(index, str(pose[index*3+1]), str(pose[index*3+2])))
            txtWrite.writelines('\n')
    if save_joints_info:
        txtWrite.close()
        
    cv2.imwrite(imageSavePath, canvas)