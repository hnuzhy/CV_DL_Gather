import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.ndimage.filters import gaussian_filter

import cv2
import numpy as np
import math

import src.util as util
from src.config_reader import config_reader

param_, model_ = config_reader()
PKlist, limbSeq, mapIdx, colors = util.get_connection_info(param_['pk_mode'])
     
def make_inference(oriImg, model):
    
    # resize original image to do inference
    oriImg = cv2.resize(oriImg, (0,0), fx=param_['scale_ratio'], fy=param_['scale_ratio'], interpolation=cv2.INTER_CUBIC)
    height, width = oriImg.shape[:2]

    with torch.no_grad():
        imageToTest = T.transpose(T.transpose(T.unsqueeze(torch.from_numpy(oriImg).float(),0),2,3),1,2).cuda()
        
    multiplier = [x * model_['boxsize'] / height for x in param_['scale_search']]
    if param_['pk_mode'] == 'fullKP':
        heatmap_avg = torch.zeros((len(multiplier), 19, height, width)).cuda()
        paf_avg = torch.zeros((len(multiplier), 38, height, width)).cuda()
    if param_['pk_mode'] == 'PK12':
        heatmap_avg = torch.zeros((len(multiplier), len(PKlist), height, width)).cuda()  # 19 --> 12
        paf_avg = torch.zeros((len(multiplier), 2*len(limbSeq), height, width)).cuda()  # 38 --> 22
    
    for m in range(len(multiplier)):
        scale = multiplier[m]
        h ,w = int(height*scale), int(width*scale)
        pad_h = 0 if (h%model_['stride']==0) else model_['stride'] - (h % model_['stride']) 
        pad_w = 0 if (w%model_['stride']==0) else model_['stride'] - (w % model_['stride'])
        new_h, new_w = h+pad_h, w+pad_w

        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
        imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5
        
        feed = Variable(T.from_numpy(imageToTest_padded)).cuda()      
        output1, output2 = model(feed)
        heatmap = nn.UpsamplingBilinear2d((height, width)).cuda()(output2)
        paf = nn.UpsamplingBilinear2d((height, width)).cuda()(output1)      
        
        if param_['pk_mode'] == 'fullKP':
            heatmap_avg[m] = heatmap[0].data
            paf_avg[m] = paf[0].data
        if param_['pk_mode'] == 'PK12':
            heatmap_avg[m] = T.from_numpy( np.vstack(( (heatmap[0].data)[0:8], (heatmap[0].data)[14:18] )) )
            # paf_avg[m] = T.from_numpy( np.vstack(( (paf[0].data)[12:18], (paf[0].data)[20:26], (paf[0].data)[28:38] )) )
            paf_avg[m] = (paf[0].data)[12:38]
        
    heatmap_avg = T.transpose(T.transpose(T.squeeze(T.mean(heatmap_avg, 0)),0,1),1,2).cuda() 
    paf_avg = T.transpose(T.transpose(T.squeeze(T.mean(paf_avg, 0)),0,1),1,2).cuda() 
    heatmap_avg = heatmap_avg.cpu().numpy()
    paf_avg = paf_avg.cpu().numpy()
    
    # height is used for decode_keypoints() to serve for score_with_dist_prior()
    return heatmap_avg, paf_avg, height

def decode_keypoints(heatmap_avg, paf_avg, scaled_len):
    all_peaks = []
    peak_counter = 0
    
    for part in range(len(PKlist)):  # fullKP is 18; PK12 is 12
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)
        
        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]
        
        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(list(peaks)))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(list(peaks))
        
    connection_all = []
    special_k = []
    mid_num = param_['min_num']
    
    for k in range(len(mapIdx)):
        score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
        candA, candB = all_peaks[limbSeq[k][0]-1], all_peaks[limbSeq[k][1]-1]
        nA, nB = len(candA), len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + 0.00001) # avoid zero division error
                    vec = np.divide(vec, norm)
                    
                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num))
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*scaled_len/norm-1, 0)
                    # score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    #score_with_dist_prior = sum(score_midpts)/(len(score_midpts)+1) + min(0.5*oriImg.shape[0]/norm, 0)
                    criterion1 = len(np.nonzero(score_midpts > param_['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
 
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, len(PKlist)+2))  # fullKP is 20; PK12 is 14
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    # limb_num = len(limbSeq)-2 if param_['pk_mode'] == 'fullKP' else len(limbSeq)  # fullKP is 19-2=17; PK12 is 11
    limb_num = len(limbSeq)-2
    
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
                
                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    # print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                # if find no partA in the subset, create a new subset
                elif not found and k < limb_num:  # fullKP is 17; PK12 is 11
                    row = -1 * np.ones(len(PKlist)+2)  # fullKP is 20; PK12 is 14 
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < param_['mid_num'] or subset[i][-2]/subset[i][-1] < param_['thre3']:
        #if subset[i][-1] < 1 or subset[i][-2]/subset[i][-1] < 0.1:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    all_keypoints = process_candidate_subset(subset, candidate)
    return all_keypoints


def process_candidate_subset(subset, candidate):
    all_keypoints = []
    # limb_num = len(limbSeq)-2 if param_['pk_mode'] == 'fullKP' else len(limbSeq)  # fullKP is 19-2=17; PK12 is 11
    limb_num = len(limbSeq)-2
    for n in range(len(subset)):
        keyPointsDict = {}
        for i in range(limb_num):  # fullKP is 17; PK12 is 11
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            point1, point2 = np.array(limbSeq[i])-1
            if not keyPointsDict.has_key(point1):
                keyPointsDict[point1] = [Y[0]/param_['scale_ratio'], X[0]/param_['scale_ratio']]
            if not keyPointsDict.has_key(point2):
                keyPointsDict[point2] = [Y[1]/param_['scale_ratio'], X[1]/param_['scale_ratio']]			
        # fill non-occured keypoints with position (0,0)
        keypoints_arr = np.zeros(len(PKlist)*3)
        for index in PKlist:
            keypoints_arr[3*index] = index
            if keyPointsDict.has_key(index):
                x_value, y_value = keyPointsDict[index]
                keypoints_arr[3*index+1] = int(x_value)
                keypoints_arr[3*index+2] = int(y_value)
        all_keypoints.append(keypoints_arr)

    return all_keypoints