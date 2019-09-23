import os
import cv2
import torch
import time

from src.paf_model_pytorch import get_paf_model_dict
from src.decode_heatmaps_pafs import make_inference, decode_keypoints
from src.plot_pose import plot_pose

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] ="0"

if __name__ == "__main__":

    weight_name = './model/paf_pose_model_wieghts.pth'
    # weight_name = '../pytorch_Realtime_MPPE/model/pose_model.pth'
    folderList = ['./test_imgs/classroom_test']
    # folderList = ['../pytorch_Realtime_MPPE/sample_image/random_test']
    
    # load model network and weights
    print('Loading model and weights ...')
    tic = time.time()
    torch.set_num_threads(torch.get_num_threads())        
    model = get_paf_model_dict()    
    model.load_state_dict(torch.load(weight_name))
    model.cuda()
    toc = time.time()
    print('Load model and weights time is %.3f'%(toc-tic))
    
    # process images under folders in folderList
    for folderNum in range(len(folderList)):
        oriImagePath = folderList[folderNum] + '/'
        resultSavePath = folderList[folderNum] + '_result/'
        if not os.path.exists(resultSavePath):
            os.mkdir(resultSavePath)

        imageFiles = os.listdir(oriImagePath)
        for index, imageName in enumerate(imageFiles):
            test_image = oriImagePath + imageName
            tic = time.time()
            oriImg = cv2.imread(test_image) # B,G,R order
            canvas = oriImg.copy()
            
            heatmap_avg, paf_avg, scaled_len = make_inference(oriImg, model)
            all_keypoints = decode_keypoints(heatmap_avg, paf_avg, scaled_len)
            plot_pose(all_keypoints, canvas, imageName, resultSavePath, save_joints_info=True)
            toc = time.time()
            print('%s : Inference %s time is %.3f'%(str(index).zfill(4), test_image, toc-tic))
