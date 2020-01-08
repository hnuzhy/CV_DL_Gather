"""Predict poses for given images."""

# bug 2020-01-08: AttributeError: 'Shell' object has no attribute 'io_scales'
# We should install lib openpifpaf==0.9.0, the version must not be unmatching

import os
import cv2
import time

from tools import predict

if __name__ == '__main__':
    model_path = './models/resnet101_block5-pif-paf.pkl'
    # img_path = os.path.join(os.getcwd(), 'test_imgs')
    img_root_path = './test_imgs'
    # img_names = ['04_0397_Student.jpg', '04_0466_Student.jpg', '05_0014_Student.jpg', '07_0123.jpg',
        # '07_0265.jpg', '10_0047.jpg', '10_0261.jpg', '022_ch40_2655_0.8.jpg']
    img_names = ['04_0397_Student.jpg', '04_0466_Student.jpg', '022_ch40_2655_0.8.jpg']
    
    for index, img_name in enumerate(img_names):
        tic = time.time()
        imgs_path_list = []
        imgs_path_list.append(os.path.join(img_root_path, img_name))
        json_dict = predict.inference(model_path, imgs_path_list)
        print(index, img_name, ", json_dict-length: ", len(json_dict), ", Time: ", time.time()-tic)
    
    # imgs_path_list = []
    # for index, img_name in enumerate(img_names):
        # imgs_path_list.append(os.path.join(img_root_path, img_name))
    # tic = time.time()
    # json_dict_list = predict.inference(model_path, imgs_path_list)
    # print(index, img_name, ", json_dict_list-length: ", len(json_dict_list), ", Time: ", time.time()-tic)