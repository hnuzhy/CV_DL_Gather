import os
import cv2
import numpy as np

# Use command line below to install pycocotools in Windows (Must have Visual Studio Build Tools)
# pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
from pycocotools.coco import COCO

import h5py
from tqdm import tqdm

root_path = 'D:/dataset/public_private_datasets/COCO/'
ANNO_FILE = root_path + 'json/annotations_trainval2017/person_keypoints_train2017.json'
IMG_DIR = root_path + 'images/train2017'
H5_DATASET = './coco2017_personlab_train.h5'
# ANNO_FILE = root_path + 'json/annotations_trainval2017/person_keypoints_val2017.json'
# IMG_DIR = root_path + 'images/val2017'
# H5_DATASET = './coco2017_personlab_val.h5'

coco = COCO(ANNO_FILE)
img_ids = list(coco.imgs.keys())

data = h5py.File(H5_DATASET, 'w')
h5_root = data.create_group(name='coco')

total_anno_info = len(img_ids)
print("total_anno_info", total_anno_info)

total_anno_keypoints, total_anno_person_images = 0, 0
cnt = 0
for i, img_id in enumerate(tqdm(img_ids)):
    filepath = os.path.join(IMG_DIR, coco.imgs[img_id]['file_name'])
    # img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_COLOR) # error in OpenCV3
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    h, w, c = img.shape
    
    # crowd_mask is not in instance_masks, and also not annotated with keypoints.
    # unannotated_mask is in instance_masks, but not annotated with keypoints.
    crowd_mask = np.zeros((h, w), dtype='bool')
    unannotated_mask = np.zeros((h,w), dtype='bool')
    instance_masks = []  
    keypoints = []

    img_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    if len(img_anns) == 0:
        continue
    total_anno_person_images += 1
    for anno in img_anns:
        if anno['area']==0:
            continue
        mask = coco.annToMask(anno) # mask give value 1
        # print(img_id, "| mask max:", np.max(mask), "| mask mean:", np.sum(mask)/(mask>0).sum())

        # if crowd, don't compute loss
        if anno['iscrowd'] == 1:
            crowd_mask = np.logical_or(crowd_mask, mask)
        # if tiny instance, don't compute loss
        elif anno['num_keypoints'] == 0:
            unannotated_mask = np.logical_or(unannotated_mask, mask)
            instance_masks.append(mask)
            keypoints.append(anno['keypoints'])
            total_anno_keypoints += 1
        else:
            instance_masks.append(mask)
            keypoints.append(anno['keypoints'])
            total_anno_keypoints += 1

    # Construct encoding: mask index id (non-zero) will be encoding values for corresponding positions
    encoding = np.argmax(np.stack([np.zeros((h,w))]+instance_masks, axis=-1), axis=-1).astype('uint8')
    encoding = np.unpackbits(np.expand_dims(encoding, axis=-1), axis=-1)
    
    # No image has more than 63 instance annotations, so the first 2 channels are zeros
    encoding[:,:,0] = unannotated_mask.astype('uint8')
    encoding[:,:,1] = crowd_mask.astype('uint8')
    encoding = np.packbits(encoding, axis=-1)

    np_data = np.concatenate([img, encoding], axis=-1)
    this_data = h5_root.create_dataset(name=str(img_id), data=np_data, dtype='uint8')
    # keypoints = np.array(keypoints).astype(int)
    this_data.attrs['keypoints'] = keypoints
    # data.create_dataset(name=str(img_id), data=np_data, dtype='uint8')
    # data.attrs['keypoints'] = keypoints
    
    cnt += 1
    temp_instance = np.argmax(np.stack([np.zeros((h,w))]+instance_masks, axis=-1), axis=-1)
    temp_instance = np.expand_dims(temp_instance, axis=-1)
    temp_instance[temp_instance > 0] = 255  # should inlcude both annotated person and unannotated_mask
    print(cnt, img.shape, np_data.shape, encoding.shape, temp_instance.shape, keypoints)
    cv2.imwrite('./test_imgs/img_'+str(cnt)+'.jpg', np_data.astype(np.uint8))
    cv2.imwrite('./test_imgs/img_'+str(cnt)+'_mask.jpg', encoding.astype(np.uint8))
    cv2.imwrite('./test_imgs/img_'+str(cnt)+'_instance_mask.jpg', temp_instance.astype(np.uint8))
    if cnt==3: break

data.close()
print("total_anno_info", total_anno_info)
print("total_anno_keypoints", total_anno_keypoints)
print("total_anno_person_images", total_anno_person_images)

'''
dataset format: imgs, mask_encoding, keypoints

val2017
total_anno_info 5000
total_anno_keypoints 10777
total_anno_person_images 2693

train2017
total_anno_info 118287
total_anno_keypoints 257252
total_anno_person_images 64115
'''
