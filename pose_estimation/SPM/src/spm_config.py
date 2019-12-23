#!/usr/bin/python3
# encoding: utf-8

spm_config = {}


'''
shape=128*128, batch_size=8 ==> P40 is ok, heatmap 128*128 is too small.
shape=384*384, batch_size=2 ==> P40 is ok, heatmap 384*384 is out of memory, stage1-3 has warning.
shape=256*256, batch_size=2 ==> P40 is ok, heatmap 256*256 is ok
'''

# SPM hyper parameters
spm_config['height'] = 256
spm_config['width'] = 256
spm_config['joint_weight'] = 1
spm_config['root_weight'] = 80.0
spm_config['sigma'] = 5.0


# Training parameters
spm_config['joints'] = 17
spm_config['batch_size'] = 2
spm_config['learning_rate'] = 1e-4
spm_config['total_epoch'] = 100

spm_config['train_json_file'] = '/data/zhouhuayi/data/COCO/json/person_keypoints_train2017.json'
spm_config['train_img_path'] = '/data/zhouhuayi/data/COCO/images/train2017'

spm_config['val_json_file'] = '/data/zhouhuayi/data/COCO/json/person_keypoints_val2017.json'
spm_config['val_img_path'] = '/data/zhouhuayi/data/COCO/images/val2017'

spm_config['finetune'] = None
spm_config['ckpt'] = '/data/zhouhuayi/pose_estimation/SPM/ckpt/'


# Test or decoding parameters
spm_config['nms_score'] = 0.5  # Choose location (x,y) in heatmap that root_score large than this
spm_config['nms_dist'] = 10.0  # From high score to low, choose all location (x,y) with distance lager than this mutually
# spm_config['scale_search'] = [0.75, 1, 1.25, 1.5, 2]
spm_config['scale_search'] = [1.5]


'''

COCO joints format:
1-‘nose’
2-‘left_eye’
3-‘right_eye’
4-‘left_ear’
5-‘right_ear’
6-‘left_shoulder’
7-‘right_shoulder’
8-‘left_elbow’
9-‘right_elbow’
10-‘left_wrist’
11-‘right_wrist’
12-‘left_hip’
13-‘right_hip’
14-‘left_knee’
15-‘right_knee’
16-‘left_ankle’
17-‘right_ankle’


This is the COCO format in OpenPose
0: 'nose',
1: 'neck', 
2: 'Rshoulder',
3: 'Relbow',
4: 'Rwrist',
5: 'Lshoulder',
6: 'Lelbow',
7: 'Lwrist',
8: 'Rhip',
9: 'Rknee',
10: 'Rankle',
11: 'Lhip',
12: 'Lknee',
13: 'Lankle',
14: 'Leye',
15: 'Reye',
16:'Lear',
17:'Rear'

val2017
total_anno_info 5000
total_anno_keypoints 10777
total_anno_person_images 2693

train2017
total_anno_info 118287
total_anno_keypoints 257252
total_anno_person_images 64115

'''
