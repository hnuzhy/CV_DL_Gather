from keras.models import load_model
from src.config import config
from src.model import get_personlab
from utils.resnet101 import get_resnet101_base
from scipy.ndimage.filters import gaussian_filter
import cv2
import numpy as np
from time import time
import os

from matplotlib import pyplot as plt
from utils.plot import apply_mask, plot_poses, plot_instance_masks, \
        visualize_short_offsets, visualize_mid_offsets, visualize_long_offsets, \
        pad_img, overlay
from utils.post_proc import compute_heatmaps, compute_heatmaps_v2,\
        get_keypoints, group_skeletons, get_instance_masks

os.environ["CUDA_VISIBLE_DEVICES"] ="0"


# Load model with Train=False.
# This model was trained with intermediate supervision from the layer specified,
# so we'll load the whole thing including those extra prediction layers.
tic = time()
model = get_personlab(train=False, with_preprocess_lambda=True,
                      intermediate_supervision=config.INTER_SUPERVISION,
                      intermediate_layer='res4b12_relu',
                      build_base_func=get_resnet101_base, output_stride=16)
model.load_weights(config.TRAIN_MODEL_PATH)
print('Loading Network and Model Weight time: {}'.format(time()-tic))

for one_image in config.TEST_IMAGES:
    img = cv2.imread(one_image)
    img = cv2.resize(img, (0,0), fx=config.TEST_SCALE, fy=config.TEST_SCALE, interpolation=cv2.INTER_CUBIC)
    img = pad_img(img)
    print('Image shape: {}'.format(img.shape))

    tic = time()
    outputs = model.predict(img[np.newaxis,...])
    print('Prediction one image takes time: {}'.format(time()-tic))
    tic = time()
    # Remove batch axes and remove intermediate predictions
    # outputs = [o[0] for o in outputs][5:]
    outputs = [o[0] for o in outputs]
    # H = compute_heatmaps_v2(kp_maps=outputs[0], short_offsets=outputs[1])
    H = compute_heatmaps(kp_maps=outputs[0], short_offsets=outputs[1])
    print('Heatmaps takes time: {}'.format(time()-tic))

    tic1 = time()
    pred_kp = get_keypoints(H)
    print('Keypoints takes time: {}'.format(time()-tic1))
    
    tic2 = time()
    pred_skels = group_skeletons(keypoints=pred_kp, mid_offsets=outputs[2])
    print('Group Skeletons takes time: {}'.format(time()-tic2))
    print('Group Joints of one image takes time: {}'.format(time()-tic))
    img = img[:,:,[2,1,0]]  # BGR to RGB

    # continue

    # plt.figure('Original Image')
    # plt.imshow(img)

    # Here is the output map for right shoulder
    Rshoulder_map = outputs[0][:,:,config.KEYPOINTS.index('Rshoulder')]
    plt.figure('Output Map: Rshoulder')
    plt.imshow(overlay(img, Rshoulder_map, alpha=0.7))

    plt.figure('Heatmaps: Rshoulder')
    plt.imshow(H[:,:,config.KEYPOINTS.index('Rshoulder')])

    # The heatmaps are computed using the short offsets predicted by the network
    # Here are the right shoulder offsets
    visualize_short_offsets(offsets=outputs[1], heatmaps=H, keypoint_id='Rshoulder', img=img, every=8)

    # The connections between keypoints are computed via the mid-range offsets.
    # We can visuzalize them as well; for example right shoulder -> right hip
    visualize_mid_offsets(offsets= outputs[2], heatmaps=H, from_kp='Rshoulder', to_kp='Rhip', img=img, every=8)

    # And we can see the reverse connection (Rhip -> Rshjoulder) as well
    visualize_mid_offsets(offsets= outputs[2], heatmaps=H, to_kp='Rshoulder', from_kp='Rhip', img=img, every=8)


    joints_num_list = [(skel[:,2]>0).sum() for skel in pred_skels]
    print('Number of detected skeletons: {}, joints_num_list: {}'.format(len(pred_skels), joints_num_list))
    pred_skels = [skel for skel in pred_skels if (skel[:,2]>0).sum() > config.JOINTS_NUM_THRE]
    print('Number of detected skeletons (above threshold number): {}'.format(len(pred_skels)))
    plot_poses(img, pred_skels)
    print("Pose Estimation is Done.")

    if config.MODE is 1:
        # Finally, we can use the predicted skeletons along with the long-range offsets and binary segmentation mask
        # to compute the instance masks. First, let's look at the binary mask predicted by the net.
        plt.figure('Binary Mask')
        plt.imshow(apply_mask(img, outputs[4][:,:,0]>0.5, color=[1,0,0]))
        
        # We can visuzalize long offsets, here are the right shoulder offsets
        visualize_long_offsets(offsets=outputs[3], keypoint_id='Rshoulder', seg_mask=outputs[4], img=img, every=8)
        
        instance_masks = get_instance_masks(pred_skels, outputs[-1][:,:,0], outputs[-2])
        plot_instance_masks(instance_masks, img)
        
        print("Instance Segmentation is Done.")

    print(one_image, "All is Done.")