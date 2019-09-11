# Pose Estimation

## 1) Introduction
The general definition of `Human Pose Estimation` is as follows: Given a picture containing human body, the position of human joints can be detected, and the joints can be connected in a predetermined order to form pose. If there are multiple people, the joints should not be confused. Therefore, the pose estimation can be divided into two categories, `single` and `multiple`.

![example1](./materials/single_person_pose_estimation-stacked_hourglass.jpg)
![example2](./materials/multi_person_pose_estimation-PAF_openpose.jpg)

In terms of implementation method, pose estimation has two branches: `Top-down` and `Bottom-up`. `Top-down` first uses the human detector to get bounding boxes of bodies in the image, and then estimates the pose of each person. `Bottom-up` directly predicts the position of all human joints in the image, and then use post-processing algorithm to link them into complete poses. The performance of the former method is mainly dominated by detectors and easy to slaughter on public datasets than the latter, but its inference time of single image increases linearly with the number of people, and it does not perform well in crowded, cluttered and occluded scenes. The latter method is easy to perform poorly because of the problem of joint point connection algorithm, but its detection time is relatively stable, and there will be no big error in the case of crowding.

![example3](./materials/method_comparing.jpg)

## 2) Datasets
At present, the mainstream schemes for pose estimation are all based on data-driven deep learning methods. And there are two popular public datasets, [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/)(CVPR2014)[1] and [COCO: Common Objects in Context](http://cocodataset.org/)(ECCV2014)[2]. Although there is a detailed description of datasets on the official website, here is a brief summary of the important information for a quick start.

**MPII Human Pose Dataset**

![example4](./materials/MPII_keypoints.jpg)

The dataset includes around *25K images* (precisely 24984) containing over *40K people* with annotated body joints. Overall the dataset covers *410 human activities* and each image is provided with an activity label. Each image was extracted from a YouTube video which is not very clear. And according to standard practice, authors withhold the test annotations to prevent overfitting and tuning on the test set. For external testing, an automatic evaluation server and performance analysis tools are provided.
'''joints_name_dict = {0: 'r ankle', 1: 'r knee', 2: 'r hip', 3: 'l hip', 4: 'l knee', 5: 'l ankle', 6: 'pelvis', 7: 'thorax', 8: 'upper neck', 9: 'head top', 10: 'r wrist', 11: 'r elbow', 12: 'r shoulder', 13: 'l shoulder', 14: 'l elbow', 15: 'l wrist'} # 16 points'''
'''limb_connection_list = [[0, 1], [1, 2], [2, 6], [3, 6], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [8, 12], [8, 13], [10, 11], [11, 12], [13, 14], [14, 15]] # 15 pairs'''

**COCO: Common Objects in Context**

![example5](./materials/COCO_keypoints.jpg)


## 3) SOTA Algorithms


## 4) References
[1] M. Andriluka, L. Pishchulin, P. Gehler, and B. Schiele. 2d human pose estimation: New benchmark and state of the art analysis. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2014.

[2] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740-755. Springer, 2014.
