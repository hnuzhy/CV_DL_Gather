# PersonLab

## 1) Introduction

This Bottom-up approach is an ECCV2018 paper named `Personlab: Person pose estimation and instance segmentation with a bottomup, part-based, geometric embedding model`. PersonLab is simple and hence fast, it outperforms some SOTAs in both keypoint localization and instance segmentation task on COCO. In particular, the pipeline of this algorithm includes six main stages (not in order):

1. Predict all keypoints of all persons in the image using fully convolutional network;
2. Also predict the relative displacement between each pair of keypoints with a novel recurrent scheme to improve long-range predictions;
3. Use a greedy decoding process to group keypoints into instances;
4. Predict instance segmentation masks for each person;
5. Also predict offset vectors to each of the K keypoints of the corresponding person instance for every person pixel;
6. Use an efficient association algorithm to do instance segmentation decoding.

Note: For working better in clutter, greedy decoding starts from the most confident detection keypoint instead of a definite landmark liking nose. Below is an overview of PersonLab method.

![example1](./materials/network_architecture.jpg)

The PersonLab system consists of a CNN model that predicts: (1) keypoint heatmaps, (2) short-range offsets, (3) mid-range pairwise offsets, (4) person segmentation maps, and (5) long-range offsets. The first three predictions are used by the _Pose Estimation Module_ in order to detect human poses while the latter two, along with the human pose detections, are used by the _Instance Segmentation Module_ in order to predict person instance segmentation masks.


## 2) Experiment
