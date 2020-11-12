# UDP-Pose(CVPR2020)

Paper URL: https://arxiv.org/abs/1911.07524

Authors are all Chinese from Institute of Automation, Chinese Academy of Sciences. The paper full name is **The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation**, and UDP here means Unbiased Data Processing. The official code is released in [HuangJunJie2017/UDP-Pose](https://github.com/HuangJunJie2017/UDP-Pose). 

## 0) Abstract

<img src="./materials/Abstract.png" width = "500" alt="" align=center />

*Recently, the leading performance of human pose estimation is dominated by top-down methods. Being a fundamental component in training and inference, data processing has not been systematically considered in pose estimation community, to the best of our knowledge. In this paper, we focus on this problem and find that the devil of top-down pose estimator is in the biased data processing. Specifically, by investigating the standard data processing in state-of-the-art approaches mainly including data transformation and encoding-decoding, we find that the results obtained by common flipping strategy are unaligned with the original ones in inference. Moreover, there is statistical error in standard encoding-decoding during both training and inference. Two problems couple together and significantly degrade the pose estimation performance. Based on quantitative analyses, we then formulate a principled way to tackle this dilemma. Data is processed based on unit length instead of pixel, and an offset-based strategy is adopted to perform encoding-decoding. The Unbiased Data Processing (UDP) for human pose estimation can be achieved by combining the two together. UDP not only boosts the performance of existing methods by a large margin but also plays a important role in result reproducing and future exploration. As a model-agnostic approach, UDP promotes SimpleBaseline-ResNet-50-256x192 by 1.5 AP (70.2 to 71.7) and HRNet-W32-256x192 by 1.7 AP (73.5 to 75.2) on COCO test-dev set. The HRNet-W48-384x288 equipped with UDP achieves 76.5 AP and sets a new state-of-the-art for human pose estimation. The code will be released.*

## 1) Introduction

UDP-Pose is either a new backbone for pose estimation nor a novel independent algorithm. It focuses on *Unbiased Data Processing* before data training and also subsequent infenence. UDP-Pose works as a plug-in for SOTA MPPE methods including HRNet and SimpleBaseline to futher improve their mAP with less GFLOPS burden increase.

<img src="./materials/UDP_plug-in.png" width = "500" alt="" align=center />

The main contribution of UDP-Pose is to discover the defects of current pose estimation algorithms during data processing both in training and infenence, and then propose an unbiased data transformation pipeline. This method can be used as a model independent plug-in, combined with other methods, to enhance their mAP on the common datasets. Illustration for the processes between *standard biased data transformation* and proposed *unbiased data transformation* is below. In this paper, a lot of complicated mathematical formulas of error analysis are involved. Comprehension of my reading will be showed later.

<img src="./materials/UnbiasedDataProcessing.png" height = "400" alt="" align=center />


## 2) Intensive Reading


## 3) Citation
Please cite the paper in your publications if it helps your research:
```
@InProceedings{Huang_2020_CVPR,
author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
