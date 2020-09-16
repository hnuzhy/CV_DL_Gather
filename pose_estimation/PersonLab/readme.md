# PersonLab(ECCV2018) 
Paper URL: https://arxiv.org/abs/1803.08225

For codes, go straight to [**3) Implementation**](https://github.com/hunzhy/Pose_Estimation_Depository/tree/master/PersonLab#3-implementation) in my another project.

**Note**: `PersonLab` is the work from Google with the same author with the top-down method `G-RMI`(CVPR2017). So we will find their intuitions about pose estimation are similar.

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

## 2) Impression & Understanding

PersonLab does both pose estimation and instance segmentation task in one system. We only focus on the effect of the former branch pose estimation. Here are some topics what I think are important after reading the paper.

1. **Hough Voting**: Aggregate the detected heatmap and short-range offsets into 2D Hough score maps.
2. **Bilinear Interpolation Kernel**: Compute Hough score using short-range offsets array.
3. **Recurrent Offset Refinement**: Refine the mid-range pairwise offsets using the more accurate short-range offsets. *pairwise displacements = mid-range pairwise offsets + short-range offsets.*
4. **Fast Greedy Decoding**: Group keypoints into detected person instances.
5. **Non-Maximum Suppression**: Use a NMS radius of *r = 10* pixels to reject redundant candidate points.
6. **Object Keypoint Similarity**: Evaluation metric OKS is used in the COCO keypoints task and penalizes localization errors for different keypoint types with different accuracy thresholds.

Focus on these points will help to facilitate the subsequent separation of pose estimation part.

### 3) Citation

```
@article{Papandreou2018PersonLab,
  title={PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model},
  author={Papandreou, George and Zhu, Tyler and Chen, Liang Chieh and Gidaris, Spyros and Tompson, Jonathan and Murphy, Kevin},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2018},
}
```

