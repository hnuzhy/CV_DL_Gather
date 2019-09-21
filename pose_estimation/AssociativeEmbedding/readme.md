# AssociativeEmbedding(NIPS2017)

## 1) Introduction

`Associative embedding: End-to-end learning for joint detection and grouping`. Firstly, this paper is the most simple and direct bottom-up method that I’ve ever read. Both writing and methodological design follow a logical train of thought. 

It keeps the previous network structure design of *stacked hourglass* network for single person pose estimation, and extends it to multi-person joints heatmaps detection. *Associative Embedding* in title is an exclusive novel "tag" for different people to identify detected keypoints group assignment. Briefly speaking, although this is an approach performing detection first and grouping second, it actually has only one stage using a generic network that includes no special complicated design for grouping.

In fact, the authors of this article are also the authors of the previous *stacked hourglass* method. They belong to an advanced group [Vision & Learning Lab @ Princeton University](https://pvl.cs.princeton.edu/). Thank them for their excellent works. Interestingly, the article concludes by saying that the main bottleneck of the system is keypoints detection rather than learning high quality grouping. Is it because they have to use their group's *stacked hourglass*? In other words, could we switch to other networks (ResNet, InceptionV3 or MobileNet) to learn heatmaps detection?


## 2) Experiment
