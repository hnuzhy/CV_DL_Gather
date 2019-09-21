# AssociativeEmbedding(NIPS2017)

## 1) Introduction

`Associative embedding: End-to-end learning for joint detection and grouping`. Firstly, this paper is the most simple and direct bottom-up method that I’ve ever read. Both writing and methodological design follow a logical train of thought. It keeps the previous network structure design of *stacked hourglass* network for single person pose estimation, and extends it to multi-person joints heatmaps detection. *Associative Embedding* in title is an exclusive novel "tag" for different people to identify detected keypoints group assignment. Briefly speaking, this two-stage pipelines perform detection first and grouping second.

In fact, the authors of this article are also the authors of the previous *stacked hourglass* method. They belong to an advanced group [Vision & Learning Lab @ Princeton University](https://pvl.cs.princeton.edu/). Thank them for their excellent works.


## 2) Experiment
