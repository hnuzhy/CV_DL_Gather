# SPM(ICCV2019)

## 1) Introduction

Paper Name: [`Single-Stage Multi-Person Pose Machines`](https://arxiv.org/abs/1908.09220). As introduced in the root README file, this paper gives the first single stage multi-person pose estimation method. The abbreviation SPM comes from its name, and there are main two kinds of pipeline: **Structured Pose Representation (SPR)** and **Hierarchical SPR**.

Different from the traditional two-stage multi-person pose estimation methods (one stage for proposal generation and the other for allocating poses to corresponding persons), SPM can simplify the pipeline and improve the efficiency. Specifically, SPM has proposed a novel architecture **Structured Pose Representation (SPR)** which can unify person instance and their keypoints position together. SPM is designed to predict the **Structured Pose** of every person in one image in a single stage directly. So the inference time is almost the same with the backbone deep model goes forward once. This is surely faster and more elegant than both Top-Down and Bottom-Up algorithms.

SPM is to some extent inspired by the one-stage anchor-free object detection algorithms appeared recent months, such as FOCS, CornerNet, CenterNet. In particular, SPR defines the root joints to indicate all persons and their corresponding body joints are encoded into offsets or displacements *w.r.t* the root joints. Considering that some keypoints are far away from the root joint, the author thought *hierarchical representations* which is called **Hierarchical SPR**. In this way, the CNN model can predict keypoints in the outer ring from the root joint layer by layer. Note that root joints and displacements are outputed by model simultaneously. The idea in the paper can easily be adjusted from 2D to 3D pose estimation.

## 2) Method Description

Below is the diagram example of SPM. (b) gives the conventional pose representation which should predict all joints of every person in the input image. (c) shows the results of SPR. (d) is Hierarchical SPR. This keypoints annotation format comes from MPII.

![example1](./imgs/SPM_example_diagram.jpg)

[comment]: <> (MathJax Plugin for Github in Google Chrome)
Suppose that we have known how the Top-Down and Bottom-Up method work. We just use simple equations to explain them briefly.

*Conventional Pose Representation:* $\mathcal {P} = \{P^1_i, P^2_i, ..., P^K_i\}^N_{i=1}$, where $N$ is the number of persons in image $I$, $K$ is joint catagories, and $P^i_j$ is the coordinates of $j$th joint of person $i$. Note that $P^i_j$ can be either 2D $(x^i_j, y^i_j)$ or 3D $(x^i_j, y^i_j, z^i_j)$. To obtain $\mathcal {P}$, there are mainly two two-stage methods: Top-Down and Bottom-Up.

*Top-Down:* It first uses a person detector $f$ to localize person instances $\mathcal {B}$ (usually a set of bounding boxes) with function $\{f:I \rightarrow \mathcal {B}\}$, and then conducts single-person pose estimation with model $g$ to get corresponding keypoints of every person respectively. $\{g:\mathcal {B},I \rightarrow \mathcal {P}\}$.

*Bottom-Up:* This method first utilizes a keypoints extimator $g'$ to localize all body joints $\mathcal {J}$ in one image with function $\{g':I \rightarrow \mathcal {J,C}\}$. $\mathcal {C}$ indicates additional auxiliary vectors for assigning joint candidates to person instances. Then it performs joint allocation by sloving a graph partition problem (greedy matching in CPU) using strategy $f'$ which is formulated as $\{f':\mathcal {J,C} \rightarrow \mathcal {P}\}$.



## 3) Experiment
