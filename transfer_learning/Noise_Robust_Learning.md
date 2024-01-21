# Noise Robust Learning
also named `Partial Label Learning` or `Sample Selection Learning`, `Label Noise Tolerance Learning` or `Label-Noise Learning (LNL)`

## Papers

### ▶ for Classification

* 👍**Co-Teaching(NIPS2018)** Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels [[paper link](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html)][[arxiv link](https://arxiv.org/abs/1804.06872)][[code|official](https://github.com/bhanML/Co-teaching)][`University of Technology Sydney`, first author [`Bo Han`](https://bhanml.github.io/)][It uses two networks and selects samples with loss value below a threshold in one network to train the other. In Co-Teaching, the threshold is chosen based on the knowledge of `noise rates`.]

* **Co-Teaching+(ICML2019)** How does Disagreement Help Generalization against Label Corruption? [[paper link](https://proceedings.mlr.press/v97/yu19b.html)][[arxiv link](https://arxiv.org/abs/1901.04215)][`University of Technology Sydney`, first author [`Xingrui Yu`](https://xingruiyu.github.io/)][It uses two networks and selects samples with loss value below a threshold in one network to train the other. The same threshold is used in Co-Teaching+ (as in Co-Teaching) but the sample selection is based on `disagreement` between the two networks.]

* **(NIPS2019)** Gradient based sample selection for online continual learning [[paper link](https://proceedings.neurips.cc/paper_files/paper/2019/hash/e562cd9c0768d5464b64cf61da7fc6bb-Abstract.html)][[arxiv link](https://arxiv.org/abs/1903.08671)][[code|official](https://github.com/rahafaljundi/Gradient-based-Sample-Selection)][`Rahaf Aljundi` and `Yoshua Bengio`, for the task `online continual learning`]

* **(NIPS2021)** Sample Selection for Fair and Robust Training [[paper link](https://proceedings.neurips.cc/paper/2021/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html)][[arxiv link](https://arxiv.org/abs/2110.14222)][[code|official](https://github.com/yuji-roh/fair-robust-selection)][`KAIST`, first author [`Yuji Roh`](https://www.yujiroh.com/)][Here, the `Fairness` aims to address the `biased distribution` of different samples in datasets, and the `Robustness` aims to train with using noisy labels which are `manually produced` by random flipping or adversarial attacks. This work mainly focuses on the former, rather than the latter.]

* **Survey(TNNLS2022)** Learning From Noisy Labels With Deep Neural Networks: A Survey [[paper link](https://ieeexplore.ieee.org/abstract/document/9729424)][[arxiv link](https://arxiv.org/abs/2007.08199)][[github link](https://github.com/songhwanjun/Awesome-Noisy-Labels)][`KAIST`, first author [`Hwanjun Song`](https://songhwanjun.github.io/)]

* 👍**BARE(WACV2023)** Adaptive Sample Selection for Robust Learning under Label Noise [[paper link](https://openaccess.thecvf.com/content/WACV2023/html/Patel_Adaptive_Sample_Selection_for_Robust_Learning_Under_Label_Noise_WACV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2106.15292)][[code|official](https://github.com/dbp1994/bare-wacv-2023)][`Indian Institute of Science`, This algorithm is motivated
by [`curriculum learning` (ICML2009, Yoshua Bengio)](https://dl.acm.org/doi/abs/10.1145/1553374.1553380) (also see [Self-Paced Learning for Latent Variable Models (NIPS2010)](https://proceedings.neurips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html) wherein easiness is decided upon based on how small the loss values are.) and can be thought of as a way to design an `adaptive curriculum`.][The core selection strategy is choosing a threshold in each mini-batch by calaulating `mean + 1*var` (super easy!!!)][tested on datasets `MNIST`, `CIFAR10` and `Clothing-1M`, where the Label Noise is manually made with a given `noisy rate`!!! (not realistic datasets with noisy labels!!!)(except the `Clothing-1M`)]

* 👍**RTME(TPAMI2023)** Regularly Truncated M-Estimators for Learning With Noisy Labels [[paper link](https://ieeexplore.ieee.org/document/10375792)][[arxiv link](https://arxiv.org/abs/2309.00894)][[code|official](https://github.com/xiaoboxia/RTM_LNL)][`University of Sydney`, first author [`Xiaobo Xia`](https://xiaoboxia.github.io/)][`label-noise-tolerant`]

* **NPN(arxiv2023.12)(AAAI2024)** Adaptive Integration of Partial Label Learning and Negative Learning for Enhanced Noisy Label Learning [[arxiv link](https://arxiv.org/abs/2312.09505)][[code|official](https://github.com/NUST-Machine-Intelligence-Laboratory/NPN)]

* **(arxiv2023.12)** Sample selection with noise rate estimation in noise learning of medical image analysis [[arxiv link](https://arxiv.org/abs/2312.15233)][`Medical Image Analysis` (mainly focusing on `classification`)]

* **ERASE(arxiv2023.12)** ERASE: Error-Resilient Representation Learning on Graphs for Label Noise Tolerance [[arxiv link](https://arxiv.org/abs/2312.08852)][[project link](https://eraseai.github.io/ERASE-page/)][[code|official](https://github.com/eraseai/erase)][`Tsinghua University, Shenzhen`, first author [`Ling-Hao CHEN`](https://lhchen.top/)][for tackling `graph-related` tasks; Essentially, ERASE is a semi-supervised framwork]

* 👍**NMTune(arxiv2023.09)(ICLR2024)** Understanding and Mitigating the Label Noise in Pre-training on Downstream Tasks [[openreview link](https://openreview.net/forum?id=TjhUtloBZU)][[arxiv link](https://arxiv.org/abs/2309.17002)][`Carnegie Mellon University + Microsoft Research Asia`]

* **NRAT(MachineLearning2024)** Nrat: towards adversarial training with inherent label noise [[paper link](https://link.springer.com/article/10.1007/s10994-023-06437-3)][[code|official](https://github.com/TrustAI/NRAT)][`Liverpool University`, proposed by the [`TrustAI`](https://github.com/TrustAI)]

* **LNL-flywheel(arxiv2024.01)** Learning with Noisy Labels: Interconnection of Two Expectation-Maximizations [[arxiv link](https://arxiv.org/abs/2401.04390)][`Samsung Advanced Institute of Technology + Seoul National Univ`][with a given `noisy rate`]

* 👍**BadLabels(TPAMI2024)** BadLabel: A Robust Perspective on Evaluating and Enhancing Label-Noise Learning [[paper link](https://ieeexplore.ieee.org/document/10404058)][[arxiv link](https://arxiv.org/abs/2305.18377)][[code|official](https://github.com/zjfheart/BadLabels)][`University of Auckland`, the first author [`Jingfeng Zhang (张景锋)`](https://zjfheart.github.io/), `Label-Noise Learning (LNL)`][It introduced a novel label noise type called `BadLabel`. BadLabel is crafted based on the `label-flipping attack` against standard classification.]



### ▶ for Detection

* 👍**MonoLSS(arxiv2023.12)** MonoLSS: Learnable Sample Selection For Monocular 3D Detection [[arxiv link](https://arxiv.org/abs/2312.14474)][`Baidu Inc.`, tested in datasets `Waymo` and `KITTI-nuScenes`][2D detector is based on `CenterNet` + `DLANet-34`][The proposed LSS module is largely based on [Gumbel-Softmax (ICLR2017)](https://openreview.net/forum?id=rkE3y85ee) and [top-k Gumbel-Softmax (ICML2019)](https://proceedings.mlr.press/v97/kool19a.html), which are useful and common theories.]

