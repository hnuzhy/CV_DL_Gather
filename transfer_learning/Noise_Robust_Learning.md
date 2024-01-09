# Noise Robust Learning
also named `Sample Selection Learning`

## Papers

### ▶ for Classification

* **(NIPS2019)** Gradient based sample selection for online continual learning [[paper link](https://proceedings.neurips.cc/paper_files/paper/2019/hash/e562cd9c0768d5464b64cf61da7fc6bb-Abstract.html)][[arxiv link](https://arxiv.org/abs/1903.08671)][`Rahaf Aljundi` and `Yoshua Bengio`]

* **(NIPS2021)** Sample Selection for Fair and Robust Training [[paper link](https://proceedings.neurips.cc/paper/2021/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html)][[arxiv link](https://arxiv.org/abs/2110.14222)][[code|official](https://github.com/yuji-roh/fair-robust-selection)][`KAIST`, first author [`Yuji Roh`](https://www.yujiroh.com/), ]

* **Survey(TNNLS2022)** Learning From Noisy Labels With Deep Neural Networks: A Survey [[paper link](https://ieeexplore.ieee.org/abstract/document/9729424)][[arxiv link](https://arxiv.org/abs/2007.08199)][[github link](https://github.com/songhwanjun/Awesome-Noisy-Labels)][`KAIST`, first author [`Hwanjun Song`](https://songhwanjun.github.io/)]

* **BARE(WACV2023)** Adaptive Sample Selection for Robust Learning under Label Noise [[paper link](https://openaccess.thecvf.com/content/WACV2023/html/Patel_Adaptive_Sample_Selection_for_Robust_Learning_Under_Label_Noise_WACV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2106.15292)][[code|official](https://github.com/dbp1994/bare-wacv-2023)][`Indian Institute of Science`, This algorithm is motivated
by [`curriculum learning` (ICML2009, Yoshua Bengio)](https://dl.acm.org/doi/abs/10.1145/1553374.1553380) (also see [Self-Paced Learning for Latent Variable Models (NIPS2010)](https://proceedings.neurips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html) wherein easiness is decided upon based on how small the loss values are.) and can be thought of as a way to design an `adaptive curriculum`.][The core selection strategy is choosing a threshold in each mini-batch by calaulating `mean + 1*var` (super easy!!!)][tested on datasets `MNIST`, `CIFAR10` and `Clothing-1M`, where the Label Noise is manually made with a given `noisy rate`!!! (not realistic datasets with noisy labels!!!)(except the `Clothing-1M`)]

* **(arxiv2023.12)** Sample selection with noise rate estimation in noise learning of medical image analysis [[arxiv link](https://arxiv.org/abs/2312.15233)][`Medical Image Analysis` (mainly focusing on `classification`)]


### ▶ for Detection

* 👍**MonoLSS(arxiv2023.12)** MonoLSS: Learnable Sample Selection For Monocular 3D Detection [[arxiv link](https://arxiv.org/abs/2312.14474)][`Baidu Inc.`, tested in datasets `Waymo` and `KITTI-nuScenes`][2D detector is based on `CenterNet` + `DLANet-34`][The proposed LSS module is largely based on [Gumbel-Softmax (ICLR2017)](https://openreview.net/forum?id=rkE3y85ee) and [top-k Gumbel-Softmax (ICML2019)](https://proceedings.mlr.press/v97/kool19a.html), which are useful and common theories.]

