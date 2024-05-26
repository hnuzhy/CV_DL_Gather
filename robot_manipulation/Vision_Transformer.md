# Vision Transformer


## Materials

## Papers

### Theories and Backbones

* **(ICCV2019)** Local Relation Networks for Image Recognition [[arxiv link](https://arxiv.org/abs/1904.11491)][[code|official](https://github.com/microsoft/Swin-Transformer/tree/LR-Net)][`Microsoft`, `the first full-attention visual backbone`]

* **LeViT (ICCV2021)** LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.html)][[code|offficial](https://github.com/facebookresearch/LeViT)][`facebookresearch`]

* 👍 **Swin-Transformer (Shifted Window)(ICCV2021)** Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [[arxiv link](https://arxiv.org/abs/2103.14030)][[code|offficial](https://github.com/microsoft/swin-transformer)][[Swin Transformers inference implemented in FasterTransformer by Nvidia](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/swin_guide.md)][`Microsoft`]

* **Swin-Transformer-V2 (CVPR2022)** Swin Transformer V2: Scaling Up Capacity and Resolution [[arxiv link](https://arxiv.org/abs/2111.09883)][[code|offficial](https://github.com/microsoft/swin-transformer)][`Microsoft`]

* **SimMIM (CVPR2022)** SimMIM: A Simple Framework for Masked Image Modeling [[arxiv link](https://arxiv.org/abs/2111.09886)][[code|offficial](https://github.com/microsoft/SimMIM)][`Microsoft`, `a self-supervised approach that enables SwinV2-G`]

* **EfficientViT (CVPR2023)** EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention [[arxiv link](https://arxiv.org/abs/2305.07027)][[code|official](https://github.com/microsoft/Cream/tree/main/EfficientViT)][`Microsoft`]


### Self-Supervsied Learning

* 👍**SimCLR (ICML2020)** A Simple Framework for Contrastive Learning of Visual Representations [[paper link](http://proceedings.mlr.press/v119/chen20j.html)][[paperswithcode link](https://paperswithcode.com/paper/a-simple-framework-for-contrastive-learning)][[code|official](https://github.com/google-research/simclr)][[official blog](https://blog.research.google/2020/04/advancing-self-supervised-and-semi.html)][`Geoffrey Hinton`, `Contrastive Learning`, `Pre-training`, `Self-Supervised Learning`]

* 👍**MoCo (CVPR2020)** Momentum Contrast for Unsupervised Visual Representation Learning [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html)][[arxiv link](http://arxiv.org/abs/1911.05722)][[code|official](https://github.com/facebookresearch/moco)][`Kaiming He + Ross Girshick`, `Contrastive Learning`, `Pre-training`, `Self-Supervised Learning`]

* 👍**BYOL (NIPS2020)** Bootstrap your own latent: A new approach to self-supervised Learning [[paper link](https://papers.nips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)][[arxiv link](https://arxiv.org/abs/2006.07733)][[code|official](https://github.com/deepmind/deepmind-research/tree/master/byol)][`Contrastive Learning`, `Pre-training`, `Self-Supervised Learning`]

* 👍**SwAV (NIPS2020)** Unsupervised Learning of Visual Features by Contrasting Cluster Assignments [[paper link](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)][[arxiv link](https://arxiv.org/abs/2006.09882)]
[[code|official](https://github.com/facebookresearch/swav)][including `contrastive learning`]

* **CARE(NIPS2021)** Revitalizing CNN Attention via Transformers in Self-Supervised Visual Representation Learning [[paper link](https://proceedings.neurips.cc/paper_files/paper/2021/hash/21be992eb8016e541a15953eee90760e-Abstract.html)][[openreview link](https://openreview.net/forum?id=sRojdWhXJx)][[arxiv link](https://arxiv.org/abs/2110.05340)][[`Chongjian GE 葛崇剑`](https://chongjiange.github.io/)][In order to make the training process of `Mean-Teacher` more stable, it slowly increases α from 0.999 to 1 through `cosine` design.]

* 👍👍**DINO (ICCV2021)** Emerging Properties in Self-Supervised Vision Transformers [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)][`ViT-based`, `a form of self-distillation with no labels`, `self-supervised pre-training`]

* **MoCo-v3(ICCV2021)** An Empirical Study of Training Self-Supervised Vision Transformers [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_An_Empirical_Study_of_Training_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)][`ViT-based`, `self-supervised pre-training`]

* 👍**SimSiam (CVPR2021)** Exploring Simple Siamese Representation Learning [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2011.10566)][[code|official](https://github.com/facebookresearch/simsiam)][`Kaiming He`, `Contrastive Learning`, `Pre-training`, `Self-Supervised Learning`]

* 👍**MAE (CVPR2022)** Masked Autoencoders Are Scalable Vision Learners [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)][`ViT-based`, `FAIR`, `He Kaiming`， `It reconstructs the original signal given its partial observation`, `self-supervised pre-training`]


### Lightweight and Efficient Training 

* **EfficientTrain++(TPAMI2024)(arxiv2024.05)** EfficientTrain++: Generalized Curriculum Learning for Efficient Visual Backbone Training [[paper link](https://ieeexplore.ieee.org/abstract/document/10530470/)][[arxiv link](https://arxiv.org/pdf/2405.08768)][[weixin blog](https://mp.weixin.qq.com/s/FJj0F2NcW9ftmT_lbO1R3w)][[code|official](https://github.com/LeapLabTHU/EfficientTrain)][`THU + BAAI`, used the `generalized curriculum learning`][The conference (EfficientTrain, ICCV2023) version [EfficientTrain: Exploring Generalized Curriculum Learning for Training Visual Backbones](https://arxiv.org/abs/2211.09703)]
