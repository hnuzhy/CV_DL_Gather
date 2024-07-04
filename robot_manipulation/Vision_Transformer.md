# Vision Transformer

## Materials

## Papers

### Theories and Backbones

* **(ICCV2019)** Local Relation Networks for Image Recognition [[paper link](http://openaccess.thecvf.com/content_ICCV_2019/html/Hu_Local_Relation_Networks_for_Image_Recognition_ICCV_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1904.11491)][[code|official](https://github.com/microsoft/Swin-Transformer/tree/LR-Net)][`Microsoft + THU`, `the first full-attention visual backbone`]

* 👍**ViT(ICLR2021 Oral)(arxiv2020.10)** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [[paper link](https://openreview.net/forum?id=YicbFdNTTy)][[arxiv link](https://arxiv.org/abs/2010.11929)][[code|official](https://github.com/google-research/vision_transformer )]

* **LeViT (ICCV2021)** LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.html)][[code|official](https://github.com/facebookresearch/LeViT)][`facebookresearch`]

* 👍**Swin-Transformer (Shifted Window)(ICCV2021)** Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper)][[arxiv link](https://arxiv.org/abs/2103.14030)][[code|official](https://github.com/microsoft/swin-transformer)][[Swin Transformers inference implemented in FasterTransformer by Nvidia](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/swin_guide.md)][`Microsoft`]

* **Swin-Transformer-V2 (CVPR2022)** Swin Transformer V2: Scaling Up Capacity and Resolution [[paper link](http://openaccess.thecvf.com/content/CVPR2022/html/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2111.09883)][[code|official](https://github.com/microsoft/swin-transformer)][`Microsoft`]

* **TinyViT(ECCV2022)(arxiv2022.07)** TinyViT: Fast Pretraining Distillation for Small Vision Transformers [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19803-8_5)][[arxiv link](https://arxiv.org/abs/2207.10666)][[code|official](https://github.com/microsoft/Cream/tree/main/TinyViT)][`Microsoft Research + Microsoft Cloud+AI`]

* **EfficientViT (CVPR2023)(arxiv2023.05)** EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2305.07027)][[code|official](https://github.com/microsoft/Cream/tree/main/EfficientViT)][`CUHK + Microsoft`]

* **RoFormer / RoPE (Neurocomputing2024)(arxiv2021.04)(with very high influence)** RoFormer: Enhanced transformer with Rotary Position Embedding [[paper link](https://www.sciencedirect.com/science/article/abs/pii/S0925231223011864)][[arxiv link](https://arxiv.org/abs/2104.09864)][[code|official](https://huggingface.co/docs/transformers/model_doc/roformer)][`Zhuiyi Technology`; It is widely used in modern `transformer` designs, for example using `RoPE` to get the reformed [`RoPE-ViT`](https://github.com/naver-ai/rope-vit) by the work [`(arxiv2024.03) Rotary Position Embedding for Vision Transformer`](https://arxiv.org/abs/2403.13298)]

* 👍**StarNet(CVPR2024)(arxiv2024.03)** Rewrite the Stars [[arxiv link](https://arxiv.org/abs/2403.19967)][[weixin blog](https://mp.weixin.qq.com/s/SemsRFsrGQ0WJf_yQN6p4A)][[code|official](https://github.com/ma-xu/Rewrite-the-Stars)][`microsoft`; superior than transformer-based conunterparts `FasterViT`, `EdgeViT`, and `Mobile-Former`]

* **CoPE(arxiv2024.05)** Contextual Position Encoding: Learning to Count What's Important [[arxiv link](https://arxiv.org/abs/2405.18719)][`FAIR at Meta`; It aims to enhance the performance of `RoPE`]

* 👍**YOCO(arxiv2024.05)** You Only Cache Once: Decoder-Decoder Architectures for Language Models [[arxiv link](https://arxiv.org/abs/2405.05254)][[weixin blog](https://mp.weixin.qq.com/s/X4HSyEreN4L4xTizC-_mow)][[code|official](https://github.com/microsoft/unilm/tree/master/YOCO)][`microsoft`; partially based on [`Flash-Attention`](https://github.com/Dao-AILab/flash-attention)]


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

* **SimMIM (CVPR2022)(arxiv2021.11)** SimMIM: A Simple Framework for Masked Image Modeling [[arxiv link](https://arxiv.org/abs/2111.09886)][[code|official](https://github.com/microsoft/SimMIM)][`Microsoft`, `a self-supervised approach that enables SwinV2-G`]

* 👍**MAE (CVPR2022)** Masked Autoencoders Are Scalable Vision Learners [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)][`ViT-based`, `FAIR`, `He Kaiming`， `It reconstructs the original signal given its partial observation`, `self-supervised pre-training`]

* **RSP(ICML2024)(arxiv2024.06)** Visual Representation Learning with Stochastic Frame Prediction [[arxiv link](https://arxiv.org/abs/2406.07398)][[project link](https://sites.google.com/view/2024rsp)][[code|official](https://github.com/huiwon-jang/RSP)][`KAIST + UC Berkeley + Dyson Robot Learning Lab`][It can be used for `Vision-based Robot Learning` inlcuding the `RLBench`]

### Lightweight and Efficient Training 

* **EfficientTrain++(TPAMI2024)(arxiv2024.05)** EfficientTrain++: Generalized Curriculum Learning for Efficient Visual Backbone Training [[paper link](https://ieeexplore.ieee.org/abstract/document/10530470/)][[arxiv link](https://arxiv.org/pdf/2405.08768)][[weixin blog](https://mp.weixin.qq.com/s/FJj0F2NcW9ftmT_lbO1R3w)][[code|official](https://github.com/LeapLabTHU/EfficientTrain)][`THU + BAAI`, used the `generalized curriculum learning`][The conference (EfficientTrain, ICCV2023) version [EfficientTrain: Exploring Generalized Curriculum Learning for Training Visual Backbones](https://arxiv.org/abs/2211.09703)]

