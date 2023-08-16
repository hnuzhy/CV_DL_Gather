# Semi-Supervised_Learning
also including `self-supervised learning` and `unsupervised learning`

---

## Materials

* [(github) SemiSeg: a list of "Awesome Semi-Supervised Semantic Segmentation" works](https://github.com/LiheYoung/UniMatch/blob/main/docs/SemiSeg.md)

---

## Papers (Semi-Supervised_Learning)

### ▶for Data Augmentation

* **AdaIN (ICCV2017)** Arbitrary Style Transfer in Real-Time With Adaptive Instance Normalization [[paper link](https://openaccess.thecvf.com/content_iccv_2017/html/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html)][[arxiv link](https://arxiv.org/abs/1703.06868v2)][[code|official](https://github.com/xunhuang1995/AdaIN-style)]

* 👍**Cutout (arxiv2017.08)** Improved Regularization of Convolutional Neural Networks with Cutout [[arxiv link](https://arxiv.org/abs/1708.04552)][[code|official](https://github.com/uoguelph-mlrg/Cutout)]

* 👍**Mixup (ICLR2018)** Mixup: Beyond Empirical Risk Minimization [[openreview link](https://openreview.net/forum?id=r1Ddp1-Rb)][[arxiv link](https://arxiv.org/abs/1710.09412)][[code|official](https://github.com/facebookresearch/mixup-cifar10)]

* **AdaMixUp (AAAI2019)** MixUp as Locally Linear Out-of-Manifold Regularization [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/4256)][[arxiv link](https://arxiv.org/abs/1809.02499)][[code|official](https://github.com/SITE5039/AdaMixUp)]

* 👍**CutMix (ICCV2019)** CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1905.04899)][[code|official](https://github.com/clovaai/CutMix-PyTorch)]

* **Manifold Mixup (ICML2019)** Manifold Mixup: Better Representations by Interpolating Hidden States [[paper link](https://proceedings.mlr.press/v97/verma19a.html)][[arxiv link](https://arxiv.org/abs/1806.05236)][[code|official](https://github.com/vikasverma1077/manifold_mixup)]

* 👍**RandAugment (CVPRW2020)** Randaugment: Practical Automated Data Augmentation With a Reduced Search Space [[paper link](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html)][[arxiv link](https://arxiv.org/abs/1909.13719)][[code|official](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)]


### ▶for Image Classification

* 👍👍**Mean Teachers (NIPS2017)** Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results [[[paper link](https://proceedings.neurips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html)][[arxiv link](https://arxiv.org/abs/1703.01780)][` the teacher is the moving average of the student which can be timely updated in every iteration`, `But their performance is limited because the two models tend to converge to the same point and stop further exploration`]

* **VAT (TPAMI2018)** Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning [[paper link](https://ieeexplore.ieee.org/abstract/document/8417973)][[arxiv link](https://arxiv.org/abs/1704.03976)][[code|official vat_chainer](https://github.com/takerum/vat_chainer)][[code|official vat_tf](https://github.com/takerum/vat_tf)]

* **DCT (Deep Co-Training)(ECCV2018)** Deep Co-Training for Semi-Supervised Image Recognition [[paper link](https://openaccess.thecvf.com/content_ECCV_2018/html/Siyuan_Qiao_Deep_Co-Training_for_ECCV_2018_paper.html)][[arxiv link](https://arxiv.org/abs/1803.05984v1)][`learn two different models by minimizing their prediction discrepancy`, `learn from different initializations to avoid the case where the two models converge to the same point`]

* **Dual-Student (ICCV2019)** Dual Student: Breaking the Limits of the Teacher in Semi-Supervised Learning [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Ke_Dual_Student_Breaking_the_Limits_of_the_Teacher_in_Semi-Supervised_ICCV_2019_paper.html)][`learn two different models by minimizing their prediction discrepancy`, `add view difference constraints to avoid the case where the two models converge to the same point`]

* 👍**MixMatch (NIPS2019)** MixMatch: A Holistic Approach to Semi-Supervised Learning [[paper link](https://proceedings.neurips.cc/paper/2019/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html)][[arxiv link](https://arxiv.org/abs/1905.02249)][[code|official](https://github.com/google-research/mixmatch)][`Google`, The first author is `David Berthelot`, `Combining Existing Useful SSL Techniques`]

* 👍**ReMixMatch (NIPS2020)** ReMixMatch: Semi-Supervised Learning with Distribution Matching and Augmentation Anchoring [[openreview link](https://openreview.net/forum?id=HklkeR4KPB)][[arxiv link](https://arxiv.org/abs/1911.09785)][[code|official](https://github.com/google-research/remixmatch)][`Google`, The first author is `David Berthelot`, `Applying Multiple Strong Augmentations for the Same Input Batch`]

* 👍👍**FixMatch (NIPS2020)** FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence [[paper link](https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html)][[arxiv link](https://arxiv.org/abs/2001.07685)][[code|official](https://github.com/google-research/fixmatch)][`Google`, The first author is `David Berthelot`, `Weak-Strong Augmentation Pairs`, `pseudo-labeling based (also called self-training)`]

* 👍**UDA (NIPS2020)** Unsupervised Data Augmentation for Consistency Training [[paper link](https://proceedings.neurips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html)][[arxiv link](https://arxiv.org/abs/1904.12848)][[code|official](https://github.com/google-research/uda)]

* 👍**FlexMatch (NIPS2021)** FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling [[paper link](https://proceedings.neurips.cc/paper/2021/hash/995693c15f439e3d189b06e89d145dd5-Abstract.html)][[arxiv link](https://arxiv.org/abs/2110.08263)][[code|official](https://github.com/TorchSSL/TorchSSL)]

* **Dash (ICML2021)** Dash: Semi-Supervised Learning with Dynamic Thresholding [[paper link](https://proceedings.mlr.press/v139/xu21e.html)][[arxiv link](https://arxiv.org/abs/2109.00650)][`It proposes dynamic and adaptive pseudo label filtering, better suited for the training process (similar to the FixMatch)`] 

* **SimPLE (CVPR2021)** SimPLE: Similar Pseudo Label Exploitation for Semi-Supervised Classification [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_SimPLE_Similar_Pseudo_Label_Exploitation_for_Semi-Supervised_Classification_CVPR_2021_paper.html)][[arxiv link](http://arxiv.org/abs/2103.16725)][[code|official](https://github.com/zijian-hu/SimPLE)][`It proposes the paired loss minimizing the statistical distance between confident and similar pseudo labels`]

* **SemCo (CVPR2021)** All Labels Are Not Created Equal: Enhancing Semi-Supervision via Label Grouping and Co-Training [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Nassar_All_Labels_Are_Not_Created_Equal_Enhancing_Semi-Supervision_via_Label_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2104.05248)][[code|official](https://github.com/islam-nassar/semco)][`It considers label semantics to prevent the degradation of pseudo label quality for visually similar classes in a co-training manner`]

* **EMAN (CVPR2021)** Exponential Moving Average Normalization for Self-Supervised and Semi-Supervised Learning [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Cai_Exponential_Moving_Average_Normalization_for_Self-Supervised_and_Semi-Supervised_Learning_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2101.08482)][[code|official](https://github.com/amazon-science/exponential-moving-average-normalization)]

* **** [[]()][[]()][[]()]

### ▶for Object Detection

* **MUM (CVPR2022)** MUM: Mix Image Tiles and UnMix Feature Tiles for Semi-Supervised Object Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_MUM_Mix_Image_Tiles_and_UnMix_Feature_Tiles_for_Semi-Supervised_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2111.10958)][[code|official](https://github.com/JongMokKim/mix-unmix)][`data augmentation`, [(arxiv2022.03) Pose-MUM: Reinforcing Key Points Relationship for Semi-Supervised Human Pose Estimation](https://arxiv.org/abs/2203.07837)]]

* 👍**Unbiased Teacher (ICLR2021)** Unbiased Teacher for Semi-Supervised Object Detection [[openreview link](https://openreview.net/forum?id=MJIve1zgR_)][[arxiv link](https://arxiv.org/abs/2102.09480)][[project link](https://ycliu93.github.io/projects/unbiasedteacher.html)][[code|official](https://github.com/facebookresearch/unbiased-teacher)]


* **** [[]()][[]()][[]()]

### ▶for Semantic Segmentation

* **UniMatch (CVPR2023)** Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2208.09910)][[code|official](https://github.com/LiheYoung/UniMatch)]


### ▶for Pose Estimation

* Please refer [[Domain Adaptive Multiple Person Pose Estimation](https://github.com/hnuzhy/CV_DL_Gather/blob/master/pose_estimation/readme_details.md#-domain-adaptive-multiple-person-pose-estimation)]

---

## Papers (Self-Supervised Learning or Unsupervised Learning)

### ▶for Image Classification

* **SwAV (NIPS2020)** Unsupervised Learning of Visual Features by Contrasting Cluster Assignments [[paper link](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)][[arxiv link](https://arxiv.org/abs/2006.09882)]
[[code|official](https://github.com/facebookresearch/swav)][including `contrastive learning`]

