# Semi-Supervised_Learning
also including `self-supervised learning` and `unsupervised learning`

## Contents

* **[1) Papers (Semi-Supervised_Learning)](#papers-semi-supervised_learning)**
  * **[▶ for Data Augmentation](#for-data-augmentation)**
  * **[▶ for Image Classification](#for-image-classification)**
  * **[▶ for Object Detection](#for-object-detection)**
  * **[▶ for Semantic Segmentation](#for-semantic-segmentation)**
  * **[▶ for Pose Estimation](#for-pose-estimation)**
  * **[▶ for 3D Object Detection](#for-3d-object-detection)**
  * **[▶ for 6D Object Pose Estimation](#for-6d-object-pose-estimation)**
  * **[▶ for Rotation Regression](#for-rotation-regression)**
  * **[▶ for Crowd Counting](#for-crowd-counting)**
  * **[▶ for 3D Hand-Object](#for-3d-hand-object)**
  * **[▶ for 3D Reconstruction](#for-3d-reconstruction)**
* **[2) Papers (Self-Supervised Learning or Unsupervised Learning)](#papers-self-supervised-learning-or-unsupervised-learning)**
  * **[▶ for Image Classification](#for-image-classification-1)**

---

## Materials

* [(github) SemiSeg: a list of "Awesome Semi-Supervised Semantic Segmentation" works](https://github.com/LiheYoung/UniMatch/blob/main/docs/SemiSeg.md)

---

## Papers (Semi-Supervised_Learning)

### ▶for Data Augmentation

* 👍**AdaIN (ICCV2017)** Arbitrary Style Transfer in Real-Time With Adaptive Instance Normalization [[paper link](https://openaccess.thecvf.com/content_iccv_2017/html/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html)][[arxiv link](https://arxiv.org/abs/1703.06868v2)][[code|official](https://github.com/xunhuang1995/AdaIN-style)]

* 👍**Cutout (arxiv2017.08)** Improved Regularization of Convolutional Neural Networks with Cutout [[arxiv link](https://arxiv.org/abs/1708.04552)][[code|official](https://github.com/uoguelph-mlrg/Cutout)]

* 👍**Mixup (ICLR2018)** Mixup: Beyond Empirical Risk Minimization [[openreview link](https://openreview.net/forum?id=r1Ddp1-Rb)][[arxiv link](https://arxiv.org/abs/1710.09412)][[code|official](https://github.com/facebookresearch/mixup-cifar10)]

* **AdaMixUp (AAAI2019)** MixUp as Locally Linear Out-of-Manifold Regularization [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/4256)][[arxiv link](https://arxiv.org/abs/1809.02499)][[code|official](https://github.com/SITE5039/AdaMixUp)]

* 👍**CutMix (ICCV2019)** CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1905.04899)][[code|official](https://github.com/clovaai/CutMix-PyTorch)]

* **Manifold Mixup (ICML2019)** Manifold Mixup: Better Representations by Interpolating Hidden States [[paper link](https://proceedings.mlr.press/v97/verma19a.html)][[arxiv link](https://arxiv.org/abs/1806.05236)][[code|official](https://github.com/vikasverma1077/manifold_mixup)]

* **AutoAugment (CVPR2019)** AutoAugment: Learning Augmentation Policies from Data [[paper link]](https://research.google/pubs/pub47890/)][[arxiv link](https://arxiv.org/abs/1805.09501)][[code|official](https://github.com/tensorflow/models/tree/master/research/autoaugment)][`google`]

* 👍**RandAugment (CVPRW2020)** Randaugment: Practical Automated Data Augmentation With a Reduced Search Space [[paper link](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html)][[arxiv link](https://arxiv.org/abs/1909.13719)][[code|official](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)]

* **PuzzleMix(ICML2020)** Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup [[paper link](https://proceedings.mlr.press/v119/kim20b.html)][[arxiv link](https://arxiv.org/abs/2009.06962)][[code|official](https://github.com/snu-mllab/PuzzleMix)]

* **AugMix(ICLR2020)** AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty [[paper link](https://openreview.net/forum?id=S1gmrxHFvB)][[arxiv link](https://arxiv.org/abs/1912.02781)][[code|official](https://github.com/google-research/augmix)][for `improving robustness and uncertainty` and `domain generalization`]

* **AugMax(NIPS2021)** AugMax: Adversarial Composition of Random Augmentations for Robust Training [[paper link](https://proceedings.neurips.cc/paper/2021/hash/01e9565cecc4e989123f9620c1d09c09-Abstract.html)][[openreview link](https://openreview.net/forum?id=P5MtdcVdFZ4)][[code|official](https://github.com/VITA-Group/AugMax)][`AugMax-DuBIN`, compared to `AugMix`]

* **MixStyle(ICLR2021)** Domain Generalization with MixStyle [[openreview link](https://openreview.net/forum?id=6xHJ37MVxxp)][[arxiv link](https://arxiv.org/abs/2104.02008)][[slides link](https://iclr.cc/media/Slides/iclr/2021/virtual(03-08-00)-03-08-00UTC-2738-domain_generali.pdf)][[code}official](https://github.com/KaiyangZhou/mixstyle-release)][originally designed for `domain generalization`, inspired by `AdaIN (ICCV2017)`]

* **StyleMix(CVPR2021)** StyleMix: Separating Content and Style for Enhanced Data Augmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Hong_StyleMix_Separating_Content_and_Style_for_Enhanced_Data_Augmentation_CVPR_2021_paper.html)][[code|official](https://github.com/alsdml/StyleMix)]

* **KeepAugment(CVPR2021)** KeepAugment: A Simple Information-Preserving Data Augmentation Approach [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Gong_KeepAugment_A_Simple_Information-Preserving_Data_Augmentation_Approach_CVPR_2021_paper.html)][[arxiv link](http://arxiv.org/abs/2011.11778)]

* **Augmentation-for-LNL(CVPR2021)** Augmentation Strategies for Learning With Noisy Labels [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Nishi_Augmentation_Strategies_for_Learning_With_Noisy_Labels_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2103.02130)][[code|official](https://github.com/KentoNishi/Augmentation-for-LNL)]

* **TrivialAugment(ICCV2021 oral)** TrivialAugment: Tuning-Free Yet State-of-the-Art Data Augmentation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Muller_TrivialAugment_Tuning-Free_Yet_State-of-the-Art_Data_Augmentation_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2103.10158)][[code|official](https://github.com/automl/trivialaugment)]

* **RegMixup(NIPS2022)** Using Mixup as a Regularizer Can Surprisingly Improve Accuracy & Out-of-Distribution Robustness [[paper link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5ddcfaad1cb72ce6f1a365e8f1ecf791-Abstract-Conference.html)][[arxiv link](https://arxiv.org/abs/2206.14502)][[code|official](https://github.com/FrancescoPinto/RegMixup)]

* 👍**C-Mixup(NIPS2022)** C-Mixup: Improving Generalization in Regression [[paper link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html)][[arxiv link](https://arxiv.org/abs/2210.05775)][[code|official](https://github.com/huaxiuyao/C-Mixup)][for the `Regression` task]

* **DeepAA(ICLR2022)** Deep AutoAugment [[openreview link](https://openreview.net/forum?id=St-53J9ZARf)][[code|official](https://github.com/AIoT-MLSys-Lab/DeepAA)][`Michigan State University + Amazon Web Services`]

* **AutoMix(ECCV2022)** AutoMix: Unveiling the Power of Mixup for Stronger Classifiers [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_26)][[arxiv link](https://arxiv.org/abs/2103.13027)][[code|official](https://github.com/Westlake-AI/openmixup)][`Westlake University`]

* **TokenMix(ECCV2022)** TokenMix: Rethinking Image Mixing for Data Augmentation in Vision Transformers [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19809-0_26)][[arxiv link](https://arxiv.org/abs/2207.08409)][[code|official](https://github.com/Sense-X/TokenMix)]

* **TransMix(CVPR2022)** TransMix: Attend To Mix for Vision Transformers [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_TransMix_Attend_To_Mix_for_Vision_Transformers_CVPR_2022_paper.html)][[arxiv link](http://arxiv.org/abs/2111.09833)][[code|official](https://github.com/Beckschen/TransMix)]


### ▶for Image Classification

* 👍👍**Mean Teachers (NIPS2017)** Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results [[paper link](https://proceedings.neurips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html)][[arxiv link](https://arxiv.org/abs/1703.01780)][` the teacher is the moving average of the student which can be timely updated in every iteration`, `But their performance is limited because the two models tend to converge to the same point and stop further exploration`]

* **VAT (TPAMI2018)** Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning [[paper link](https://ieeexplore.ieee.org/abstract/document/8417973)][[arxiv link](https://arxiv.org/abs/1704.03976)][[code|official vat_chainer](https://github.com/takerum/vat_chainer)][[code|official vat_tf](https://github.com/takerum/vat_tf)]

* **DCT (Deep Co-Training)(ECCV2018)** Deep Co-Training for Semi-Supervised Image Recognition [[paper link](https://openaccess.thecvf.com/content_ECCV_2018/html/Siyuan_Qiao_Deep_Co-Training_for_ECCV_2018_paper.html)][[arxiv link](https://arxiv.org/abs/1803.05984v1)][`learn two different models by minimizing their prediction discrepancy`, `learn from different initializations to avoid the case where the two models converge to the same point`]

* **Dual-Student (ICCV2019)** Dual Student: Breaking the Limits of the Teacher in Semi-Supervised Learning [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Ke_Dual_Student_Breaking_the_Limits_of_the_Teacher_in_Semi-Supervised_ICCV_2019_paper.html)][`learn two different models by minimizing their prediction discrepancy`, `add view difference constraints to avoid the case where the two models converge to the same point`]

* 👍**MixMatch (NIPS2019)** MixMatch: A Holistic Approach to Semi-Supervised Learning [[paper link](https://proceedings.neurips.cc/paper/2019/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html)][[arxiv link](https://arxiv.org/abs/1905.02249)][[code|official](https://github.com/google-research/mixmatch)][`Google`, The first author is `David Berthelot`, `Combining Existing Useful SSL Techniques`]

* 👍**ReMixMatch (NIPS2020)** ReMixMatch: Semi-Supervised Learning with Distribution Matching and Augmentation Anchoring [[openreview link](https://openreview.net/forum?id=HklkeR4KPB)][[arxiv link](https://arxiv.org/abs/1911.09785)][[code|official](https://github.com/google-research/remixmatch)][`Google`, The first author is `David Berthelot`, `Applying Multiple Strong Augmentations for the Same Input Batch`]

* 👍👍**FixMatch (NIPS2020)** FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence [[paper link](https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html)][[arxiv link](https://arxiv.org/abs/2001.07685)][[code|official](https://github.com/google-research/fixmatch)][`Google`, The first author is `David Berthelot`, `Weak-Strong Augmentation Pairs`, `pseudo-labeling based (also called self-training)`]

* **FeatMatch(ECCV2020)** FeatMatch: Feature-Based Augmentation for Semi-supervised Learning [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58523-5_28)][[arxiv link](https://arxiv.org/abs/2007.08505)][[project link](https://sites.google.com/view/chiawen-kuo/home/featmatch)][[code|official](https://github.com/GT-RIPL/FeatMatch)]

* 👍**UDA (NIPS2020)** Unsupervised Data Augmentation for Consistency Training [[paper link](https://proceedings.neurips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html)][[arxiv link](https://arxiv.org/abs/1904.12848)][[code|official](https://github.com/google-research/uda)]

* 👍**FlexMatch (NIPS2021)** FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling [[paper link](https://proceedings.neurips.cc/paper/2021/hash/995693c15f439e3d189b06e89d145dd5-Abstract.html)][[arxiv link](https://arxiv.org/abs/2110.08263)][[code|official](https://github.com/TorchSSL/TorchSSL)]

* **Dash (ICML2021)** Dash: Semi-Supervised Learning with Dynamic Thresholding [[paper link](https://proceedings.mlr.press/v139/xu21e.html)][[arxiv link](https://arxiv.org/abs/2109.00650)][`It proposes dynamic and adaptive pseudo label filtering, better suited for the training process (similar to the FixMatch)`] 

* **SimPLE (CVPR2021)** SimPLE: Similar Pseudo Label Exploitation for Semi-Supervised Classification [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_SimPLE_Similar_Pseudo_Label_Exploitation_for_Semi-Supervised_Classification_CVPR_2021_paper.html)][[arxiv link](http://arxiv.org/abs/2103.16725)][[code|official](https://github.com/zijian-hu/SimPLE)][`It proposes the paired loss minimizing the statistical distance between confident and similar pseudo labels`]

* **SemCo (CVPR2021)** All Labels Are Not Created Equal: Enhancing Semi-Supervision via Label Grouping and Co-Training [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Nassar_All_Labels_Are_Not_Created_Equal_Enhancing_Semi-Supervision_via_Label_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2104.05248)][[code|official](https://github.com/islam-nassar/semco)][`It considers label semantics to prevent the degradation of pseudo label quality for visually similar classes in a co-training manner`]

* **EMAN (CVPR2021)** Exponential Moving Average Normalization for Self-Supervised and Semi-Supervised Learning [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Cai_Exponential_Moving_Average_Normalization_for_Self-Supervised_and_Semi-Supervised_Learning_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2101.08482)][[code|official](https://github.com/amazon-science/exponential-moving-average-normalization)][`may not that generic`]

* **Classification-SemiCLS (CVPR2022)** Class-Aware Contrastive Semi-Supervised Learning [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Class-Aware_Contrastive_Semi-Supervised_Learning_CVPR_2022_paper.html)][[arxiv link](http://arxiv.org/abs/2203.02261)][[code|official](https://github.com/TencentYoutuResearch/Classification-SemiCLS)][based on `FixMatch`, `THU + Tencent Youtu Lab`]

* **SimMatch (CVPR2022)** SimMatch: Semi-Supervised Learning With Similarity Matching [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_SimMatch_Semi-Supervised_Learning_With_Similarity_Matching_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2203.06915)][[code|official](https://github.com/mingkai-zheng/SimMatch)][`The University of Sydney`]

* **FreeMatch (ICLR2023)** FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning [[openreview link](https://openreview.net/forum?id=PDrUPTXJI_A)][[arxiv link](https://arxiv.org/abs/2205.07246)][[code|official](https://github.com/microsoft/Semi-supervised-learning)][`microsoft`]

* **SoftMatch (ICLR2023)** SoftMatch: Addressing the Quantity-Quality Trade-off in Semi-supervised Learning [[openreview link](https://openreview.net/forum?id=ymt1zQXBDiF)][[arxiv link](https://arxiv.org/abs/2301.10921)][[code|official](https://github.com/microsoft/Semi-supervised-learning)][`microsoft`]

* **FullMatch (CVPR2023)** Boosting Semi-Supervised Learning by Exploiting All Unlabeled Data [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Boosting_Semi-Supervised_Learning_by_Exploiting_All_Unlabeled_Data_CVPR_2023_paper.html)][[arxiv link](http://arxiv.org/abs/2303.11066)][[code|official](https://github.com/megvii-research/FullMatch)][based on `FixMatch`, `megvii-research`]

* **CHMatch (CVPR2023)** CHMATCH: Contrastive Hierarchical Matching and Robust Adaptive Threshold Boosted Semi-Supervised Learning [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_CHMATCH_Contrastive_Hierarchical_Matching_and_Robust_Adaptive_Threshold_Boosted_Semi-Supervised_CVPR_2023_paper.html)][[code|official](https://github.com/sailist/CHMatch)][based on `FixMatch` and `FlexMatch`, `Harbin Institute of Technology (Shenzhen)`]

* 👍**Suave-Daino(CVPR2023)** Semi-Supervised Learning Made Simple With Self-Supervised Clustering [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Fini_Semi-Supervised_Learning_Made_Simple_With_Self-Supervised_Clustering_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2306.07483)][[code|official](https://github.com/pietroastolfi/suave-daino)][based on `Self-Supervised` methods such as `SwAV` or `DINO`]

* **ProtoCon(CVPR2023)** PROTOCON: Pseudo-label Refinement via Online Clustering and Prototypical Consistency for Efficient Semi-supervised Learning [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Nassar_ProtoCon_Pseudo-Label_Refinement_via_Online_Clustering_and_Prototypical_Consistency_for_CVPR_2023_paper.html)][[arxiv link](http://arxiv.org/abs/2303.13556)][code is unavailable][`Monash University, Australia`]

* **SAA (ICCV2023)** Enhancing Sample Utilization through Sample Adaptive Augmentation in Semi-Supervised Learning [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Gui_Enhancing_Sample_Utilization_through_Sample_Adaptive_Augmentation_in_Semi-Supervised_Learning_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2309.03598)][[code|official](https://github.com/GuanGui-nju/SAA)][based on `FixMatch` and `FlexMatch`]

* **ShrinkMatch (ICCV2023)** Shrinking Class Space for Enhanced Certainty in Semi-Supervised Learning [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Shrinking_Class_Space_for_Enhanced_Certainty_in_Semi-Supervised_Learning_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2308.06777)][[code|official](https://github.com/LiheYoung/ShrinkMatch)][based on `FixMatch`]

* **SimMatchV2 (ICCV2023)** SimMatchV2: Semi-Supervised Learning with Graph Consistency [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_SimMatchV2_Semi-Supervised_Learning_with_Graph_Consistency_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2308.06692)][[code|official](https://github.com/mingkai-zheng/SimMatchV2)][`The University of Sydney`]

* **** [[]()][[]()][[]()]


### ▶for Object Detection

* **ssl_detection(arxiv2020.12)** A Simple Semi-Supervised Learning Framework for Object Detection [[arxiv link](https://arxiv.org/abs/2005.04757)][[code|official](https://github.com/google-research/ssl_detection/)][`google-research`]

* 👍**Unbiased Teacher (ICLR2021)** Unbiased Teacher for Semi-Supervised Object Detection [[openreview link](https://openreview.net/forum?id=MJIve1zgR_)][[arxiv link](https://arxiv.org/abs/2102.09480)][[project link](https://ycliu93.github.io/projects/unbiasedteacher.html)][[code|official](https://github.com/facebookresearch/unbiased-teacher)]

* **Unbiased Teacher v2 (CVPR2022)** Unbiased Teacher v2: Semi-Supervised Object Detection for Anchor-Free and Anchor-Based Detectors [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Unbiased_Teacher_v2_Semi-Supervised_Object_Detection_for_Anchor-Free_and_Anchor-Based_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2206.09500)][[project link](https://ycliu93.github.io/projects/unbiasedteacher2.html)]

* **MUM (CVPR2022)** MUM: Mix Image Tiles and UnMix Feature Tiles for Semi-Supervised Object Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_MUM_Mix_Image_Tiles_and_UnMix_Feature_Tiles_for_Semi-Supervised_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2111.10958)][[code|official](https://github.com/JongMokKim/mix-unmix)][`data augmentation`, [(arxiv2022.03) Pose-MUM: Reinforcing Key Points Relationship for Semi-Supervised Human Pose Estimation](https://arxiv.org/abs/2203.07837)]]

* **ARSL (CVPR2023)** Ambiguity-Resistant Semi-Supervised Learning for Dense Object Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Ambiguity-Resistant_Semi-Supervised_Learning_for_Dense_Object_Detection_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.14960)][[code|official of PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)][`PaddleDetection`]

* **MixTeacher (CVPR2023)** MixTeacher: Mining Promising Labels With Mixed Scale Teacher for Semi-Supervised Object Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_MixTeacher_Mining_Promising_Labels_With_Mixed_Scale_Teacher_for_Semi-Supervised_CVPR_2023_paper.html)][[arxiv link](http://arxiv.org/abs/2303.09061)][[code|official](https://github.com/lliuz/MixTeacher)]

* **Semi-DETR (CVPR2023)** Semi-DETR: Semi-Supervised Object Detection With Detection Transformers [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Semi-DETR_Semi-Supervised_Object_Detection_With_Detection_Transformers_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2307.08095)][[code|official](https://github.com/JCZ404/Semi-DETR)][`SYSU`]

* **Consistent-Teacher (CVPR2023)** Consistent-Teacher: Towards Reducing Inconsistent Pseudo-Targets in Semi-Supervised Object Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Consistent-Teacher_Towards_Reducing_Inconsistent_Pseudo-Targets_in_Semi-Supervised_Object_Detection_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2209.01589)][[project link](https://adamdad.github.io/consistentteacher/)][[code|official](https://github.com/Adamdad/ConsistentTeacher)][`SenseTime Research`]


### ▶for Semantic Segmentation

* **PS-MT(CVPR2022)** Perturbed and Strict Mean Teachers for Semi-Supervised Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Perturbed_and_Strict_Mean_Teachers_for_Semi-Supervised_Semantic_Segmentation_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2111.12903)][[code|official](https://github.com/yyliu01/PS-MT)]

* **UniMatch (CVPR2023)** Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Revisiting_Weak-to-Strong_Consistency_in_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2208.09910)][[code|official](https://github.com/LiheYoung/UniMatch)]


### ▶for Pose Estimation

* Please refer [[Transfer Learning of Multiple Person Pose Estimation](https://github.com/hnuzhy/CV_DL_Gather/blob/master/pose_estimation/readme_details.md#-transfer-learning-of-multiple-person-pose-estimation)]

### ▶for 3D Object Detection

* 👍**SESS (CVPR2020 oral)** SESS: Self-Ensembling Semi-Supervised 3D Object Detection [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhao_SESS_Self-Ensembling_Semi-Supervised_3D_Object_Detection_CVPR_2020_paper.html)][[arxiv link](https://arxiv.org/abs/1912.11803)][[code|official](https://github.com/Na-Z/sess)][`National University of Singapore`, comparing to `SESS`, on datasets `ScanNet` and `SUNRGB-D`, using `Mean-Teacher`]

* 👍👍**3DIoUMatch (CVPR2021)** 3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_3DIoUMatch_Leveraging_IoU_Prediction_for_Semi-Supervised_3D_Object_Detection_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2012.04355)][[code|official](https://github.com/yezhen17/3DIoUMatch)][`Stanford University + Tsinghua University + NVIDIA`, on datasets `ScanNet`, `SUNRGB-D` and `KITTI`, using `Mean-Teacher` and `FixMatch`]

* **Proficient-Teachers (ECCV2022)** Semi-supervised 3D Object Detection with Proficient Teachers [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19839-7_42)][[arxiv link](https://arxiv.org/abs/2207.12655)][[code|official (not really released!!!👎)](https://github.com/yinjunbo/ProficientTeachers)][`Beijing Institute of Technology`, comparing to `SESS` and `3DIoUMatch` yet no code, on datasets `ONCE` and `Waymo Open`, revised based on `Mean-Teacher`]

* **UpCycling (ICCV2023)** UpCycling: Semi-supervised 3D Object Detection without Sharing Raw-level Unlabeled Scenes [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Hwang_UpCycling_Semi-supervised_3D_Object_Detection_without_Sharing_Raw-level_Unlabeled_Scenes_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2211.11950)][`Seoul National University`, comparing to `3DIoUMatch` yet no code, on datasets `ScanNet`, `SUNRGB-D` and `KITTI`]

* **ViT-WSS3D (ICCV2023)** A Simple Vision Transformer for Weakly Semi-supervised 3D Object Detection
 [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_A_Simple_Vision_Transformer_for_Weakly_Semi-supervised_3D_Object_Detection_ICCV_2023_paper.html)][`HUST`, comparing to `3DIoUMatch` yet no code, on datasets `SUNRGB-D` and `KITTI`]

* **Side-Aware (ICCV2023)** Not Every Side Is Equal: Localization Uncertainty Estimation for Semi-Supervised 3D Object Detection [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Not_Every_Side_Is_Equal_Localization_Uncertainty_Estimation_for_Semi-Supervised_ICCV_2023_paper.html)][`USTC`, comparing to `SESS` and `3DIoUMatch` yet no code, using `Mean-Teacher`]

* **NoiseDet (ICCV2023)** Learning from Noisy Data for Semi-Supervised 3D Object Detection [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Learning_from_Noisy_Data_for_Semi-Supervised_3D_Object_Detection_ICCV_2023_paper.html)][[code|official (not really released!!!👎)](https://github.com/zehuichen123/NoiseDet)][`USTC`, comparing to `SESS` and `3DIoUMatch` yet no code, on datasets `ONCE` and `Waymo Open`]

* 👍👍**DQS3D (ICCV2023)** DQS3D: Densely-matched Quantization-aware Semi-supervised 3D Detection [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_DQS3D_Densely-matched_Quantization-aware_Semi-supervised_3D_Detection_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2304.13031)][[code|official](https://github.com/AIR-DISCOVER/DQS3D)][`Institute for AI Industry Research (AIR), Tsinghua University`, comparing to `SESS` and `3DIoUMatch`, on datasets `ScanNet` and `SUNRGB-D`, using `Mean-Teacher`]


### ▶for 6D Object Pose Estimation

* **multipath(CVPR2020)** Multi-Path Learning for Object Pose Estimation Across Domains [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Sundermeyer_Multi-Path_Learning_for_Object_Pose_Estimation_Across_Domains_CVPR_2020_paper.html)][[code|official](https://github.com/DLR-RM/AugmentedAutoencoder/tree/multipath)][`Domain Adaptation`, `6D Object Detection`, `3D Object Pose Estimation`]

* **Self6D(ECCV2020)** Self6D: Self-Supervised Monocular 6D Object Pose Estimation [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58452-8_7)][[arxiv link](https://arxiv.org/abs/2004.06468)][[code|official (Self6D-Diff-Renderer)](https://github.com/THU-DA-6D-Pose-Group/Self6D-Diff-Renderer)][`THU`]

* **Self6D++(TPAMI2021)** Occlusion-Aware Self-Supervised Monocular 6D Object Pose Estimation [[paper link](https://ieeexplore.ieee.org/document/9655492)][[arxiv link](https://arxiv.org/abs/2203.10339)][[code|official](https://github.com/THU-DA-6D-Pose-Group/self6dpp)][`THU`]

* **NVSM(NIPS2021)** Neural View Synthesis and Matching for Semi-Supervised Few-Shot Learning of 3D Pose [[paper link](https://proceedings.neurips.cc/paper_files/paper/2021/hash/3a61ed715ee66c48bacf237fa7bb5289-Abstract.html)][[arxiv link](https://arxiv.org/abs/2110.14213)][[code|official](https://github.com/Angtian/NeuralVS)]

* **ViewNet(ICCV2021)** ViewNet: Unsupervised Viewpoint Estimation From Conditional Generation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Mariotti_ViewNet_Unsupervised_Viewpoint_Estimation_From_Conditional_Generation_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2212.00435)][`Unsupervised Learning`, trained on `PASCAL3D+`, `University of Edinburgh`]

* **DSC-PoseNet(CVPR2021)** DSC-PoseNet: Learning 6DoF Object Pose Estimation via Dual-scale Consistency [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_DSC-PoseNet_Learning_6DoF_Object_Pose_Estimation_via_Dual-Scale_Consistency_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2104.03658)][[code is not available]()][`Baidu Research`, a self-supervised manner, needing 2D bounding boxes]

* **zero-shot-pose(ECCV2022)** Zero-Shot Category-Level Object Pose Estimation [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19842-7_30)][[arxiv link](https://arxiv.org/abs/2204.03635)][[code|official](https://github.com/applied-ai-lab/zero-shot-pose)][`Zero-Shot Learning`, `University of Oxford`, on the dataset `CO3D`, `the authors re-annotated 10 sequences from each of 20 categories with ground-truth poses.`, `all baselines are reproduced by the authors.`]

* **Self-DPDN(ECCV2022)** Category-Level 6D Object Pose and Size Estimation Using Self-supervised Deep Prior Deformation Networks [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_2)][[arxiv link](https://arxiv.org/abs/2207.05444)][[code|official](https://github.com/JiehongLin/Self-DPDN)][`Self-Supervised`, `Domain Adaptation`, `South China University of Technology`, another work [`VI-Net (ICCV2023)`](https://github.com/JiehongLin/VI-Net) with title `VI-Net: Boosting Category-level 6D Object Pose Estimation via Learning Decoupled Rotations on the Spherical Representations`]

* 👍**Wild6D + RePoNet (NIPS2022)** Category-Level 6D Object Pose Estimation in the Wild: A Semi-Supervised Learning Approach and A New Dataset [[paper link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/afe99e55be23b3523818da1fefa33494-Abstract-Conference.html)][[arxiv link](https://arxiv.org/abs/2206.15436)][[project link](https://oasisyang.github.io/semi-pose)][[code|official](https://github.com/OasisYang/Wild6D)][`University of California San Diego`, a new dataset `Wild6D`, [`Xiaolong Wang`](https://xiaolonw.github.io/), [`Yang Fu 付旸`](https://oasisyang.github.io/), based on the [NOCS](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.html)]

* **UDA-COPE(CVPR2022)** UDA-COPE: Unsupervised Domain Adaptation for Category-level Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_UDA-COPE_Unsupervised_Domain_Adaptation_for_Category-Level_Object_Pose_Estimation_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2111.12580)][[project link](https://sites.google.com/view/taeyeop-lee/udacope)][[no code]()][[`Domain Adaptation`, `Taeyeop Lee`](https://sites.google.com/view/taeyeop-lee/), based on the [NOCS](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.html)]

* **SSC-6D(AAAI2022)** Self-Supervised Category-Level 6D Object Pose Estimation with Deep Implicit Shape Representation [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/20104)][[code|official](https://github.com/swords123/SSC-6D)][`Dalian University of Technology`]

* **MAST(IJCAI2023)** Manifold-Aware Self-Training for Unsupervised Domain Adaptation on Regressing 6D Object Pose [[paper link](https://www.ijcai.org/proceedings/2023/0193.pdf)][[arxiv link](https://arxiv.org/abs/2305.10808)][`Domain Adaptation`, `Self-Training`]

* 👍**self-pose(ICLR2023)(arxiv 2022.10)** Self-Supervised Geometric Correspondence for Category-Level 6D Object Pose Estimation in the Wild [[openreview link](https://openreview.net/forum?id=ZKDUlVMqG_O)][[arxiv link](https://arxiv.org/abs/2210.07199)][[project link](https://kywind.github.io/self-pose)][[code|official](https://github.com/kywind/self-corr-pose)][training and testing on `Wild6D`, [`Kaifeng Zhang`](https://kywind.github.io/), second author is [`Yang Fu 付旸`](https://oasisyang.github.io/)]

* **TTA-COPE (CVPR2023)** TTA-COPE: Test-Time Adaptation for Category-Level Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_TTA-COPE_Test-Time_Adaptation_for_Category-Level_Object_Pose_Estimation_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.16730)][[project link](https://sites.google.com/view/taeyeop-lee/ttacope)][[Code is not available]()][`Test-Time Adaptation`, [`Taeyeop Lee`](https://sites.google.com/view/taeyeop-lee/), `The proposed pose ensemble and the self-training loss improve category-level object pose performance during test time under both semi-supervised and unsupervised settings.`, based on the [NOCS](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.html)]

* 👍**PseudoFlow(ICCV2023)** Pseudo Flow Consistency for Self-Supervised 6D Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Hai_Pseudo_Flow_Consistency_for_Self-Supervised_6D_Object_Pose_Estimation_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2308.10016)][[code|official](https://github.com/yanghai-1218/pseudoflow)][[`Yang Hai(海洋)`](https://yanghai-1218.github.io/), [`Yinlin Hu (胡银林)`](https://yinlinhu.github.io/)]

* **SA6D(CoRL2023)** SA6D: Self-Adaptive Few-Shot 6D Pose Estimator for Novel and Occluded Objects [[openreview link](https://openreview.net/forum?id=gdkKi_F55h)][[project link](https://sites.google.com/view/sa6d)][[arxiv link](https://arxiv.org/abs/2308.16528)][`Bosch Center for AI`, `robotic manipulation`, `few-shot pose estimation (FSPE)`, inputs: `a small number of cluttered reference images`]

* **Cas6D (arxiv2023.06)** Learning to Estimate 6DoF Pose from Limited Data: A Few-Shot, Generalizable Approach using RGB Images [[arxiv link](https://arxiv.org/abs/2306.07598)][`Few-Shot Learning`, `ByteDance`, compared to `OnePose++` and `Gen6D`, trained on datasets `LINEMOD` and `GenMOP`, inputs: ` sparse support views`]


### ▶for Rotation Regression

* 👍**FisherMatch(CVPR2022 Oral)** FisherMatch: Semi-Supervised Rotation Regression via Entropy-Based Filtering [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Yin_FisherMatch_Semi-Supervised_Rotation_Regression_via_Entropy-Based_Filtering_CVPR_2022_paper.html)][[arxiv link](http://arxiv.org/abs/2203.15765)][[project link](https://yd-yin.github.io/FisherMatch/)][[code|official](https://github.com/yd-yin/FisherMatch)][`3DoF rotation estimation`, based on `FixMatch` and `Semi_Human_Pose`, maybe suitable for `3D head pose estimation`, the `Semi-Supervised Rotation Regression` task][based on the `matrix Fisher distribution` theory introduced in [(NIPS2020) An Analysis of SVD for Deep Rotation Estimation](https://proceedings.neurips.cc/paper/2020/hash/fec3392b0dc073244d38eba1feb8e6b7-Abstract.html) and [(NIPS2020) Probabilistic Orientation Estimation with Matrix Fisher Distributions](https://proceedings.neurips.cc/paper/2020/hash/33cc2b872dfe481abef0f61af181dfcf-Abstract.html)]

* **UCVME(AAAI2023)** Semi-Supervised Deep Regression with Uncertainty Consistency and Variational Model Ensembling via Bayesian Neural Networks [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/25890/)][[arxiv link](https://arxiv.org/abs/2302.07579)][[code | official](https://github.com/xmed-lab/UCVME)][`Semi-Supervised Rotation Regression`]


### ▶for Crowd Counting

* **IRAST(ECCV2020)** Semi-supervised Crowd Counting via Self-training on Surrogate Tasks [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_15)][[arxiv link](https://arxiv.org/abs/2007.03207)][`Sichuan University`]

* **UA_crowd_counting(ICCV2021)** Spatial Uncertainty-Aware Semi-Supervised Crowd Counting [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Meng_Spatial_Uncertainty-Aware_Semi-Supervised_Crowd_Counting_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2107.13271)][[code|official](https://github.com/smallmax00/SUA_crowd_counting)]

* **MTCP(TNNLS2023)** Multi-Task Credible Pseudo-Label Learning for Semi-Supervised Crowd Counting [[paper link](https://ieeexplore.ieee.org/abstract/document/10040995)][[code|official](https://github.com/ljq2000/MTCP)][`TJU`]

* **OPT(CVPR2023)** Optimal Transport Minimization: Crowd Localization on Density Maps for Semi-Supervised Counting [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Optimal_Transport_Minimization_Crowd_Localization_on_Density_Maps_for_Semi-Supervised_CVPR_2023_paper.html)][[code|official](https://github.com/Elin24/OT-M)]

* **CrowdCLIP(CVPR2023)** CrowdCLIP: Unsupervised Crowd Counting via Vision-Language Model [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Liang_CrowdCLIP_Unsupervised_Crowd_Counting_via_Vision-Language_Model_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2304.04231)][[code|official](https://github.com/dk-liang/CrowdCLIP)][`HUST`]

* **SSCC (ICCV2023)** Calibrating Uncertainty for Semi-Supervised Crowd Counting [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/LI_Calibrating_Uncertainty_for_Semi-Supervised_Crowd_Counting_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2308.09887)][`Stony Brook University`]


### ▶for 3D Hand-Object

* **Semi-Hand-Object(CVPR2021)** Semi-Supervised 3D Hand-Object Poses Estimation With Interactions in Time [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Semi-Supervised_3D_Hand-Object_Poses_Estimation_With_Interactions_in_Time_CVPR_2021_paper.html)][[arxiv link](http://arxiv.org/abs/2106.05266)][[project link](https://stevenlsw.github.io/Semi-Hand-Object/)][[code|official](https://github.com/stevenlsw/Semi-Hand-Object)][trained on `HO3D` dataset, `UC San Diego` and `NVIDIA`]

* **S2Contact(ECCV2022)** S2Contact: Graph-based Network for 3D Hand-Object Contact Estimation with Semi-Supervised Learning [[paper link]](https://link.springer.com/chapter/10.1007/978-3-031-19769-7_33)][[arxiv link](https://arxiv.org/abs/2208.00874)][[project link](https://eldentse.github.io/s2contact/)][[code|official](https://github.com/eldentse/s2contact)][`University of Birmingham, UNIST, SUSTech`]


### ▶for 3D Reconstruction

* **SSP3D(ECCV2022)** Semi-supervised Single-View 3D Reconstruction via Prototype Shape Priors [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19769-7_31)][[arxiv link](https://arxiv.org/abs/2209.15383)][[code|official](https://github.com/ChenHsing/SSP3D)][[`Zhen Xing`](https://chenhsing.github.io/), on datasets `ShapeNet` and `Pix3D`]


---

## Papers (Self-Supervised Learning or Unsupervised Learning)

### ▶for Image Classification

* 👍**SimCLR (ICML2020)** A Simple Framework for Contrastive Learning of Visual Representations [[paper link](http://proceedings.mlr.press/v119/chen20j.html)][[paperswithcode link](https://paperswithcode.com/paper/a-simple-framework-for-contrastive-learning)][[code|official](https://github.com/google-research/simclr)][[official blog](https://blog.research.google/2020/04/advancing-self-supervised-and-semi.html)][`Geoffrey Hinton`, `Contrastive Learning`, `Pre-training`, `Self-Supervised Learning`]

* 👍**MoCo (CVPR2020)** Momentum Contrast for Unsupervised Visual Representation Learning [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html)][[arxiv link](http://arxiv.org/abs/1911.05722)][[code|official](https://github.com/facebookresearch/moco)][`Kaiming He + Ross Girshick`, `Contrastive Learning`, `Pre-training`, `Self-Supervised Learning`]

* 👍**BYOL (NIPS2020)** Bootstrap your own latent: A new approach to self-supervised Learning [[paper link](https://papers.nips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)][[arxiv link](https://arxiv.org/abs/2006.07733)][[code|official](https://github.com/deepmind/deepmind-research/tree/master/byol)][`Contrastive Learning`, `Pre-training`, `Self-Supervised Learning`]

* 👍**SwAV (NIPS2020)** Unsupervised Learning of Visual Features by Contrasting Cluster Assignments [[paper link](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)][[arxiv link](https://arxiv.org/abs/2006.09882)]
[[code|official](https://github.com/facebookresearch/swav)][including `contrastive learning`]

* 👍👍**DINO (ICCV2021)** Emerging Properties in Self-Supervised Vision Transformers [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)][`ViT-based`, `a form of self-distillation with no labels`, `self-supervised pre-training`]

* **MoCo-v3(ICCV2021)** An Empirical Study of Training Self-Supervised Vision Transformers [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_An_Empirical_Study_of_Training_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)][`ViT-based`, `self-supervised pre-training`]

* 👍**SimSiam (CVPR2021)** Exploring Simple Siamese Representation Learning [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2011.10566)][[code|official](https://github.com/facebookresearch/simsiam)][`Kaiming He`, `Contrastive Learning`, `Pre-training`, `Self-Supervised Learning`]

* 👍**MAE (CVPR2022)** Masked Autoencoders Are Scalable Vision Learners [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)][`ViT-based`, `FAIR`, `He Kaiming`， `It reconstructs the original signal given its partial observation`, `self-supervised pre-training`]
