# Contents for Domain Adaptation

* **[1) Defination](#1-Defination)**
* **[2) Pioneers and Experts](#2-Pioneers-and-Experts)**
* **[3) Datasets](#3-Datasets)**
* **[4) Materials](#4-Materials)**
* **[5) Papers](#5-Papers)**
  * **[▶ ① ⭐⭐⭐Domain Adaptation for Image Classification](#-Domain-Adaptation-for-Image-Classification)**
  * **[▶ ② ⭐⭐Domain Adaptation for Object Detection](#-Domain-Adaptation-for-Object-Detection)**
  * **[▶ ③ ⭐⭐Domain Adaptation for Semantic Segmentation](#-Domain-Adaptation-for-Semantic-Segmentation)**
  * **[▶ ④ ⭐Domain Generalization Methods](#-Domain-Generalization-Methods)**
    * ▶ [4.1 Image Classification](#41-Image-Classification)
    * ▶ [4.2 Object Detection](#42-Object-Detection)
    * ▶ [4.3 Semantic Segmentation](#43-Semantic-Segmentation)
    * ▶ [4.4 Keypoints Detection](#44-Keypoints-Detection)
  * **[▶ ⑤ ⭐Source-Free Domain Adaptation Methods](#-Source-Free-Domain-Adaptation-Methods)**
    * ▶ [5.1 Image Classification](#51-Image-Classification)
    * ▶ [5.2 Object Detection](#52-Object-Detection)
    * ▶ [5.3 Semantic Segmentation](#53-Semantic-Segmentation)
    * ▶ [5.4 Keypoints Detection](#54-Keypoints-Detection)
  * **[▶ ⑥ ⭐Domain Adaptation for Other Fields](#-Domain-Adaptation-for-Other-Fields)**


## 1) Defination

`Domain Adaptation` belongs to `Semi-supervised` or `Un-supervised Learning` / `Transfer Learning` / `Few-shot Learning`. We especially focus on domain adaptative object detection for building robust object detection methods in real application.

## 2) Pioneers and Experts

[[Mingsheng Long](http://ise.thss.tsinghua.edu.cn/~mlong/)] 

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## 3) Datasets

* [GTA5 Dataset(ECCV2016)](https://download.visinf.tu-darmstadt.de/data/from_games/): Playing for Data: Ground Truth from Computer Games [[paper link](https://download.visinf.tu-darmstadt.de/data/from_games/data/eccv-2016-richter-playing_for_data.pdf)]
* [CityScapes(CVPR2016)](https://www.cityscapes-dataset.com/login/): The Cityscapes Dataset for Semantic Urban Scene Understanding [[paper link](http://openaccess.thecvf.com/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html)]
* [SYNTHIA-RAND-CITYSCAPES(CVPR2016)](https://synthia-dataset.net/downloads/): The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes [[paper link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Ros_The_SYNTHIA_Dataset_CVPR_2016_paper.html)]
* [Foggy Cityscapes(ECCV2018)](https://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/): Model Adaptation with Synthetic and Real Data for Semantic Dense Foggy Scene Understanding [[paper link (ECCV2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Christos_Sakaridis_Semantic_Scene_Understanding_ECCV_2018_paper.html)][[paper link (IJCV2020)](https://link.springer.com/article/10.1007/s11263-019-01182-4)]
* [NightCity(TIP2021)](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html): Night-time Scene Parsing with a Large Real Dataset [[paper link (journal)](https://ieeexplore.ieee.org/abstract/document/9591338)][[paper link (arxiv)](https://arxiv.org/abs/2003.06883)]
* [Roboflow-100(Arxiv2022)](https://arxiv.org/abs/2211.13523): Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark [[blogs: Roboflow 100](https://blog.roboflow.com/roboflow-100/)]

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## 4) Materials

* [(github) A collection of AWESOME things about domian adaptation](https://github.com/zhaoxin94/awesome-domain-adaptation)
* [(github) A collection of AWESOME things about domian adaptation object detection](https://github.com/zhaoxin94/awesome-domain-adaptation#object-detection)
* [(github) A collection of AWESOME things about domian adaptation semantic segmentation](https://github.com/zhaoxin94/awesome-domain-adaptation#semantic-segmentation)
* [(zhihu) 【目标检测与域适应】论文及代码整理](https://zhuanlan.zhihu.com/p/371721493)
* [(github) Unsupervised Domain Adaptation Papers and Code](https://github.com/barebell/DA)
* [(github) Best transfer learning and domain adaptation resources (papers, tutorials, datasets, etc.)](https://github.com/artix41/awesome-transfer-learning)
* [(github) Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)
* [(github) (YOLO-Seg) YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://github.com/WongKinYiu/yolov7/tree/u7/seg)


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## 5) Papers

`*********************************`

### ① ⭐⭐⭐Domain Adaptation for Image Classification

* ❤**Model Evaluation(CVPR2021)** Are Labels Necessary for Classifier Accuracy Evaluation?(测试集没有标签，可以拿来测试模型吗？) [[arxiv link](https://arxiv.org/abs/2007.02915)][[CSDN blog](https://zhuanlan.zhihu.com/p/328686799)]

* ❤**PCS-FUDA(CVPR2021)** Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation [[arxiv link](https://arxiv.org/pdf/2103.16765.pdf)][[project link](http://xyue.io/pcs-fuda/)][[codes|official PyTorch](https://github.com/zhengzangw/PCS-FUDA)]

* ❤**SHOT++(TPAMI2021)** Source Data-Absent Unsupervised Domain Adaptation Through Hypothesis Transfer and Labeling Transfer [[paper link](https://ieeexplore.ieee.org/abstract/document/9512429)][[codes|official](https://github.com/tim-learn/SHOT-plus)]

* **PTMDA(TIP2022)** Multi-Source Unsupervised Domain Adaptation via Pseudo Target Domain [[paper link](https://ieeexplore.ieee.org/abstract/document/9720154)]

* **DINE(CVPR2022)** DINE: Domain Adaptation From Single and Multiple Black-Box Predictors [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Liang_DINE_Domain_Adaptation_From_Single_and_Multiple_Black-Box_Predictors_CVPR_2022_paper.html)][[codes|official](https://github.com/tim-learn/DINE/)]


`*********************************`

### ② ⭐⭐Domain Adaptation for Object Detection

* ❤**DA-FasterRCNN(CVPR2018)(Baseline & Milestone)** Domain Adaptive Faster R-CNN for Object Detection in the Wild [[arxiv link](https://arxiv.org/abs/1803.03243)][[paper link](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.html)][[codes|official Caffe](https://github.com/yuhuayc/da-faster-rcnn)][[Zhihu blog](https://zhuanlan.zhihu.com/p/371721493)]

* **SCL(arxiv2019)** SCL: Towards Accurate Domain Adaptive Object Detection via Gradient Detach Based Stacked Complementary Losses [[paper link](https://arxiv.org/abs/1911.02559)]
[[code|official](https://github.com/harsh-99/SCL)]

* **MAF(ICCV2019)** Multi-adversarial Faster-RCNN for Unrestricted Object Detection [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Multi-Adversarial_Faster-RCNN_for_Unrestricted_Object_Detection_ICCV_2019_paper.pdf)][`No code`]

* **DM(CVPR2019)** Diversify and Match: A Domain Adaptive Representation Learning Paradigm for Object Detection [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Kim_Diversify_and_Match_A_Domain_Adaptive_Representation_Learning_Paradigm_for_CVPR_2019_paper.html)]

* **Strong-Weak DA(CVPR2019)** Strong-Weak Distribution Alignment for Adaptive Object Detection [[arxiv link](https://arxiv.org/pdf/1812.04798.pdf)][[project link](http://cs-people.bu.edu/keisaito/research/CVPR2019.html)][[codes|official PyTorch](https://github.com/VisionLearningGroup/DA_Detection)]
 
* **MEAA(ACMMM2020)** Domain-Adaptive Object Detection via Uncertainty-Aware Distribution Alignment [[paper link](https://basiclab.lab.nycu.edu.tw/assets/MEAA_MM2020.pdf)][`No code`]

* **(ECCV2020)** YOLO in the Dark: Domain Adaptation Method for Merging Multiple Models [[paper link](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660341.pdf)][`No code`]

* **ATF(ECCV2020)** Domain Adaptive Object Detection via Asymmetric Tri-Way Faster-RCNN [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58586-0_19)][`No code`]

* **DA-FCOS(ECCV2020)** One-Shot Unsupervised Cross-Domain Detection [[paper link](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610715.pdf)]

* **CDRA(CVPR2020)** Exploring Categorical Regularization for Domain Adaptive Object Detection[[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Exploring_Categorical_Regularization_for_Domain_Adaptive_Object_Detection_CVPR_2020_paper.html)][[code|official](https://github.com/Megvii-Nanjing/CR-DA-DET)]

* **HTCN(CVPR2020)** Harmonizing Transferability and Discriminability for Adapting Object Detectors [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Harmonizing_Transferability_and_Discriminability_for_Adapting_Object_Detectors_CVPR_2020_paper.html)][[codes|official PyTorch](https://github.com/chaoqichen/HTCN)][[CSDN blog](https://blog.csdn.net/moutain9426/article/details/120587123)]

* **PA-ATF(TCSVT2021)** Partial Alignment for Object Detection in the Wild [[paper link](https://ieeexplore.ieee.org/abstract/document/9663266/)][`No code`]

* ❤**Divide-and-Merge Spindle Network(DMSN)(ICCV2021)** Multi-Source Domain Adaptation for Object Detection [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Multi-Source_Domain_Adaptation_for_Object_Detection_ICCV_2021_paper.html)]

* ❤**UMT(CVPR2021)** Unbiased Mean Teacher for Cross-domain Object Detection [[arxiv link](https://arxiv.org/abs/2003.00707)][[paper link](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_Unbiased_Mean_Teacher_for_Cross-Domain_Object_Detection_CVPR_2021_paper.pdf)][[codes|official PyTorch](https://github.com/kinredon/umt)]

* **Survey(arxiv2021)** Unsupervised Domain Adaptation of Object Detectors: A Survey [[paper link](https://arxiv.org/pdf/2105.13502.pdf)]

* **MS-DAYOLO(ICIP2021)(YOLOV4)** Multiscale Domain Adaptive YOLO for Cross-Domain Object Detection [[arxiv link](https://arxiv.org/abs/2106.01483)][[csdn blog](https://cloud.tencent.com/developer/article/1843695)]

* **DAYOLO(ACML2021)(YOLOV3)** Domain Adaptive YOLO for One-Stage Cross-Domain Detection [[paper link](https://proceedings.mlr.press/v157/zhang21c.html)]

* **US-DAF(ACMMM2022)** Universal Domain Adaptive Object Detector [[paper link](https://arxiv.org/abs/2207.01756)][`No code`]

* **SCAN(AAAI2022)** SCAN: Cross Domain Object Detection with Semantic Conditioned Adaptation [[paper link](https://www.aaai.org/AAAI22Papers/AAAI-902.LiW.pdf)][[codes|official PyTorch](https://github.com/CityU-AIM-Group/SCAN)]

* **SIGMA(CVPR2022)** SIGMA: Semantic-complete Graph Matching for Domain Adaptive Object Detection [[paper link](https://arxiv.org/abs/2203.06398)][[codes|official PyTorch](https://github.com/CityU-AIM-Group/SIGMA)]

* **TIA(CVPR2022)** Task-specific Inconsistency Alignment for Domain Adaptive Object Detection [[paper link](https://arxiv.org/abs/2203.15345)][[codes|official PyTorch](https://github.com/MCG-NJU/TIA)]

* **TPKP(CVPR2022)** Target-Relevant Knowledge Preservation for Multi-Source Domain Adaptive Object Detection [[paper link](https://arxiv.org/abs/2204.07964)][[codes|(not found)]()]

* **MGADA(CVPR2022)** Multi-Granularity Alignment Domain Adaptation for Object Detection [[paper link](https://arxiv.org/abs/2203.16897)][[codes|(not found)](https://github.com/tiankongzhang/MGADA)][[related journal link](https://arxiv.org/abs/2301.00371)]

* **TDD(CVPR2022)** Cross Domain Object Detection by Target-Perceived Dual Branch Distillation [[paper link](https://arxiv.org/abs/2205.01291)][[codes|official PyTorch](https://github.com/Feobi1999/TDD)]

* **AT(CVPR2022)** Cross-Domain Adaptive Teacher for Object Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection_CVPR_2022_paper.html)][`No code`]

* ❤**PT(ICML2022)** Learning Domain Adaptive Object Detection with Probabilistic Teacher [[paper link](https://arxiv.org/abs/2206.06293)][[code|official](https://github.com/hikvision-research/ProbabilisticTeacher)][`Probabilistic Teacher`, `Knowledge Distillation Framework`]

* **DICN(TPAMI2022)** Dual Instance-Consistent Network for Cross-Domain Object Detection [[paper link](https://ieeexplore.ieee.org/abstract/document/9935311)]

* **DenseTeacher(ECCV2022)** DenseTeacher: Dense Pseudo-Label for Semi-supervised Object Detection [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_3)][[code|official](https://github.com/Megvii-BaseDetection/DenseTeacher)]

* **SSDA-YOLO(CVIU2023)** SSDA-YOLO: Semi-supervised Domain Adaptive YOLO for Cross-Domain Object Detection [[paper link](https://www.sciencedirect.com/science/article/abs/pii/S1077314223000292)][[arxiv link](https://arxiv.org/abs/2211.02213v2)][[code|official](https://github.com/hnuzhy/SSDA-YOLO)]

* **DETR-GA(CVPR2023)** DETR with Additional Global Aggregation for Cross-domain Weakly Supervised Object Detection [[paper link](https://arxiv.org/abs/2304.07082)][`cross-domain weakly supervised object detection (CDWSOD)`]

* **2PCNet(CVPR2023)** 2PCNet: Two-Phase Consistency Training for Day-to-Night Unsupervised Domain Adaptive Object Detection [[arxiv link](https://arxiv.org/abs/2303.13853)][[code|official](https://github.com/mecarill/2pcnet)]

`*********************************`

### ③ ⭐⭐Domain Adaptation for Semantic Segmentation

* **FCNs in the Wild(arxiv2016)** FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation [[paper link](https://arxiv.org/abs/1612.02649)][`both global and category specific adaptation techniques`, `pioneering`]

* **CDA(ICCV2017)** Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes [[paper link](https://openaccess.thecvf.com/content_iccv_2017/html/Zhang_Curriculum_Domain_Adaptation_ICCV_2017_paper.html)][[code|official](https://github.com/YangZhang4065/AdaptationSeg)][`curriculum domain adaptation`] 

* **CyCADA(ICML2018)** CyCADA: Cycle-Consistent Adversarial Domain Adaptation [[paper link](https://proceedings.mlr.press/v80/hoffman18a)][`adversarial training`]

* **AdaptSegNet(CVPR2018)** Learning to Adapt Structured Output Space for Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content_cvpr_2018/html/Tsai_Learning_to_Adapt_CVPR_2018_paper.html)][[code|official](https://github.com/wasidennis/AdaptSegNet)][`adversarial learning`]

* **ADVENT(CVPR2019 oral)** ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.html)][[code|official](https://github.com/valeoai/ADVENT)][`adversarial training`]

* **BDL(CVPR2019)** Bidirectional Learning for Domain Adaptation of Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Bidirectional_Learning_for_Domain_Adaptation_of_Semantic_Segmentation_CVPR_2019_paper.html)][[code|official](https://github.com/liyunsheng13/BDL)][`adversarial training`]

* **TGCF-DA(ICCV2019)** Self-Ensembling With GAN-Based Data Augmentation for Domain Adaptation in Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Choi_Self-Ensembling_With_GAN-Based_Data_Augmentation_for_Domain_Adaptation_in_Semantic_ICCV_2019_paper.html)][`GAN-Based Data Augmentation`]

* **Adapt-Seg(ICCV2019 Oral)** Domain Adaptation for Structured Output via Discriminative Patch Representations [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tsai_Domain_Adaptation_for_Structured_Output_via_Discriminative_Patch_Representations_ICCV_2019_paper.pdf)][[project link](https://www.nec-labs.com/~mas/adapt-seg/adapt-seg.html)][`adversarial learning scheme`]

* **FDA(CVPR2020)** FDA: Fourier Domain Adaptation for Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html)][[code|official](https://github.com/YanchaoYang/FDA)]

* **FADA(ECCV2020)** Classes Matter: A Fine-Grained Adversarial Approach to Cross-Domain Semantic Segmentation [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_38)][[codes|official PyTorch](https://github.com/JDAI-CV/FADA)][`self-training`]

* ❤**ProDA(CVPR2021)** Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Prototypical_Pseudo_Label_Denoising_and_Target_Structure_Learning_for_Domain_CVPR_2021_paper.html)][[codes|official PyTorch](https://github.com/microsoft/ProDA)][`Use prototypes to weight pseudo-labels`]

* **(CVPR2021)** Coarse-To-Fine Domain Adaptive Semantic Segmentation With Photometric Alignment and Category-Center Regularization [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Ma_Coarse-To-Fine_Domain_Adaptive_Semantic_Segmentation_With_Photometric_Alignment_and_Category-Center_CVPR_2021_paper.html)][`self-training`]

* **PixMatch(CVPR2021)** PixMatch: Unsupervised Domain Adaptation via Pixelwise Consistency Training [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Melas-Kyriazi_PixMatch_Unsupervised_Domain_Adaptation_via_Pixelwise_Consistency_Training_CVPR_2021_paper.html)][[codes|official PyTorch](https://github.com/lukemelas/pixmatch)][`self-training`]

* **DA-SAC(CVPR2021)** Self-Supervised Augmentation Consistency for Adapting Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Araslanov_Self-Supervised_Augmentation_Consistency_for_Adapting_Semantic_Segmentation_CVPR_2021_paper.html)][[codes|official PyTorch](https://github.com/visinf/da-sac)][`self-training`]

* ❤**DAFormer(CVPR2022)** DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.html)][[codes|official PyTorch](https://github.com/lhoyer/DAFormer)][`Rare Class Sampling (RCS) + Thing-Class ImageNet Feature Distance (FD) + Learning Rate Warmup`]

* **SimT(CVPR2022)** SimT: Handling Open-Set Noise for Domain Adaptive Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_SimT_Handling_Open-Set_Noise_for_Domain_Adaptive_Semantic_Segmentation_CVPR_2022_paper.html)][[codes|official PyTorch](https://github.com/CityU-AIM-Group/SimT)][`self-training`]

* **CPSL(CVPR2022)** Class-Balanced Pixel-Level Self-Labeling for Domain Adaptive Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Class-Balanced_Pixel-Level_Self-Labeling_for_Domain_Adaptive_Semantic_Segmentation_CVPR_2022_paper.html)][[codes|official PyTorch](https://github.com/lslrh/CPSL)][`self-training`]

* ❤**ProCA(ECCV2022)** Prototypical Contrast Adaptation for Domain Adaptive Semantic Segmentation [[paper link](https://arxiv.org/abs/2207.06654)][[codes|official PyTorch](https://github.com/jiangzhengkai/ProCA)][`Prototype to feature contrastive`]

* ❤**HRDA(ECCV2022)** HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation  [[paper link](https://arxiv.org/pdf/2204.13132)][[codes|official PyTorch](https://github.com/lhoyer/HRDA)][`Based on DAFormer`]

* **DecoupleNet(ECCV2022)** DecoupleNet: Decoupled Network for Domain Adaptive Semantic Segmentation [[paper link](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930362.pdf)][[codes|official PyTorch](https://github.com/dvlab-research/DecoupleNet)][`self-training`]

* **DDB(NIPS2022)** Deliberated Domain Bridging for Domain Adaptive Semantic Segmentation [[paper link](https://arxiv.org/abs/2209.07695)][[codes|official PyTorch](https://github.com/xiaoachen98/DDB)][`self-training`]

* **BiSMAP(ACMMM2022)** Bidirectional Self-Training with Multiple Anisotropic Prototypes for Domain Adaptive Semantic Segmentation [[paper link](https://arxiv.org/abs/2204.07730)][`Use gaussian mixture model as prototypes to generate pseudo-labels`]

* **SePiCo(TPAMI2023)** SePiCo: Semantic-Guided Pixel Contrast for Domain Adaptive Semantic Segmentation [[paper link](https://arxiv.org/abs/2204.08808)][[codes|official PyTorch](https://github.com/BIT-DA/SePiCo)][`Contrastive with centroid, memory band and gaussian`]

* **WSDA_semantic(MTA2023)** On exploring weakly supervised domain adaptation strategies for semantic segmentation using synthetic data [[paper link](https://link.springer.com/article/10.1007/s11042-023-14662-0)][[code|official](http://www-vpu.eps.uam.es/publications/WSDA_semantic/)]

`*********************************`

### ④ ⭐Domain Generalization Methods

#### ▶4.1 Image Classification

#### ▶4.2 Object Detection

#### ▶4.3 Semantic Segmentation

* **IBN-Net(ECCV2018)** Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net [[paper link](https://openaccess.thecvf.com/content_ECCV_2018/html/Xingang_Pan_Two_at_Once_ECCV_2018_paper.html)][[codes|official PyTorch](https://github.com/XingangPan/IBN-Net)]

* **SW(Switchable Whitening)(ICCV2019)** Switchable Whitening for Deep Representation Learning [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Pan_Switchable_Whitening_for_Deep_Representation_Learning_ICCV_2019_paper.html)]

* **DRPC(ICCV2019)** Domain Randomization and Pyramid Consistency: Simulation-to-Real Generalization Without Accessing Target Domain Data [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Yue_Domain_Randomization_and_Pyramid_Consistency_Simulation-to-Real_Generalization_Without_Accessing_Target_ICCV_2019_paper.html)][[codes|official PyTorch](https://github.com/xyyue/DRPC)] 

* ❤**GTR-LTR(Global Texture Randomization, Local Texture Randomization)(TIP2021)** Global and Local Texture Randomization for Synthetic-to-Real Semantic Segmentation [[paper link](https://ieeexplore.ieee.org/abstract/document/9489280)][[arxiv link](https://arxiv.org/abs/2108.02376)][[codes|official PyTorch](https://github.com/leolyj/GTR-LTR)][author `leolyj`]

* **FSDR(CVPR2021)** FSDR: Frequency Space Domain Randomization for Domain Generalization [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_FSDR_Frequency_Space_Domain_Randomization_for_Domain_Generalization_CVPR_2021_paper.html)][[codes|official PyTorch](https://github.com/jxhuang0508/FSDR)]

* ❤**RobustNet(CVPR2021 Oral)** RobustNet: Improving Domain Generalization in Urban-Scene Segmentationvia Instance Selective Whitening [[paper link](https://arxiv.org/abs/2103.15597)][[codes|official PyTorch](https://github.com/shachoi/RobustNet)]

* **WildNet(CVPR2022)** WildNet: Learning Domain Generalized Semantic Segmentation From the Wild [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_WildNet_Learning_Domain_Generalized_Semantic_Segmentation_From_the_Wild_CVPR_2022_paper.html)][[codes|official PyTorch](https://github.com/suhyeonlee/WildNet)]

* **SAN-SAW(CVPR2022 Oral)** Semantic-Aware Domain Generalized Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.html)][[codes|official PyTorch](https://github.com/leolyj/SAN-SAW)][author `leolyj`]

* **SHADE(ECCV2022)** Style-Hallucinated Dual Consistency Learning for Domain Generalized Semantic Segmentation [[paper link](https://arxiv.org/pdf/2204.02548.pdf)][[codes|official PyTorch](https://github.com/HeliosZhao/SHADE)][`Style Consistency` and `Retrospection Consistency`]

#### ▶4.4 Keypoints Detection


`*********************************`

### ⑤ ⭐Source-Free Domain Adaptation Methods

#### ▶5.1 Image Classification

* 👍**USFDA(CVPR2020)** Universal Source-Free Domain Adaptation [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Kundu_Universal_Source-Free_Domain_Adaptation_CVPR_2020_paper.html)][[project link](https://sites.google.com/view/usfda-cvpr2020)]

* **DIPE (Domain-Invariant Parameter Exploring)(CVPR2022)** Exploring Domain-Invariant Parameters for Source Free Domain Adaptation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Exploring_Domain-Invariant_Parameters_for_Source_Free_Domain_Adaptation_CVPR_2022_paper.html)]

#### ▶5.2 Object Detection

* 👍**IRG-SFDA(CVPR2023)** Instance Relation Graph Guided Source-Free Domain Adaptive Object Detection [[arxiv link](https://arxiv.org/abs/2203.15793)][[project link](https://viudomain.github.io/irg-sfda-web/)][[code|official](https://github.com/Vibashan/irg-sfda)][`Johns Hopkins University`]

#### ▶5.3 Semantic Segmentation

* **SFDA(MICCAI2020)** Source-Relaxed Domain Adaptation for Image Segmentation [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_48)][[code|official](https://github.com/mathilde-b/SFDA)]

* 👍**SFDA-Seg(CVPR2021)** Source-Free Domain Adaptation for Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Source-Free_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2103.16372)][`only a well-trained source model and an unlabeled target domain dataset are available for adaptation`]


#### ▶5.4 Keypoints Detection

* **(CVPR2021)** Regressive Domain Adaptation for Unsupervised Keypoint Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Jiang_Regressive_Domain_Adaptation_for_Unsupervised_Keypoint_Detection_CVPR_2021_paper.html)][[project link](https://github.com/thuml/Transfer-Learning-Library)][`Single Hand Keypoint Detection Transfer Test`, `Single Human Keypoint Detection Transfer Test`]

`*********************************`

### ⑥ ⭐Domain Adaptation for Other Fields

* **SSDA3D(AAAI2023)** SSDA3D: Semi-supervised Domain Adaptation for 3D Object Detection from Point Cloud [[paper link](https://arxiv.org/abs/2212.02845)][[codes|official (not released)](https://github.com/yinjunbo/SSDA3D)][`Domain Adaptation for 3D Object Detection`]

* **CMOM(WACV2023)** Domain Adaptive Video Semantic Segmentation via Cross-Domain Moving Object Mixing [[paper link](https://arxiv.org/abs/2211.02307)][`Domain Adaptation for Video Semantic Segmentation`]

