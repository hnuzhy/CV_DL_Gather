# Domain Adaptation

## Defination

`Domain Adaptation` belongs to `Semi-supervised` or `Un-supervised Learning` / `Transfer Learning` / `Few-shot Learning`. We especially focus on domain adaptative object detection for building robust object detection methods in real application.

## Experts

[[Mingsheng Long](http://ise.thss.tsinghua.edu.cn/~mlong/)] 

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## Datasets

* [NightCity(TIP2021)](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html): Night-time Scene Parsing with a Large Real Dataset


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## Materials

* [(zhihu) 【目标检测与域适应】论文及代码整理](https://zhuanlan.zhihu.com/p/371721493)
* [(github) Unsupervised Domain Adaptation Papers and Code](https://github.com/barebell/DA)
* [(github) A collection of AWESOME things about domian adaptation](https://github.com/zhaoxin94/awesome-domain-adaptation)
* [(github) Best transfer learning and domain adaptation resources (papers, tutorials, datasets, etc.)](https://github.com/artix41/awesome-transfer-learning)
* [(github) Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)
* [(github) (YOLO-Seg) YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://github.com/WongKinYiu/yolov7/tree/u7/seg)

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## Papers

### ① ⭐⭐⭐Domain Adaptation for Image Classification

* ❤**Model Evaluation(CVPR2021)** Are Labels Necessary for Classifier Accuracy Evaluation?(测试集没有标签，可以拿来测试模型吗？) [[arxiv link](https://arxiv.org/abs/2007.02915)][[CSDN blog](https://zhuanlan.zhihu.com/p/328686799)]

* ❤**PCS-FUDA(CVPR2021)** Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation [[arxiv link](https://arxiv.org/pdf/2103.16765.pdf)][[project link](http://xyue.io/pcs-fuda/)][[codes|official PyTorch](https://github.com/zhengzangw/PCS-FUDA)]

* ❤**SHOT++(TPAMI2021)** Source Data-Absent Unsupervised Domain Adaptation Through Hypothesis Transfer and Labeling Transfer [[paper link](https://ieeexplore.ieee.org/abstract/document/9512429)][[codes|official](https://github.com/tim-learn/SHOT-plus)]

* **PTMDA(TIP2022)** Multi-Source Unsupervised Domain Adaptation via Pseudo Target Domain [[paper link](https://ieeexplore.ieee.org/abstract/document/9720154)]

* **DINE(CVPR2022)** DINE: Domain Adaptation From Single and Multiple Black-Box Predictors [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Liang_DINE_Domain_Adaptation_From_Single_and_Multiple_Black-Box_Predictors_CVPR_2022_paper.html)][[codes|official](https://github.com/tim-learn/DINE/)]



### ② ⭐⭐Domain Adaptation for Object Detection

* ❤**DA-FasterRCNN(CVPR2018)(Baseline&Milestone)** Domain Adaptive Faster R-CNN for Object Detection in the Wild [[arxiv link](https://arxiv.org/abs/1803.03243)][[paper link](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.html)][[codes|official Caffe](https://github.com/yuhuayc/da-faster-rcnn)][[Zhihu blog](https://zhuanlan.zhihu.com/p/371721493)]

* **SCL(arxiv2019)** SCL: Towards Accurate Domain Adaptive Object Detection via Gradient Detach Based Stacked Complementary Losses [[paper link](https://arxiv.org/abs/1911.02559)]
[[code|official](https://github.com/harsh-99/SCL)]

* **MAF(ICCV2019)** Multi-adversarial Faster-RCNN for Unrestricted Object Detection [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Multi-Adversarial_Faster-RCNN_for_Unrestricted_Object_Detection_ICCV_2019_paper.pdf)][`No code`]

* **DM(CVPR2019)** Diversify and Match: A Domain Adaptive Representation Learning Paradigm for Object Detection [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Kim_Diversify_and_Match_A_Domain_Adaptive_Representation_Learning_Paradigm_for_CVPR_2019_paper.html)]

* **Strong-Weak DA(CVPR2019)** Strong-Weak Distribution Alignment for Adaptive Object Detection [[arxiv link](https://arxiv.org/pdf/1812.04798.pdf)][[project link](http://cs-people.bu.edu/keisaito/research/CVPR2019.html)][[codes|official PyTorch](https://github.com/VisionLearningGroup/DA_Detection)]
 
* **MEAA(ACMMM2020)** Domain-Adaptive Object Detection via Uncertainty-Aware Distribution Alignment [[paper link](https://basiclab.lab.nycu.edu.tw/assets/MEAA_MM2020.pdf)][`No code`]

* **(ECCV2020)** YOLO in the Dark - Domain Adaptation Method for Merging Multiple Models [[paper link](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660341.pdf)]

* **ATF(ECCV2020)** Domain Adaptive Object Detection via Asymmetric Tri-Way Faster-RCNN [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58586-0_19)][`No code`]

* **DA-FCOS(ECCV2020)** One-Shot Unsupervised Cross-Domain Detection [[paper link](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610715.pdf)]

* **CDRA(CVPR2020)** Exploring Categorical Regularization for Domain Adaptive Object Detection[[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Exploring_Categorical_Regularization_for_Domain_Adaptive_Object_Detection_CVPR_2020_paper.html)][[code|official](https://github.com/Megvii-Nanjing/CR-DA-DET)]

* **HTCN(CVPR2020)** Harmonizing Transferability and Discriminability for Adapting Object Detectors [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Harmonizing_Transferability_and_Discriminability_for_Adapting_Object_Detectors_CVPR_2020_paper.html)][[codes|official PyTorch](https://github.com/chaoqichen/HTCN)][[CSDN blog](https://blog.csdn.net/moutain9426/article/details/120587123)]

* **CRDA(CVPR2020)** Exploring Categorical Regularization for Domain Adaptive Object Detection [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Exploring_Categorical_Regularization_for_Domain_Adaptive_Object_Detection_CVPR_2020_paper.pdf)][[codes|official PyTorch](https://github.com/Megvii-Nanjing/CR-DA-DET)]

* **HTCN(CVPR2020)** Harmonizing Transferability and Discriminability for Adapting Object Detectors [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Harmonizing_Transferability_and_Discriminability_for_Adapting_Object_Detectors_CVPR_2020_paper.html)]

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

* **MGADA(CVPR2022)** Multi-Granularity Alignment Domain Adaptation for Object Detection [[paper link](https://arxiv.org/abs/2203.16897)][[codes|(not found)](https://github.com/tiankongzhang/MGADA)]

* **TDD(CVPR2022)** Cross Domain Object Detection by Target-Perceived Dual Branch Distillation [[paper link](https://arxiv.org/abs/2205.01291)][[codes|official PyTorch](https://github.com/Feobi1999/TDD)]

* **AT(CVPR2022)** Cross-Domain Adaptive Teacher for Object Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection_CVPR_2022_paper.html)][`No code`]

* ❤**PT(ICML2022)** Learning Domain Adaptive Object Detection with Probabilistic Teacher [[paper link](https://arxiv.org/abs/2206.06293)][[code|official](https://github.com/hikvision-research/ProbabilisticTeacher)][`Probabilistic Teacher`, `Knowledge Distillation Framework`]


### ③ ⭐⭐Domain Adaptation for Semantic Segmentation


* **ProDA(CVPR2021)** Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Prototypical_Pseudo_Label_Denoising_and_Target_Structure_Learning_for_Domain_CVPR_2021_paper.html)][[codes|official PyTorch](https://github.com/microsoft/ProDA)][`Use prototypes to weight pseudo-labels`]

* ❤**DAFormer(CVPR2022)** DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.html)][[codes|official PyTorch](https://github.com/lhoyer/DAFormer)][`Rare Class Sampling (RCS) + Thing-Class ImageNet Feature Distance (FD) + Learning Rate Warmup`]

* **ProCA(ECCV2022)** Prototypical Contrast Adaptation for Domain Adaptive Semantic Segmentation [[paper link](https://arxiv.org/abs/2207.06654)][[codes|official PyTorch](https://github.com/jiangzhengkai/ProCA)][`Prototype to feature contrastive`]

* **SePiCo(arxiv2022)** SePiCo: Semantic-Guided Pixel Contrast for Domain Adaptive Semantic Segmentation [[paper link](https://arxiv.org/abs/2204.08808)][[codes|official PyTorch](https://github.com/BIT-DA/SePiCo)][`Contrastive with centroid, memory band and gaussian`]

* **BiSMAP(arxiv2022)** Bidirectional Self-Training with Multiple Anisotropic Prototypes for Domain Adaptive Semantic Segmentation [[paper link](https://arxiv.org/abs/2204.07730)][[]()][`Use gaussian mixture model as prototypes to generate pseudo-labels`]


