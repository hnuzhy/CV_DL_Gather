# Contents

* **[1) Pubilc Datasets and Challenges](#1-Pubilc-Datasets-and-Challenges)**
* **[2) Annotation Tools](#2-Annotation-Tools)**
* **[3) Pioneers and Experts](#3-Pioneers-and-Experts)**
* **[4) Blogs and Videos](#4-Blogs-and-Videos)**
  * **[▶ GitHub](#-GitHub)**
  * **[▶ CSDN or zhihu](#-CSDN-or-zhihu)**
* **[5) Papers and Sources Codes](#5-Papers-and-Sources-Codes)**
  * **[▶ Related Survey](#-Related-Survey)**
  * **[▶ Two-stage Anchor based](#-Two-stage-Anchor-based)**
  * **[▶ One-stage Anchor based](#-One-stage-Anchor-based)**
  * **[▶ One-stage Anchor free](#-One-stage-Anchor-free)**
  * **[▶ Detection Transformer (DETR)](#-Detection-Transformer-DETR-fully-end-to-end)**
  * **[▶ YOLO Series Algorithms](#-YOLO-Series-Algorithms)**


## 1) Pubilc Datasets and Challenges

* [CrowdHuman Dataset (A Benchmark for Detecting Human in a Crowd)(拥挤人群人体检测)](http://www.crowdhuman.org/download.html)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/) [[github - CityscapesScripts](https://github.com/mcordts/cityscapesScripts)]
* [TT100K (Traffic-Sign Detection and Classification in the Wild)(中国交通信号标志数据集)](https://cg.cs.tsinghua.edu.cn/traffic-sign/)

## 2) Annotation Tools

* [CSAILVision/LabelMeAnnotationTool](https://github.com/CSAILVision/LabelMeAnnotationTool) [Source code for the LabelMe annotation tool.]
* [tzutalin/LabelImg](https://github.com/tzutalin/labelImg) [🖍️ LabelImg is a graphical image annotation tool and label object bounding boxes in images]
* [wkentaro/Labelme](https://github.com/wkentaro/labelme) [Image Polygonal Annotation with Python (polygon, rectangle, circle, line, point and image-level flag annotation)]
* [openvinotoolkit/CVAT](https://github.com/openvinotoolkit/cvat) [A Powerful and efficient Computer Vision Annotation Tool]
* [Ericsson/EVA](https://github.com/Ericsson/eva) [A web-based tool for efficient annotation of videos and image sequences and has an additional tracking capabilities]

## 3) Pioneers and Experts

[👍Martin Danelljan](https://martin-danelljan.github.io/)


## 4) Blogs and Videos

### ▶ GitHub
* [(github) High-resolution networks (HRNets) for object detection](https://github.com/HRNet/HRNet-Object-Detection)
* [(github) MMDetection: an open source object detection toolbox based on PyTorch by CUHK](https://github.com/open-mmlab/mmdetection)
* [(github) TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
* [(github) Yolov5 Yolov4 Yolov3 TensorRT Implementation](https://github.com/enazoe/yolo-tensorrt)
* [(github) A Faster Pytorch Implementation of Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch)
* [(github) FAIR's research platform for object detection research, implementing popular algorithms](https://github.com/facebookresearch/Detectron)
* [(github) Awesome Detection Transformer for Computer Vision (CV) (by `IDEA-Research`)](https://github.com/IDEA-Research/awesome-detection-transformer)
* [(github) detrex is a research platform for Transformer-based Instance Recognition algorithms (by `IDEA-Research`)](https://github.com/IDEA-Research/detrex)
* [(github) (yolo-gradcam) yolo model with gradcam visual. 即插即用,不需要对源码进行任何修改!](https://github.com/z1069614715/objectdetection_script/tree/master/yolo-gradcam)
* 👍[(github) A collection of some awesome public YOLO object detection series projects](https://github.com/sjinzh/awesome-yolo-object-detection)

### ▶ CSDN or zhihu
* [(CSDN blog) 理解COCO的评价指标：AP，AP50，AP70，mAP，AP[.50:.05:.95]](https://blog.csdn.net/qq_27095227/article/details/105450470)
* [(CSDN blog) 目标检测——One-stage和Two-stage的详解](https://blog.csdn.net/gaoyu1253401563/article/details/86485851)
* [(CSDN blog) Anchor-free的目标检测文章](https://blog.csdn.net/qq_33547191/article/details/90548564)
* [(CSDN blog) 目标检测Anchor-free分支：基于关键点的目标检测（最新网络全面超越YOLOv3）](https://blog.csdn.net/qiu931110/article/details/89430747)
* [(CSDN blog) YOLO V4 Tiny改进版来啦！速度294FPS精度不减YOLO V4 Tiny](https://blog.csdn.net/Yong_Qi2015/article/details/109685373)
* [(blog) Custom Object Detection Tutorial with YOLO V5](https://pub.towardsai.net/yolo-v5-is-here-custom-object-detection-tutorial-with-yolo-v5-12666ee1774e)
* [(zhihu) 如何评价YOLOv5？](https://www.zhihu.com/question/399884529)


## 5) Papers and Sources Codes

### ▶ Related Survey

* **(ACM Computing Surveys 2022.09)** Transformers in Vision: A Survey [[paper link](https://dl.acm.org/doi/abs/10.1145/3505244)][[arxiv link (2021.01)](https://arxiv.org/abs/2101.01169)][`Australian`, `USA`]

* **(ACM Computing Surveys 2022.12)** Efficient Transformers: A Survey [[paper link](https://dl.acm.org/doi/abs/10.1145/3530811)][[arxiv link (2020.09)](https://arxiv.org/abs/2009.06732)][`Google`]

* **(AI Open 2022)** A Survey of Transformers [[paper link](https://www.sciencedirect.com/science/article/pii/S2666651022000146)][[arxiv link (2021.06](https://arxiv.org/abs/2106.04554)][`FDU`]

* **(TPAMI 2022)** A Survey on Vision Transformer [[paper link](https://ieeexplore.ieee.org/abstract/document/9716741)][[arxiv link (2020.12)](https://arxiv.org/abs/2012.12556)][`Huawei`, `PKU`, `Dacheng Tao`]


### ▶ Two-stage Anchor based

* ❤**R-FCN(NIPS2016)** R-FCN: Object Detection via Region-based Fully Convolutional Networks [[arxiv link](https://arxiv.org/abs/1605.06409)][[Codes|Caffe&MATLAB(offical)](https://github.com/daijifeng001/R-FCN)][[Codes|Caffe(unoffical)](https://github.com/YuwenXiong/py-R-FCN)]

* **DCN(ICCV2017)** Deformable Convolutional Networks [[arxiv link](https://arxiv.org/abs/1703.06211)][[Codes|MXNet(offical based on R-FCN)](https://github.com/msracver/Deformable-ConvNets)][[Codes|MXNet(unoffical based on R-FCN)](https://github.com/bharatsingh430/Deformable-ConvNets)]

* ❤**MaskRCNN(ICCV2017)** Mask R-CNN [[paper link](https://arxiv.org/abs/1703.06870)][[codes|official](https://github.com/matterport/Mask_RCNN)]

### ▶ One-stage Anchor based

* **RetinaNet(ICCV2017)** Focal Loss for Dense Object Detection [[arxiv link](https://arxiv.org/abs/1708.02002)][[Codes|PyTorch(unoffical)](https://github.com/yhenon/pytorch-retinanet)][[[Codes|Keras(unoffical)](https://github.com/fizyr/keras-retinanet)]

* **Repulsion(CVPR2018)** Repulsion Loss: Detecting Pedestrians in a Crowd [[arxiv link](https://arxiv.org/abs/1711.07752)][[Codes|PyTorch(unoffical using SSD)](https://github.com/bailvwangzi/repulsion_loss_ssd)][[Codes|PyTorch(unoffical using RetinaNet)](https://github.com/rainofmine/Repulsion_Loss)][[CSDN blog](https://blog.csdn.net/gbyy42299/article/details/83956648)]


### ▶ One-stage Anchor free

* **CornerNet(ECCV2018)** CornerNet: Detecting Objects as Paired Keypoints [[arxiv link](https://arxiv.org/abs/1808.01244)][[Codes|PyTorch(offical)](https://github.com/princeton-vl/CornerNet)][[Codes|PyTorch(offical CornerNet-Lite)](https://github.com/princeton-vl/CornerNet-Lite)]

* ❤**CenterNet(arxiv2019)** Objects as Points [[arxiv link](https://arxiv.org/abs/1904.07850)][[Codes|PyTorch(offical)](https://github.com/xingyizhou/CenterNet)]

* **CenterNet(arxiv2019)** CenterNet: Keypoint Triplets for Object Detection [[arxiv link](https://arxiv.org/abs/1904.07850)][[Codes|PyTorch(offical)](https://github.com/Duankaiwen/CenterNet)]

* ❤**FCOS(ICCV2019)** FCOS: Fully Convolutional One-Stage Object Detection [[arxiv link](https://arxiv.org/abs/1904.01355)][[Codes|PyTorch_MASK_RCNN(offical)](https://github.com/tianzhi0549/FCOS)][[Codes|PyTorch(unoffical improved)](https://github.com/yqyao/FCOS_PLUS)][[Codes|PyTorch(unoffical using HRNet as backbone)](https://github.com/HRNet/HRNet-FCOS)][[blog_zhihu](https://zhuanlan.zhihu.com/p/63868458)]

* **VFNet(arxiv2020)** VarifocalNet: An IoU-aware Dense Object Detector [[arxiv link](https://arxiv.org/abs/2008.13367)][[Codes|offical with MMDetection & PyTorch](https://github.com/hyz-xmaster/VarifocalNet)]

* 👍**ATSS(CVPR2020 Oral, Best Paper Nomination)** Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection [[paper link](http://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.html)][[arxiv link](https://arxiv.org/abs/1912.02424)][[code|official](https://github.com/sfzhang15/ATSS)][[CSDN blog](https://blog.csdn.net/u014380165/article/details/103795048)][[`Shifeng Zhang`](http://www.cbsr.ia.ac.cn/users/sfzhang/)]

* **GFLV1 (NIPS2020)** Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection [[paper link](https://proceedings.neurips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html)][[arxiv link](https://arxiv.org/abs/2006.04388)][[code|official](https://github.com/implus/GFocal)][[`Li Xiang`](https://implus.github.io/), based on `ATSS `]

* **GFLV2 (CVPR2021)** Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.html)][[arxiv link](http://arxiv.org/abs/2011.12885)][[code|official](https://github.com/implus/GFocalV2)][[`Li Xiang`](https://implus.github.io/), based on `ATSS `]

* 👍**TOOD(ICCV2021 Oral)** TOOD: Task-aligned One-stage Object Detection [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2108.07755)][[project link](https://fcjian.github.io/tood/)][[code|official](https://github.com/fcjian/TOOD)]

* **GFL(QFL + DFL)(TPAMI2022)** Generalized Focal Loss: Towards Efficient Representation Learning for Dense Object Detection [[paper link](https://ieeexplore.ieee.org/abstract/document/9792391/)][[`Li Xiang`](https://implus.github.io/), based on `ATSS `]


### ▶ Detection Transformer (DETR, fully end-to-end)
`DETRs --> Pros: eliminates the hand-designed anchor and NMS components; Cons: slow training convergence and hard-to-optimize queries. Thus, researchers should put their efforts of optimizing transformer-based detectors in accelerating training convergence and reducing optimization difficulty`

* ❤**DEtection TRansformer(DETR) (ECCV2020 BestPaper)** End-to-End Object Detection with Transformers [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58452-8_13)][[codes|official](https://github.com/facebookresearch/detr)][[bilibili paper reading video](https://www.bilibili.com/video/BV133411m7VP/)][[bilibili paper reading video 2](https://www.bilibili.com/video/BV1sx4y1G76p/)][`facebookresearch`]

* ❤**DeformableDETR (ICLR2021 OralPaper)** Deformable DETR: Deformable Transformers for End-to-End Object Detection [[paper link](https://arxiv.org/abs/2010.04159)] [[codes|official](https://github.com/fundamentalvision/Deformable-DETR)][[bilibili paper reading video](https://www.bilibili.com/video/BV133411m7VP/)]

* **DAB-DETR (ICLR2022)** DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR [[paper link](https://arxiv.org/abs/2201.12329)][[codes|official](https://github.com/SlongLiu/DAB-DETR)]

* **DN-DETR (CVPR2022 OralPaper)** DN-DETR: Accelerate DETR Training by Introducing Query DeNoising [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Li_DN-DETR_Accelerate_DETR_Training_by_Introducing_Query_DeNoising_CVPR_2022_paper.html)] [[codes|official](https://github.com/IDEA-Research/DN-DETR)]

* **HDETR (CVPR2023)** DETRs with Hybrid Matching [[arxiv link](https://arxiv.org/abs/2207.13080)][[code|official](https://github.com/HDETR)][[code|official (`H-Deformable-DETR`)](https://github.com/HDETR/H-Deformable-DETR)][[code|official (`H-PETR-3D`)](https://github.com/HDETR/H-PETR-3D)][[code|official (`H-PETR-Pose`)](https://github.com/HDETR/H-PETR-Pose)]

* **DDQ-DETR (CVPR2023)** Dense Distinct Query for End-to-End Object Detection [[arxiv link](https://arxiv.org/abs/2303.12776)][[code|official](https://github.com/jshilong/DDQ)]

* ❤**RT-DETR(arxiv2023)** DETRs Beat YOLOs on Real-time Object Detection [[arxiv link](https://arxiv.org/abs/2304.08069)][[code|official whole-repo](https://github.com/PaddlePaddle/PaddleDetection)][[code|official branch-repo](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr)][[code|reproduced by ultralytics](https://docs.ultralytics.com/models/rtdetr/#overview)][`Baidu`, `PaddlePaddle`, `PaddleDetection`]


### ▶ YOLO Series Algorithms
> *Anchor-base*: `YOLOv2`, `YOLOv3`, `YOLOv4`, `Scaled-YOLOv4`,`YOLOv5`, `YOLOR`, `TPH-YOLOv5`, `YOLOv5-Lite`, `YOLOv7`, `YOLOv8`; *Anchor-Free*: `YOLOv1`, `YOLOX`, `PP-YOLOE`, `YOLOv6`

* ❤**YOLOv1 (CVPR2016)** You Only Look Once: Unified, Real-Time Object Detection [[paper link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)][[project link (darknet)](https://pjreddie.com/darknet/yolo/)][[code|official by darknet](https://github.com/pjreddie/darknet)][`Joseph Redmon`, `Santosh Divvala`, `Ross Girshick`, `Ali Farhadi`]

* ❤**YOLOv2 (CVPR2017)** YOLO9000: Better, Faster, Stronger [[paper link](https://openaccess.thecvf.com/content_cvpr_2017/html/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.html)][[project link (darknet)](https://pjreddie.com/darknet/yolov2/)][[code|unofficial PyTorch](https://github.com/longcw/yolo2-pytorch)][`Joseph Redmon`, `Ali Farhadi`]
  
* **YOLOv3 (arxiv2018.04)** YOLOv3: An Incremental Improvement [[paper link](https://arxiv.org/abs/1804.02767)][[project link (darknet)](https://pjreddie.com/darknet/yolo/)][[code|unofficial PyTorch (github 2018.12)](https://github.com/ultralytics/yolov3)][code|unofficial PyTorch](https://github.com/eriklindernoren/PyTorch-YOLOv3)][paper: `Joseph Redmon`, `Ali Farhadi`][code: `ultralytics`, `Glenn Jocher`]

* **YOLOv4 (arxiv2020.04)** YOLOv4: Optimal Speed and Accuracy of Object Detection [[paper link](https://arxiv.org/abs/2004.10934)][[code|official DarkNet](https://github.com/AlexeyAB/darknet)][[code|official PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)][[code|unofficial PyTorch](https://github.com/Tianxiaomo/pytorch-YOLOv4)][`Alexey Bochkovskiy`, `Chien-Yao Wang`, `Hong-Yuan Mark Liao`, `Taiwan + Intel`, `DarkNet`]

* **Scaled-YOLOv4 (CVPR2021)** Scaled-YOLOv4: Scaling Cross Stage Partial Network [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Scaled-YOLOv4_Scaling_Cross_Stage_Partial_Network_CVPR_2021_paper.html)][[code|official](https://github.com/WongKinYiu/ScaledYOLOv4)][`Chien-Yao Wang`, `Alexey Bochkovskiy`, `Hong-Yuan Mark Liao`, `Taiwan + Intel`]

* ❤**YOLOv5 (github 2020.06)** YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite [[No published paper]()][[code|official by ultralytics](https://github.com/ultralytics/yolov5)][[Docs|official by ultralytics](https://docs.ultralytics.com/)][[author: Glenn Jocher](https://github.com/glenn-jocher)][`ultralytics`, `Glenn Jocher`]

* **YOLOR (arxiv2021.05)** You Only Learn One Representation: Unified Network for Multiple Tasks [[paper link](https://arxiv.org/abs/2105.04206)][[code|official](https://github.com/WongKinYiu/yolor)][`Chien-Yao Wang`, `I-Hau Yeh`, `Hong-Yuan Mark Liao`, `Taiwan`]

* ❤**YOLOX (arxiv2021.07)** YOLOX: Exceeding YOLO Series in 2021 [[paper link](https://arxiv.org/abs/2107.08430)][[code|official by MegEngine-BaseDetection](https://github.com/Megvii-BaseDetection/YOLOX)][[code|official by MegEngine](https://github.com/MegEngine/YOLOX)][[Docs|official](https://yolox.readthedocs.io/en/latest/)][`Megvii`, `Jian Sun`]

* **TPH-YOLOv5 (ICCVW2021)** TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-Captured Scenarios [[paper link](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/html/Zhu_TPH-YOLOv5_Improved_YOLOv5_Based_on_Transformer_Prediction_Head_for_Object_ICCVW_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2108.11539)][[code|official PyTorch](https://github.com/cv516Buaa/tph-yolov5)][`Transformer`, `SwinTransformer`]

* **YOLOv5-Lite (github 2021.08)** YOLOv5-Lite：Lighter, faster and easier to deploy [[No published paper]()][[code|official](https://github.com/ppogg/YOLOv5-Lite)]
  
* **PP-YOLOE (arxiv2022.03)** PP-YOLOE: An evolved version of YOLO [[paper link](https://arxiv.org/abs/2203.16250)][[code|official](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe)][`PaddlePaddle`, `PaddleDetection`, `Baidu`, based on the `TAL (task-aligned assignment loss)` proposed in `TOOD`]
  
* **YOLOv6 (arxiv2022.09)** YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications [[paper link](https://arxiv.org/abs/2209.02976)][[code|official by meituan](https://github.com/meituan/YOLOv6)][[YOLOv6 v3.0: A Full-Scale Reloading (arxiv2023.01)](https://arxiv.org/abs/2301.05586)][`meituan`] 

* ❤**YOLOv7 (arxiv2022.07) (CVPR2023)** YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors [[paper link](https://arxiv.org/abs/2207.02696)][[code|official](https://github.com/WongKinYiu/yolov7)][[code|official - pose](https://github.com/WongKinYiu/yolov7/tree/pose) , [blog - ipynb](https://github.com/WongKinYiu/yolov7/blob/main/tools/keypoint.ipynb)][[code|official - seg](https://github.com/WongKinYiu/yolov7/tree/mask), [blog - ipynb](https://github.com/WongKinYiu/yolov7/blob/main/tools/instance.ipynb)][[code|YOLOv7-Seg based on YOLOv5](https://github.com/WongKinYiu/yolov7/tree/u7/seg)][[code|YOLOv7-Pose-Estimation by RizwanMunawar](https://github.com/RizwanMunawar/yolov7-pose-estimation)][[code|YOLOv7-Segmentation by RizwanMunawar](https://github.com/RizwanMunawar/yolov7-segmentation)][[blog|YOLOv7-Segmentation by RizwanMunawar](https://medium.com/augmented-startups/train-yolov7-segmentation-on-custom-data-b91237bd2a29)][`Chien-Yao Wang`, `Alexey Bochkovskiy`, `Hong-Yuan Mark Liao`, `Taiwan + Intel`][`re-parameterized module` + `dynamic label assignment` + `trainable bag-of-freebies`][`Following the YOLOv5 project arch`]

* **DAMO-YOLO (arxiv 2022.11)** DAMO-YOLO : A Report on Real-Time Object Detection Design [[arxiv link](https://arxiv.org/abs/2211.15444)][[project link (github)](https://github.com/tinyvision/DAMO-YOLO)]

* ❤**YOLOv8 (github 2023.01)** YOLOv8 🚀 in PyTorch > ONNX > CoreML > TFLite [[No published paper]()][[code|official by ultralytics](https://github.com/ultralytics/ultralytics)][[Docs|official by ultralytics](https://docs.ultralytics.com/)][`ultralytics`, `Glenn Jocher`]

* **YOLO-NAS (github 2023.05)** A Next-Generation, Object Detection Foundational Model generated by Deci’s Neural Architecture Search Technology [[project link](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md)][[super-gradients github](https://github.com/Deci-AI/super-gradients/)][[zhihu blog](https://zhuanlan.zhihu.com/p/628090349)]
