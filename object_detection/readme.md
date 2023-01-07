#  List for public implementation of various algorithms

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

* [(CSDNblog) 理解COCO的评价指标：AP，AP50，AP70，mAP，AP[.50:.05:.95]](https://blog.csdn.net/qq_27095227/article/details/105450470)
* [(github) High-resolution networks (HRNets) for object detection](https://github.com/HRNet/HRNet-Object-Detection)
* [(github) MMDetection: an open source object detection toolbox based on PyTorch by CUHK](https://github.com/open-mmlab/mmdetection)
* [(github) TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
* [(CSDN blog) 目标检测——One-stage和Two-stage的详解](https://blog.csdn.net/gaoyu1253401563/article/details/86485851)
* [(CSDN blog) Anchor-free的目标检测文章](https://blog.csdn.net/qq_33547191/article/details/90548564)
* [(CSDN blog) 目标检测Anchor-free分支：基于关键点的目标检测（最新网络全面超越YOLOv3）](https://blog.csdn.net/qiu931110/article/details/89430747)
* [(CSDN blog) YOLO V4 Tiny改进版来啦！速度294FPS精度不减YOLO V4 Tiny](https://blog.csdn.net/Yong_Qi2015/article/details/109685373)
* [(blog) YOLO V5 is Here! Custom Object Detection Tutorial with YOLO V5](https://pub.towardsai.net/yolo-v5-is-here-custom-object-detection-tutorial-with-yolo-v5-12666ee1774e)
* [(github) Yolo v4, v3 and v2 for Windows and Linux](https://github.com/AlexeyAB/darknet)
* [(github) Darknet & Scaled-YOLOv4 & YOLOv4](https://github.com/pjreddie/darknet)
* [(github) Yolov5 Yolov4 Yolov3 TensorRT Implementation](https://github.com/enazoe/yolo-tensorrt)
* [(github) YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/yolov5)
* [(github) A Faster Pytorch Implementation of Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch)
* [(zhihu) 如何评价YOLOv5？](https://www.zhihu.com/question/399884529)
* [(github) FAIR's research platform for object detection research, implementing popular algorithms like Mask R-CNN and RetinaNet.](https://github.com/facebookresearch/Detectron)


## 5) Papers and Sources Codes

### ▶ Two-stage Anchor based

* **R-FCN(NIPS2016)** R-FCN: Object Detection via Region-based Fully Convolutional Networks [[arxiv link](https://arxiv.org/abs/1605.06409)][[Codes|Caffe&MATLAB(offical)](https://github.com/daijifeng001/R-FCN)][[Codes|Caffe(unoffical)](https://github.com/YuwenXiong/py-R-FCN)]

* **DCN(ICCV2017)** Deformable Convolutional Networks [[arxiv link](https://arxiv.org/abs/1703.06211)][[Codes|MXNet(offical based on R-FCN)](https://github.com/msracver/Deformable-ConvNets)][[Codes|MXNet(unoffical based on R-FCN)](https://github.com/bharatsingh430/Deformable-ConvNets)]

* **MaskRCNN(ICCV2017)** Mask R-CNN [[paper link](https://arxiv.org/abs/1703.06870)][[codes|official](https://github.com/matterport/Mask_RCNN)]

### ▶ One-stage Anchor based

* **RetinaNet(ICCV2017)** Focal Loss for Dense Object Detection [[arxiv link](https://arxiv.org/abs/1708.02002)][[Codes|PyTorch(unoffical)](https://github.com/yhenon/pytorch-retinanet)][[[Codes|Keras(unoffical)](https://github.com/fizyr/keras-retinanet)]

* **Repulsion(CVPR2018)** Repulsion Loss: Detecting Pedestrians in a Crowd [[arxiv link](https://arxiv.org/abs/1711.07752)][[Codes|PyTorch(unoffical using SSD)](https://github.com/bailvwangzi/repulsion_loss_ssd)][[Codes|PyTorch(unoffical using RetinaNet)](https://github.com/rainofmine/Repulsion_Loss)][[CSDN blog](https://blog.csdn.net/gbyy42299/article/details/83956648)]


### ▶ One-stage Anchor free

* **CornerNet(ECCV2018)** CornerNet: Detecting Objects as Paired Keypoints [[arxiv link](https://arxiv.org/abs/1808.01244)][[Codes|PyTorch(offical)](https://github.com/princeton-vl/CornerNet)][[Codes|PyTorch(offical CornerNet-Lite)](https://github.com/princeton-vl/CornerNet-Lite)]

* ❤ **CenterNet(arxiv2019)** Objects as Points [[arxiv link](https://arxiv.org/abs/1904.07850)][[Codes|PyTorch(offical)](https://github.com/xingyizhou/CenterNet)]

* **CenterNet(arxiv2019)** CenterNet: Keypoint Triplets for Object Detection [[arxiv link](https://arxiv.org/abs/1904.07850)][[Codes|PyTorch(offical)](https://github.com/Duankaiwen/CenterNet)]

* **FCOS(ICCV2019)** FCOS: Fully Convolutional One-Stage Object Detection [[arxiv link](https://arxiv.org/abs/1904.01355)][[Codes|PyTorch_MASK_RCNN(offical)](https://github.com/tianzhi0549/FCOS)][[Codes|PyTorch(unoffical improved)](https://github.com/yqyao/FCOS_PLUS)][[Codes|PyTorch(unoffical using HRNet as backbone)](https://github.com/HRNet/HRNet-FCOS)][[blog_zhihu](https://zhuanlan.zhihu.com/p/63868458)]

* **VFNet(arxiv2020)** VarifocalNet: An IoU-aware Dense Object Detector [[arxiv link](https://arxiv.org/abs/2008.13367)][[Codes|offical with MMDetection & PyTorch](https://github.com/hyz-xmaster/VarifocalNet)]


### ▶ Detection Transformer (DETR, fully end-to-end)

* ❤ **DEtection TRansformer(DETR) (ECCV2020 BestPaper)** End-to-End Object Detection with Transformers [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58452-8_13)][[codes|official](https://github.com/facebookresearch/detr)][[bilibili paper reading video](https://www.bilibili.com/video/BV133411m7VP/)]

* ❤ **DeformableDETR (ICLR2021 OralPaper)** Deformable DETR: Deformable Transformers for End-to-End Object Detection [[paper link](https://arxiv.org/abs/2010.04159)] [[codes|official](https://github.com/fundamentalvision/Deformable-DETR)][[bilibili paper reading video](https://www.bilibili.com/video/BV133411m7VP/)]

* **DAB-DETR (ICLR2022)** DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR [[paper link](https://arxiv.org/abs/2201.12329)][[codes|official](https://github.com/SlongLiu/DAB-DETR)]

* **DN-DETR (CVPR2022 OralPaper)** DN-DETR: Accelerate DETR Training by Introducing Query DeNoising [[paper link]((https://openaccess.thecvf.com/content/CVPR2022/html/Li_DN-DETR_Accelerate_DETR_Training_by_Introducing_Query_DeNoising_CVPR_2022_paper.html))] [[codes|official](https://github.com/IDEA-Research/DN-DETR)]

