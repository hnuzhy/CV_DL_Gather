#  List for public implementation of various algorithms

## 1) Pubilc Datasets and Challenges

* [CrowdHuman Dataset](http://www.crowdhuman.org/download.html)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/) [[github - CityscapesScripts](https://github.com/mcordts/cityscapesScripts)]



## 2) Pioneers and Experts

[👍Martin Danelljan](https://martin-danelljan.github.io/)


## 3) Blogs and Videos

* [(github) High-resolution networks (HRNets) for object detection](https://github.com/HRNet/HRNet-Object-Detection)
* [(github) MMDetection: an open source object detection toolbox based on PyTorch by CUHK](https://github.com/open-mmlab/mmdetection)
* [(github) TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
* [(CSDN blog) 目标检测——One-stage和Two-stage的详解](https://blog.csdn.net/gaoyu1253401563/article/details/86485851)
* [(CSDN blog) Anchor-free的目标检测文章](https://blog.csdn.net/qq_33547191/article/details/90548564)
* [(CSDN blog) 目标检测Anchor-free分支：基于关键点的目标检测（最新网络全面超越YOLOv3）](https://blog.csdn.net/qiu931110/article/details/89430747)


## 4) Papers and Sources Codes

### ▶ Two-stage Anchor based

* **R-FCN(NIPS2016)** R-FCN: Object Detection via Region-based Fully Convolutional Networks [[arxiv link](https://arxiv.org/abs/1605.06409)][[Codes|Caffe&MATLAB(offical)](https://github.com/daijifeng001/R-FCN)][[Codes|Caffe(unoffical)](https://github.com/YuwenXiong/py-R-FCN)]

* **DCN(ICCV2017)** Deformable Convolutional Networks [[arxiv link](https://arxiv.org/abs/1703.06211)][[Codes|MXNet(offical based on R-FCN)](https://github.com/msracver/Deformable-ConvNets)][[Codes|MXNet(unoffical based on R-FCN)](https://github.com/bharatsingh430/Deformable-ConvNets)]





### ▶ One-stage Anchor based

* **RetinaNet(ICCV2017)** Focal Loss for Dense Object Detection [[arxiv link](https://arxiv.org/abs/1708.02002)][[Codes|PyTorch(unoffical)](https://github.com/yhenon/pytorch-retinanet)][[[Codes|Keras(unoffical)](https://github.com/fizyr/keras-retinanet)]

* **Repulsion(CVPR2018)** Repulsion Loss: Detecting Pedestrians in a Crowd [[arxiv link](https://arxiv.org/abs/1711.07752)][[Codes|PyTorch(unoffical using SSD)](https://github.com/bailvwangzi/repulsion_loss_ssd)][[Codes|PyTorch(unoffical using RetinaNet)](https://github.com/rainofmine/Repulsion_Loss)][[CSDN blog](https://blog.csdn.net/gbyy42299/article/details/83956648)]



### ▶ One-stage Anchor free

* **CornerNet(ECCV2018)** CornerNet: Detecting Objects as Paired Keypoints [[arxiv link](https://arxiv.org/abs/1808.01244)][[Codes|PyTorch(offical)](https://github.com/princeton-vl/CornerNet)][[Codes|PyTorch(offical CornerNet-Lite)](https://github.com/princeton-vl/CornerNet-Lite)]

* **CenterNet(arxiv2019)** Objects as Points [[arxiv link](https://arxiv.org/abs/1904.07850)][[Codes|PyTorch(offical)](https://github.com/xingyizhou/CenterNet)]

* **CenterNet(arxiv2019)** CenterNet: Keypoint Triplets for Object Detection [[arxiv link](https://arxiv.org/abs/1904.07850)][[Codes|PyTorch(offical)](https://github.com/Duankaiwen/CenterNet)]

* **FCOS(ICCV2019)** FCOS: Fully Convolutional One-Stage Object Detection [[arxiv link](https://arxiv.org/abs/1904.01355)][[Codes|PyTorch_MASK_RCNN(offical)](https://github.com/tianzhi0549/FCOS)][[Codes|PyTorch(unoffical improved)](https://github.com/yqyao/FCOS_PLUS)][[Codes|PyTorch(unoffical using HRNet as backbone)](https://github.com/HRNet/HRNet-FCOS)][[blog_zhihu](https://zhuanlan.zhihu.com/p/63868458)]

* **VFNet(arxiv2020)** VarifocalNet: An IoU-aware Dense Object Detector [[arxiv link](https://arxiv.org/abs/2008.13367)][[Codes|offical with MMDetection & PyTorch](https://github.com/hyz-xmaster/VarifocalNet)]

