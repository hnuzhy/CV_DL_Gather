# Contents

* **[1) Pubilc Datasets and Challenges](#1-Pubilc-Datasets-and-Challenges)**
  * **[⭐For Head Pose Estimation](#For-Head-Pose-Estimation)**
  * **[⭐For Head Detection Only](#For-Head-Detection-Only)**
  * **[⭐For Head Detection or Crowd Counting](#For-Head-Detection-or-Crowd-Counting)**
* **[2) Pioneers and Experts](#2-Pioneers-and-Experts)**
* **[3) Related Materials (Papers, Sources Code, Blogs, Videos and Applications)](#3-Related-Materials-Papers-Sources-Code-Blogs-Videos-and-Applications)**
  * **[▶ Beautify Face](#-Beautify-Face)**
  * **[▶ Eye Gaze Estimation](#-Eye-Gaze-Estimation)**
  * **[▶ Face Alignment](#-Face-Alignment)**
  * **[▶ Face Detection](#-Face-Detection)**
  * **[▶ Face Recognition](#-Face-Recognition)**
  * **[▶ Face Reconstruction (3D)](#-Face-Reconstruction-3D)**
  * **[▶ Head Detector](#-Head-Detector)**
  * **[▶ Head Pose Estimation](#-Head-Pose-Estimation)**


# List of public algorithms and datasets

## 1) Pubilc Datasets and Challenges

### ⭐**For Head Pose Estimation**
* [BIWI RGBD-ID Dataset](http://robotics.dei.unipd.it/reid/index.php): The BIWI RGBD-ID Dataset is a RGB-D dataset of people targeted to long-term people re-identification from RGB-D cameras. It contains 50 training and 56 testing sequences of 50 different people.
* [300W-LP & AFLW2000-3D](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3ddfa/main.htm): 300W-LP has the synthesized large-pose face images from 300W. AFLW2000-3D is the fitted 3D faces of the first 2000 AFLW samples, which can be used for 3D face alignment evaluation.
* [CMU Panoptic Studio Dataset](http://domedb.perception.cs.cmu.edu/index.html): Currently, 480 VGA videos, 31 HD videos, 3D body pose, and calibration data are available. PointCloud DB from 10 Kinects (with corresponding 41 RGB videos) is also available (6+ hours of data). Please refer the official website for details. Dataset paper link [Panoptic studio: A massively multiview system for social interaction capture](https://arxiv.org/pdf/1612.03153.pdf).

### ⭐**For Head Detection Only**
* [HollywoodHead dataset](https://www.di.ens.fr/willow/research/headdetection/): HolleywoodHeads dataset is a head detection datset. HollywoodHeads dataset contains 369846 human heads annotated in 224740 video frames from 21 Hollywood movies.
* [Brainwash dataset](https://exhibits.stanford.edu/data/catalog/sx925dc9385): Brainwash dataset is related for face detection. Brainwash dataset contains 11917 images with 91146 labeled people.
* [SCUT-HEAD-Dataset-Release](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release): SCUT-HEAD is a large-scale head detection dataset, including 4405 images labeld with 111251 heads. The dataset consists of two parts. PartA includes 2000 images sampled from monitor videos of classrooms in an university with 67321 heads annotated. PartB includes 2405 images crawled from Internet with 43930 heads annotated.

### ⭐**For Head Detection or Crowd Counting**
* [ShanghaiTech dataset](https://github.com/desenzhou/ShanghaiTechDataset): Dataset appeared in Single Image Crowd Counting via Multi Column Convolutional Neural Network(MCNN) in CVPR2016. 【情况介绍】：包含标注图片 1198 张，共 330165 人，分为 A 和 B 两个部分，A 包含 482 张图片，均为网络下载的含高度拥挤人群的场景图片，人群数量从 33 到 3139 个不等，训练集包含 300 张图片和测试集包含 182 张图片。B 包含 716 张图片，这些图片的人流场景相对稀疏，拍摄于街道的固定摄像头，群体数量从 12 到 578 不等。训练集包含 400 张图像，测试集包含 316 张图像。
* [UCF-QNRF - A Large Crowd Counting Data Set](https://www.crcv.ucf.edu/data/ucf-qnrf/): It contains 1535 images which are divided into train and test sets of 1201 and 334 images respectively. Paper is published in ECCV2018. 【情况介绍】：这是最新发布的最大人群数据集。它包含 1535 张来自 Flickr、网络搜索和 Hajj 片段的密集人群图像。数据集包含广泛的场景，拥有丰富的视角、照明变化和密度多样性，计数范围从 49 到 12865 不等，这使该数据库更加困难和现实。此外，图像分辨率也很大，因此导致头部尺寸出现大幅变化。
* [UCSD Pedestrian Dataset](http://visal.cs.cityu.edu.hk/downloads/): Video of people on pedestrian walkways at UCSD, and the corresponding motion segmentations. Currently two scenes are available. 【情况介绍】：由 2000 帧监控摄像机拍摄的照片组成，尺寸为 238×158。这个数据集的密度相对较低，每幅图像 11 到 46 人不等，平均约 25 人。在所有帧中，帧 601 到 1400 为训练集，其余帧为测试集。
* [Megvii CrowdHuman](https://www.crowdhuman.org/): CrowdHuman is a benchmark dataset to better evaluate detectors in crowd scenarios. The CrowdHuman dataset is large, rich-annotated and contains high diversity. CrowdHuman contains 15000, 4370 and 5000 images for training, validation, and testing, respectively. There are a total of 470K human instances from train and validation subsets and 23 persons per image, with various kinds of occlusions in the dataset. Each human instance is annotated with a head bounding-box, human visible-region bounding-box and human full-body bounding-box. We hope our dataset will serve as a solid baseline and help promote future research in human detection tasks.


## 2) Pioneers and Experts

[👍Jian Sun](http://www.jiansun.org/) [👍Gang YU](http://www.skicyyu.org/) [👍Yuliang Xiu 修宇亮](https://xiuyuliang.cn/)



## 3) Related Materials (Papers, Sources Code, Blogs, Videos and Applications)

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Beautify Face

#### Materials

* [(github) BeautifyFaceDemo](https://github.com/Guikunzhi/BeautifyFaceDemo)
* [(CSDN blogs) 图像滤镜艺术---换脸算法资源收集](https://blog.csdn.net/scythe666/article/details/81021041)

#### Papers


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Eye Gaze Estimation

#### Materials


#### Papers

* **ETH-XGaze(ECCV2020)** ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation [[arxiv link](https://arxiv.org/abs/2007.15837)][[project link](https://ait.ethz.ch/projects/2020/ETH-XGaze/)][[Codes|PyTorch(official)](https://github.com/xucong-zhang/ETH-XGaze)]

* **EVE(ECCV2020)** Towards End-to-end Video-based Eye-tracking [[arxiv link](https://arxiv.org/abs/2007.13120)][[project link](https://ait.ethz.ch/projects/2020/EVE/)][[Codes|PyTorch(official)](https://github.com/swook/EVE)]

* **❤ MEBOW(CVPR2020)** MEBOW: Monocular Estimation of Body Orientation in the Wild [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Wu_MEBOW_Monocular_Estimation_of_Body_Orientation_in_the_Wild_CVPR_2020_paper.html)][[project link](https://chenyanwu.github.io/MEBOW/)][[codes|official](https://github.com/ChenyanWu/MEBOW)][`COCO-MEBOW, Body Orientation Estimation`]

* **RUDA(CVPR2022)** Generalizing Gaze Estimation With Rotation Consistency [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Bao_Generalizing_Gaze_Estimation_With_Rotation_Consistency_CVPR_2022_paper.html)]

* **❤ GazeOnce/MPSGaze(CVPR2022)** GazeOnce: Real-Time Multi-Person Gaze Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_GazeOnce_Real-Time_Multi-Person_Gaze_Estimation_CVPR_2022_paper.html)][[codes|official](https://github.com/mf-zhang/GazeOnce)][`The MPSGaze is a synthetic dataset (ETH-XGaze + WiderFace) containing full images (instead of only cropped faces) that provides ground truth 3D gaze directions for multiple people in one image.`]

* **❤ GAFA(CVPR2022)** Dynamic 3D Gaze From Afar: Deep Gaze Estimation From Temporal Eye-Head-Body Coordination [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Nonaka_Dynamic_3D_Gaze_From_Afar_Deep_Gaze_Estimation_From_Temporal_CVPR_2022_paper.html)][[project link](https://vision.ist.i.kyoto-u.ac.jp/research/gafa/)][[codes|official](https://github.com/kyotovision-public/dynamic-3d-gaze-from-afar)][`The GAze From Afar (GAFA) dataset consists of surveillance videos of freely moving people with automatically annotated 3D gaze, head, and body orientations.`]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**


### ▶ Face Alignment

#### Materials

* [(jianshu) 人脸关键点对齐](https://www.jianshu.com/p/e4b9317a817f)
* Procrustes Analysis [[CSDN blog](https://blog.csdn.net/u011808673/article/details/80733686)][[wikipedia](https://en.wikipedia.org/wiki/Procrustes_analysis)][[scipy.spatial.procrustes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html)][[github](https://github.com/Ahmer-444/Action-Recognition-ProcrustesAnalysis)]
* [(website) Procrustes Analysis and its application in computer graphaics](https://max.book118.com/html/2017/0307/94565569.shtm)
* [(github) ASM-for-human-face-feature-points-matching](https://github.com/JiangtianPan/ASM-for-human-face-feature-points-matching)
* [(github) align_dataset_mtcnn](https://github.com/davidsandberg/facenet/blob/master/src/align/align_dataset_mtcnn.py)
* [(Website) Face Alignment Across Large Poses: A 3D Solution (official website)](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3ddfa/main.htm)
* [(github) 🔥🔥The pytorch implement of the head pose estimation(yaw,roll,pitch) and emotion detection](https://github.com/WIKI2020/FacePose_pytorch)

#### Papers

* **Dlib(CVPR2014)** One Millisecond Face Alignment with an Ensemble of Regression Trees [[paper link](https://openaccess.thecvf.com/content_cvpr_2014/html/Kazemi_One_Millisecond_Face_2014_CVPR_paper.html)][[codes|official C++](https://github.com/davisking/dlib)][`pip install dlib`]

* **3000FPS(CVPR2014)** Face Alignment at 3000 FPS via Regressing Local Binary Features [[paper link](http://www.cse.psu.edu/~rtc12/CSE586/papers/regr_cvpr14_facealignment.pdf)][[Codes|opencv(offical)](https://github.com/freesouls/face-alignment-at-3000fps)][[Codes|liblinear(unoffical)](https://github.com/jwyang/face-alignment)][[CSDN blog](https://blog.csdn.net/lzb863/article/details/49890369)]

* **3DDFA(CVPR2016)** Face Alignment Across Large Poses: A 3D Solution [[paper link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780392)][[project link](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)][[codes|PyTorch 3DDFA](https://github.com/cleardusk/3DDFA)]

* **FAN(ICCV2017)** How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks) [[paper link](https://www.adrianbulat.com/downloads/FaceAlignment/FaceAlignment.pdf)][[Adrian Bulat](https://www.adrianbulat.com/)][[Codes|PyTorch(offical)](https://github.com/1adrianb/face-alignment)][[CSDN blogs](https://www.cnblogs.com/molakejin/p/8027573.html)][`pip install face-alignment`]

* **PRNet(ECCV2018)** Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network [[arxiv link](https://arxiv.org/abs/1803.07835)][[Codes|TensorFlow(offical)](https://github.com/YadiraF/PRNet)]

* **3DDFA_V2(ECCV2020)** Towards Fast, Accurate and Stable 3D Dense Face Alignment [[paper link](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640154.pdf)][[codes|PyTorch 3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)]

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**


### ▶ Face Detection

#### Materials

* [(github) A-Light-and-Fast-Face-Detector-for-Edge-Devices](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices)
* [(website) FDDB: Face Detection Data Set and Benchmark Home](http://vis-www.cs.umass.edu/fddb/)
* [(CSDN blogs) 人脸检测（十八）--TinyFace(S3FD,SSH,HR,RSA,Face R-CNN,PyramidBox)](https://blog.csdn.net/App_12062011/article/details/80534351)
* [(github) e2e-joint-face-detection-and-alignment](https://github.com/KaleidoZhouYN/e2e-joint-face-detection-and-alignment)
* [(github) libfacedetection in PyTorch](https://github.com/ShiqiYu/libfacedetection/)
* [(github) 1MB lightweight face detection model (1MB轻量级人脸检测模型)](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
* [(blog) LFFD 再升级！新增行人和人头检测模型，及优化的C++实现](https://www.zhuanzhi.ai/document/d36c78507cc5d09dcac3fb7241344f3b)
* [(github) YOLO-FaceV2: A Scale and Occlusion Aware Face Detector](https://github.com/Krasjet-Yu/YOLO-FaceV2)[[paper link](https://arxiv.org/abs/2208.02019)]

#### Papers

* **MTCNN(SPL2016)** Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks [[paper link](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)][[project link](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)][[Codes|Caffe&Matlab(offical)](https://github.com/kpzhang93/MTCNN_face_detection_alignment)][[Codes|MXNet(unoffical)](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection)][[Codes|Tensorflow(unoffical)](https://github.com/AITTSMD/MTCNN-Tensorflow)][[CSDN blog](https://blog.csdn.net/qq_36782182/article/details/83624357)]

* **TinyFace(CVPR2017)** Finding Tiny Faces [[arxiv link](https://arxiv.org/abs/1612.04402)][[preject link](https://www.cs.cmu.edu/~peiyunh/tiny/)][[Codes|MATLAB(offical)](https://github.com/peiyunh/tiny)][[Codes|PyTorch(unoffical)](https://github.com/varunagrawal/tiny-faces-pytorch)][[Codes|MXNet(unoffical)](https://github.com/chinakook/hr101_mxnet)][[Codes|Tensorflow(unoffical)](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow)]

* **FaceBoxes(IJCB2017)** FaceBoxes: A CPU Real-time Face Detector with High Accuracy [[arxiv link](https://arxiv.org/abs/1708.05234)][[Codes|Caffe(offical)](https://github.com/sfzhang15/FaceBoxes)][[Codes|PyTorch(unoffical)](https://github.com/zisianw/FaceBoxes.PyTorch)]

* **SSH(ICCV2017)** SSH: Single Stage Headless Face Detector [[arxiv link](https://arxiv.org/abs/1708.03979)][[Codes|Caffe(offical)](https://github.com/mahyarnajibi/SSH)][[Codes|MXNet(unoffical SSH with Alignment)](https://github.com/ElegantGod/SSHA)][[Codes|(unoffical enhanced-ssh-mxnet)](https://github.com/deepinx/enhanced-ssh-mxnet)]

* **S3FD(ICCV2017)** S³FD: Single Shot Scale-invariant Face Detector [[arxiv link](https://arxiv.org/abs/1708.05237)][[Codes|Caffe(offical)](https://github.com/sfzhang15/SFD)]

* **RSA(ICCV2017)** Recurrent Scale Approximation (RSA) for Object Detection [[arxiv link](https://arxiv.org/abs/1707.09531)][[Codes|Caffe(offical)](https://github.com/liuyuisanai/RSA-for-object-detection)]

* **DSFD(CVPR2019)** DSFD: Dual Shot Face Detector [[arxiv link](https://arxiv.org/abs/1810.10220)][[Codes|PyTorch(offical)](https://github.com/yxlijun/DSFD.pytorch)][[CSDN blog](https://blog.csdn.net/wwwhp/article/details/83757286)]

* **LFFD(arxiv2019)** LFFD: A Light and Fast Face Detector for Edge Devices [[arxiv link](https://arxiv.org/abs/1904.10633)][[Codes|PyTorch, offical V1](https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices)][[Codes|PyTorch, offical V2](https://github.com/YonghaoHe/LFD-A-Light-and-Fast-Detector)]



**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Face Recognition

#### Materials

* [(website) EyeKey 眼神科技](http://www.eyekey.com/)
* [(CSDN blogs) 人脸比对（1:N）](https://blog.csdn.net/intflojx/article/details/81278330)
* [(github) Face Recognition (dlib with deep learning reaching 99.38% acc in LFW)](https://github.com/ageitgey/face_recognition)
* [(website) face_recognition package](https://face-recognition.readthedocs.io/en/latest/face_recognition.html)

###3 Papers

* **ArcFace/InsightFace(CVPR2019)** ArcFace: Additive Angular Margin Loss for Deep Face Recognition [[arxiv link](https://arxiv.org/abs/1801.07698)][[Codes|MXNet(offical insightface)](https://github.com/deepinsight/insightface)][[Codes|MXNet(offical ArcFace)](https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace)][[CSDN blog](https://blog.csdn.net/fire_light_/article/details/79602705)]

* **SubCenter-ArcFace(ECCV2020)** Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces [[paper link](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf)][[Codes|MXNet(offical SubCenter-ArcFace)](https://github.com/deepinsight/insightface/tree/master/recognition/SubCenter-ArcFace)][[CSDN blogs](https://blog.csdn.net/XBB102910/article/details/109400771)]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Face Reconstruction (3D)

#### Materials

* [(CSDNblogs) 3D人脸重建--学习笔记](https://blog.csdn.net/u011681952/article/details/82623328)
* [(CSDNblogs) PRNet人脸重建学习笔记](https://blog.csdn.net/johnyu024/article/details/100511408)
* [(github) Python tools for 3D face: 3DMM, Mesh processing(transform, camera, light, render), 3D face representations.](https://github.com/YadiraF/face3d)
* [(zhihu) 1.利用3D mesh生成2D图像](https://zhuanlan.zhihu.com/p/463003032) [2.人脸3DMM](https://zhuanlan.zhihu.com/p/463145736) [3. 2D图像的3D重建(3DMM)](https://zhuanlan.zhihu.com/p/465224205)


#### Papers

* **VRN(ICCV2017)** Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression [[arxiv link](https://arxiv.org/abs/1703.07834)][[project link](http://aaronsplace.co.uk/papers/jackson2017recon/)][[online website](https://cvl-demos.cs.nott.ac.uk/vrn/)][[Codes|Torch7(offical)](https://github.com/AaronJackson/vrn)]

* **PRNet(ECCV2018)** Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network [[arxiv link](https://arxiv.org/abs/1803.07835)][[Codes|TensorFlow(offical)](https://github.com/YadiraF/PRNet)]

* **RingNet(CVPR2019)** Learning to Regress 3D Face Shape and Expression from an Image without 3D Supervision [[paper link](https://arxiv.org/abs/1905.06817)][[project link](https://ringnet.is.tue.mpg.de/index.html)][[codes|official Tensorflow ](https://github.com/soubhiksanyal/RingNet)][`NoW dataset`]

* **REALY(ECCV2022)** REALY: Rethinking the Evaluation of 3D Face Reconstruction [[paper link](https://arxiv.org/abs/2203.09729)][[project link](https://www.realy3dface.com/)][[codes|official](https://github.com/czh-98/REALY)][[blogs|zhihu](https://zhuanlan.zhihu.com/p/549704170)]

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Head Detector

### Materials

* [(zhihu) 一文读懂YOLO V5 与 YOLO V4](https://zhuanlan.zhihu.com/p/161083602?d=1605663864267)
* [(zhihu) 如何评价YOLOv5？](https://www.zhihu.com/question/399884529)
* [(csdn blog) YOLO/V1、V2、V3目标检测系列介绍](https://blog.csdn.net/qq26983255/article/details/82119232)
* [(csdn blog) 睿智的目标检测26——Pytorch搭建yolo3目标检测平台](https://blog.csdn.net/weixin_44791964/article/details/105310627)
* [(csdn blog) 睿智的目标检测30——Pytorch搭建YoloV4目标检测平台](https://blog.csdn.net/weixin_44791964/article/details/106214657/)

### Papers

* **ReInspect, Lhungarian(CVPR2016)** End-To-End People Detection in Crowded Scenes [[arxiv link](https://arxiv.org/abs/1506.04878)]

* **DA-RCNN(CVPR2018)** Double Anchor R-CNN for Human Detection in a Crowd [[arxiv link](https://arxiv.org/abs/1909.09998)][[CSDN blog1](https://blog.csdn.net/Suan2014/article/details/103987896)][[CSDN blog2](https://blog.csdn.net/Megvii_tech/article/details/103485685)]

* **FCHD(arxiv2018,ICIP2019)** FCHD: Fast and accurate head detection in crowded scenes [[arxiv link](https://arxiv.org/abs/1809.08766)][[Codes|PyTorch(official)](https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector)][[CSDN blog](https://blog.csdn.net/javastart/article/details/82865858)]

* **LSC-CNN(TPAMI2020)** Locate, Size and Count: Accurately Resolving People in Dense Crowds via Detection [[arxiv link](https://arxiv.org/abs/1906.07538)][[Codes|Pytorch(official)](https://github.com/val-iisc/lsc-cnn)]
 
* **YOLOv5(2020)** YOLOv5 is from the family of object detection architectures YOLO and has no paper [[YOLOv5 Docs](https://docs.ultralytics.com/)][[Code|PyTorch(official)](https://github.com/ultralytics/yolov5)]

* **JointDet(AAAI2020)** Relational Learning for Joint Head and Human Detection [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/6691)][[codes|not released](https://github.com/ChiCheng123/JointDet)]

* **BFJDet(ICCV2021)** Body-Face Joint Detection via Embedding and Head Hook [[paper link](https://openaccess.thecvf.com/content/ICCV2021/papers/Wan_Body-Face_Joint_Detection_via_Embedding_and_Head_Hook_ICCV_2021_paper.pdf)][[codes|official](https://github.com/AibeeDetect/BFJDet)][`joint detection of person head and body`]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Head Pose Estimation

#### Materials

* [(tutorial & blog) Head Pose Estimation using OpenCV and Dlib](https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)
* [(blogs) 基于Dlib和OpenCV的人脸姿态估计(HeadPoseEstimation))](https://blog.csdn.net/u013512448/article/details/77804161)
* [(blogs) 使用opencv和dlib进行人脸姿态估计(python)](https://blog.csdn.net/yuanlulu/article/details/82763170)
* [(cnblogs) paper 154：姿态估计（Hand Pose Estimation）相关总结](https://www.cnblogs.com/molakejin/p/8021574.html)
* [(blogs) solvepnp三维位姿估算 | PnP 单目相机位姿估计（一、二、三）](https://blog.csdn.net/cocoaqin/article/details/77485436)
* [(github) OpenFace 2.2.0: a facial behavior analysis toolkit](https://github.com/TadasBaltrusaitis/OpenFace)
* [(github) Deepgaze contains useful packages including Head Pose Estimation](https://github.com/mpatacchiola/deepgaze)
* [(github) [Suggestion] Annotate rigid objects in 2D image with standard 3D cube](https://github.com/openvinotoolkit/cvat/issues/3387)
* [(github) head pose estimation system based on 3d facial landmarks (3DDFA_v2)](https://github.com/bubingy/HeadPoseEstimate)
* [(paper-CVPR2019) On the Continuity of Rotation Representations in Neural Networks (6D表征头姿最合适)](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.html)
* [(blogs) What is The Difference Between 2D and 3D Image Annotations: Use Cases](https://anolytics.home.blog/2019/07/18/difference-between-2d-and-3d-image-annotations-use-cases/)
* [(zhihu) 如何通俗地解释欧拉角？之后为何要引入四元数？](https://www.zhihu.com/question/47736315)
* [(blogs) 四元数与欧拉角（Yaw、Pitch、Roll）的转换](https://blog.csdn.net/xiaoma_bk/article/details/79082629)


#### Datasets
* [BIWI Kinect Head Pose Database](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)
* [300W-LP & AFLW2000](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)


#### Papers(Journal)

* **Survey(TPAMI2009)** Head Pose Estimation in Computer Vision: A Survey [[paper link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4497208)][[CSDN blog](https://blog.csdn.net/weixin_41703033/article/details/83215043)]

* **HyperFace(TPAMI2017)** HyperFace: A Deep Multi-Task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition [[paper link](https://ieeexplore.ieee.org/abstract/document/8170321)]

* **(Neurocomputing2018)** Appearance based pedestrians head pose and body orientation estimation using deep learning [[paper link](https://www.sciencedirect.com/science/article/pii/S0925231217312869)]

* **HeadFusion(TPAMI2018)** HeadFusion: 360 Head Pose Tracking Combining 3D Morphable Model and 3D Reconstruction [[paper link](https://www.idiap.ch/~odobez/publications/YuFunesOdobez-PAMI2018.pdf)]

* ⭐**QuatNet(TMM2019)** Quatnet: Quaternion-based head pose estimation with multiregression loss [[paper link](https://ieeexplore.ieee.org/abstract/document/8444061)]

* **(IVC2020)** Improving head pose estimation using two-stage ensembles with top-k regression [[paper link](https://www.sciencedirect.com/sdfe/reader/pii/S0262885619304202/pdf)]

* ⭐**MNN(TPAMI2020)** Multi-Task Head Pose Estimation in-the-Wild [[paper link](https://bobetocalo.github.io/pdf/paper_pami20.pdf)][[codes|Tensorflow / C++](https://github.com/bobetocalo/bobetocalo_pami20)]

#### Papers(Conference)

* **(ITSC2014)** Head detection and orientation estimation for pedestrian safety [[paper link](https://www.mrt.kit.edu/z/publ/download/2014/RehderKloedenStiller2014itsc.pdf)]

* **Dlib(68 points)(CVPR2014)** One Millisecond Face Alignment with an Ensemble of Regression Trees [[paper link](https://openaccess.thecvf.com/content_cvpr_2014/html/Kazemi_One_Millisecond_Face_2014_CVPR_paper.html)]

* ⭐**3DDFA(CVPR2016)** Face Alignment Across Large Poses: A 3D Solution [[paper link](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhu_Face_Alignment_Across_CVPR_2016_paper.html)]

* ⭐**FAN(12 points)(ICCV2017)** How Far Are We From Solving the 2D & 3D Face Alignment Problem? (And a Dataset of 230,000 3D Facial Landmarks) [[paper link](https://openaccess.thecvf.com/content_iccv_2017/html/Bulat_How_Far_Are_ICCV_2017_paper.html)]

* **KEPLER(FG2017)** KEPLER: Keypoint and Pose Estimation of Unconstrained Faces by Learning Efficient H-CNN Regressors [[paper link](https://ieeexplore.ieee.org/abstract/document/7961750)]

* **FasterRCNN+regression(ACCV2018)** Simultaneous Face Detection and Head Pose Estimation: A Fast and Unified Framework [[paper link](https://link.springer.com/content/pdf/10.1007%2F978-3-030-20887-5_12.pdf)][dataset|[AFW](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.661.3510&rep=rep1&type=pdf) and [ALFW](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.384.2988&rep=rep1&type=pdf) dataset: coarse face pose by using Subcategory to generate 12 clusters]

* **WNet(ACCVW2018)** WNet: Joint Multiple Head Detection and Head Pose Estimation from a Spectator Crowd Image [[paper link](https://stevenputtemans.github.io/AMV2018/presentations/wnet_presentation.pdf)][[dataset|spectator crowd S-HOCK dataset: rough orientation labels](https://iris.unitn.it/retrieve/handle/11572/187463/470794/Shock_r2.pdf)]

* **SSR-Net-MD(IJCAI2018)** SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation [[paper link](https://www.ijcai.org/proceedings/2018/0150.pdf)][[codes|Tensorflow+Dlib+MTCNN](https://github.com/shamangary/SSR-Net)]

* **HeadPose(FG2019)** Improving Head Pose Estimation with a Combined Loss and Bounding Box Margin Adjustment [[paper link](https://ieeexplore.ieee.org/abstract/document/8756605)][[codes|TensorFlow](https://github.com/MingzhenShao/HeadPose)]

* ⭐**HopeNet(CVPRW2018)** Fine-Grained Head Pose Estimation Without Keypoints [[arxiv link](https://arxiv.org/abs/1710.00925)][[Codes|PyTorch(official)](https://github.com/natanielruiz/deep-head-pose)][[CSDN blog](https://blog.csdn.net/qq_42189368/article/details/84849638)]

* ⭐**FSA-Net(CVPR2019)** FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image [[paper link](https://github.com/shamangary/FSA-Net/blob/master/0191.pdf)][[Codes|Keras&Tensorflow(official)](https://github.com/shamangary/FSA-Net)][[Codes|PyTorch(unofficial)](https://github.com/omasaht/headpose-fsanet-pytorch)]

* ⭐**WHENet(BMVC2020)** WHENet: Real-time Fine-Grained Estimation for Wide Range Head Pose [[arxiv link](https://arxiv.org/abs/2005.10353)][[Codes|Kears&tensorflow(official)](https://github.com/Ascend-Research/HeadPoseEstimation-WHENet)][[codes|PyTorch(unofficial)](https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino)]

* **RAFA-Net(ACCV2020)** Rotation Axis Focused Attention Network (RAFA-Net) for Estimating Head Pose [[paper link](https://openaccess.thecvf.com/content/ACCV2020/html/Behera_Rotation_Axis_Focused_Attention_Network_RAFA-Net_for_Estimating_Head_Pose_ACCV_2020_paper.html)][[codes|keras+tensorflow](https://github.com/ArdhenduBehera/RAFA-Net)]

* ⭐**FDN(AAAI2020)** FDN: Feature decoupling network for head pose estimation [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/6974)]

* **Rankpose(BMVC2020)** RankPose: Learning Generalised Feature with Rank Supervision for Head Pose Estimation [[paper link](https://www.bmvc2020-conference.com/assets/papers/0401.pdf)][[codes|PyTorch](https://github.com/seathiefwang/RankPose)]

* **EVA-GCN(CVPRW2021)** EVA-GCN: Head Pose Estimation Based on Graph Convolutional Networks [[paper link](http://openaccess.thecvf.com/content/CVPR2021W/AMFG/html/Xin_EVA-GCN_Head_Pose_Estimation_Based_on_Graph_Convolutional_Networks_CVPRW_2021_paper.html)][[codes|PyTorch](https://github.com/stoneMo/EVA-GCN)]

* ⭐**TriNet(WACV2021)** A Vector-Based Representation to Enhance Head Pose Estimation
 [[paper link](http://openaccess.thecvf.com/content/WACV2021/html/Chu_A_Vector-Based_Representation_to_Enhance_Head_Pose_Estimation_WACV_2021_paper.html)][[codes|Tensorflow+Keras](https://github.com/anArkitek/TriNet_WACV2021)]

* ⭐**img2pose(CVPR2021)** img2pose: Face Alignment and Detection via 6DoF, Face Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Albiero_img2pose_Face_Alignment_and_Detection_via_6DoF_Face_Pose_Estimation_CVPR_2021_paper.html)][[codes|PyTorch](http://github.com/vitoralbiero/img2pose)]

* ⭐**OsGG-Net(ACMMM2021)** OsGG-Net: One-step Graph Generation Network for Unbiased Head Pose Estimation [[paper link](https://dl.acm.org/doi/abs/10.1145/3474085.3475417)][[codes|PyTorch](https://github.com/stoneMo/OsGG-Net)]

* **(KSE2021)** Simultaneous face detection and 360 degree head pose estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/9648838)]【文章使用了FPN+Multi-task的方式，同时检测人头和识别人头姿态，数据集主要使用了CMU-Panoptic，300WLP和BIWI。头姿表示形式上，除了欧拉角，还使用了Rotation Matrix】

* **(KSE2021)** UET-Headpose: A sensor-based top-view head pose dataset [[paper link](https://ieeexplore.ieee.org/abstract/document/9648656)] 【全文均在阐述获取数据集的硬件系统，但数据集未公布；HPE算法为FSA-Net，并根据WHENet中的思路拓展为full-range 360°单人头部姿态估计方法】

* **(FG2021)** Relative Pose Consistency for Semi-Supervised Head Pose Estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/9666992/)]

* ⭐**SynergyNet(3DV2021)** Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry [[paper link](https://www.computer.org/csdl/proceedings-article/3dv/2021/268800a453/1zWEnuGbFte)][[project link](https://choyingw.github.io/works/SynergyNet)][[codes|PyTorch](https://github.com/choyingw/SynergyNet)]

* **MOS(BMVC2021)** MOS: A Low Latency and Lightweight Framework for Face Detection, Landmark Localization, and Head Pose Estimation [[paper link](https://www.bmvc2021-virtualconference.com/assets/papers/0580.pdf)][[codes|PyTorch](https://github.com/lyp-deeplearning/MOS-Multi-Task-Face-Detect)]

* **MTGLS(WACV2022)** MTGLS: Multi-Task Gaze Estimation With Limited Supervision [[paper link](https://openaccess.thecvf.com/content/WACV2022/html/Ghosh_MTGLS_Multi-Task_Gaze_Estimation_With_Limited_Supervision_WACV_2022_paper.html)]

* **LwPosr(WACV2022)** LwPosr: Lightweight Efficient Fine Grained Head Pose Estimation [[paper link](https://openaccess.thecvf.com/content/WACV2022/html/Dhingra_LwPosr_Lightweight_Efficient_Fine_Grained_Head_Pose_Estimation_WACV_2022_paper.html)]

* **HHP-Net(WACV2022)** HHP-Net: A Light Heteroscedastic Neural Network for Head Pose Estimation With Uncertainty [[paper link](https://openaccess.thecvf.com/content/WACV2022/html/Cantarini_HHP-Net_A_Light_Heteroscedastic_Neural_Network_for_Head_Pose_Estimation_WACV_2022_paper.html)][[codes|TensorFlow](https://github.com/cantarinigiorgio/HHP-Net)]

* ⭐**6DRepNet(ICIP2022)** 6D Rotation Representation For Unconstrained Head Pose Estimation [[paper link](https://arxiv.org/abs/2202.12555)][[codes|PyTorch+RepVGG](https://github.com/thohemp/6DRepNet)]

* ⭐**DAD-3DNet(CVPR2022)** DAD-3DHeads: A Large-scale Dense, Accurate and Diverse Dataset for 3D Head Alignment from a Single Image [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Martyniuk_DAD-3DHeads_A_Large-Scale_Dense_Accurate_and_Diverse_Dataset_for_3D_CVPR_2022_paper.html)][[project link](https://www.pinatafarm.com/research/dad-3dheads)][[codes|official PyTorch](https://github.com/PinataFarms/DAD-3DHeads)]
