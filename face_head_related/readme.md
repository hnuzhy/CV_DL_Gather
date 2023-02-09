# Contents

* **[1) Pubilc Datasets and Challenges](#1-Pubilc-Datasets-and-Challenges)**
  * **[⭐For Face Alignment or Landmark Detection](#For-Face-Alignment-or-Landmark-Detection)**
  * **[⭐For Head Pose Estimation](#For-Head-Pose-Estimation)**
  * **[⭐For Head Detection Only](#For-Head-Detection-Only)**
  * **[⭐For Head Detection or Crowd Counting](#For-Head-Detection-or-Crowd-Counting)**
* **[2) Pioneers and Experts](#2-Pioneers-and-Experts)**
* **[3) Related Materials (Papers, Sources Code, Blogs, Videos and Applications)](#3-Related-Materials-Papers-Sources-Code-Blogs-Videos-and-Applications)**
  * **[▶ Beautify Face](#-Beautify-Face)**
  * **[▶ Body Orientation Estimation](#-Body-Orientation-Estimation)**
  * **[▶ Eye Gaze Estimation](#-Eye-Gaze-Estimation)**
  * **[▶ Face Alignment](#-Face-Alignment)**
  * **[▶ Face Detection](#-Face-Detection)**
  * **[▶ Face Recognition](#-Face-Recognition)**
  * **[▶ Face Reconstruction (3D)](#-Face-Reconstruction-3D)**
  * **[▶ Head Detector](#-Head-Detector)**
  * **[▶ Head Pose Estimation](#-Head-Pose-Estimation)**


# List of public algorithms and datasets

## 1) Pubilc Datasets and Challenges

### ⭐**For Face Alignment or Landmark Detection**
* [Flickr-Faces-HQ (FFHQ) Dataset](https://github.com/NVlabs/ffhq-dataset): Flickr-Faces-HQ (FFHQ) is a high-quality image dataset of human faces, originally created as a benchmark for`generative adversarial networks (GAN)`. The dataset consists of `70,000` high-quality PNG images at 1024×1024 resolution and contains considerable variation in terms of age, ethnicity and image background. It also has good coverage of accessories such as eyeglasses, sunglasses, hats, etc. The images were crawled from `Flickr`, thus inheriting all the biases of that website, and automatically aligned and cropped using `dlib`. [(CVPR2019) A Style-Based Generator Architecture for Generative Adversarial Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html)

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

[👍Michael Black](https://ps.is.mpg.de/person/black); [👍Jian Sun](http://www.jiansun.org/); [👍Gang YU](http://www.skicyyu.org/); [👍Yuliang Xiu 修宇亮](https://xiuyuliang.cn/); [👍(website) face-rec](https://www.face-rec.org/databases/)



## 3) Related Materials (Papers, Sources Code, Blogs, Videos and Applications)

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Beautify Face

#### Materials

* [(github) BeautifyFaceDemo](https://github.com/Guikunzhi/BeautifyFaceDemo)
* [(CSDN blogs) 图像滤镜艺术---换脸算法资源收集](https://blog.csdn.net/scythe666/article/details/81021041)

#### Papers


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Body Orientation Estimation

#### Materials


#### Papers

* **TUD(CVPR2010)** Monocular 3D Pose Estimation and Tracking by Detection [[paper link](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.171.187&rep=rep1&type=pdf)][`TUD Dataset`]

* **(ICCV2015)** Uncovering Interactions and Interactors: Joint Estimation of Head, Body Orientation and F-Formations From Surveillance Videos [[paper link](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Ricci_Uncovering_Interactions_and_ICCV_2015_paper.html)]

* **AKRF-VW(IJCV2017)** Growing Regression Tree Forests by Classification for Continuous Object Pose Estimation [[paper link](https://link.springer.com/article/10.1007/s11263-016-0942-1)]

* **CPOEHK(ISCAS2019)** Continuous Pedestrian Orientation Estimation using Human Keypoints [[paper link](https://ieeexplore.ieee.org/abstract/document/8702175/)]

* **❤ MEBOW(CVPR2020)** MEBOW: Monocular Estimation of Body Orientation in the Wild [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Wu_MEBOW_Monocular_Estimation_of_Body_Orientation_in_the_Wild_CVPR_2020_paper.html)][[project link](https://chenyanwu.github.io/MEBOW/)][[codes|official](https://github.com/ChenyanWu/MEBOW)][`COCO-MEBOW dataset, Body Orientation Estimation`]

* **PedRecNet(IV2022)** PedRecNet: Multi-task deep neural network for full 3D human pose and orientation estimation [[paper link](https://arxiv.org/abs/2204.11548)][[codes|official](https://github.com/noboevbo/PedRec)]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Eye Gaze Estimation

#### Materials


#### Papers

* **HGM(CVPR2018)** A Hierarchical Generative Model for Eye Image Synthesis and Eye Gaze Estimation [[paper link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_A_Hierarchical_Generative_CVPR_2018_paper.pdf)]

* **ETH-XGaze(ECCV2020)** ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation [[arxiv link](https://arxiv.org/abs/2007.15837)][[project link](https://ait.ethz.ch/projects/2020/ETH-XGaze/)][[Codes|PyTorch(official)](https://github.com/xucong-zhang/ETH-XGaze)]

* **EVE(ECCV2020)** Towards End-to-end Video-based Eye-tracking [[arxiv link](https://arxiv.org/abs/2007.13120)][[project link](https://ait.ethz.ch/projects/2020/EVE/)][[Codes|PyTorch(official)](https://github.com/swook/EVE)]

* **MTGLS(WACV2022)** MTGLS: Multi-Task Gaze Estimation With Limited Supervision [[paper link](https://openaccess.thecvf.com/content/WACV2022/html/Ghosh_MTGLS_Multi-Task_Gaze_Estimation_With_Limited_Supervision_WACV_2022_paper.html)]

* **RUDA(CVPR2022)** Generalizing Gaze Estimation With Rotation Consistency [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Bao_Generalizing_Gaze_Estimation_With_Rotation_Consistency_CVPR_2022_paper.html)]

* **❤ GazeOnce/MPSGaze(CVPR2022)** GazeOnce: Real-Time Multi-Person Gaze Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_GazeOnce_Real-Time_Multi-Person_Gaze_Estimation_CVPR_2022_paper.html)][[codes|official](https://github.com/mf-zhang/GazeOnce)][`The MPSGaze is a synthetic dataset (ETH-XGaze + WiderFace) containing full images (instead of only cropped faces) that provides ground truth 3D gaze directions for multiple people in one image.`]

* **❤ GAFA(CVPR2022)** Dynamic 3D Gaze From Afar: Deep Gaze Estimation From Temporal Eye-Head-Body Coordination [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Nonaka_Dynamic_3D_Gaze_From_Afar_Deep_Gaze_Estimation_From_Temporal_CVPR_2022_paper.html)][[project link](https://vision.ist.i.kyoto-u.ac.jp/research/gafa/)][[codes|official](https://github.com/kyotovision-public/dynamic-3d-gaze-from-afar)][`The GAze From Afar (GAFA) dataset consists of surveillance videos of freely moving people with automatically annotated 3D gaze, head, and body orientations.`]

* **NeRF-Gaze(arxiv2022)** NeRF-Gaze: A Head-Eye Redirection Parametric Model for Gaze Estimation [[paper link](https://arxiv.org/abs/2212.14710)][`HKVision`]

* **GazeNeRF(arxiv2022)** GazeNeRF: 3D-Aware Gaze Redirection with Neural Radiance Fields [[paper link](https://arxiv.org/abs/2212.04823)][`ETH`]

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

#### Datasets
 
* **300-W(ICCV2013)** 300 Faces In-the-Wild Challenge (300-W), ICCV 2013 [[project link](https://ibug.doc.ic.ac.uk/resources/300-W/)] [[(IMAVIS) 300 faces In-the-wild challenge: Database and results](https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_2016_imavis.pdf)] [[(ICCV-W) 300 Faces in-the-Wild Challenge: The first facial landmark localization Challenge](https://www.cv-foundation.org/openaccess/content_iccv_workshops_2013/W11/html/Sagonas_300_Faces_in-the-Wild_2013_ICCV_paper.html)]

* **FaceSynthetics(ICCV2021)** Fake It Till You Make It: Face analysis in the wild using synthetic data alone [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Wood_Fake_It_Till_You_Make_It_Face_Analysis_in_the_ICCV_2021_paper.html)][[project link](https://microsoft.github.io/FaceSynthetics/)][[code|official](https://github.com/microsoft/FaceSynthetics)]

#### Papers

* **Dlib(CVPR2014)** One Millisecond Face Alignment with an Ensemble of Regression Trees [[paper link](https://openaccess.thecvf.com/content_cvpr_2014/html/Kazemi_One_Millisecond_Face_2014_CVPR_paper.html)][[codes|official C++](https://github.com/davisking/dlib)][`pip install dlib`]

* **3000FPS(CVPR2014)** Face Alignment at 3000 FPS via Regressing Local Binary Features [[paper link](http://www.cse.psu.edu/~rtc12/CSE586/papers/regr_cvpr14_facealignment.pdf)][[Codes|opencv(offical)](https://github.com/freesouls/face-alignment-at-3000fps)][[Codes|liblinear(unoffical)](https://github.com/jwyang/face-alignment)][[CSDN blog](https://blog.csdn.net/lzb863/article/details/49890369)]

* ❤**3DDFA(CVPR2016)** Face Alignment Across Large Poses: A 3D Solution [[paper link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780392)][[project link](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)][[codes|PyTorch 3DDFA](https://github.com/cleardusk/3DDFA)]

* **FAN(ICCV2017)** How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks) [[paper link](https://www.adrianbulat.com/downloads/FaceAlignment/FaceAlignment.pdf)][[Adrian Bulat](https://www.adrianbulat.com/)][[Codes|PyTorch(offical)](https://github.com/1adrianb/face-alignment)][[CSDN blogs](https://www.cnblogs.com/molakejin/p/8027573.html)][`pip install face-alignment`]

* **PRNet(ECCV2018)** Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network [[arxiv link](https://arxiv.org/abs/1803.07835)][[Codes|TensorFlow(offical)](https://github.com/YadiraF/PRNet)]

* ❤**3DDFA_V2(ECCV2020)** Towards Fast, Accurate and Stable 3D Dense Face Alignment [[paper link](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640154.pdf)][[codes|PyTorch 3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)]

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

#### Datasets

* **WIDER FACE(CVPR2016)** WIDER FACE: A Face Detection Benchmark [[paper link](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/paper.pdf)][[project link origin](http://shuoyang1213.me/WIDERFACE/)][[project link new](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)]


#### Papers

* ❤**MTCNN(SPL2016)** Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks [[paper link](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)][[project link](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)][[Codes|Caffe&Matlab(offical)](https://github.com/kpzhang93/MTCNN_face_detection_alignment)][[Codes|MXNet(unoffical)](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection)][[Codes|Tensorflow(unoffical)](https://github.com/AITTSMD/MTCNN-Tensorflow)][[CSDN blog](https://blog.csdn.net/qq_36782182/article/details/83624357)]

* **TinyFace(CVPR2017)** Finding Tiny Faces [[arxiv link](https://arxiv.org/abs/1612.04402)][[preject link](https://www.cs.cmu.edu/~peiyunh/tiny/)][[Codes|MATLAB(offical)](https://github.com/peiyunh/tiny)][[Codes|PyTorch(unoffical)](https://github.com/varunagrawal/tiny-faces-pytorch)][[Codes|MXNet(unoffical)](https://github.com/chinakook/hr101_mxnet)][[Codes|Tensorflow(unoffical)](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow)]

* **FaceBoxes(IJCB2017)** FaceBoxes: A CPU Real-time Face Detector with High Accuracy [[arxiv link](https://arxiv.org/abs/1708.05234)][[Codes|Caffe(offical)](https://github.com/sfzhang15/FaceBoxes)][[Codes|PyTorch(unoffical)](https://github.com/zisianw/FaceBoxes.PyTorch)]

* **SSH(ICCV2017)** SSH: Single Stage Headless Face Detector [[arxiv link](https://arxiv.org/abs/1708.03979)][[Codes|Caffe(offical)](https://github.com/mahyarnajibi/SSH)][[Codes|MXNet(unoffical SSH with Alignment)](https://github.com/ElegantGod/SSHA)][[Codes|(unoffical enhanced-ssh-mxnet)](https://github.com/deepinx/enhanced-ssh-mxnet)]

* ❤**S3FD(ICCV2017)** S³FD: Single Shot Scale-invariant Face Detector [[arxiv link](https://arxiv.org/abs/1708.05237)][[Codes|Caffe(offical)](https://github.com/sfzhang15/SFD)]

* **RSA(ICCV2017)** Recurrent Scale Approximation (RSA) for Object Detection [[arxiv link](https://arxiv.org/abs/1707.09531)][[Codes|Caffe(offical)](https://github.com/liuyuisanai/RSA-for-object-detection)]

* **DSFD(CVPR2019)** DSFD: Dual Shot Face Detector [[arxiv link](https://arxiv.org/abs/1810.10220)][[Codes|PyTorch(offical)](https://github.com/yxlijun/DSFD.pytorch)][[CSDN blog](https://blog.csdn.net/wwwhp/article/details/83757286)]

* **LFFD(arxiv2019)** LFFD: A Light and Fast Face Detector for Edge Devices [[arxiv link](https://arxiv.org/abs/1904.10633)][[Codes|PyTorch, offical V1](https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices)][[Codes|PyTorch, offical V2](https://github.com/YonghaoHe/LFD-A-Light-and-Fast-Detector)]

* ❤**RetinaFace(CVPR2020)** RetinaFace: Single-shot Multi-level Face Localisation in the Wild [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html)][[Github - insightface](https://github.com/deepinsight/insightface)][[Project - insightface](https://insightface.ai/retinaface)][[codes|PyTorch(not official)](https://github.com/biubug6/Pytorch_Retinaface)][[codes|MXNet(official)](https://github.com/deepinsight/insightface/tree/master/detection/retinaface)][`RetinaFace: Single-stage Dense Face Localisation in the Wild` is the same work released in Arxiv2019]


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
* [(website) searching '3D Face Reconstruction' in the website catalyzex](https://www.catalyzex.com/s/3D%20Face%20Reconstruction)
* [(github) Awesome-Talking-Face (papers, code and projects)](https://github.com/JosephPai/Awesome-Talking-Face)


#### Datasets

* [**Papers With Code Ranks**][[NoW Benchmark](https://paperswithcode.com/dataset/now-benchmark)] [[FaceScape](https://paperswithcode.com/dataset/facescape)] [[D3DFACS](https://paperswithcode.com/dataset/d3dfacs)] [[AFLW2000-3D](https://paperswithcode.com/dataset/aflw2000-3d)]
* [**CelebA**] [(ICCV2015) Deep Learning Face Attributes in the Wild](https://openaccess.thecvf.com/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html) [[project link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)] [[zhihu-zhuanlan](https://zhuanlan.zhihu.com/p/35975956)] [(ICLR2018 by NVIDIA) [CelebA-HQ (paperswithcode)](https://paperswithcode.com/dataset/celeba-hq), [CelebA-HQ (tensorflow-download)](https://www.tensorflow.org/datasets/catalog/celeb_a_hq), [CelebA-HQ (how to generate this dataset?)](https://zhuanlan.zhihu.com/p/52188519), [CelebA-HQ (upload by somebody)](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P)] [(CVPR2020 by MMLab) [CelebAMask-HQ (codes)](https://github.com/switchablenorms/CelebAMask-HQ)] [(CVPR2021 by MMLab) [Multi-Modal-CelebA-HQ (codes)](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset)] [`not a face reconstruction dataset`]
* [**Feng et al. using Stirling meshes (Stirling/ESRC Benchmark)**] [(FG2018) Evaluation of Dense 3D Reconstruction from 2D Face Images in the Wild](https://ieeexplore.ieee.org/abstract/document/8373916) [[pdf page](https://arxiv.org/pdf/2204.06607.pdf)]
* [**NoW ("Not quite in-the-Wild")**] [RingNet(CVPR2019) Learning to Regress 3D Face Shape and Expression from an Image without 3D Supervision](https://ringnet.is.tue.mpg.de/index.html) [[NoW Challenge](https://now.is.tue.mpg.de/nonmetricalevaluation.html)]
* [**FaceScape**] [FaceScape(CVPR2020) FaceScape: A Large-Scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction](https://facescape.nju.edu.cn/)
* [**DAD-3DHeads**] [DAD-3DNet(CVPR2022) DAD-3DHeads: A Large-scale Dense, Accurate and Diverse Dataset for 3D Head Alignment from a Single Image](https://www.pinatafarm.com/research/dad-3dheads)
* [**FaceSynthetics**] [FaceSynthetics(ICCV2021) Fake It Till You Make It: Face analysis in the wild using synthetic data alone](https://microsoft.github.io/FaceSynthetics/) [`synthetic face image with 70 landmarks`]


#### Survey

* **Survey of optimization-based methods(CGFroum2018)** State of the Art on Monocular 3D Face Reconstruction, Tracking, and Applications [[paper link](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13382)][[pdf page](http://zollhoefer.com/papers/EG18_FaceSTAR/paper.pdf)]

* **Survey of face models(TOG2020)** 3D Morphable Face Models—Past, Present, and Future [[paper link](https://dl.acm.org/doi/abs/10.1145/3395208)][pdf page](https://www.researchgate.net/profile/Adam-Kortylewski/publication/342056152_3D_Morphable_Face_Models-Past_Present_and_Future/links/5f73174492851c14bc9d26c9/3D-Morphable-Face-Models-Past-Present-and-Future.pdf)]

* **Survey of regression-based methods(CSReview2021)** Survey on 3D face reconstruction from uncalibrated images [[paper link](https://www.sciencedirect.com/science/article/pii/S157401372100040X)][[pdf page](https://arxiv.org/pdf/2011.05740.pdf)]

* **Survey on SOTA 3D reconstruction with single RGB image (arxiv2022)** State of the Art in Dense Monocular Non-Rigid 3D Reconstruction [[paper link](https://arxiv.org/abs/2210.15664)]


#### Papers (Conference and Journey)

* **Blanz et al.(SIGGRAPH1999)** A morphable model for the synthesis of 3D faces [[paper link](https://dl.acm.org/doi/pdf/10.1145/311535.311556)][`3DMM of face/head`][`The seminal work of 3DMM`]

* ⭐**BFM(AVSS2009)** A 3D Face Model for Pose and Illumination Invariant Face Recognition [[paper link](https://web.archive.org/web/20170813045339id_/http://gravis.dmi.unibas.ch/publications/2009/BFModel09.pdf)][[project link](https://faces.dmi.unibas.ch/bfm/)][[bfm2019 model downloading](https://faces.dmi.unibas.ch/bfm/bfm2019.html)][[Basel Face Model 2019 Viewer](https://github.com/unibas-gravis/basel-face-model-viewer)][`3DMM of face/head (BFM)`][Well-known 3DMM by `University of Basel, Switzerland`]

* **LSFM(CVPR2016)** A 3D Morphable Model learnt from 10,000 faces [[paper link](https://ibug.doc.ic.ac.uk/media/uploads/documents/0002.pdf)][[project link](https://ibug.doc.ic.ac.uk/resources/lsfm/)][[code|official](https://github.com/menpo/lsfm)][[(IJCV2017) Large Scale 3D Morphable Models](https://link.springer.com/article/10.1007/s11263-017-1009-7)][`3DMM of face/head (LSFM)`][By `the iBUG group at Imperial, UK`]

* ⭐**FLAME(SIGGRAPH2017)** Learning a model of facial shape and expression from 4D scans [[paper link](https://ps.is.mpg.de/uploads_file/attachment/attachment/400/paper.pdf)][[project link](https://flame.is.tue.mpg.de/)][[code|official Chumpy FLAME fitting](https://github.com/Rubikplayer/flame-fitting)][[code|official FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch)][[code|official FLAME texture fitting](https://github.com/HavenFeng/photometric_optimization)][`3DMM of face/head (FLAME)`][`MPII 马普所`]

* **3DMM-CNN(CVPR2017)** Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network [[paper link](https://arxiv.org/abs/1612.04904)][[code|official](https://github.com/anhttran/3dmm_cnn)]

* **MoFA(ICCV2017)** MoFA: Model-Based Deep Convolutional Face Autoencoder for Unsupervised Monocular Reconstruction [[paper link](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w19/html/Tewari_MoFA_Model-Based_Deep_ICCV_2017_paper.html)]

* **VRN(ICCV2017)** Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression [[arxiv link](https://arxiv.org/abs/1703.07834)][[project link](http://aaronsplace.co.uk/papers/jackson2017recon/)][[online website](https://cvl-demos.cs.nott.ac.uk/vrn/)][[Codes|Torch7(offical)](https://github.com/AaronJackson/vrn)]

* **BIP(IJCV2018)** Occlusion-Aware 3D Morphable Models and an Illumination Prior for Face Image Analysis [[paper link](https://link.springer.com/article/10.1007/s11263-018-1064-8)][[project link](https://shapemodelling.cs.unibas.ch/web/)][[code|official](https://github.com/unibas-gravis/parametric-face-image-generator)][`Basel Illumination Prior 2017`]

* **PRNet(ECCV2018)** Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network [[arxiv link](https://arxiv.org/abs/1803.07835)][[Codes|TensorFlow(offical)](https://github.com/YadiraF/PRNet)]

* **LYHM(IJCV2019)** Statistical Modeling of Craniofacial Shape and Texture [[paper link](https://link.springer.com/article/10.1007/s11263-019-01260-7)][[project link](https://www-users.cs.york.ac.uk/~nep/research/LYHM/)][`3DMM of face/head (LYHM)`][By `Liverpool-York: Liverpool (UK) and the University of York (UK)`]

* ⭐**Syn&Real(ICCV2019)** 3D Face Modeling From Diverse Raw Scan Data [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_3D_Face_Modeling_From_Diverse_Raw_Scan_Data_ICCV_2019_paper.html)][[codes|official](https://github.com/liuf1990/3DFC)][`A subset of Stirling/ESRC 3D face database`]

* 👍**Deep3DFaceRecon(CVPRW2019)** Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set [[paper link](https://arxiv.org/abs/1903.08527)][[code|official](https://github.com/microsoft/Deep3DFaceReconstruction)][[code|not official, a better version using PyTorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)]

* ⭐**RingNet(CVPR2019)** Learning to Regress 3D Face Shape and Expression from an Image without 3D Supervision [[paper link](https://arxiv.org/abs/1905.06817)][[project link](https://ringnet.is.tue.mpg.de/index.html)][[codes|official Tensorflow ](https://github.com/soubhiksanyal/RingNet)][[NoW evaluation code](https://github.com/soubhiksanyal/now_evaluation)][[NoW challenge page](https://now.is.tue.mpg.de/nonmetricalevaluation.html)][`NoW dataset`][`FLAME based`][`MPII 马普所`]

* **FaceScape(CVPR2020)** FaceScape: A Large-Scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FaceScape_A_Large-Scale_High_Quality_3D_Face_Dataset_and_Detailed_CVPR_2020_paper.html)][[project link](https://facescape.nju.edu.cn/)][[codes|official](https://github.com/zhuhao-nju/facescape)][`3DMM of face/head (FaceScape)` and `3D face dataset (FaceScape)`][By `NJU`]

* **UMDFA(ECCV2020)** “Look Ma, no landmarks!”–Unsupervised, model-based dense face alignment [[paper link](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470681.pdf)][[code|official (not released)](https://github.com/kzmttr/UMDFA)]

* **MGCNet(ECCV2020)** Self-Supervised Monocular 3D Face Reconstruction by Occlusion-Aware Multi-view Geometry Consistency [[paper link](https://arxiv.org/abs/2007.12494)][[code|official](https://github.com/jiaxiangshang/MGCNet)]

* ⭐**3DDFA_V2(ECCV2020)** Towards Fast, Accurate and Stable 3D Dense Face Alignment [[paper link](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640154.pdf)][[codes|PyTorch 3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)]

* ⭐**SynergyNet(3DV2021)** Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry [[paper link](https://www.computer.org/csdl/proceedings-article/3dv/2021/268800a453/1zWEnuGbFte)][[project link](https://choyingw.github.io/works/SynergyNet)][[codes|PyTorch](https://github.com/choyingw/SynergyNet)]

* **(ICCV2021)** Towards High Fidelity Monocular Face Reconstruction with Rich Reflectance using Self-supervised Learning and Ray Tracing [[paper link](https://arxiv.org/abs/2103.15432)][`MPII 马普所`]

* **HIFI3D(TOG2021)** High-Fidelity 3D Digital Human Head Creation from RGB-D Selfies [[paper link](https://dl.acm.org/doi/abs/10.1145/3472954)][[project link](https://tencent-ailab.github.io/hifi3dface_projpage/)][[codes|official](https://github.com/tencent-ailab/hifi3dface)][`3DMM of face/head (HIFI3D)`][By `Tencent`]

* ⭐**DECA(TOG2021)(SIGGRAPH2021)** Learning an animatable detailed 3D face model from in-the-wild images [[paper link](https://dl.acm.org/doi/abs/10.1145/3450626.3459936)][[project link](https://deca.is.tue.mpg.de/)][[code|official](https://github.com/YadiraF/DECA)][`MPII 马普所`]

* 👍**FOCUS(arxiv2021)** To fit or not to fit: Model-based Face Reconstruction and Occlusion Segmentation from Weak Supervision [[paper link](https://arxiv.org/abs/2106.09614)][[code|official](https://github.com/unibas-gravis/Occlusion-Robust-MoFA)]

* ⭐**DAD-3DNet(CVPR2022)** DAD-3DHeads: A Large-scale Dense, Accurate and Diverse Dataset for 3D Head Alignment from a Single Image [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Martyniuk_DAD-3DHeads_A_Large-Scale_Dense_Accurate_and_Diverse_Dataset_for_3D_CVPR_2022_paper.html)][[project link👍](https://www.pinatafarm.com/research/dad-3dheads)][[codes|official PyTorch](https://github.com/PinataFarms/DAD-3DHeads)][[benchmark challenge👍](https://github.com/PinataFarms/DAD-3DHeads/tree/main/dad_3dheads_benchmark)][`DAD-3DHeads dataset`][By `pinatafarm`]

* **REALY(ECCV2022)** REALY: Rethinking the Evaluation of 3D Face Reconstruction [[paper link](https://arxiv.org/abs/2203.09729)][[project link](https://www.realy3dface.com/)][[codes|official](https://github.com/czh-98/REALY)][[blogs|zhihu](https://zhuanlan.zhihu.com/p/549704170)][`3DMM of face/head (HIFI3D++)` and `3D face dataset (REALY)`][By `Tsinghua`]

* **DenseLandmarks(ECCV2022)** 3D Face Reconstruction with Dense Landmarks [[paper link](https://arxiv.org/abs/2204.02776)][[project link](https://microsoft.github.io/DenseLandmarks/)][`Microsoft`]

* **MICA(ECCV2022)** Towards Metrical Reconstruction of Human Faces [[paper link](https://arxiv.org/abs/2204.06607)][[project link](https://zielon.github.io/mica/)][[code|official](https://github.com/Zielon/MICA)][[used multiple datasets](https://github.com/Zielon/MICA/tree/master/datasets/)][`SoTA results in NoW`][`MPII 马普所`]

* **JMLR(ECCVW2022)** Perspective Reconstruction of Human Faces by Joint Mesh and Landmark Regression [[paper link](https://arxiv.org/abs/2208.07142)][[code|official](https://github.com/deepinsight/insightface/tree/master/reconstruction/jmlr)]



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

* **DA-RCNN(arxiv2018)** Double Anchor R-CNN for Human Detection in a Crowd [[arxiv link](https://arxiv.org/abs/1909.09998)][[CSDN blog1](https://blog.csdn.net/Suan2014/article/details/103987896)][[CSDN blog2](https://blog.csdn.net/Megvii_tech/article/details/103485685)]

* **FCHD(arxiv2018,ICIP2019)** FCHD: Fast and accurate head detection in crowded scenes [[arxiv link](https://arxiv.org/abs/1809.08766)][[Codes|PyTorch(official)](https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector)][[CSDN blog](https://blog.csdn.net/javastart/article/details/82865858)]

* **LSC-CNN(TPAMI2020)** Locate, Size and Count: Accurately Resolving People in Dense Crowds via Detection [[arxiv link](https://arxiv.org/abs/1906.07538)][[Codes|Pytorch(official)](https://github.com/val-iisc/lsc-cnn)]
 
* **PedHunter(AAAI2020)** PedHunter: Occlusion Robust Pedestrian Detector in Crowded Scenes [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/6690)][`joint body-head detection`]
 
* **YOLOv5(2020)** YOLOv5 is from the family of object detection architectures YOLO and has no paper [[YOLOv5 Docs](https://docs.ultralytics.com/)][[Code|PyTorch(official)](https://github.com/ultralytics/yolov5)]

* **JointDet(AAAI2020)** Relational Learning for Joint Head and Human Detection [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/6691)][[codes|not released](https://github.com/ChiCheng123/JointDet)]

* **FastNFusion(PRCV2021)** Fast and Fusion: Real-Time Pedestrian Detector Boosted by Body-Head Fusion [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-88004-0_6)][`Pedestrian Detector using Body-Head Association`]

* **BFJDet(ICCV2021)** Body-Face Joint Detection via Embedding and Head Hook [[paper link](https://openaccess.thecvf.com/content/ICCV2021/papers/Wan_Body-Face_Joint_Detection_via_Embedding_and_Head_Hook_ICCV_2021_paper.pdf)][[codes|official](https://github.com/AibeeDetect/BFJDet)][`joint detection of person body, head and face`]

* **BodyHands(CVPR2022)** Whose Hands Are These? Hand Detection and Hand-Body Association in the Wild [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Narasimhaswamy_Whose_Hands_Are_These_Hand_Detection_and_Hand-Body_Association_in_CVPR_2022_paper.html)][[project link](http://vision.cs.stonybrook.edu/~supreeth/BodyHands/)][[code|official](https://github.com/cvlab-stonybrook/BodyHands)][`joint detection of person body and hands`]

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
* [(blogs) 四元数（Quaternion）和旋转 + 欧拉角](https://www.cnblogs.com/jins-note/p/9512719.html)
* [(blogs) Understanding Quaternions 中文翻译《理解四元数》](https://www.qiujiawei.com/understanding-quaternions/)


#### Datasets
* [BIWI Kinect Head Pose Database: (IJCV2013) Random forests for real time 3d face analysis](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)[`pitch-yaw-roll`]
* [300W-LP & AFLW2000: (CVPR2016) Face Alignment Across Large Poses: A 3D Solution](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)[`pitch-yaw-roll`]
* [LPHD: (ICME2019) LPHD: A Large-Scale Head Pose Dataset for RGB Images](https://ieeexplore.ieee.org/abstract/document/8784950)[`pitch-yaw-roll`][`un-released`]
* [S-HOCK: (CVIU2017) The S-Hock dataset: A new benchmark for spectator crowd analysis](https://www.sciencedirect.com/science/article/pii/S1077314217300024)[[paper link](https://iris.unitn.it/retrieve/handle/11572/187463/470794/Shock_r2.pdf)][`far left, left, frontal, right, far right, away, down`]
* [SynHead: (CVPR2017) Dynamic Facial Analysis: From Bayesian Filtering to Recurrent Neural Network](https://research.nvidia.com/publication/2017-07_dynamic-facial-analysis-bayesian-filtering-recurrent-neural-networks)[[paper link](https://openaccess.thecvf.com/content_cvpr_2017/html/Gu_Dynamic_Facial_Analysis_CVPR_2017_paper.html)][`NVIDIA Synthetic Head Dataset (SynHead)`]

#### Papers(Survey)

* ⭐**Survey(TPAMI2009)** Head Pose Estimation in Computer Vision: A Survey [[paper link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4497208)][[CSDN blog](https://blog.csdn.net/weixin_41703033/article/details/83215043)]

* **Survey(SPI2021)** Head pose estimation: A survey of the last ten years [[paper link](https://www.sciencedirect.com/science/article/abs/pii/S0923596521002332)]

* **Survey(PR2022)** Head pose estimation: An extensive survey on recent techniques and applications [[paper link](https://www.sciencedirect.com/science/article/pii/S0031320322000723)]


#### Papers(Journal)

* **HyperFace(TPAMI2017)** HyperFace: A Deep Multi-Task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition [[paper link](https://ieeexplore.ieee.org/abstract/document/8170321)]

* **(Neurocomputing2018)** Appearance based pedestrians head pose and body orientation estimation using deep learning [[paper link](https://www.sciencedirect.com/science/article/pii/S0925231217312869)][`eight orientation bins`]

* **HeadFusion(TPAMI2018)** HeadFusion: 360 Head Pose Tracking Combining 3D Morphable Model and 3D Reconstruction [[paper link](https://www.idiap.ch/~odobez/publications/YuFunesOdobez-PAMI2018.pdf)]

* ⭐**QuatNet(TMM2019)** Quatnet: Quaternion-based head pose estimation with multiregression loss [[paper link](https://ieeexplore.ieee.org/abstract/document/8444061)][`unit quaternion representation`]

* **(IVC2020)** Improving head pose estimation using two-stage ensembles with top-k regression [[paper link](https://www.sciencedirect.com/sdfe/reader/pii/S0262885619304202/pdf)]

* **MLD(TPAMI2020)** Head Pose Estimation Based on Multivariate Label Distribution [[paper link](https://ieeexplore.ieee.org/abstract/document/9217984)]

* ⭐**MNN(TPAMI2021)** Multi-Task Head Pose Estimation in-the-Wild [[paper link](https://bobetocalo.github.io/pdf/paper_pami20.pdf)][[codes|Tensorflow / C++](https://github.com/bobetocalo/bobetocalo_pami20)]

* ⭐**MFDNet(TMM2021)** MFDNet: Collaborative Poses Perception and Matrix Fisher Distribution for Head Pose Estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/9435939/)][`matrix representation`]

* ⭐**2DHeadPose(NN2023)** 2DHeadPose: A simple and effective annotation method for the head pose in RGB images and its dataset [[paper link](https://www.sciencedirect.com/science/article/pii/S0893608022005214)][[codes|official](https://github.com/youngnuaa/2DHeadPose)][`annotation tool, dataset, and source code`]


#### Papers(Conference)

* **(ITSC2014)** Head detection and orientation estimation for pedestrian safety [[paper link](https://www.mrt.kit.edu/z/publ/download/2014/RehderKloedenStiller2014itsc.pdf)]

* **Dlib(68 points)(CVPR2014)** One Millisecond Face Alignment with an Ensemble of Regression Trees [[paper link](https://openaccess.thecvf.com/content_cvpr_2014/html/Kazemi_One_Millisecond_Face_2014_CVPR_paper.html)]

* ⭐**3DDFA(CVPR2016)** Face Alignment Across Large Poses: A 3D Solution [[paper link](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhu_Face_Alignment_Across_CVPR_2016_paper.html)]

* ⭐**FAN(12 points)(ICCV2017)** How Far Are We From Solving the 2D & 3D Face Alignment Problem? (And a Dataset of 230,000 3D Facial Landmarks) [[paper link](https://openaccess.thecvf.com/content_iccv_2017/html/Bulat_How_Far_Are_ICCV_2017_paper.html)]

* **KEPLER(FG2017)** KEPLER: Keypoint and Pose Estimation of Unconstrained Faces by Learning Efficient H-CNN Regressors [[paper link](https://ieeexplore.ieee.org/abstract/document/7961750)]

* **FasterRCNN+regression(ACCV2018)** Simultaneous Face Detection and Head Pose Estimation: A Fast and Unified Framework [[paper link](https://link.springer.com/content/pdf/10.1007%2F978-3-030-20887-5_12.pdf)][dataset|[AFW](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.661.3510&rep=rep1&type=pdf) and [ALFW](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.384.2988&rep=rep1&type=pdf) dataset: from coarse face pose by using Subcategory to generate 12 clusters to fine Euler angles prediction][`following the HyperFace`]

* **WNet(ACCVW2018)** WNet: Joint Multiple Head Detection and Head Pose Estimation from a Spectator Crowd Image [[paper link](https://stevenputtemans.github.io/AMV2018/presentations/wnet_presentation.pdf)][[dataset|spectator crowd S-HOCK dataset: rough orientation labels](https://iris.unitn.it/retrieve/handle/11572/187463/470794/Shock_r2.pdf)]

* **SSR-Net-MD(IJCAI2018)** SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation [[paper link](https://www.ijcai.org/proceedings/2018/0150.pdf)][[codes|Tensorflow+Dlib+MTCNN](https://github.com/shamangary/SSR-Net)][`Inspiring the FSA-Net`]

* ⭐**HopeNet(CVPRW2018)** Fine-Grained Head Pose Estimation Without Keypoints [[arxiv link](https://arxiv.org/abs/1710.00925)][[Codes|PyTorch(official)](https://github.com/natanielruiz/deep-head-pose)][[CSDN blog](https://blog.csdn.net/qq_42189368/article/details/84849638)]

* **HeadPose(FG2019)** Improving Head Pose Estimation with a Combined Loss and Bounding Box Margin Adjustment [[paper link](https://ieeexplore.ieee.org/abstract/document/8756605)][[codes|TensorFlow](https://github.com/MingzhenShao/HeadPose)]

* ⭐**FSA-Net(CVPR2019)** FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image [[paper link](https://github.com/shamangary/FSA-Net/blob/master/0191.pdf)][[Codes|Keras&Tensorflow(official)](https://github.com/shamangary/FSA-Net)][[Codes|PyTorch(unofficial)](https://github.com/omasaht/headpose-fsanet-pytorch)]

* **PADACO(ICCV2019)** Deep Head Pose Estimation Using Synthetic Images and Partial Adversarial Domain Adaption for Continuous Label Spaces [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Kuhnke_Deep_Head_Pose_Estimation_Using_Synthetic_Images_and_Partial_Adversarial_ICCV_2019_paper.html)][[project link](http://www.tnt.uni-hannover.de/papers/view_paper.php?id=1419)][`SynHead and BIWI --> SynHead++, SynBiwi+, Biwi+`]

* ⭐**WHENet(BMVC2020)** WHENet: Real-time Fine-Grained Estimation for Wide Range Head Pose [[arxiv link](https://arxiv.org/abs/2005.10353)][[Codes|Kears&tensorflow(official)](https://github.com/Ascend-Research/HeadPoseEstimation-WHENet)][[codes|PyTorch(unofficial)](https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino)][[codes|DMHead(unofficial)](https://github.com/PINTO0309/DMHead)]

* **RAFA-Net(ACCV2020)** Rotation Axis Focused Attention Network (RAFA-Net) for Estimating Head Pose [[paper link](https://openaccess.thecvf.com/content/ACCV2020/html/Behera_Rotation_Axis_Focused_Attention_Network_RAFA-Net_for_Estimating_Head_Pose_ACCV_2020_paper.html)][[codes|keras+tensorflow](https://github.com/ArdhenduBehera/RAFA-Net)]

* ⭐**FDN(AAAI2020)** FDN: Feature decoupling network for head pose estimation [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/6974)]

* **Rankpose(BMVC2020)** RankPose: Learning Generalised Feature with Rank Supervision for Head Pose Estimation [[paper link](https://www.bmvc2020-conference.com/assets/papers/0401.pdf)][[codes|PyTorch](https://github.com/seathiefwang/RankPose)][`vector representation`]

* ⭐**3DDFA_V2(ECCV2020)** Towards Fast, Accurate and Stable 3D Dense Face Alignment [[paper link](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640154.pdf)][[codes|PyTorch 3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)][`3D Dense Face Alignment`, `3D Face Reconstruction`, `3DMM`, `Lightweight`]

* **EVA-GCN(CVPRW2021)** EVA-GCN: Head Pose Estimation Based on Graph Convolutional Networks [[paper link](http://openaccess.thecvf.com/content/CVPR2021W/AMFG/html/Xin_EVA-GCN_Head_Pose_Estimation_Based_on_Graph_Convolutional_Networks_CVPRW_2021_paper.html)][[codes|PyTorch](https://github.com/stoneMo/EVA-GCN)]

* ⭐**TriNet(WACV2021)** A Vector-Based Representation to Enhance Head Pose Estimation
 [[paper link](http://openaccess.thecvf.com/content/WACV2021/html/Chu_A_Vector-Based_Representation_to_Enhance_Head_Pose_Estimation_WACV_2021_paper.html)][[codes|Tensorflow+Keras](https://github.com/anArkitek/TriNet_WACV2021)][`vector representation`]

* ⭐**img2pose(CVPR2021)** img2pose: Face Alignment and Detection via 6DoF, Face Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Albiero_img2pose_Face_Alignment_and_Detection_via_6DoF_Face_Pose_Estimation_CVPR_2021_paper.html)][[codes|PyTorch](http://github.com/vitoralbiero/img2pose)]

* ⭐**OsGG-Net(ACMMM2021)** OsGG-Net: One-step Graph Generation Network for Unbiased Head Pose Estimation [[paper link](https://dl.acm.org/doi/abs/10.1145/3474085.3475417)][[codes|PyTorch](https://github.com/stoneMo/OsGG-Net)]

* **(KSE2021)** Simultaneous face detection and 360 degree head pose estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/9648838)]【文章使用了FPN+Multi-task的方式，同时检测人头和识别人头姿态，数据集主要使用了CMU-Panoptic，300WLP和BIWI。头姿表示形式上，除了欧拉角，还使用了Rotation Matrix】

* **(KSE2021)** UET-Headpose: A sensor-based top-view head pose dataset [[paper link](https://ieeexplore.ieee.org/abstract/document/9648656)] 【全文均在阐述获取数据集的硬件系统，但数据集未公布；HPE算法为FSA-Net，并根据WHENet中的思路拓展为full-range 360°单人头部姿态估计方法】

* **(FG2021)** Relative Pose Consistency for Semi-Supervised Head Pose Estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/9666992/)]

* ⭐**SynergyNet(3DV2021)** Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry [[paper link](https://www.computer.org/csdl/proceedings-article/3dv/2021/268800a453/1zWEnuGbFte)][[project link](https://choyingw.github.io/works/SynergyNet)][[codes|PyTorch](https://github.com/choyingw/SynergyNet)]

* ⭐**MOS(BMVC2021)** MOS: A Low Latency and Lightweight Framework for Face Detection, Landmark Localization, and Head Pose Estimation [[paper link](https://www.bmvc2021-virtualconference.com/assets/papers/0580.pdf)][[codes|PyTorch](https://github.com/lyp-deeplearning/MOS-Multi-Task-Face-Detect)][`re-annotate the WIDER FACE with head pose label`]

* **LwPosr(WACV2022)** LwPosr: Lightweight Efficient Fine Grained Head Pose Estimation [[paper link](https://openaccess.thecvf.com/content/WACV2022/html/Dhingra_LwPosr_Lightweight_Efficient_Fine_Grained_Head_Pose_Estimation_WACV_2022_paper.html)]

* **HHP-Net(WACV2022)** HHP-Net: A Light Heteroscedastic Neural Network for Head Pose Estimation With Uncertainty [[paper link](https://openaccess.thecvf.com/content/WACV2022/html/Cantarini_HHP-Net_A_Light_Heteroscedastic_Neural_Network_for_Head_Pose_Estimation_WACV_2022_paper.html)][[codes|TensorFlow](https://github.com/cantarinigiorgio/HHP-Net)]

* ⭐**6DRepNet(ICIP2022)** 6D Rotation Representation For Unconstrained Head Pose Estimation [[paper link](https://arxiv.org/abs/2202.12555)][[codes|PyTorch+RepVGG](https://github.com/thohemp/6DRepNet)][`vector representation`]

* ⭐**DAD-3DNet(CVPR2022)** DAD-3DHeads: A Large-scale Dense, Accurate and Diverse Dataset for 3D Head Alignment from a Single Image [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Martyniuk_DAD-3DHeads_A_Large-Scale_Dense_Accurate_and_Diverse_Dataset_for_3D_CVPR_2022_paper.html)][[project link👍](https://www.pinatafarm.com/research/dad-3dheads)][[codes|official PyTorch](https://github.com/PinataFarms/DAD-3DHeads)][[benchmark challenge👍](https://github.com/PinataFarms/DAD-3DHeads/tree/main/dad_3dheads_benchmark)][`DAD-3DHeads dataset`][By `pinatafarm`]


