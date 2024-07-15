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
  * **[▶ Crowd Counting](#-Crowd-Counting)**
  * **[▶ Eye Gaze Estimation](#-Eye-Gaze-Estimation)**
  * **[▶ Face Alignment](#-Face-Alignment)**
  * **[▶ Face Detection](#-Face-Detection)**
  * **[▶ Face Recognition](#-Face-Recognition)**
  * **[▶ Face Reconstruction (3D)](#-Face-Reconstruction-3D)**
  * **[▶ Hand/Head/Person Detection](#-HandHeadPerson-Detection)**
  * **[▶ Hand Pose Estimation](#-Hand-Pose-Estimation)**
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

### ▶ Crowd Counting

#### Materials


#### Papers 

* **DM-Count(NIPS2020)** Distribution Matching for Crowd Counting [[paper link](https://proceedings.neurips.cc/paper_files/paper/2020/file/118bd558033a1016fcc82560c65cca5f-Paper.pdf)][[arxiv link](https://arxiv.org/pdf/2009.13077.pdf)][[code|official](https://github.com/cvlab-stonybrook/DM-Count)][[CVLab@StonyBrook](https://github.com/cvlab-stonybrook)]

* **LearningToCountEverything(CVPR2021)** Learning To Count Everything [[arxiv link](https://arxiv.org/pdf/2104.08391.pdf)][[code|official](https://github.com/cvlab-stonybrook/LearningToCountEverything)][[CVLab@StonyBrook](https://github.com/cvlab-stonybrook)]

* **CrowdCounting-P2PNet(ICCV2021 Oral)** Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework [[paper link](https://arxiv.org/abs/2107.12746)][[code|official](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)][`Tencent Youtu Research`]

* **ZeroShotCounting(CVPR2023)** Zero-shot Object Counting [[arxiv link](https://arxiv.org/abs/2303.02001)] [[code|official](https://github.com/cvlab-stonybrook/zero-shot-counting)][[CVLab@StonyBrook](https://github.com/cvlab-stonybrook)]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Eye Gaze Estimation

#### Materials

* [Gaze CVPR Workshop: International Workshop on Gaze Estimation and Prediction in the Wild](https://gazeworkshop.github.io/2023/)

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

* **PARKS-Gaze(arxiv2023)** Towards Precision in Appearance-based Gaze Estimation in the Wild [[paper link](https://arxiv.org/abs/2302.02353)][[code|official](https://github.com/lrdmurthy/PARKS-Gaze)][`PARKS-Gaze` dataset]

* **CUDA-GHR(WACV2023)** CUDA-GHR: Controllable Unsupervised Domain Adaptation for Gaze and Head Redirection [[paper link](https://arxiv.org/abs/2106.10852)][[code|official](https://github.com/jswati31/cuda-ghr)]

* 👍**PJAE(ICCV2023)** Interaction-aware Joint Attention Estimation Using People Attributes [[paper link]()][[arxiv link link](https://arxiv.org/abs/2308.05382)][[project link](https://www.toyota-ti.ac.jp/Lab/Denshi/iim/ukita/selection/ICCV2023-PJAE.html)][[code|official](https://github.com/chihina/PJAE)][`Japan`, `Toyota Technological Institute and University of Hyogo`]


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

* ❤**SPIGA(BMVC2022)** Shape Preserving Facial Landmarks with Graph Attention Networks [[paper link](https://arxiv.org/abs/2210.07233)][[project link](https://bmvc2022.mpi-inf.mpg.de/155/)][[codes|official PyTorch](https://github.com/andresprados/SPIGA)]


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
* [(github) awesome 3d human reconstruction --> 3d_human_face](https://github.com/rlczddl/awesome-3d-human-reconstruction?tab=readme-ov-file#3d_human_face)

#### Datasets

* [**Papers With Code Ranks**][[NoW Benchmark](https://paperswithcode.com/dataset/now-benchmark)] [[FaceScape](https://paperswithcode.com/dataset/facescape)] [[D3DFACS](https://paperswithcode.com/dataset/d3dfacs)] [[AFLW2000-3D](https://paperswithcode.com/dataset/aflw2000-3d)]
* [**CelebA**] [(ICCV2015) Deep Learning Face Attributes in the Wild](https://openaccess.thecvf.com/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html) [[project link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)] [[zhihu-zhuanlan](https://zhuanlan.zhihu.com/p/35975956)] [(ICLR2018 by NVIDIA) [CelebA-HQ (paperswithcode)](https://paperswithcode.com/dataset/celeba-hq), [CelebA-HQ (tensorflow-download)](https://www.tensorflow.org/datasets/catalog/celeb_a_hq), [CelebA-HQ (how to generate this dataset?)](https://zhuanlan.zhihu.com/p/52188519), [CelebA-HQ (upload by somebody)](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P)] [(CVPR2020 by MMLab) [CelebAMask-HQ (codes)](https://github.com/switchablenorms/CelebAMask-HQ)] [(CVPR2021 by MMLab) [Multi-Modal-CelebA-HQ (codes)](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset)] [`not a face reconstruction dataset`]
* [**Feng et al. using Stirling meshes (Stirling/ESRC Benchmark)**] [(FG2018) Evaluation of Dense 3D Reconstruction from 2D Face Images in the Wild](https://ieeexplore.ieee.org/abstract/document/8373916) [[pdf page](https://arxiv.org/pdf/2204.06607.pdf)]
* [**NoW ("Not quite in-the-Wild")**] [RingNet(CVPR2019) Learning to Regress 3D Face Shape and Expression from an Image without 3D Supervision](https://ringnet.is.tue.mpg.de/index.html) [[NoW Challenge](https://now.is.tue.mpg.de/nonmetricalevaluation.html)]
* [**FaceScape**] [FaceScape(CVPR2020) FaceScape: A Large-Scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction](https://facescape.nju.edu.cn/)
* [**FaceSynthetics**] [FaceSynthetics(ICCV2021) Fake It Till You Make It: Face analysis in the wild using synthetic data alone](https://microsoft.github.io/FaceSynthetics/) [`synthetic face image with 70 landmarks`]
* [**DAD-3DHeads**] [DAD-3DNet(CVPR2022) DAD-3DHeads: A Large-scale Dense, Accurate and Diverse Dataset for 3D Head Alignment from a Single Image](https://www.pinatafarm.com/research/dad-3dheads)
* [**Multiface**] [Multiface(arxiv2022.07) Multiface: A Dataset for Neural Face Rendering](https://arxiv.org/abs/2207.11243) [[github link](https://github.com/facebookresearch/multiface)] [`Facebook`]

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

* 👍**H3D-Net(ICCV2021)** H3D-Net: Few-Shot High-Fidelity 3D Head Reconstruction [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Ramon_H3D-Net_Few-Shot_High-Fidelity_3D_Head_Reconstruction_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2107.12512)][[project link](https://crisalixsa.github.io/h3d-net/)][[`H3DS Dataset`](https://github.com/CrisalixSA/h3ds)]

* **(ICCV2021)** Towards High Fidelity Monocular Face Reconstruction with Rich Reflectance using Self-supervised Learning and Ray Tracing [[paper link](https://arxiv.org/abs/2103.15432)][`MPII 马普所`]

* **ToFu(ICCV2021)** ToFu: Topologically Consistent Multi-View Face Inference Using Volumetric Sampling [[paper link](http://openaccess.thecvf.com/content/ICCV2021/html/Li_Topologically_Consistent_Multi-View_Face_Inference_Using_Volumetric_Sampling_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2110.02948)][[project link](https://tianyeli.github.io/tofu)][[code|official](https://github.com/tianyeli/tofu)][`USC Institute for Creative Technologies` and `MPII 马普所`]

* **HIFI3D(TOG2021)** High-Fidelity 3D Digital Human Head Creation from RGB-D Selfies [[paper link](https://dl.acm.org/doi/abs/10.1145/3472954)][[project link](https://tencent-ailab.github.io/hifi3dface_projpage/)][[codes|official](https://github.com/tencent-ailab/hifi3dface)][`3DMM of face/head (HIFI3D)`][By `Tencent`]

* ⭐**DECA(TOG2021)(SIGGRAPH2021)** Learning an animatable detailed 3D face model from in-the-wild images [[paper link](https://dl.acm.org/doi/abs/10.1145/3450626.3459936)][[project link](https://deca.is.tue.mpg.de/)][[code|official](https://github.com/YadiraF/DECA)][`MPII 马普所`]

* **(TOG2021)** Semi-supervised video-driven facial animation transfer for production [[paper link](https://dl.acm.org/doi/abs/10.1145/3478513.3480515)][`Digital Domain`, `transfer of facial expressions`, based on `unsupervised image-to-image translation`]

* 👍**FOCUS(arxiv2021)** To fit or not to fit: Model-based Face Reconstruction and Occlusion Segmentation from Weak Supervision [[paper link](https://arxiv.org/abs/2106.09614)][[code|official](https://github.com/unibas-gravis/Occlusion-Robust-MoFA)]

* ⭐**DAD-3DNet(CVPR2022)** DAD-3DHeads: A Large-scale Dense, Accurate and Diverse Dataset for 3D Head Alignment from a Single Image [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Martyniuk_DAD-3DHeads_A_Large-Scale_Dense_Accurate_and_Diverse_Dataset_for_3D_CVPR_2022_paper.html)][[project link👍](https://www.pinatafarm.com/research/dad-3dheads)][[codes|official PyTorch](https://github.com/PinataFarms/DAD-3DHeads)][[benchmark challenge👍](https://github.com/PinataFarms/DAD-3DHeads/tree/main/dad_3dheads_benchmark)][`DAD-3DHeads dataset`][By `pinatafarm`]

* **ImFace(CVPR2022)** ImFace: A Nonlinear 3D Morphable Face Model with Implicit Neural Representations [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_ImFace_A_Nonlinear_3D_Morphable_Face_Model_With_Implicit_Neural_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2203.14510)][[code | official](https://github.com/MingwuZheng/ImFace)][`Beihang University`]

* **REALY(ECCV2022)** REALY: Rethinking the Evaluation of 3D Face Reconstruction [[paper link](https://arxiv.org/abs/2203.09729)][[project link](https://www.realy3dface.com/)][[codes|official](https://github.com/czh-98/REALY)][[blogs|zhihu](https://zhuanlan.zhihu.com/p/549704170)][`3DMM of face/head (HIFI3D++)` and `3D face dataset (REALY)`][By `Tsinghua`]

* **DenseLandmarks(ECCV2022)** 3D Face Reconstruction with Dense Landmarks [[paper link](https://arxiv.org/abs/2204.02776)][[project link](https://microsoft.github.io/DenseLandmarks/)][`Microsoft`]

* **MICA(ECCV2022)** Towards Metrical Reconstruction of Human Faces [[paper link](https://arxiv.org/abs/2204.06607)][[project link](https://zielon.github.io/mica/)][[code|official](https://github.com/Zielon/MICA)][[used multiple datasets](https://github.com/Zielon/MICA/tree/master/datasets/)][`SoTA results in NoW`][`MPII 马普所`]

* **JMLR(ECCVW2022)** Perspective Reconstruction of Human Faces by Joint Mesh and Landmark Regression [[paper link](https://arxiv.org/abs/2208.07142)][[code|official](https://github.com/deepinsight/insightface/tree/master/reconstruction/jmlr)]

* ⭐**DSFNet(CVPR2023)** DSFNet: Dual Space Fusion Network for Occlusion-Robust Dense 3D Face Alignment [[paper link](http://openaccess.thecvf.com/content/CVPR2023/html/Li_DSFNet_Dual_Space_Fusion_Network_for_Occlusion-Robust_3D_Dense_Face_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2305.11522)][[paperwithcode link](https://paperswithcode.com/paper/dsfnet-dual-space-fusion-network-for-1)][[code|official](https://github.com/lhyfst/DSFNet)][`Head Pose Estimation` + `Face Alignment` + `3D Face Reconstruction`]

* **TEMPEH(CVPR2023)** Instant Multi-View Head Capture Through Learnable Registration [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Bolkart_Instant_Multi-View_Head_Capture_Through_Learnable_Registration_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2306.07437)][[project link](https://tempeh.is.tue.mpg.de/)][[code|official](https://github.com/TimoBolkart/TEMPEH)][`MPII 马普所`, based on `ToFu(ICCV2021)`]

* **3DDFA+ & DAD-3DNet+ (CVPR2023)** 3D-Aware Facial Landmark Detection via Multi-View Consistent Training on Synthetic Data [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Zeng_3D-Aware_Facial_Landmark_Detection_via_Multi-View_Consistent_Training_on_Synthetic_CVPR_2023_paper.html)][[project link](https://people.engr.tamu.edu/nimak/Papers/CVPR2023_Landmark/index.html)][`Texas A&M University`, new dataset `DAD-3DHeads-Syn` based on `NeRF`]

* **FOCUS(CVPR2023)** Robust Model-based Face Reconstruction through Weakly-Supervised Outlier Segmentation [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Robust_Model-Based_Face_Reconstruction_Through_Weakly-Supervised_Outlier_Segmentation_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2106.09614)][[code|official](https://github.com/unibas-gravis/Occlusion-Robust-MoFA)][the accepted paper of `FOCUS(arxiv2021)`, `Weakly-Supervised Learning`]

* **TokenHead (ICCV2023)** Accurate 3D Face Reconstruction with Facial Component Tokens [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Accurate_3D_Face_Reconstruction_with_Facial_Component_Tokens_ICCV_2023_paper.html)][`THU(Shenzhen)` + `IDEA`]

* **HiFace (ICCV2023)** HiFace: High-Fidelity 3D Face Reconstruction by Learning Static and Dynamic Details [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Chai_HiFace_High-Fidelity_3D_Face_Reconstruction_by_Learning_Static_and_Dynamic_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.11225)][[project link](https://project-hiface.github.io/)][`MicroSoft`, based on `DenseLandmarks(ECCV2022)`]

* **SIRA++(arxiv2023.10)** Implicit Shape and Appearance Priors for Few-Shot Full Head Reconstruction [[arxiv link](https://arxiv.org/abs/2310.08784)][`few-shot learning`][extended journal on `SIRA(WACV2023)` --> SIRA: Relightable Avatars From a Single Image [[paper link](https://openaccess.thecvf.com/content/WACV2023/html/Caselles_SIRA_Relightable_Avatars_From_a_Single_Image_WACV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2209.03027)]]

* **3DDFA-V3(arxiv2023.12)(CVPR2024)** 3D Face Reconstruction with the Geometric Guidance of Facial Part Segmentation [[arxiv link](https://arxiv.org/abs/2312.00311)][[code|official](https://github.com/wang-zidu/3DDFA-V3)][tested on the dataset `REALY`]

* **PPR-CNet(Computers & Graphics 2023)(CCF C)** 3D face reconstruction from a single image based on hybrid-level contextual information with weak supervision [[paper link](https://www.sciencedirect.com/science/article/pii/S0097849323002881)][`no code is available`, `Xinjiang University`]

* **ImFace++(arxiv2023.12)** ImFace++: A Sophisticated Nonlinear 3D Morphable Face Model with Implicit Neural Representations [[arxiv link](https://arxiv.org/abs/2312.04028)][[code|official](https://github.com/MingwuZheng/ImFace/tree/imface%2B%2B)][`Beihang University`, the extended journal version of `ImFace`]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Hand/Head/Person Detection

### Materials

* [(zhihu) 一文读懂YOLO V5 与 YOLO V4](https://zhuanlan.zhihu.com/p/161083602?d=1605663864267)
* [(zhihu) 如何评价YOLOv5？](https://www.zhihu.com/question/399884529)
* [(csdn blog) YOLO/V1、V2、V3目标检测系列介绍](https://blog.csdn.net/qq26983255/article/details/82119232)
* [(csdn blog) 睿智的目标检测26——Pytorch搭建yolo3目标检测平台](https://blog.csdn.net/weixin_44791964/article/details/105310627)
* [(csdn blog) 睿智的目标检测30——Pytorch搭建YoloV4目标检测平台](https://blog.csdn.net/weixin_44791964/article/details/106214657/)
* [**YOLOv5(2020)** YOLOv5 is from the family of object detection architectures YOLO and has no paper](https://github.com/ultralytics/yolov5) [[YOLOv5 Docs](https://docs.ultralytics.com/)]

### Papers

#### ▶Applications

* **ThroughHand (CHI2021)** ThroughHand: 2D Tactile Interaction to Simultaneously Recognize and Touch Multiple Objects [[paper link](https://dl.acm.org/doi/10.1145/3411764.3445530)][`a novel tactile interaction that enables users with visual impairments to interact with multiple dynamic objects in real time`, `utilize the potential of the ` **human tactile sense**, `enable users to perceive the objects using the` **palm**]

* 👍**SoloFinger (CHI2021)** SoloFinger: Robust Microgestures while Grasping Everyday Objects [[paper link](https://dl.acm.org/doi/10.1145/3411764.3445197)][[project link](https://hci.cs.uni-saarland.de/projects/solofinger/)][`Input / Spatial Interaction / Practice Support`, `36 everyday hand-object actions`, `simple SoloFinger gestures can relieve the need for complex finger configurations or delimiting gestures`]

* **Gaze-Supported (CHI2021)** Gaze-Supported 3D Object Manipulation in Virtual Reality [[paper link](https://dl.acm.org/doi/abs/10.1145/3411764.3445343)][`Input / Spatial Interaction / Practice Support`, `investigates integration, coordination, and transition strategies of gaze and hand input for 3D object manipulation in VR`, `help guide the design of future VR systems that incorporate gaze input for 3D object manipulation`]

* **ARnnotate (UIST2022)(CCF-A)** ARnnotate: An Augmented Reality Interface for Collecting Custom Dataset of 3D Hand-Object Interaction Pose Estimation [[paper link](https://dl.acm.org/doi/abs/10.1145/3526113.3545663)][[pdf link](https://engineering.purdue.edu/cdesign/wp/wp-content/uploads/2022/11/ARnnotate_CameraReady.pdf)][`Purdue University`, application in `Augmented Reality`]

* **Ubi-TOUCH (UIST2023)(CCF-A)** Ubi-TOUCH: Ubiquitous Tangible Object Utilization through Consistent Hand-object interaction in Augmented Reality [[paper link](https://dl.acm.org/doi/abs/10.1145/3586183.3606793)][`Purdue University`, application in `Augmented Reality`]

* **InstruMentAR (CHI2023)** InstruMentAR: Auto-Generation of Augmented Reality Tutorials for Operating Digital Instruments Through Recording Embodied Demonstration [[paper link](https://dl.acm.org/doi/abs/10.1145/3544548.3581442)][[pdf link](https://3dvar.com/Liu2023InstruMentAR.pdf)][`Purdue University`, application in `Augmented Reality`]


#### ▶Body/Person
including `Crowd Person Detection`, `Pedestrian Detection`， `Crowded Pedestrian Detection`

* **ReInspect, Lhungarian(CVPR2016)** End-To-End People Detection in Crowded Scenes [[arxiv link](https://arxiv.org/abs/1506.04878)]

* **PRNet(ECCV2020)** Progressive Refinement Network for Occluded Pedestrian Detection [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58592-1_3)][[code|official](https://github.com/sxlpris/PRNet)][for `Crowded Human Detection`]

* **Pedestron(CVPR2021)** Generalizable Pedestrian Detection: The Elephant In The Room [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Hasan_Generalizable_Pedestrian_Detection_The_Elephant_in_the_Room_CVPR_2021_paper.html)][[code|official](https://github.com/hasanirtiza/Pedestron)][`Pedestrian Detection`]

* **OTP-NMS(TIP2023)** OTP-NMS: Towards Optimal Threshold Prediction of NMS for Crowded Pedestrian Detection [[paper link](https://ieeexplore.ieee.org/abstract/document/10130101)][`CrowdHuman and CityPersons datasets`, `HNU`]

* **VLPD(CVPR2023)** VLPD: Context-Aware Pedestrian Detection via Vision-Language Semantic Self-Supervision [[arxiv link](https://arxiv.org/abs/2304.03135)][[code|official](https://github.com/lmy98129/VLPD)][`Vision-Language semantic self-supervision for context-aware Pedestrian Detection`]

* **LSFM (Localized Semantic Feature Mixers)(CVPR2023)** Localized Semantic Feature Mixers for Efficient Pedestrian Detection in Autonomous Driving [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Khan_Localized_Semantic_Feature_Mixers_for_Efficient_Pedestrian_Detection_in_Autonomous_CVPR_2023_paper.html)][`Caltech, CityPersons, Euro City Persons, and TJU-Traffic-Pedestrian datasets`][`LSFM beats the human baseline for the first time in the history of pedestrian detection`]

* **SSCP (Sample Selection for Crowded Pedestrians)(arxiv2023.05)** Selecting Learnable Training Samples is All DETRs Need in Crowded Pedestrian Detection [[arxiv link](https://arxiv.org/abs/2305.10801)][`Crowdhuman and Citypersons datasets`]

* **OPL (Optimal Proposal Learning)(CVPR2023)** Optimal Proposal Learning for Deployable End-to-End Pedestrian Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Song_Optimal_Proposal_Learning_for_Deployable_End-to-End_Pedestrian_Detection_CVPR_2023_paper.html)][[code is not available]()][`BUPT`]

* **LOAF (ICCV2023)** Large-Scale Person Detection and Localization Using Overhead Fisheye Cameras [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Large-Scale_Person_Detection_and_Localization_Using_Overhead_Fisheye_Cameras_ICCV_2023_paper.html)][[project link](https://loafisheye.github.io/)][[arxiv link](https://arxiv.org/abs/2307.08252)][[code|official](https://github.com/BUPT-PRIV/LOAF)][`dataset`, `BUPT-PRIV`]

#### ▶Hand Part
including `Hand Detection`, `Hand Tracking`, `Hand-Object Contact`, `Hand Pressure Estimation`, `Hand-Object Interaction`, `Hand Contact Reconstruction` and `Hand-Object Manipulation`

* **Hand_detection_rotation_estimation(TIP2017)** Joint Hand Detection and Rotation Estimation Using CNN [[paper link](https://ieeexplore.ieee.org/abstract/document/8128503)][[arxiv link](https://arxiv.org/abs/1612.02742)]

* **Hand-CNN(hand_det_attention)(ICCV2019)** Contextual Attention for Hand Detection in the Wild [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Narasimhaswamy_Contextual_Attention_for_Hand_Detection_in_the_Wild_ICCV_2019_paper.html)][[project](https://www3.cs.stonybrook.edu/~cvl/projects/hand_det_attention/)][[code|official](https://github.com/SupreethN/Hand-CNN)]

* ⭐**BodyHands(CVPR2022)** Whose Hands Are These? Hand Detection and Hand-Body Association in the Wild [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Narasimhaswamy_Whose_Hands_Are_These_Hand_Detection_and_Hand-Body_Association_in_CVPR_2022_paper.html)][[project link](http://vision.cs.stonybrook.edu/~supreeth/BodyHands/)][[code|official](https://github.com/cvlab-stonybrook/BodyHands)][[CVLab@StonyBrook](https://github.com/cvlab-stonybrook)][`joint detection of person body and hands`][`BodyHands` dataset]

* ⭐**HandLer(CVPR2022)** Forward Propagation, Backward Regression, and Pose Association for Hand Tracking in the Wild [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Forward_Propagation_Backward_Regression_and_Pose_Association_for_Hand_Tracking_CVPR_2022_paper.html)][[project link](https://vision.cs.stonybrook.edu/~mingzhen/handler/)][[code|official](https://github.com/cvlab-stonybrook/HandLer)][[CVLab@StonyBrook](https://github.com/cvlab-stonybrook)][`YoutubeHands` dataset, Hand-tracking]

#### ▶Head Part
including `Head Detection`, `Head Counting`

* **HollywoodHeads(ICCV2015)** Context-Aware CNNs for Person Head Detection [[paper link](https://openaccess.thecvf.com/content_iccv_2015/html/Vu_Context-Aware_CNNs_for_ICCV_2015_paper.html)][[project link](https://www.di.ens.fr/willow/research/headdetection/)][`It introduces a large dataset with 369,846 human heads annotated in 224,740 movie frames.`]

* **DA-RCNN(arxiv2018)** Double Anchor R-CNN for Human Detection in a Crowd [[arxiv link](https://arxiv.org/abs/1909.09998)][[CSDN blog1](https://blog.csdn.net/Suan2014/article/details/103987896)][[CSDN blog2](https://blog.csdn.net/Megvii_tech/article/details/103485685)]

* **FCHD(arxiv2018,ICIP2019)** FCHD: Fast and accurate head detection in crowded scenes [[arxiv link](https://arxiv.org/abs/1809.08766)][[Codes|PyTorch(official)](https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector)][[CSDN blog](https://blog.csdn.net/javastart/article/details/82865858)]

* **LSC-CNN(TPAMI2020)** Locate, Size and Count: Accurately Resolving People in Dense Crowds via Detection [[arxiv link](https://arxiv.org/abs/1906.07538)][[Codes|Pytorch(official)](https://github.com/val-iisc/lsc-cnn)]

* **PedHunter(AAAI2020)** PedHunter: Occlusion Robust Pedestrian Detector in Crowded Scenes [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/6690)][`joint body-head detection`]

* ⭐**JointDet(AAAI2020)** Relational Learning for Joint Head and Human Detection [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/6691)][[codes|not released](https://github.com/ChiCheng123/JointDet)]

* **FastNFusion(PRCV2021)** Fast and Fusion: Real-Time Pedestrian Detector Boosted by Body-Head Fusion [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-88004-0_6)][`Pedestrian Detector using Body-Head Association`]

* ⭐**BFJDet(ICCV2021)** Body-Face Joint Detection via Embedding and Head Hook [[paper link](https://openaccess.thecvf.com/content/ICCV2021/papers/Wan_Body-Face_Joint_Detection_via_Embedding_and_Head_Hook_ICCV_2021_paper.pdf)][[codes|official](https://github.com/AibeeDetect/BFJDet)][`joint detection of person body, head and face`]

* **HeadHunter(CVPR2021)** Tracking Pedestrian Heads in Dense Crowd [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Sundararaman_Tracking_Pedestrian_Heads_in_Dense_Crowd_CVPR_2021_paper.html)][[project link](https://project.inria.fr/crowdscience/project/dense-crowd-head-tracking/)][[code|official](https://github.com/Sentient07/HeadHunter)][[Head_Tracking_21 challenge](https://motchallenge.net/data/Head_Tracking_21/)][`Pedestrian Tracking`]

* 👍**Head-body-Tracking(arxiv2023.04)** Handling Heavy Occlusion in Dense Crowd Tracking by Focusing on the Heads [[arxiv link](https://arxiv.org/abs/2304.07705)]

* 👍👍**PanoHead(CVPR2023)** PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360∘ [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/An_PanoHead_Geometry-Aware_3D_Full-Head_Synthesis_in_360deg_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.13071)][[project link](https://sizhean.github.io/panohead)][[code|official](https://github.com/SizheAn/PanoHead)]


#### ▶Human Parts
including `Human-Parts Detection`, `Human Activity Understanding`, `Human and Object Reconstruction`, `Human-Aware Object Placement`, `Human-Scene Contact`, `Human-Object Contact`, `Human-Object Interaction Tracking` and `Close Human Interaction`

* **DID-Net(ACCV2018)** Detector-in-Detector: Multi-level Analysis for Human-Parts [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-20890-5_15)][[code | official](https://github.com/xiaojie1017/Human-Parts)][`HumanParts` dataset]

* **PROX(ICCV2019)** Resolving 3D Human Pose Ambiguities with 3D Scene Constraints [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Hassan_Resolving_3D_Human_Pose_Ambiguities_With_3D_Scene_Constraints_ICCV_2019_paper.html)][[project link](https://prox.is.tue.mpg.de./)][`MPII`, `The contact constraint encourages specific parts of the body to be in contact with scene surfaces if they are close enough in distance and orientation.`]

* ⭐**Hier-R-CNN(TIP2020)** Hier R-CNN: Instance-Level Human Parts Detection and A New Benchmark [[paper link](https://ieeexplore.ieee.org/abstract/document/9229236)][[code|official](https://github.com/soeaver/Hier-R-CNN)][`Mask R-CNN` as Backbone][`FCOS` as Hier Branch which needs many hand-crafted tricks][`COCOHumanParts` dataset]

* **ContactDynamics(ECCV2020)** Contact and Human Dynamics from Monocular Video [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58558-7_5)][[project link](https://geometry.stanford.edu/projects/human-dynamics-eccv-2020/)][[code|official](https://github.com/davrempe/contact-human-dynamics)][`Stanford University`, `Adobe Research`]

* **PaStaNet(CVPR2020)** PaStaNet: Toward Human Activity Knowledge Engine [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_PaStaNet_Toward_Human_Activity_Knowledge_Engine_CVPR_2020_paper.html)][[project link](http://hake-mvig.cn/)][`SJTU`, `body-part state annotations in the context of HOI`][`HAKE 1.0` (Human Activity Knowledge Engine) dataset]

* **CHORE(ECCV2022)** CHORE: Contact, Human and Object Reconstruction from a Single RGB Image [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20086-1_8)][[project link](https://virtualhumans.mpi-inf.mpg.de/chore/)][`MPII`, `single-person`, reason the interactions and recover the spatial arrangement, fine-grained contacts between the human and the object]

* **MOVER(CVPR2022)** Human-Aware Object Placement for Visual Environment Reconstruction [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Yi_Human-Aware_Object_Placement_for_Visual_Environment_Reconstruction_CVPR_2022_paper.html)][[project link](https://mover.is.tue.mpg.de/)][[code|official](https://github.com/yhw-yhw/mover)][`human-scene interactions (HSIs)`, `MPII`]

* 👍**BSTRO(Body-Scene contact TRansfOrmer)(CVPR2022)** Capturing and Inferring Dense Full-Body Human-Scene Contact [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Capturing_and_Inferring_Dense_Full-Body_Human-Scene_Contact_CVPR_2022_paper.html)][[project link](https://rich.is.tue.mpg.de/)][[code|official](https://github.com/paulchhuang/bstro)][dataset `RICH`, `Interaction-Contact-Humans`, `MPII`, `single-person`]

* **HAKE(TPAMI2023)** HAKE: A Knowledge Engine Foundation for Human Activity Understanding [[paper link](https://ieeexplore.ieee.org/abstract/document/10002711)][[arxiv link](https://arxiv.org/abs/2202.06851)][[project link](http://hake-mvig.cn/)][`HAKE 2.0` (Human Activity Knowledge Engine) dataset]

* 👍**HOT(CVPR2023)** Detecting Human-Object Contact in Images [[paper link](https://arxiv.org/abs/2303.03373)][[project link](https://hot.is.tue.mpg.de/)][`马普所`, `HOT` dataset, `single-person`]

* **VisTracker(CVPR2023)** Visibility Aware Human-Object Interaction Tracking from Single RGB Camera [[arxiv link](https://arxiv.org/abs/2303.16479)][[project link](https://virtualhumans.mpi-inf.mpg.de/VisTracker/)][`MPII`, An approach to jointly track the human, the object and the contacts between them, in 3D, from a monocular RGB video.]

* **Hi4D(Humans interacting in 4D)(CVPR2023)** Hi4D: 4D Instance Segmentation of Close Human Interaction [[arxiv link](https://arxiv.org/abs/2303.15380)][[project link](https://yifeiyin04.github.io/Hi4D/)][`ETH Zürich`, A dataset of humans in close physical interaction]



**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Hand Pose Estimation
also `2D/3D Hand Keypoints Detection` or `Hand Shape Estimation` or `3D Hand Shape and Pose Regression`

#### Materials

* 👍 **(github)(Hand3DResearch) Recent Progress in 3D Hand Tasks** [[github link](https://github.com/SeanChenxy/Hand3DResearch)]
* **(github) awesome 3d human reconstruction --> 3d_human_hand**  [[github link](https://github.com/rlczddl/awesome-3d-human-reconstruction?tab=readme-ov-file#3d_human_hand)]


#### Datasets

* [**HANDS17: (arxiv2017)**] [The 2017 Hands in the Million Challenge on 3D Hand Pose Estimation](http://icvl.ee.ic.ac.uk/hands17/challenge/) [[arxiv link](https://arxiv.org/abs/1707.02237)][`3D Hand Pose Estimation`, `21 joints`]
* [**FreiHAND: (ICCV2019)**] [FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images](https://lmb.informatik.uni-freiburg.de/projects/freihand/) [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Zimmermann_FreiHAND_A_Dataset_for_Markerless_Capture_of_Hand_Pose_and_ICCV_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1909.04349)][`University of Freiburg`, `A dataset that uses MANO`]
* [**ObMan: (CVPR2019)**] [Learning joint reconstruction of hands and manipulated objects](https://hassony2.github.io/obman) [[paper link](http://openaccess.thecvf.com/content_CVPR_2019/html/Hasson_Learning_Joint_Reconstruction_of_Hands_and_Manipulated_Objects_CVPR_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1904.05767)][`MPII`, `hand-object manipulations`, `A dataset that uses MANO`, `A new large-scale synthetic dataset with hand-object manipulations`]
* [**InterHand2.6M: (ECCV2020)**] [InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image](https://mks0601.github.io/InterHand2.6M/) [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58565-5_33)][[github link](https://github.com/facebookresearch/InterHand2.6M)][`facebookresearch`, `A dataset that uses MANO`]
* [**GanHand or YCB_Affordance: (CVPR2020)**] [GanHand: Predicting Human Grasp Affordances in Multi-Object Scenes](http://www.iri.upc.edu/people/ecorona/ganhand/) [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Corona_GanHand_Predicting_Human_Grasp_Affordances_in_Multi-Object_Scenes_CVPR_2020_paper.html)][[github link (method)](https://github.com/enriccorona/GanHand)][[github link (dataset)](https://github.com/enriccorona/YCB_Affordance)][`A dataset that uses MANO`, `Human Grasp Affordances`]
* [**HO-3D: (CVPR2020)**] [HOnnotate: A method for 3D Annotation of Hand and Object Poses](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation) [[arxiv link](https://arxiv.org/abs/1907.01481)][[github link 1](https://github.com/shreyashampali/HOnnotate)][[github link2](https://github.com/shreyashampali/ho3d)][`The first markerless dataset of color images with 3D annotations of both hand and object`, `This dataset is currently made of 80,000 frames, 65 sequences, 10 persons, and 10 objects`]
* [**YouTube 3D Hands: (CVPR2020 Oral)**] [Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild](https://www.arielai.com/mesh_hands/) [[arxiv link](https://arxiv.org/abs/2004.01946)][[github link](https://github.com/arielai/youtube_3d_hands)][`Ariel AI`]
* [**HanCo: (GCPR2021)**] [Contrastive Representation Learning for Hand Shape Estimation](https://lmb.informatik.uni-freiburg.de/projects/contra-hand/) [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-92659-5_16)][[arxiv link](https://arxiv.org/abs/2106.04324)][`University of Freiburg`, `A dataset that uses MANO`, `An extended version of FreiHAND with calibration and multiple-views`]
* [**DARTset: (NIPS2022)**] [DART: Articulated Hand Model with Diverse Accessories and Rich Textures](https://dart2022.github.io/) [[arxiv link](https://arxiv.org/abs/2210.07650)][[github link](https://github.com/DART2022/DART)][`Alibaba XR Lab + MPII + SJTU`][`A dataset (DARTset) that uses MANO and proposes a new hand morphable model DART`, `for hand pose estimation & surface reconstruction tasks`, `with large-scale (800K), diverse, and high-fidelity hand images, paired with perfect-aligned 3D labels`]


#### Papers

##### ▶Hand Related Survey

* **Survey(IJCV2023)** Efficient Annotation and Learning for 3D Hand Pose Estimation: A Survey [[paper link](https://link.springer.com/article/10.1007/s11263-023-01856-0)][[arxiv link](https://arxiv.org/abs/2206.02257)][[slice link](https://drive.google.com/file/d/15gHEeyeCuzFyGYBMK971U_524M55iR1d/view)][`The code is not available`][`University of Tokyo + ETH`, the first author [`Take Ohkawa (大川 武彦)`](https://tkhkaeio.github.io/)]

##### ▶Hand Modeling Methods

* **MANO (TOG2017, SIGGRAPH ASIA 2017)** Embodied Hands: Modeling and Capturing Hands and Bodies Together [[paper link](https://dl.acm.org/doi/abs/10.1145/3130800.3130883)][[arxiv link](https://arxiv.org/abs/2201.02610)][[project link (keep updating)](http://mano.is.tue.mpg.de/)][`MPII`, `It attempts to learn hand shape variation with Linear Blend Skinning (LBS)` [[SIGGRAPH 2000](https://dl.acm.org/doi/abs/10.1145/344779.344862)]][`it learns from a large variety of high-quality hand scans and represents the geometric changes in the low-dimensional pose and shape space`]

* **NIMBLE (TOG2022)** NIMBLE: A Non-rigid Hand Model with Bones and Muscles [[paper link](https://dl.acm.org/doi/abs/10.1145/3528223.3530079)][[arxiv link](https://arxiv.org/abs/2202.04533)][[project link](https://liyuwei.cc/proj/nimble)][[code|official](https://github.com/reyuwei/NIMBLE_model)][`ShanghaiTech University`]

##### ▶Hand Keypoints Detection

* **hand3d(ICCV2017)** Learning to Estimate 3D Hand Pose From Single RGB Images [[paper link](https://openaccess.thecvf.com/content_iccv_2017/html/Zimmermann_Learning_to_Estimate_ICCV_2017_paper.html)][[arxiv link](https://arxiv.org/abs/1705.01389v3)][[project link](https://lmb.informatik.uni-freiburg.de/projects/hand3d/)][[code|official](https://github.com/lmb-freiburg/hand3d)][`University of Freiburg`, new dataset `Rendered Hand Pose Dataset (RHD)`, `3D Hand Keypoints Detection`]

##### ▶3D Hand Reconstruction
also `3D Hand Shape and Pose Regression`

* **(ECCV2018)** Hand Pose Estimation via Latent 2.5D Heatmap Regression [[paper link](https://openaccess.thecvf.com/content_ECCV_2018/html/Umar_Iqbal_Hand_Pose_Estimation_ECCV_2018_paper.html)][[arxiv link](https://arxiv.org/abs/1804.09534)][`No code is available`, `NVIDIA`]

* **(CVPR2019)** 3D Hand Shape and Pose From Images in the Wild [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Boukhayma_3D_Hand_Shape_and_Pose_From_Images_in_the_Wild_CVPR_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1902.03451)][`No code is available`, based on `MANO`]

* **(CVPR2019)** Pushing the Envelope for RGB-Based Dense 3D Hand Pose Estimation via Neural Rendering [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Baek_Pushing_the_Envelope_for_RGB-Based_Dense_3D_Hand_Pose_Estimation_CVPR_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1904.04196)][`No code is available`, based on `MANO`]

* **Hand+Object(CVPR2019)** H+O: Unified Egocentric Recognition of 3D Hand-Object Poses and Interactions [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Tekin_HO_Unified_Egocentric_Recognition_of_3D_Hand-Object_Poses_and_Interactions_CVPR_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1904.05349)][`No code is available`, `6DoF Object Pose Estimation` + `3D Hand Keypoints Detection`]

* 👍**hand-graph-cnn(CVPR2019)** 3D Hand Shape and Pose Estimation From a Single RGB Image [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Ge_3D_Hand_Shape_and_Pose_Estimation_From_a_Single_RGB_CVPR_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1903.00812)][[code|official](https://github.com/3d-hand-shape/hand-graph-cnn)][based on `MANO`, `2D/3D Hand Keypoints Detection` + `3D Hand Mesh`]

* **HAMR(ICCV2019)** End-to-End Hand Mesh Recovery From a Monocular RGB Image [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_End-to-End_Hand_Mesh_Recovery_From_a_Monocular_RGB_Image_ICCV_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1902.09305)][[code|official](https://github.com/MandyMo/HAMR)][based on `MANO`, `2D/3D Hand Keypoints Detection` + `3D Hand Mesh`]

* 👍**MobileHand(ICONIP2020)** MobileHand: Real-time 3D Hand Shape and Pose Estimation from Color Image [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-63820-7_52)][[project link](https://gmntu.github.io/mobilehand/)][[code|official](https://github.com/gmntu/mobilehand)][`anyang Technological University`][based on `MANO`, `2D/3D Hand Keypoints Detection` + `3D Hand Mesh`]

* **I2L-MeshNet(ECCV2020)** I2L-MeshNet: Image-to-Lixel Prediction Network for Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58571-6_44)][[arxiv link](https://arxiv.org/abs/2008.03713)][[code|official](https://github.com/mks0601/I2L-MeshNet_RELEASE)][`Seoul National University`, `whole body and related hands`]

* **mesh_hands(CVPR2020)** Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Kulon_Weakly-Supervised_Mesh-Convolutional_Hand_Reconstruction_in_the_Wild_CVPR_2020_paper.html)][[arxiv link](https://arxiv.org/abs/2004.01946)][[project link](https://arielai.com/mesh_hands)][based on `MANO`]

* **RGB2Hands(SIGGRAPH Asia 2020)** RGB2Hands: Real-Time Tracking of 3D Hand Interactions from Monocular RGB Video [[paper link](https://dl.acm.org/doi/abs/10.1145/3414685.3417852)][[arxiv link](https://arxiv.org/abs/2106.11725)][[project link](https://handtracker.mpi-inf.mpg.de/projects/RGB2Hands/)][new dataset `RGB2Hands`, based on `MANO`]

* **InterShape(ICCV2021)** Interacting Two-Hand 3D Pose and Shape Reconstruction From Single Color Image [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Interacting_Two-Hand_3D_Pose_and_Shape_Reconstruction_From_Single_Color_ICCV_2021_paper.html)][[pdf link](https://www.yangangwang.com/papers/ZHANG-ITH-2021-08.pdf)][[project link](https://baowenz.github.io/Intershape/)][[code|official](https://github.com/BaowenZ/Two-Hand-Shape-Pose)][`Yangang Wang`, based on `MANO`, using the dataset `InterHand2.6M`]

* 👍**MobRecon(CVPR2022)** MobRecon: Mobile-Friendly Hand Mesh Reconstruction From Monocular Image [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_MobRecon_Mobile-Friendly_Hand_Mesh_Reconstruction_From_Monocular_Image_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2112.02753)][[code|official](https://github.com/SeanChenxy/HandMesh)][`Kuaishou Technology`]

* **IntagHand(CVPR2022)** Interacting Attention Graph for Single Image Two-Hand Reconstruction [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Interacting_Attention_Graph_for_Single_Image_Two-Hand_Reconstruction_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2203.09364)][[code|official](https://github.com/Dw1010/IntagHand)][based on `MANO`, using the dataset `InterHand2.6M`]

* 👍**HandOccNet(CVPR2022)** HandOccNet: Occlusion-Robust 3D Hand Mesh Estimation Network [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Park_HandOccNet_Occlusion-Robust_3D_Hand_Mesh_Estimation_Network_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2203.14564)][[code|official](https://github.com/namepllet/HandOccNet)][based on `MANO`]

* **MeMaHand(CVPR2023)** MeMaHand: Exploiting Mesh-Mano Interaction for Single Image Two-Hand Reconstruction [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_MeMaHand_Exploiting_Mesh-Mano_Interaction_for_Single_Image_Two-Hand_Reconstruction_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.15718)][`ByteDance`, `No code is available`, based on `MANO`, compared to `IntagHand` and `InterShape`]

* **Im2Hands(CVPR2023)** Im2Hands: Learning Attentive Implicit Representation of Interacting Two-Hand Shapes [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Im2Hands_Learning_Attentive_Implicit_Representation_of_Interacting_Two-Hand_Shapes_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2302.14348)][[project link](https://jyunlee.github.io/projects/implicit-two-hands/)][[code|official](https://github.com/jyunlee/Im2Hands)][`KAIST`, compared to `IntagHand`, based on [`HALO: A Skeleton-Driven Neural Occupancy Representation for Articulated Hands (3DV 2021)`](https://github.com/korrawe/halo) and [`Occupancy Networks`](https://github.com/autonomousvision/occupancy_networks)]

* **ACR(CVPR2023)** ACR: Attention Collaboration-Based Regressor for Arbitrary Two-Hand Reconstruction [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_ACR_Attention_Collaboration-Based_Regressor_for_Arbitrary_Two-Hand_Reconstruction_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.05938)][[code|official](https://github.com/ZhengdiYu/Arbitrary-Hands-3D-Reconstruction)][`Tencent AI Lab`, based on `MANO`, compared to `IntagHand`]

* **InterWild(CVPR2023)** Bringing Inputs to Shared Domains for 3D Interacting Hands Recovery in the Wild [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Moon_Bringing_Inputs_to_Shared_Domains_for_3D_Interacting_Hands_Recovery_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.13652)][[code|official](https://github.com/facebookresearch/InterWild)][`facebookresearch`, single author `Gyeongsik Moon`, based on `MANO`, compared to `IntagHand`]

* **H2ONet(CVPR2023)** H2ONet: Hand-Occlusion-and-Orientation-aware Network for Real-time 3D Hand Mesh Reconstruction [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_H2ONet_Hand-Occlusion-and-Orientation-Aware_Network_for_Real-Time_3D_Hand_Mesh_Reconstruction_CVPR_2023_paper.html)][[code|official](https://github.com/hxwork/H2ONet_Pytorch)][`CUHK`, first author [`Hao XU (徐昊)`](https://hxwork.github.io/), tested on datasets `DexYCB` and `HO3D`]

* **DIR (ICCV2023 Oral)** Decoupled Iterative Refinement Framework for Interacting Hands Reconstruction from a Single RGB Image [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Ren_Decoupled_Iterative_Refinement_Framework_for_Interacting_Hands_Reconstruction_from_a_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2302.02410)][[project link](https://pengfeiren96.github.io/DIR/)][[code|official](https://github.com/PengfeiRen96/DIR)][`PICO IDL ByteDance` + `BUPT`]

* **HMP(WACV2024)** HMP: Hand Motion Priors for Pose and Shape Estimation From Video [[paper link](https://openaccess.thecvf.com/content/WACV2024/html/Duran_HMP_Hand_Motion_Priors_for_Pose_and_Shape_Estimation_From_WACV_2024_paper.html)][[project link](https://hmp.is.tue.mpg.de/)][[code|official](https://github.com/enesduran/HMP)][`MPII`, taking video as the input, tested on datasets `HO3D` and `DexYCB`, mainly focusing on `hand occlusions`]

* **Ev2Hands(3DV2024)** 3D Pose Estimation of Two Interacting Hands from a Monocular Event Camera [[arxiv link](https://arxiv.org/abs/2312.14157)][[project link](https://4dqv.mpi-inf.mpg.de/Ev2Hands/)][[code|official](https://github.com/Chris10M/Ev2Hands)][`MPII`, a new `synthetic` large-scale dataset of two interacting hands, `Ev2Hands-S`, and a new real benchmark with real event streams and ground-truth 3D annotations, `Ev2Hands-R`.]

* 👍**HaMeR(CVPR2024)(arxiv2023.12)** Reconstructing Hands in 3D with Transformers [[paper link](http://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2312.05251)][[project link](https://geopavlakos.github.io/hamer/)][[code|official](https://github.com/geopavlakos/hamer)][the first author [`Georgios Pavlakos`](https://geopavlakos.github.io/), `University of California, Berkeley`, a new dataset `HInt` which is built by sampling frames from `New Days of Hands`, `EpicKitchens-VISOR` and `Ego4D` and annotating the hands with `2D keypoints`.]

* **OHTA(CVPR2024)(arxiv2024.02)** OHTA: One-shot Hand Avatar via Data-driven Implicit Priors [[arxiv link](https://arxiv.org/abs/2402.18969)][[project link](https://zxz267.github.io/OHTA/)][[code|official](https://github.com/zxz267/OHTA)][`ByteDance`][To test OHTA’s performance for the challenging `in-the-wild` images, they take the whole-body version of `MSCOCO` for experiments. They utilize the pose estimation results provided by `InterWild` and generate the masks using `SAM`]


##### ▶Sign Language Understanding
including `Sign Language Recognition` and `Sign Language Translation`

* **BSL(ECCV2020)** BSL-1K: Scaling Up Co-articulated Sign Language Recognition Using Mouthing Cues [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58621-8_3)]

* **HMA(AAAI2021)** Hand-Model-Aware Sign Language Recognition [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/16247)][`Sign Language Recognition (SLR)`]

* **SignBERT (ICCV2021)** SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_SignBERT_Pre-Training_of_Hand-Model-Aware_Representation_for_Sign_Language_Recognition_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2110.05382)][`Sign Language Recognition (SLR)`]

* 👍**SignBERT+ (TPAMI2023)** SignBERT+: Hand-model-aware Self-supervised Pre-training for Sign Language Understanding [[paper link](https://ieeexplore.ieee.org/abstract/document/10109128)][[arxvi link](https://arxiv.org/abs/2305.04868)][[project link](https://signbert-zoo.github.io/)][`Sign Language Understanding (SLU)`]


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
* [[Head Pose Estimation on AFLW2000](https://paperswithcode.com/sota/head-pose-estimation-on-aflw2000)], [[Head Pose Estimation on BIWI ranking](https://paperswithcode.com/sota/head-pose-estimation-on-biwi)]
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

* **6dof_face(TIP2023)** Towards 3D Face Reconstruction in Perspective Projection: Estimating 6DoF Face Pose from Monocular Image [[paper link](https://ieeexplore.ieee.org/abstract/document/10127617)][[code|official](https://github.com/cbsropenproject/6dof_face)]

* **CIT(IJCV2023)** Cascaded Iterative Transformer for Jointly Predicting Facial Landmark, Occlusion Probability and Head Pose [[paper link](https://link.springer.com/article/10.1007/s11263-023-01935-2)][[code|official](https://github.com/Iron-LYK/CIT)][`SYSU`, `Facial Landmark` + `Head Pose`]

* **TokenHPE(TIP2023)** Orientation Cues-Aware Facial Relationship Representation for Head Pose Estimation via Transformer [[paper link](https://ieeexplore.ieee.org/abstract/document/10318055)][The journal version of the conference paper `TokenHPE(CVPR2023)`]

* **OPAL(SRHP+WRHP)(PR2024)** On the representation and methodology for wide and short range head pose estimation [[paper link](https://www.sciencedirect.com/science/article/pii/S0031320324000141)][[arxiv link](https://arxiv.org/abs/2401.05807)][`Universidad Politécnica de Madrid`]

* **HeadDiff(TIP2024)** HeadDiff: Exploring Rotation Uncertainty with Diffusion Models for Head Pose Estimation [[paper link](https://ieeexplore.ieee.org/document/10462910)][`Ningxia University`]

* **HHP-Net-Plus(CVIU2024)** Head pose estimation with uncertainty and an application to dyadic interaction detection [[paper link](https://www.sciencedirect.com/science/article/pii/S1077314224000808)][[code|official](https://github.com/Malga-Vision/HHP-Net)][`Università degli Studi di Genova, Italy`, the extended journal of `HHP-Net(WACV2022)`]


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

* **(FG2021)** Relative Pose Consistency for Semi-Supervised Head Pose Estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/9666992/)][[pdf link](https://www.tnt.uni-hannover.de/papers/data/1544/RCRwFG2021.pdf)][`Semi-Supervised`]

* **HeadPosr(FG2021)** HeadPosr: End-to-end Trainable Head Pose Estimation using Transformer Encoders [[paper link](https://ieeexplore.ieee.org/document/9667080)][[arxiv link](https://arxiv.org/abs/2202.03548)][`Naina Dhingra`]

* ⭐**SynergyNet(3DV2021)** Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry [[paper link](https://www.computer.org/csdl/proceedings-article/3dv/2021/268800a453/1zWEnuGbFte)][[project link](https://choyingw.github.io/works/SynergyNet)][[codes|PyTorch](https://github.com/choyingw/SynergyNet)]

* ⭐**MOS(BMVC2021)** MOS: A Low Latency and Lightweight Framework for Face Detection, Landmark Localization, and Head Pose Estimation [[paper link](https://www.bmvc2021-virtualconference.com/assets/papers/0580.pdf)][[codes|PyTorch](https://github.com/lyp-deeplearning/MOS-Multi-Task-Face-Detect)][`re-annotate the WIDER FACE with head pose label`]

* **LwPosr(WACV2022)** LwPosr: Lightweight Efficient Fine Grained Head Pose Estimation [[paper link](https://openaccess.thecvf.com/content/WACV2022/html/Dhingra_LwPosr_Lightweight_Efficient_Fine_Grained_Head_Pose_Estimation_WACV_2022_paper.html)][`Naina Dhingra`]

* **HHP-Net(WACV2022)** HHP-Net: A Light Heteroscedastic Neural Network for Head Pose Estimation With Uncertainty [[paper link](https://openaccess.thecvf.com/content/WACV2022/html/Cantarini_HHP-Net_A_Light_Heteroscedastic_Neural_Network_for_Head_Pose_Estimation_WACV_2022_paper.html)][[codes|TensorFlow](https://github.com/cantarinigiorgio/HHP-Net)]

* ⭐**6DRepNet(ICIP2022)** 6D Rotation Representation For Unconstrained Head Pose Estimation [[paper link](https://arxiv.org/abs/2202.12555)][[codes|PyTorch+RepVGG](https://github.com/thohemp/6DRepNet)][Journal Version ([6DRepNet360](https://github.com/thohemp/6DRepNet360)) --> [`Towards Robust and Unconstrained Full Range of Rotation Head Pose Estimation`](https://arxiv.org/abs/2309.07654)][`vector representation`]

* ⭐**DAD-3DNet(CVPR2022)** DAD-3DHeads: A Large-scale Dense, Accurate and Diverse Dataset for 3D Head Alignment from a Single Image [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Martyniuk_DAD-3DHeads_A_Large-Scale_Dense_Accurate_and_Diverse_Dataset_for_3D_CVPR_2022_paper.html)][[project link👍](https://www.pinatafarm.com/research/dad-3dheads)][[codes|official PyTorch](https://github.com/PinataFarms/DAD-3DHeads)][[benchmark challenge👍](https://github.com/PinataFarms/DAD-3DHeads/tree/main/dad_3dheads_benchmark)][`DAD-3DHeads dataset`, by `pinatafarm`][used as an `off-the-shelf head pose estimator` in [HairNeRF(ICCV2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Chang_HairNeRF_Geometry-Aware_Image_Synthesis_for_Hairstyle_Transfer_ICCV_2023_paper.html)]

* **TokenHPE(CVPR2023)** TokenHPE: Learning Orientation Tokens for Efficient Head Pose Estimation via Transformers [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_TokenHPE_Learning_Orientation_Tokens_for_Efficient_Head_Pose_Estimation_via_CVPR_2023_paper.html)][[code|official](https://github.com/zc2023/TokenHPE)][`Transformer-based method`]

* ⭐**DSFNet(CVPR2023)** DSFNet: Dual Space Fusion Network for Occlusion-Robust Dense 3D Face Alignment [[paper link](http://openaccess.thecvf.com/content/CVPR2023/html/Li_DSFNet_Dual_Space_Fusion_Network_for_Occlusion-Robust_3D_Dense_Face_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2305.11522)][[paperwithcode link](https://paperswithcode.com/paper/dsfnet-dual-space-fusion-network-for-1)][[code|official](https://github.com/lhyfst/DSFNet)][`Head Pose Estimation` + `Face Alignment` + `3D Face Reconstruction`]

* **PFA(arxiv2023.08)** 3D Face Alignment Through Fusion of Head Pose Information and Features [[arxiv link](https://arxiv.org/abs/2308.13327)][`Soongsil University`]

* **OrdinalRegression(ICASSP2024)** Language-Driven Ordinal Learning for Imbalanced Head Pose Estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/10448404)][`Ningxia University`]

* **StructuredLight(ICASSP2024)** Adaptive Head Pose Estimation with Real-Time Structured Light [[paper link](https://ieeexplore.ieee.org/abstract/document/10446561)][`Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, China`]

* **FaceXFormer(arxiv2024.03)** FaceXFormer: A Unified Transformer for Facial Analysis [[arxiv link](https://arxiv.org/abs/2403.12960)][[project link](https://kartik-3004.github.io/facexformer_web/)][[code|official](https://github.com/Kartik-3004/facexformer)][`Johns Hopkins University`]


