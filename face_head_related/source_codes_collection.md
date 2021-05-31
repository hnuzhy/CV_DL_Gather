#  List of public algorithms and datasets

## 1) Pubilc Datasets and Challenges

* [BIWI RGBD-ID Dataset](http://robotics.dei.unipd.it/reid/index.php)
* [300W-3D & 300W-3D-Face & 300W-LP & AFLW2000-3D](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3ddfa/main.htm)


## 2) Pioneers and Experts

[👍Alejandro Newell](https://www.alejandronewell.com/)



## 3) Blogs, Videos and Applications
 
* [(Website) Face Alignment Across Large Poses: A 3D Solution (official website)](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3ddfa/main.htm)
* [(blog) LFFD 再升级！新增行人和人头检测模型，及优化的C++实现](https://www.zhuanzhi.ai/document/d36c78507cc5d09dcac3fb7241344f3b)


## 4) Papers and Sources Codes

### ▶ Beautify Face

#### Materials

* [(github) BeautifyFaceDemo](https://github.com/Guikunzhi/BeautifyFaceDemo)
* [(CSDN blogs) 图像滤镜艺术---换脸算法资源收集](https://blog.csdn.net/scythe666/article/details/81021041)

#### Papers

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Face Alignment

#### Materials

* [(jianshu) 人脸关键点对齐](https://www.jianshu.com/p/e4b9317a817f)
* Procrustes Analysis [[CSDN blog](https://blog.csdn.net/u011808673/article/details/80733686)][[wikipedia](https://en.wikipedia.org/wiki/Procrustes_analysis)][[scipy.spatial.procrustes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html)][[github](https://github.com/Ahmer-444/Action-Recognition-ProcrustesAnalysis)]
* (website) Greek Mythology 浙江大学数学科学学院希腊神话浙江大学数学科学学院.ppt [[Procrustes Analysis and its application in computer graphaics](https://max.book118.com/html/2017/0307/94565569.shtm)]
* [(github) ASM-for-human-face-feature-points-matching](https://github.com/JiangtianPan/ASM-for-human-face-feature-points-matching)
* [(github) align_dataset_mtcnn](https://github.com/davidsandberg/facenet/blob/master/src/align/align_dataset_mtcnn.py)

#### Papers

* **3000FPS(CVPR2014)** Face Alignment at 3000 FPS via Regressing Local Binary Features [[paper link](http://www.cse.psu.edu/~rtc12/CSE586/papers/regr_cvpr14_facealignment.pdf)][[Codes|opencv(offical)](https://github.com/freesouls/face-alignment-at-3000fps)][[Codes|liblinear(unoffical)](https://github.com/jwyang/face-alignment)][[CSDN blog](https://blog.csdn.net/lzb863/article/details/49890369)]

* **3DDFA(CVPR2016)** Face Alignment Across Large Poses: A 3D Solution [[paper link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780392)][[project link](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)]

* **face-alignment(ICCV2017)** How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks) [[paper link](https://www.adrianbulat.com/downloads/FaceAlignment/FaceAlignment.pdf)][[Adrian Bulat](https://www.adrianbulat.com/)][[Codes|PyTorch(offical)](https://github.com/1adrianb/face-alignment)][[CSDN blogs](https://www.cnblogs.com/molakejin/p/8027573.html)]

* **PRNet(ECCV2018)** Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network [[arxiv link](https://arxiv.org/abs/1803.07835)][[Codes|TensorFlow(offical)](https://github.com/YadiraF/PRNet)]

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Face Detection

#### Materials

* [(github) A-Light-and-Fast-Face-Detector-for-Edge-Devices](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices)
* [(website) FDDB: Face Detection Data Set and Benchmark Home](http://vis-www.cs.umass.edu/fddb/)
* [(CSDN blogs) 人脸检测（十八）--TinyFace(S3FD,SSH,HR,RSA,Face R-CNN,PyramidBox)](https://blog.csdn.net/App_12062011/article/details/80534351)
* [(github) e2e-joint-face-detection-and-alignment](https://github.com/KaleidoZhouYN/e2e-joint-face-detection-and-alignment)
* [(github) libfacedetection in PyTorch](https://github.com/ShiqiYu/libfacedetection/)
* [(github) 1MB lightweight face detection model (1MB轻量级人脸检测模型)](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

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

#### Papers

* **VRN(ICCV2017)** Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression [[arxiv link](https://arxiv.org/abs/1703.07834)][[project link](http://aaronsplace.co.uk/papers/jackson2017recon/)][[online website](https://cvl-demos.cs.nott.ac.uk/vrn/)][[Codes|Torch7(offical)](https://github.com/AaronJackson/vrn)]

* **PRNet(ECCV2018)** Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network [[arxiv link](https://arxiv.org/abs/1803.07835)][[Codes|TensorFlow(offical)](https://github.com/YadiraF/PRNet)]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

### ▶ Head Detector

### Materials


### Papers

* **FCHD(arxiv2018,ICIP2019)** FCHD: Fast and accurate head detection in crowded scenes [[arxiv link](https://arxiv.org/abs/1809.08766)][[Codes|PyTorch(offical)](https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector)][[CSDN blog](https://blog.csdn.net/javastart/article/details/82865858)]

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

#### Papers

* **Survey(TPAMI2019)** Head Pose Estimation in Computer Vision: A Survey [[paper link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4497208)][[CSDN blog](https://blog.csdn.net/weixin_41703033/article/details/83215043)]

* **HopeNet(CVPRW2018)** Fine-Grained Head Pose Estimation Without Keypoints [[arxiv link](https://arxiv.org/abs/1710.00925)][[Codes|PyTorch(offical)](https://github.com/natanielruiz/deep-head-pose)][[CSDN blog](https://blog.csdn.net/qq_42189368/article/details/84849638)]

* **FSA-Net(CVPR2019)** FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image [[paper link](https://github.com/shamangary/FSA-Net/blob/master/0191.pdf)][[Codes|Keras&Tensorflow(offical)](https://github.com/shamangary/FSA-Net)]
