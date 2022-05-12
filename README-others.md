# Here are collections about other CV topics

**Contents Hyperlinks**

* [⭐3D Object Reconstruction](#3d-object-reconstruction)
* [⭐6D Object Pose Estimation](#6d-object-pose-estimation)
* [⭐Deep Neural Networks](#deep-neural-networks)
* [⭐Eye Gaze Estimation and Tracking](#eye-gaze-estimation-and-tracking)
* [⭐Generative Adversarial Network](#generative-adversarial-network)
* [⭐Image Mosaic](#image-mosaic)
* [⭐Image Restoration](#image-restoration)
* [⭐Lane Detection](#lane-detection)
* [⭐Pedestrian Localization](#pedestrian-localization)
* [⭐Person ReID](#person-reid)
* [⭐Scene Text Detection](#scene-text-detection)
* [⭐Semantic Segmentation](#semantic-segmentation)
* [⭐SLAM (Simultaneous Localization and Mapping)](#slam-simultaneous-localization-and-mapping)
* [⭐Sound Source Localization](#sound-source-localization)
* [⭐Traffic Violation Detection](#traffic-violation-detection)


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐3D Object Reconstruction

### Materials

### Papers

* **PSVH-3d-reconstruction(AAAI2019)** Deep Single-View 3D Object Reconstruction with Visual Hull Embedding [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/4922)][[codes|official TensorFlow](https://github.com/HanqingWangAI/PSVH-3d-reconstruction)]

* **(arxiv2022)** A Real World Dataset for Multi-view 3D Reconstruction [[paper link](https://arxiv.org/abs/2203.11397)]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐6D Object Pose Estimation

### Materials

* [(website) A Comprehensive List of 3D Sensors Commonly Leveraged in ROS Development](https://rosindustrial.org/3d-camera-survey)
* [(CSDN blogs) 【深度相机系列六】深度相机哪家强？附详细参数对比清单](https://blog.csdn.net/electech6/article/details/78907463)
* [(website) List of RGBD datasets](http://www.michaelfirman.co.uk/RGBDdatasets/) (written by a domain expert [Michael Firman](http://www.michaelfirman.co.uk/index.html))
* [(CSDN blogs) 6D姿态估计算法汇总（上）](https://blog.csdn.net/qq_29462849/article/details/103740960)
* [(CSDN blogs) 6D姿态估计算法汇总（下）](https://blog.csdn.net/qq_29462849/article/details/103741059)
* [(zhihu) VR设备常说的3DOF和6DOF到底是什么？](https://zhuanlan.zhihu.com/p/114650000)
* [(github) Awesome work on object 6 DoF pose estimation](https://github.com/ZhongqunZHANG/awesome-6d-object)
* [(github) MediaPipe: About Cross-platform, customizable ML solutions for live and streaming media](https://github.com/google/mediapipe)
* [(tools) ARCore: The AR session captures camera poses, point-clouds, and surface planes](http://developers.google.com/ar/)
* [(tools) ARKit: The AR session captures camera poses, point-clouds, and surface planes](https://developer.apple.com/augmented-reality/)
* [(website) paperswithcode: 6D Pose Estimation using RGB](https://paperswithcode.com/task/6d-pose-estimation)
* [(toolkit) 3D Annotation Of Arbitrary Objects In The Wild (inputs are RGB-D)](https://docs.strayrobots.io/toolkit/index.html) [[paper link](https://arxiv.org/abs/2109.07165)]
* [(algorithm) EPnP: Efficient Perspective-n-Point Camera Pose Estimation](https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/)
* [(blogs) 3D Object Detection和6D Pose Estimation有什么异同？](https://www.bilibili.com/read/cv5287260)

### Datasets

* [3DObject (ICCV2007)](http://vision.stanford.edu/resources_links.html): 3D generic object categorization, localization and pose estimation [***It provides discretized viewpoint annotations for 10 everyday object categories***]
* [⭐LineMOD (ACCV2012)](https://campar.in.tum.de/Main/StefanHinterstoisser): Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes [***The most commonly used dataset for object pose estimation***]
* [Pascal3D+ (WACV2014)](https://cvgl.stanford.edu/projects/pascal3d): Beyond PASCAL - A Benchmark for 3D Object Detection in the Wild [***It adds 3D pose annotations to the Pascal VOC and a few images from the ImageNet dataset***]
* [⭐ShapeNet (arxiv2015)](https://shapenet.org/): ShapeNet - An Information-Rich 3D Model Repository [***It includes synthetic CAD models for many objects and has been widely used***]
* [IC-BIN dataset (CVPR2016)](https://bop.felk.cvut.cz/leaderboards/bop19_ic-bin/): Recovering 6D Object Pose and Predicting Next-Best-View in the Crowd [***It adds a few more categories based on LineMOD***]
* [Rutgers APC (ICRA2016)](https://robotics.cs.rutgers.edu/pracsys/rutgers-apc-rgb-d-dataset/): A Dataset for Improved RGBD-based Object Detection and Pose Estimation for Warehouse Pick-and-Place [***It contains 14 textured objects from the Amazon picking challenge***]
* [ObjectNet3D (ECCV2016)](https://cvgl.stanford.edu/projects/objectnet3d/): ObjectNet3D - A Large Scale Database for 3D Object Recognition [***It contains 3D object poses from images***]
* [⭐T-LESS (WACV2017)](https://cmp.felk.cvut.cz/t-less/): T-LESS - An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects [***It features industrialized objects that lack texture or color***]
* [YCB (IJRR2017)](http://www.ycbbenchmarks.org/): Yale-CMU-Berkeley dataset for robotic manipulation research [***It contains videos of objects and their poses in a controlled environment***]
* [ScanNet(CVPR2017)](http://www.scan-net.org/): ScanNet - Richly-Annotated 3D Reconstructions of Indoor Scenes [***A large scale video dataset of indoor scenes with semantic annotations***]
* [⭐BOP Challenge (ECCV2018)](https://bop.felk.cvut.cz/home/): BOP - Benchmark for 6D Object Pose Estimation [***It consists of a set of benchmark for 3D object detection and combines many of these smaller datasets into a larger one***]
* [Pix3D (CVPR2018)](http://pix3d.csail.mit.edu/): Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling [***It contains pixel-level 2D-3D pose alignment***]
* [Scan2CAD (CVPR2019)](scan2cad.org): Scan2CAD - Learning CAD Model Alignment in RGB-D Scans [***It annotates the original scans in ScanNet with ShapeNetCore models to label each object’s pose***]
* [RIO (ICCV2019)](https://waldjohannau.github.io/RIO/): RIO - 3D Object Instance Re-Localization in Changing Indoor Environments [***Another dataset that contains indoor scans annotated with an object’s 3D pose***]
* [⭐Objectron (CVPR2021)](https://github.com/google-research-datasets/Objectron): A Large Scale Dataset of Object-Centric Videos in the Wild With Pose Annotations



### Papers

* **Pascal3D+(WACV2014)** Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild [[paper link](http://roozbehm.info/papers/Xiang14wacv.pdf)][[project link](https://cvgl.stanford.edu/projects/pascal3d)]

* **(ICCVW2017)** 3D Pose Regression Using Convolutional Neural Networks [[paper link](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w31/html/Mahendran_3D_Pose_Regression_ICCV_2017_paper.html)]

* **PoseCNN(RSS2018)** PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes [[paper link](https://yuxng.github.io/xiang_rss18.pdf)][[project link](https://rse-lab.cs.washington.edu/projects/posecnn/)]

* **(ECCV2018)** Occlusion Resistant Object Rotation Regression from Point Cloud Segments [[paper link](https://openaccess.thecvf.com/content_eccv_2018_workshops/w6/html/Gao_Occlusion_Resistant_Object_Rotation_Regression_from_Point_Cloud_Segments_ECCVW_2018_paper.html)]

* **DeepIM(ECCV2018)** DeepIM: Deep Iterative Matching for 6D Pose Estimation [[paper link](https://openaccess.thecvf.com/content_ECCV_2018/html/Yi_Li_DeepIM_Deep_Iterative_ECCV_2018_paper.html)][[project link](https://rse-lab.cs.washington.edu/projects/deepim/)]

* **⭐PVNet(CVPR2019)** PVNet: Pixel-Wise Voting Network for 6DoF Pose Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Peng_PVNet_Pixel-Wise_Voting_Network_for_6DoF_Pose_Estimation_CVPR_2019_paper.html)][[codes|official](https://zju3dv.github.io/pvnet/)]

* **DPOD(ICCV2019)** DPOD: 6D Pose Object Detector and Refiner [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Zakharov_DPOD_6D_Pose_Object_Detector_and_Refiner_ICCV_2019_paper.html)][[codes|PyTorch](https://github.com/zakharos/DPOD)]

* **HybridPose(CVPR2020)** HybridPose: 6D Object Pose Estimation Under Hybrid Representations [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Song_HybridPose_6D_Object_Pose_Estimation_Under_Hybrid_Representations_CVPR_2020_paper.html)]

* **single-stage-pose(CVPR2020)** Single-Stage 6D Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Hu_Single-Stage_6D_Object_Pose_Estimation_CVPR_2020_paper.html)][[codes|official PyTorch](https://github.com/cvlab-epfl/single-stage-pose)]

* **CosyPose(ECCV2020)** CosyPose: Consistent Multi-view Multi-object 6D Pose Estimation [[paper link](https://hal.inria.fr/hal-02950800/)][[project link](https://www.di.ens.fr/willow/research/cosypose)][[codes|official PyTorch](https://github.com/ylabbe/cosypose)]

* **MobilePose(arxiv2020)** MobilePose: Real-Time Pose Estimation for Unseen Objects with Weak Shape Supervision [[paper link](https://arxiv.org/abs/2003.03522)]

* **SGPA(ICCV2021)** SGPA: Structure-Guided Prior Adaptation for Category-Level 6D Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_SGPA_Structure-Guided_Prior_Adaptation_for_Category-Level_6D_Object_Pose_Estimation_ICCV_2021_paper.html)][[codes|PyTorch](https://github.com/leo94-hk/SGPA)]

* **⭐Objectron(CVPR2021)(Training Codes ╳)(Annotation Tool ╳)]** Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild With Pose Annotations [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Ahmadyan_Objectron_A_Large_Scale_Dataset_of_Object-Centric_Videos_in_the_CVPR_2021_paper.html)][[codes|PyTorch+TensorFlow](https://github.com/google-research-datasets/Objectron)][[official blog 1: MediaPipe](https://mediapipe.dev/)][[official blog 2: MediaPipe Objectron](https://google.github.io/mediapipe/solutions/objectron)]

* **CenterPose(arxiv2021)(Training with CenterNet and Objectron)** Single-stage Keypoint-based Category-level Object Pose Estimation from an RGB Image [[paper link](https://arxiv.org/abs/2109.06161)][[codes|official PyTorch ](https://github.com/NVlabs/CenterPose)]

* **SAR-Net(CVPR2022)** SAR-Net: Shape Alignment and Recovery Network for Category-level 6D Object Pose and Size Estimation [[paper link]()][[project link](https://hetolin.github.io/SAR-Net/)][[codes|official](https://github.com/hetolin/SAR-Net)]

* **OVE6D-pose(CVPR2022)** OVE6D: Object Viewpoint Encoding for Depth-based 6D Object Pose Estimation [[paper link](https://arxiv.org/pdf/2203.01072.pdf)][[project link](https://dingdingcai.github.io/ove6d-pose/)][[codes|official](https://github.com/dingdingcai/OVE6D-pose)]

* **Gen6D(arxiv2022)** Gen6D: Generalizable Model-Free 6-DoF Object Pose Estimation from RGB Images [[paper link](https://arxiv.org/abs/2204.10776)][[project link](https://liuyuan-pal.github.io/Gen6D/)][[codes|on the way]()]




**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Eye Gaze Estimation and Tracking

### Materials


### Papers

* **ETH-XGaze(ECCV2020)** ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation [[arxiv link](https://arxiv.org/abs/2007.15837)][[project link](https://ait.ethz.ch/projects/2020/ETH-XGaze/)][[Codes|PyTorch(official)](https://github.com/xucong-zhang/ETH-XGaze)]

* **EVE(ECCV2020)** Towards End-to-end Video-based Eye-tracking [[arxiv link](https://arxiv.org/abs/2007.13120)][[project link](https://ait.ethz.ch/projects/2020/EVE/)][[Codes|PyTorch(official)](https://github.com/swook/EVE)]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Generative Adversarial Network

### Materials

* [(blog) Test and Train CycleGAN](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb#scrollTo=OzSKIPUByfiN)
* [(CSDNblog) CycleGAN论文的阅读与翻译，无监督风格迁移](https://zhuanlan.zhihu.com/p/45394148)
* [(CSDNblog) 生成对抗网络(四)CycleGAN讲解](https://blog.csdn.net/qq_40520596/article/details/104714762)

### Papers

* **CycleGAN(ICCV2017)** Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [[arxiv link](https://arxiv.org/pdf/1703.10593.pdf)][[project link](https://junyanz.github.io/CycleGAN/)][[Codes|PyTorch(official)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]

* **CUT(ECCV2020)** Contrastive Learning for Unpaired Image-to-Image Translation [[arxiv link](https://arxiv.org/abs/2007.15651)][[project link](http://taesung.me/ContrastiveUnpairedTranslation/)][[Codes|PyTorch(official)](https://github.com/taesungp/contrastive-unpaired-translation)]



**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Deep Neural Networks

### Frameworks

* **PyTorch** [Home Page](https://pytorch.org/), [Offical Documentation](https://pytorch.org/docs/stable/index.html)
* **TensorFlow** [Home Page](https://tensorflow.google.cn/), [Offical Documentation](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf)

### Materials

* [Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://arxiv.org/pdf/1703.09039.pdf)
* [(CSDN blog) 五大经典卷积神经网络介绍：LeNet / AlexNet / GoogLeNet / VGGNet/ ResNet](https://blog.csdn.net/fendouaini/article/details/79807830)
* [(cnblogs) Deep Learning回顾#之LeNet、AlexNet、GoogLeNet、VGG、ResNet](https://www.cnblogs.com/52machinelearning/p/5821591.html)
* [(github) HRNet: HRNet-Applications-Collection](https://github.com/HRNet/HRNet-Applications-Collection)

### Papers



**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Image Mosaic


### Materials

* [(zhihu) 基于图像的三维建模——特征点检测与匹配](https://zhuanlan.zhihu.com/p/128937547)
* [(website) 图像拼接算法的综述 - A survey on image mosaicing techniques](http://s1nh.org/post/A-survey-on-image-mosaicing-techniques/)
* [(cnblogs) OpenCV探索之路（二十四）图像拼接和图像融合技术](https://www.cnblogs.com/skyfsm/p/7411961.html)
* [(zhihu - YaqiLYU) 图像拼接现在还有研究的价值吗？有哪些可以研究的点？现在技术发展如何？](https://www.zhihu.com/question/34535199/answer/135169187)
* [(zhihu - YaqiLYU) 目前最成熟的全景视频拼接技术是怎样的？](https://www.zhihu.com/question/34573969/answer/136464893)
* [(opencv docs) Feature Detection and Description](https://docs.opencv.org/master/db/d27/tutorial_py_table_of_contents_feature2d.html)
* [(github) [Real-Time Image Stitching] CS205 Computing Foundations for Computational Science Final Project(C++)](https://github.com/ziqiguo/CS205-ImageStitching)
* [(github) [Image and Video Stitching] Conducts image stitching upon an input video to generate a panorama in 3D(Python)](https://github.com/WillBrennan/ImageStitching)
* [(github) Multiple Image stitching in Python](https://github.com/kushalvyas/Python-Multiple-Image-Stitching)


### Papers

* **NISwGSP(ECCV2016)** Natural Image Stitching with the Global Similarity Prior [[paper link](https://link.springer.com/chapter/10.1007%2F978-3-319-46454-1_12)][[Codes|offical C++ & Matlab](https://github.com/nothinglo/NISwGSP)]

* **VFSMS(CMS2019)** A Fast Algorithm for Material Image Sequential Stitching [[paper link](http://www.sciencedirect.com/science/article/pii/S0927025618307158)][[software](https://www.mgedata.cn/app_entrance/microscope)][[Codes|offical python & C++](https://github.com/Keep-Passion/ImageStitch)]

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Image Restoration

Image restoration includes **image in-painting**, **pixel interpolation**, **image deblurring**, and **image denoising**.
 
### Materials

* [(github) CNN-For-End-to-End-Deblurring (Keras)](https://github.com/axium/CNN-For-End-to-End-Deblurring--Keras)

### Papers

* **DnCNN(TIP2017)** Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising [[paper link](http://www4.comp.polyu.edu.hk/~cslzhang/paper/DnCNN.pdf)][[Codes|MATLAB(offical)](https://github.com/cszn/DnCNN)]

* **MemNet(ICCV2017)** MemNet: A Persistent Memory Network for Image Restoration [[paper link](http://cvlab.cse.msu.edu/pdfs/Image_Restoration%20using_Persistent_Memory_Network.pdf)][[Codes|Matlab(offical)](https://github.com/tyshiwo/MemNet)]

* **pix2pix(CVPR2017)** Image-to-Image Translation with Conditional Adversarial Nets [[arxiv link](https://arxiv.org/abs/1611.07004)][[project link](https://phillipi.github.io/pix2pix/)][[Codes|Torch(offical)](https://github.com/phillipi/pix2pix)][[Codes|PyTorch(offical)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]

* **DeepDeblur(CVPR2017)** [[paper link](https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)][[Codes|Torch(offical)](https://github.com/SeungjunNah/DeepDeblur_release)][[Codes|PyTorch(offical)](https://github.com/SeungjunNah/DeepDeblur-PyTorch)]

* **ImageDeblurring(ICCV2017)** Deep Generative Filter for motion deblurring [[arxiv link](https://arxiv.org/abs/1709.03481)][[Codes|Keras&Tensorflow(offical)](https://github.com/leftthomas/ImageDeblurring)]

* **DeblurGAN(CVPR2017)** DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks [[arxiv link](https://arxiv.org/pdf/1711.07064.pdf)][[Codes|PyTorch(offical)](https://github.com/KupynOrest/DeblurGAN)]

* **SRN-Deblur(CVPR2018)** Scale-recurrent Network for Deep Image Deblurring [[paper link](http://www.xtao.website/projects/srndeblur/srndeblur_cvpr18.pdf)][[Codes|Tensorflow(offical)](https://github.com/jiangsutx/SRN-Deblur)]

* **RNN-Deblur(CVPR2018)** Dynamic Scene Deblurring Using Spatially Variant Recurrent Neural Networks [[paper link](https://www.cs.cityu.edu.hk/~rynson/papers/cvpr18c.pdf)][[Codes|Matcaffe(offical)](https://github.com/zhjwustc/cvpr18_rnn_deblur_matcaffe)]

* **Deep-Semantic-Face(CVPR2018)** Deep Semantic Face Deblurring [[paper link](https://research.nvidia.com/sites/default/files/pubs/2018-06_Deep-Semantic-Face//DeepSemanticFaceDeblur_CVPR18.pdf)][[project link](https://research.nvidia.com/publication/2018-06_Deep-Semantic-Face)][[Codes|Matlab(offical)](https://github.com/joanshen0508/Deep-Semantic-Face-Deblurring)]

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Lane Detection

### Materials

### Datasets

* [**CULane**: SCNN(AAAI2018) Spatial As Deep: Spatial CNN for Traffic Scene Understanding](https://xingangpan.github.io/projects/CULane.html)

### Papers
 
* **SCNN(AAAI2018)** Spatial As Deep: Spatial CNN for Traffic Scene Understanding [[arxiv link](https://arxiv.org/abs/1712.06080)][[Codes|offical Torch & Matlab](https://github.com/XingangPan/SCNN)]

* **LaneNet(IVS2018)** Towards End-to-End Lane Detection: an Instance Segmentation Approach [[arxiv link](https://arxiv.org/abs/1802.05591)][[project link](https://maybeshewill-cv.github.io/lanenet-lane-detection/)][[Codes|unoffical TF](https://github.com/MaybeShewill-CV/lanenet-lane-detection)]

* **UltraLane(ECCV2020)** Ultra Fast Structure-aware Deep Lane Detection [[arxiv link](https://arxiv.org/abs/2004.11757)][[Codes|offical PyTorch](https://github.com/cfzd/Ultra-Fast-Lane-Detection)]

* **BézierLaneNet(CVPR2022)** Rethinking Efficient Lane Detection via Curve Modeling [[paper link](https://arxiv.org/abs/2203.02431)][[codes|official PyTorch](https://github.com/voldemortX/pytorch-auto-drive)]



**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Pedestrian Localization

### Materials


### Papers

* **Monoloco(ICCV2019)** MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation [[arxiv link](https://arxiv.org/abs/1906.06059)][[Codes|PyTorch(offical)](https://github.com/vita-epfl/monoloco)]

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Person ReID

### Materials

* [(zhihu) 从零开始行人重识别](https://zhuanlan.zhihu.com/p/50387521)
* [(zhihu) (转)行人重识别(ReID) ——技术实现及应用场景](https://zhuanlan.zhihu.com/p/64362368)
* [(zhihu) 一些想法：关于行人检测与重识别](https://zhuanlan.zhihu.com/p/39282286)
* [(zhihu) 零基础实战行人重识别ReID项目-基于Milvus的以图搜图](https://zhuanlan.zhihu.com/p/141204192)
* [(csdnblog) 行人重识别（Person Re-ID）【一】：常用评测指标](https://blog.csdn.net/qq_38451119/article/details/83000061)
* [(csdnblog) 云从科技：详解跨镜追踪（ReID）技术实现及应用场景](https://edu.csdn.net/course/detail/8426)
* [(tencent cloud) 云从科技资深算法研究员：详解跨镜追踪(ReID)技术实现及难点 | 公开课笔记](https://cloud.tencent.com/developer/article/1160607)

### Datasets

* [Market1501 [Tsinghua University; 32217 images; 1501 persons; 6 cameras]](http://liangzheng.com.cn/Project/project_reid.html)
* [DukeMTMC-ReID [Duke University; 36441 images; 1812 persons; 8 cameras]](https://github.com/sxzrt/DukeMTMC-reID_evaluation#download-dataset)
* [CUHK03 [CUHK University; 13164 images; 1467 persons; 10 cameras]](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)

### Papers

* **(TOMM2017)** A Discriminatively Learned CNN Embedding for Person Re-identification [[arxiv link](https://arxiv.org/pdf/1611.05666.pdf)][[Codes|caffe+keras(official)](https://github.com/layumi/2016_person_re-ID)][[CSDN blog](https://blog.csdn.net/weixin_41427758/article/details/80091596)]



**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Scene Text Detection

### Materials

* [Commonly used four datasets in scene text detection: ICDAR2015, CTW1500, Total-Text and MSRA-TD500]
* [(github) 🚀PaddleOCR: Awesome multilingual OCR toolkits based on PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR)

### Papers

#### Regression-based

* **R2CNN(arxiv2017)** R2CNN: Rotational Region CNN for Orientation Robust Scene Text Detection [[paper link](https://arxiv.org/abs/1706.09579)]

* **TextBoxes(AAAI2017)** TextBoxes: A Fast Text Detector with a Single Deep Neural Network [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/11196)]

* **EAST(CVPR2017)** EAST: An Efficient and Accurate Scene Text Detector [[paper link](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_EAST_An_Efficient_CVPR_2017_paper.html)]

* **ContourNet(CVPR2020)** ContourNet: Taking a Further Step Toward Accurate Arbitrary-Shaped Scene Text Detection [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ContourNet_Taking_a_Further_Step_Toward_Accurate_Arbitrary-Shaped_Scene_Text_CVPR_2020_paper.html)][[codes|official](https://github.com/wangyuxin87/ContourNet)]

* **ABCNet(CVPR2020)** ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_ABCNet_Real-Time_Scene_Text_Spotting_With_Adaptive_Bezier-Curve_Network_CVPR_2020_paper.pdf)][[codes|Detectron2 & AdelaiDet Toolbox](https://github.com/aim-uofa/AdelaiDet)]

* **ABCNet_v2(TPAMI2021)** ABCNet v2: Adaptive Bezier-Curve Network for Real-time End-to-end Text Spotting [[paper link](https://ieeexplore.ieee.org/abstract/document/9525302)][[codes|Detectron2 & AdelaiDet Toolbox](https://github.com/aim-uofa/AdelaiDet)]

#### Segmentation-based

* **TextSnake(ECCV2018)** TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes [[paper link](https://openaccess.thecvf.com/content_ECCV_2018/html/Shangbang_Long_TextSnake_A_Flexible_ECCV_2018_paper.html)]

* **TextDragon(ICCV2019)** TextDragon: An End-to-End Framework for Arbitrary Shaped Text Spotting [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Feng_TextDragon_An_End-to-End_Framework_for_Arbitrary_Shaped_Text_Spotting_ICCV_2019_paper.html)]

* **PANet(ICCV2019)** Efficient and Accurate Arbitrary-Shaped Text Detection With Pixel Aggregation Network [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Efficient_and_Accurate_Arbitrary-Shaped_Text_Detection_With_Pixel_Aggregation_Network_ICCV_2019_paper.html)]

* **PSENet(CVPR2019)** Shape Robust Text Detection With Progressive Scale Expansion Network [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html)]

* **DBNet(AAAI2020)** Real-Time Scene Text Detection with Differentiable Binarization [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/6812)][[codes|official](https://github.com/MhLiao/DB)]

* **FCENet(CVPR2021)** Fourier Contour Embedding for Arbitrary-Shaped Text Detection [paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Fourier_Contour_Embedding_for_Arbitrary-Shaped_Text_Detection_CVPR_2021_paper.html)]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**


## ⭐Semantic Segmentation

### Materials

* [(CSDN blogs) 语义分割 - Semantic Segmentation Papers](https://blog.csdn.net/langb2014/article/details/82414918)

### Papers

* **FCIS(CVPR2017)** Fully Convolutional Instance-aware Semantic Segmentation [[arxiv link](https://arxiv.org/abs/1611.07709)][[Codes|MXNet(offical based on RFCN)](https://github.com/msracver/FCIS)][[CSDN blog](https://blog.csdn.net/jiongnima/article/details/78961147)]

* **BezierSeg(arxiv2021)** BezierSeg: Parametric Shape Representation for Fast Object Segmentation in Medical Images [[paper link](https://arxiv.org/abs/2108.00760)]

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**


## ⭐SLAM (Simultaneous Localization and Mapping)

### Materials

* [(cnblogs) 视觉SLAM漫谈](https://blog.csdn.net/weixin_41537599/article/details/110819969)

### Papers

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Sound Source Localization

### Materials

* [(cnblogs) 【论文导读】Learning to Localize Sound Source in Visual Scenes】&soundnet的复现](https://blog.csdn.net/zzc15806/article/details/80772152)
* [(cnblogs) 论文【Learning to Localize Sound Source in Visual Scenes】&soundnet的复现](https://www.cnblogs.com/gaoxiang12/p/3695962.html)

### Papers

* **SoundNet(NIPS2016)** SoundNet: Learning Sound Representations from Unlabeled Video [[arxiv link](https://arxiv.org/pdf/1610.09001.pdf)][[Codes|offical TensorFlow](https://github.com/cvondrick/soundnet)][[CSDN blog](https://blog.csdn.net/zzc15806/article/details/80669883)]

* **SoundLocation(CVPR2018)** Learning to Localize Sound Source in Visual Scenes [[arxiv link](https://arxiv.org/pdf/1803.03849.pdf)][[Codes|offical PyTorch based on SoundNet](https://github.com/ardasnck/learning_to_localize_sound_source)][[Codes|unoffical PyTorch](https://github.com/liyidi/soundnet_localize_sound_source)]

* **avobjects(ECCV2020)** Self-Supervised Learning of Audio-Visual Objects from Video [[paper link](https://arxiv.org/abs/2008.04237)][[project link](https://www.robots.ox.ac.uk/~vgg/research/avobjects/)][[Oxford VGG](https://www.robots.ox.ac.uk/~vgg/research/)][[codes|official PyTorch](https://github.com/afourast/avobjects)]

* **VGG-Sound(CVPR2021)** Localizing Visual Sounds the Hard Way [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Localizing_Visual_Sounds_the_Hard_Way_CVPR_2021_paper.html)][[codes|official PyTorch ](https://github.com/hche11/Localizing-Visual-Sounds-the-Hard-Way)]

* **EZ-VSL(arxiv2022)** Localizing Visual Sounds the Easy Way [[paper link](https://arxiv.org/abs/2203.09324)][[codes|official](https://github.com/stoneMo/EZ-VSL)]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Traffic Violation Detection

### Materials

* [(github) Traffic-Rule-Violation-Detection-System (Tensorflow + OpenALPR )](https://github.com/ShreyAmbesh/Traffic-Rule-Violation-Detection-System)
* [(github) Traffic-Signal-Violation-Detection-System (Tensorflow based YOLOv3)](https://github.com/anmspro/Traffic-Signal-Violation-Detection-System)
* [(github) Traffic-Rules-Violation-Detection (mobilenet-v1)](https://github.com/rahatzamancse/Traffic-Rules-Violation-Detection)
* [(github) Traffic-Rules-Violation-Detection-System (mobilenet-v1)](https://github.com/sakibreza/Traffic-Rules-Violation-Detection-System)
* [(github) Fully-Automated-red-light-Violation-Detection (Tensorflow based YOLOv3)](https://github.com/AhmadYahya97/Fully-Automated-red-light-Violation-Detection)
* [(github) yolov3-vehicle-detection-paddle](https://github.com/Sharpiless/yolov3-vehicle-detection-paddle) [[CSDN link](https://blog.csdn.net/weixin_45449540/article/details/107345738)]

### Papers



