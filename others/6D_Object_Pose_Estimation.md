# ⭐6D Object Pose Estimation 
Also named ***3D Object Detection***

## Materials

* [(zhihu) 话题：6DOF姿态估计](https://www.zhihu.com/collection/274088096)
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

## Datasets

* [3DObject (ICCV2007)](http://vision.stanford.edu/resources_links.html): 3D generic object categorization, localization and pose estimation [***It provides discretized viewpoint annotations for 10 everyday object categories***]
* [❤ LineMOD (ACCV2012)](https://campar.in.tum.de/Main/StefanHinterstoisser): Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes [***The most commonly used dataset for object pose estimation***]
* [Pascal3D+ (WACV2014)](https://cvgl.stanford.edu/projects/pascal3d): Beyond PASCAL - A Benchmark for 3D Object Detection in the Wild [***It adds 3D pose annotations to the Pascal VOC and a few images from the ImageNet dataset***]
* [❤ ShapeNet (arxiv2015)](https://shapenet.org/): ShapeNet - An Information-Rich 3D Model Repository [***It includes synthetic CAD models for many objects and has been widely used***]
* [IC-BIN dataset (CVPR2016)](https://bop.felk.cvut.cz/leaderboards/bop19_ic-bin/): Recovering 6D Object Pose and Predicting Next-Best-View in the Crowd [***It adds a few more categories based on LineMOD***]
* [Rutgers APC (ICRA2016)](https://robotics.cs.rutgers.edu/pracsys/rutgers-apc-rgb-d-dataset/): A Dataset for Improved RGBD-based Object Detection and Pose Estimation for Warehouse Pick-and-Place [***It contains 14 textured objects from the Amazon picking challenge***]
* [ObjectNet3D (ECCV2016)](https://cvgl.stanford.edu/projects/objectnet3d/): ObjectNet3D - A Large Scale Database for 3D Object Recognition [***It contains 3D object poses from images***]
* [❤ T-LESS (WACV2017)](https://cmp.felk.cvut.cz/t-less/): T-LESS - An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects [***It features industrialized objects that lack texture or color***]
* [YCB (IJRR2017)](http://www.ycbbenchmarks.org/): Yale-CMU-Berkeley dataset for robotic manipulation research [***It contains videos of objects and their poses in a controlled environment***]
* [ScanNet(CVPR2017)](http://www.scan-net.org/): ScanNet - Richly-Annotated 3D Reconstructions of Indoor Scenes [***A large scale video dataset of indoor scenes with semantic annotations***]
* [❤ BOP Challenge (ECCV2018)](https://bop.felk.cvut.cz/home/): BOP - Benchmark for 6D Object Pose Estimation [***It consists of a set of benchmark for 3D object detection and combines many of these smaller datasets into a larger one***]
* [Pix3D (CVPR2018)](http://pix3d.csail.mit.edu/): Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling [***It contains pixel-level 2D-3D pose alignment***]
* [Scan2CAD (CVPR2019)](scan2cad.org): Scan2CAD - Learning CAD Model Alignment in RGB-D Scans [***It annotates the original scans in ScanNet with ShapeNetCore models to label each object’s pose***]
* [RIO (ICCV2019)](https://waldjohannau.github.io/RIO/): RIO - 3D Object Instance Re-Localization in Changing Indoor Environments [***Another dataset that contains indoor scans annotated with an object’s 3D pose***]
* [NOCS (CVPR2019 Oral)](https://geometry.stanford.edu/projects/NOCS_CVPR2019/): Normalized Object Coordinate Space (NOCS) - a shared canonical representation for all possible object instances within a category [***It is a fully annotated real-world RGB-D dataset with large environment and instance variation***]
* [👍 ❤ Objectron (CVPR2021)(by google)](https://github.com/google-research-datasets/Objectron): A Large Scale Dataset of Object-Centric Videos in the Wild With Pose Annotations
* [👍 ❤ CO3D (ICCV2021)(by facebook)](https://github.com/facebookresearch/co3d): Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction [***It has diverse instances in different categories and is collected without recording the depth***]
* [❤ PhoCaL (CVPR2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_PhoCaL_A_Multi-Modal_Dataset_for_Category-Level_Object_Pose_Estimation_With_CVPR_2022_paper.html): PhoCaL: A Multi-Modal Dataset for Category-Level Object Pose Estimation With Photometrically Challenging Objects [***A novel robot-supported multi-modal (RGB, depth, polarisation) benchmark with challenging scenes supporting RGB-D and monocular RGB methods***]
* [❤ ABO (Amazon Berkeley Objects) (CVPR2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Collins_ABO_Dataset_and_Benchmarks_for_Real-World_3D_Object_Understanding_CVPR_2022_paper.html) ABO: Dataset and Benchmarks for Real-World 3D Object Understanding [[project link](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)][[github link](https://github.com/jazcollins/amazon-berkeley-objects)][[paperswithcode homepage](https://paperswithcode.com/dataset/abo)][***A large-scale dataset designed for material prediction and multi-view retrieval experiments. The dataset contains Blender renderings of 30 viewpoints for each of the 7,953 3D objects, as well as camera intrinsics and extrinsic for each rendering.***]
* [Objaverse (arxiv2022)](https://arxiv.org/abs/2212.08051): Objaverse: A Universe of Annotated 3D Objects [[project link](https://objaverse.allenai.org/)][[paperswithcode homepage](https://paperswithcode.com/dataset/objaverse)][***A large dataset of objects with 800K+ (and growing) 3D models with descriptive captions, tags, and animations.***]
* [👍 ❤ Wild6D (NIPS2022)](https://oasisyang.github.io/semi-pose/): Category-Level 6D Object Pose Estimation in the Wild: A Semi-Supervised Learning Approach and A New Dataset [***This dataset consists of a large number of object-centric RGBD videos***]
* [❤ Omni3D (CVPR2023)(by facebook)](https://github.com/facebookresearch/omni3d): Omni3D: A Large Benchmark and Model for 3D Object Detection in the Wild [***Omni3D re-purposes and combines existing datasets resulting in 234k images annotated with more than 3 million instances and 98 categories. 3D detection at such scale is challenging due to variations in camera intrinsics and the rich diversity of scene and object types.***]



## Papers

### ⭐3D Object Detection

* ❤ **3D-BoundingBox(CVPR2017)** 3D Bounding Box Estimation Using Deep Learning and Geometry [[paper link](https://arxiv.org/abs/1612.00496)][[codes|official PyTorch](https://github.com/skhadem/3D-BoundingBox)]

* ❤ **SMOKE(CVPRW2020)** SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation [[paper link](https://openaccess.thecvf.com/content_CVPRW_2020/html/w60/Liu_SMOKE_Single-Stage_Monocular_3D_Object_Detection_via_Keypoint_Estimation_CVPRW_2020_paper.html)][[codes|official PyTorch](https://github.com/lzccccc/SMOKE)]

* **MonoPair(CVPR2020)** MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_MonoPair_Monocular_3D_Object_Detection_Using_Pairwise_Spatial_Relationships_CVPR_2020_paper.html)][[codes is not available]()]

* **RTM3D(ECCV2020)** RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving [[paper link](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480647.pdf)][[codes|]()]

* ❤ **FADNet(TIV2022)** Monocular 3D Object Detection with Sequential Feature Association and Depth Hint Augmentation [[paper link](https://arxiv.org/abs/2011.14589)][[codes|official](https://github.com/gtzly/FADNet)]

### ⭐6D Object Pose Estimation

* **Pascal3D+(WACV2014)** Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild [[paper link](http://roozbehm.info/papers/Xiang14wacv.pdf)][[project link](https://cvgl.stanford.edu/projects/pascal3d)]

* **(ICCVW2017)** 3D Pose Regression Using Convolutional Neural Networks [[paper link](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w31/html/Mahendran_3D_Pose_Regression_ICCV_2017_paper.html)]

* **PoseCNN(RSS2018)** PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes [[paper link](https://yuxng.github.io/xiang_rss18.pdf)][[project link](https://rse-lab.cs.washington.edu/projects/posecnn/)]

* **(ECCV2018)** Occlusion Resistant Object Rotation Regression from Point Cloud Segments [[paper link](https://openaccess.thecvf.com/content_eccv_2018_workshops/w6/html/Gao_Occlusion_Resistant_Object_Rotation_Regression_from_Point_Cloud_Segments_ECCVW_2018_paper.html)]

* **DeepIM(ECCV2018)** DeepIM: Deep Iterative Matching for 6D Pose Estimation [[paper link](https://openaccess.thecvf.com/content_ECCV_2018/html/Yi_Li_DeepIM_Deep_Iterative_ECCV_2018_paper.html)][[project link](https://rse-lab.cs.washington.edu/projects/deepim/)]

* ❤**YOLO-6D(CVPR2018)** Real-Time Seamless Single Shot 6D Object Pose Prediction [[paper link](https://openaccess.thecvf.com/content_cvpr_2018/html/Tekin_Real-Time_Seamless_Single_CVPR_2018_paper.html)][[codes|official PyTorch](https://github.com/microsoft/singleshotpose)][[codes|unofficial TensorFlow](https://github.com/Mmmofan/YOLO_6D)][`YOLOv2`]

* ❤**YOLO-Seg(CVPR2019)** Segmentation-Driven 6D Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Hu_Segmentation-Driven_6D_Object_Pose_Estimation_CVPR_2019_paper.html)][[codes|official](https://github.com/cvlab-epfl/segmentation-driven-pose)][`YOLOv2`]

* ❤**PVNet(CVPR2019 Oral)** PVNet: Pixel-Wise Voting Network for 6DoF Pose Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Peng_PVNet_Pixel-Wise_Voting_Network_for_6DoF_Pose_Estimation_CVPR_2019_paper.html)][[codes|official](https://zju3dv.github.io/pvnet/)]

* 👍👍**NOCS(CVPR2019 Oral)** Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.html)][[project link](https://geometry.stanford.edu/projects/NOCS_CVPR2019/)][[codes & datasets|official keras and tensorflow](https://github.com/hughw19/NOCS_CVPR2019)][[`He Wang (王鹤)`](https://hughw19.github.io/)]

* **DPOD(ICCV2019)** DPOD: 6D Pose Object Detector and Refiner [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Zakharov_DPOD_6D_Pose_Object_Detector_and_Refiner_ICCV_2019_paper.html)][[codes|PyTorch](https://github.com/zakharos/DPOD)]

* **CDPN(ICCV2019)** CDPN: Coordinates-Based Disentangled Pose Network for Real-Time RGB-Based 6-DoF Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_CDPN_Coordinates-Based_Disentangled_Pose_Network_for_Real-Time_RGB-Based_6-DoF_Object_ICCV_2019_paper.html)][[codes|official PyTorch](https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi)][`YOLOv3`]

* **HybridPose(CVPR2020)** HybridPose: 6D Object Pose Estimation Under Hybrid Representations [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Song_HybridPose_6D_Object_Pose_Estimation_Under_Hybrid_Representations_CVPR_2020_paper.html)]

* **single-stage-pose(CVPR2020)** Single-Stage 6D Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Hu_Single-Stage_6D_Object_Pose_Estimation_CVPR_2020_paper.html)][[codes|official PyTorch](https://github.com/cvlab-epfl/single-stage-pose)]

* **CosyPose(ECCV2020)** CosyPose: Consistent Multi-view Multi-object 6D Pose Estimation [[paper link](https://hal.inria.fr/hal-02950800/)][[project link](https://www.di.ens.fr/willow/research/cosypose)][[codes|official PyTorch](https://github.com/ylabbe/cosypose)]

* 👍**SPD(ECCV2020)** Shape Prior Deformation for Categorical 6D Object Pose and Size Estimation [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58589-1_32)][[arxiv link](https://arxiv.org/abs/2007.08454)][[code|official](https://github.com/mentian/object-deformnet)]

* **MobilePose(arxiv2020)** MobilePose: Real-Time Pose Estimation for Unseen Objects with Weak Shape Supervision [[paper link](https://arxiv.org/abs/2003.03522)]

* **SGPA(ICCV2021)** SGPA: Structure-Guided Prior Adaptation for Category-Level 6D Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_SGPA_Structure-Guided_Prior_Adaptation_for_Category-Level_6D_Object_Pose_Estimation_ICCV_2021_paper.html)][[codes|PyTorch](https://github.com/leo94-hk/SGPA)]

* 👍**DualPoseNet(ICCV2021)** DualPoseNet: Category-level 6D Object Pose and Size Estimation Using Dual Pose Network with Refined Learning of Pose Consistency [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Lin_DualPoseNet_Category-Level_6D_Object_Pose_and_Size_Estimation_Using_Dual_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2103.06526)][[code|official](https://github.com/Gorilla-Lab-SCUT/DualPoseNet)]

* ❤**Objectron(CVPR2021)(Training Codes ╳)(Annotation Tool ╳)]** Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild With Pose Annotations [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Ahmadyan_Objectron_A_Large_Scale_Dataset_of_Object-Centric_Videos_in_the_CVPR_2021_paper.html)][[codes|PyTorch+TensorFlow](https://github.com/google-research-datasets/Objectron)][[official blog 1: MediaPipe](https://mediapipe.dev/)][[official blog 2: MediaPipe Objectron](https://google.github.io/mediapipe/solutions/objectron)]

* **SAR-Net(CVPR2022)** SAR-Net: Shape Alignment and Recovery Network for Category-level 6D Object Pose and Size Estimation [[paper link](https://arxiv.org/abs/2106.14193)][[project link](https://hetolin.github.io/SAR-Net/)][[codes|official](https://github.com/hetolin/SAR-Net)]

* **OVE6D-pose(CVPR2022)** OVE6D: Object Viewpoint Encoding for Depth-based 6D Object Pose Estimation [[paper link](https://arxiv.org/pdf/2203.01072.pdf)][[project link](https://dingdingcai.github.io/ove6d-pose/)][[codes|official](https://github.com/dingdingcai/OVE6D-pose)]

* **OnePose(CVPR2022)** OnePose: One-Shot Object Pose Estimation without CAD Models [[paper link](https://arxiv.org/pdf/2205.12257.pdf)][[project link](https://zju3dv.github.io/onepose/)][[codes|official](https://github.com/zju3dv/OnePose)][`ZJU + Objectron`]

* **Gen6D(ECCV2022)** Gen6D: Generalizable Model-Free 6-DoF Object Pose Estimation from RGB Images [[paper link](https://arxiv.org/abs/2204.10776)][[project link](https://liuyuan-pal.github.io/Gen6D/)][[codes|on the way]()]

* ❤**CenterSnap(ICRA2022)** CenterSnap: Single-Shot Multi-Object 3D Shape Reconstruction and Categorical 6D Pose and Size Estimation [[paper link](https://arxiv.org/abs/2203.01929)][[project link](https://zubair-irshad.github.io/projects/CenterSnap.html)][[codes|official PyTorch](https://github.com/zubair-irshad/CenterSnap)]

* ❤**CenterPose(ICRA2022)(Training with CenterNet and Objectron)** Single-stage Keypoint-based Category-level Object Pose Estimation from an RGB Image [[paper link](https://arxiv.org/abs/2109.06161)][[project link](https://sites.google.com/view/centerpose)][[author homepage](https://yunzhi.netlify.app/)][[codes|official PyTorch](https://github.com/NVlabs/CenterPose)][`Nvidia + Objectron + one-stage + end2end`][based on `FADNet` (https://github.com/gtzly/FADNet) and `CenterNet` (https://github.com/xingyizhou/CenterNet)]

* 👍**EPro-PnP(CVPR2022 Oral, Best Student Paper)** EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_EPro-PnP_Generalized_End-to-End_Probabilistic_Perspective-N-Points_for_Monocular_Object_Pose_Estimation_CVPR_2022_paper.pdf)][[code|official](https://github.com/tjiiv-cprg/EPro-PnP)]

* 👍**VI-Net(ICCV2023)** VI-Net: Boosting Category-level 6D Object Pose Estimation via Learning Decoupled Rotations on the Spherical Representations [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_VI-Net_Boosting_Category-level_6D_Object_Pose_Estimation_via_Learning_Decoupled_ICCV_2023_paper.html)][[arxivl ink](https://arxiv.org/abs/2308.09916)][[code|official](https://github.com/JiehongLin/VI-Net)]

* **IST-Net(ICCV2023)** IST-Net: Prior-free Category-level Pose Estimation with Implicit Space Transformation [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_IST-Net_Prior-Free_Category-Level_Pose_Estimation_with_Implicit_Space_Transformation_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.13479)][[project link](https://sites.google.com/view/cvmi-ist-net/)][[code|official](https://github.com/CVMI-Lab/IST-Net)][`The University of Hong Kong`, `It investigates whether 3D shape priors are necessary (The anwser is No)`]
