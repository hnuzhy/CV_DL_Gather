# Here are collections about other CV topics

**Contents Hyperlinks**

* [⭐3D Body Model Regression](#3d-body-model-regression)
* [⭐3D Reconstruction](#3d-reconstruction)
* [⭐6D Object Pose Estimation](#6d-object-pose-estimation)
* [⭐Aerial Autonomous Navigation](#aerial-autonomous-navigation)
* [⭐Automatic Speech Recognition](#automatic-speech-recognition)
* [⭐Camera Pose Estimation (SLAM)](#camera-pose-estimation-slam)
* [⭐Deep Neural Networks](#deep-neural-networks)
* [⭐Generative Adversarial Network](#generative-adversarial-network)
* [⭐Human Object Interaction Detection](#human-object-interaction-detection)
* [⭐Image Mosaic](#image-mosaic)
* [⭐Image Restoration](#image-restoration)
* [⭐Lane Detection](#lane-detection)
* [⭐Pedestrian Localization](#pedestrian-localization)
* [⭐Person ReID](#person-reid)
* [⭐Reinforcement Learning](#reinforcement-learning)
* [⭐Scene Text Detection](#scene-text-detection)
* [⭐Semantic Segmentation](#semantic-segmentation)
* [⭐Sound Source Localization](#sound-source-localization)
* [⭐Traffic Violation Detection](#traffic-violation-detection)


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐3D Body Model Regression
Also named ***3D Human Pose and Shape Regression*** or ***3D Human Pose and Shape Estimation*** or ***Human Mesh Recovery (HMS)***

### Materials


* **(blogs) OBJ Files** [[Everything You Need to Know About Using OBJ Files](https://www.marxentlabs.com/obj-files/)]
* **(blogs) OBJ Files** [[6 Best Free OBJ Editor Software For Windows](https://listoffreeware.com/free-obj-editor-software-windows/)]
* **(models) SMPL family, i.e. SMPL, SMPL+H, SMPL-X** [[codes|official github](https://github.com/vchoutas/smplx/tree/main/transfer_model)]
* **(survey)(arxiv2022) Recovering 3D Human Mesh from Monocular Images: A Survey** [[paper link](https://arxiv.org/abs/2203.01923)] [[project link](https://github.com/tinatiansjz/hmr-survey)] [[CVPR 2022 related works](https://github.com/tinatiansjz/hmr-survey/issues/1)]
 

### Papers

* **SMPL(SIGGRAPH2015)** SMPL: A Skinned Multi-Person Linear Model [[paper link](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf)][[project link](https://smpl.is.tue.mpg.de/)][`MPII 马普所`]

* **SMPL-X(CVPR2019)** Expressive Body Capture: 3D Hands, Face, and Body from a Single Image [[paper link](https://ps.is.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf)][[project link](https://smpl-x.is.tue.mpg.de/)][[codes|official](https://github.com/vchoutas/smplify-x)][`MPII 马普所`]

* **SPIN(ICCV2019)** Learning to Reconstruct 3D Human Pose and Shape via Model-Fitting in the Loop [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.html)][[project link](https://www.seas.upenn.edu/~nkolot/projects/spin/)][[codes|official](https://github.com/nkolot/SPIN)][`MPII 马普所`]

* **STAR(ECCV2020)** STAR: A Sparse Trained Articulated Human Body Regressor [[paper link](https://ps.is.mpg.de/uploads_file/attachment/attachment/618/star_paper.pdf)][[project link](https://star.is.tue.mpg.de/)][[codes|official](https://github.com/ahmedosman/STAR)][`MPII 马普所`]

* **ExPose(ECCV2020)** Monocular Expressive Body Regression through Body-driven Attention [[paper linkl](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_2)][[project link](https://expose.is.tue.mpg.de/)][[codes|official](https://github.com/vchoutas/expose)][`MPII 马普所`][`the pioneering work (regression-based method) for the full-body mesh recovery task`]

* **GTRS(ACMMM2021)** A Lightweight Graph Transformer Network for Human Mesh Reconstruction from 2D Human Pose [[paper link](https://arxiv.org/pdf/2111.12696.pdf)][[code|official](https://github.com/zczcwh/GTRS)]

* **DetNet(CVPR2021)** Monocular Real-Time Full Body Capture With Inter-Part Correlations [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Monocular_Real-Time_Full_Body_Capture_With_Inter-Part_Correlations_CVPR_2021_paper.html)][`no official code`]

* **PIXIE(3DV2021)** Collaborative regression of expressive bodies using moderation [[paper link](https://ps.is.mpg.de/uploads_file/attachment/attachment/667/PIXIE_3DV_CR.pdf)][[project link](https://pixie.is.tue.mpg.de/)][[codes|official](https://github.com/YadiraF/PIXIE)][`MPII 马普所`]

* **FrankMocap(ICCVW2021)** FrankMocap: A monocular 3D whole-body pose estimation system via regression and integration [[paper link](https://openaccess.thecvf.com/content/ICCV2021W/ACVR/html/Rong_FrankMocap_A_Monocular_3D_Whole-Body_Pose_Estimation_System_via_Regression_ICCVW_2021_paper.html)][[codes|official](https://github.com/facebookresearch/frankmocap)][`facebookresearch`]

* **LightweightMHMS(ICCV2021)** Lightweight Multi-Person Total Motion Capture Using Sparse Multi-View Cameras [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Lightweight_Multi-Person_Total_Motion_Capture_Using_Sparse_Multi-View_Cameras_ICCV_2021_paper.html)][`taking multi-view RGB sequences and body estimation results as inputs`, `using full-body model SMPL-X`, `Openpose + FaceAlignment + SRHandNet + HandHMR`]

* ❤**ROMP(ICCV2021)** Monocular, One-stage, Regression of Multiple 3D People [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Sun_Monocular_One-Stage_Regression_of_Multiple_3D_People_ICCV_2021_paper.html)][[codes|official](https://github.com/Arthur151/ROMP)][`related with MPII 马普所`]

* **PyMAF(ICCV2021 Oral)** PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop [[paper link](https://arxiv.org/pdf/2103.16507.pdf)][[project link](https://hongwenzhang.github.io/pymaf/)][[codes|official](https://github.com/HongwenZhang/PyMAF)]

* ❤**PyMAF-X(arxiv2022)** PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images [[paper link](https://arxiv.org/pdf/2207.06400.pdf)][[project link](https://www.liuyebin.com/pymaf-x/)][[codes|official](https://github.com/HongwenZhang/PyMAF)]

* **Hand4Whole(CVPRW2022)** Accurate 3D Hand Pose Estimation for Whole-body 3D Human Mesh Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022W/ABAW/html/Moon_Accurate_3D_Hand_Pose_Estimation_for_Whole-Body_3D_Human_Mesh_CVPRW_2022_paper.html)][[codes|official](https://github.com/mks0601/Hand4Whole_RELEASE)]

* ❤**BEV(CVPR2022)** Putting People in their Place: Monocular Regression of 3D People in Depth [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_Putting_People_in_Their_Place_Monocular_Regression_of_3D_People_CVPR_2022_paper.html)][[project link](https://arthur151.github.io/BEV/BEV.html)][[codes|official](https://github.com/Arthur151/ROMP)][[Relative Human dataset](https://github.com/Arthur151/Relative_Human)][`related with MPII 马普所`]

* ❤**hmr-benchmarks(NIPS2022)** Benchmarking and Analyzing 3D Human Pose and Shape Estimation Beyond Algorithms [[paper link](https://openreview.net/forum?id=rjBYortWdRV)][[codes|official](https://github.com/smplbody/hmr-benchmarks)]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐3D Reconstruction

### Materials


### Papers

* **PSVH-3d-reconstruction(AAAI2019)** Deep Single-View 3D Object Reconstruction with Visual Hull Embedding [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/4922)][[codes|official TensorFlow](https://github.com/HanqingWangAI/PSVH-3d-reconstruction)]

* **(arxiv2022)** A Real World Dataset for Multi-view 3D Reconstruction [[paper link](https://arxiv.org/abs/2203.11397)]

* **SDF(CVPR2022)** Neural 3D Scene Reconstruction with the Manhattan-world Assumption [[paper link](https://arxiv.org/abs/2205.02836)][[project link](https://zju3dv.github.io/manhattan_sdf/)][[codes|official PyTorch](https://github.com/zju3dv/manhattan_sdf)]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐6D Object Pose Estimation 

(or 3D Object Detection)

### Materials

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

### Datasets

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
* [❤ Objectron (CVPR2021)](https://github.com/google-research-datasets/Objectron): A Large Scale Dataset of Object-Centric Videos in the Wild With Pose Annotations
* [❤ PhoCaL (CVPR2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_PhoCaL_A_Multi-Modal_Dataset_for_Category-Level_Object_Pose_Estimation_With_CVPR_2022_paper.html): PhoCaL: A Multi-Modal Dataset for Category-Level Object Pose Estimation With Photometrically Challenging Objects [***A novel robot-supported multi-modal (RGB, depth, polarisation) benchmark with challenging scenes supporting RGB-D and monocular RGB methods***]
* [❤ ABO (Amazon Berkeley Objects) (CVPR2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Collins_ABO_Dataset_and_Benchmarks_for_Real-World_3D_Object_Understanding_CVPR_2022_paper.html) ABO: Dataset and Benchmarks for Real-World 3D Object Understanding [[project link](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)][[github link](https://github.com/jazcollins/amazon-berkeley-objects)][[paperswithcode homepage](https://paperswithcode.com/dataset/abo)][***A large-scale dataset designed for material prediction and multi-view retrieval experiments. The dataset contains Blender renderings of 30 viewpoints for each of the 7,953 3D objects, as well as camera intrinsics and extrinsic for each rendering.***]
* [❤ Objaverse (arxiv2022)](https://arxiv.org/abs/2212.08051): Objaverse: A Universe of Annotated 3D Objects [[project link](https://objaverse.allenai.org/)][[paperswithcode homepage](https://paperswithcode.com/dataset/objaverse)][***A large dataset of objects with 800K+ (and growing) 3D models with descriptive captions, tags, and animations.***]


### Papers

#### ⭐3D Object Detection

* ❤ **3D-BoundingBox(CVPR2017)** 3D Bounding Box Estimation Using Deep Learning and Geometry [[paper link](https://arxiv.org/abs/1612.00496)][[codes|official PyTorch](https://github.com/skhadem/3D-BoundingBox)]

* ❤ **SMOKE(CVPRW2020)** SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation [[paper link](https://openaccess.thecvf.com/content_CVPRW_2020/html/w60/Liu_SMOKE_Single-Stage_Monocular_3D_Object_Detection_via_Keypoint_Estimation_CVPRW_2020_paper.html)][[codes|official PyTorch](https://github.com/lzccccc/SMOKE)]

* **MonoPair(CVPR2020)** MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_MonoPair_Monocular_3D_Object_Detection_Using_Pairwise_Spatial_Relationships_CVPR_2020_paper.html)][[codes|]()]

* **RTM3D(ECCV2020)** RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving [[paper link](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480647.pdf)][[codes|]()]

* ❤ **FADNet(TIV2022)** Monocular 3D Object Detection with Sequential Feature Association and Depth Hint Augmentation [[paper link](https://arxiv.org/abs/2011.14589)][[codes|official](https://github.com/gtzly/FADNet)]

#### ⭐6D Object Pose Estimation

* **Pascal3D+(WACV2014)** Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild [[paper link](http://roozbehm.info/papers/Xiang14wacv.pdf)][[project link](https://cvgl.stanford.edu/projects/pascal3d)]

* **(ICCVW2017)** 3D Pose Regression Using Convolutional Neural Networks [[paper link](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w31/html/Mahendran_3D_Pose_Regression_ICCV_2017_paper.html)]

* **PoseCNN(RSS2018)** PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes [[paper link](https://yuxng.github.io/xiang_rss18.pdf)][[project link](https://rse-lab.cs.washington.edu/projects/posecnn/)]

* **(ECCV2018)** Occlusion Resistant Object Rotation Regression from Point Cloud Segments [[paper link](https://openaccess.thecvf.com/content_eccv_2018_workshops/w6/html/Gao_Occlusion_Resistant_Object_Rotation_Regression_from_Point_Cloud_Segments_ECCVW_2018_paper.html)]

* **DeepIM(ECCV2018)** DeepIM: Deep Iterative Matching for 6D Pose Estimation [[paper link](https://openaccess.thecvf.com/content_ECCV_2018/html/Yi_Li_DeepIM_Deep_Iterative_ECCV_2018_paper.html)][[project link](https://rse-lab.cs.washington.edu/projects/deepim/)]

* ❤**YOLO-6D(CVPR2018)** Real-Time Seamless Single Shot 6D Object Pose Prediction [[paper link](https://openaccess.thecvf.com/content_cvpr_2018/html/Tekin_Real-Time_Seamless_Single_CVPR_2018_paper.html)][[codes|official PyTorch](https://github.com/microsoft/singleshotpose)][[codes|unofficial TensorFlow](https://github.com/Mmmofan/YOLO_6D)][`YOLOv2`]

* ❤**YOLO-Seg(CVPR2019)** Segmentation-Driven 6D Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Hu_Segmentation-Driven_6D_Object_Pose_Estimation_CVPR_2019_paper.html)][[codes|official](https://github.com/cvlab-epfl/segmentation-driven-pose)][`YOLOv2`]

* ❤**PVNet(CVPR2019 Oral)** PVNet: Pixel-Wise Voting Network for 6DoF Pose Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Peng_PVNet_Pixel-Wise_Voting_Network_for_6DoF_Pose_Estimation_CVPR_2019_paper.html)][[codes|official](https://zju3dv.github.io/pvnet/)]

* **NOCS(CVPR2019 Oral)** Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.html)][[project link](https://geometry.stanford.edu/projects/NOCS_CVPR2019/)][[codes & datasets|official keras and tensorflow](https://github.com/hughw19/NOCS_CVPR2019)]

* **DPOD(ICCV2019)** DPOD: 6D Pose Object Detector and Refiner [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Zakharov_DPOD_6D_Pose_Object_Detector_and_Refiner_ICCV_2019_paper.html)][[codes|PyTorch](https://github.com/zakharos/DPOD)]

* **CDPN(ICCV2019)** CDPN: Coordinates-Based Disentangled Pose Network for Real-Time RGB-Based 6-DoF Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_CDPN_Coordinates-Based_Disentangled_Pose_Network_for_Real-Time_RGB-Based_6-DoF_Object_ICCV_2019_paper.html)][[codes|official PyTorch](https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi)][`YOLOv3`]

* **HybridPose(CVPR2020)** HybridPose: 6D Object Pose Estimation Under Hybrid Representations [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Song_HybridPose_6D_Object_Pose_Estimation_Under_Hybrid_Representations_CVPR_2020_paper.html)]

* **single-stage-pose(CVPR2020)** Single-Stage 6D Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Hu_Single-Stage_6D_Object_Pose_Estimation_CVPR_2020_paper.html)][[codes|official PyTorch](https://github.com/cvlab-epfl/single-stage-pose)]

* **CosyPose(ECCV2020)** CosyPose: Consistent Multi-view Multi-object 6D Pose Estimation [[paper link](https://hal.inria.fr/hal-02950800/)][[project link](https://www.di.ens.fr/willow/research/cosypose)][[codes|official PyTorch](https://github.com/ylabbe/cosypose)]

* **MobilePose(arxiv2020)** MobilePose: Real-Time Pose Estimation for Unseen Objects with Weak Shape Supervision [[paper link](https://arxiv.org/abs/2003.03522)]

* **SGPA(ICCV2021)** SGPA: Structure-Guided Prior Adaptation for Category-Level 6D Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_SGPA_Structure-Guided_Prior_Adaptation_for_Category-Level_6D_Object_Pose_Estimation_ICCV_2021_paper.html)][[codes|PyTorch](https://github.com/leo94-hk/SGPA)]

* ❤**Objectron(CVPR2021)(Training Codes ╳)(Annotation Tool ╳)]** Objectron: A Large Scale Dataset of Object-Centric Videos in the Wild With Pose Annotations [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Ahmadyan_Objectron_A_Large_Scale_Dataset_of_Object-Centric_Videos_in_the_CVPR_2021_paper.html)][[codes|PyTorch+TensorFlow](https://github.com/google-research-datasets/Objectron)][[official blog 1: MediaPipe](https://mediapipe.dev/)][[official blog 2: MediaPipe Objectron](https://google.github.io/mediapipe/solutions/objectron)]

* **SAR-Net(CVPR2022)** SAR-Net: Shape Alignment and Recovery Network for Category-level 6D Object Pose and Size Estimation [[paper link](https://arxiv.org/abs/2106.14193)][[project link](https://hetolin.github.io/SAR-Net/)][[codes|official](https://github.com/hetolin/SAR-Net)]

* **OVE6D-pose(CVPR2022)** OVE6D: Object Viewpoint Encoding for Depth-based 6D Object Pose Estimation [[paper link](https://arxiv.org/pdf/2203.01072.pdf)][[project link](https://dingdingcai.github.io/ove6d-pose/)][[codes|official](https://github.com/dingdingcai/OVE6D-pose)]

* **OnePose(CVPR2022)** OnePose: One-Shot Object Pose Estimation without CAD Models [[paper link](https://arxiv.org/pdf/2205.12257.pdf)][[project link](https://zju3dv.github.io/onepose/)][[codes|official](https://github.com/zju3dv/OnePose)][`ZJU + Objectron`]

* **Gen6D(ECCV2022)** Gen6D: Generalizable Model-Free 6-DoF Object Pose Estimation from RGB Images [[paper link](https://arxiv.org/abs/2204.10776)][[project link](https://liuyuan-pal.github.io/Gen6D/)][[codes|on the way]()]

* ❤**CenterSnap(ICRA2022)** CenterSnap: Single-Shot Multi-Object 3D Shape Reconstruction and Categorical 6D Pose and Size Estimation [[paper link](https://arxiv.org/abs/2203.01929)][[project link](https://zubair-irshad.github.io/projects/CenterSnap.html)][[codes|official PyTorch](https://github.com/zubair-irshad/CenterSnap)]

* ❤**CenterPose(ICRA2022)(Training with CenterNet and Objectron)** Single-stage Keypoint-based Category-level Object Pose Estimation from an RGB Image [[paper link](https://arxiv.org/abs/2109.06161)][[project link](https://sites.google.com/view/centerpose)][[author homepage](https://yunzhi.netlify.app/)][[codes|official PyTorch](https://github.com/NVlabs/CenterPose)][`Nvidia + Objectron + one-stage + end2end`][based on `FADNet` (https://github.com/gtzly/FADNet) and `CenterNet` (https://github.com/xingyizhou/CenterNet)]

* 👍**EPro-PnP(CVPR2022 Oral, Best Student Paper)** EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_EPro-PnP_Generalized_End-to-End_Probabilistic_Perspective-N-Points_for_Monocular_Object_Pose_Estimation_CVPR_2022_paper.pdf)][[code|official](https://github.com/tjiiv-cprg/EPro-PnP)]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Aerial Autonomous Navigation

### Materials

* [(github) CMU: Leveraging system development and robot deployment for aerial autonomous navigation.](https://github.com/caochao39/aerial_navigation_development_environment) [[demo video](https://www.bilibili.com/video/BV1tZ4y187HR)]

### Papers


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Automatic Speech Recognition

### Materials

* [(github) **Whisper by OpenAI** Robust Speech Recognition via Large-Scale Weak Supervision](https://github.com/openai/whisper) [[paper link](https://arxiv.org/abs/2212.04356)][[blogs](https://openai.com/blog/whisper/)]

### Papers


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Camera Pose Estimation (SLAM)

### Materials

* [(cnblogs) 视觉SLAM(Simultaneous Localization and Mapping)漫谈](https://blog.csdn.net/weixin_41537599/article/details/110819969)

### Papers

* **ORB-SLAM(TRO2015)** ORB-SLAM: a Versatile and Accurate Monocular SLAM System [[paper link](https://arxiv.org/abs/1502.00956)][[project link](http://webdiis.unizar.es/~raulmur/orbslam/)][[codes|official ROS](https://github.com/raulmur/ORB_SLAM)]

* **ORB-SLAM2(TRO2017)** ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras [[paper link](https://arxiv.org/abs/1610.06475)][[project link](http://webdiis.unizar.es/~raulmur/orbslam/)][[codes|official ROS](https://github.com/raulmur/ORB_SLAM2)]

* **GeoNet(CVPR2018)** GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose [[paper link](https://openaccess.thecvf.com/content_cvpr_2018/html/Yin_GeoNet_Unsupervised_Learning_CVPR_2018_paper.html)][[codes|official Tensorflow](https://github.com/yzcjtr/GeoNet)]

* **DeepVO(ICRA2017)** DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks [[paper link](https://arxiv.org/pdf/1709.08429.pdf)][[project link](http://senwang.gitlab.io/DeepVO/)][[codes|unofficial PyTorch 1](https://github.com/ChiWeiHsiao/DeepVO-pytorch)][[codes|unofficial PyTorch 2](https://github.com/krrish94/DeepVO)]

* **BiLevelOpt(3DV2020)** Joint Unsupervised Learning of Optical Flow and Egomotion with Bi-Level Optimization [[paper link](https://arxiv.org/abs/2002.11826)]

* **TartanVO(CoRL2021)** TartanVO: A Generalizable Learning-based VO [[paper link](https://proceedings.mlr.press/v155/wang21h.html)][[codes|official PyTorch](https://github.com/castacks/tartanvo)]

* **❤ DiffPoseNet(CVPR2022)** DiffPoseNet: Direct Differentiable Camera Pose Estimation [[paper link](https://arxiv.org/abs/2203.11174)][[first author](https://analogicalnexus.github.io/)][[project link](https://nitinjsanket.github.io/research.html)][[codes|official PyTorch]()]


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

* **FFA(Hinton 2022)** The Forward-Forward Algorithm: Some Preliminary Investigations [[paper link](https://www.cs.toronto.edu/~hinton/FFA13.pdf)]


**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Generative Adversarial Network

### Materials

* [(blog) Test and Train CycleGAN](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb#scrollTo=OzSKIPUByfiN)
* [(CSDNblog) CycleGAN论文的阅读与翻译，无监督风格迁移](https://zhuanlan.zhihu.com/p/45394148)
* [(CSDNblog) 生成对抗网络(四)CycleGAN讲解](https://blog.csdn.net/qq_40520596/article/details/104714762)
* [(blog) What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

### Papers

#### ▲ GAN-based

* ❤ **CycleGAN(ICCV2017)** Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [[arxiv link](https://arxiv.org/pdf/1703.10593.pdf)][[project link](https://junyanz.github.io/CycleGAN/)][[Codes|PyTorch(official)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]

* ❤ **CUT(ECCV2020)** Contrastive Learning for Unpaired Image-to-Image Translation [[arxiv link](https://arxiv.org/abs/2007.15651)][[project link](http://taesung.me/ContrastiveUnpairedTranslation/)][[Codes|PyTorch(official)](https://github.com/taesungp/contrastive-unpaired-translation)]


#### ▲ Diffusion-based

* ❤ **GET3D(NIPS2022)** GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images [[paper link](https://nv-tlabs.github.io/GET3D/assets/paper.pdf)][[project link](https://nv-tlabs.github.io/GET3D/)][[codes|official PyTorch](https://github.com/nv-tlabs/GET3D)][`NVIDIA`]

* ❤ **SCAM(ECCV2022)** SCAM! Transferring humans between images with Semantic Cross Attention Modulation [[paper link](https://arxiv.org/abs/2210.04883)][[project link](https://imagine.enpc.fr/~dufourn/publications/scam.html)][[codes|official PyTorch](https://github.com/nicolas-dufour/SCAM)]

* **SDEdit(ICLR2022)** SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations [[paper link](https://arxiv.org/abs/2108.01073)][[project link](https://sde-image-editing.github.io/)][`Partial StyleGAN`]

* **HumanDiffusion(arxiv2022)** HumanDiffusion: a Coarse-to-Fine Alignment Diffusion Framework for Controllable Text-Driven Person Image Generation [[paper link](https://arxiv.org/abs/2211.06235)][`Human related image generation`]

* **Dream3D(arxiv2022)** Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models [[paper link](https://arxiv.org/abs/2212.14704)][[project link](https://bluestyle97.github.io/dream3d/)]


#### ▲ NeRF-based

* 👍**NeRF(ECCV2020)** NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis [[paper link](https://dl.acm.org/doi/abs/10.1007/978-3-030-58452-8_24)]

* **NerfCap(TVCG2022)** NerfCap: Human Performance Capture With Dynamic Neural Radiance Fields [[paper link](https://ieeexplore.ieee.org/abstract/document/9870173)]

* **HumanNeRF(CVPR2022)** HumanNeRF: Efficiently Generated Human Radiance Field from Sparse Inputs [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_HumanNeRF_Efficiently_Generated_Human_Radiance_Field_From_Sparse_Inputs_CVPR_2022_paper.html)][`Human related image generation`]

* 👍**Humannerf(CVPR2022 Oral)** HumanNeRF: Free-Viewpoint Rendering of Moving People From Monocular Video [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Weng_HumanNeRF_Free-Viewpoint_Rendering_of_Moving_People_From_Monocular_Video_CVPR_2022_paper.html)][[project link](https://grail.cs.washington.edu/projects/humannerf/)][[code|official](https://github.com/chungyiweng/humannerf)][`Human related image generation`]

* 👍**NeuMan(ECCV2022)** NeuMan: Neural Human Radiance Field from a Single Video [[paper link](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920400.pdf)][[code|official](https://github.com/apple/ml-neuman)][`Human related image generation`]

* **Neural-Sim(ECCV2022)** Neural-Sim: Learning to Generate Training Data with NeRF [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20050-2_28)][[code|official](https://github.com/gyhandy/Neural-Sim-NeRF)]

* ⭐**MoFaNeRF(ECCV2022)** MoFaNeRF:Morphable Facial Neural Radiance Field [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20062-5_16)][[code|official](https://github.com/zhuhao-nju/mofanerf)][`Face or head related NeRF`]

* **headshot(arxiv2022)** Novel View Synthesis for High-fidelity Headshot Scenes [[paper link](https://arxiv.org/abs/2205.15595)][[code|official](https://github.com/showlab/headshot)][`Face or head related NeRF`]

* **FLNeRF(arxiv2022)** FLNeRF: 3D Facial Landmarks Estimation in Neural Radiance Fields [[paper link](https://arxiv.org/abs/2211.11202)][[project link](https://github.com/ZHANG1023/FLNeRF)][`Face or head related NeRF`]

* **HexPlane(arxiv2023)** HexPlane: A Fast Representation for Dynamic Scenes [[paper link](https://arxiv.org/abs/2301.09632)][[project link](https://caoang327.github.io/HexPlane)]

* **K-Planes(arxiv2023)** K-Planes: Explicit Radiance Fields in Space, Time, and Appearance  [[paper link](https://arxiv.org/abs/2301.10241)][[project link](https://sarafridov.github.io/K-Planes/)]

* **MAV3D(Make-A-Video3D)(arxiv2023)** Text-To-4D Dynamic Scene Generation [[paper link](https://arxiv.org/abs/2301.11280)][[project link](https://make-a-video3d.github.io/)]



**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Human Object Interaction Detection

### Materials

* [(DEtection TRansformer(DETR) ECCV2020 BestPaper) End-to-End Object Detection with Transformers](https://link.springer.com/chapter/10.1007/978-3-030-58452-8_13) [[codes|official](https://github.com/facebookresearch/detr)]
* [(DeformableDETR ICLR2021 OralPaper) Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159) [[codes|official](https://github.com/fundamentalvision/Deformable-DETR)][[bilibili paper reading video](https://www.bilibili.com/video/BV133411m7VP/)]
* [(DAB-DETR ICLR2022) DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR](https://arxiv.org/abs/2201.12329) [[codes|official](https://github.com/SlongLiu/DAB-DETR)]
* [(DN-DETR CVPR2022 OralPaper) DN-DETR: Accelerate DETR Training by Introducing Query DeNoising](https://openaccess.thecvf.com/content/CVPR2022/html/Li_DN-DETR_Accelerate_DETR_Training_by_Introducing_Query_DeNoising_CVPR_2022_paper.html) [[codes|official](https://github.com/IDEA-Research/DN-DETR)]

### Datasets

* **V-COCO (arxiv2015)** [Visual Semantic Role Labeling](https://arxiv.org/abs/1505.04474) [[github link](https://github.com/s-gupta/v-coco)][[paperswithcode page](https://paperswithcode.com/dataset/v-coco)]
* **HICO-DET (WACV2018)** [Learning to Detect Human-Object Interactions](https://ieeexplore.ieee.org/abstract/document/8354152) [[project link](http://www-personal.umich.edu/~ywchao/hico/)][[csdn blogs](https://blog.csdn.net/irving512/article/details/115122416)][[paperswithcode page](https://paperswithcode.com/dataset/hico-det)]
* **HAKE (2018~2022)** [HAKE: Human Activity Knowledge Engine](http://hake-mvig.cn/home/) [[github link](https://github.com/DirtyHarryLYL/HAKE)]

### Papers

* **UnionDet(ECCV2020)** UnionDet: Union-Level Detector Towards Real-Time Human-Object Interaction Detection [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_30)][[codes|official]()]

* **MSTR(CVPR2022)** MSTR: Multi-Scale Transformer for End-to-End Human-Object Interaction Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_MSTR_Multi-Scale_Transformer_for_End-to-End_Human-Object_Interaction_Detection_CVPR_2022_paper.html)][[codes|official]()]

* **DisTrans(CVPR2022)** Human-Object Interaction Detection via Disentangled Transformer [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Human-Object_Interaction_Detection_via_Disentangled_Transformer_CVPR_2022_paper.html)][[codes|official]()]


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

* **❤ LaneNet(IVS2018)** Towards End-to-End Lane Detection: an Instance Segmentation Approach [[arxiv link](https://arxiv.org/abs/1802.05591)][[project link](https://maybeshewill-cv.github.io/lanenet-lane-detection/)][[Codes|unoffical TF](https://github.com/MaybeShewill-CV/lanenet-lane-detection)]

* **❤ UltraLane(ECCV2020)** Ultra Fast Structure-aware Deep Lane Detection [[arxiv link](https://arxiv.org/abs/2004.11757)][[Codes|offical PyTorch](https://github.com/cfzd/Ultra-Fast-Lane-Detection)]

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

## ⭐Reinforcement Learning
<details>
<summary>Click Here to Show All</summary>
* **[Definition]** Reinforcement learning (RL) is used to describe and solve the problem that agents use learning strategies to take actions to maximize reward or achieve specific goals in the process of interaction with the environment.

* **[Supplement]** The common model of RL is standard **Markov Decision Process** (MDP). According to the given conditions, RL can be divided into **model-based RL** and **model-free RL**. The algorithms used to solve RL problems can be divided into strategy search algorithm and value function algorithm. Deep learning model can be used in RL to form deep reinforcement learning. Inspired by **behaviorist psychology**, RL focuses on online learning and tries to maintain a balance between exploration and exploitation. Unlike **supervised learning** and **unsupervised learning**, RL does not require any given data in advance, but obtains learning information and updates model parameters by receiving reward (feedback) from environment. RL has been discussed in the fields of **information theory**, **game theory** and **automatic control**. It is used to explain the **equilibrium state under bounded rationality**, design **recommendation system** and robot interaction system. Some complex RL algorithms have general intelligence to solve complex problems to a certain extent, which can reach the human level in go and electronic games. The learning cost and training cost of RL are very high.
</details>

```
Lectures and Courses

**2020-11-11 强化学习课程tips**
* 好的机械臂十分昂贵，一般都是机械动力与自动化专业研发设计，不适合采用模拟环境下诞生的强化学习方法来指导生成智能化机械臂；
* 国内技术还是被卡脖子的，智能机械臂的生产领头行业在日本，即使清华的自动化系，也需要去向日本采购昂贵的单个近百万的机械臂；

**2020-11-25 Imitation Learning 模仿学习**
* Behavior Cloning 相当于是监督学习
* Inverse Reinforcement Learning(IRL) 逆强化学习，类似于生成对抗网络GAN
* 实验室研究进展分享 -- 沉浸式视频（360度全景视频）：处理和传输
  * 多目相机（8目或12目）购买成本还很高
  * 多角度实时拼接的时延问题：接近10K带宽传输、拼接算法、编解码算法
  * 平面卷积不适用：球面到平面的投影，使用主流的方法（如地球仪投影ERP等）
  * 或者抛开平面卷积的思路：使用球面卷积核及其算子；旋转等变的图卷积网络（需要转换到频域）
  * 直接逼近和多项式回归逼近，来定量估计神经网络复杂度，指导设计神经网络的层数和总节点数（构建多元高阶多项式与神经元组之间的等价关系）
```

### Materials

### Papers

**-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-|-+-**

## ⭐Scene Text Detection

### Materials

* [Commonly used four datasets in scene text detection: ICDAR2015, CTW1500, Total-Text and MSRA-TD500]
* [(github) 🚀PaddleOCR: Awesome multilingual OCR toolkits based on PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR)

### Papers

#### Regression-based

* **R2CNN(arxiv2017)** R2CNN: Rotational Region CNN for Orientation Robust Scene Text Detection [[paper link](https://arxiv.org/abs/1706.09579)]

* **TextBoxes(AAAI2017)** TextBoxes: A Fast Text Detector with a Single Deep Neural Network [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/11196)]

* **❤ EAST(CVPR2017)** EAST: An Efficient and Accurate Scene Text Detector [[paper link](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_EAST_An_Efficient_CVPR_2017_paper.html)]

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

## ⭐Sound Source Localization

### Materials

* [(cnblogs) 【论文导读】Learning to Localize Sound Source in Visual Scenes】&soundnet的复现](https://blog.csdn.net/zzc15806/article/details/80772152)
* [(cnblogs) 论文【Learning to Localize Sound Source in Visual Scenes】&soundnet的复现](https://www.cnblogs.com/gaoxiang12/p/3695962.html)
* [(CSDNblogs) 麦克风阵列声源定位 GCC-PHAT](https://blog.csdn.net/u010592995/article/details/79735198)
* [(online PPT) 语音识别技术的前世今生(made by 王赟(Maigo))](https://zhihu-live.zhimg.com/0af15bfda98f5885ffb509acd470b0fa)


### Datasets

* **Columbia dataset (ECCV2016)** Cross-modal Supervision for Learning Active Speaker Detection in Video [[paper link](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_18)]

* **LRS2 (TPAMI2018)** Deep Audio-Visual Speech Recognition [[paper link](https://ieeexplore.ieee.org/abstract/document/8585066)][[dataset link](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/)]

* **LRS3 (arxiv2018)** LRS3-TED: a large-scale dataset for visual speech recognition [[paper link](https://arxiv.org/abs/1809.00496)][[dataset link](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/)]

* **dataset annotation tool: VIA (ACMMM2019)** The VIA Annotation Software for Images, Audio and Video [[paper link](https://www.robots.ox.ac.uk/~adutta/data/postdoc/dutta2019vgg.pdf)][[project link](https://www.robots.ox.ac.uk/~vgg/software/via/)]


### Papers

* **❤ SoundNet(NIPS2016)** SoundNet: Learning Sound Representations from Unlabeled Video [[arxiv link](https://arxiv.org/pdf/1610.09001.pdf)][[project link](http://projects.csail.mit.edu/soundnet/)][[Codes|offical TensorFlow](https://github.com/cvondrick/soundnet)][[CSDN blog](https://blog.csdn.net/zzc15806/article/details/80669883)](`Dataset: SoundNet`)

* **AVC(ICCV2017)** Look, Listen and Learn [[paper link](https://openaccess.thecvf.com/content_iccv_2017/html/Arandjelovic_Look_Listen_and_ICCV_2017_paper.html)]

* **Multisensory(ECCV2018)** Audio-Visual Scene Analysis with Self-Supervised Multisensory Features [[paper link](https://openaccess.thecvf.com/content_ECCV_2018/html/Andrew_Owens_Audio-Visual_Scene_Analysis_ECCV_2018_paper.html)][[project link](https://andrewowens.com/multisensory/)][[codes|officical TensorFlow](https://github.com/andrewowens/multisensory)]

* **❤ SoundLocation or Attention10k(CVPR2018)** Learning to Localize Sound Source in Visual Scenes [[arxiv link](https://arxiv.org/pdf/1803.03849.pdf)][[Codes|offical PyTorch based on SoundNet](https://github.com/ardasnck/learning_to_localize_sound_source)][[Codes|unofficial PyTorch](https://github.com/liyidi/soundnet_localize_sound_source)](`Dataset: Flickr-SoundNet`)

* **DMC(CVPR2019)** Deep Multimodal Clustering for Unsupervised Audiovisual Learning [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Hu_Deep_Multimodal_Clustering_for_Unsupervised_Audiovisual_Learning_CVPR_2019_paper.html)]

* **MSSL(ECCV2020)** Multiple Sound Sources Localization from Coarse to Fine [[paper link](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650290.pdf)][[codes|official PyTorch](https://github.com/shvdiwnkozbw/Multi-Source-Sound-Localization)][[Author - [Weiyao Lin]](https://weiyaolin.github.io/)]

* **AVVP(ECCV2020)** Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing [[paper link](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480443.pdf)][[codes|official PyTorch](https://github.com/YapengTian/AVVP-ECCV20)]

* **AVObjects(ECCV2020)** Self-Supervised Learning of Audio-Visual Objects from Video [[paper link](https://arxiv.org/abs/2008.04237)][[project link](https://www.robots.ox.ac.uk/~vgg/research/avobjects/)][[Oxford VGG](https://www.robots.ox.ac.uk/~vgg/)][[codes|official PyTorch](https://github.com/afourast/avobjects)]

* **❤ LVS or VGG-Sound(CVPR2021)** Localizing Visual Sounds the Hard Way [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Localizing_Visual_Sounds_the_Hard_Way_CVPR_2021_paper.html)][[codes|official PyTorch](https://github.com/hche11/Localizing-Visual-Sounds-the-Hard-Way)](`Dataset: VGG-Sound Source`)

* **vanilla-LVS or HardPos(ICASSP2022)** Learning Sound Localization Better from Semantically Similar Samples [[paper link](https://ieeexplore.ieee.org/abstract/document/9747867)]

* **❤ EZ-VSL(arxiv2022)** Localizing Visual Sounds the Easy Way [[paper link](https://arxiv.org/abs/2203.09324)][[codes|official PyTorch](https://github.com/stoneMo/EZ-VSL)][[multiple-instance-learning](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf)]

* **IEr(AAAI2022)** Visual Sound Localization in the Wild by Cross-Modal Interference Erasing [[paper link](https://www.aaai.org/AAAI22Papers/AAAI-140.LiuX.pdf)][[codes|official](https://github.com/alvinliu0/Visual-Sound-Localization-in-the-Wild)]

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



