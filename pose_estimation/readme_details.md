#  List for public implementation of various algorithms

## 1) Pubilc Datasets and Challenges

* [LIP(Look Into Person)](http://www.sysu-hcp.net/lip/index.php)
* [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#)
* [COCO - Common Objects in Context](https://cocodataset.org/)
* [PoseTrack: Dataset and Benchmark](https://posetrack.net/)
* [CMU Panoptic Studio Dataset (3D single and multiple real person pose in the lab)](http://domedb.perception.cs.cmu.edu/)
* [SURREAL dataset (CVPR2017) (3D single synthetic person pose in the indoor)](https://www.di.ens.fr/willow/research/surreal/)[[paper link](https://arxiv.org/abs/1701.01370)]
* [Drive&Act dataset (ICCV2019) (3D openpose single real person pose in the car with 5 views)](https://www.driveandact.com/)[[paper link](https://www.driveandact.com/publication/2019_iccv_drive_and_act/2019_iccv_drive_and_act.pdf)]
* [COCO-WholeBody (ECCV2020) (re-annotated based on keypoints in COCO dataset)](https://github.com/jin-s13/COCO-WholeBody)[[paper link](https://arxiv.org/abs/2007.11858)]
* [IKEA ASSEMBLY DATASET (WACV2021) (3D single and multiple real person pose in the lab with 3 views)](https://ikeaasm.github.io/)[[paper link](https://arxiv.org/abs/2007.00394)][[google drive](https://drive.google.com/drive/folders/1xkDp--QuUVxgl4oJjhCDb2FWNZTkYANq)]
* [Yoga-82: A New Dataset for Fine-grained Classification of Human Poses](https://sites.google.com/view/yoga-82/home)[[kaggle](https://www.kaggle.com/shrutisaxena/yoga-pose-image-classification-dataset)]
* [UAV-Human Dataset (CVPR2021) (not all appeared persons are annotated)](https://github.com/SUTDCV/UAV-Human)[[paper link](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_UAV-Human_A_Large_Benchmark_for_Human_Behavior_Understanding_With_Unmanned_CVPR_2021_paper.pdf)][[google drive](https://drive.google.com/drive/folders/1QeYXeM_pbWBSSmpRr_rKHurMpI2TxAKs)]
* [Mirrored-Human Dataset (CVPR2021 Oral)](https://zju3dv.github.io/Mirrored-Human/)[[paper link](https://arxiv.org/pdf/2104.00340.pdf)]

## 2) Pioneers and Experts

[👍Alejandro Newell](https://www.alejandronewell.com/)
[👍Jia Deng](https://www.cs.princeton.edu/~jiadeng/)
[👍Zhe Cao](https://people.eecs.berkeley.edu/~zhecao/)
[👍Tomas Simon](http://www.cs.cmu.edu/~tsimon/)
[👍tensorboy](https://github.com/tensorboy)
[👍murdockhou](https://github.com/murdockhou)



## 3) Blogs, Videos and Applications


* [(B站 video) 张锋-2D单人人体姿态估计及其应用](https://www.bilibili.com/video/av19006542/)
* [(B站 video) 人工智能 | 基于人体骨架的行为识别](https://www.bilibili.com/video/BV1wt411p7Ut/?spm_id_from=333.788.videocard.0)
* [(Website) 姿态估计交流网站ilovepose](http://www.ilovepose.cn/)
* [(CSDN blog) Paper List：CVPR 2018 人体姿态估计相关](https://blog.csdn.net/BockSong/article/details/80899689)
* [(blog) ECCV 2020 论文大盘点-姿态估计与动作捕捉篇](https://my.oschina.net/u/4580264/blog/4654293)
* [(blog) ECCV 2020 论文大盘点-3D人体姿态估计篇](https://xw.qq.com/cmsid/20200930A03Q3Y00)
* [(github) Awesome Human Pose Estimation (cbsudux)](https://github.com/cbsudux/awesome-human-pose-estimation)
* [(github) Awesome Human Pose Estimation (wangzheallen)](https://github.com/wangzheallen/awesome-human-pose-estimation)
* [(real time pose in github) tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)
* [(real time pose in github) 💃 Real-time single person pose estimation for Android and iOS](https://github.com/edvardHua/PoseEstimationForMobile)
* [(real time pose in github) Real-time 2D MPPE on CPU: Lightweight OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
* [(Application) FXMirror虚拟试衣解决方案](http://fxmirror.net/zh/features)
* [(Application) 3D试衣间：人工智能虚拟试衣系统](http://3d.oleoad.com/3dshiyi.asp)


## 4) Papers and Sources Codes

### ▶ Single Person Pose Estimation

* **Modeep(ACCV2014)(video based)** MoDeep: A Deep Learning Framework Using Motion Features for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1409.7963)]

* **(NIPS2014)(heatmaps)** Joint Training of a Convolutional Network and a Graphical Model for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1406.2984v1)]

* ⭐**PoseMachines(ECCV2014)(regression)** Pose Machines: Articulated Pose Estimation via Inference Machines [[paper link](https://www.ri.cmu.edu/pub_files/2014/7/poseMachines.pdf)][[project link](http://www.cs.cmu.edu/~vramakri/poseMachines.html)]

* ⭐**DeepPose(CVPR2014)(AlexNet based)(regression)** DeepPose: Human Pose Estimation via Deep Neural Networks [[arxiv link](https://arxiv.org/abs/1312.4659)][[Codes|OpenCV(unoffical)](https://github.com/mitmul/deeppose)]

* **(ICCV2015)(video based)** Flowing ConvNets for Human Pose Estimation in Videos [[arxiv link](https://arxiv.org/abs/1506.02897)]

* **(ECCV2016)(heatmaps)** Human Pose Estimation using Deep Consensus Voting [[arxiv link](https://arxiv.org/abs/1603.08212)]

* **(CVPR2016)(structure information)** End-To-End Learning of Deformable Mixture of Parts and Deep Convolutional Neural Networks for Human Pose Estimation [[paper link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Yang_End-To-End_Learning_of_CVPR_2016_paper.html)]

* **(CVPR2016)(structure information)** Structured Feature Learning for Pose Estimation [[paper link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Chu_Structured_Feature_Learning_CVPR_2016_paper.html)]

* **IEF(CVPR2016)(GoogleNet Based)(regression)** Human Pose Estimation with Iterative Error Feedback [[arxiv link](https://arxiv.org/abs/1507.06550)]

* ⭐**CPM(CVPR2016)(heatmaps)** Convolutional Pose Machines [[arxiv link](https://arxiv.org/abs/1602.00134)][[Codes|Caffe(offical)](https://github.com/shihenw/convolutional-pose-machines-release)][[Codes|Tensorflow(unoffical)](https://github.com/psycharo/cpm)]

* ⭐**StackedHourglass(ECCV2016)(heatmaps)** Stacked Hourglass Networks for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1603.06937)][[Codes|Torch7(offical old)](https://github.com/princeton-vl/pose-hg-train)][[Codes|PyTorch(offical new)](https://github.com/princeton-vl/pytorch_stacked_hourglass)][[Codes|Tensorflow(unoffical)](https://github.com/wbenbihi/hourglasstensorlfow)]

* **HourglassResidualUnits(HRUs)(CVPR2017)(heatmaps)** Multi-context Attention for Human Pose Estimation [[arciv link](https://arxiv.org/abs/1702.07432)]

* **PyraNet(ICCV2017)(heatmaps)** Learning Feature Pyramids for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1708.01101)][[Codes|Torch(offical)](https://github.com/bearpaw/PyraNet)]

* **(ICCV2017)(ResNet-50 Based)(regression)** Compositional Human Pose Regression [[arxiv link](https://arxiv.org/abs/1704.00159)]

* ⭐**Adversarial-PoseNet(ICCV2017)(GAN)** Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1705.00389)][[Codes|PyTorch(unoffical)](https://github.com/rohitrango/Adversarial-Pose-Estimation)]

* **(ECCV2018)(structure information)** Multi-Scale Structure-Aware Network for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1803.09894)]

* **(ECCV2018)(structure information)** Deeply Learned Compositional Models for Human Pose Estimation [[paper link](https://link.springer.com/chapter/10.1007%2F978-3-030-01219-9_12)]

* **(CVPR2018)(multi-task/video based)(regression)** 2D/3D Pose Estimation and Action Recognition using Multitask Deep Learning [[arxiv link](https://arxiv.org/abs/1802.09232)]

* **(CVPR2019)(structure information)** Does Learning Specific Features for Related Parts Help Human Pose Estimation? [[paper link](eeexplore.ieee.org/document/8953713)]

* **(arxiv2020)(video based)** Key Frame Proposal Network for Efficient Pose Estimation in Videos [[arxiv link](https://arxiv.org/abs/2007.15217)]

* **UniPose(CVPR2020)(video based)** UniPose: Unified Human Pose Estimation in Single Images and Videos [[arxiv link](https://arxiv.org/abs/2001.08095)][[Codes|PyTorch(offical)](https://github.com/bmartacho/UniPose)]



### ▶ Two-Stage [Top-Down] Multiple Person Pose Estimation

* **(ECCV2016)** Multi-Person Pose Estimation with Local Joint-to-Person Associations [[arxiv link](https://arxiv.org/abs/1608.08526)]

* ⭐**DeeperCut(ECCV2016)** DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model [[arxiv link](http://arxiv.org/abs/1605.03170)][[project link](http://pose.mpi-inf.mpg.de/#)][[Codes|Tensorflow(offical)](https://github.com/eldar/pose-tensorflow)]

* **(CVPR2017)** Towards Accurate Multi-person Pose Estimation in the Wild [[arxiv link](https://arxiv.org/abs/1701.01779)]

* **(ICCV2017)** A Coarse-Fine Network for Keypoint Localization [[paper link](https://openaccess.thecvf.com/content_iccv_2017/html/Huang_A_Coarse-Fine_Network_ICCV_2017_paper.html)]

* ⭐**AlphaPose/RMPE(ICCV2017)** RMPE: Regional Multi-person Pose Estimation [[arxiv link](https://arxiv.org/abs/1612.00137)][[project link](https://www.mvig.org/research/alphapose.html)][[Codes|PyTorch(offical)](https://github.com/MVIG-SJTU/AlphaPose)]

* ⭐**SimpleBaseline(ECCV2018)** Simple Baselines for Human Pose Estimation and Tracking [[arxiv link](https://arxiv.org/abs/1804.06208)][[Codes|PyTorch(offical)](https://github.com/Microsoft/human-pose-estimation.pytorch)][[Codes|PyTorch(flowtrack part)](https://github.com/simochen/flowtrack.pytorch)]

* ⭐**CPN(CVPR2018)** Cascaded Pyramid Network for Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/1711.07319)][[Codes|Tensorflow(offical)](https://github.com/chenyilun95/tf-cpn)][[Codes|Tensorflow(offical megvii)](https://github.com/megvii-detection/tf-cpn)][[zhihu blogs](https://zhuanlan.zhihu.com/p/37582402)]

* ⭐**HRNet(CVPR2019)** Deep High-Resolution Representation Learning for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1902.09212)][[Codes|PyTorch(offical)](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)][[Codes|(Repositories using HRNet as backbone)](https://github.com/HRNet)][[Codes|Tensorflow for fun](https://github.com/VXallset/deep-high-resolution-net.TensorFlow)][[Codes|Tensorflow HRNet-V2(unoffical)](https://github.com/AI-Chen/HRNet-V2)]

* **(CVPR2019)** Multi-Person Pose Estimation with Enhanced Channel-wise and Spatial Information [[arxiv link](https://arxiv.org/abs/1905.03466)]

* **(CVPR2019)** PoseFix: Model-Agnostic General Human Pose Refinement Network [[paper link](https://www.researchgate.net/publication/338506497_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network)]
 
* **(arxiv2019)** Rethinking on Multi-Stage Networks for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1901.00148v1)]

* ⭐**DarkPose(CVPR2020)** Distribution-Aware Coordinate Representation for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1910.06278)][[project link](https://ilovepose.github.io/coco/)][[Codes|PyTorch(offical)](https://github.com/ilovepose/DarkPose)]

* ⭐**UDP-Pose(CVPR2020)** The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1911.07524)][[Codes|](https://github.com/HuangJunJie2017/UDP-Pose)]

* **Graph-PCNN(arxiv 2020)** Graph-PCNN: Two Stage Human Pose Estimation with Graph Pose Refinement [[arxiv link](http://arxiv.org/abs/2007.10599)]

* **RSN-PRM(arxiv2020)** Learning Delicate Local Representations for Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/2003.04030v3)]

* **OPEC-Net(arxiv2020)** Peeking into occluded joints: A novel framework for crowd pose estimation [[arxiv link](https://arxiv.org/abs/2003.10506)]

* **(arxiv2020)(video based)** Self-supervised Keypoint Correspondences for Multi-Person Pose Estimation and Tracking in Videos [[arxiv link](https://arxiv.org/abs/2004.12652)]

* **OmniPose(arxiv2021)** OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/2103.10180)]


### ▶  Two-Stage [Bottom-Up] Multiple Person Pose Estimation

* **DeepCut(CVPR2016)** DeepCut: Joint Subset Partition and Labeling for Multi Person Pose Estimation [[arxiv link](https://arxiv.org/abs/1511.06645)]

* **DeeperCut(ECCV2016)** DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model [[arxiv link](https://arxiv.org/abs/1605.03170)]

* **ArtTrack(CVPR2017)** ArtTrack: Articulated Multi-Person Tracking in the Wild [[paper link](https://openaccess.thecvf.com/content_cvpr_2017/papers/Insafutdinov_ArtTrack_Articulated_Multi-Person_CVPR_2017_paper.pdf)]

* ⭐**OpenPose(CVPR2017)** Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields [[arxiv link](https://arxiv.org/abs/1611.08050)][[Codes|Caffe&Matlab(offical)](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)][[Codes|Caffe(offical only for testing)](https://github.com/CMU-Perceptual-Computing-Lab/openpose)][Codes|PyTorch(unoffical by tensorboy)](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)]

* ⭐**AssociativeEmbedding(NIPS2017)** Associative Embedding: End-to-end Learning for Joint Detection and Grouping [[arxiv link](https://arxiv.org/abs/1611.05424)][[Codes|PyTorch(offical)](https://github.com/princeton-vl/pose-ae-train)]

* **(ICCVW2017)** Multi-Person Pose Estimation for PoseTrack with Enhanced Part Affinity Fields [[paper link](https://posetrack.net/workshops/iccv2017/pdfs/ML_Lab.pdf)][[CSDN blog](https://blog.csdn.net/m0_37644085/article/details/82928933)]

* **(CVPRW2018)** Learning to Refine Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1804.07909)]

* ⭐**MultiPoseNet(ECCV2018)(multi-task)** MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network [[arxiv link](https://arxiv.org/abs/1807.04067)][[Codes|PyTorch(offical)](https://github.com/salihkaragoz/pose-residual-network-pytorch)]

* ⭐**PersonLab(ECCV2018)(multi-task)** PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model [[arxiv link](https://arxiv.org/abs/1803.08225)][[Codes|Keras&Tensorflow(unoffical by octiapp)](https://github.com/octiapp/KerasPersonLab)][[Codes|Tensorflow(unoffical)](https://github.com/scnuhealthy/Tensorflow_PersonLab)]

 * **DirectPose(arxiv2019)** DirectPose: Direct End-to-End Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/1911.07451v2)]

* ⭐**OpenPifPaf(CVPR2019)** PifPaf: Composite Fields for Human Pose Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.pdf)][[Codes|PyTorch(offical)](https://github.com/vita-epfl/openpifpaf)]

* ⭐**HigherHRNet(CVPR2020)** HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1908.10357)][[Codes|PyTorch(offical)](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)]

* ⭐**MDN3(CVPR2020)** Mixture Dense Regression for Object Detection and Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1912.00821)][[Codes|PyTorch(offical)](https://github.com/alivaramesh/MixtureDenseRegression)]

* **HGG(arxiv2020)** Differentiable Hierarchical Graph Grouping for Multi-person Pose Estimation [[arxiv link](https://arxiv.org/abs/2007.11864)]

* **SAHR(CVPR2021)(arxiv2021)** Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation [[arxiv link](https://arxiv.org/abs/2012.15175)][[Codes | official pytorch](https://github.com/greatlog/SWAHR-HumanPose)]

### ▶  Single-Stage Multiple Person Pose Estimation

* **SPM(ICCV2019)** Single-Stage Multi-Person Pose Machines [[arxiv link](https://arxiv.org/abs/1908.09220)][[Codes|PyTorch(offical not released)](https://github.com/NieXC/pytorch-spm)][[Codes|Tensorflow(unoffical)](https://github.com/murdockhou/Single-Stage-Multi-person-Pose-Machines)][[CSDN blog](https://blog.csdn.net/Murdock_C/article/details/100545377)]

* **CenterNet(arxiv2019)** Objects as Points [[arxiv link](https://arxiv.org/abs/1904.07850)]

* **KAPAO(arxiv2021)** Rethinking Keypoint Representations: Modeling Keypoints and Poses as Objects for Multi-Person Human Pose Estimation [[arxiv link](https://arxiv.org/abs/2111.08557)][[codes|(official pytorch using YOLOv5)](https://github.com/wmcnally/kapao)]



### ▶  3D Multiple Person Pose Estimation

* **mvpose(CVPR2019)(monocular multi-view)** Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views [[arxiv link](https://arxiv.org/abs/1901.04111)][[project link](https://zju3dv.github.io/mvpose/)][[Codes|Torch&Tensorflow(offical)](https://github.com/zju3dv/mvpose)]

* **EpipolarPose(CVPR2019)(monocular multi-view)** Self-Supervised Learning of 3D Human Pose using Multi-view Geometry [[arxiv link](https://arxiv.org/abs/1903.02330)][[project link](https://mkocabas.github.io/epipolarpose.html)][[Codes|PyTorch(offical)](https://github.com/mkocabas/EpipolarPose)]



### ▶  Special Multiple Person Pose Estimation

* **PoseTrack(CVPR2017)** PoseTrack: Joint Multi-Person Pose Estimation and Tracking [[arxiv link](https://arxiv.org/abs/1611.07727)][[Codes|Matlab&Caffe](https://github.com/iqbalu/PoseTrack-CVPR2017)]

* **Detect-and-Track(CVPR2018)** Detect-and-Track: Efficient Pose Estimation in Videos [[arxiv link](https://arxiv.org/abs/1712.09184)][[project link](https://rohitgirdhar.github.io/DetectAndTrack/)][[Codes|Detectron(offical)](https://github.com/facebookresearch/DetectAndTrack/)]

* **PoseFlow(BMVC2018)** Pose Flow: Efficient Online Pose Tracking [[arxiv link](https://arxiv.org/abs/1802.00977)][[Codes|AlphaPose(offical)](https://github.com/YuliangXiu/PoseFlow)]

* **DensePose(CVPR2018)** DensePose: Dense Human Pose Estimation In The Wild [[arxiv link](https://arxiv.org/abs/1802.00434)][[project link](http://densepose.org/)][[Codes|Caffe2(offical)](https://github.com/facebookresearch/Densepose)]

* **RF-Pose(CVPR2018)(radio frequency)** Through-Wall Human Pose Estimation Using Radio Signals [[paper link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Through-Wall_Human_Pose_CVPR_2018_paper.pdf)][[project link](http://rfpose.csail.mit.edu/)]

* **DoubleFusion(TPAMI2019)(3D single-view real-time depth-sensor)** DoubleFusion: Real-time Capture of Human Performances with Inner Body Shapes from a Single Depth Sensor [[arxiv link](https://arxiv.org/pdf/1804.06023.pdf)]


### ▶  Domain Adaptive Multiple Person Pose Estimation

* **Pose-ADHNN(ACMMM2021)** Alleviating Human-level Shift: A Robust Domain Adaptation Method for Multi-person Pose Estimation [[paper link](https://dl.acm.org/doi/abs/10.1145/3394171.3414040)][[Codes|PyTorch](https://github.com/Sophie-Xu/Pose-ADHNN)]


