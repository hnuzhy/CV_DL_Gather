# Contents

* **[1) Pubilc Datasets and Challenges](#1-Pubilc-Datasets-and-Challenges)**
  * **[⭐Slim and Simple](#Slim-and-Simple)**
  * **[⭐Mixed, Synthetic and Complicated](#Mixed-Synthetic-and-Complicated)**
* **[2) Pioneers and Experts](#2-Pioneers-and-Experts)**
* **[3) Blogs, Videos and Applications](#3-Blogs-Videos-and-Applications)**
* **[4) Papers and Sources Codes](#4-Papers-and-Sources-Codes)**
  * **[▶ Related Survey](#-Related-Survey)**
  * **[▶ Single Person Pose Estimation](#-Single-Person-Pose-Estimation)**
  * **[▶ Two-Stage [Top-Down] Multiple Person Pose Estimation](#-Two-Stage-Top-Down-Multiple-Person-Pose-Estimation)**
  * **[▶ Two-Stage [Bottom-Up] Multiple Person Pose Estimation](#-Two-Stage-Bottom-Up-Multiple-Person-Pose-Estimation)**
  * **[▶ Single-Stage Multiple Person Pose Estimation](#-Single-Stage-Multiple-Person-Pose-Estimation)**
  * **[▶ Simultaneous Multiple Person Pose Estimation and Instance Segmentation](#-Simultaneous-Multiple-Person-Pose-Estimation-and-Instance-Segmentation)**
  * **[▶ 3D Multiple Person Pose Estimation](#-3D-Multiple-Person-Pose-Estimation)**
  * **[▶ Special Multiple Person Pose Estimation](#-Special-Multiple-Person-Pose-Estimation)**
  * **[▶ Transfer Learning of Multiple Person Pose Estimation](#-Transfer-Learning-of-Multiple-Person-Pose-Estimation)**
    * **[※ Active Learning for Pose](#-active-learning-for-pose)**
    * **[※ Pose in Real Classroom](#-pose-in-real-classroom)**
    * **[※ Animal Pose Estimation](#-animal-pose-estimation)**
    * **[※ Hand Pose Estimation](#-hand-pose-estimation)**
    * **[※ Head Pose Estimation / Eye Gaze Estimation](#-head-pose-estimation--eye-gaze-estimation)**
    * **[※ 3D Human Pose Estimation](#-3d-human-pose-estimation)**
    * **[※ 2D Human Pose Estimation (Single and Multiple)](#-2d-human-pose-estimation-single-and-multiple)**
  * **[▶ Keypoints Meet Large Language Model](#-Keypoints-Meet-Large-Language-Model)**
  * **[▶ Keypoints for Human Motion Generation](#-Keypoints-for-Human-Motion-Generation)**

  
#  List for public implementation of various algorithms

## 1) Pubilc Datasets and Challenges

### Slim and Simple

* [LIP(Look Into Person)](http://www.sysu-hcp.net/lip/index.php)
* [Human3.6M (TPAMI2014) (3D single person)](http://vision.imar.ro/human3.6m/description.php)
* [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#) [[Annotations(Matlab-->Python)](https://github.com/bearpaw/pytorch-pose#installation)]
* [COCO - Common Objects in Context](https://cocodataset.org/)
* [AI Challenger (arxiv2017 & ICME2019)](https://github.com/AIChallenger/AI_Challenger_2017)[[paper link](https://arxiv.org/abs/1711.06475)]
* [MHP - Multi-Human Parsing (ACMMM2018)](https://lv-mhp.github.io/dataset)
* [DensePose-COCO Dataset (CVPR2018)](http://densepose.org/#dataset)
* [PoseTrack: Dataset and Benchmark](https://posetrack.net/) [[challenges links](https://posetrack.net/workshops/eccv2018/#challenges)][[paper link](https://openaccess.thecvf.com/content_cvpr_2018/html/Andriluka_PoseTrack_A_Benchmark_CVPR_2018_paper.html)][[github link](https://github.com/facebookresearch/DensePose/tree/main/PoseTrack)]

### Mixed, Synthetic and Complicated

* ⭐[OCHuman(Occluded Human) Dataset (CVPR2019)](https://arxiv.org/abs/1803.10683) [[github link](https://github.com/liruilong940607/OCHumanApi)]
* ⭐[CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark (CVPR2019)](https://github.com/Jeff-sjtu/CrowdPose) [[paper link](https://arxiv.org/abs/1812.00324)]
* ⭐[JTA(Joint Track Auto) - A synthetical dataset from GTA-V (ECCV2018)](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=25)[[paper link](https://openaccess.thecvf.com/content_ECCV_2018/papers/Matteo_Fabbri_Learning_to_Detect_ECCV_2018_paper.pdf)][[github link](https://github.com/fabbrimatteo/JTA-Dataset)][[JTA-Extension](https://github.com/thomasgolda/Human-Pose-Estimation-for-Real-World-Crowded-Scenarios)]
* [Mannequin RGB and IRS in-bed pose estimation dataset](https://github.com/ostadabbas/in-bed-pose-estimation)
* ⭐[CMU Panoptic Studio Dataset (3D single and multiple real person pose in the lab)](http://domedb.perception.cs.cmu.edu/) [[github link](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox)]
* [SURREAL dataset (CVPR2017) (3D single synthetic person pose in the indoor)](https://www.di.ens.fr/willow/research/surreal/)[[paper link](https://arxiv.org/abs/1701.01370)]
* [Drive&Act dataset (ICCV2019) (3D openpose single real person pose in the car with 5 views)](https://www.driveandact.com/)[[paper link](https://www.driveandact.com/publication/2019_iccv_drive_and_act/2019_iccv_drive_and_act.pdf)]
* ⭐[COCO-WholeBody (ECCV2020) (re-annotated based on keypoints in COCO dataset)](https://github.com/jin-s13/COCO-WholeBody)[[paper link](https://arxiv.org/abs/2007.11858)][[ZoomNAS(TPAMI2022)](https://ieeexplore.ieee.org/document/9852279)]
* [Halpe-FullBody (CVPR2020) (full body human pose estimation and human-object interaction detection dataset)](https://github.com/Fang-Haoshu/Halpe-FullBody)[[paper link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_PaStaNet_Toward_Human_Activity_Knowledge_Engine_CVPR_2020_paper.pdf)]
* [IKEA ASSEMBLY DATASET (WACV2021) (3D single and multiple real person pose in the lab with 3 views)](https://ikeaasm.github.io/)[[paper link](https://arxiv.org/abs/2007.00394)][[google drive](https://drive.google.com/drive/folders/1xkDp--QuUVxgl4oJjhCDb2FWNZTkYANq)]
* [Yoga-82: A New Dataset for Fine-grained Classification of Human Poses](https://sites.google.com/view/yoga-82/home)[[kaggle](https://www.kaggle.com/shrutisaxena/yoga-pose-image-classification-dataset)]
* [UAV-Human Dataset (CVPR2021) (not all appeared persons are annotated)](https://github.com/SUTDCV/UAV-Human)[[paper link](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_UAV-Human_A_Large_Benchmark_for_Human_Behavior_Understanding_With_Unmanned_CVPR_2021_paper.pdf)][[google drive](https://drive.google.com/drive/folders/1QeYXeM_pbWBSSmpRr_rKHurMpI2TxAKs)]
* [Mirrored-Human Dataset: Reconstructing 3D Human Pose by Watching Humans in the Mirror (CVPR2021 Oral)](https://zju3dv.github.io/Mirrored-Human/)[[paper link](https://arxiv.org/pdf/2104.00340.pdf)]
* ⭐[AGORA: A synthetic human pose and shape dataset (CVPR2021)](https://agora.is.tue.mpg.de/) [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Patel_AGORA_Avatars_in_Geography_Optimized_for_Regression_Analysis_CVPR_2021_paper.html)][[github link](https://github.com/pixelite1201/agora_evaluation)][[STAR (ECCV2020)](https://star.is.tue.mpg.de/)][[SMPL-X (CVPR2019)](https://smpl-x.is.tue.mpg.de/)][[FLAME (SIGGRAPH2017)](https://flame.is.tue.mpg.de/)][[SMPL (SIGGRAPH2015)](https://smpl.is.tue.mpg.de/)][[rankers webpage](https://paperswithcode.com/dataset/agora)]
* [InfiniteForm: Open Source Dataset for Human Pose Estimation (NIPSW2021)](https://pixelate.ai/InfiniteForm) [[paper link](https://arxiv.org/abs/2110.01330)][[github link](https://github.com/toinfinityai/infiniteform)]
* [Lower Body Rehabilitation Dataset and Model Optimization (ICME2021)](https://ieeexplore.ieee.org/abstract/document/9428432/) [`The first human keypoints detection dataset for physical therapy, in particular lower body rehabilitation`]
* ⭐[UrbanPose: A new benchmark for VRU pose estimation in urban traffic scenes (IEEE Intelligent Vehicles Symposium (IV) 2021)](http://urbanpose-dataset.com/info/Datasets/198) [[paper link](https://ieeexplore.ieee.org/abstract/document/9575469)]
* [HMR-Benchmarks: Benchmarking 3D Pose and Shape Estimation Beyond Algorithms (NIPS2022)](https://github.com/smplbody/hmr-benchmarks) [[paper link](https://openreview.net/pdf?id=rjBYortWdRV)]
* [SynPose: A Large-Scale and Densely Annotated Synthetic Dataset for Human Pose Estimation in Classroom (ICASSP2022)](https://yuzefang96.github.io/SynPose/) [[paper link](https://ieeexplore.ieee.org/abstract/document/9747453)][`Based on GTA-V, CycleGAN, ST-GCN and DEKR`]
* ⭐[JRDB-Pose: A Large-scale Dataset for Multi-Person Pose Estimation and Tracking (ICCV2019 & CVPR2021 & ECCV2022 & CVPR2023)](https://jrdb.erc.monash.edu/) [[paper link](https://openaccess.thecvf.com/content/CVPR2023/papers/Vendrow_JRDB-Pose_A_Large-Scale_Dataset_for_Multi-Person_Pose_Estimation_and_Tracking_CVPR_2023_paper.pdf)][[arxiv link](https://arxiv.org/abs/2210.11940)][[dataset details](https://jrdb.erc.monash.edu/dataset/)]
* ⭐[Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes (CVPR2023)](https://arxiv.org/abs/2303.02760) [[paper link](https://arxiv.org/abs/2303.02760)][[By IDEA-Research](https://github.com/IDEA-Research)]


## 2) Pioneers and Experts

[👍Alejandro Newell](https://www.alejandronewell.com/)
[👍Jia Deng](https://www.cs.princeton.edu/~jiadeng/)
[👍Zhe Cao](https://people.eecs.berkeley.edu/~zhecao/)
[👍Tomas Simon](http://www.cs.cmu.edu/~tsimon/)
[👍tensorboy](https://github.com/tensorboy)
[👍murdockhou](https://github.com/murdockhou)
[👍张兆翔](https://people.ucas.ac.cn/~zhangzhaoxiang)


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
* [(blog) A Comprehensive Guide to Human Pose Estimation](https://www.v7labs.com/blog/human-pose-estimation-guide)
* [(blog) (MMPose) 2D BODY KEYPOINT DATASETS](https://mmpose.readthedocs.io/en/latest/tasks/2d_body_keypoint.html)
* [(github) (coco-annotator) Web-based image segmentation tool for object detection, localization, and keypoints](https://github.com/jsbroks/coco-annotator/)


## 4) Papers and Sources Codes

### ▶ Related Survey

* **ComputingSurveys 2022** Recent Advances of Monocular 2D and 3D Human Pose Estimation: A Deep Learning Perspective [[paper link](https://dl.acm.org/doi/full/10.1145/3524497)][[arxiv link](https://arxiv.org/abs/2104.11536)][`JD` + `HIT`]

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

* **(ECCVW2016)** Multi-Person Pose Estimation with Local Joint-to-Person Associations [[arxiv link](https://arxiv.org/abs/1608.08526)]

* **(CVPR2017)** Towards Accurate Multi-person Pose Estimation in the Wild [[arxiv link](https://arxiv.org/abs/1701.01779)]

* **(ICCV2017)** A Coarse-Fine Network for Keypoint Localization [[paper link](https://openaccess.thecvf.com/content_iccv_2017/html/Huang_A_Coarse-Fine_Network_ICCV_2017_paper.html)]

* ⭐**AlphaPose/RMPE(ICCV2017)** RMPE: Regional Multi-person Pose Estimation [[arxiv link](https://arxiv.org/abs/1612.00137)][[project link](https://www.mvig.org/research/alphapose.html)][[Codes|PyTorch(offical)](https://github.com/MVIG-SJTU/AlphaPose)]

* ⭐**SimpleBaseline(ECCV2018)** Simple Baselines for Human Pose Estimation and Tracking [[arxiv link](https://arxiv.org/abs/1804.06208)][[Codes|PyTorch(offical)](https://github.com/Microsoft/human-pose-estimation.pytorch)][[Codes|PyTorch(flowtrack part)](https://github.com/simochen/flowtrack.pytorch)]

* ⭐**CPN(CVPR2018)** Cascaded Pyramid Network for Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/1711.07319)][[Codes|Tensorflow(offical)](https://github.com/chenyilun95/tf-cpn)][[Codes|Tensorflow(offical megvii)](https://github.com/megvii-detection/tf-cpn)][[zhihu blogs](https://zhuanlan.zhihu.com/p/37582402)]

* ⭐**HRNet(CVPR2019)** Deep High-Resolution Representation Learning for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1902.09212)][[Codes|PyTorch(offical)](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)][[Codes|(Repositories using HRNet as backbone)](https://github.com/HRNet)][[Codes|Tensorflow for fun](https://github.com/VXallset/deep-high-resolution-net.TensorFlow)][[Codes|Tensorflow HRNet-V2(unoffical)](https://github.com/AI-Chen/HRNet-V2)]

* ⭐**CrowdPose(CVPR2019)** CrowdPose: Efficient Crowded Scenes Pose Estimation and a New Benchmark [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_CrowdPose_Efficient_Crowded_Scenes_Pose_Estimation_and_a_New_Benchmark_CVPR_2019_paper.html)][[codes|(SJTU) official PyTorch](https://github.com/Jeff-sjtu/CrowdPose)]

* **(CVPR2019)** Multi-Person Pose Estimation with Enhanced Channel-wise and Spatial Information [[arxiv link](https://arxiv.org/abs/1905.03466)]

* **(CVPR2019)** PoseFix: Model-Agnostic General Human Pose Refinement Network [[paper link](https://www.researchgate.net/publication/338506497_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network)]
 
* **(arxiv2019)** Rethinking on Multi-Stage Networks for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1901.00148v1)]

* ⭐**DarkPose(CVPR2020)** Distribution-Aware Coordinate Representation for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1910.06278)][[project link](https://ilovepose.github.io/coco/)][[Codes|PyTorch(offical)](https://github.com/ilovepose/DarkPose)]

* ⭐**UDP-Pose(CVPR2020)** The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1911.07524)][[Codes|](https://github.com/HuangJunJie2017/UDP-Pose)][`A model-agnostic approach`, `Plug-and-Play`]

* **Graph-PCNN(arxiv 2020)** Graph-PCNN: Two Stage Human Pose Estimation with Graph Pose Refinement [[arxiv link](http://arxiv.org/abs/2007.10599)]

* **RSN-PRM(arxiv2020)** Learning Delicate Local Representations for Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/2003.04030v3)]

* **OPEC-Net(arxiv2020)(ECCV2020)** Peeking into occluded joints: A novel framework for crowd pose estimation [[arxiv link](https://arxiv.org/abs/2003.10506)][for `Crowded Human Pose Estimation`]

* **(arxiv2020)(video based)** Self-supervised Keypoint Correspondences for Multi-Person Pose Estimation and Tracking in Videos [[arxiv link](https://arxiv.org/abs/2004.12652)]

* ⭐**PoseNAS(ACMMM2020)** Pose-native Network Architecture Search for Multi-person Human Pose Estimation [[paper link](https://dl.acm.org/doi/abs/10.1145/3394171.3413842)][[codes|official PyTorch](https://github.com/for-code0216/PoseNAS)][`Network Architecture Search (NAS) based two-stage MPPE`]

* **CCM(IJCV2021)** Towards High Performance Human Keypoint Detection [[paper link](https://link.springer.com/article/10.1007/s11263-021-01482-8)][[codes|official (not released)](https://github.com/chaimi2013/CCM)]

* **OmniPose(arxiv2021)** OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/2103.10180)]

* **RLE(ICCV2021)** Human Pose Regression With Residual Log-Likelihood Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Human_Pose_Regression_With_Residual_Log-Likelihood_Estimation_ICCV_2021_paper.html)][[code|official](https://github.com/Jeff-sjtu/res-loglikelihood-regression)]

* **MIPNet(ICCV2021)** Multi-Instance Pose Networks: Rethinking Top-Down Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Khirodkar_Multi-Instance_Pose_Networks_Rethinking_Top-Down_Pose_Estimation_ICCV_2021_paper.html)][[project link](https://rawalkhirodkar.github.io/mipnet/)][[codes|official demo](https://github.com/rawalkhirodkar/MIPNet)]

* ⭐**TransPose(ICCV2021)** TransPose: Keypoint Localization via Transformer [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_TransPose_Keypoint_Localization_via_Transformer_ICCV_2021_paper.html)][[codes|official PyTroch](https://github.com/yangsenius/TransPose)][`Transformer based two-stage MPPE (light-weight)`]

* ⭐**TokenPose(ICCV2021)** TokenPose: Learning Keypoint Tokens for Human Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Li_TokenPose_Learning_Keypoint_Tokens_for_Human_Pose_Estimation_ICCV_2021_paper.html)][[codes|official PyTroch](https://github.com/leeyegy/TokenPose)][`Token representation based two-stage MPPE (light-weight)`]

* ⭐**Lite-HRNet(CVPR2021)** Lite-HRNet: A Lightweight High-Resolution Network [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Yu_Lite-HRNet_A_Lightweight_High-Resolution_Network_CVPR_2021_paper.html)][[codes|official PyTorch](https://github.com/HRNet/Lite-HRNet)][`This work is done by the original group of HRNet`]

* **HRFormer(NIPS2021)** HRFormer: High-Resolution Transformer for Dense Prediction [[paper link](https://arxiv.org/abs/2110.09408)][[code|official](https://github.com/HRNet/HRFormer)][`multi-task`, `2D Human Pose Estimation`, `Semantic Segmentation`]

* ⭐**LitePose(CVPR2022)** Lite Pose: Efficient Architecture Design for 2D Human Pose Estimation [[paper link](https://tinyml.mit.edu/wp-content/uploads/2022/04/CVPR2022__Lite_Pose.pdf)][[project link](https://tinyml.mit.edu/publications/)][[codes|official PyTorch](https://github.com/mit-han-lab/litepose)][`Model quantization and compression on Qualcomm Snapdragon`]

* **CID(CVPR2022)** Contextual Instance Decoupling for Robust Multi-Person Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Contextual_Instance_Decoupling_for_Robust_Multi-Person_Pose_Estimation_CVPR_2022_paper.html)][[codes|official](https://github.com/kennethwdk/CID)][[(TPAMI2023) Contextual Instance Decoupling for Instance-Level Human Analysis](https://ieeexplore.ieee.org/abstract/document/10040902)][[First Author: Dongkai Wang](https://kennethwdk.github.io/)]

* **Poseur(ECCV2022)** Poseur: Direct Human Pose Regression with Transformers [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20068-7_5)][[code|official](https://github.com/aim-uofa/Poseur)][`RLE-based`, `DETR-based top-down framework`]

* **SCIO(ECCV2022)** Self-Constrained Inference Optimization on Structural Groups for Human Pose Estimation [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_42)][[arxiv link](https://arxiv.org/abs/2207.02425)][`Test-Time Adaptation`, the same author of [`SCAI`](https://arxiv.org/abs/2303.11180)]

* **Swin-Pose(arxiv2022)(MIPR2022)** Swin-Pose: Swin Transformer Based Human Pose Estimation [[paper link](https://arxiv.org/abs/2201.07384)][`Swin Transformer`]

* ⭐**ViTPose(NIPS2022)** ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation [[paper link](https://openreview.net/forum?id=6H2pBoPtm0s)][[arxiv link](https://arxiv.org/abs/2204.12484)][[code|official](https://github.com/ViTAE-Transformer/ViTPose)][[ViTPose+: Vision Transformer Foundation Model for Generic Body Pose Estimation (arxiv2022.12)](https://arxiv.org/abs/2212.04246) ([TPAMI2023](https://ieeexplore.ieee.org/abstract/document/1030864))][`Tao Dacheng`, `plain vision transformer`]

* **PCT(CVPR2023)** Human Pose as Compositional Tokens [[arxiv link](https://arxiv.org/abs/2303.11638)][[code|official](https://github.com/Gengzigang/PCT)][`Transformer-based`]

* **DistilPose(CVPR2023)** DistilPose: Tokenized Pose Regression With Heatmap Distillation [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Ye_DistilPose_Tokenized_Pose_Regression_With_Heatmap_Distillation_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.02455)][[code|offical](https://github.com/yshMars/DistilPose)][`Xia Men University`, `Regression-based`, Transformer]

* **BCIR(Bias Compensated Integral Regression)(TPAMI2023)** Bias-Compensated Integral Regression for Human Pose Estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/10093110)][[arxiv link](https://arxiv.org/abs/2301.10431)][`A model-agnostic approach`, `Plug-and-Play`]

* **ICON(AAAI2023)** Inter-image Contrastive Consistency for Multi-Person Pose Estimation [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/25410)][`Xixia Xu`, `No code`, sever as a play-in-plug]


### ▶ Two-Stage [Bottom-Up] Multiple Person Pose Estimation

* **DeepCut(CVPR2016)** DeepCut: Joint Subset Partition and Labeling for Multi Person Pose Estimation [[arxiv link](https://arxiv.org/abs/1511.06645)]

* ⭐**DeeperCut(ECCV2016)** DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model [[arxiv link](http://arxiv.org/abs/1605.03170)][[project link](http://pose.mpi-inf.mpg.de/#)][[Codes|Tensorflow(offical)](https://github.com/eldar/pose-tensorflow)]

* **ArtTrack(CVPR2017)** ArtTrack: Articulated Multi-Person Tracking in the Wild [[paper link](https://openaccess.thecvf.com/content_cvpr_2017/papers/Insafutdinov_ArtTrack_Articulated_Multi-Person_CVPR_2017_paper.pdf)]

* ⭐**OpenPose(CVPR2017)** Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields [[arxiv link](https://arxiv.org/abs/1611.08050)][[Codes|Caffe&Matlab(offical)](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)][[Codes|Caffe(offical only for testing)](https://github.com/CMU-Perceptual-Computing-Lab/openpose)][Codes|PyTorch(unoffical by tensorboy)](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)]

* ⭐**AssociativeEmbedding(NIPS2017)** Associative Embedding: End-to-end Learning for Joint Detection and Grouping [[arxiv link](https://arxiv.org/abs/1611.05424)][[Codes|PyTorch(offical)](https://github.com/princeton-vl/pose-ae-train)]

* **(ICCVW2017)** Multi-Person Pose Estimation for PoseTrack with Enhanced Part Affinity Fields [[paper link](https://posetrack.net/workshops/iccv2017/pdfs/ML_Lab.pdf)][[CSDN blog](https://blog.csdn.net/m0_37644085/article/details/82928933)]

* **PPN(ECCV2018)** Pose Partition Networks for Multi-Person Pose Estimation [[paper link](https://openaccess.thecvf.com/content_ECCV_2018/html/Xuecheng_Nie_Pose_Partition_Networks_ECCV_2018_paper.html)][`To partition all keypoint detections using dense regressions from keypoint candidates to centroids of persons`, `similar to SPM`]

* **(CVPRW2018)** Learning to Refine Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1804.07909)]

* ⭐**MultiPoseNet(ECCV2018)(multi-task)** MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network [[arxiv link](https://arxiv.org/abs/1807.04067)][[Codes|PyTorch(offical)](https://github.com/salihkaragoz/pose-residual-network-pytorch)]

* **OpenPoseTrain(ICCV2019)** Single-Network Whole-Body Pose Estimation [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Hidalgo_Single-Network_Whole-Body_Pose_Estimation_ICCV_2019_paper.html)][[codes|official](https://github.com/CMU-Perceptual-Computing-Lab/openpose_train)][`simultaneous localization of body, face, hands, and feet keypoints`]

* ⭐**OpenPifPaf(CVPR2019)** PifPaf: Composite Fields for Human Pose Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.pdf)][[Codes|PyTorch(offical)](https://github.com/vita-epfl/openpifpaf)]

* ⭐**HigherHRNet(CVPR2020)** HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1908.10357)][[Codes|PyTorch(offical)](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)]

* ⭐**MDN3(CVPR2020)** Mixture Dense Regression for Object Detection and Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1912.00821)][[Codes|PyTorch(offical)](https://github.com/alivaramesh/MixtureDenseRegression)]

* **HGG(arxiv2020)(ECCV2020)** Differentiable Hierarchical Graph Grouping for Multi-person Pose Estimation [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58571-6_42)][[arxiv link](https://arxiv.org/abs/2007.11864)]

* ⭐**EfficientHRNet(arxiv2020)** EfficientHRNet: Efficient Scaling for Lightweight High-Resolution Multi-Person Pose Estimation [[paper link](https://arxiv.org/abs/2007.08090)]

* **SimplePose(AAAI2020)** Simple pose: Rethinking and improving a bottom-up approach for multi-person pose estimation [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/6797)][[codes|official PyTorch](https://github.com/hellojialee/Improved-Body-Parts)][`An improved OpenPose based on Stacked Hourglass and proposed Body Parts`]

* **DGCN(AAAI2020)** DGCN: Dynamic Graph Convolutional Network for Efficient Multi-Person Pose Estimation [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/6867)][`Graph based two-stage MPPE`]

* ⭐**CenterGroup(ICCV2021)** The Center of Attention: Center-Keypoint Grouping via Attention for Multi-Person Pose Estimation [[paper link](https://arxiv.org/abs/2110.05132)][[codes|official PyTorch based on mmpose and HigherHRNet](https://github.com/dvl-tum/center-group)]

* ⭐**SWAHR(CVPR2021)** Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation [[arxiv link](https://arxiv.org/abs/2012.15175)][[Codes|official pytorch based on HigherHRNet](https://github.com/greatlog/SWAHR-HumanPose)]

* ⭐**DEKR(CVPR2021)** Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression [[arxiv link](https://arxiv.org/abs/2104.02300)][[Codes|official pytorch](https://github.com/HRNet/DEKR)]

* **PINet(NIPS2021)** Robust Pose Estimation in Crowded Scenes with Direct Pose-Level Inference [[paper link](https://proceedings.neurips.cc/paper/2021/hash/31857b449c407203749ae32dd0e7d64a-Abstract.html)][[codes|official PyTorch](https://github.com/kennethwdk/PINet)][[First Author: Dongkai Wang](https://kennethwdk.github.io/)][For `Crowded Scenes`, Following `HigherHRNet` and `DEKR`]

* **DAC(arxiv2022)** Bottom-Up 2D Pose Estimation via Dual Anatomical Centers for Small-Scale Persons [[arxiv link](https://arxiv.org/abs/2208.11975)][`Dual Anatomical Centers (Head + Body)`]

* **CoupledEmbedding(ECCV2022)** Regularizing Vector Embedding in Bottom-Up Human Pose Estimation [[paper link](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660105.pdf)][[codes|official PyTorch](https://github.com/CR320/CoupledEmbedding)]

* **DEKRv2(ICIP2022)** DEKRv2: More Fast or Accurate than DEKR [[paper link](https://ieeexplore.ieee.org/abstract/document/9897550)][[codes|official PyTorch](https://github.com/chaowentao/DEKRv2)]

* **HrHRNet-CF(CVPR2023)** A Characteristic Function-Based Method for Bottom-Up Human Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2023/papers/Qu_A_Characteristic_Function-Based_Method_for_Bottom-Up_Human_Pose_Estimation_CVPR_2023_paper.pdf)]

* **BUCTD(ICCV2023)** Rethinking Pose Estimation in Crowds: Overcoming the Detection Information Bottleneck and Ambiguity [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Rethinking_Pose_Estimation_in_Crowds_Overcoming_the_Detection_Information_Bottleneck_ICCV_2023_paper.html)][[project link](https://amathislab.github.io/BUCTD/)][[arxiv link](https://arxiv.org/abs/2306.07879)][[code|official](https://github.com/amathislab/BUCTD)][`EPFL`]


### ▶ Single-Stage Multiple Person Pose Estimation

* **DirectPose(arxiv2019)** DirectPose: Direct End-to-End Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/1911.07451v2)][`DirectPose proposes to directly regress the instance-level keypoints by considering the keypoints as a special bounding-box with more than two corners.`]

* **SPM(ICCV2019)** Single-Stage Multi-Person Pose Machines [[arxiv link](https://arxiv.org/abs/1908.09220)][[Codes|PyTorch(offical not released)](https://github.com/NieXC/pytorch-spm)][[Codes|Tensorflow(unoffical)](https://github.com/murdockhou/Single-Stage-Multi-person-Pose-Machines)][[CSDN blog](https://blog.csdn.net/Murdock_C/article/details/100545377)]

* **CenterNet(arxiv2019)** Objects as Points [[arxiv link](https://arxiv.org/abs/1904.07850)]

* **Point-Set Anchors(ECCV2020)** Point-Set Anchors for Object Detection, Instance Segmentation and Pose Estimation [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_31)]

* **POET(arxiv2021)** End-to-End Trainable Multi-Instance Pose Estimation with Transformers [[arxiv link](https://arxiv.org/abs/2103.12115)][`DETR-based`, `regression`]

* **TFPose(arxiv2021)** TFPose: Direct Human Pose Estimation with Transformers [[arxiv link](https://arxiv.org/abs/2103.15320)][[project link](https://github.com/aim-uofa/AdelaiDet/)][`It adopts Detection Transformers to estimate the cropped single-person images as a query-based regression task`][`end2end top-down`]

* **InsPose(ACMMM2021)** InsPose: Instance-Aware Networks for Single-Stage Multi-Person Pose Estimation [[paper link](https://dl.acm.org/doi/abs/10.1145/3474085.3475447)][[code|official](https://github.com/hikvision-research/opera/tree/main/configs/inspose)][`It designs
instance-aware dynamic networks to adaptively adjust part of the network parameters for each instance`]

* **DeepDarts(CVPRW2021)** DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts using a Single Camera [[paper link](https://openaccess.thecvf.com/content/CVPR2021W/CVSports/papers/McNally_DeepDarts_Modeling_Keypoints_as_Objects_for_Automatic_Scorekeeping_in_Darts_CVPRW_2021_paper.pdf)]

* ⭐**FCPose(CVPR2021)** FCPose: Fully Convolutional Multi-Person Pose Estimation With Dynamic Instance-Aware Convolutions [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Mao_FCPose_Fully_Convolutional_Multi-Person_Pose_Estimation_With_Dynamic_Instance-Aware_Convolutions_CVPR_2021_paper.html)][[codes|official](https://github.com/aim-uofa/AdelaiDet/)]

* **PRTR(CVPR2021)** Pose Recognition With Cascade Transformers [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Pose_Recognition_With_Cascade_Transformers_CVPR_2021_paper.html)][[codes|official](https://github.com/mlpc-ucsd/PRTR)][`transformer-based`, `high input resolution and stacked attention modules`, `high complexity and require huge memory during the training phase`][`end2end top-down`]

* ⭐**KAPAO(ECCV2022)** Rethinking Keypoint Representations: Modeling Keypoints and Poses as Objects for Multi-Person Human Pose Estimation [[arxiv link](https://arxiv.org/abs/2111.08557)][[codes|(official pytorch using YOLOv5)](https://github.com/wmcnally/kapao)]

* **YOLO-Pose(CVPRW2022)** YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss [[paper link](https://arxiv.org/abs/2204.06806)][[codes|official edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)][[codes|official edgeai-yolov5](https://github.com/TexasInstruments/edgeai-yolov5)]

* ⭐**AdaptivePose(AAAI2022)** AdaptivePose: Human Parts as Adaptive Points [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/20185)][[codes|official PyTorch](https://github.com/buptxyb666/AdaptivePose)]

* ⭐**AdaptivePose++(TCSVT2022)** AdaptivePose++: A Powerful Single-Stage Network for Multi-Person Pose Regression [[paper link](https://arxiv.org/abs/2210.04014)][[codes|official PyTorch](https://github.com/buptxyb666/AdaptivePose)]

* **LOGO-CAP(CVPR2022)** Learning Local-Global Contextual Adaptation for Multi-Person Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Xue_Learning_Local-Global_Contextual_Adaptation_for_Multi-Person_Pose_Estimation_CVPR_2022_paper.html)][[codes|official PyTorch](https://github.com/cherubicXN/logocap)]

* **PETR(CVPR2022)** End-to-End Multi-Person Pose Estimation With Transformers [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Shi_End-to-End_Multi-Person_Pose_Estimation_With_Transformers_CVPR_2022_paper.html)][[codes|official PyTorch](https://github.com/hikvision-research/opera)][`transformer-based`, `high input resolution and stacked attention modules`, `high complexity and require huge memory during the training phase`][`fully end2end`]

* **QueryPose(NIPS2022)** QueryPose: Sparse Multi-Person Pose Regression via Spatial-Aware Part-Level Query [[openreview link](https://openreview.net/forum?id=tbId-oAOZo)][[arxiv link](https://arxiv.org/abs/2212.07855)][[code|official](https://github.com/buptxyb666/QueryPose)][`fully end2end`]

* ⭐**ED-Pose(ICLR2023)** Explicit Box Detection Unifies End-to-End Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/2302.01593)][[openreview link](https://openreview.net/forum?id=s4WVupnJjmX)][[code|official](https://github.com/IDEA-Research/ED-Pose)][`IDEA-Research`][`fully end2end`]

* **PolarPose(TIP2023)** PolarPose: Single-Stage Multi-Person Pose Estimation in Polar Coordinates [[paper link](https://ieeexplore.ieee.org/abstract/document/10034548)]

* 👍**GroupPose(ICCV2023)** Group Pose: A Simple Baseline for End-to-End Multi-person Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Group_Pose_A_Simple_Baseline_for_End-to-End_Multi-Person_Pose_Estimation_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2308.07313)][[code|official Paddle](https://github.com/Michel-liu/GroupPose-Paddle)][[code|official PyTorch](https://github.com/Michel-liu/GroupPose)]


### ▶ Simultaneous Multiple Person Pose Estimation and Instance Segmentation

* ⭐**Mask R-CNN(ICCV2017)(multi-task)** Mask R-CNN [[paper link](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)]

* ⭐**PersonLab(ECCV2018)(multi-task)** PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model [[arxiv link](https://arxiv.org/abs/1803.08225)][[Codes|Keras&Tensorflow(unoffical by octiapp)](https://github.com/octiapp/KerasPersonLab)][[Codes|Tensorflow(unoffical)](https://github.com/scnuhealthy/Tensorflow_PersonLab)]

* **ACPNet(ICME2019)** ACPNet: Anchor-Center Based Person Network for Human Pose Estimation and Instance Segmentation [[paper link](https://ieeexplore.ieee.org/abstract/document/8784943)][`based on Mask R-CNN`]

* **Pose2Seg(CVPR2019)** Pose2Seg: Detection Free Human Instance Segmentation [[paper link](https://arxiv.org/abs/1803.10683)][[codes|official](https://github.com/liruilong940607/OCHumanApi)]

* **PointSetNet(ECCV2020)** Point-Set Anchors for Object Detection, Instance Segmentation and Pose Estimation [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_31)][`Not a multi-task end-to-end network`, `The proposed Point-Set Anchors can be applied to object detection, instance segmentation and human pose estimation tasks separately`]

* **MG-HumanParsing(CVPR2021)** Differentiable Multi-Granularity Human Representation Learning for Instance-Aware Human Semantic Parsing [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Differentiable_Multi-Granularity_Human_Representation_Learning_for_Instance-Aware_Human_Semantic_Parsing_CVPR_2021_paper.html)][[code|official](https://github.com/tfzhou/MG-HumanParsing)]

* **Multitask-CenterNet(ICCVW2021)** MultiTask-CenterNet (MCN): Efficient and Diverse Multitask Learning Using an Anchor Free Approach [[paper link](https://openaccess.thecvf.com/content/ICCV2021W/ERCVAD/html/Heuer_MultiTask-CenterNet_MCN_Efficient_and_Diverse_Multitask_Learning_Using_an_Anchor_ICCVW_2021_paper.html)][`based on the CenterNet`]

* **MDSP(IVS2022 Oral)** Multitask Network for Joint Object Detection, Semantic Segmentation and Human Pose Estimation in Vehicle Occupancy Monitoring [[paper link](https://arxiv.org/abs/2205.01515)]

* **PosePlusSeg(AAAI2022)** Joint Human Pose Estimation and Instance Segmentation with PosePlusSeg [[paper link](https://www.aaai.org/AAAI22Papers/AAAI-6681.AhmadN.pdf)][[codes|official tensorflow](https://github.com/RaiseLab/PosePlusSeg)][`similarly with the PersonLab`, `Niaz Ahmad`, `suspected of plagiarism`]

* **MultiPoseSeg(ICPR2022)** MultiPoseSeg: Feedback Knowledge Transfer for Multi-Person Pose Estimation and Instance Segmentation [[paper link](https://ieeexplore.ieee.org/abstract/document/9956648)][[code|official](https://github.com/RaiseLab/MultiPoseSeg)][`similarly with the PersonLab`, `Niaz Ahmad`, `suspected of plagiarism`]

* **HCQNet(Human-Centric Query)(arxiv2023.03)** Object-Centric Multi-Task Learning for Human Instances [[paper link](https://arxiv.org/abs/2303.06800)][based on the `Mask2Former (CVPR2022) (Masked-attention mask transformer for universal image segmentation)`]


### ▶ 3D Multiple Person Pose Estimation

* **mvpose(CVPR2019)(monocular multi-view)** Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views [[arxiv link](https://arxiv.org/abs/1901.04111)][[project link](https://zju3dv.github.io/mvpose/)][[Codes|Torch&Tensorflow(offical)](https://github.com/zju3dv/mvpose)]

* **EpipolarPose(CVPR2019)(monocular multi-view)** Self-Supervised Learning of 3D Human Pose using Multi-view Geometry [[arxiv link](https://arxiv.org/abs/1903.02330)][[project link](https://mkocabas.github.io/epipolarpose.html)][[Codes|PyTorch(offical)](https://github.com/mkocabas/EpipolarPose)]

* **SMAP(ECCV2020)** SMAP: Single-Shot Multi-person Absolute 3D Pose Estimation [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_33)][[project link](https://zju3dv.github.io/SMAP/)][[codes|official PyTorch](https://github.com/zju3dv/SMAP)]

* **(multi-views)(ICCV2021)** Shape-aware Multi-Person Pose Estimation from Multi-View Images [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Dong_Shape-Aware_Multi-Person_Pose_Estimation_From_Multi-View_Images_ICCV_2021_paper.html)][[project link](https://ait.ethz.ch/projects/2021/multi-human-pose/)][[codes|official](https://github.com/zj-dong/Multi-Person-Pose-Estimation)]

* **MVP(NIPS2021)** Direct Multi-view Multi-person 3D Pose Estimation [[paper link](https://proceedings.neurips.cc/paper/2021/hash/6da9003b743b65f4c0ccd295cc484e57-Abstract.html)][[codes|official PyTorch](https://github.com/sail-sg/mvp)]

* **InverseKinematics(ECCV2022)** Multi-Person 3D Pose and Shape Estimation via Inverse Kinematics and Refinement [[paper link](https://arxiv.org/abs/2210.13529)][datasets `3DPW`, `MuCo-3DHP` and `AGORA`][`transformer`]

* **HUPOR(ECCV2022)** Explicit Occlusion Reasoning for Multi-person 3D Human Pose Estimation [[arxiv link](https://arxiv.org/abs/2208.00090)][[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_29)][[code|official](https://github.com/qihao067/HUPOR)] 

* **POTR3D(ICCV2023)** Towards Robust and Smooth 3D Multi-Person Pose Estimation from Monocular Videos in the Wild [[paper link]()][[arxiv link](https://arxiv.org/abs/2309.08644)][`Seoul National University`]


### ▶ Special Multiple Person Pose Estimation

* **PoseTrack(CVPR2017)** PoseTrack: Joint Multi-Person Pose Estimation and Tracking [[arxiv link](https://arxiv.org/abs/1611.07727)][[Codes|Matlab&Caffe](https://github.com/iqbalu/PoseTrack-CVPR2017)]

* **Detect-and-Track(CVPR2018)** Detect-and-Track: Efficient Pose Estimation in Videos [[arxiv link](https://arxiv.org/abs/1712.09184)][[project link](https://rohitgirdhar.github.io/DetectAndTrack/)][[Codes|Detectron(offical)](https://github.com/facebookresearch/DetectAndTrack/)][[codes|official](https://github.com/wmcnally/deep-darts)]

* **PoseFlow(BMVC2018)** Pose Flow: Efficient Online Pose Tracking [[arxiv link](https://arxiv.org/abs/1802.00977)][[Codes|AlphaPose(offical)](https://github.com/YuliangXiu/PoseFlow)]

* **DensePose(CVPR2018)** DensePose: Dense Human Pose Estimation In The Wild [[arxiv link](https://arxiv.org/abs/1802.00434)][[project link](http://densepose.org/)][[Codes|Caffe2(offical)](https://github.com/facebookresearch/Densepose)]

* **RF-Pose(CVPR2018)(radio frequency)** Through-Wall Human Pose Estimation Using Radio Signals [[paper link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Through-Wall_Human_Pose_CVPR_2018_paper.pdf)][[project link](http://rfpose.csail.mit.edu/)]

* 👍**LIP_JPPNet(TPAMI2019)** Look into Person: Joint Body Parsing & Pose Estimation Network and a New Benchmark [[paper link](https://ieeexplore.ieee.org/abstract/document/8327922)][[Lab Homepage](http://www.sysu-hcp.net/)][[code|official](https://github.com/Engineering-Course/LIP_JPPNet)][`Joint Body Parsing & Pose Estimation`]

* **DoubleFusion(TPAMI2019)(3D single-view real-time depth-sensor)** DoubleFusion: Real-time Capture of Human Performances with Inner Body Shapes from a Single Depth Sensor [[arxiv link](https://arxiv.org/pdf/1804.06023.pdf)]

* **Keypoint-Communities(ICCV2019)** Keypoint Communities [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Zauss_Keypoint_Communities_ICCV_2021_paper.html)][`Model all keypoints belonging to a human or an object (the pose) as a graph`]

* **BlazePose (CVPRW2020)** BlazePose: On-device Real-time Body Pose tracking [[paper link](https://arxiv.org/abs/2006.10204)][[project link](https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html)]

* **ODKD(arxiv2021)** Orderly Dual-Teacher Knowledge Distillation for Lightweight Human Pose Estimation [[paper link](https://arxiv.org/abs/2104.10414)][`Knowledge Distillation of MPPE based on HRNet`]

* **DDP(3DV2021)** Direct Dense Pose Estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/9665828)][`Dense human pose estimation`]

* **MEVADA(ICCV2021)** Single View Physical Distance Estimation using Human Pose [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Fei_Single_View_Physical_Distance_Estimation_Using_Human_Pose_ICCV_2021_paper.html)][[project link](https://feixh.github.io/projects/physical_distance/)]

* **Unipose+(TPAMI2022)** UniPose+: A Unified Framework for 2D and 3D Human Pose Estimation in Images and Videos [[paper link](https://ieeexplore.ieee.org/abstract/document/9599531)][[author given link](https://par.nsf.gov/biblio/10322977)]

* 👍**HTCorrM(Human Task Correlation Machine)(TPAMI2022)** On the Correlation among Edge, Pose and Parsing [[paper link](https://ieeexplore.ieee.org/abstract/document/9527074/)][[pdf link](http://www.jdl.link/doc/2011/20220112_On%20the%20Correlation%20among%20Edge,%20Pose%20and%20Parsing.pdf)][`Multi-tasks Learning`]

* **PoseTrack21(CVPR2022)** PoseTrack21: A Dataset for Person Search, Multi-Object Tracking and Multi-Person Pose Tracking [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Doring_PoseTrack21_A_Dataset_for_Person_Search_Multi-Object_Tracking_and_Multi-Person_CVPR_2022_paper.html)][[codes|official](https://github.com/andoer/PoseTrack21)][`jointly person search, multi-object tracking and multi-person pose tracking`]

* **PoseTrans(ECCV2022)** PoseTrans: A Simple Yet Effective Pose Transformation Augmentation for Human Pose Estimation [[paper link](https://arxiv.org/abs/2208.07755)]

* ⭐**DeciWatch(ECCV2022)** DeciWatch: A Simple Baseline for 10x Efficient 2D and 3D Pose Estimation [[paper link](https://arxiv.org/abs/2203.08713)][[code|official](https://github.com/cure-lab/DeciWatch)][[project link](https://ailingzeng.site/deciwatch)][`Video based human pose estimation`]

* **PPT(ECCV2022)** PPT: Token-Pruned Pose Transformer for Monocular and Multi-view Human Pose Estimation [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_25)][[code|official](https://github.com/HowieMa/PPT)]
 
* **QuickPose(SIGGRAPH2022)** QuickPose: Real-time Multi-view Multi-person Pose Estimation in Crowded Scenes [[paper link](https://dl.acm.org/doi/abs/10.1145/3528233.3530746)][`ZJU`]
 
* **TDMI-ST(CVPR2023)** Mutual Information-Based Temporal Difference Learning for Human Pose Estimation in Video [[paper link](https://arxiv.org/abs/2303.08475)][`PoseTrack2017, PoseTrack2018, and PoseTrack21`, `video-based HPE`]

* **MG-HumanParsing(TPAMI2023)** Differentiable Multi-Granularity Human Parsing [[paper link](https://ieeexplore.ieee.org/abstract/document/10032235)][[code|official](https://github.com/tfzhou/MG-HumanParsing)][`Human Parsing`]

* **Obj2Seq(NIPS2022)** Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks [[openreview link](https://openreview.net/forum?id=cRNl08YWRKq)][[arxiv link](https://arxiv.org/abs/2209.13948)][[code|official](https://github.com/CASIA-IVA-Lab/Obj2Seq)][`ViT-based`, `Multi-task model`]

* 👍**AutoLink(NIPS2022)** AutoLink: Self-supervised Learning of Human Skeletons and Object Outlines by Linking Keypoints [[arxiv link](https://arxiv.org/abs/2205.10636)][[openreview link](https://openreview.net/forum?id=mXP-qQcYCBN)][[project link](https://xingzhehe.github.io/autolink/)]


### ▶ Transfer Learning of Multiple Person Pose Estimation
**Domain Adaptive / Unsupervised / Self-Supervised / Semi-Supervised / Weakly-Supervised / Generalizable**

#### ※ Active Learning for Pose

* **VL4Pose(BMVC2022)** VL4Pose: Active Learning Through Out-Of-Distribution Detection For Pose Estimation [[arxiv link](https://arxiv.org/abs/2210.06028)][[code|official](https://github.com/meghshukla/ActiveLearningForHumanPose)][with tasks of single `human pose` and `hand pose`]

#### ※ Pose in Real Classroom

* 👍**SynPose(ICASSP2022)** Synpose: A Large-Scale and Densely Annotated Synthetic Dataset for Human Pose Estimation in Classroom [[paper link](https://ieeexplore.ieee.org/abstract/document/9747453)][[project link](https://yuzefang96.github.io/SynPose/)][`Based on GTA-V, CycleGAN, ST-GCN and DEKR`]

* 👍**CC-PoseNet(ICASSP2023)** CC-PoseNet: Towards Human Pose Estimation in Crowded Classrooms [[paper link](https://ieeexplore.ieee.org/abstract/document/10095734)]

#### ※ Animal Pose Estimation

* **WS-CDA(ICCV2019)** Cross-Domain Adaptation for Animal Pose Estimation [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Cao_Cross-Domain_Adaptation_for_Animal_Pose_Estimation_ICCV_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1908.05806)][[project link](https://sites.google.com/view/animal-pose/)][[code|official](https://github.com/noahcao/animal-pose-dataset)][`Animal Pose Dataset`, `Leverages human pose data and a partially annotated animal pose dataset to perform semi-supervised domain adaptation`]

* 👍**CC-SSL(CVPR2020)** Learning From Synthetic Animals [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Mu_Learning_From_Synthetic_Animals_CVPR_2020_paper.html)][[arxiv link](https://arxiv.org/abs/1912.08265)][[code|official](https://github.com/JitengMu/Learning-from-Synthetic-Animals)][`Animal Pose`][`It proposed invariance and equivariance consistency learning with respect to transformations as well as temporal consistency learning with a video`; `It employs a single end-to-end trained network`]

* 👍**MDAM, UDA-Animal-Pose(CVPR2021)** From Synthetic to Real: Unsupervised Domain Adaptation for Animal Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Li_From_Synthetic_to_Real_Unsupervised_Domain_Adaptation_for_Animal_Pose_CVPR_2021_paper.html)][[codes|PyTorch](https://github.com/chaneyddtt/UDA-Animal-Pose)][`Animal Pose`][`ResNet + Hourglass`][`It proposed a refinement module and a self-feedback loop to obtain reliable pseudo labels`; `It addresses the teacher-student paradigm alongside a novel pseudo-label strategy`]

* ⚡**DeepLabCut (Nature Methods 2022)** Multi-animal pose estimation, identification and tracking with DeepLabCut [[paper link](https://www.nature.com/articles/s41592-022-01443-0)]

* ⚡**Social LEAP Estimates Animal Poses (SLEAP) (Nature Methods 2022)** SLEAP: A deep learning system for multi-animal pose tracking [[paper link](https://www.nature.com/articles/s41592-022-01426-1)]

* **SemiMultiPose(arxiv2022)** SemiMultiPose: A Semi-supervised Multi-animal Pose Estimation Framework [[paper link](https://arxiv.org/abs/2204.07072)][`Semi-Supervised Keypoint Localization`]

* **AnimalKingdom (CVPR2022)** Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Ng_Animal_Kingdom_A_Large_and_Diverse_Dataset_for_Animal_Behavior_CVPR_2022_paper.html)][[project link](https://sutdcv.github.io/Animal-Kingdom)][[arxiv link](https://arxiv.org/abs/2204.08129)][[code|official](https://github.com/sutdcv/Animal-Kingdom)]

* ⭐**ScarceNet(CVPR2023)** ScarceNet: Animal Pose Estimation With Scarce Annotations [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Li_ScarceNet_Animal_Pose_Estimation_With_Scarce_Annotations_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.15023)][[code|official](https://github.com/chaneyddtt/ScarceNet)][`Animal Pose`, `Semi-Supervised Keypoint Localization`, based on `HRNet`][`small-loss trick for reliability check` + `agreement check to identify reusable samples` + `student-teacher network (Mean Teacher) to enforce a consistency constraint`]

* **AnimalTrack (IJCV2023)** AnimalTrack: A Benchmark for Multi-Animal Tracking in the Wild [[arxiv link](https://arxiv.org/abs/2205.00158)][[project link](https://hengfan2010.github.io/projects/AnimalTrack/)][[download page](https://hengfan2010.github.io/projects/AnimalTrack/download.html)][`Animal dataset`]

* **LoTE-Animal (ICCV2023)** LoTE-Animal: A Long Time-span Dataset for Endangered Animal Behavior Understanding [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_LoTE-Animal_A_Long_Time-span_Dataset_for_Endangered_Animal_Behavior_Understanding_ICCV_2023_paper.html)][[project link](https://lote-animal.github.io/)][Animal dataset]

* **Animal3D (ICCV2023)** Animal3D: A Comprehensive Dataset of 3D Animal Pose and Shape [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Animal3D_A_Comprehensive_Dataset_of_3D_Animal_Pose_and_Shape_ICCV_2023_paper.html)][[arxiv link](http://arxiv.org/abs/2308.11737)][[project link](https://xujiacong.github.io/Animal3D/)][based on the `SMAL` model, Animal dataset]

* ⚡**Social Behavior Atlas (SBeA) (Nature Machine Intelligence 2024)** Multi-animal 3D social pose estimation, identification and behaviour embedding with a few-shot learning framework [[paper link](https://www.nature.com/articles/s42256-023-00776-5)]


#### ※ Hand Pose Estimation

* **(ECCV2018)** Weakly-supervised 3D Hand Pose Estimation from Monocular RGB Images [[paper link](https://openaccess.thecvf.com/content_ECCV_2018/html/Yujun_Cai_Weakly-supervised_3D_Hand_ECCV_2018_paper.html)][`No code is available`, `Nanyang Technological University`, `a weakly-supervised method with the aid of depth images`, `3D Hand Pose Estimation`, `Keypoints`]

* **SO-HandNet(ICCV2019)** SO-HandNet: Self-Organizing Network for 3D Hand Pose Estimation With Semi-Supervised Learning [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_SO-HandNet_Self-Organizing_Network_for_3D_Hand_Pose_Estimation_With_Semi-Supervised_ICCV_2019_paper.html)][`No code is available`, `Wuhan University`, `3D Hand Pose Estimation`, `Keypoints`, based on [`SO-Net`](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.html) and 3D point clouds]

* **weak_da_hands(CVPR2020)** Weakly-Supervised Domain Adaptation via GAN and Mesh Model for Estimating 3D Hand Poses Interacting Objects [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Baek_Weakly-Supervised_Domain_Adaptation_via_GAN_and_Mesh_Model_for_Estimating_CVPR_2020_paper.html)][[code|official (not available)](https://github.com/bsrvision/weak_da_hands)]

* **SemiHand(ICCV2021)** SemiHand: Semi-Supervised Hand Pose Estimation With Consistency [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_SemiHand_Semi-Supervised_Hand_Pose_Estimation_With_Consistency_ICCV_2021_paper.html)][`No code is available`, `semi-supervised hand pose estimation`]

* **MarsDA(TCSVT2022)** Multibranch Adversarial Regression for Domain Adaptative Hand Pose Estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/9732951)][based on `RegDA`, `hand datasets (RHD→H3D)`, `It applies a teacher-student approach to edit RegDA`]

* 👍**C-GAC(ECCV2022)** Domain Adaptive Hand Keypoint and Pixel Localization in the Wild [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_5)][[arxiv link](https://arxiv.org/abs/2203.08344)][[project link](https://tkhkaeio.github.io/projects/22-hand-ps-da/)][based on `Stacked Hourglass`， `all compared methods are reproduced by the author`, `no code is available`]

* **DM-HPE(CVPR2023)** Cross-Domain 3D Hand Pose Estimation With Dual Modalities [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Cross-Domain_3D_Hand_Pose_Estimation_With_Dual_Modalities_CVPR_2023_paper.html)][`No code is available`, `cross-domain semi-supervised hand pose estimation`, `Dual Modalities`]


#### ※ Head Pose Estimation / Eye Gaze Estimation
belonging to the `Domain Adaptive Regression (DGA)` or `Semi-Supervised Rotation Regression` problem

* **PADACO(ICCV2019)** Deep Head Pose Estimation Using Synthetic Images and Partial Adversarial Domain Adaption for Continuous Label Spaces [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Kuhnke_Deep_Head_Pose_Estimation_Using_Synthetic_Images_and_Partial_Adversarial_ICCV_2019_paper.html)][[code|official](http://www.tnt.uni-hannover.de/project/headposeplus)][An adversarial training approach based on [`domain adversarial neural networks`](http://proceedings.mlr.press/v37/ganin15.html) is used to force the extraction of domain-invariant features]

* 👍**Gaze360(ICCV2019)** Gaze360: Physically Unconstrained Gaze Estimation in the Wild [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Kellnhofer_Gaze360_Physically_Unconstrained_Gaze_Estimation_in_the_Wild_ICCV_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1910.10088)][[project link](http://gaze360.csail.mit.edu/)][`dataset Gaze360`, `Domain Adaptive Gaze Estimation`]

* **few_shot_gaze(ICCV2019 oral)** Few-Shot Adaptive Gaze Estimation [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1905.01941)][[code|official](https://github.com/NVlabs/few_shot_gaze)][`Domain Adaptive Gaze Estimation`]

* **DeepDAR(SpringerBook2020)** Deep Domain Adaptation for Regression [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-31764-5_4)][[pdf link](https://link.springer.com/content/pdf/10.1007/978-3-030-31764-5.pdf#page=99)][`Domain Adaptive Regression (DGA)` theory, `Age Estimation` and `Head Pose Estimation`][book title [`Development and Analysis of Deep Learning Architectures`](https://link.springer.com/content/pdf/10.1007/978-3-030-31764-5.pdf)]

* **DAGEN(ACCV2020)** Domain Adaptation Gaze Estimation by Embedding with Prediction Consistency [[paper link](https://openaccess.thecvf.com/content/ACCV2020/html/Guo_Domain_Adaptation_Gaze_Estimation_by_Embedding_with_Prediction_Consistency_ACCV_2020_paper.html)][[arxiv link](http://arxiv.org/abs/2011.07526)][`Eye Gaze Estimation`]

* **(FG2021)** Relative Pose Consistency for Semi-Supervised Head Pose Estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/9666992/)][[pdf link](https://www.tnt.uni-hannover.de/papers/data/1544/RCRwFG2021.pdf)][`Semi-Supervised`]

* **PnP-GA(ICCV2021)** Generalizing Gaze Estimation With Outlier-Guided Collaborative Adaptation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Generalizing_Gaze_Estimation_With_Outlier-Guided_Collaborative_Adaptation_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2107.13780)][[code|official](https://github.com/DreamtaleCore/PnP-GA)][`Domain Adaptive Gaze Estimation`]

* 👍**RSD(ICML2021)** Representation Subspace Distance for Domain Adaptation Regression [[paper link](http://proceedings.mlr.press/v139/chen21u.html)][[code|official](https://github.com/thuml/Domain-Adaptation-Regression)][`Domain Adaptive Regression (DGA)` theory, `Mingsheng Long`, datasets [dSprites](https://github.com/deepmind/dsprites-dataset)(a standard 2D synthetic dataset for deep representation learning) and [MPI3D](https://github.com/rr-learning/disentanglement_dataset)(a simulation-to-real dataset of 3D objects)]

* **DINO-INIT & DINO-TRAIN(NIPS2022)** Distribution-Informed Neural Networks for Domain Adaptation Regression [[paper link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/410bbba8388369d8bb5875544d1d4428-Abstract-Conference.html)][`Domain Adaptive Regression (DGA)` theory]

* **SynGaze(CVPRW2022)** Learning-by-Novel-View-Synthesis for Full-Face Appearance-Based 3D Gaze Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022W/GAZE/html/Qin_Learning-by-Novel-View-Synthesis_for_Full-Face_Appearance-Based_3D_Gaze_Estimation_CVPRW_2022_paper.html)][[arxiv link](http://arxiv.org/abs/2201.07927)][`The University of Tokyo`, `Eye Gaze Estimation`, No code]

* **RUDA(CVPR2022)** Generalizing Gaze Estimation With Rotation Consistency [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Bao_Generalizing_Gaze_Estimation_With_Rotation_Consistency_CVPR_2022_paper.html)][`Eye Gaze Estimation`, No code]

* **CRGA(CVPR2022)** Contrastive Regression for Domain Adaptation on Gaze Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Contrastive_Regression_for_Domain_Adaptation_on_Gaze_Estimation_CVPR_2022_paper.html)][`SJTU`, `Eye Gaze Estimation`, No code]

* **(TBIOM2023)** Domain Adaptation for Head Pose Estimation Using Relative Pose Consistency [[paper link](https://ieeexplore.ieee.org/abstract/document/10021684)]

* **AdaptiveGaze(arxiv2023.05)** Domain-Adaptive Full-Face Gaze Estimation via Novel-View-Synthesis and Feature Disentanglement [[arxiv link](https://arxiv.org/abs/2305.16140)][[code|official](https://github.com/utvision/AdaptiveGaze)][`The University of Tokyo`, `Eye Gaze Estimation`]

* 👍**DARE-GRAM(CVPR2023)** DARE-GRAM: Unsupervised Domain Adaptation Regression by Aligning Inverse Gram Matrices [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Nejjar_DARE-GRAM_Unsupervised_Domain_Adaptation_Regression_by_Aligning_Inverse_Gram_Matrices_CVPR_2023_paper.html)][[code|official](https://github.com/ismailnejjar/DARE-GRAM)][HPE domain transfer test for Male --> Female on `BIWI` dataset]

* **(AAAI2023)** Learning a Generalized Gaze Estimator from Gaze-Consistent Feature [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/25406)]

* 👍**UnReGA(CVPR2023)** Source-Free Adaptive Gaze Estimation by Uncertainty Reduction [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Cai_Source-Free_Adaptive_Gaze_Estimation_by_Uncertainty_Reduction_CVPR_2023_paper.html)][[paperswithcode link](https://paperswithcode.com/paper/source-free-adaptive-gaze-estimation-by)][[code|official (not released)](https://github.com/caixin1998/UnReGA)]

* **PnP-GA+(TPAMI2023)** PnP-GA+: Plug-and-Play Domain Adaptation for Gaze Estimation using Model Variants [[paper link](https://ieeexplore.ieee.org/abstract/document/10378867)][`Domain Adaptive Gaze Estimation`, extended based on `PnP-GA(ICCV2021)`]

#### ※ 3D Human Pose Estimation

* **pose-hg-3d(ICCV2017)** Towards 3D Human Pose Estimation in the Wild: A Weakly-Supervised Approach [[paper link](https://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Towards_3D_Human_ICCV_2017_paper.html)][[code|official](https://github.com/xingyizhou/pose-hg-3d)][`3D keypoints detection`, `weakly-supervised domain adaptation with a 3D geometric constraint-induced loss`]

* **3DKeypoints-DA(ECCV2018)** Unsupervised Domain Adaptation for 3D Keypoint Estimation via View Consistency [[paper link](https://openaccess.thecvf.com/content_ECCV_2018/html/Xingyi_Zhou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.html)][[arxiv link](https://arxiv.org/abs/1712.05765v2)][[code|official](https://github.com/xingyizhou/3DKeypoints-DA)][`It utilizes view-consistency to regularize predictions from unlabeled target domain in 3D keypoints detection, but depth scans and images from different views are required on the target domain`]

* **(ACMMM2019)** Unsupervised Domain Adaptation for 3D Human Pose Estimation [[paper link](http://zju-capg.org/unsupervised_domain_adaptation/main.pdf)][`3D keypoints detection`]

* **(CVPR2020)** Weakly-Supervised 3D Human Pose Learning via Multi-View Images in the Wild [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Iqbal_Weakly-Supervised_3D_Human_Pose_Learning_via_Multi-View_Images_in_the_CVPR_2020_paper.html)][[arxiv link](https://arxiv.org/abs/2003.07581)][`NVIDIA`, `It focuses on unlabelled multi-view images`, `Self-supervised learning for 3D human pose estimation`]

* **AdaptPose(CVPR2022)** AdaptPose: Cross-Dataset Adaptation for 3D Human Pose Estimation by Learnable Motion Generation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Gholami_AdaptPose_Cross-Dataset_Adaptation_for_3D_Human_Pose_Estimation_by_Learnable_CVPR_2022_paper.html)][`3D keypoints detection`]

* **FewShot3DKP(CVPR2023)** Few-Shot Geometry-Aware Keypoint Localization [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/He_Few-Shot_Geometry-Aware_Keypoint_Localization_CVPR_2023_paper.html)][[project link](https://xingzhehe.github.io/FewShot3DKP/)][`Few-Shot Learning`, `3D Keypoint Localization`, `human faces, eyes, animals, cars, and never-before-seen mouth interior (teeth) localization tasks`]

* **ACSM-Plus(CVPR2023)** Learning Articulated Shape With Keypoint Pseudo-Labels From Web Images [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Stathopoulos_Learning_Articulated_Shape_With_Keypoint_Pseudo-Labels_From_Web_Images_CVPR_2023_paper.html)][`2D Keypoints for downstream application`, `3D Reconstruction / Shape Recovery from 2D images`]

* **PoseDA (ICCV2023)** Global Adaptation meets Local Generalization: Unsupervised Domain Adaptation for 3D Human Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Chai_Global_Adaptation_Meets_Local_Generalization_Unsupervised_Domain_Adaptation_for_3D_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.16456)][[code|official](https://github.com/rese1f/PoseDA)][`ZJU`]

* 👍**3D-Pose-Transfer (ICCV2023)** Weakly-supervised 3D Pose Transfer with Keypoints [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Weakly-supervised_3D_Pose_Transfer_with_Keypoints_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2307.13459)][[project link](https://jinnan-chen.github.io/ws3dpt/)][[code|official](https://github.com/jinnan-chen/3D-Pose-Transfer)][`National University of Singapore`]


#### ※ 2D Human Pose Estimation (Single and Multiple)

* **DataDistill, Pseudo-Labeling, PL(CVPR2018)** Data Distillation: Towards Omni-Supervised Learning [[paper link](https://openaccess.thecvf.com/content_cvpr_2018/html/Radosavovic_Data_Distillation_Towards_CVPR_2018_paper.html)][[arxiv link](https://arxiv.org/abs/1712.04440)][`Omni-Supervised Learning`, `a special regime of semi-supervised learning`, with tasks `human keypoint detection` and `general object detection`]

* **Pose_DomainAdaption(ACMMM2020)** Alleviating Human-level Shift: A Robust Domain Adaptation Method for Multi-person Pose Estimation [[paper link](https://dl.acm.org/doi/abs/10.1145/3394171.3414040)][[Codes|PyTorch (not available)](https://github.com/Sophie-Xu/Pose_DomainAdaption)][[(TMM2022 extended journal version) Structure-enriched Topology Learning for Cross-domain Multi-person Pose estimation](https://ieeexplore.ieee.org/abstract/document/9894704)]

* ⭐**SSKL(ICLR2021)** Semi-supervised Keypoint Localization [[openreview link](https://openreview.net/forum?id=yFJ67zTeI2)][[arxiv link](https://arxiv.org/abs/2101.07988)][[code|official](https://github.com/olgamoskvyak/tf_equivariance_loss)][[author Olga Moskvyak's homepage](https://olgamoskvyak.github.io/)][`single hand datasets`, `single person datasets`, `Semi-Supervised Keypoint Localization`]

* ⭐**Semi_Human_Pose(ICCV2021)** An Empirical Study of the Collapsing Problem in Semi-Supervised 2D Human Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Xie_An_Empirical_Study_of_the_Collapsing_Problem_in_Semi-Supervised_2D_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2011.12498)][[codes|official PyTorch](https://github.com/xierc/Semi_Human_Pose)][`Semi-Supervised 2D Human Pose Estimation`]

* 👍❤**RegDA(CVPR2021)** Regressive Domain Adaptation for Unsupervised Keypoint Detection [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Jiang_Regressive_Domain_Adaptation_for_Unsupervised_Keypoint_Detection_CVPR_2021_paper.html)][[project library](https://github.com/thuml/Transfer-Learning-Library)][[code|official](https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/keypoint_detection)][`hand datasets (RHD→H3D)`, `human datasets (SURREAL→Human3.6M, SURREAL→LSP)`][`ResNet101 + Simple Baseline`][based on the DA classification method [disparity discrepancy (DD)](https://proceedings.mlr.press/v97/zhang19i.html) (ICML2019, authors including Mingsheng Long and Michael Jordan)][`It utilizes one shared feature extractor and two separate regressors`; `It made changes in DD for human and hand pose estimation tasks, which measures discrepancy by estimating false predictions on the target domain`]

* 👍**HPE-AdaptOR(arxiv2021.08)(Medical Image Analysis2022)** Unsupervised domain adaptation for clinician pose estimation and instance segmentation in the operating room [[paper link](https://www.sciencedirect.com/science/article/pii/S1361841522001724)][[arxiv link](https://arxiv.org/abs/2108.11801)][[code|official](https://github.com/CAMMA-public/HPE-AdaptOR)]

* **TransPar(TIP2022)** Learning Transferable Parameters for Unsupervised Domain Adaptation [[paper link](https://ieeexplore.ieee.org/abstract/document/9807644)][[arxiv link](https://arxiv.org/abs/2108.06129)][evaluation on tasks `image classification` and `regression tasks (keypoint detection)`][`hand datasets (RHD→H3D)`, `It emphasizes transferable parameters using a similar structure as RegDA which has one shared feature extractor and two separate regressors`]

* 👍❤**UniFrame, UDA_PoseEstimation(ECCV2022)** A Unified Framework for Domain Adaptive Pose Estimation [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19827-4_35)][[arxiv link](https://arxiv.org/abs/2204.00172)][[code|official](https://github.com/VisionLearningGroup/UDA_PoseEstimation)][`hand datasets (RHD→H3D)`, `human datasets (SURREAL→Human3.6M, SURREAL→LSP)`, `animal datasets (SynAnimal→TigDog, SynAnimal→AnimalPose)`, based on `RegDA`][`ResNet101 + Simple Baseline`][[AdaIN (ICCV2017)](https://github.com/xunhuang1995/AdaIN-style) `for image style transfer` + `Mean Teacher for student model updating`; `It modifies the classic Mean-Teacher model by combining it with style transfer AdaIN`]

* ⭐**iart-semi-pose(ACMMM2022)** Semi-supervised Human Pose Estimation in Art-historical Images [[arxiv link](https://arxiv.org/abs/2207.02976)][[code|official](https://github.com/TIBHannover/iart-semi-pose)][`Germany`, `Semi-Supervised 2D Human Pose Estimation`]

* ⭐**PLACL(ICLR2022)** Pseudo-Labeled Auto-Curriculum Learning for Semi-Supervised Keypoint Localization [[openreview link](https://openreview.net/forum?id=6Q52pZ-Th7N)][[arxiv link](https://arxiv.org/abs/2201.08613)][[author Sheng Jin's homepage](https://jin-s13.github.io/)][`Semi-Supervised Keypoint Localization`, backbone `HRNet-w32`, `Curriculum Learning` + `Reinforcement Learning`, slightly better than `SSKL(ICLR2020)`][largely based on [`(Curriculum-Labeling, AAAI2021) Curriculum Labeling: Revisiting Pseudo-Labeling for Semi-Supervised Learning`](https://arxiv.org/abs/2001.06001)]

* **ADHNN(AAAI2022)** Adaptive Hypergraph Neural Network for Multi-person Pose Estimation [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/20201)][[Codes|PyTorch (not available)](https://github.com/Sophie-Xu/Pose-ADHNN)]

* **(WACV2022)** Transfer Learning for Pose Estimation of Illustrated Characters [[paper link](https://openaccess.thecvf.com/content/WACV2022/html/Chen_Transfer_Learning_for_Pose_Estimation_of_Illustrated_Characters_WACV_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2108.01819)][[codes|official PyTorch](https://github.com/ShuhongChen/bizarre-pose-estimator)]

* **CD_HPE(ICASSP2022)** Towards Accurate Cross-Domain in-Bed Human Pose Estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/9747183)][[arxiv link](https://arxiv.org/abs/2110.03578)][[code|official](https://github.com/MohamedAfham/CD_HPE)]

* **EdgeTrans4Mark(ECCV2022)** One-Shot Medical Landmark Localization by Edge-Guided Transform and Noisy Landmark Refinement [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19803-8_28)][[arxiv link](https://arxiv.org/abs/2208.00453)][[code|official](https://github.com/GoldExcalibur/EdgeTrans4Mark)][`PKU`, `Landmark Localization`, `Medical Image`]

* ⭐**SSPCM(CVPR2023)** Semi-Supervised 2D Human Pose Estimation Driven by Position Inconsistency Pseudo Label Correction Module [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Semi-Supervised_2D_Human_Pose_Estimation_Driven_by_Position_Inconsistency_Pseudo_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.04346)][[code|official](https://github.com/hlz0606/SSPCM)][`Semi-Supervised 2D Human Pose Estimation`]

* **SCAI(self-correctable and adaptable inference)(CVPR2023)** Self-Correctable and Adaptable Inference for Generalizable Human Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2023/papers/Kan_Self-Correctable_and_Adaptable_Inference_for_Generalizable_Human_Pose_Estimation_CVPR_2023_paper.pdf)][[arxiv link](https://arxiv.org/abs/2303.11180)][`Domain Generalization`][`It works as a play-in-plug for top-down human pose estimation methods like SimpleBaseline and HRNet`, the same author of [`SCIO`](https://arxiv.org/abs/2207.02425)]

* **Full-DG(full-view data generation)(TNNLS2023)** Overcoming Data Deficiency for Multi-Person Pose Estimation [[paper link](https://ieeexplore.ieee.org/abstract/document/10122653)][Full-DG can help improve pose estimators’ `robustness` and `generalizability`]

* **MAPS(arxiv2023.02)** MAPS: A Noise-Robust Progressive Learning Approach for Source-Free Domain Adaptive Keypoint Detection [[arxiv link](https://arxiv.org/abs/2302.04589)][[code|official](https://github.com/YuheD/MAPS)][`hand datasets (RHD→H3D)`, `human datasets (SURREAL→LSP)`, `animal datasets (SynAnimal→TigDog, SynAnimal→AnimalPose)`, based on `RegDA` and `UniFrame`]

* **ImSty(Implicit Stylization)(ICLRW2023)** Implicit Stylization for Domain Adaptation [[openreview link](https://openreview.net/forum?id=fkFFh4fAbH)][[pdf link](https://openreview.net/pdf?id=fkFFh4fAbH)][[workshop homepage](https://openreview.net/group?id=ICLR.cc/2023/Workshop/DG)]

* ⭐**SF-DAPE(ICCV2023)** Source-free Domain Adaptive Human Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Peng_Source-free_Domain_Adaptive_Human_Pose_Estimation_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2308.03202)][[code|official](https://github.com/davidpengucf/SFDAHPE)][`Source-free Domain Adaptation`, `hand datasets (RHD→H3D, RHD→FreiHand)`, `human datasets (SURREAL→Human3.6M, SURREAL→LSP)`]

* **POST(ICCV2023)** Prior-guided Source-free Domain Adaptation for Human Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Raychaudhuri_Prior-guided_Source-free_Domain_Adaptation_for_Human_Pose_Estimation_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2308.13954)][`Source-free Domain Adaptation`, `Self-training`, `human datasets (SURREAL→Human3.6M, SURREAL→LSP)`]

* **Pseudo-Heatmaps(arxiv2023.10)** Denoising and Selecting Pseudo-Heatmaps for Semi-Supervised Human Pose Estimation [[arxiv link](https://arxiv.org/abs/2310.00099)][based on the `DualPose (ICCV2021)`, do not compare with `SSPCM(CVPR2023)`]

* **MDSs(arxiv2023.10)(under review in ICLR2024)** Modeling the Uncertainty with Maximum Discrepant Students for Semi-supervised 2D Pose Estimation [[arxiv link](https://arxiv.org/abs/2311.01770)][[code|official](https://github.com/Qi2019KB/MDSs/tree/master)][based on the `DualPose (ICCV2021)`, do not compare with `SSPCM(CVPR2023)`]


### ▶ Keypoints Meet Large Language Model
**Large Language Model / Large Vision Model / Vision-Language Model for Human / Animals / Anything**

* 👍**CLAMP(CVPR2023)** CLAMP: Prompt-Based Contrastive Learning for Connecting Language and Animal Pose [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_CLAMP_Prompt-Based_Contrastive_Learning_for_Connecting_Language_and_Animal_Pose_CVPR_2023_paper.html)][[arxiv link](http://arxiv.org/abs/2206.11752)][[code|official](https://github.com/xuzhang1199/CLAMP)][`CLIP`, `Tao Dacheng`, trained and tested on dataset [`AP-10K`](https://github.com/AlexTheBad/AP-10K), also see [`APT-36K`](https://github.com/pandorgan/APT-36K)]

* **PoseFix(ICCV2023)** PoseFix: Correcting 3D Human Poses with Natural Language [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Delmas_PoseFix_Correcting_3D_Human_Poses_with_Natural_Language_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2309.08480)][[code|official](https://europe.naverlabs.com/research/computer-vision/posefix/)]

* **UniAP(arxiv2023.08)** UniAP: Towards Universal Animal Perception in Vision via Few-shot Learning [[arxiv link](https://arxiv.org/abs/2308.09953)][`CLIP`, `ZJU`, `Few-shot Learning`, `various perception tasks including pose estimation, segmentation, and classification tasks`]

* **KDSM(arxiv2023.10)** Language-driven Open-Vocabulary Keypoint Detection for Animal Body and Face [[arxiv link](https://arxiv.org/abs/2310.05056)][`CLIP`, `XJU + Shanghai AI Lab`, `Open-Vocabulary Keypoint Detection`]

* **UniPose(arxiv2023.10)(under review in ICLR2024)** UniPose: Detecting Any Keypoints [[openreview link](https://openreview.net/forum?id=v2J205zwlu)][[arxiv link](https://arxiv.org/abs/2310.08530)][[project link](https://yangjie-cv.github.io/UniPose/)][[code|official](https://github.com/IDEA-Research/UniPose)][`IDEA-Research`, `using visual or textual prompts`]


### ▶ Keypoints for Human Motion Generation
**Motion Synthesis / Motion Diffusion Model**

* **MoFusion(CVPR2023)** MoFusion: A Framework for Denoising-Diffusion-based Motion Synthesis [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Dabral_Mofusion_A_Framework_for_Denoising-Diffusion-Based_Motion_Synthesis_CVPR_2023_paper.html)][[arxiv link](http://arxiv.org/abs/2212.04495)][[project link](https://vcai.mpi-inf.mpg.de/projects/MoFusion/)][code is not avaliable][`MPII`]

* **GMD(ICCV2023)** Guided Motion Diffusion for Controllable Human Motion Synthesis [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Karunratanakul_Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2305.12577)][[project link](https://korrawe.github.io/gmd-project/)][[code|official](https://github.com/korrawe/guided-motion-diffusion)][`ETH`]

* **PhysDiff(ICCV2023 Oral)** PhysDiff: Physics-Guided Human Motion Diffusion Model [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_PhysDiff_Physics-Guided_Human_Motion_Diffusion_Model_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2212.02500)][[project link](https://nvlabs.github.io/PhysDiff/)][code is not avaliable][`NVIDIA`]

* **InterDiff(ICCV2023)** InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Xu_InterDiff_Generating_3D_Human-Object_Interactions_with_Physics-Informed_Diffusion_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2308.16905)][[project link](https://sirui-xu.github.io/InterDiff/)][[code|official](https://github.com/Sirui-Xu/InterDiff)][`University of Illinois at Urbana-Champaign`, `Human-Object Interactions`]

* **OmniControl(arxiv2023.10)** OmniControl: Control Any Joint at Any Time for Human Motion Generation [[arxiv link](https://arxiv.org/abs/2310.08580)][[project link](https://neu-vi.github.io/omnicontrol/)][[code|official](https://github.com/neu-vi/omnicontrol)][`Northeastern University + Google Research`]

