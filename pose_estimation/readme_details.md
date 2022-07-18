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
* ⭐[COCO-WholeBody (ECCV2020) (re-annotated based on keypoints in COCO dataset)](https://github.com/jin-s13/COCO-WholeBody)[[paper link](https://arxiv.org/abs/2007.11858)]
* [Halpe-FullBody (CVPR2020) (full body human pose estimation and human-object interaction detection dataset)](https://github.com/Fang-Haoshu/Halpe-FullBody)[[paper link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_PaStaNet_Toward_Human_Activity_Knowledge_Engine_CVPR_2020_paper.pdf)]
* [IKEA ASSEMBLY DATASET (WACV2021) (3D single and multiple real person pose in the lab with 3 views)](https://ikeaasm.github.io/)[[paper link](https://arxiv.org/abs/2007.00394)][[google drive](https://drive.google.com/drive/folders/1xkDp--QuUVxgl4oJjhCDb2FWNZTkYANq)]
* [Yoga-82: A New Dataset for Fine-grained Classification of Human Poses](https://sites.google.com/view/yoga-82/home)[[kaggle](https://www.kaggle.com/shrutisaxena/yoga-pose-image-classification-dataset)]
* [UAV-Human Dataset (CVPR2021) (not all appeared persons are annotated)](https://github.com/SUTDCV/UAV-Human)[[paper link](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_UAV-Human_A_Large_Benchmark_for_Human_Behavior_Understanding_With_Unmanned_CVPR_2021_paper.pdf)][[google drive](https://drive.google.com/drive/folders/1QeYXeM_pbWBSSmpRr_rKHurMpI2TxAKs)]
* [Mirrored-Human Dataset: Reconstructing 3D Human Pose by Watching Humans in the Mirror (CVPR2021 Oral)](https://zju3dv.github.io/Mirrored-Human/)[[paper link](https://arxiv.org/pdf/2104.00340.pdf)]
* ⭐[AGORA: A synthetic human pose and shape dataset (CVPR2021)](https://agora.is.tue.mpg.de/) [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Patel_AGORA_Avatars_in_Geography_Optimized_for_Regression_Analysis_CVPR_2021_paper.html)][[github link](https://github.com/pixelite1201/agora_evaluation)][[SMPL-X (CVPR2019)](https://smpl-x.is.tue.mpg.de/)][[SMPL (SIGGRAPH2015)](https://smpl.is.tue.mpg.de/)][[STAR (ECCV2020)](https://star.is.tue.mpg.de/)]
* [InfiniteForm: Open Source Dataset for Human Pose Estimation (NIPSW2021)](https://pixelate.ai/InfiniteForm) [[paper link](https://arxiv.org/abs/2110.01330)][[github link](https://github.com/toinfinityai/infiniteform)]
* ⭐[UrbanPose: A new benchmark for VRU pose estimation in urban traffic scenes (IEEE Intelligent Vehicles Symposium (IV) 2021)](http://urbanpose-dataset.com/info/Datasets/198) [[paper link](https://ieeexplore.ieee.org/abstract/document/9575469)]

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

* ⭐**CrowdPose(CVPR2019)** CrowdPose: Efficient Crowded Scenes Pose Estimation and a New Benchmark [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_CrowdPose_Efficient_Crowded_Scenes_Pose_Estimation_and_a_New_Benchmark_CVPR_2019_paper.html)][[codes|(SJTU) official PyTorch](https://github.com/Jeff-sjtu/CrowdPose)]

* **(CVPR2019)** Multi-Person Pose Estimation with Enhanced Channel-wise and Spatial Information [[arxiv link](https://arxiv.org/abs/1905.03466)]

* **(CVPR2019)** PoseFix: Model-Agnostic General Human Pose Refinement Network [[paper link](https://www.researchgate.net/publication/338506497_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network)]
 
* **(arxiv2019)** Rethinking on Multi-Stage Networks for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1901.00148v1)]

* ⭐**DarkPose(CVPR2020)** Distribution-Aware Coordinate Representation for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1910.06278)][[project link](https://ilovepose.github.io/coco/)][[Codes|PyTorch(offical)](https://github.com/ilovepose/DarkPose)]

* ⭐**UDP-Pose(CVPR2020)** The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1911.07524)][[Codes|](https://github.com/HuangJunJie2017/UDP-Pose)]

* **Graph-PCNN(arxiv 2020)** Graph-PCNN: Two Stage Human Pose Estimation with Graph Pose Refinement [[arxiv link](http://arxiv.org/abs/2007.10599)]

* **RSN-PRM(arxiv2020)** Learning Delicate Local Representations for Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/2003.04030v3)]

* **OPEC-Net(arxiv2020)** Peeking into occluded joints: A novel framework for crowd pose estimation [[arxiv link](https://arxiv.org/abs/2003.10506)]

* **(arxiv2020)(video based)** Self-supervised Keypoint Correspondences for Multi-Person Pose Estimation and Tracking in Videos [[arxiv link](https://arxiv.org/abs/2004.12652)]

* ⭐**PoseNAS(ACMMM2020)** Pose-native Network Architecture Search for Multi-person Human Pose Estimation [[paper link](https://dl.acm.org/doi/abs/10.1145/3394171.3413842)][[codes|official PyTorch](https://github.com/for-code0216/PoseNAS)][`Network Architecture Search (NAS) based two-stage MPPE`]

* **CCM(IJCV2021)** Towards High Performance Human Keypoint Detection [[paper link](https://link.springer.com/article/10.1007/s11263-021-01482-8)][[codes|official (not released)](https://github.com/chaimi2013/CCM)]

* **OmniPose(arxiv2021)** OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/2103.10180)]

* **MIPNet(ICCV2021)** Multi-Instance Pose Networks: Rethinking Top-Down Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Khirodkar_Multi-Instance_Pose_Networks_Rethinking_Top-Down_Pose_Estimation_ICCV_2021_paper.html)][[project link](https://rawalkhirodkar.github.io/mipnet/)][[codes|official demo](https://github.com/rawalkhirodkar/MIPNet)]

* ⭐**TransPose(ICCV2021)** TransPose: Keypoint Localization via Transformer [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_TransPose_Keypoint_Localization_via_Transformer_ICCV_2021_paper.html)][[codes|official PyTroch](https://github.com/yangsenius/TransPose)][`Transformer based two-stage MPPE (light-weight)`]

* ⭐**TokenPose(ICCV2021)** TokenPose: Learning Keypoint Tokens for Human Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Li_TokenPose_Learning_Keypoint_Tokens_for_Human_Pose_Estimation_ICCV_2021_paper.html)][[codes|official PyTroch](https://github.com/leeyegy/TokenPose)][`Token representation based two-stage MPPE (light-weight)`]

* ⭐**Lite-HRNet(CVPR2021)** Lite-HRNet: A Lightweight High-Resolution Network [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Yu_Lite-HRNet_A_Lightweight_High-Resolution_Network_CVPR_2021_paper.html)][[codes|official PyTorch](https://github.com/HRNet/Lite-HRNet)][`This work is done by the original group of HRNet`]

* ⭐**LitePose(CVPR2022)** Lite Pose: Efficient Architecture Design for 2D Human Pose Estimation [[paper link](https://tinyml.mit.edu/wp-content/uploads/2022/04/CVPR2022__Lite_Pose.pdf)][[project link](https://tinyml.mit.edu/publications/)][[codes|official PyTorch](https://github.com/mit-han-lab/litepose)][`Model quantization and compression on Qualcomm Snapdragon`]

* **CID(CVPR2022)** Contextual Instance Decoupling for Robust Multi-Person Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Contextual_Instance_Decoupling_for_Robust_Multi-Person_Pose_Estimation_CVPR_2022_paper.html)][[codes|official](https://github.com/kennethwdk/CID)][[First Author: Dongkai Wang](https://kennethwdk.github.io/)]



### ▶  Two-Stage [Bottom-Up] Multiple Person Pose Estimation

* **DeepCut(CVPR2016)** DeepCut: Joint Subset Partition and Labeling for Multi Person Pose Estimation [[arxiv link](https://arxiv.org/abs/1511.06645)]

* **DeeperCut(ECCV2016)** DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model [[arxiv link](https://arxiv.org/abs/1605.03170)]

* **ArtTrack(CVPR2017)** ArtTrack: Articulated Multi-Person Tracking in the Wild [[paper link](https://openaccess.thecvf.com/content_cvpr_2017/papers/Insafutdinov_ArtTrack_Articulated_Multi-Person_CVPR_2017_paper.pdf)]

* ⭐**OpenPose(CVPR2017)** Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields [[arxiv link](https://arxiv.org/abs/1611.08050)][[Codes|Caffe&Matlab(offical)](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)][[Codes|Caffe(offical only for testing)](https://github.com/CMU-Perceptual-Computing-Lab/openpose)][Codes|PyTorch(unoffical by tensorboy)](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)]

* ⭐**AssociativeEmbedding(NIPS2017)** Associative Embedding: End-to-end Learning for Joint Detection and Grouping [[arxiv link](https://arxiv.org/abs/1611.05424)][[Codes|PyTorch(offical)](https://github.com/princeton-vl/pose-ae-train)]

* **(ICCVW2017)** Multi-Person Pose Estimation for PoseTrack with Enhanced Part Affinity Fields [[paper link](https://posetrack.net/workshops/iccv2017/pdfs/ML_Lab.pdf)][[CSDN blog](https://blog.csdn.net/m0_37644085/article/details/82928933)]

* **(CVPRW2018)** Learning to Refine Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1804.07909)]

* ⭐**MultiPoseNet(ECCV2018)(multi-task)** MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network [[arxiv link](https://arxiv.org/abs/1807.04067)][[Codes|PyTorch(offical)](https://github.com/salihkaragoz/pose-residual-network-pytorch)]

* **DirectPose(arxiv2019)** DirectPose: Direct End-to-End Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/1911.07451v2)]

* **OpenPoseTrain(ICCV2019)** Single-Network Whole-Body Pose Estimation [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Hidalgo_Single-Network_Whole-Body_Pose_Estimation_ICCV_2019_paper.html)][[codes|official](https://github.com/CMU-Perceptual-Computing-Lab/openpose_train)][`simultaneous localization of body, face, hands, and feet keypoints`]

* ⭐**OpenPifPaf(CVPR2019)** PifPaf: Composite Fields for Human Pose Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.pdf)][[Codes|PyTorch(offical)](https://github.com/vita-epfl/openpifpaf)]

* ⭐**HigherHRNet(CVPR2020)** HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1908.10357)][[Codes|PyTorch(offical)](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)]

* ⭐**MDN3(CVPR2020)** Mixture Dense Regression for Object Detection and Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1912.00821)][[Codes|PyTorch(offical)](https://github.com/alivaramesh/MixtureDenseRegression)]

* **HGG(arxiv2020)** Differentiable Hierarchical Graph Grouping for Multi-person Pose Estimation [[arxiv link](https://arxiv.org/abs/2007.11864)]

* ⭐**EfficientHRNet(arxiv2020)** EfficientHRNet: Efficient Scaling for Lightweight High-Resolution Multi-Person Pose Estimation [[paper link](https://arxiv.org/abs/2007.08090)]

* **SimplePose(AAAI2020)** Simple pose: Rethinking and improving a bottom-up approach for multi-person pose estimation [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/6797)][[codes|official PyTorch](https://github.com/hellojialee/Improved-Body-Parts)][`An improved OpenPose based on Stacked Hourglass and proposed Body Parts`]

* **DGCN(AAAI2020)** DGCN: Dynamic Graph Convolutional Network for Efficient Multi-Person Pose Estimation [[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/6867)][`Graph based two-stage MPPE`]

* **PINet(NIPS2021)** Robust Pose Estimation in Crowded Scenes with Direct Pose-Level Inference [[paper link](https://proceedings.neurips.cc/paper/2021/hash/31857b449c407203749ae32dd0e7d64a-Abstract.html)][[codes|official PyTorch](https://github.com/kennethwdk/PINet)][[First Author: Dongkai Wang](https://kennethwdk.github.io/)]

* **CenterGroup(ICCV2021)** The Center of Attention: Center-Keypoint Grouping via Attention for Multi-Person Pose Estimation [[paper link](https://arxiv.org/abs/2110.05132)][[codes|official PyTorch based on mmpose and HigherHRNet](https://github.com/dvl-tum/center-group)]

* **SWAHR(CVPR2021)** Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation [[arxiv link](https://arxiv.org/abs/2012.15175)][[Codes|official pytorch based on HigherHRNet](https://github.com/greatlog/SWAHR-HumanPose)]

* **DERK(CVPR2021)** Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression [[arxiv link](https://arxiv.org/abs/2104.02300)][[Codes|official pytorch](https://github.com/HRNet/DEKR)]


### ▶  Single-Stage Multiple Person Pose Estimation

* **SPM(ICCV2019)** Single-Stage Multi-Person Pose Machines [[arxiv link](https://arxiv.org/abs/1908.09220)][[Codes|PyTorch(offical not released)](https://github.com/NieXC/pytorch-spm)][[Codes|Tensorflow(unoffical)](https://github.com/murdockhou/Single-Stage-Multi-person-Pose-Machines)][[CSDN blog](https://blog.csdn.net/Murdock_C/article/details/100545377)]

* **CenterNet(arxiv2019)** Objects as Points [[arxiv link](https://arxiv.org/abs/1904.07850)]

* **DeepDarts(CVPRW2021)** DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts using a Single Camera [[paper link](https://openaccess.thecvf.com/content/CVPR2021W/CVSports/papers/McNally_DeepDarts_Modeling_Keypoints_as_Objects_for_Automatic_Scorekeeping_in_Darts_CVPRW_2021_paper.pdf)]

* ⭐**KAPAO(ECCV2022)** Rethinking Keypoint Representations: Modeling Keypoints and Poses as Objects for Multi-Person Human Pose Estimation [[arxiv link](https://arxiv.org/abs/2111.08557)][[codes|(official pytorch using YOLOv5)](https://github.com/wmcnally/kapao)]

* **YOLO-Pose(CVPRW2022)** YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss [[paper link](https://arxiv.org/abs/2204.06806)][[codes|official edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)][[codes|official edgeai-yolov5](https://github.com/TexasInstruments/edgeai-yolov5)]

### ▶  Simultaneous Multiple Person Pose Estimation and Instance Segmentation

* ⭐**Mask R-CNN(ICCV2017)(multi-task)** Mask R-CNN [[paper link](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)]

* ⭐**PersonLab(ECCV2018)(multi-task)** PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model [[arxiv link](https://arxiv.org/abs/1803.08225)][[Codes|Keras&Tensorflow(unoffical by octiapp)](https://github.com/octiapp/KerasPersonLab)][[Codes|Tensorflow(unoffical)](https://github.com/scnuhealthy/Tensorflow_PersonLab)]

* **ACPNet(ICME2019)** ACPNet: Anchor-Center Based Person Network for Human Pose Estimation and Instance Segmentation [[paper link](https://ieeexplore.ieee.org/abstract/document/8784943)][`based on Mask R-CNN`]

* **Pose2Seg(CVPR2019)** Pose2Seg: Detection Free Human Instance Segmentation [[paper link](https://arxiv.org/abs/1803.10683)][[codes|official](https://github.com/liruilong940607/OCHumanApi)]

* **PointSetNet(ECCV2020)** Point-Set Anchors for Object Detection, Instance Segmentation and Pose Estimation [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_31)][`Not a multi-task end-to-end network`]

* **Multitask-CenterNet(ICCVW2021)** MultiTask-CenterNet (MCN): Efficient and Diverse Multitask Learning Using an Anchor Free Approach [[paper link](https://openaccess.thecvf.com/content/ICCV2021W/ERCVAD/html/Heuer_MultiTask-CenterNet_MCN_Efficient_and_Diverse_Multitask_Learning_Using_an_Anchor_ICCVW_2021_paper.html)][`based on the CenterNet`]

* **PosePlusSeg(AAAI2022)** Joint Human Pose Estimation and Instance Segmentation with PosePlusSeg [[paper link](https://www.aaai.org/AAAI22Papers/AAAI-6681.AhmadN.pdf)][[codes|official tensorflow](https://github.com/RaiseLab/PosePlusSeg)][`similarly with the PersonLab`]


### ▶  3D Multiple Person Pose Estimation

* **mvpose(CVPR2019)(monocular multi-view)** Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views [[arxiv link](https://arxiv.org/abs/1901.04111)][[project link](https://zju3dv.github.io/mvpose/)][[Codes|Torch&Tensorflow(offical)](https://github.com/zju3dv/mvpose)]

* **EpipolarPose(CVPR2019)(monocular multi-view)** Self-Supervised Learning of 3D Human Pose using Multi-view Geometry [[arxiv link](https://arxiv.org/abs/1903.02330)][[project link](https://mkocabas.github.io/epipolarpose.html)][[Codes|PyTorch(offical)](https://github.com/mkocabas/EpipolarPose)]

* **SMAP(ECCV2020)** SMAP: Single-Shot Multi-person Absolute 3D Pose Estimation [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_33)][[project link](https://zju3dv.github.io/SMAP/)][[codes|official PyTorch](https://github.com/zju3dv/SMAP)]

* **(multi-views)(ICCV2021)** Shape-aware Multi-Person Pose Estimation from Multi-View Images [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Dong_Shape-Aware_Multi-Person_Pose_Estimation_From_Multi-View_Images_ICCV_2021_paper.html)][[project link](https://ait.ethz.ch/projects/2021/multi-human-pose/)][[codes|official](https://github.com/zj-dong/Multi-Person-Pose-Estimation)]

* **MVP(NIPS2021)** Direct Multi-view Multi-person 3D Pose Estimation [[paper link](https://proceedings.neurips.cc/paper/2021/hash/6da9003b743b65f4c0ccd295cc484e57-Abstract.html)][[codes|official PyTorch](https://github.com/sail-sg/mvp)]


### ▶  Special Multiple Person Pose Estimation

* **PoseTrack(CVPR2017)** PoseTrack: Joint Multi-Person Pose Estimation and Tracking [[arxiv link](https://arxiv.org/abs/1611.07727)][[Codes|Matlab&Caffe](https://github.com/iqbalu/PoseTrack-CVPR2017)]

* **Detect-and-Track(CVPR2018)** Detect-and-Track: Efficient Pose Estimation in Videos [[arxiv link](https://arxiv.org/abs/1712.09184)][[project link](https://rohitgirdhar.github.io/DetectAndTrack/)][[Codes|Detectron(offical)](https://github.com/facebookresearch/DetectAndTrack/)][[codes|official](https://github.com/wmcnally/deep-darts)]

* **PoseFlow(BMVC2018)** Pose Flow: Efficient Online Pose Tracking [[arxiv link](https://arxiv.org/abs/1802.00977)][[Codes|AlphaPose(offical)](https://github.com/YuliangXiu/PoseFlow)]

* **DensePose(CVPR2018)** DensePose: Dense Human Pose Estimation In The Wild [[arxiv link](https://arxiv.org/abs/1802.00434)][[project link](http://densepose.org/)][[Codes|Caffe2(offical)](https://github.com/facebookresearch/Densepose)]

* **RF-Pose(CVPR2018)(radio frequency)** Through-Wall Human Pose Estimation Using Radio Signals [[paper link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Through-Wall_Human_Pose_CVPR_2018_paper.pdf)][[project link](http://rfpose.csail.mit.edu/)]

* **DoubleFusion(TPAMI2019)(3D single-view real-time depth-sensor)** DoubleFusion: Real-time Capture of Human Performances with Inner Body Shapes from a Single Depth Sensor [[arxiv link](https://arxiv.org/pdf/1804.06023.pdf)]

* **Keypoint-Communities(ICCV2019)** Keypoint Communities [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Zauss_Keypoint_Communities_ICCV_2021_paper.html)][`Model all keypoints belonging to a human or an object (the pose) as a graph`]

* **BlazePose (CVPRW2020)** BlazePose: On-device Real-time Body Pose tracking [[paper link](https://arxiv.org/abs/2006.10204)][[project link](https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html)]

* **ODKD(arxiv2021)** Orderly Dual-Teacher Knowledge Distillation for Lightweight Human Pose Estimation [[paper link](https://arxiv.org/abs/2104.10414)][`Knowledge Distillation of MPPE based on HRNet`]

* **MEVADA(ICCV2021)** Single View Physical Distance Estimation using Human Pose [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Fei_Single_View_Physical_Distance_Estimation_Using_Human_Pose_ICCV_2021_paper.html)][[project link](https://feixh.github.io/projects/physical_distance/)]


### ▶  Domain Adaptive Multiple Person Pose Estimation

* **(ICCV2019)** Cross-Domain Adaptation for Animal Pose Estimation [[arxiv link](https://arxiv.org/abs/1908.05806)][[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Cao_Cross-Domain_Adaptation_for_Animal_Pose_Estimation_ICCV_2019_paper.html)]

* **(ACMMM2019)** Unsupervised Domain Adaptation for 3D Human Pose Estimation [[paper link](http://zju-capg.org/unsupervised_domain_adaptation/main.pdf)]

* **Semi_Human_Pose(ICCV2021)** An Empirical Study of the Collapsing Problem in Semi-Supervised 2D Human Pose Estimation [[paper link](https://arxiv.org/abs/2011.12498)][[codes|official PyTorch](https://github.com/xierc/Semi_Human_Pose)]

* **Pose_DomainAdaption(ACMMM2021)** Alleviating Human-level Shift: A Robust Domain Adaptation Method for Multi-person Pose Estimation [[paper link](https://dl.acm.org/doi/abs/10.1145/3394171.3414040)][[Codes|PyTorch](https://github.com/Sophie-Xu/Pose_DomainAdaption)]

* **ADHNN(AAAI2022)** Adaptive Hypergraph Neural Network for Multi-person Pose Estimation [[paper link](https://www.aaai.org/AAAI22Papers/AAAI-1201.XuX.pdf)][[Codes|PyTorch](https://github.com/Sophie-Xu/Pose-ADHNN)]

* **UDA-Animal-Pose(CVPR2021)** From Synthetic to Real: Unsupervised Domain Adaptation for Animal Pose Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Li_From_Synthetic_to_Real_Unsupervised_Domain_Adaptation_for_Animal_Pose_CVPR_2021_paper.html)][[codes|PyTorch](https://github.com/chaneyddtt/UDA-Animal-Pose)]

* **(WACV2022)** Transfer Learning for Pose Estimation of Illustrated Characters [[arxiv link](https://arxiv.org/abs/2108.01819)][[paper link](https://openaccess.thecvf.com/content/WACV2022/html/Chen_Transfer_Learning_for_Pose_Estimation_of_Illustrated_Characters_WACV_2022_paper.html)][[codes|official PyTorch](https://github.com/ShuhongChen/bizarre-pose-estimator)]

* **AdaptOR(arxiv2021)** Unsupervised domain adaptation for clinician pose estimation and instance segmentation in the OR [[paper link](https://arxiv.org/abs/2108.11801)]

* **SemiMultiPose(arxiv2022)** SemiMultiPose: A Semi-supervised Multi-animal Pose Estimation Framework [[paper link](https://arxiv.org/abs/2204.07072)]
