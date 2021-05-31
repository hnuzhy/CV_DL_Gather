#  List for public implementation of various algorithms

## 1) Pubilc Datasets and Challenges

* [LIP(Look Into Person)](http://www.sysu-hcp.net/lip/index.php)
* [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#)
* [COCO - Common Objects in Context](https://cocodataset.org/)
* [PoseTrack: Dataset and Benchmark](https://posetrack.net/)


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
* [(github) Awesome Human Pose Estimation](https://github.com/cbsudux/awesome-human-pose-estimation)
* [(real time pose in github) tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)
* [(real time pose in github) PoseEstimationForMobile](https://github.com/edvardHua/PoseEstimationForMobile)
* [(real time pose in github) Real-time 2D MPPE on CPU: Lightweight OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
* [(Application) FXMirror虚拟试衣解决方案](http://fxmirror.net/zh/features)
* [(Application) 3D试衣间：人工智能虚拟试衣系统](http://3d.oleoad.com/3dshiyi.asp)



## 4) Papers and Sources Codes

### ▶ Single Person Pose Estimation

* **PoseMachines(ECCV2014)** Pose Machines: Articulated Pose Estimation via Inference Machines [[paper link](https://www.ri.cmu.edu/pub_files/2014/7/poseMachines.pdf)][[project link](http://www.cs.cmu.edu/~vramakri/poseMachines.html)]

* **DeepPose(CVPR2014)** DeepPose: Human Pose Estimation via Deep Neural Networks [[arxiv link](https://arxiv.org/abs/1312.4659)][[Codes|OpenCV(unoffical)](https://github.com/mitmul/deeppose)]

* **CPM(CVPR2016)** Convolutional Pose Machines [[arxiv link](https://arxiv.org/abs/1602.00134)][[Codes|Caffe(offical)](https://github.com/shihenw/convolutional-pose-machines-release)][[Codes|Tensorflow(unoffical)](https://github.com/psycharo/cpm)]

* **StackHourglass(ECCV2016)** Stacked Hourglass Networks for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1603.06937)][[Codes|Torch7(offical old)](https://github.com/princeton-vl/pose-hg-train)][[Codes|PyTorch(offical new)](https://github.com/princeton-vl/pytorch_stacked_hourglass)][[Codes|Tensorflow(unoffical)](https://github.com/wbenbihi/hourglasstensorlfow)]

* **PyraNet(ICCV2017)** Learning Feature Pyramids for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1708.01101)][[Codes|Torch(offical)](https://github.com/bearpaw/PyraNet)]

* **Adversarial-PoseNet(ICCV2017)** Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1705.00389)][[Codes|PyTorch(unoffical)](https://github.com/rohitrango/Adversarial-Pose-Estimation)]
 
 

### ▶ Two-Stage Top-Down Multiple Person Pose Estimation

* **DeeperCut(ECCV2016)** DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model [[arxiv link](http://arxiv.org/abs/1605.03170)][[project link](http://pose.mpi-inf.mpg.de/#)][[Codes|Tensorflow(offical)](https://github.com/eldar/pose-tensorflow)]

* **AlphaPose(ICCV2017)** RMPE: Regional Multi-person Pose Estimation [[arxiv link](https://arxiv.org/abs/1612.00137)][[project link](https://www.mvig.org/research/alphapose.html)][[Codes|PyTorch(offical)](https://github.com/MVIG-SJTU/AlphaPose)]

* **SimpleBaseline(ECCV2018)** Simple Baselines for Human Pose Estimation and Tracking [[arxiv link](https://arxiv.org/abs/1804.06208)][[Codes|PyTorch(offical)](https://github.com/Microsoft/human-pose-estimation.pytorch)][[Codes|PyTorch(flowtrack part)](https://github.com/simochen/flowtrack.pytorch)]

* **CPN(CVPR2018)** Cascaded Pyramid Network for Multi-Person Pose Estimation [[arxiv link](https://arxiv.org/abs/1711.07319)][[Codes|Tensorflow(offical)](https://github.com/chenyilun95/tf-cpn)][[Codes|Tensorflow(offical megvii)](https://github.com/megvii-detection/tf-cpn)][[zhihu blogs](https://zhuanlan.zhihu.com/p/37582402)]

* **HRNet(CVPR2019)** Deep High-Resolution Representation Learning for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1902.09212)][[Codes|PyTorch(offical)](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)][[Codes|(Repositories using HRNet as backbone)](https://github.com/HRNet)][[Codes|Tensorflow for fun](https://github.com/VXallset/deep-high-resolution-net.TensorFlow)]

* **DarkPose(CVPR2020)** Distribution-Aware Coordinate Representation for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1910.06278)][[project link](https://ilovepose.github.io/coco/)][[Codes|PyTorch(offical)](https://github.com/ilovepose/DarkPose)]

* **UDP-Pose(CVPR2020)** The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1911.07524)][[Codes|](https://github.com/HuangJunJie2017/UDP-Pose)]


### ▶  Two-Stage Bottom-Up Multiple Person Pose Estimation

* **OpenPose(CVPR2017)** Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields [[arxiv link](https://arxiv.org/abs/1611.08050)][[Codes|Caffe&Matlab(offical)](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)][[Codes|Caffe(offical only for testing)](https://github.com/CMU-Perceptual-Computing-Lab/openpose)][Codes|PyTorch(unoffical by tensorboy)](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)]

* **AssociativeEmbedding(NIPS2017)** Associative Embedding: End-to-end Learning for Joint Detection and Grouping [[arxiv link](https://arxiv.org/abs/1611.05424)][[Codes|PyTorch(offical)](https://github.com/princeton-vl/pose-ae-train)]

* **MultiPoseNet(ECCV2018)** MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network [[arxiv link](https://arxiv.org/abs/1807.04067)][[Codes|PyTorch(offical)](https://github.com/salihkaragoz/pose-residual-network-pytorch)]

* **PersonLab(ECCV2018)** PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model [[arxiv link](https://arxiv.org/abs/1803.08225)][[Codes|Keras&Tensorflow(unoffical by octiapp)](https://github.com/octiapp/KerasPersonLab)][[Codes|Tensorflow(unoffical)](https://github.com/scnuhealthy/Tensorflow_PersonLab)]

* **OpenPifPaf(CVPR2019)** PifPaf: Composite Fields for Human Pose Estimation [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.pdf)][[Codes|PyTorch(offical)](https://github.com/vita-epfl/openpifpaf)]

* **HigherHRNet(CVPR2020)** HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1908.10357)][[Codes|PyTorch(offical)](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)]

* **MDN3(CVPR2020)** Mixture Dense Regression for Object Detection and Human Pose Estimation [[arxiv link](https://arxiv.org/abs/1912.00821)][[Codes|PyTorch(offical)](https://github.com/alivaramesh/MixtureDenseRegression)]



### ▶  Single-Stage Multiple Person Pose Estimation

* **SPM(ICCV2019)** Single-Stage Multi-Person Pose Machines [[arxiv link](https://arxiv.org/abs/1908.09220)][[Codes|PyTorch(offical not released)](https://github.com/NieXC/pytorch-spm)][[Codes|Tensorflow(unoffical)](https://github.com/murdockhou/Single-Stage-Multi-person-Pose-Machines)][[CSDN blog](https://blog.csdn.net/Murdock_C/article/details/100545377)]


### ▶  3D Multiple Person Pose Estimation

* **mvpose(CVPR2019)** Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views [[arxiv link](https://arxiv.org/abs/1901.04111)][[project link](https://zju3dv.github.io/mvpose/)][[Codes|Torch&Tensorflow(offical)](https://github.com/zju3dv/mvpose)]

* **EpipolarPose(CVPR2019)** Self-Supervised Learning of 3D Human Pose using Multi-view Geometry [[arxiv link](https://arxiv.org/abs/1903.02330)][[project link](https://mkocabas.github.io/epipolarpose.html)][[Codes|PyTorch(offical)](https://github.com/mkocabas/EpipolarPose)]



### ▶  Special Multiple Person Pose Estimation

* **PoseTrack(CVPR2017)** PoseTrack: Joint Multi-Person Pose Estimation and Tracking [[arxiv link](https://arxiv.org/abs/1611.07727)][[Codes|Matlab&Caffe](https://github.com/iqbalu/PoseTrack-CVPR2017)]

* **Detect-and-Track(CVPR2018)** Detect-and-Track: Efficient Pose Estimation in Videos [[arxiv link](https://arxiv.org/abs/1712.09184)][[project link](https://rohitgirdhar.github.io/DetectAndTrack/)][[Codes|Detectron(offical)](https://github.com/facebookresearch/DetectAndTrack/)]

* **PoseFlow(BMVC2018)** Pose Flow: Efficient Online Pose Tracking [[arxiv link](https://arxiv.org/abs/1802.00977)][[Codes|AlphaPose(offical)](https://github.com/YuliangXiu/PoseFlow)]

* **DensePose(CVPR2018)** DensePose: Dense Human Pose Estimation In The Wild [[arxiv link](https://arxiv.org/abs/1802.00434)][[project link](http://densepose.org/)][[Codes|Caffe2(offical)](https://github.com/facebookresearch/Densepose)]

* **RF-Pose(CVPR2018)** Through-Wall Human Pose Estimation Using Radio Signals [[paper link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Through-Wall_Human_Pose_CVPR_2018_paper.pdf)][[project link](http://rfpose.csail.mit.edu/)]


