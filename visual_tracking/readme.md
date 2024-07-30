#  List for public implementation of various algorithms

## 1) Pubilc Datasets and Challenges

* [Human Action Video Dataset](http://videodatasets.org/)
* [Awesome-Video-Datasets](https://github.com/xiaobai1217/Awesome-Video-Datasets) [[Video Dataset Overview](https://www.di.ens.fr/~miech/datasetviz/)]
* [VOT Challenge](https://votchallenge.net/index.html) [[VOT git repos](https://github.com/votchallenge)]
* [TAO: A Large-Scale Benchmark for Tracking Any Object (ECCV2020)](https://taodataset.org/)
* [MOT challenge: Multiple Object Tracking Benchmark!](https://motchallenge.net/)

## 2) Pioneers and Experts

[[👍Martin Danelljan](https://martin-danelljan.github.io/)] [[VIS Lab People](https://ivi.fnwi.uva.nl/vislab/people/)]


## 3) Blogs and Videos

* [(B站) 2019-2020年目标跟踪资源全汇总（论文、模型代码、优秀实验室）](https://www.bilibili.com/read/cv7636814)
* [(B站) 张志鹏：Ocean/Ocean+：实时目标跟踪分割算法，小代价，大增益](https://www.bilibili.com/video/BV1354y1e7wU)
* [(github) Visual Tracking Paper List](https://github.com/foolwood/benchmark_results)
* [(github) PyQt5_YoLoV5_DeepSort](https://github.com/BioMeasure/PyQt5_YoLoV5_DeepSort)
* [(github) [YOLOv5 + DeepSORT] Yolov5 deepsort inference，使用YOLOv5+Deepsort实现车辆行人追踪和计数](https://github.com/Sharpiless/Yolov5-deepsort-inference)
* [(github) [YOLOv5 + DeepSORT] Real-time multi-object tracker using YOLO v5 and deep sort](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)[[Evaluation Page]](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/wiki/Evaluation)
* [(github) [Official DeepSort] MOT tracking using deepsort and yolov3 with pytorch](https://github.com/ZQPei/deep_sort_pytorch)
* [(github) [YOLOv5 + SORT] ClassySORT is a simple real-time multi-object tracker (MOT)](https://github.com/tensorturtle/classy-sort-yolov5)
* [(github) [FastMOT] High-performance multiple object tracking based on YOLO, Deep SORT, and KLT](https://github.com/GeekAlexis/FastMOT)
* [(github} [IJCV-2021] FairMOT: On the Fairness of Detection and Re-Identification in Multi-Object Tracking](https://github.com/ifzhang/FairMOT)
* [(arxiv) CountingMOT: Joint Counting, Detection and Re-Identification for Multiple Object Tracking](https://arxiv.org/abs/2212.05861)


## 4) Papers and Sources Codes

### ▶ Survey


### ▶ Optical Flow

* **FlowNet2.0(CVPR2017)** FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks [[arxiv link](https://arxiv.org/abs/1612.01925)][[Codes|PyTorch(offical)](https://github.com/NVIDIA/flownet2-pytorch)]


### ▶ Siamese Based

* **SiamMask(CVPR2019)** Fast Online Object Tracking and Segmentation: A Unifying Approach [[arxiv link](https://arxiv.org/abs/1812.05050)][[project link](http://www.robots.ox.ac.uk/~qwang/SiamMask/)][[Codes|PyTorch(offical)](https://github.com/foolwood/SiamMask)][[blog in wechat](https://mp.weixin.qq.com/s/tn3DBGQ-bfj8UCuupK-vHg)][[blog in bilibili](https://www.bilibili.com/video/av45602011/)]



### ▶ CF Based

* **ECO(CVPR2017)** ECO: Efficient Convolution Operators for Tracking [[arxiv link](https://arxiv.org/abs/1611.09224)][[Codes|Matlab(offical)](https://github.com/martin-danelljan/ECO)][[Codes|MXNet(unoffical)](https://github.com/StrangerZhang/pyECO)][[CSDN blog](https://blog.csdn.net/zixiximm/article/details/54378397)]


### ▶ Subfield

* **SiamCAR(CVPR2020)** SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking [[arxiv link](http://arxiv.org/abs/1911.07241v2)][[cvpr link](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_SiamCAR_Siamese_Fully_Convolutional_Classification_and_Regression_for_Visual_Tracking_CVPR_2020_paper.html)][[Codes|offical PyTorch](https://github.com/ohhhyeahhh/SiamCAR)]

### ▶ Point Based
*Point tracking is a new field with a few notable works released around the same time.*

* **PIPs(ECCV2022 Oral)(arxiv2022.04)** Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20047-2_4)][[arxiv link](https://arxiv.org/abs/2204.04153)][[project link](https://particle-video-revisited.github.io/)][[code|official](https://github.com/aharley/pips)][`Carnegie Mellon University`][It was an inspiration for the work `TAPIR`]

* 👍**OmniMotion(ICCV2023 Oral, Best Student Paper)(arxiv2023.06)** Tracking Everything Everywhere All at Once [[paper link](http://openaccess.thecvf.com/content/ICCV2023/html/Wang_Tracking_Everything_Everywhere_All_at_Once_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2306.05422)][[project link](https://omnimotion.github.io/)][[code|official](https://github.com/qianqianwang68/omnimotion)][`Cornell University + Google Research + UC Berkeley`][It doesn't perform as well as `TAPIR` and is substantially slower, but it provides `pseudo-3D reconstructions`, and could potentially be used on top of TAPIR tracks to further improve performance.]

* **TAPIR(ICCV2023)(arxiv2023.06)** TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement [[paper link](http://openaccess.thecvf.com/content/ICCV2023/html/Doersch_TAPIR_Tracking_Any_Point_with_Per-Frame_Initialization_and_Temporal_Refinement_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2306.08637)][[project link](https://deepmind-tapir.github.io/)][[code|official](https://github.com/deepmind/tapnet)][`Google DeepMind + University of Oxford`]

* **MFT(WACV2024)(arxiv2023.05)** MFT: Long-Term Tracking of Every Pixel [[paper link](https://openaccess.thecvf.com/content/WACV2024/html/Neoral_MFT_Long-Term_Tracking_of_Every_Pixel_WACV_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2305.12998)][[project link](https://cmp.felk.cvut.cz/~serycjon/MFT/)][[code|official](https://github.com/serycjon/MFT)][`Czech Technical University in Prague`][`Multi-Flow Tracking` hypothesizes many flow fields between different pairs of frames and scores them; multiple hypotheses leads to improved robustness.]

* **LEAP-VO(CVPR2024)(arxiv2024.01)** LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_LEAP-VO_Long-term_Effective_Any_Point_Tracking_for_Visual_Odometry_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2401.01887)][[project link](https://chiaki530.github.io/projects/leapvo)][[code|official](https://github.com/chiaki530/leapvo)][`TU Munich + Munich Center for Machine Learning + MPI for Intelligent Systems + Microsoft`][`Tracking + Visual Odometry`, Long-term Effective Any Point Tracking (LEAP)]

* 👍**CoTracker(ECCV2024)(arxiv2023.07)** CoTracker: It is Better to Track Together [[paper link]()][[arxiv link](https://arxiv.org/abs/2307.07635)][[project link](https://co-tracker.github.io/)][[code|official](https://github.com/facebookresearch/co-tracker)][`Meta AI + Visual Geometry Group, University of Oxford`]

* 👍**DINO-Tracker(ECCV2024)(arxiv2024.03)** DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video [[paper link]()][[arxiv link](https://arxiv.org/abs/2403.14548)][[project link](https://dino-tracker.github.io/)][[code|official](https://github.com/AssafSinger94/dino-tracker)][`Weizmann Institute of Science, Israel`]

* **BootsTAP(arxiv2024.02)** BootsTAP: Bootstrapped Training for Tracking-Any-Point [[arxiv link](https://arxiv.org/abs/2402.00847)][[project link](https://bootstap.github.io/)][[code|official](https://github.com/google-deepmind/tapnet?tab=readme-ov-file#colab-demo)][`Google DeepMind + University of Oxford`][It is based on the `TAPIR`]


* **** [[paper link]()][[arxiv link]()][[project link]()][[code|official]()]

