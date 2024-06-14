# Grasp Pose Detection
*also belonging to `Manipulation Strategy` and `Grasp Prediction/Generation`*

## Materials

* [**paperswithcode** The Ranking of Robotic Grasping on GraspNet-1Billion](https://paperswithcode.com/sota/robotic-grasping-on-graspnet-1billion)

***

## Datasets

### ※ Widely Used Robot Grasp Datasets

* 👍**GraspNet-1Billion(CVPR2020)** GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html)][[project link](www.graspnet.net)][[code|official](https://github.com/graspnet/graspnet-baseline)][[paperswithcode link](https://paperswithcode.com/dataset/graspnet-1billion)][`SJTU`]

* 👍**ACRONYM(ICRA2021)(arxiv2020.11)** ACRONYM: A Large-Scale Grasp Dataset Based on Simulation [[paper link](https://ieeexplore.ieee.org/abstract/document/9560844/)][[arxiv link](https://arxiv.org/abs/2011.09584)][[project link](https://sites.google.com/nvidia.com/graspdataset)][[code|official](https://github.com/NVlabs/acronym)][`NVIDIA + University of Washington`]
 
* **SuctionNet-1Billion(RAL2021)(arxiv2021.03)** SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping [[paper link](https://ieeexplore.ieee.org/abstract/document/9547830/)][[arxiv link](https://arxiv.org/abs/2103.12311)][[project link](https://graspnet.net/suction)][[code|official](https://github.com/graspnet/suctionnet-baseline)][`SJTU`]
   
* **REGRAD(RAL2022)(arxiv2021.04)** REGRAD: A Large-Scale Relational Grasp Dataset for Safe and Object-Specific Robotic Grasping in Clutter
  [[paper link](https://ieeexplore.ieee.org/abstract/document/9681218/)][[arxiv link](https://arxiv.org/abs/2104.14118)][[dataset link](https://stuxjtueducn-my.sharepoint.com/personal/chaser123_stu_xjtu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchaser123%5Fstu%5Fxjtu%5Fedu%5Fcn%2FDocuments%2FREGRAD%5Fv1&ga=1)][[code|official](https://github.com/poisonwine/REGRAD)][`XJTU`; `RElational GRAsps Dataset (REGRAD)`]

### ※ Other New Robot Grasp Datasets

* **TransCG(RAL2022)(arxiv2022.02)** TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and a Grasping Baseline [[paper link](https://ieeexplore.ieee.org/abstract/document/9796631)][[arxiv link](https://arxiv.org/abs/2202.08471)][[project link](http://www.graspnet.net/transcg)][[code|official](https://github.com/galaxies99/TransCG)][`SJTU`][`Completion of Depth`]

* **DA2-Dataset(RAL2022)(arxiv2022.08)** DA2 Dataset: Toward Dexterity-Aware Dual-Arm Grasping [[paper link](https://ieeexplore.ieee.org/abstract/document/9826816/)][[arxiv link](https://arxiv.org/abs/2208.00408)][[project link](https://sites.google.com/view/da2dataset)][[dataset link](https://sites.google.com/view/da2dataset/dataset)][[code|official](https://sites.google.com/view/da2dataset/code)][`Technical University of Munich +  Tencent Robotics X + Imperial College London + ZJU + JHU`; Dual-Arm Manipulation]
 
* 👍**OCRTOC(RAL2022)(arxiv2021.04)** OCRTOC: A Cloud-Based Competition and Benchmark for Robotic Grasping and Manipulation [[paper link](https://ieeexplore.ieee.org/abstract/document/9619915/)][[arxiv link](https://arxiv.org/abs/2104.11446)][[project link](https://www.ocrtoc.org/)][[code|official](https://github.com/OCRTOC/OCRTOC_software_package)][`Alibaba AI Labs + UC San Diego + University of Edinburgh + German Aerospace Center`]
   
* **ClothObjectSet(RAL2022)(arxiv2021.11)** Household Cloth Object Set: Fostering Benchmarking in Deformable Object Manipulation [[paper link](https://ieeexplore.ieee.org/abstract/document/9732698/)][[arxiv link](https://arxiv.org/abs/2111.01527)][[project link](https://www.iri.upc.edu/groups/perception/ClothObjectSet/)][`Spain`]

* 👍**DexGraspNet(ICRA2023)(arxiv2022.10)** DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation [[paper link](https://ieeexplore.ieee.org/abstract/document/10160982/)][[arxiv link](https://arxiv.org/abs/2210.02697)][[project link](https://pku-epic.github.io/DexGraspNet/)][[code|official](https://github.com/PKU-EPIC/DexGraspNet)][`PKU`]

* **MetaGraspNetV2(TASE2023)** MetaGraspNetV2: All-in-One Dataset Enabling Fast and Reliable Robotic Bin Picking via Object Relationship Reasoning and Dexterous Grasping [[paper link](https://ieeexplore.ieee.org/abstract/document/10309974/)][[code|official](https://github.com/maximiliangilles/MetaGraspNet)][`Karlsruher Institut für Technologie (KIT)`]
 
* 👍**Sim-Suction(TRO2023)(arxiv2023.05)** Sim-Suction: Learning a Suction Grasp Policy for Cluttered Environments Using a Synthetic Benchmark [[paper link](https://ieeexplore.ieee.org/abstract/document/10314015)][[arxiv link](https://arxiv.org/abs/2305.16378)][[project link](https://junchengli1.github.io/Sim-Suction/)][[code|official](https://github.com/junchengli1/Sim-Suction-API)][`Purdue University`]

* **CoAS-Net(RAL2024)** CoAS-Net: Context-Aware Suction Network With a Large-Scale Domain Randomized Synthetic Dataset [[paper link](https://ieeexplore.ieee.org/abstract/document/10333337)][[code|official](https://github.com/SonYeongGwang/CoAS-Net.git)][`Sungkyunkwan University`]

* 👍**Grasp-Anything(ICRA2024)(arxiv2023.09)** Grasp-Anything: Large-scale Grasp Dataset from Foundation Models [[arxiv link](https://arxiv.org/abs/2309.09818)][[project link](https://grasp-anything-2023.github.io/)][`FPT Software - AIC Lab (Hanoi, Vietnam)`]

### ※ Datasets Closely Related to the Grasp Task

* **StereOBJ-1M(ICCV2021)** StereOBJ-1M: Large-Scale Stereo Image Dataset for 6D Object Pose Estimation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_StereOBJ-1M_Large-Scale_Stereo_Image_Dataset_for_6D_Object_Pose_Estimation_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2109.10115)][[project link](https://sites.google.com/view/stereobj-1m)][[code|official](https://github.com/xingyul/stereobj-1m)][`The Robotics Institute of Carnegie Mellon University`]

* 👍**TO-Scene(ECCV2022)** TO-Scene: A Large-scale Dataset for Understanding 3D Tabletop Scenes [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19812-0_20)][[arxiv link](https://arxiv.org/abs/2203.09440)][[code|official](https://github.com/GAP-LAB-CUHK-SZ/TO-Scene)][`CUHK-SZ`]

* **PhoCaL(CVPR2022)** PhoCaL: A Multi-Modal Dataset for Category-Level Object Pose Estimation with Photometrically Challenging Objects [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_PhoCaL_A_Multi-Modal_Dataset_for_Category-Level_Object_Pose_Estimation_With_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2205.08811)][[project link](https://www.campar.in.tum.de/public_datasets/2022_cvpr_wang/)][`TUM`]

* **HAMMER-dataset(CVPR2023)** On the Importance of Accurate Geometry Data for Dense 3D Vision Tasks [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Jung_On_the_Importance_of_Accurate_Geometry_Data_for_Dense_3D_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.14840)][[dataset link](https://github.com/Junggy/HAMMER-dataset)][`TUM + 3Dwe.ai + Huawei Noah’s Ark Lab + Siemens AG`][`HAMMER : Highly Accurate Multi-Modal Dataset for DEnse 3D Scene Regression`]

* **FantasticBreaks(CVPR2023)** Fantastic Breaks: A Dataset of Paired 3D Scans of Real-World Broken Objects and Their Complete Counterparts [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Lamb_Fantastic_Breaks_A_Dataset_of_Paired_3D_Scans_of_Real-World_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.14152)][[project link](https://terascale-all-sensing-research-studio.github.io/FantasticBreaks/)][[dataset link](https://drive.google.com/drive/folders/1mGldCURVSN77ZKYfvnEYl4l-QPJRgsJK?usp=sharing)][`Terascale All-sensing Research Studio (TARS) at Clarkson University, USA`]


***

## Papers

* 👍**graspnet-baseline(CVPR2020)** GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html)][[project link](www.graspnet.net)][[code|official](https://github.com/graspnet/graspnet-baseline)][[paperswithcode link](https://paperswithcode.com/dataset/graspnet-1billion)][`SJTU`]

* **Graspness(ICCV2021)** Graspness Discovery in Clutters for Fast and Accurate Grasp Detection [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.html)][[graspnetAPI link](https://github.com/graspnet/graspnetAPI)][[project link](https://graspnet.net/)][[code|official](https://github.com/rhett-chen/graspness_implementation)][`SJTU`]

* **Dex-NeRF(CoRL2021)(arxiv2021.10)** Dex-NeRF: Using a Neural Radiance field to Grasp Transparent Objects [[paper link](https://proceedings.mlr.press/v164/ichnowski22a.html)][[arxiv link](https://arxiv.org/abs/2110.14217)][[project link](https://sites.google.com/view/dex-nerf)][[code|official](https://github.com/BerkeleyAutomation/dex-nerf-datasets/)][`University of California Berkeley`][using the `NeRF`]

* **NDF(ICRA2022)(arxiv2021.12)** Neural Descriptor Fields: SE(3)-Equivariant Object Representations for Manipulation [[paper link](https://ieeexplore.ieee.org/abstract/document/9812146)][[arxiv link](https://arxiv.org/abs/2112.05124)][[project link](https://yilundu.github.io/ndf/)][[code|official](https://github.com/anthonysimeonov/ndf_robot)][`MIT + Google Research  + University of Toronto`][using the `NeRF`]

* ❤**ImplicitPCDA(CVPR2022)(arxiv2021.12)** Domain Adaptation on Point Clouds via Geometry-Aware Implicits [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Shen_Domain_Adaptation_on_Point_Clouds_via_Geometry-Aware_Implicits_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2112.09343)][[code|official](https://github.com/Jhonve/ImplicitPCDA)][`ZJU + Stanford + PKU`][`Point Cloud Adaptation`]

* **NeuralGrasps(CoRL2022)(arxiv2022.07)** NeuralGrasps: Learning Implicit Representations for Grasps of Multiple Robotic Hands [[paper link](https://proceedings.mlr.press/v205/khargonkar23a.html)][[arxiv link](https://arxiv.org/abs/2207.02959)][[project link](https://irvlutd.github.io/NeuralGrasps)][[code|official](https://github.com/IRVLUTD/neuralgrasps-model)][[HandNet-Pipeline](https://github.com/IRVLUTD/handnet-pipeline)][`The University of Texas at Dallas + St. Mark's School of Texas`][using the `NeRF`]

* **Scale-Balanced-Grasp(CoRL2022)(arxiv2022.12)** Towards Scale Balanced 6-DoF Grasp Detection in Cluttered Scenes [[paper link](https://proceedings.mlr.press/v205/ma23a.html)][[arxiv link](https://arxiv.org/abs/2212.05275)][[project link]()][[code|official](https://github.com/mahaoxiang822/Scale-Balanced-Grasp)][`Beihang University`]

* **GPDAN(RAL2023)** GPDAN: Grasp Pose Domain Adaptation Network for Sim-to-Real 6-DoF Object Grasping [[paper link](https://ieeexplore.ieee.org/abstract/document/10153686)][`University of Chinese Academy of Sciences`][`Grasp Pose Domain Adaptation`] 

* 👍**NGDF(ICRA2023)(arxiv2022.11)** Neural Grasp Distance Fields for Robot Manipulation [[paper link](https://ieeexplore.ieee.org/abstract/document/10160217)][[arxiv link](https://arxiv.org/abs/2211.02647)][[project link](https://sites.google.com/view/neural-grasp-distance-fields)][[code|official](https://github.com/facebookresearch/NGDF/)][`Meta AI + Carnegie Mellon University`][using the `NeRF`]

* **DefGraspNets(ICRA2023)(arxiv2023.03)** DefGraspNets: Grasp Planning on 3D Fields with Graph Neural Nets [[paper link](https://ieeexplore.ieee.org/abstract/document/10160986/)][[arxiv link](https://arxiv.org/abs/2303.16138)][[project link](https://sites.google.com/view/defgraspnets)][`University of California, Berkeley + NVIDIA`][using the `NeRF`]

* **GraspNeRF(ICRA2023)(arxiv2022.10)** GraspNeRF: Multiview-based 6-DoF Grasp Detection for Transparent and Specular Objects Using Generalizable NeRF [[paper link](https://ieeexplore.ieee.org/abstract/document/10160842)][[arxiv link](https://arxiv.org/abs/2210.06575)][[project link](https://pku-epic.github.io/GraspNeRF/)][[code|official](https://github.com/PKU-EPIC/GraspNeRF)][`Peking University + Beijing Academy of Artificial Intelligence + National University of Defense Technology  `][using the `NeRF`]

* **Vision-Language-Grasping(ICRA2023)(arxiv2023.02)** A Joint Modeling of Vision-Language-Action for Target-oriented Grasping in Clutter [[paper link](https://ieeexplore.ieee.org/abstract/document/10161041)][[arxiv link](https://arxiv.org/abs/2302.12610)][[code|official](https://github.com/xukechun/Vision-Language-Grasping)][`ZJU + University of Texas at Austin`]

* **3DSGrasp(ICRA2023)(arxiv2023.01)** 3DSGrasp: 3D Shape-Completion for Robotic Grasp [[paper link](https://ieeexplore.ieee.org/abstract/document/10160350)][[arxiv link](https://arxiv.org/abs/2301.00866)][[code|official](https://github.com/NunoDuarte/3DSGrasp)][`Universidade de Lisboa, Portugal + University of Genoa, Italy`][`Completion of Shape`]

* **GraspAda(ICRA2023)** GraspAda: Deep Grasp Adaptation through Domain Transfer [[paper link](https://ieeexplore.ieee.org/abstract/document/10160213/)][`Wuhan University + THU + Chalmers University of Technology + UCL + CUHK`][`Grasp Pose Domain Adaptation`]

* **SPARTN(CVPR2023)(arxiv2023.01)** NeRF in the Palm of Your Hand: Corrective Augmentation for Robotics via Novel-View Synthesis[[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_NeRF_in_the_Palm_of_Your_Hand_Corrective_Augmentation_for_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2301.08556)][[project link](https://bland.website/spartn/)][`Stanford + MIT + Google`][using the `NeRF`]

* ❤**HyperPC(CVPR2023)(arxiv2023.07)** Hyperspherical Embedding for Point Cloud Completion [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Hyperspherical_Embedding_for_Point_Cloud_Completion_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2307.05634)][[project link](https://haomengz.github.io/hyperpc/index.html)][[code|official](https://github.com/haomengz/HyperPC)][`University of Michigan + CMU`][`Completion of Point Cloud`]

* ❤**PartManip(CVPR2023)(arxiv2023.03)** PartManip: Learning Cross-Category Generalizable Part Manipulation Policy From Point Cloud Observations [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Geng_PartManip_Learning_Cross-Category_Generalizable_Part_Manipulation_Policy_From_Point_Cloud_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2303.16958)][[project link](https://pku-epic.github.io/PartManip/)][[code|official](https://github.com/PKU-EPIC/PartManip)][`PKU + Beijing Academy of Artificial Intelligence`][`Point Cloud Generalization`]

* **M2T2(CoRL2023)(arxiv2023.11)** M2T2: Multi-Task Masked Transformer for Object-centric Pick and Place [[paper link](https://proceedings.mlr.press/v229/yuan23a.html)][[arxiv link](https://arxiv.org/abs/2311.00926)][[project link](https://m2-t2.github.io/)][[code|official](https://github.com/NVlabs/M2T2)][`University of Washington + NVIDIA`]

* **OCID-VLG(CoRL2023)(arxiv2023.11)** Language-guided Robot Grasping: CLIP-based Referring Grasp Synthesis in Clutter [[paper link](https://proceedings.mlr.press/v229/tziafas23a.html)][[arxiv link](https://arxiv.org/abs/2311.05779)][[dataset link](https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view?usp=share_link)][[code|official](https://github.com/gtziafas/OCID-VLG)][`University of Groningen + University of Edinburgh + University College London`]

* **(RAL2023)(arxiv2023.11)** Anthropomorphic Grasping with Neural Object Shape Completion [[paper link](https://ieeexplore.ieee.org/abstract/document/10271524)][[arxiv link](https://arxiv.org/abs/2311.02510)][`TUM`][using the `NeRF`][`Completion of Shape`]

* 👍👍**HGGD(RAL2023)(arxiv2024.03)** [[paper link](https://ieeexplore.ieee.org/document/10168242)][[arxiv link](https://arxiv.org/pdf/2403.18546)][[bilibili link](https://www.bilibili.com/video/BV1hH4y1H7qv/)][[code|official](https://github.com/THU-VCLab/HGGD)][`THU`][`Attention`: HGGD detects grasps only from `heatmap guidance`, without `any workspace mask` (adopted in [`Graspness`](https://github.com/rhett-chen/graspness_implementation)) or `object/foreground segmentation` method (adopted in [`Scale-balanced Grasp`](https://github.com/mahaoxiang822/scale-balanced-grasp)). It may be useful to add some of this prior information to get better results.]

* **RGBGrasp(RAL2024)(arxiv2023.11)** RGBGrasp: Image-Based Object Grasping by Capturing Multiple Views During Robot arm Movement With Neural Radiance Fields[[paper link](https://ieeexplore.ieee.org/abstract/document/10517376)][[arxiv link](https://arxiv.org/abs/2311.16592)][[project link](https://sites.google.com/view/rgbgrasp)][`PKU + Imperial College London + Oxford University`][using the `NeRF`]

* **TCRNet(TASE2024)** TCRNet: Transparent Object Depth Completion With Cascade Refinements [[paper link](https://ieeexplore.ieee.org/abstract/document/10464368)][`Beijing Institute of Technology`][`Completion of Depth`]

* **COT(WACV2024 Oral)(arxiv2023.08)** Synergizing Contrastive Learning and Optimal Transport for 3D Point Cloud Domain Adaptation [[paper link](https://openaccess.thecvf.com/content/WACV2024/html/Katageri_Synergizing_Contrastive_Learning_and_Optimal_Transport_for_3D_Point_Cloud_WACV_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2308.14126)][[project link](https://siddharthkatageri.github.io/COT/)][[code|official](https://github.com/siddharthKatageri/COT)][`IIIT Hyderabad, India + Fujitsu Research India`][`Point Cloud Adaptation`]

* 👍**NeuGraspNet(RSS2024)(arxiv2023.06)** Learning Any-View 6DoF Robotic Grasping in Cluttered Scenes via Neural Surface Rendering [[paper link]()][[arxiv link](https://arxiv.org/abs/2306.07392)][[project link](https://sites.google.com/view/neugraspnet)][`TU Darmstadt, Germany + NIT Trichy, India + Hessian.AI, Darmstadt, Germany`][using the `NeRF`]

* **(ICRA2024)(arxiv2024.02)** Articulated Object Manipulation with Coarse-to-fine Affordance for Mitigating the Effect of Point Cloud Noise [[arxiv link](https://arxiv.org/abs/2402.18699)][`PKU + Huawei`][`Noisy Label Learning`]

* **GL-MSDA(ICRA2024)(arxiv2024.03)** Sim-to-Real Grasp Detection with Global-to-Local RGB-D Adaptation [[arxiv link](https://arxiv.org/abs/2403.11511)][[code|official](https://github.com/mahaoxiang822/GL-MSDA)][`Beihang University + HIT`][`RGB-D Adaptation`]

* **APEX(IROS2024)(arxiv2024.04)** APEX: Ambidextrous Dual-Arm Robotic Manipulation Using Collision-Free Generative Diffusion Models [[arxiv link](https://arxiv.org/abs/2404.02284)][[project link](https://sites.google.com/view/apex-dual-arm/home)][`University of Central Florida`][It designed and used the `Collision-Free Generative Diffusion Models`; Dual-Arm Manipulation]

* **PhyGrasp(arxiv2024.02)** PhyGrasp: Generalizing Robotic Grasping with Physics-informed Large Multimodal Models [[arxiv link](https://arxiv.org/abs/2402.16836)][[project link](https://sites.google.com/view/phygrasp)][[code|official](https://github.com/dkguo/PhyGrasp)][`Mechanical Engineering + University of California + Berkeley`]

* 👍**FlexLoG(arxiv2024.03)** Rethinking 6-Dof Grasp Detection: A Flexible Framework for High-Quality Grasping [[arxiv link](https://arxiv.org/abs/2403.15054)][`THU + 3Shanghai AI Lab`]

* 👍**GaussianGrasper(arxiv2024.03)** GaussianGrasper: 3D Language Gaussian Splatting for Open-vocabulary Robotic Grasping [[paper link]()][[arxiv link](https://arxiv.org/abs/2403.09637)][[project link](https://mrsecant.github.io/GaussianGrasper/)][[dataset link](https://drive.google.com/file/d/1Zsl0yCXezqwLrAYzOb-u33q8gdjvRfAC/view?usp=drive_link)][[code|official](https://github.com/MrSecant/GaussianGrasper)][`Beihang University + EncoSmart + HKU + CASIA + Tsinghua AIR + Imperial College London`]

* 👍**CGDF(arxiv2024.04)** Constrained 6-DoF Grasp Generation on Complex Shapes for Improved Dual-Arm Manipulation [[arxiv link](https://arxiv.org/abs/2404.04643)][[project link](https://constrained-grasp-diffusion.github.io/)][`IIIT Hyderabad + MIT + Brown University`][`CGDF: Constrained Grasp Diffusion Fields`; Dual-Arm Manipulation]

* **SemGrasp(arxiv2024.04)** SemGrasp: Semantic Grasp Generation via Language Aligned Discretization [[arxiv link](https://arxiv.org/abs/2404.03590)][[project link](https://kailinli.github.io/SemGrasp/)][`SJTU + Shanghai AI Lab`]

* **DexGYS(arxiv2024.05)** Grasp as You Say: Language-guided Dexterous Grasp Generation [[arxiv link](https://arxiv.org/abs/2405.19291)][[project link](https://sites.google.com/stanford.edu/dexgys)][`Sun Yat-sen University  + Stanford University + Wuhan University`; `Dexterous Grasp as You Say (DexGYS)`]


* **** [[paper link]()][[arxiv link]()][[project link]()][[code|official]()]
