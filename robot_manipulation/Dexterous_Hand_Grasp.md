# ⭐Dexterous Hand Grasp
*also closely related to `Dexterous Robotic Manipulation`, `Dexterous (Bimanual) Hand Manipulation`, `Long-Horizon Robotic Manipulation` and `Contact-Rich Articulated Object Manipulation`*

## Softwares / Hardwares

* [**ShadowHand** Shadow Dexterous Hand Series - Research and Development Tool](https://www.shadowrobot.com/dexterous-hand-series/) [As a widely used five-finger robotic dexterous hand, `ShadowHand` amounts to `26 degrees of freedom (DoF)`, in contrast with `7 DoF` for a typical `parallel gripper`. Such high dimensionality magnifies the difficulty in both generating valid grasp poses and planning the execution trajectories, and thus distinguishes the dexterous grasping task from its counterpart for parallel grippers.]

* [**Isaac Gym** provides Benchmark Environments and the corresponding simulator](https://developer.nvidia.com/isaac-gym) [[Technical Paper](https://arxiv.org/abs/2108.10470)][[github link](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)][openreview link (NIPS2021 Track Datasets and Benchmarks Round2)](https://openreview.net/forum?id=fgFBtYgJQX_)][By `NVIDIA`]


## Materials

* 👍 **(github)(Hand3DResearch) Recent Progress in 3D Hand Tasks** [[github link](https://github.com/SeanChenxy/Hand3DResearch)]
* **(Website) GraspNet通用物体抓取(GraspNet-1Billion + AnyGrasp + SuctionNet-1Billion + TransCG)**  [[Homepage link](https://graspnet.net/index.html)]

## Datasets

* **Bi-DexHands / DexterousHands(NIPS2022 Datasets and Benchmarks Track)(arxiv2022.06)** Towards Human-Level Bimanual Dexterous Manipulation with Reinforcement Learning [[openreview link](https://openreview.net/forum?id=D29JbExncTP)][[paper link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/217a2a387f52c30755c37b0a73430291-Abstract-Datasets_and_Benchmarks.html)][[arxiv link](https://arxiv.org/abs/2206.08686)][[project link](https://bi-dexhands.ai/)][[code|official](https://github.com/PKU-MARL/DexterousHands)][`PKU-MARL`]

* **DexGraspNet (ICRA2023)** [DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation](https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/) [[paper link](https://ieeexplore.ieee.org/abstract/document/10160982)][[project link](https://pku-epic.github.io/DexGraspNet/)][`PKU`; related to [`He Wang`](https://hughw19.github.io/) group]

* **DFCData (UniDexGrasp CVPR2023)** [UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy](https://mirrors.pku.edu.cn/dl-release/) [[paper link](http://openaccess.thecvf.com/content/CVPR2023/html/Xu_UniDexGrasp_Universal_Robotic_Dexterous_Grasping_via_Learning_Diverse_Proposal_Generation_CVPR_2023_paper.html)][[project link](https://pku-epic.github.io/UniDexGrasp/)][`PKU`; related to [`He Wang`](https://hughw19.github.io/) group]

* **TACO (CVPR2024)(arxiv2024.01)** [TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding](https://taco2024.github.io/) [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_TACO_Benchmarking_Generalizable_Bimanual_Tool-ACtion-Object_Understanding_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2401.08399)][[weixin blog](https://mp.weixin.qq.com/s/WdbK93Z3T4a9_AfPuo0WOQ)][[code|official](https://github.com/leolyliu/TACO-Instructions)][`THU + Shanghai AI Lab + Shanghai Qi Zhi`]

* **SHOWMe(CVIU2024)** SHOWMe: Robust object-agnostic hand-object 3D reconstruction from RGB video [[paper link](https://www.sciencedirect.com/science/article/abs/pii/S1077314224001541)][[code|official](https://download.europe.naverlabs.com/showme/)][`NAVER LABS Europe + Inria centre at the University Grenoble Alpes`][It is based on the conference version [`(ICCVW2023) SHOWMe: Benchmarking Object-agnostic Hand-Object 3D Reconstruction`](https://openaccess.thecvf.com/content/ICCV2023W/ACVR/html/Swamy_SHOWMe_Benchmarking_Object-Agnostic_Hand-Object_3D_Reconstruction_ICCVW_2023_paper.html) with the [project link](https://europe.naverlabs.com/research/showme/) link]

## Papers

### ▶ Hand Object Contact/Interaction

* **ContactDB(CVPR2019 Oral, Best Paper Finalist)** ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging [[arxiv link](https://arxiv.org/abs/1904.06830)][[project link](https://contactdb.cc.gatech.edu/)][`Georgia Tech Robotics`][`the first large-scale dataset that records detailed contact maps for functional human grasps`]

* **ObMan(CVPR2019)(arxiv2019.04)** Learning Joint Reconstruction of Hands and Manipulated Objects [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Hasson_Learning_Joint_Reconstruction_of_Hands_and_Manipulated_Objects_CVPR_2019_paper.html)][[arxiv link](https://arxiv.org/abs/1904.05767)][[project link](https://hassony2.github.io/obman)][[code|official](https://github.com/hassony2/obman)][`Inria + PSL Research University + MPII`; This is also a new synthetic dataset]

* 👍**100DOH(CVPR2020 oral)** Understanding Human Hands in Contact at Internet Scale [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Shan_Understanding_Human_Hands_in_Contact_at_Internet_Scale_CVPR_2020_paper.html)][[arxiv link](https://arxiv.org/abs/2006.06669)][[project link](http://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/)][[code|official 1](https://github.com/ddshan/hand_object_detector)][[code|official 2](https://github.com/ddshan/hand_detector.d2)][`University of Michigan + Johns Hopkins University`][`hand-contact understanding`][`100DOH` dataset]

* 👍**ContactHands(NIPS2020)** Detecting Hands and Recognizing Physical Contact in the Wild [[paper link](https://proceedings.neurips.cc/paper/2020/hash/595373f017b659cb7743291e920a8857-Abstract.html)][[project link](http://vision.cs.stonybrook.edu/~supreeth/ContactHands_data_website/)][[code|official](https://github.com/cvlab-stonybrook/ContactHands)][[CVLab@StonyBrook](https://github.com/cvlab-stonybrook)][`ContactHands` dataset][`hand contact estimation`]

* **ContactPose(ECCV2020)** ContactPose: A Dataset of Grasps with Object Contact and Hand Pose [[paper link](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_22)][[arxiv link](https://arxiv.org/abs/2007.09545)][[project link](https://contactpose.cc.gatech.edu/)][`hand contact estimation`][`ContactPose` dataset][the first dataset of hand-object contact paired with hand pose, object pose, and RGB-D images]

* **Hand-Object Contact Prediction(BMVC2021)** Hand-Object Contact Prediction via Motion-Based Pseudo-Labeling and Guided Progressive Label Correction [[arxiv link](https://arxiv.org/abs/2110.10174)][[code|official](https://github.com/takumayagi/hand_object_contact_prediction)][`Hand-Object Contact Prediction`]

* **ContactOpt(CVPR2021)** ContactOpt: Optimizing Contact To Improve Grasps [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Grady_ContactOpt_Optimizing_Contact_To_Improve_Grasps_CVPR_2021_paper.html)][[code|official](https://github.com/facebookresearch/contactopt)][`hand contact`, `grasp`]

* **TUCH(Towards Understanding Contact in Humans)(CVPR2021)** On Self-Contact and Human Pose [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Muller_On_Self-Contact_and_Human_Pose_CVPR_2021_paper.html)][[project link](https://tuch.is.tue.mpg.de)][A dataset of `3D Contact Poses (3DCP)`, `hand contact estimation`, `MPII`, `single-person`]

* **GraspTTA&GraspNet(ICCV2021 Oral)(arxiv2021.04)** Hand-Object Contact Consistency Reasoning for Human Grasps Generation [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Jiang_Hand-Object_Contact_Consistency_Reasoning_for_Human_Grasps_Generation_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2104.03304)][[project link](https://hwjiang1510.github.io/GraspTTA/)][[code|official](https://github.com/hwjiang1510/GraspTTA)][`UC San Diego`; [`Xiaolong Wang`](https://xiaolonw.github.io/) group]

* **PressureVision(ECCV2022)** PressureVision: Estimating Hand Pressure from a Single RGB Image [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20068-7_19)][[code|official](https://github.com/facebookresearch/pressurevision)][`facebookresearch`, `Hand Pressure`]

* 👍**EgoHOS(ECCV2022)** Fine-Grained Egocentric Hand-Object Segmentation: Dataset, Model, and Applications [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19818-2_8)][[project link](https://www.seas.upenn.edu/~shzhou2/projects/eos_dataset/)][[code|official](https://github.com/owenzlz/EgoHOS)][`Hand-Object Segmentation`]

* **SOS(ECCV2022)** SOS! Self-supervised Learning over Sets of Handled Objects in Egocentric Action Recognition [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19778-9_35)][[arxiv link](https://arxiv.org/abs/2204.04796)][`Self-Supervised Learning Over Sets (SOS)`]

* **VISOR(NIPS2022)** EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations [[paper link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/590a7ebe0da1f262c80d0188f5c4c222-Abstract-Datasets_and_Benchmarks.html)][[project link](https://epic-kitchens.github.io/VISOR/)][`EPIC-KITCHENS`, a new set of `challenges` not encountered in current `video segmentation datasets`]

* 👍**HOIG or HOGAN(NIPS2022 spotlight)** Hand-Object Interaction Image Generation [[openreview link](https://openreview.net/forum?id=DDEwoD608_l)][[arxiv link](https://arxiv.org/abs/2211.15663)][[project link](https://play-with-hoi-generation.github.io/)][[code|official](https://github.com/play-with-HOI-generation/HOIG)]

* 👍**SCR(CVPR2022)** Stability-Driven Contact Reconstruction From Monocular Color Images [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Stability-Driven_Contact_Reconstruction_From_Monocular_Color_Images_CVPR_2022_paper.html)][[project link](https://www.yangangwang.com/papers/ZZM-SCR-2022-03.html)][[Corresponding Author](https://www.yangangwang.com/)][`Southeast University`, `CBF dataset` (hand-object Contact with Balancing Force recording, version 0.1)]

* **OakInk(CVPR2022)(arxiv2022.03)** OakInk: A Large-scale Knowledge Repository for Understanding Hand-Object Interaction [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_OakInk_A_Large-Scale_Knowledge_Repository_for_Understanding_Hand-Object_Interaction_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2203.15709)][[project link](https://oakink.net/)][[code|official](https://github.com/oakink/OakInk)][`SJTU`]

* **ARCTIC(CVPR2023)** ARCTIC: A Dataset for Dexterous Bimanual Hand-Object Manipulation [[arxiv link](https://arxiv.org/abs/2204.13662)][[project link](https://arctic.is.tue.mpg.de/)][[code|official](https://github.com/zc-alexfan/arctic)][`MPII`, `Hand-Object Manipulation`]

* **RUFormer(ICCV2023)** Nonrigid Object Contact Estimation With Regional Unwrapping Transformer [[paper link]()][[arxiv link](https://arxiv.org/abs/2308.14074)][`Southeast University`, `Nonrigid Object Hand Contact`]

* **HO-NeRF(ICCV2023)** Novel-view Synthesis and Pose Estimation for Hand-Object Interaction from Sparse Views [[paper link]()][[arxiv link](https://arxiv.org/abs/2308.11198)][[project link](https://iscas3dv.github.io/HO-NeRF/)][`Hand-Object Interaction`, `NeRF`]

* **USST(ICCV2023)** Uncertainty-aware State Space Transformer for Egocentric 3D Hand Trajectory Forecasting [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Bao_Uncertainty-aware_State_Space_Transformer_for_Egocentric_3D_Hand_Trajectory_Forecasting_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2307.08243)][[project link](https://actionlab-cv.github.io/EgoHandTrajPred/)][[code|official](https://github.com/oppo-us-research/USST)][`3D Ego-Hand Trajectory Forecasting`]

* **Hands23(NIPS2023)** Towards A Richer 2D Understanding of Hands at Scale [[paper link](https://papers.nips.cc/paper_files/paper/2023/hash/612a7948f3294a02a63d970566ca8536-Abstract-Conference.html)][[openreview link](https://openreview.net/forum?id=6ldTxwhgtP)][`University of Michigan + Addis Ababa University + New York University`]

* **HOIDiffusion(CVPR2024)(arxiv2024.03)** HOIDiffusion: Generating Realistic 3D Hand-Object Interaction Data [[arxiv link](https://arxiv.org/abs/2403.12011)][[project link](https://mq-zhang1.github.io/HOIDiffusion/)][`UC San Diego + HKUST`; [`Xiaolong Wang`](https://xiaolonw.github.io/) group]

* 👍**GeneOH-Diffusion(ICLR2024)(arxiv2024.02)** GeneOH Diffusion: Towards Generalizable Hand-Object Interaction Denoising via Denoising Diffusion [[openreview link](https://openreview.net/forum?id=FvK2noilxT)][[arxiv link](https://arxiv.org/abs/2402.14810)][[project link](https://meowuu7.github.io/GeneOH-Diffusion/)][[code|official](https://github.com/Meowuu7/GeneOH-Diffusion)][[blog|weixin](https://mp.weixin.qq.com/s/9LOUNGHYCSuHk-bTq1veUQ)][`THU + Shanghai AI Lab + Shanghai Qi Zhi`]

* **HOISDF(CVPR2024)(arxiv2024.02)** HOISDF: Constraining 3D Hand-Object Pose Estimation with Global Signed Distance Fields [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Qi_HOISDF_Constraining_3D_Hand-Object_Pose_Estimation_with_Global_Signed_Distance_CVPR_2024_paper.html)][[arxiv link](http://arxiv.org/abs/2402.17062)][[project link](https://amathislab.github.io/HOISDF/)][[code|official](https://github.com/amathislab/HOISDF)][`EPFL`][It achieved state-of-the-art results on the `DexYCB` and `HO3Dv2` datasets]

* **DiffH2O(arxiv2024.03)** DiffH2O: Diffusion-Based Synthesis of Hand-Object Interactions from Textual Descriptions [[arxiv link](https://arxiv.org/abs/2403.17827)][[project link](https://diffh2o.github.io/)][`Meta, Switzerland + ETH`]

* **MCC-HO(MCC-Hand-Object)(arxiv2024.04)** Reconstructing Hand-Held Objects in 3D [[arxiv link](https://arxiv.org/abs/2404.06507)][[project link](https://janehwu.github.io/mcc-ho/)][[code|official](https://github.com/janehwu/mcc-ho)][`University of California, Berkeley + University of Texas at Austin + California Institute of Technology`][It jointly `reconstructs hand and object geometry` given `a single RGB image` and `inferred 3D hand` as inputs.][The proposed alignment method `Retrieval-Augmented Reconstruction (RAR)` can be used to `automatically obtain 3D labels` for in-the-wild images of `hand-object interactions`.]

* **HO-Cap(arxiv2024.06)** HO-Cap: A Capture System and Dataset for 3D Reconstruction and Pose Tracking of Hand-Object Interaction [[arxiv link](https://arxiv.org/abs/2406.06843)][[project link](https://irvlutd.github.io/HOCap/)][[dataset link](https://utdallas.box.com/v/hocap-dataset-release)][[code|official](https://github.com/IRVLUTD/HO-Cap)][`University of Texas at Dallas + NVIDIA`][It can be used to study `3D reconstruction and pose tracking of hands and objects in videos`.][It also can be used as `human demonstrations` for `embodied AI and robot manipulation` research.]



### ▶ Dexterous Hand Grasp/Manipulation

* **DAPG(RSS2018)(arxiv2017.09)** Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations [[paper link](https://www.roboticsproceedings.org/rss14/p49.pdf)][[arxiv link](https://arxiv.org/abs/1709.10087)][[project link](https://sites.google.com/view/deeprl-dexterous-manipulation)][[code|official](github.com/aravindr93/hand_dapg)][DAPG for `Dexterous Hand Manipulation`]

* **(ICCV2021)** Toward Human-Like Grasp: Dexterous Grasping via Semantic Representation of Object-Hand [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Toward_Human-Like_Grasp_Dexterous_Grasping_via_Semantic_Representation_of_Object-Hand_ICCV_2021_paper.html)][`Dalian University of Technology`]

* **DexMV(ECCV2022)(arxiv2021.08)** DexMV: Imitation Learning for Dexterous Manipulation from Human Videos [[paper link]()][[arxiv link](https://arxiv.org/abs/2108.05877)][[project link](https://yzqin.github.io/dexmv/)][[code|official](https://github.com/yzqin/dexmv-sim)][`UC San Diego`; related to [`Xiaolong Wang`](https://xiaolonw.github.io/) group]

* **VRL3(NIPS2022)(arxiv2022.02)** VRL3: A Data-Driven Framework for Visual Deep Reinforcement Learning [[paper link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d4cc7a2d0d70736e29a3b48c3729bc06-Abstract-Conference.html)][[arxiv link](https://arxiv.org/abs/2202.10324)][[project link](https://sites.google.com/nyu.edu/vrl3)][[code|official](https://github.com/microsoft/VRL3)][`NYU + NYU Shanghai + Microsoft Research Asia`]

* **HORA(CoRL2022)(arxiv2022.10)** In-Hand Object Rotation via Rapid Motor Adaptation  [[openreview link](https://openreview.net/forum?id=Xux9gSS7WE0)][[paper link](https://proceedings.mlr.press/v205/qi23a.html)][[arxiv link](https://arxiv.org/abs/2210.04887)][[project link](https://haozhi.io/hora/)][[code|official](https://github.com/HaozhiQi/hora)][`UC Berkeley + Meta AI`]

* **ImplicitAugmentation(CoRL2022)(arxiv2022.10)** Learning Robust Real-World Dexterous Grasping Policies via Implicit Shape Augmentation [[paper link](https://proceedings.mlr.press/v205/chen23b.html)][[arxiv link](https://arxiv.org/abs/2210.13638)][[project link](https://sites.google.com/view/implicitaugmentation/home)][`University of Washington + NVIDIA`]

* **DexPoint(CoRL2022)(arxiv2022.11)** DexPoint: Generalizable Point Cloud Reinforcement Learning for Sim-to-Real Dexterous Manipulation [[paper link](https://proceedings.mlr.press/v205/qin23a.html)][[arxiv link](https://arxiv.org/abs/2211.09423)][[project link](https://yzqin.github.io/dexpoint)][[code|official](https://github.com/yzqin/dexpoint_sim)][`UC San Diego + HKUST`; related to [`Xiaolong Wang`](https://xiaolonw.github.io/) group]

* **VideoDex(CoRL2022)(arxiv2022.12)** VideoDex: Learning Dexterity from Internet Videos [[paper link](https://proceedings.mlr.press/v205/shaw23a.html)][[arxiv link](https://arxiv.org/abs/2212.04498)][[openreview link](https://openreview.net/forum?id=qUhkhHw8Dz)][[project link](https://video-dex.github.io/)][`CMU`, The journal verison in [IJRR2024](https://journals.sagepub.com/doi/abs/10.1177/02783649241227559)]

* **AnyGrasp(TRO2023)(arxiv2022.12)** AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains [[paper link](https://ieeexplore.ieee.org/abstract/document/10167687)][[arxiv link](https://arxiv.org/abs/2212.08333)][[project link](https://graspnet.net/anygrasp.html)][[code|official](https://github.com/graspnet/anygrasp_sdk)][`SJTU, Cuwu Lu`]

* **DexGraspNet(ICRA2023 Outstanding Manipulation Paper Award Finalist)(arxiv2022.10)** DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation [[paper link](https://ieeexplore.ieee.org/abstract/document/10160982)][[arxiv link](https://arxiv.org/abs/2210.02697)][[project link](https://pku-epic.github.io/DexGraspNet/)][[code|official](https://github.com/PKU-EPIC/DexGraspNet)][[dataset](https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/)][`PKU`; related to [`He Wang`](https://hughw19.github.io/) group]

* **DexDeform(ICLR2023)(arxiv2023.04)** DexDeform: Dexterous Deformable Object Manipulation with Human Demonstrations and Differentiable Physics [[openreview link](https://openreview.net/forum?id=LIV7-_7pYPl)][[arxiv link](https://arxiv.org/abs/2304.03223)][[code|official](https://github.com/sizhe-li/DexDeform)][`MIT + UC San Diego + Tsinghua University + Shanghai Qi Zhi`]

* **DexArt(CVPR2023)(arxiv2023.05)** DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Bao_DexArt_Benchmarking_Generalizable_Dexterous_Manipulation_With_Articulated_Objects_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2305.05706)][[project link](https://www.chenbao.tech/dexart/)][[code|official](https://github.com/Kami-code/dexart-release)][`SJTU` + `THU`; related to [`Xiaolong Wang`](https://xiaolonw.github.io/) group][It proposes a new benchmark for `Dexterous manipulation with Articulated objects (DexArt)` in a `physical simulator`]

* 👍**UniDexGrasp(CVPR2023)(arxiv2023.03)** UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy [[paper link](http://openaccess.thecvf.com/content/CVPR2023/html/Xu_UniDexGrasp_Universal_Robotic_Dexterous_Grasping_via_Learning_Diverse_Proposal_Generation_CVPR_2023_paper.html)][[arxiv link](http://arxiv.org/abs/2303.00938)][[project link](https://pku-epic.github.io/UniDexGrasp/)][[code|official](https://github.com/PKU-EPIC/UniDexGrasp)][[dataset|DFCData](https://mirrors.pku.edu.cn/dl-release/UniDexGrasp_CVPR2023)][`PKU`; related to [`He Wang`](https://hughw19.github.io/) group][it executes the grasp step in the `Isaac Gym physics simulator`][The first work that can demonstrate `universal` and `diverse dexterous grasping` that can well `generalize` to unseen objects.]

* 👍**UniDexGrasp++(ICCV2023 Best Paper Finalist)(arxiv2023.04)** UniDexGrasp++: Improving Dexterous Grasping Policy Learning via Geometry-aware Curriculum and Iterative Generalist-Specialist Learning [[paper link](http://openaccess.thecvf.com/content/ICCV2023/html/Wan_UniDexGrasp_Improving_Dexterous_Grasping_Policy_Learning_via_Geometry-Aware_Curriculum_and_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2304.00464)][[project link](https://pku-epic.github.io/UniDexGrasp++/)][[code|official](https://github.com/PKU-EPIC/UniDexGrasp2)][`PKU`; related to [`He Wang`](https://hughw19.github.io/) group][Its proposed `iGSL` is largely based on the [`Generalist-Specialist Learning (GSL)(ICML2022)`](https://proceedings.mlr.press/v162/jia22a.html); using the `Allegro Hand` to conduct their real robot experiments.]

* **GraspGF(NIPS2023)(arxiv2023.09)** GraspGF: Learning Score-based Grasping Primitive for Human-assisting Dexterous Grasping [[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/464012c83279e19be4cd42c25f341c92-Abstract-Conference.html)][[arxiv link](https://openreview.net/forum?id=fwvfxDbUFw)][[arxiv link](https://arxiv.org/abs/2309.06038)][[project link](https://sites.google.com/view/graspgf)][[code|official](https://github.com/tianhaowuhz/human-assisting-dex-grasp/)][`PKU + Beijing Academy of Artificial Intelligence`]

* **H-InDex(NIPS2023)(arxiv2023.10)** H-InDex: Visual Reinforcement Learning with
Hand-Informed Representations for Dexterous Manipulation [[openreview link](https://openreview.net/forum?id=lvvaNwnP6M)][[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/eb4b1f7feadcd124a59de6ff7b9196f3-Abstract-Conference.html)][[arxiv link](https://arxiv.org/abs/2310.01404)][[project link](https://yanjieze.com/H-InDex/)][[code|official](https://github.com/YanjieZe/H-InDex)][`Shanghai Qi Zhi + SJTU + THU + Renmin University of China + CMU`]

* **KODex(CoRL2023 Oral)(arxiv2023.03)** On the Utility of Koopman Operator Theory in Learning Dexterous Manipulation Skills [[paper link](https://proceedings.mlr.press/v229/han23a.html)][[openreview link](https://openreview.net/forum?id=pw-OTIYrGa)][[arxiv link](https://arxiv.org/abs/2303.13446)][[project link](https://sites.google.com/view/kodex-corl)][[code|official](https://github.com/GT-STAR-Lab/KODex)][`Georgia Institute of Technology`]

* 👍**SeqDex(CoRL2023)(arxiv2023.09)** Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon Manipulation [[paper link](https://proceedings.mlr.press/v229/chen23e.html)][[arxiv link](https://arxiv.org/abs/2309.00987)][[project link](https://sequential-dexterity.github.io/)][[code|official](https://github.com/sequential-dexterity/SeqDex)][`Stanford University`, `Fei-Fei Li`; using the `Allegro Hand` to conduct their real robot experiments.]

* **DexFunc(CoRL2023)(arxiv2023.12)** Dexterous Functional Grasping [[paper link](https://proceedings.mlr.press/v229/agarwal23a.html)][[openreview link](https://openreview.net/forum?id=93qz1k6_6h)][[arxiv link](https://arxiv.org/abs/2312.02975)][[project link](https://dexfunc.github.io/)][`CMU`; the real robot experiments are based on their self-designed [`LEAP Hand`](https://leaphand.com/) which is proposed and presented in [`(RSS2023) LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning`](https://leap-hand.github.io/) ]

* 👍**AnyTeleop(RSS2023)(arxiv2023.07)** AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System [[paper link](https://roboticsconference.org/program/papers/015/)][[arxiv link](https://arxiv.org/abs/2307.04577)][[](https://yzqin.github.io/anyteleop/)][[code|official](https://github.com/dexsuite/dex-retargeting)][`UC San Diego + NVIDIA`; related to [`Xiaolong Wang`](https://xiaolonw.github.io/) group][This work can be used for `dex-retargeting`]

* **MultiGrasp(R-AL2024)(arxiv2023.10)** Grasp Multiple Objects with One Hand [[paper link](https://ieeexplore.ieee.org/abstract/document/10460998)][[arxiv link](https://arxiv.org/abs/2310.15599)][[project link](https://multigrasp.github.io/)][[code|official](https://github.com/MultiGrasp/MultiGrasp)][`BIGAI + THU + PKU`; using the `Shadow Hand` to conduct their real robot experiments.]

* **ArtiGrasp(3DV2024 Spotlight)(arxiv2023.09)** ArtiGrasp: Physically Plausible Synthesis of Bi-Manual Dexterous Grasping and Articulation [[arxiv link](https://arxiv.org/abs/2309.03891)][[project link](https://eth-ait.github.io/artigrasp/)][[code|official](https://github.com/zdchan/artigrasp)][`ETH Zurich + MPII Germany`; they did not conduct real robot experiments.]

* **CyberDemo(CVPR2024)(arxiv2024.02)** CyberDemo: Augmenting Simulated Human Demonstration for Real-World Dexterous Manipulation [[arxiv link](https://arxiv.org/abs/2402.14795)][[project link](https://cyber-demo.github.io/)][`UC San Diego + USC`; related to [`Xiaolong Wang`](https://xiaolonw.github.io/) group; using the `Allegro Hand` to conduct their real robot experiments.]

* 👍**OAKINK2(CVPR2024)(arxiv2024.03)** OAKINK2: A Dataset of Bimanual Hands-Object Manipulation in Complex Task Completion [[arxiv link](https://arxiv.org/abs/2403.19417)][[project link](https://oakink.net/v2/)][`SJTU`][They propose a task-oriented framework, `CTC`, for `complex task and motion planning`. CTC consists of a `LLM-based task interpreter` for Complex Task `decomposition` and a `diffusion-based motion generator` for Primitive `fulfillment`.][The authors expect OAKINK2 to support `large-scale language-manipulation pre-training`, improving the performance of `multi-modal` (e.g. `vision-language-action`) models for `Complex Task Completion`.]

* **(TASE2024)** LLM-Enabled Incremental Learning Framework for Hand Exoskeleton Control [[paper link](https://ieeexplore.ieee.org/abstract/document/10489910)][`Chinese Academy of Sciences`, 'IEEE Transactions on Automation Science and Engineering']

* **(TRO2024)** Learning Human-like Functional Grasping for Multi-finger Hands from Few Demonstrations [[paper link](https://ieeexplore.ieee.org/abstract/document/10577462)][[project link](https://v-wewei.github.io/sr_dexgrasp/)][`Chinese Academy of Sciences`]

* **GeneralFlow(arxiv2024.01)** General Flow as Foundation Affordance for Scalable Robot Learning [[arxiv link](https://arxiv.org/abs/2401.11439)][[project link](https://general-flow.github.io/)][[code|official](https://github.com/michaelyuancb/general_flow)][`THU + Shanghai AI Lab + Shanghai Qi Zhi`; using the traditional `parallel grippers`]

* **RealDex(arxiv2024.02)** RealDex: Towards Human-like Grasping for Robotic Dexterous Hand [[arxiv link](https://arxiv.org/abs/2402.13853)][`ShanghaiTech University + The University of Hong Kong`; using the `Shadow Hand` to conduct their real robot experiments and build the proposed dataset.]

* **UniDexFPM(arxiv2024.03)** Dexterous Functional Pre-Grasp Manipulation with Diffusion Policy [[arxiv link](https://arxiv.org/abs/2403.12421)][[project link](https://unidexfpm.github.io/)][`PKU`; It created a simulation environment based on `Isaac Gym` using `Shadow Hand` and `UR10e robots`.]

* **ShapeGrasp(arxiv2024.03)** ShapeGrasp: Zero-Shot Task-Oriented Grasping with Large Language Models through Geometric Decomposition [[arxiv link](https://arxiv.org/abs/2403.18062)][[project link](https://shapegrasp.github.io/)][`CMU`; All of the experiments are conducted with a `Kinova Jaco robotic arm` equipped with a `three-finger gripper`, coupled with a fixed `Oak-D SR passive stereo-depth camera` for `RGB` and `depth` perception.]

* 👍**DexCap(RSS2024)(arxiv2024.03)** DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation [[arxiv link](https://arxiv.org/abs/2403.07788)][[project link](https://dex-cap.github.io/)][[code|official](https://github.com/j96w/DexCap)][`Stanford`; `Li Fei-Fei`][It is a `portable hand motion capture system`, alongside `DexIL`, a novel imitation algorithm for training `dexterous robot skills` directly from `human hand mocap data`.][It showcases the system's capability to `effectively learn from in-the-wild mocap data`, paving the way for future `data collection` methods for `dexterous manipulation`.]

* **GraspXL(ECCV2024)(arxiv2024.03)** GraspXL: Generating Grasping Motions for Diverse Objects at Scale [[arxiv link](https://arxiv.org/abs/2403.19649)][[project link](https://eth-ait.github.io/graspxl/)][[code|official](https://github.com/zdchan/graspxl)][`ETH Zurich + MPII Germany`; they did not conduct real robot experiments.]

* **CIMER(arxiv2024.04)(Under review by RA-L)** Learning Prehensile Dexterity by Imitating and Emulating State-only Observations [[arxiv link](https://arxiv.org/abs/2404.05582)][[project link](https://sites.google.com/view/cimer-2024/)][[code|official](https://github.com/GT-STAR-Lab/CIMER)][`Georgia Institute of Technology`; based on their previous work `KODex`; they did not conduct real robot experiments.]

* **Bi-VLA(arxiv2024.05)** Bi-VLA: Vision-Language-Action Model-Based System for Bimanual Robotic Dexterous Manipulations [[arxiv link](https://arxiv.org/abs/2405.06039)][`Intelligent Space Robotics Laboratory, Moscow, Russia`]

* **penspin(arxiv2024.07)** Lessons from Learning to Spin "Pens" [[arxiv link](https://arxiv.org/abs/2407.18902)][[project link](https://penspin.github.io/)][[code|official](https://github.com/HaozhiQi/penspin)][`UC San Diego, + Carnegie Mellon University + UC Berkeley`; `Xiaolong Wang`]

* **VoxAct-B(arxiv2024.07)** VoxAct-B: Voxel-Based Acting and Stabilizing Policy for Bimanual Manipulation [[arxiv link](https://arxiv.org/abs/2407.04152)][[project link](https://voxact-b.github.io/)][`University of Southern California`]

* **ACE(arxiv2024.08)** ACE: A Cross-Platform Visual-Exoskeletons System for Low-Cost Dexterous Teleoperation [[arxiv link](https://arxiv.org/abs/2408.11805)][[project link](https://ace-teleop.github.io/)][[code|official](https://github.com/ACETeleop/ACETeleop)][`UC San Diego`; `Xiaolong Wang`]

* **** [[paper link]()][[arxiv link]()][[project link]()][[code|official]()]
