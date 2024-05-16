# ⭐Dexterous Hand Grasp
*also closely related to `contact-rich articulated object manipulation`*

## Softwares / Hardwares

* [**ShadowHand** Shadow Dexterous Hand Series - Research and Development Tool](https://www.shadowrobot.com/dexterous-hand-series/) [As a widely used five-finger robotic dexterous hand, `ShadowHand` amounts to `26 degrees of freedom (DoF)`, in contrast with `7 DoF` for a typical `parallel gripper`. Such high dimensionality magnifies the difficulty in both generating valid grasp poses and planning the execution trajectories, and thus distinguishes the dexterous grasping task from its counterpart for parallel grippers.]

* [**Isaac Gym** provides Benchmark Environments and the corresponding simulator](https://developer.nvidia.com/isaac-gym) [[Technical Paper](https://arxiv.org/abs/2108.10470)][[github link](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)][openreview link (NIPS2021 Track Datasets and Benchmarks Round2)](https://openreview.net/forum?id=fgFBtYgJQX_)][By `NVIDIA`]


## Materials

* 👍 **(github)(Hand3DResearch) Recent Progress in 3D Hand Tasks** [[github link](https://github.com/SeanChenxy/Hand3DResearch)]
* **(Website) GraspNet通用物体抓取(GraspNet-1Billion + AnyGrasp + SuctionNet-1Billion + TransCG)**  [[Homepage link](https://graspnet.net/index.html)]

## Datasets

* **DFCData (UniDexGrasp CVPR2023)** [UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy](https://mirrors.pku.edu.cn/dl-release/) [[project link](https://pku-epic.github.io/UniDexGrasp/)][`PKU`]

***

## Papers

* **OakInk(CVPR2022)(arxiv2022.03)** OakInk: A Large-scale Knowledge Repository for Understanding Hand-Object Interaction [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_OakInk_A_Large-Scale_Knowledge_Repository_for_Understanding_Hand-Object_Interaction_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2203.15709)][[project link](https://oakink.net/)][[code|official](https://github.com/oakink/OakInk)][`SJTU`]

* **ImplicitAugmentation(CoRL2022)(arxiv2022.10)** Learning Robust Real-World Dexterous Grasping Policies via Implicit Shape Augmentation [[paper link](https://proceedings.mlr.press/v205/chen23b.html)][[arxiv link](https://arxiv.org/abs/2210.13638)][[project link](https://sites.google.com/view/implicitaugmentation/home)][`University of Washington + NVIDIA`]

* **DexPoint(CoRL2022)(arxiv2022.11)** DexPoint: Generalizable Point Cloud Reinforcement Learning for Sim-to-Real Dexterous Manipulation [[paper link](https://proceedings.mlr.press/v205/qin23a.html)][[arxiv link](https://arxiv.org/abs/2211.09423)][[project link](https://yzqin.github.io/dexpoint)][[code|official](https://github.com/yzqin/dexpoint_sim)][`UC San Diego + HKUST`]

* **VideoDex(CoRL2022)(arxiv2022.12)** VideoDex: Learning Dexterity from Internet Videos [[paper link](https://proceedings.mlr.press/v205/shaw23a.html)][[arxiv link](https://arxiv.org/abs/2212.04498)][[openreview link](https://openreview.net/forum?id=qUhkhHw8Dz)][[project link](https://video-dex.github.io/)][`CMU`, The journal verison in [IJRR2024](https://journals.sagepub.com/doi/abs/10.1177/02783649241227559)]

* 👍**UniDexGrasp(CVPR2023)(arxiv2023.03)** UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy [[paper link](http://openaccess.thecvf.com/content/CVPR2023/html/Xu_UniDexGrasp_Universal_Robotic_Dexterous_Grasping_via_Learning_Diverse_Proposal_Generation_CVPR_2023_paper.html)][[arxiv link](http://arxiv.org/abs/2303.00938)][[project link](https://pku-epic.github.io/UniDexGrasp/)][[code|official](https://github.com/PKU-EPIC/UniDexGrasp)][[dataset|DFCData](https://mirrors.pku.edu.cn/dl-release/UniDexGrasp_CVPR2023)][`PKU`][it executes the grasp step in the `Isaac Gym physics simulator`][The first work that can demonstrate `universal` and `diverse dexterous grasping` that can well `generalize` to unseen objects.]

* 👍**UniDexGrasp++(ICCV2023 Best Paper Finalist)(arxiv2023.04)** UniDexGrasp++: Improving Dexterous Grasping Policy Learning via Geometry-aware Curriculum and Iterative Generalist-Specialist Learning [[paper link](http://openaccess.thecvf.com/content/ICCV2023/html/Wan_UniDexGrasp_Improving_Dexterous_Grasping_Policy_Learning_via_Geometry-Aware_Curriculum_and_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2304.00464)][[project link](https://pku-epic.github.io/UniDexGrasp++/)][[code|official](https://github.com/PKU-EPIC/UniDexGrasp2)][`PKU`]

* **GraspGF(NIPS2023)(arxiv2023.09)** GraspGF: Learning Score-based Grasping Primitive for Human-assisting Dexterous Grasping [[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/464012c83279e19be4cd42c25f341c92-Abstract-Conference.html)][[arxiv link](https://openreview.net/forum?id=fwvfxDbUFw)][[arxiv link](https://arxiv.org/abs/2309.06038)][[project link](https://sites.google.com/view/graspgf)][[code|official](https://github.com/tianhaowuhz/human-assisting-dex-grasp/)][`PKU + Beijing Academy of Artificial Intelligence`]

* **KODex(CoRL2023 Oral)(arxiv2023.03)** On the Utility of Koopman Operator Theory in Learning Dexterous Manipulation Skills [[paper link](https://proceedings.mlr.press/v229/han23a.html)][[openreview link](https://openreview.net/forum?id=pw-OTIYrGa)][[arxiv link](https://arxiv.org/abs/2303.13446)][[project link](https://sites.google.com/view/kodex-corl)][[code|official](https://github.com/GT-STAR-Lab/KODex)][`Georgia Institute of Technology`]

* 👍**SeqDex(CoRL2023)(arxiv2023.09)** Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon Manipulation [[paper link](https://proceedings.mlr.press/v229/chen23e.html)][[arxiv link](https://arxiv.org/abs/2309.00987)][[project link](https://sequential-dexterity.github.io/)][[code|official](https://github.com/sequential-dexterity/SeqDex)][`Stanford University`, `Fei-Fei Li`]

* **DexFunc(CoRL2023)(arxiv2023.12)** Dexterous Functional Grasping [[paper link](https://proceedings.mlr.press/v229/agarwal23a.html)][[openreview link](https://openreview.net/forum?id=93qz1k6_6h)][[arxiv link](https://arxiv.org/abs/2312.02975)][[project link](https://dexfunc.github.io/)][`CMU`]

* **MultiGrasp(R-AL2024)(arxiv2023.10)** Grasp Multiple Objects with One Hand [[paper link](https://ieeexplore.ieee.org/abstract/document/10460998)][[arxiv link](https://arxiv.org/abs/2310.15599)][[project link](https://multigrasp.github.io/)][[code|official](https://github.com/MultiGrasp/MultiGrasp)][`BIGAI + THU + PKU`]

* **ArtiGrasp(3DV2024)(arxiv2023.09)** ArtiGrasp: Physically Plausible Synthesis of Bi-Manual Dexterous Grasping and Articulation [[arxiv link](https://arxiv.org/abs/2309.03891)][[project link](https://eth-ait.github.io/artigrasp/)][[code|official](https://github.com/zdchan/artigrasp)][`ETH Zurich + MPII Germany`]

* 👍**OAKINK2(CVPR2024)(arxiv2024.03)** OAKINK2: A Dataset of Bimanual Hands-Object Manipulation in Complex Task Completion [[arxiv link](https://arxiv.org/abs/2403.19417)][[project link](https://oakink.net/v2/)][`SJTU`]

* **CyberDemo(CVPR2024)(arxiv2024.02)** CyberDemo: Augmenting Simulated Human Demonstration for Real-World Dexterous Manipulation [[arxiv link](https://arxiv.org/abs/2402.14795)][[project link](https://cyber-demo.github.io/)][`UC San Diego + USC`]

* **GeneralFlow(arxiv2024.01)** General Flow as Foundation Affordance for Scalable Robot Learning [[arxiv link](https://arxiv.org/abs/2401.11439)][[project link](https://general-flow.github.io/)][[code|official](https://github.com/michaelyuancb/general_flow)][`THU + Shanghai AI Lab + Shanghai Qi Zhi`]

* **RealDex(arxiv2024.02)** RealDex: Towards Human-like Grasping for Robotic Dexterous Hand [[arxiv link](https://arxiv.org/abs/2402.13853)][`ShanghaiTech University + The University of Hong Kong`]

* **UniDexFPM(arxiv2024.03)** Dexterous Functional Pre-Grasp Manipulation with Diffusion Policy [[arxiv link](https://arxiv.org/abs/2403.12421)][[project link](https://unidexfpm.github.io/)][`PKU`]

* **ShapeGrasp(arxiv2024.03)** ShapeGrasp: Zero-Shot Task-Oriented Grasping with Large Language Models through Geometric Decomposition [[arxiv link](https://arxiv.org/abs/2403.18062)][[project link](https://shapegrasp.github.io/)][`CMU`]

* **GraspXL(arxiv2024.03)** GraspXL: Generating Grasping Motions for Diverse Objects at Scale [[arxiv link](https://arxiv.org/abs/2403.19649)][[project link](https://eth-ait.github.io/graspxl/)][[code|official](https://github.com/zdchan/graspxl)][`ETH Zurich + MPII Germany`]

* **CIMER(arxiv2024.04)(Under review by RA-L)** Learning Prehensile Dexterity by Imitating and Emulating State-only Observations [[arxiv link](https://arxiv.org/abs/2404.05582)][[project link](https://sites.google.com/view/cimer-2024/)][[code|official](https://github.com/GT-STAR-Lab/CIMER)][`Georgia Institute of Technology`; based on their previous work `KODex`]

* **** [[paper link]()][[arxiv link]()][[project link]()][[code|official]()]
