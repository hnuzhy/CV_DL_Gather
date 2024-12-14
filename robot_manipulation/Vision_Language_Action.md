# ⭐Vision Language Action
*this is the most popular paradigm for achieving `robot manipulation`, also similar to `image-to-action policy models,` and `state-to-action mappings`*

## ▶Materials

### ※ Useful Collections
* **Github** [Recent LLM-based CV and related works.](https://github.com/DirtyHarryLYL/LLM-in-Vision)
* **Github** [Must-read Papers on Large Language Model(LLM) Agents.](https://github.com/zjunlp/LLMAgentPapers)
* **Github** [Awesome-Robotics-Foundation-Models](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models) [[the survey paper](https://arxiv.org/abs/2312.07843)]
* **Github** [Survey Paper of foundation models for robotics](https://github.com/JeffreyYH/robotics-fm-surve) [[the survey paper](https://arxiv.org/abs/2312.08782)]
* **Github** [Robotic Grasping Papers and Codes - Grasp Detection](https://github.com/rhett-chen/Robotic-grasping-papers?tab=readme-ov-file#3-grasp-detection)
* **Github** [CV & Geometry-based 6DOF Robotic Grasping - 6D Grasp Pose Detection](https://github.com/kidpaul94/My-Robotic-Grasping?tab=readme-ov-file#6d-grasp-pose-detection)
* **Github** [Diffusion-Literature-for-Robotics - Summary of key papers and blogs](https://github.com/mbreuss/diffusion-literature-for-robotics)
* **Github** [Awesome-Touch - Tactile Sensor and Simulator; Visual Tactile Manipulation; Open Source.](https://github.com/linchangyi1/Awesome-Touch)

### ※ Representative Blogs
* **website (CCF)** [具身智能 | CCF专家谈术语 (Cewu Lu)](https://www.ccf.org.cn/Media_list/gzwyh/jsjsysdwyh/2023-07-22/794317.shtml)
* **website** [GraspNet通用物体抓取(GraspNet-1Billion + AnyGrasp + SuctionNet-1Billion + TransCG)](https://graspnet.net/index.html)

### ※ Simulator Toolkits
* **Gensim** [Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora.](https://radimrehurek.com/gensim/) [[github](https://github.com/piskvorky/gensim)]
* **Gym** [Gym is a standard API for reinforcement learning, and a diverse collection of reference environments](https://www.gymlibrary.dev/) [[github](https://github.com/openai/gym)]
* 👍👍**Gymnasium** [Gymnasium is an API standard for reinforcement learning with a diverse collection of reference environments](https://gymnasium.farama.org/) [[github](https://github.com/Farama-Foundation/Gymnasium)]
* **Gymnasium-Robotics** [Gymnasium-Robotics is a collection of robotics simulation environments for Reinforcement Learning](https://robotics.farama.org/) [[github](https://github.com/Farama-Foundation/Gymnasium-Robotics)]
* 👍**ManiSkill** [SAPIEN Manipulation Skill Framework, a GPU parallelized robotics simulator and benchmark](https://github.com/haosulab/ManiSkill) [[ManiSkill readthedocs](https://maniskill.readthedocs.io/en/latest/index.html)]
* 👍**PyRep** [PyRep is a toolkit for robot learning research, built on top of CoppeliaSim (previously called V-REP).](https://github.com/stepjam/PyRep)
* **CoppeliaSim** [CoppeliaSim supports you in testing and validating complex robotics systems via algorithms prototyping, kinematic design and digital twin creation.](https://www.coppeliarobotics.com/)
* **Deoxys** [A modular, real-time controller library for Franka Emika Panda robots, aiming to facilitate a wide range of robot learning research.](https://github.com/UT-Austin-RPL/deoxys_control)
* **ManiSkill Research** [ManiSkill helps propel groundbreaking research in generalizable robotic manipulation.](https://www.maniskill.ai/research)
* **Hillbot** [Scaling Robot Foundation Models via Simulation](https://www.hillbot.ai/home)


***

## ▶Datasets

* **MetaWorld(CoRL2019)(arxiv2019.10)** Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning [[paper link](https://proceedings.mlr.press/v100/yu20a.html)][[arxiv link](https://arxiv.org/abs/1910.10897)][[project link](http://meta-world.github.io/)][[baseline method](https://github.com/rlworkgroup/garage)][[code|official](https://github.com/rlworkgroup/metaworld)][`Stanford University + UC Berkeley + Columbia University + University of Southern California + Robotics at Google`; `Chelsea Finn + Sergey Levine`]

* 👍**RLBench(RAL2020)(arxiv2019.09)** RLBench: The Robot Learning Benchmark & Learning Environment [[paper link](https://ieeexplore.ieee.org/abstract/document/9001253)][[arxiv link](https://arxiv.org/abs/1909.12271)][[project link](https://sites.google.com/view/rlbench)][[code|official](https://github.com/stepjam/RLBench)][`Dyson Robotics Lab, Imperial College London`][This dataset is based on `Coppliasim 4.1.0` and `PyRep`]

* **Ravens(TransporterNets)(CoRL2020)(arxiv2020.10)** Transporter Networks: Rearranging the Visual World for Robotic Manipulation [[paper link](https://proceedings.mlr.press/v155/zeng21a.html)][[arxiv link](https://arxiv.org/abs/2010.14406)][[project link](https://transporternets.github.io/)][[code|official](https://github.com/google-research/ravens)][`Robotics at Google`][It trained robotic agents to learn `pick` and `place` with deep learning for `vision-based manipulation` in `PyBullet`.]

* 👍**CALVIN(RAL2022)(Best Paper Award)(arxiv2021.12)** Calvin: A Benchmark for Language-conditioned Policy Learning for Long-horizon Robot Manipulation Tasks [[paper link](https://ieeexplore.ieee.org/abstract/document/9788026/)][[arxiv link](https://arxiv.org/abs/2112.03227)][[project link](http://calvin.cs.uni-freiburg.de/)][[code|official](https://github.com/mees/calvin)][`University of Freiburg, Germany`]

* **VLMbench(NIPS2022 Datasets and Benchmarks)(arxiv2022.06)** VLMbench: A Compositional Benchmark for Vision-and-Language Manipulation [[paper link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/04543a88eae2683133c1acbef5a6bf77-Abstract-Datasets_and_Benchmarks.html)][[arxiv link](https://arxiv.org/abs/2206.08522)][[project link](https://sites.google.com/ucsc.edu/vlmbench/home)][[code|official](https://github.com/eric-ai-lab/vlmbench)][`University of California + University of Michigan`, It proposed the baseline method named `6D-CLIPort`][This dataset is based on `Coppliasim 4.1.0` and `PyRep`]

* **Language-Table(RAL2023)(arxiv2022.10)** Interactive Language: Talking to Robots in Real Time [[paper link](https://ieeexplore.ieee.org/abstract/document/10182264)][[arxiv link](https://arxiv.org/abs/2210.06407)][[project link](https://interactive-language.github.io/)][[code|official](https://github.com/google-research/language-table)][`Robotics at Google`][It is a suite of `human-collected datasets` and a `multi-task continuous control benchmark` for `open vocabulary visuolinguomotor learning`.]

* **RT-1(RSS2023)(arxiv2022.12)** RT-1: Robotics Transformer for Real-World Control at Scale [[paper link](https://www.roboticsproceedings.org/rss19/p025.pdf)][[arxiv link](https://arxiv.org/abs/2212.06817)][[project link](https://robotics-transformer1.github.io/)][[released datasets](https://console.cloud.google.com/storage/browser/gresearch/rt-1-data-release)][[code|official](https://github.com/google-research/robotics_transformer)][by `Google DeepMind`]

* 👍**ARNOLD(ICCV2023)(arxiv2023.04)** ARNOLD: A Benchmark for Language-Grounded Task Learning with Continuous States in Realistic 3D Scenes [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Gong_ARNOLD_A_Benchmark_for_Language-Grounded_Task_Learning_with_Continuous_States_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2304.04321)][[project link](https://arnold-benchmark.github.io/)][[code|official](https://github.com/arnold-benchmark/arnold)][[dataset|official](https://drive.google.com/drive/folders/1yaEItqU9_MdFVQmkKA6qSvfXy_cPnKGA)][[challenges|official](https://sites.google.com/view/arnoldchallenge/)][`UCLA + PKU + THU + Columbia University + BIGAI`]

* **RH20T(CoRLW2023)(arxiv2023.07)** RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot [[paper link](https://openreview.net/forum?id=Sg9qzrodL9)][[arxiv link](https://arxiv.org/abs/2307.00595)][[project link](https://rh20t.github.io/)][`SJTU`][Its `150 skills` were either selected from `RLBench` and `MetaWorld`, or `proposed by themselves`.]

* 👍**BridgeData-V2(CoRL2023)(arxiv2023.08)** BridgeData V2: A Dataset for Robot Learning at Scale [[openreview link](https://openreview.net/forum?id=f55MlAT1Lu)][[paper link](https://proceedings.mlr.press/v229/walke23a.html)][[arxiv link](https://arxiv.org/abs/2308.12952)][[project link](https://rail-berkeley.github.io/bridgedata/)][[code|official](https://github.com/rail-berkeley/bridge_data_v2)][`UC Berkeley + Stanford + Google DeepMind + CMU`][It is based on the `(arxiv2021.09) Bridge data: Boosting generalization of robotic skills with cross-domain datasets` with [[arxiv link](https://arxiv.org/abs/2109.13396)] and [[project link](https://sites.google.com/view/bridgedata)]]

* **LoHoRavens(arxiv2023.10)** LoHoRavens: A Long-Horizon Language-Conditioned Benchmark for Robotic Tabletop Manipulation [[arxiv link](https://arxiv.org/abs/2310.12020)][[project link](https://cisnlp.github.io/lohoravens-webpage/)][[code|official](https://github.com/Shengqiang-Zhang/LoHo-Ravens)][`LMU Munich + TUM`][The code is largely based on method `CLIPort-batchify(CoRL2021)(arxiv2021.09)` and dataset `Ravens(TransporterNets)(CoRL2020)`]

* **D4PAS(arxiv2023.12)** Multi-level Reasoning for Robotic Assembly: From Sequence Inference to Contact Selection [[arxiv link](https://arxiv.org/abs/2312.10571)][`UC Berkeley`; It is a large-scale `Dataset for Part Assembly Sequences (D4PAS)`]

* **Safety-Gymnasium(NIPS2023 Datasets and Benchmarks)(arxiv2023.10)** Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark [[openreview link](https://openreview.net/forum?id=WZmlxIuIGR)][[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html)][[arxiv link](https://arxiv.org/abs/2310.12567)][[project link](https://sites.google.com/view/safety-gymnasium)][[code|official](https://github.com/PKU-Alignment/safety-gymnasium)][`PKU`, Safety-Gymnasium is a `highly scalable` and `customizable` Safe Reinforcement Learning (`SafeRL`) library.]

* 👍**Open X-Embodiment(RT-2-X)(arxiv2023.10)** Open X-Embodiment: Robotic Learning Datasets and RT-X Models [[arxiv link](https://arxiv.org/abs/2310.08864)][[project link](https://robotics-transformer-x.github.io/)][[code|official](https://github.com/google-deepmind/open_x_embodiment)][by `Google DeepMind`]

* 👍👍**FMB(IJRR2024)(arxiv2024.01)** FMB: A Functional Manipulation Benchmark for Generalizable Robotic Learning [[arxiv link](https://arxiv.org/abs/2401.08553)][[project link](https://functional-manipulation-benchmark.github.io/)][[Materials and CAD Files](https://functional-manipulation-benchmark.github.io/files/index.html)][[dataset link](https://functional-manipulation-benchmark.github.io/dataset/index.html)][[code|official](https://github.com/rail-berkeley/fmb)][`University of California, Berkeley (BAIR)`]

* 👍**DROID(arxiv2024.03)** DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset [[arxiv link](https://arxiv.org/abs/2403.12945)][[project link](https://droid-dataset.github.io/)][[dataset visualizer](https://droid-dataset.github.io/visualizer/)][[code|official](https://github.com/droid-dataset/droid_policy_learning)][`Stanford + Berkeley + Toyota` and many other universities; It used the [`diffusion policy`](https://diffusion-policy.cs.columbia.edu/) for policy learning]

* **Open6DOR(CVPRW2024 Oral)** Open6DOR: Benchmarking Open-instruction 6-DoF Object Rearrangement and A VLM-based Approach [[openreview link](https://openreview.net/forum?id=RclUiexKMt)][[project link](https://pku-epic.github.io/Open6DOR)][`PKU`, by the [`He Wang`](https://hughw19.github.io/) group][This is a work published in the `First Vision and Language for Autonomous Driving and Robotics Workshop`]

* **BRMData(arxiv2024.05)** Empowering Embodied Manipulation: A Bimanual-Mobile Robot Manipulation Dataset for Household Tasks [[arxiv link](https://arxiv.org/abs/2405.18860)][[project link](https://embodiedrobot.github.io/)][[dataset link](http://box.jd.com/sharedInfo/1147DC284DDAEE91DC759E209F58DD60)][`JD Explore Academy`][It proposed `BRMData`, a `Bimanual-mobile Robot Manipulation Dataset` specifically designed for `household applications`.]


***

## ▶Papers

### ※ 1) Survey

* **Survey(IJCAI2024)(arxiv2024.02)** A Comprehensive Survey of Cross-Domain Policy Transfer for Embodied Agents [[arxiv link](https://arxiv.org/abs/2402.04580)][[code|official](https://github.com/t6-thu/awesome-cross-domain-policy-transfer-for-embodied-agents)][`THU`]

* **Survey(arxiv2024.05)** A Survey on Vision-Language-Action Models for Embodied AI [[arxiv link](https://arxiv.org/abs/2405.14093)][`CUHK +  Huawei Noah’s Ark Lab`]

* **Survey(arxiv2024.05)** Neural Scaling Laws for Embodied AI [[arxiv link](https://arxiv.org/abs/2405.14005)][`TUM + MIT`][This paper presents the first study to `quantify scaling laws` for `Robot Foundation Models (RFMs)` and the use of `LLMs` in `robotics` tasks.]

* **** [[openreview link]()][[paper link]()][[arxiv link]()][[project link]()][[code|official]()]


### ※ 2) Robot Pose Estimation / Hand-eye Calibration
*This line of research may open the possibility of on-line hand-eye calibration, which is more robust and scalable then classic hand-eye calibration systems*

* **DREAM(ICRA2020)(arxiv2019.11)** Camera-to-Robot Pose Estimation from a Single Image [[paper link](https://ieeexplore.ieee.org/abstract/document/9196596)][[arxiv link](https://arxiv.org/abs/1911.09231)][[project link](https://research.nvidia.com/publication/2020-05_camera-robot-pose-estimation-single-image)][[code|official](https://github.com/NVlabs/DREAM)][`NVIDIA + CMU`]

* **RoMa(3DV)(arxiv2021.03)** Deep Regression on Manifolds: A 3D Rotation Case Study [[paper link](https://ieeexplore.ieee.org/abstract/document/9665892)][[arxiv link](https://arxiv.org/abs/2103.16317)][[project link](https://naver.github.io/roma/)][[code|official](https://github.com/naver/roma)]

* **RoboPose(CVPR2021 oral)(arxiv2021.04)** Single-View Robot Pose and Joint Angle Estimation via Render & Compare [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Labbe_Single-View_Robot_Pose_and_Joint_Angle_Estimation_via_Render__CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2104.09359)][[project link](https://www.di.ens.fr/willow/research/robopose/)][[code|official](https://github.com/ylabbe/robopose)][`ENS/Inria + LIGM, ENPC + CIIRC CTU`]

* **EasyHeC(RAL2023)(arxiv2023.05)** EasyHeC: Accurate and Automatic Hand-eye Calibration via Differentiable Rendering and Space Exploration [[paper link](https://ieeexplore.ieee.org/abstract/document/10251600)][[arxiv link](https://arxiv.org/abs/2305.01191)][[project link](https://ootts.github.io/easyhec/)][[code|official](https://github.com/ootts/EasyHeC)][`State Key Lab of CAD & CG, Zhejiang University + University of California, San Diego`; `Xiaowei Zhou +  Hao Su`]

* **HolisticRoboPose(ECCV2024)(arxiv2024.02)** Real-time Holistic Robot Pose Estimation with Unknown States [[arxiv link](https://arxiv.org/abs/2402.05655)][[project link](https://oliverbansk.github.io/Holistic-Robot-Pose/)][[code|official](https://github.com/Oliverbansk/Holistic-Robot-Pose-Estimation)][`Peking University + Shanghai Jiao Tong University`]

* **BoT(RSS2024 Workshop)** Body Transformer: LeveragingRobot Embodiment for Policy Learning [[openreview link](https://openreview.net/forum?id=IbXqRpANPD)][[project link](https://bodytransformer.site/)][`UC Berkeley`]

* **Kalib(arxiv2024.08)** Kalib: Markerless Hand-Eye Calibration with Keypoint Tracking [[arxiv link](https://arxiv.org/abs/2408.10562)][[project link](https://sites.google.com/view/hand-eye-kalib)][[code|official](https://github.com/robotflow-initiative/Kalib)][`SJTU`; `Cewu Lu`]



### ※ 3) Tactile Signals Sensing/Simulation

* **TACTO(RAL2022)(arxiv2020.12)** TACTO: A Fast, Flexible, and Open-Source Simulator for High-Resolution Vision-Based Tactile Sensors [[paper link](https://ieeexplore.ieee.org/abstract/document/9697425)][[arxiv link](https://arxiv.org/abs/2012.08456)][[code|official](https://github.com/facebookresearch/tacto)][`facebook`][using the `Tactile` signal as the input]

* **TactileSim(CoRL2022)** Efficient Tactile Simulation with Differentiability for Robotic Manipulation [[openreview link](https://proceedings.mlr.press/v205/xu23b.html)][[paper link](https://openreview.net/forum?id=6BIffCl6gsM)][[poster link](https://people.csail.mit.edu/jiex/papers/TactileSim/poster.pdf)][`MIT + Texas A&M University`][using the `Tactile` signal as the input]

* **RobotSynesthesia(ICRA2024)(arxiv2023.12)** Robot Synesthesia: In-Hand Manipulation with Visuotactile Sensing [[paper link](https://ieeexplore.ieee.org/abstract/document/10610532)][[arxiv link](https://arxiv.org/abs/2312.01853)][[project link](https://github.com/YingYuan0414/in-hand-rotation)][[code|official](https://github.com/YingYuan0414/in-hand-rotation)][`UC San Diego + Tsinghua University + University of Illinois Urbana-Champaign + UC Berkeley + Dongguk University`; `Xiaolong Wang`]

* 👍**ViTaM(Nature Communications 2024)** Capturing forceful interaction with deformable objects using a deep learning-powered stretchable tactile array [[paper link](https://www.nature.com/articles/s41467-024-53654-y)][[code|official](https://github.com/jeffsonyu/ViTaM)][[weixin blog](https://mp.weixin.qq.com/s/bWFFd3ZUNnGTU0Ule57CmQ)][`Cewu Lu`][`Visual-Tactile recording and tracking system for Manipulation`]

* **HATO(arxiv2024.04)** Learning Visuotactile Skills with Two Multifingered Hands [[arxiv link](https://arxiv.org/abs/2404.16823)][[paper link](https://toruowo.github.io/hato/)][[dataset link](https://berkeley.app.box.com/s/379cf57zqm1akvr00vdcloxqxi3ucb9g?sortColumn=name&sortDirection=ASC)][[code|official](https://github.com/toruowo/hato)][`UC Berkeley`][They repurpose `two prosthetic hands` with `touch sensing` for research use, develop a `bimanual multifingered hands teleoperation system` to collect `visuotactile` data, and learn cool policies.]

* **TacSL(arxiv2024.08)** TacSL: A Library for Visuotactile Sensor Simulation and Learning [[arxiv link](https://arxiv.org/abs/2408.06506)][[project link](https://iakinola23.github.io/tacsl/)][[code|official](https://github.com/isaac-sim/IsaacGymEnvs/blob/tacsl/isaacgymenvs/tacsl_sensors/install/tacsl_setup.md)][`NVIDIA Research + University of Washington`; `Dieter Fox`][using `Isaac Gym`; for `Visuotactile Sensor Simulation`]

* **TacDiffusion(arxiv2024.09)** TacDiffusion: Force-domain Diffusion Policy for Precise Tactile Manipulation [[arxiv link](https://arxiv.org/abs/2409.11047)][[code|official](https://github.com/popnut123/TacDiffusion)][`Technical University of Munich`]

* **3DTacDex(arxiv2024.09)** Canonical Representation and Force-Based Pretraining of 3D Tactile for Dexterous Visuo-Tactile Policy Learning [[arxiv link](https://arxiv.org/abs/2409.17549)][[project link](https://3dtacdex.github.io/)][`Peking University`; `Hao Dong`]



### ※ 4) Assemblely/Rearrangement Related Generation/Manipulation

* **IKEA Furniture Assembly(ICRA2021)** IKEA Furniture Assembly Environment for Long-Horizon Complex Manipulation Tasks [[paper link](https://ieeexplore.ieee.org/abstract/document/9560986/)][[arxiv link](https://arxiv.org/abs/1911.07246)][[project link](https://clvrai.github.io/furniture/)][[code|official](https://github.com/clvrai/furniture)][`Cognitive Learning for Vision and Robotics (CLVR), University of Southern California`]

* **Factory(RSS2022)(arxiv2022.05)** Factory: Fast Contact for Robotic Assembly [[arxiv link](https://arxiv.org/abs/2205.03532)][[projec link](https://sites.google.com/nvidia.com/factory)][[code|official](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/docs/factory.md)][`NVIDIA`; `Isaac Gym`]

* **LEGO-Net(CVPR2023)(arxiv2023.01)** LEGO-Net: Learning Regular Rearrangements of Objects in Rooms [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Wei_LEGO-Net_Learning_Regular_Rearrangements_of_Objects_in_Rooms_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2301.09629)][[project link](https://ivl.cs.brown.edu/research/lego-net.html)][[code|official](https://github.com/QiuhongAnnaWei/LEGO-Net)][`Brown University + Stanford University`]

* **CabiNet(ICRA2023)(arxiv2023.04)** CabiNet: Scaling Neural Collision Detection for Object Rearrangement with Procedural Scene Generation [[paper link](https://ieeexplore.ieee.org/abstract/document/10161528/)][[arxiv link](https://arxiv.org/abs/2304.09302)][[project link](https://cabinet-object-rearrangement.github.io/)][[code|official](https://github.com/NVlabs/cabi_net)][`NVIDIA`; `Dieter Fox`][`Procedural Scene Generation`]

* **IndustReal(RSS2023)(arxiv2023.05)** IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to Reality [[arxiv link](https://arxiv.org/abs/2305.17110)][[project link](https://sites.google.com/nvidia.com/industreal)][[IndustRealKit link](https://github.com/NVlabs/industrealkit)][[IndustRealSim link](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/industreal.md)][[IndustRealLib link](https://github.com/NVLabs/industreallib)][`University of Southern California + Stanford University + NVIDIA + University of Sydney + University of Washington`; `Isaac Gym`][following the previous work `Factory(RSS2022)`]

* **DiffAssemble(CVPR2024)(arxiv2024.02)** DiffAssemble: A Unified Graph-Diffusion Model for 2D and 3D Reassembly [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Scarpellini_DiffAssemble_A_Unified_Graph-Diffusion_Model_for_2D_and_3D_Reassembly_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2402.19302)][[project link](https://iit-pavis.github.io/DiffAssemble/)][[code|official](https://github.com/IIT-PAVIS/DiffAssemble)][`Pattern Analysis and Computer Vision (PAVIS)` + `Istituto Italiano di Tecnologia (IIT)`][It focused on 2D and 3D `reassembly` tasks]

* **SCANet(IROS2024, Oral)(arxiv2024.03)** SCANet: Correcting LEGO Assembly Errors with Self-Correct Assembly Network [[arxiv link](https://arxiv.org/abs/2403.18195)][[project link](https://scanet-iros2024.github.io/)][[code|official](https://github.com/Yaser-wyx/SCANet)][`Southeast University + Peking University`; `Dong Hao`]

* **ClutterGen(CoRL2024)(arxiv2024.07)** ClutterGen: A Cluttered Scene Generator for Robot Learning [[arxiv link](https://arxiv.org/abs/2407.05425)][[project link](http://www.generalroboticslab.com/blogs/blog/2024-07-06-cluttergen/index.html)][[code|official](https://github.com/generalroboticslab/ClutterGen)][`Duke University`]

* **AutoMate(arxiv2024.07)** AutoMate: Specialist and Generalist Assembly Policies over Diverse Geometries [[arxiv link](https://arxiv.org/abs/2407.08028)][[project link](https://bingjietang718.github.io/automate/)][`University of Southern California + NVIDIA Corporation + University of Washington + University of Sydney`; `Dieter Fox`][`Assembly`]

* **DegustaBot(arxiv2024.07)** DegustaBot: Zero-Shot Visual Preference Estimation for Personalized Multi-Object Rearrangement [[arxiv link](https://arxiv.org/abs/2407.08876)][`Carnegie Mellon University + Hello Robot`][`Multi-Object Rearrangement`]

* **ARCH(arxiv2024.09)** ARCH: Hierarchical Hybrid Learning for Long-Horizon Contact-Rich Robotic Assembly [[arxiv link](https://arxiv.org/abs/2409.16451)][[project link](https://long-horizon-assembly.github.io/)][`Stanford University + MIT + University of Michigan + Autodesk Research`]



### ※ 5) Simulation/Synthesis/Generation for Embodied AI

* **robosuite(2020.09)** robosuite: A Modular Simulation Framework and Benchmark for Robot Learning [[white paper](https://robosuite.ai/assets/whitepaper.pdf)][[arxiv link](https://arxiv.org/abs/2009.12293)][[project link](https://robosuite.ai/)][[documentation link](https://robosuite.ai/docs/overview.html)][[github link](https://github.com/ARISE-Initiative/robosuite)][`robosuite.ai`; robosuite is a `simulation framework` powered by the `MuJoCo physics engine` for robot learning. It also offers a suite of `benchmark environments` for reproducible research.]

* 👍**SAPIEN(CVPR2020)(arxiv2020.03)** SAPIEN: A SimulAted Part-based Interactive ENvironment [[paper link](http://openaccess.thecvf.com/content_CVPR_2020/html/Xiang_SAPIEN_A_SimulAted_Part-Based_Interactive_ENvironment_CVPR_2020_paper.html)][[arxiv link](https://arxiv.org/abs/2003.08515)][[project link](https://sapien.ucsd.edu/)][[code|official](https://github.com/haosulab/SAPIEN)][`UC San Diego + Stanford University + Simon Fraser University + Google Research + UC Los Angeles`][SAPIEN is a `realistic` and `physics-rich` simulated environment that hosts a large-scale set for `articulated objects`. It enables various `robotic vision and interaction tasks` that require detailed `part-level understanding`. SAPIEN is a collaborative effort between researchers at `UCSD`, `Stanford` and `SFU`.]

* **GATO(TMLR2022)(arxiv2022.05)** A Generalist Agent [[openreview link](https://openreview.net/forum?id=1ikK0kHjvj)]][[arxiv link](https://arxiv.org/abs/2205.06175)][[offifial blog](https://deepmind.google/discover/blog/a-generalist-agent/)][[code|not official](https://github.com/LAS1520/Gato-A-Generalist-Agent)][`Deepmind`]

* **MPiNets(CoRL2023)(arxiv2022.10)** Motion Policy Networks [[openreview link](https://openreview.net/forum?id=aQnn9cIVTRJ)][[paper link](https://proceedings.mlr.press/v205/fishman23a.html)][[arxiv link](https://arxiv.org/abs/2210.12209)][[project link](https://mpinets.github.io/)][[code|official](https://github.com/NVlabs/motion-policy-networks)][`University of Washington + NVIDIA`; `Dieter Fox`]

* **MimicGen(CoRL2023)(arxiv2023.10)** MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations [[openreview link](https://openreview.net/forum?id=dk-2R1f_LR)][[paper link](https://proceedings.mlr.press/v229/mandlekar23a.html)][[arxiv link](https://arxiv.org/abs/2310.17596)][[project link](https://mimicgen.github.io/)][[code|official](https://github.com/NVlabs/mimicgen_environments)][`NVIDIA + The University of Texas at Austin`]

* 👍**Gen2Sim(ICRA2024)(arxiv2023.10)** Gen2Sim: Scaling up Robot Learning in Simulation with Generative Models [[arxiv link](https://arxiv.org/abs/2310.18308)][[project link](https://gen2sim.github.io/)][[code|official](https://github.com/pushkalkatara/Gen2Sim)][`CMU`]

* 👍**RoboGen(ICML2024)(arxiv2023.11)** RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation [[arxiv link](https://arxiv.org/abs/2311.01455)][[project link](https://robogen-ai.github.io/)][[code|official](https://github.com/Genesis-Embodied-AI/RoboGen)][`CMU + Tsinghua IIIS + MIT CSAIL + UMass Amherst + MIT-IBM AI Lab`]

* **LEO(ICML2024)(arxiv2023.11)** An Embodied Generalist Agent in 3D World [[arxiv link](https://arxiv.org/abs/2311.12871)][[project link](https://embodied-generalist.github.io/)][[code|official](https://github.com/embodied-generalist/embodied-generalist)][`BIGAI + PKU + CMU + THU`; on the simulator world]

* **GenSim(ICLR2024 spotlight)(arxiv2023.10)** GenSim: Generating Robotic Simulation Tasks via Large Language Models [[openreview link](https://openreview.net/forum?id=OI3RoHoWAN)][[arxiv link](https://arxiv.org/abs/2310.01361)][[project link](https://gen-sim.github.io/)][[data link](https://huggingface.co/spaces/Gen-Sim/Gen-Sim)][[code|official](https://github.com/liruiw/GenSim)][`MIT CSAIL + SJUT + UCSD + THU + UW + CMU`; `Xiaolong Wang`]

* **PhyRecon(arxiv2024.04)** PhyRecon: Physically Plausible Neural Scene Reconstruction [[arxiv link](https://arxiv.org/abs/2404.16666)][[project link](https://phyrecon.github.io/)][[code|official](https://github.com/PhyRecon/PhyRecon)][`BIGAI + THU + PKU`][It harnesses both `differentiable rendering` and `differentiable physics simulation` to achieve `physically plausible scene reconstruction` from `multi-view images`.]

* **DigitalTwinArt(CVPR2024)(arxiv2024.04)** Neural Implicit Representation for Building Digital Twins of Unknown Articulated Objects [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Weng_Neural_Implicit_Representation_for_Building_Digital_Twins_of_Unknown_Articulated_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2404.01440)][[project link](https://nvlabs.github.io/DigitalTwinArt/)][[code|official](https://github.com/NVlabs/DigitalTwinArt)][`NVIDIA + Stanford University`]

* **SAM-E(ICML2024)(arxiv2024.05)** SAM-E: Leveraging Visual Foundation Model with Sequence Imitation for Embodied Manipulation [[paper link](https://sam-embodied.github.io/static/SAM-E.pdf)][[arxiv link](https://arxiv.org/pdf/2405.19586)][[project link](https://sam-embodied.github.io/)][[weixin blog](https://mp.weixin.qq.com/s/bLqyLHzFoBrRBT0jgkmZMw)][[code|official](https://github.com/pipixiaqishi1/SAM-E)][`THU + Shanghai AI Lab + HKUST`][only tested on the dataset `RLBench`, and obtained inferior results than `3D Diffuser Actor`]

* **PhyScene(CVPR2024, Highlight)(arxiv2024.04)** PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_PhyScene_Physically_Interactable_3D_Scene_Synthesis_for_Embodied_AI_CVPR_2024_paper.html)][[arxiv link](http://arxiv.org/abs/2404.09465)][[project link](https://physcene.github.io/)][[code|official](https://github.com/PhyScene/PhyScene)][`BIGAI`]

* **SPIN(CVPR2024)(arxiv2024.05)** SPIN: Simultaneous Perception, Interaction and Navigation [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Uppal_SPIN_Simultaneous_Perception_Interaction_and_Navigation_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2405.07991)][[project link](https://spin-robot.github.io/)][`CMU`]

* 👍**SimplerEnv(arxiv2024.05)** Evaluating Real-World Robot Manipulation Policies in Simulation [[arxiv link](https://arxiv.org/abs/2405.05941)][[project link](https://simpler-env.github.io/)][[code|official](https://github.com/simpler-env/SimplerEnv)][`UC San Diego + Stanford University + UC Berkeley + Google DeepMind`][Evaluating and reproducing real-world robot manipulation policies (e.g., `RT-1, RT-1-X, Octo`) in simulation under common setups (e.g., `Google Robot, WidowX+Bridge`)]

* 👍**RoboCasa(RSS2024)(arxiv2024.06)** RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots [[arxiv link](https://arxiv.org/pdf/2406.02523)][[project link](https://robocasa.ai/)][[weixin blog](https://mp.weixin.qq.com/s/PPXPbJYru1ZOxgJaMtzDjg)][[zhihu blog](https://zhuanlan.zhihu.com/p/701052987)][[code|official](https://github.com/robocasa/robocasa)][`The University of Texas at Austin + NVIDIA Research`; Real2Sim2Real]

* **IRASim(arxiv2024.06)** IRASim: Learning Interactive Real-Robot Action Simulators [[arxiv link](https://arxiv.org/pdf/2406.14540)][[project link](https://gen-irasim.github.io/)][[code|official](https://github.com/bytedance/IRASim)][`ByteDance Research + HKUST`; `Video Generation as Real-Robot Simulators`]

* **SimGen(arxiv2024.06)** SimGen: Simulator-conditioned Driving Scene Generation [[arxiv link](https://arxiv.org/abs/2406.09386)][[project link](https://metadriverse.github.io/simgen/)][[code|official](https://github.com/metadriverse/SimGen)][`University of California, Los Angeles + Shanghai Jiao Tong University`; `Minyi Guo`]

* **Dreamitate(arxiv2024.06)** Dreamitate: Real-World Visuomotor Policy Learning via Video Generation [[arxiv link](https://arxiv.org/abs/2406.16862)][[project link](https://dreamitate.cs.columbia.edu/)][[code|official](https://github.com/cvlab-columbia/dreamitate)][`Columbia University + Toyota Research Institute + Stanford University`]

* **GENIMA(arxiv2024.07)** Generative Image as Action Models [[arxiv link](https://arxiv.org/abs/2407.07875)][[project link](https://genima-robot.github.io/)][[code|official](https://github.com/MohitShridhar/genima)][`Dyson Robot Learning Lab`; the last author is `Stephen James`][This is an interesting work with similar idea with `Render and Diffuse`]

* **Make-An-Agent(NIPS2024)(arxiv2024.07)** Make-An-Agent: A Generalizable Policy Network Generator with Behavior-Prompted Diffusion [[arxiv link](https://arxiv.org/abs/2407.10973)][[project link](https://cheryyunl.github.io/make-an-agent/)][[code|official](https://github.com/cheryyunl/Make-An-Agent)][`University of Maryland + College Park + Tsinghua University, IIIS + UC San Diego`; `Huazhe Xu`]

* **DiffusionForcing(arxiv2024.07)** Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion [[arxiv link](https://arxiv.org/abs/2407.01392)][[project link](https://boyuan.space/diffusion-forcing/)][[code|official](https://github.com/buoyancy99/diffusion-forcing)][`MIT`]

* **RoboStudio(arxiv2024.08)** RoboStudio: A Physics Consistent World Model for Robotic Arm with Hybrid Representation [[arxiv link](https://www.arxiv.org/abs/2408.14873)][[project link](https://robostudioapp.com/)][[code|official (not released)](https://github.com/RoboOmniSim/Robostudio)][`University of Southern California + National University of Singapore + University of Michigan + Peking University + The Hong Kong University of Science and Technology + Beijing Institute of Technology + Tsinghua University + Xiaomi Robot Technology + AiR, Tsinghua University`]

* 👍👍**Transfusion(arxiv2024.08)** Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model [[arxiv link](https://www.arxiv.org/abs/2408.11039)][[code|official](https://github.com/lucidrains/transfusion-pytorch)][`Meta + Waymo + University of Southern California`]

* **ICRT(arxiv2024.08)** In-Context Imitation Learning via Next-Token Prediction [[arxiv link](https://arxiv.org/abs/2408.15980)][[project link](https://icrt.dev/)][[code|official](https://github.com/Max-Fu/icrt)][`UC Berkeley + Autodesk`]

* 👍**PhysGen(ECCV2024)(arxiv2024.09)** PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation [[arxiv link](https://arxiv.org/abs/2409.18964)][[project link](https://stevenlsw.github.io/physgen/)][[code|official](https://github.com/stevenlsw/physgen)][`University of Illinois Urbana-Champaign`]

* **Gen2Act(arxiv2024.09)** Gen2Act: Human Video Generation in Novel Scenarios enables Generalizable Robot Manipulation [[arxiv link](https://arxiv.org/abs/2409.16283)][[project link](https://homangab.github.io/gen2act/)][`Google DeepMind + Carnegie Mellon University + Stanford University`]

* **LINGO(SIGGRAPH2024)(arxiv2024.10)** Autonomous Character-Scene Interaction Synthesis from Text Instruction [[arxiv link](https://arxiv.org/abs/2410.03187)][[project link](https://lingomotions.com/)][`Peking University + BIGAI`][This paper introduces a framework for synthesizing `multi-stage scene-aware interaction motions` and a comprehensive `language-annotated MoCap dataset (LINGO)`.]

* **GenSim2(CoRL2024)(arxiv2024.10)** GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs [[openreview link](https://openreview.net/forum?id=5u9l6U61S7)][[arxiv link](https://arxiv.org/abs/2410.03645)][[project link](https://gensim2.github.io/)][[code|official](https://github.com/GenSim2/gensim2)][`Tsinghua University + UCSD + Shanghai Jiao Tong University + MIT CSAIL`; `Weinan Zhang + Huazhe Xu`]

* **EvalTasks(arxiv2024.10)** On the Evaluation of Generative Robotic Simulations [[arxiv link](https://arxiv.org/abs/2410.08172)][[project link](https://sites.google.com/view/evaltasks)][`The University of Hong Kong + Tsinghua University IIIS + Shanghai Qi Zhi Institute + Shanghai AI Lab`; `Yi Ma + Huazhe Xu`]

* **DexMimicGen(arxiv2024.10)** DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning [[arxiv link](https://arxiv.org/abs/2410.24185)][[project link](https://dexmimicgen.github.io/)][`NVIDIA Research + UT Austin + UC San Diego`; `Yuke Zhu`]

* **RoboGSim(arxiv2024.11)** RoboGSim: A Real2Sim2Real Robotic Gaussian Splatting Simulator [[arxiv link](https://arxiv.org/abs/2411.11839)][[project link](https://robogsim.github.io/)][`Harbin Institute of Technology, Shenzhen + MEGVII Technology + Zhejiang University + Institute of Computing Technology, Chinese Academy of Sciences`]



### ※ 6) Other Mainstream Conferences

* **Where2Act(ICCV2021)(arxiv2021.01)** Where2Act: From Pixels to Actions for Articulated 3D Objects [[paper link](http://openaccess.thecvf.com/content/ICCV2021/html/Mo_Where2Act_From_Pixels_to_Actions_for_Articulated_3D_Objects_ICCV_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2101.02692)][[project link](https://cs.stanford.edu/~kaichun/where2act/)][[code|official](https://github.com/daerduoCarey/where2act)][`Stanford University + Facebook AI Research`][It is based on the simulation `SAPIEN`]

* **BC-Transformer(robomimic)(CoRL2021 oral)(arxiv2021.08)** What Matters in Learning from Offline Human Demonstrations for Robot Manipulation [[openreview link](https://openreview.net/forum?id=JrsfBJtDFdI)][[paper link](https://proceedings.mlr.press/v164/mandlekar22a.html)][[arxiv link](https://arxiv.org/abs/2108.03298)][[project link](https://arise-initiative.github.io/robomimic-web/)][[code|official](https://github.com/ARISE-Initiative/robomimic)][`Stanford University + The University of Texas at Austin`][The proposed method `BC-Transformer` is used as a baseline in `robocasa`][`robomimic`: A Framework for Robot Learning from Demonstration. It offers a broad set of `demonstration datasets` collected on `robot manipulation domains`, and learning algorithms to learn from these datasets.]

* **CLIPort(CoRL2021)(arxiv2021.09)** CLIPort: What and Where Pathways for Robotic Manipulation [[openreview link](https://openreview.net/forum?id=9uFiX_HRsIL)][[paper link](https://proceedings.mlr.press/v164/shridhar22a.html)][[arxiv link](https://arxiv.org/abs/2109.12098)][[project link](https://cliport.github.io/)][[code|official](https://github.com/cliport/cliport)][[code|not official - CLIPort-Batchify](github.com/ChenWu98/cliport-batchify)][`University of Washington + NVIDIA`]

* **BIP(IROS2022)(arxiv2022.08)** A System for Imitation Learning of Contact-Rich Bimanual Manipulation Policies [[paper link](https://ieeexplore.ieee.org/abstract/document/9981802)][[arxiv link](https://arxiv.org/abs/2208.00596)][[project link](https://bimanualmanipulation.com/)][[code|official](https://github.com/ir-lab/irl_control/tree/iros2022-dev/irl_control/learning)][`Carnegie Melon University + Intrinsic, An Alphabet Company + Arizona State University`]

* **BC-Z(CVPR2022)(arxiv2022.02)** BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning [[openreview link](https://openreview.net/forum?id=8kbp23tSGYv)][[paper link](https://proceedings.mlr.press/v164/jang22a.html)][[arxiv link](https://arxiv.org/abs/2202.02005)][[project link](https://sites.google.com/view/bc-z/home)][[code|official](https://github.com/google-research/tensor2robot/tree/master/research/bcz)][`Robotics at Google + The Moonshot Factory + UC Berkeley + Stanford University`; `Chelsea Finn`][It is based on the `TensorFlow`]

* ❤**C2FARM(CVPR2022 Oral)(arxiv2021.06)** Coarse-To-Fine Q-Attention: Efficient Learning for Visual Robotic Manipulation via Discretisation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/James_Coarse-To-Fine_Q-Attention_Efficient_Learning_for_Visual_Robotic_Manipulation_via_Discretisation_CVPR_2022_paper.html)][[arxiv link](https://arxiv.org/abs/2106.12534)][[project link](https://sites.google.com/view/c2f-q-attention)][[code|official](https://github.com/stepjam/ARM)][`Dyson Robotics Lab, Imperial College London`][It maybe the `fisrt` work to conduct `next-best keyframe detection` by the first author [`Stephen James`](https://stepjam.github.io/), who also used this “predict the next (best) keyframe action” idea in his other works [`(RAL2022) Q-attention: Enabling Efficient Learning for Vision-based Robotic Manipulation`](https://arxiv.org/abs/2105.14829) and [`(TMLR2022) Auto-Lambda: Disentangling Dynamic Task Relationships`](https://openreview.net/forum?id=KKeCMim5VN). And the `key-frames` idea fisrtly proposed in work [`(ICRA2021) Coarse-to-Fine Imitation Learning: Robot Manipulation from a Single Demonstration`](https://ieeexplore.ieee.org/abstract/document/9560942) by [`Edward Johns`](https://www.robot-learning.uk/) who leading the `Robot Learning Lab` at `Imperial College London`.]

* ❤**R3M(CoRL2022)(arxiv2022.03)** R3M: A Universal Visual Representation for Robot Manipulation [[openreview link](https://openreview.net/forum?id=tGbpgz6yOrI)][[paper link](https://proceedings.mlr.press/v205/nair23a.html)][[arxiv link](https://arxiv.org/abs/2203.12601)][[project link](https://sites.google.com/view/robot-r3m/)][[code|official](https://github.com/facebookresearch/r3m)][`Stanford University + Meta AI`; a pre-training method][We study if `visual representations pre-trained` on `diverse human videos` can enable efficient robotic manipulation. We `pre-train a single representation`, R3M, utilizing an objective that combines `time contrastive learning`, `video-language alignment`, and `a sparsity penalty`.]

* **MVP(CoRL2022 oral)(arxiv2022.10)** Real-World Robot Learning with Masked Visual Pre-training [[openreview link](https://openreview.net/forum?id=KWCZfuqshd)][[paper link](https://proceedings.mlr.press/v205/radosavovic23a.html)][[arxiv link](https://arxiv.org/abs/2210.03109)][[project link](https://tetexiao.com/projects/real-mvp)][[code|official](https://github.com/ir413/mvp)][`University of California, Berkeley`; a pre-training method][It first compiled a massive collection of `4.5 million images` from `ImageNet`, `Epic Kitchens`, `Something Something`, `100 Days of Hands`, and `Ego4D datasets`. Then, it pre-trained a model based on `masked autoencoder (MAE)`.]

* ❤**PerAct(CoRL2022)(arxiv2022.09)** Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation [[openreview link](https://openreview.net/forum?id=PS_eCS_WCvD)][[paper link](https://proceedings.mlr.press/v205/shridhar23a.html)][[arxiv link](https://arxiv.org/abs/2209.05451)][[project link](https://peract.github.io/)][[code|official](https://github.com/peract/peract)][`University of Washington + NVIDIA`; `Dieter Fox`][It proposed a 3D policy that `voxelizes the workspace` and detects the `next voxel action` through `global self-attention`.][This work is largely based on [`C2FARM (CVPR2022)`](https://sites.google.com/view/c2f-q-attention) and [`PerceiverIO (ICLR2022)`](https://openreview.net/forum?id=fILj7WpI-g); It constructs a structured observation and action space through `keyframe extraction` and `voxelization` following `C2FARM`.]

* **ToolFlowNet(CoRL2022)(arxiv2022.11)** ToolFlowNet: Robotic Manipulation with Tools via Predicting Tool Flow from Point Clouds [[openreview link](https://openreview.net/forum?id=2gfB_kMVFvP)][[paper link](https://proceedings.mlr.press/v205/seita23a.html)][[arxiv link](https://arxiv.org/abs/2211.09006)][[project link](https://sites.google.com/view/point-cloud-policy/home)][[code|official](https://github.com/DanielTakeshi/softagent_tfn)][`The Robotics Institute, Carnegie Mellon Universit`]

* 👍**CaP(ICRA2023)(arxiv2022.09)** Code as Policies: Language Model Programs for Embodied Control [[paper link](https://ieeexplore.ieee.org/abstract/document/10160591)][[arxiv link](https://arxiv.org/abs/2209.07753)][[project link](https://code-as-policies.github.io/)][[code|official](https://github.com/google-research/google-research/tree/master/code_as_policies)][`Robotics at Google`]

* 👍**ProgPrompt(ICRA2023)(arxiv2022.09)** ProgPrompt: Generating Situated Robot Task Plans using Large Language Models [[paper link](https://ieeexplore.ieee.org/abstract/document/10161317)][[arxiv link](https://arxiv.org/abs/2209.11302)][[project link](https://progprompt.github.io/)][[code|official](https://github.com/NVlabs/progprompt-vh)][`University of Southern California + NVIDIA`; It has released code for replicating the results on the `VirtualHome` dataset.]

* **PourIt(ICCV2023)(arxiv2023.07)** PourIt!: Weakly-supervised Liquid Perception from a Single Image for Visual Closed-Loop Robotic Pouring [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_PourIt_Weakly-Supervised_Liquid_Perception_from_a_Single_Image_for_Visual_ICCV_2023_paper.html)][[arxiv link](http://arxiv.org/abs/2307.11299)][[project link](https://hetolin.github.io/PourIt/)][[code|official](https://github.com/hetolin/PourIt)][`Fudan University`]

* **Voltron(RSS2023)(arxiv2023.02)** Language-Driven Representation Learning for Robotics [[paper link](https://www.roboticsproceedings.org/rss19/p032.pdf)][[arxiv link](https://arxiv.org/abs/2302.12766)][[project link](https://sites.google.com/view/voltron-robotics)][[code|official](https://github.com/siddk/voltron-robotics)][`Stanford University + Toyota Research Institute`; a pre-training method][It provides code for loading pretrained `Voltron`, `R3M`, and `MVP` representations for `adaptation to downstream tasks`, as well as code for pretraining such representations on `arbitrary datasets`.]

* **RT-1(RSS2023)(arxiv2022.12)** RT-1: Robotics Transformer for Real-World Control at Scale [[paper link](https://www.roboticsproceedings.org/rss19/p025.pdf)][[arxiv link](https://arxiv.org/abs/2212.06817)][[project link](https://robotics-transformer1.github.io/)][[code|official](https://github.com/google-research/robotics_transformer)][by `Google DeepMind`]

* 👍❤**DiffusionPolicy(RSS2023)(IJRR2024)(arxiv2023.03)** Diffusion Policy: Visuomotor Policy Learning via Action Diffusion [[paper link](https://www.roboticsproceedings.org/rss19/p026.pdf)][[arxiv link](https://arxiv.org/abs/2303.04137)][[project link](https://diffusion-policy.cs.columbia.edu/)][[code|official](https://github.com/real-stanford/diffusion_policy)][`Columbia University + Toyota Research Institute + MIT`][Did not conduct experiments on benchmarks `RLBench` and `CALVIN`]

* **ACT/ALOHA(RSS2023)(arxiv2023.04)** Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware [[paper link](https://roboticsproceedings.org/rss19/p016.pdf)][[arxiv link](https://arxiv.org/abs/2304.13705)][[project link](https://tonyzhaozh.github.io/aloha/)][[code|official](https://github.com/tonyzhaozh/act)][`Stanford University + UC Berkeley + Meta`][It adopts a `CVAE scheme` with `transformer backbones` and `ResNet image encoders` to model the variability of human data]

* **RoboNinja(RSS2023)(arxiv2023.02)** RoboNinja: Learning an Adaptive Cutting Policy for Multi-Material Objects [[paper link](https://roboticsproceedings.org/rss19/p046.pdf)][[arxiv link](https://arxiv.org/abs/2302.11553)][[project link](https://roboninja.cs.columbia.edu/)][[code|official](https://github.com/real-stanford/roboninja)][`Columbia University + CMU + UC Berkeley + UC San Diego + UMass Amherst & MIT-IBM AI Lab`]

* **MV-MWM(ICML2023)(arxiv2023.02)** Multi-View Masked World Models for Visual Robotic Manipulation [[paper link](https://proceedings.mlr.press/v202/seo23a.html)][[arxiv link](https://arxiv.org/abs/2302.02408)][[project link](https://sites.google.com/view/mv-mwm)][[code|official](https://github.com/younggyoseo/MV-MWM)][`KAIST  + Dyson Robot Learning Lab + Google Research + UC Berkeley`; It used the less-popular `TensorFlow 2`]

* **LLM-MCTS(NIPS2023)(arxiv2023.05)** Large Language Models as Commonsense Knowledge for Large-Scale Task Planning [[openreview link](https://openreview.net/forum?id=Wjp1AYB8lH)][[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/65a39213d7d0e1eb5d192aa77e77eeb7-Abstract-Conference.html)][[arxiv link](https://arxiv.org/abs/2305.14078)][[project link](https://llm-mcts.github.io/)][[code|official](https://github.com/1989Ryan/llm-mcts)][`National University of Singapore`][It used `Large Language Models` as both the `commonsense world model` and the `heuristic policy` within the `Monte Carlo Tree Search` framework, enabling better-reasoned `decision-making` for daily tasks.]

* **RoboHive(NIPS2023)(arxiv2023.10)** RoboHive: A Unified Framework for Robot Learning [[openreview link](https://openreview.net/forum?id=0H5fRQcpQ7)][[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8a84a4341c375b8441b36836bb343d4e-Abstract-Datasets_and_Benchmarks.html)][[arxiv link](https://arxiv.org/abs/2310.06828)][[project link](https://sites.google.com/view/robohive)][[code|official](https://github.com/vikashplus/robohive)][`U.Washington + UC Berkeley + CMU + UT Austin + OpenAI + GoogleAI + Meta-AI`; `Datasets and Benchmarks Track`]

* **L2M / Learning-to-Modulate(NIPS2023)(arxiv2023.06)** Learning to Modulate pre-trained Models in RL [[openreview link](https://openreview.net/forum?id=aIpGtPwXny)][[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/77e59fafe99e94f822e79bf9308ec377-Abstract-Conference.html)][[arxiv link](https://arxiv.org/abs/2306.14884)][[code|official](https://github.com/ml-jku/L2M)][`Johannes Kepler University Linz, Austria + Google DeepMind + UCL`; tyr to `adapt` the already trained `RL` models.]

* ❤**ChainedDiffuser(CoRL2023)** ChainedDiffuser: Unifying Trajectory Diffusion and Keypose Prediction for Robotic Manipulation  [[openreview link](https://openreview.net/forum?id=W0zgY2mBTA8)][[paper link](https://proceedings.mlr.press/v229/xian23a.html)][[project link](https://chained-diffuser.github.io/)][[code|official](https://github.com/zhouxian/act3d-chained-diffuser)][`CMU`, using the `Diffusion`; the first authors [`Zhou Xian`](https://www.zhou-xian.com/) and [`Nikolaos Gkanatsios`](https://nickgkan.github.io/)][It proposed to replace `motion planners`, commonly used for keypose to keypose linking, with a `trajectory diffusion model` that conditions on the `3D scene feature cloud` and the `predicted target 3D keypose` to denoise a trajectory from the current to the target keypose.]

* **MOO(CoRL2023)(arxiv2023.03)** Open-World Object Manipulation using Pre-trained Vision-Language Models [[openreview link](https://openreview.net/forum?id=9al6taqfTzr)][[paper link](https://proceedings.mlr.press/v229/stone23a.html)][[arxiv link](https://arxiv.org/abs/2303.00905)][[project link](https://robot-moo.github.io/)][`Robotics at Google`]

* **HiveFormer(CoRL2023 Oral)(arxiv2022.09)** Instruction-driven history-aware policies for robotic manipulations [[openreview link](https://openreview.net/forum?id=h0Yb0U_-Tki)][[paper link](https://proceedings.mlr.press/v205/guhur23a.html)][[arxiv link](https://arxiv.org/abs/2209.04899)][[project link](https://vlc-robot.github.io/hiveformer-corl/)][[code|official](https://github.com/vlc-robot/hiveformer-corl)][`Inria + IIIT Hyderabad`; the second author [`Shizhe Chen`](https://cshizhe.github.io/)][It is a 3D policy that enables attention `between features of different history time steps`.][It considered `74 tasks` grouped into 9 categories on `RLBench`.]

* **PolarNet(CoRL2023)(arxiv2023.09)** PolarNet: 3D Point Clouds for Language-Guided Robotic [[openreview link](https://openreview.net/forum?id=efaE7iJ2GJv)][[paper link](https://proceedings.mlr.press/v229/chen23b.html)][[arxiv link](https://arxiv.org/abs/2309.15596)][[project link](https://www.di.ens.fr/willow/research/polarnet/)][[code|official](https://github.com/vlc-robot/polarnet/)][`INRIA`; the first author [`Shizhe Chen`](https://cshizhe.github.io/); `3D Vision-Language-Action`][It is a 3D policy that computes `dense point
representations` for the robot workspace using a `PointNext` backbone.][It considered `74 tasks` grouped into 9 categories on `RLBench` following `HiveFormer`.]

* ❤**RVT(CoRL2023 Oral)(arxiv2023.06)** RVT: Robotic View Transformer for 3D Object Manipulation [[openreview link](https://openreview.net/forum?id=0hPkttoGAf)][[paper link](https://proceedings.mlr.press/v229/goyal23a.html)][[arxiv link](https://arxiv.org/abs/2306.14896)][[project link](https://robotic-view-transformer.github.io/)][[code|official](https://github.com/nvlabs/rvt)][`NVIDIA`; `Dieter Fox`][It `re-projects` the input `RGB-D` image to alternative image views, featurizes those and `lifts` the predictions to 3D to `infer 3D locations` for the robot’s end-effector.][It proposed a `3D policy` that deploys a `multi-view transformer` to predict actions and fuses those across views by `back-projecting` to 3D.]

* ❤**Act3D(CoRL2023)(arxiv2023.06)** Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation [[openreview link](https://openreview.net/forum?id=-HFJuX1uqs)][[paper link](https://proceedings.mlr.press/v229/gervet23a.html)][[arxiv link](https://arxiv.org/abs/2306.17817)][[project link](https://act3d.github.io/)][[code|official](https://github.com/zhouxian/act3d-chained-diffuser)][`CMU`; the first authors [`Theophile Gervet`](https://theophilegervet.github.io/) and [`Zhou Xian`](https://www.zhou-xian.com/) and [`Nikolaos Gkanatsios`](https://nickgkan.github.io/)][It proposed a 3D policy that featurizes the robot’s `3D workspace` using `coarse-to-fine sampling` and `featurization`.]

* **RT-2(CoRL2023)(arxiv2023.07)** RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control [[openreview link](https://openreview.net/forum?id=XMQgwiJ7KSX)][[paper link](https://proceedings.mlr.press/v229/zitkovich23a.html)][[arxiv link](https://arxiv.org/abs/2307.15818)][[project link](https://robotics-transformer2.github.io/)][[code|not official](https://github.com/kyegomez/RT-2)][by `Google DeepMind`; based on `RT-1`; it is trained on top of [`PaLM-E (12B)`](https://palm-e.github.io/); it is also trained on top of [`PaLI-X (55B)`](https://arxiv.org/abs/2305.18565); it plans to use more powerful `VLMs`, such as [`LLaVA (Large Language and Vision Assistant)`](https://llava-vl.github.io/) and `LLaVA-1.5`]

* **GRIF(CoRL2023)(arxiv2023.07)** Goal Representations for Instruction Following: A Semi-Supervised Language Interface to Control [[openreview link](https://openreview.net/forum?id=0bZaUfELuW)][[paper link](https://proceedings.mlr.press/v229/myers23a.html)][[arxiv link](https://arxiv.org/abs/2307.00117)][[project link](https://rail-berkeley.github.io/grif/)][[code|official](https://github.com/rail-berkeley/grif_release)][`University of California Berkeley + Microsoft Research`; It used `Semi-Supervised Learning` and `Contrastive Learning`, but it also used the less-popular `TensorFlow`]

* **GROOT(CoRL2023)(arxiv2023.10)** Learning Generalizable Manipulation Policies with Object-Centric 3D Representations [[openreview link](https://openreview.net/forum?id=9SM6l0HyY_)][[paper link](https://proceedings.mlr.press/v229/zhu23b.html)][[arxiv link](https://arxiv.org/abs/2310.14386)][[project link](https://ut-austin-rpl.github.io/GROOT/)][[code|official](https://github.com/UT-Austin-RPL/GROOT)][`The University of Texas, Austin + Sony AI`; It used the `SAM` for segmenting out target objects.]

* 👍**Optimus(CoRL2023)(arxiv2023.05)** Imitating Task and Motion Planning with Visuomotor Transformers [[openreview link](https://openreview.net/forum?id=QNPuJZyhFE)][[paper link](https://proceedings.mlr.press/v229/dalal23a.html)][[arxiv link](https://arxiv.org/abs/2305.16309)][[project link](https://mihdalal.github.io/optimus/)][[code|official](https://github.com/NVlabs/Optimus)][`CMU + NVIDIA`; `Dieter Fox`][It is used as the baseline method by `RoboCasa`]

* **ScalingUp(CoRL2023)(arxiv2023.07)** Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition [[openreview link](https://openreview.net/forum?id=3uwj8QZROL)][[paper link](https://proceedings.mlr.press/v229/ha23a.html)][[arxiv link](https://arxiv.org/abs/2307.14535)][[project link](https://www.cs.columbia.edu/~huy/scalingup/)][[code|official](https://github.com/real-stanford/scalingup)][`Columbia University + Google DeepMind`][It used the `Diffusion Policy` for building a robust `multi-task language-conditioned visuo-motor policy`.]

* **MimicPlay(CoRL2023 Oral)(arxiv2023.02)** MimicPlay: Long-Horizon Imitation Learning by Watching Human Play [[openreview link](https://openreview.net/forum?id=hRZ1YjDZmTo)][[paper link](https://proceedings.mlr.press/v229/wang23a.html)][[arxiv link](https://arxiv.org/abs/2302.12422)][[project link](https://mimic-play.github.io/)][[code|official](https://github.com/j96w/MimicPlay)][`Stanford + NVIDIA + Georgia Tech + UT Austin + Caltech`, by `Stanford Fei-Fei Li`]

* **VoxPoser(CoRL2023 Oral)(arxiv2023.07)** VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models [[paper link](https://proceedings.mlr.press/v229/huang23b.html)][[arxiv link](https://arxiv.org/abs/2307.05973)][[project link](https://voxposer.github.io/)][[code|official](https://github.com/huangwl18/VoxPoser)][by `Stanford Fei-Fei Li`; It extracts `affordances` and `constraints` from large language models (`LLMs`) and vision-language  models (`VLMs`) to compose `3D value maps`; It needs `Detector+Segmentor+Tracker` and thus is very `slow`]

* **GNFactor(CoRL2023 Oral)(arxiv2023.08)** GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields [[openreview link](https://openreview.net/forum?id=b1tl3aOt2R2)][[paper link](https://proceedings.mlr.press/v229/ze23a.html)][[arxiv link](https://arxiv.org/abs/2308.16891)][[project link](https://yanjieze.com/GNFactor/)][[code|official](https://github.com/YanjieZe/GNFactor)][`SJTU + UC San Diego + University of Hong Kong + AWS AI, Amazon`; a work by the `Xiaolong Wang` group][It proposed a 3D policy that co-optimizes a `neural field` for reconstructing the `3D voxels` of the input scene and a `PerAct` module for predicting actions based on `voxel representations`.]

* **HACMan(CoRL2023 Oral)(arxiv2023.05)** HACMan: Learning Hybrid Actor-Critic Maps for 6D Non-Prehensile Manipulation [[arxiv link](https://arxiv.org/abs/2305.03942)][[project link](https://hacman-2023.github.io/)][[code|official](https://github.com/HACMan-2023/HACMan)][`CMU + Meta`]

* **Diff-LfD(CoRL2023, Oral)** Diff-LfD: Contact-aware Model-based Learning from Visual Demonstration for Robotic Manipulation via Differentiable Physics-based Simulation and Rendering [[openreview link](https://openreview.net/forum?id=DYPOvNot5F)][[paper link](https://proceedings.mlr.press/v229/zhu23a.html)][[project link](https://sites.google.com/view/diff-lfd)][`UC Berkeley + USTC + Zhejiang University + Nanjing University + University of Queensland + Shanghai Jiaotong University + National University of Singapore`; `Cewu Lu`]

* **RoboCat(TMLR2023)(arxiv2023.06)** RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation [[openreview link](https://openreview.net/forum?id=vsCpILiWHu)][[arxiv link](https://arxiv.org/abs/2306.11706)][`Google DeepMind`]

* **Giving-Robots-a-Hand(arxiv2023.07)** Giving Robots a Hand: Learning Generalizable Manipulation with Eye-in-Hand Human Video Demonstrations [[arxiv link](https://arxiv.org/abs/2307.05959)][[project link](https://giving-robots-a-hand.github.io/)][`Stanford University`]

* **RoboTAP(arxiv2023.08)** RoboTAP: Tracking Arbitrary Points for Few-Shot Visual Imitation [[arxiv link](https://arxiv.org/abs/2308.15975)][[project link](https://robotap.github.io/)][[code|official](https://github.com/google-deepmind/tapnet/blob/main/colabs/tapir_clustering.ipynb)][`Google DeepMind + University College London`][It is based on the `TAPNet`]

* **DeformGS(WAFR2024)(arxiv2023.12)** DeformGS: Scene Flow in Highly Deformable Scenes for Deformable Object Manipulation [[arxiv link](https://arxiv.org/abs/2312.00583)][[project link](https://deformgs.github.io/)][[code|official](https://github.com/momentum-robotics-lab/deformgs)][`CMU + Stanford + NVIDIA + NUS + TUM`][`Deformable Object Manipulation`]

* **BimanualAssitDressing(TRO2024)(arxiv2023.01)** Do You Need a Hand? a Bimanual Robotic Dressing Assistance Scheme [[paper link](https://ieeexplore.ieee.org/document/10436357)][[arxiv link](https://arxiv.org/abs/2301.02749)][[project link](https://sites.google.com/view/bimanualassitdressing/home)][`University of York + Honda Research Institute + Cognitive Robotics, 3mE, TU Delft, the Netherlands`]

* **ERJ(RAL2024)(arxiv2024.06)** Redundancy-aware Action Spaces for Robot Learning [[arxiv link](https://arxiv.org/abs/2406.04144)][[project link](https://redundancy-actions.github.io/)][[code|official](https://github.com/mazpie/redundancy-action-spaces)][`Dyson Robot Learning Lab + Imperial College London`; `Stephen James`][This work analyses the criteria for designing `action spaces` for robot manipulation and introduces `ER (End-effector Redundancy)`, a novel action space formulation that, by addressing the `redundancies` present in the manipulator, aims to combine the advantages of both joint and task spaces, offering `fine-grained comprehensive control with overactuated robot arms` whilst achieving highly efficient robot learning.]

* **LLM-RL / LLaRP(ICLR2024)(arxiv2023.10)** Large Language Models as Generalizable Policies for Embodied Tasks [[openreview link](https://openreview.net/forum?id=u6imHU4Ebu)][[arxiv link](https://arxiv.org/abs/2310.17722)][[project link](https://llm-rl.github.io/)][[code|official](https://github.com/apple/ml-llarp)][`Apple`; `Large LAnguage model Reinforcement Learning Policy (LLaRP)`]

* 👍**RoboFlamingo(ICLR2024 Spotlight)(arxiv2023.11)** Vision-Language Foundation Models as Effective Robot Imitators [[openreview link](https://openreview.net/forum?id=lFYj0oibGR)][[arxiv link](https://arxiv.org/abs/2311.01378)][[project link](https://roboflamingo.github.io/)][[code|official](https://github.com/RoboFlamingo/RoboFlamingo)][`ByteDance + THU + SJTU 
`; based on the `OpenFlamingo`, and tested on the dataset `CALVIN`]

* 👍**SuSIE(ICLR2024)(arxiv2023.11)** Zero-Shot Robotic Manipulation with Pre-Trained Image-Editing Diffusion Models [[openreview link](https://openreview.net/forum?id=c0chJTSbci)][[arxiv link](https://arxiv.org/abs/2310.10639)][[project link](https://rail-berkeley.github.io/susie/)][[code|official](https://github.com/kvablack/susie)][`UCB + Stanford+ Google
`; using the `InstructPix2Pix` to predict future frames; using the `Diffusion` to predict action; it has beated the previous SOTA `RT-2-X`]

* 👍**GR-1(ICLR2024)(arxiv2023.12)** Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation [[openreview link](https://openreview.net/forum?id=NxoFmGgWC9)][[arxiv link](https://arxiv.org/abs/2312.13139)][[project link](https://gr1-manipulation.github.io/)][[code|official](https://github.com/bytedance/GR-1)][`ByteDance`; it adopted the `GPT-style Transformers (GPT-1)`; it adopted the released `CLIP` and `MAE`; it is pretrained on the large video dataset `Ego4D(CVPR2022)`]

* **FourierTransporter(ICLR2024)(arxiv2024.01)** Fourier Transporter: Bi-Equivariant Robotic Manipulation in 3D [[openreview link](https://openreview.net/forum?id=UulwvAU1W0)][[arxiv link](https://arxiv.org/abs/2401.12046)][[project link](https://haojhuang.github.io/fourtran_page/)][`Northeastern Univeristy`][It is tested on the `RLBench` with seleted 5 hard tasks]

* **AVDC(ICLR2024)(arxiv2023.10)** Learning to Act from Actionless Videos through Dense Correspondences [[openreview link](https://openreview.net/forum?id=Mhb5fpA1T0)][[arxiv link](https://arxiv.org/abs/2310.08576)][[project link](https://flow-diffusion.github.io/)][[code|official](https://github.com/flow-diffusion/AVDC)][`National Taiwan University + MIT`][This method is cited by `ATM(RSS2024)`, and has a inferior performance then `ATM`]

* **RT-Trajectory(ICLR2024, Spotlight)(arxiv2023.11)** RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches [[openreview link](https://openreview.net/forum?id=F1TKzG8LJO)][[arxiv link](https://arxiv.org/abs/2311.01977)][[project link](https://rt-trajectory.github.io/)][[code|official]()][`Google DeepMind + University of California San Diego + Stanford University + Intrinsic)`; `Hao Su + Chelsea Finn`]

* **EquivAct(ICRA2024)(arxiv2023.10)** EquivAct: SIM(3)-Equivariant Visuomotor Policies beyond Rigid Object Manipulation [[arxiv link](https://arxiv.org/abs/2310.16050)][[project link](https://equivact.github.io/)][[code|not available]()][`Stanford University + Princeton University`]

* **LLM-AOM(ICRA2024)(arxiv2023.11)** Kinematic-aware Prompting for Generalizable Articulated Object Manipulation with LLMs [[arxiv link](https://arxiv.org/abs/2311.02847)][[project link](https://gewu-lab.github.io/llm_for_articulated_object_manipulation/)][[code|official](https://github.com/GeWu-Lab/LLM_articulated_object_manipulation)][`Renmin University of China + Shanghai Artificial Intelligence Laboratory + Northwestern Polytechnical University`][the Demonstration Collection [scripts](https://github.com/GeWu-Lab/LLM_articulated_object_manipulation?tab=readme-ov-file#demonstration-collection) on `Isaac gym`]

* **LOTUS(ICRA2024)(arxiv2023.11)** LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery [[arxiv link](https://arxiv.org/abs/2311.02058)][[project link](https://ut-austin-rpl.github.io/Lotus/)][[code|official](https://github.com/UT-Austin-RPL/Lotus)][`The University of Texas at Austin + Peking University`][`Continual Imitation Learning`, `Lifelong Learning`]

* **HOPMan(ICRA2024, Best Paper in Robot Manipulation Finalist)(arxiv2023.12)** Towards Generalizable Zero-Shot Manipulation via Translating Human Interaction Plans [[paper link](https://ieeexplore.ieee.org/abstract/document/10610288)][[arxiv link](https://arxiv.org/abs/2312.00775)][[project link](https://homangab.github.io/hopman/)][`Carnegie Mellon University + Meta AI`]

* **DINOBot(ICRA2024)(arxiv2024.02)** DINOBot: Robot Manipulation via Retrieval and Alignment with Vision Foundation Models [[arxiv link](https://arxiv.org/abs/2402.13181)][[project link](https://sites.google.com/view/dinobot)][[project link2](https://www.robot-learning.uk/dinobot)][[code|official](https://gist.github.com/normandipalo/fbc21f23606fbe3d407e22c363cb134e)][`The Robot Learning Lab at Imperial College London`; the [homepage](https://www.robot-learning.uk/)]

* **Bi-KVIL(ICRA2024)(arxiv2024.03)** Bi-KVIL: Keypoints-based Visual Imitation Learning of Bimanual Manipulation Tasks [[paper link](https://ieeexplore.ieee.org/abstract/document/10610763/)][[arxiv link](https://arxiv.org/abs/2403.03270)][[project link](https://sites.google.com/view/bi-kvil)][[code|official](https://github.com/wyngjf/bi-kvil-pub.git)][`Karlsruhe Institute of Technology`][The proposed Bi-KVIL jointly extracts so-called `Hybrid Master-Slave Relationships (HMSR)` among objects and hands, `bimanual coordination strategies`, and `sub-symbolic task representations`]

* **PCWM(ICRA2024)(arxiv2024.04)** Point Cloud Models Improve Visual Robustness in Robotic Learners [[arxiv link](https://arxiv.org/abs/2404.18926)][[project link](https://pvskand.github.io/projects/PCWM)][`Oregon State University + University of Utah + NVIDIA`; RL-based method]

* **RPMArt(IROS2024)(arxiv2024.03)** RPMArt: Towards Robust Perception and Manipulation for Articulated Objects [[arxiv link](https://arxiv.org/abs/2403.16023)][[project link](https://r-pmart.github.io/)][[code|official](https://github.com/R-PMArt/rpmart)][`Shanghai Jiao Tong University + Stanford University + Hefei University of Technology`; `Cewu Lu`]


* **RISE(IROS)(arxiv2024.04)** RISE: 3D Perception Makes Real-World Robot Imitation Simple and Effective [[arxiv link](https://arxiv.org/abs/2404.12281)][[project link](https://rise-policy.github.io/)][[code|official](https://github.com/rise-policy/RISE)][`SJTU`; proposed by authors [`Chenxi Wang`](https://github.com/chenxi-wang), [`Hongjie Fang`](https://tonyfang.net/), [`Hao-Shu Fang`](https://fang-haoshu.github.io/), and [`Cewu Lu`](https://www.mvig.org/)][Did not conduct experiments on benchmarks `RLBench` and `CALVIN`, and compared to various baselines (2D: [`ACT`](https://tonyzhaozh.github.io/aloha/) and [`Diffusion Policy`](https://diffusion-policy.cs.columbia.edu/); 3D: [`Act3D`](https://act3d.github.io/) and [`DP3`](https://3d-diffusion-policy.github.io/)) on many tasks][It is an `end-to-end` baseline for real-world imitation learning, which `predicts continuous actions` directly from `single-view point clouds`. ]

* **Diff-Control(IROS2024)** Diff-Control: A Stateful Diffusion-based Policy for Imitation Learning [[pdf link](https://diff-control.github.io/static/videos/Diff-Control.pdf)][[project link](https://diff-control.github.io/)][[code|official](https://github.com/ir-lab/Diff-Control)][`Interactive Robotics Lab, Arizona State University + Kyushu Institute of Technology`]

* **DITTO(IROS2024)(arxiv2024.03)** DITTO: Demonstration Imitation by Trajectory Transformation [[arxiv link](https://arxiv.org/abs/2403.15203)][[project link](http://ditto.cs.uni-freiburg.de/)][`University of Freiburg, Germany`][`Learning from action labels free human videos`]

* **PreAfford(IROS2024)(arxiv2024.04)** PreAfford: Universal Affordance-Based Pre-Grasping for Diverse Objects and Environments [[arxiv link](https://arxiv.org/abs/2404.03634)][[project link](https://air-discover.github.io/PreAfford/)][[code|official](https://github.com/Robot-K/PreAfford)][`THU + PKU`]

* **ManipLLM(CVPR2024)(arxiv2023.12)** ManipLLM: Embodied Multimodal Large Language Model for Object-Centric Robotic Manipulation [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Li_ManipLLM_Embodied_Multimodal_Large_Language_Model_for_Object-Centric_Robotic_Manipulation_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2312.16217)][[project link](https://sites.google.com/view/manipllm)][[code|official](https://github.com/clorislili/ManipLLM)][`Peking University`]

* ❤**HDP(CVPR2024)(arxiv2024.03)** Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2403.03890)][[project link](https://yusufma03.github.io/projects/hdp/)][[code|official](https://github.com/dyson-ai/hdp)][`Dyson Robot Learning Lab`][It uses `PerAct` as the `high-level agent`]

* ❤**SUGAR(CVPR2024)(arxiv2024.04)** SUGAR: Pre-training 3D Visual Representations for Robotics [[arxiv link](https://arxiv.org/abs/2404.01491)][[project link](https://cshizhe.github.io/projects/robot_sugar)][[code|official](https://github.com/cshizhe/robot_sugar)][`INRIA`; the first author [`Shizhe Chen`](https://cshizhe.github.io/); `3D Vision-Language-Action`]

* **ATM(RSS2024)(arxiv2024.01)** Any-point Trajectory Modeling for Policy Learning [[arxiv link](https://arxiv.org/abs/2401.00025)][[project link](https://xingyu-lin.github.io/atm/)][[code|official](https://github.com/Large-Trajectory-Model/ATM)][`UC Berkeley + IIIS, Tsinghua University + Stanford University + Shanghai Artificial Intelligence Laboratory + Shanghai Qi Zhi Institute + CUHK`][The method is evaluated on a challenging `simulation benchmark (LIBERO)` comprised of `130 language-conditioned manipulation tasks`, and on `5 tasks` in a `real-world UR5 Kitchen` environment.]

* ❤**DP3(RSS2024)(arxiv2024.03)** 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations [[arxiv link](https://arxiv.org/abs/2403.03954)][[project link](https://3d-diffusion-policy.github.io/)][[code|official](https://github.com/YanjieZe/3D-Diffusion-Policy)][`Shanghai Qizhi + SJTU + THU + Shanghai AI Lab`][This work is also published on [`IEEE 2024 ICRA Workshop 3D Manipulation`](https://openreview.net/forum?id=Xjvcxow3sM).][Did not conduct experiments on benchmarks `RLBench` and `CALVIN`]

* **R&D / Render&Diffuse(RSS2024)(arxiv2024.05)** Render and Diffuse: Aligning Image and Action Spaces for Diffusion-based Behaviour Cloning [[arxiv link](https://arxiv.org/abs/2405.18196)][[project link](https://vv19.github.io/render-and-diffuse/)][`Dyson Robot Learning Lab + Imperial College London`][It compared to methods [`ACT`](https://tonyzhaozh.github.io/aloha/) and [`Diffusion Policy`](https://diffusion-policy.cs.columbia.edu/) on `RLBench`; It did not consider adding the 3D information into inputs.]
 
* **IMOP(RSS2024)(arxiv2024.05)** One-Shot Imitation Learning with Invariance Matching for Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2405.13178)][[project link](https://mlzxy.github.io/imop/)][[code|official](https://github.com/mlzxy/imop)][`Rutgers University`, `Invariance-Matching One-shot Policy Learning (IMOP)`][only tested on the dataset `RLBench`, and obtained inferior results than `3D Diffuser Actor`][`Learning from action labels free human videos`]

* ❤**ConsistencyPolicy(RSS2024)(arxiv2024.05)** Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation [[arxiv link](https://arxiv.org/abs/2405.07503)][[project link](https://consistency-policy.github.io/)][[code|official](https://github.com/Aaditya-Prasad/Consistency-Policy/)][`Stanford University + Princeton University`; `Consistency Policy` accelerates `Diffusion Policy `for real time inference on compute constrained robotics platforms.`]

* **MPI(RSS2024)(arxiv2024.06)** Learning Manipulation by Predicting Interaction [[arxiv link](https://arxiv.org/abs/2406.00439)][[project link](https://opendrivelab.com/MPI/)][[code|official](https://github.com/OpenDriveLab/MPI)][`Shanghai AI Lab + SJTU + Renmin University of China + PKU + Northwestern Polytechnical University`][It is tested on the benchmark `Franka Kitchen`][Given a pair of `keyframes` representing the `initial and final states`, along with `language instructions`, our algorithm `predicts the transition frame` and `detects the interaction object`, respectively. ]

* **RVT-2(RSS2024)(arxiv2024.06)** RVT-2: Learning Precise Manipulation from Few Examples [[arxiv link](https://arxiv.org/abs/2406.08545)][[project link](https://robotic-view-transformer-2.github.io/)][[code|official](https://github.com/nvlabs/rvt)][`NVIDIA`; `Dieter Fox`][It is largely based on their predecessor [`RVT`](https://robotic-view-transformer.github.io/) to make it more `performant`, `precise` and `fast`.]

* 👍**DrEureka(RSS2024)(arxiv2024.06)** DrEureka: Language Model Guided Sim-to-Real Transfer [[arxiv link](https://arxiv.org/abs/2406.01967)][[project link](https://eureka-research.github.io/dr-eureka/)][[code|official](https://github.com/eureka-research/DrEureka)][`UPenn + NVIDIA + UT Austin`][It is based on the `Isaac-Gym`; Our `LLM-guided sim-to-real` approach requires only the `physics simulation` for the target task and `automatically constructs suitable reward functions` and `domain randomization distributions` to support real-world transfer.]

* **MDT-Policy(RSS2024)(arxiv2024.07)** Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals [[arxiv link](https://arxiv.org/abs/2407.05996)][[project link](https://intuitive-robots.github.io/mdt_policy/)][[code|official](https://github.com/intuitive-robots/mdt_policy)][`Intuitive Robots Lab (IRL), Karlsruhe Institute of Technology`][It tested on benchmarks `CALVIN` and `LIBERO`.]

* 👍**MOKA(RSS2024)(arxiv2024.03)** MOKA: Open-World Robotic Manipulation through Mark-based Visual Prompting [[arxiv link](https://arxiv.org/abs/2403.03174)][[project link](https://moka-manipulation.github.io/)][[code|official](https://github.com/moka-manipulation/moka)][`Berkeley AI Research, UC Berkeley`; `Pieter Abbeel + Sergey Levine`]

* **ScrewMimic(RSS2024, Outstanding Student Paper Award Finalist)(arxiv2024.05)** ScrewMimic: Bimanual Imitation from Human Videos with Screw Space Projection [[arxiv link](https://arxiv.org/abs/2405.03666)][[project link](https://robin-lab.cs.utexas.edu/ScrewMimic/)][[code|official](https://github.com/UT-Austin-RobIn/ScrewMimic)][`The University of Texas at Austin`]

* **HACMan++(RSS2024)(arxiv2024.07)** HACMan++: Spatially-Grounded Motion Primitives for Manipulation [[arxiv link](https://arxiv.org/abs/2407.08585)][[project link](https://sgmp-rss2024.github.io/)][[code|official](https://github.com/JiangBowen0008/HACManPP)][`CMU + Meta`]

* **dmd_diffusion(RSS2024)(arxiv2024.02)** Diffusion Meets DAgger: Supercharging Eye-in-hand Imitation Learning [[arxiv link](https://arxiv.org/abs/2402.17768)][[project link](https://sites.google.com/view/diffusion-meets-dagger)][[code|official](https://github.com/ErinZhang1998/dmd_diffusion)][`University of Illinois at Urbana-Champaign`]

* **RialTo / RialToPolicyLearning(RSS2024)(arxiv2024.03)** Reconciling Reality Through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation [[arxiv link](https://arxiv.org/abs/2403.03949)][[project link](https://real-to-sim-to-real.github.io/RialTo/)][[code|official](https://github.com/real-to-sim-to-real/RialToPolicyLearning)][`Massachusetts Institute of Technology + University of Washington + TU Darmstadt`]

* **Octo(RSS2024)(arxiv2024.05)** Octo: An Open-Source Generalist Robot Policy [[paper link](https://www.roboticsproceedings.org/rss20/p090.pdf)][[arxiv link](https://arxiv.org/abs/2405.12213)][[project link](https://octo-models.github.io/)][[code|official](https://github.com/octo-models/octo)][`UC Berkeley + Stanford + CMU + Google DeepMind`][based on `RT-1-X` and `RT-2-X`; the low-level action policy is based on `Diffusion Policy`]

* **Robo-ABC(ECCV2024)(arxiv2024.01)** Robo-ABC: Affordance Generalization Beyond Categories via Semantic Correspondence for Robot Manipulation [[arxiv link](https://arxiv.org/abs/2401.07487)][[project link](https://tea-lab.github.io/Robo-ABC)][[code|official](https://github.com/TEA-Lab/Robo-ABC)][`Shanghai Qi Zhi Institute + THU + SJTU`]

* **ManiGaussian(ECCV2024)(arxiv2024.03)** ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2403.08321)][[project link](https://guanxinglu.github.io/ManiGaussian/)][[code|official](https://github.com/GuanxingLu/ManiGaussian)][[weixin blogs](https://mp.weixin.qq.com/s/HFaEoJFSkiECwsqLcJVbwg)][`PKU-SZ + CMU + PKU`][largely based on `PerAct`, `GNFactor`, and many `3DGS` projects]

* **Track2Act(ECCV2024)(arxiv2024.05)** Track2Act: Predicting Point Tracks from Internet Videos Enables Diverse Zero-shot Manipulation [[arxiv link](https://arxiv.org/abs/2405.01527)][[project link](https://homangab.github.io/track2act/)][[code|official](https://github.com/homangab/Track-2-Act/)][`Carnegie Mellon University + University of Washington + Meta`][The first author is [`Homanga Bharadhwaj`](https://homangab.github.io/) who has given a position paper in `ICML2024` named [Position: Scaling Simulation is Neither Necessary Nor Sufficient for In-the-Wild Robot Manipulation](https://proceedings.mlr.press/v235/bharadhwaj24a.html)]

* **RoboTwin(ECCV Workshop 2024 Best Paper)(arxiv2024.09)** RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins [[arxiv link](https://arxiv.org/abs/2409.02920)][[project link](https://robotwin-benchmark.github.io/early-version)][[code|official](https://github.com/TianxingChen/RoboTwin)][`The University of Hong Kong + AgileX Robotics + Shanghai AI Laboratory + Shenzhen University + Institute of Automation, Chinese Academy of Sciences`; `Ping Luo`][`AgileX Robotics (松灵机器人)`]

* **D3Fields(CoRL2024, Oral)(arxiv2023.09)** D3Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Rearrangement [[openreview link](https://openreview.net/forum?id=Uaaj4MaVIQ)][[arxiv link](https://arxiv.org/abs/2309.16118)][[project link](https://robopil.github.io/d3fields/)][[code|official](https://github.com/WangYixuan12/d3fields)]

* 👍👍**3D Diffuser Actor(CoRL2024)(arxiv2024.02)** 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations [[openreview link](https://openreview.net/forum?id=gqCQxObVz2)][[arxiv link](https://arxiv.org/abs/2402.10885)][[project link](https://3d-diffuser-actor.github.io/)][[code|official](https://github.com/nickgkan/3d_diffuser_actor)][`CMU`, using the `Diffusion`; the first authors [`Tsung-Wei Ke`](https://twke18.github.io/) and [`Nikolaos Gkanatsios`](https://nickgkan.github.io/)][This work is largely based on their previous work `Actor3D` and `ChainedDiffuser`, and also closely related with methods `PerAct`, `DiffusionPolicy`, `RVT` and `GNFactor`][It used `rotary positional embeddings` proposed by [RoFormer](https://arxiv.org/abs/2104.09864) to bulid the `3D Relative Position Denoising Transformer` module.][Comparing to `ChainedDiffuser`, It instead predicts the `next 3D keypose` for the robot’s end-effector alongside the `linking trajectory`, which is a much harder task than linking two given keyposes.][The previous version [3D Diffuser Actor](https://openreview.net/forum?id=UnsLGUCynE) is rejected by `ICLR2024` for being similar to `Actor3D`.]

* **TRANSIC(CoRL2024)(arxiv2024.05)** TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction [[openreview link](https://openreview.net/forum?id=lpjPft4RQT)][[arxiv link](https://arxiv.org/abs/2405.10315)][[project link](https://transic-robot.github.io/)][[code|official](https://github.com/transic-robot/transic)][`Stanford University`; `Jiajun Wu + Li Fei-Fei`]

* **SGRv2(CoRL2024)(arxiv2024.06)** Leveraging Locality to Boost Sample Efficiency in Robotic Manipulation [[openreview link](https://openreview.net/forum?id=Qpjo8l8AFW&noteId=Qpjo8l8AFW)][[arxiv link](https://arxiv.org/abs/2406.10615)][[project link](https://sgrv2-robot.github.io/)][[code|official](https://github.com/TongZhangTHU/sgr)][`THU + Shanghai Qi Zhi + Shanghai AI Lab`]

* **GraspSplats(CoRL2024)(arxiv2024.06)** GraspSplats: Efficient Manipulation with 3D Feature Splatting [[openreview link](https://openreview.net/forum?id=pPhTsonbXq)] [[arxiv link](https://arxiv.org/abs/2409.02084)][[project link](https://graspsplats.github.io/)][[code|official](https://github.com/jimazeyu/GraspSplats)][`UC San Diego`; `Xiaolong Wang`]

* 👍**A3VLM(CoRL2024)(arxiv2024.06)** A3VLM: Actionable Articulation-Aware Vision Language Model [[openreview link](https://openreview.net/forum?id=lyhS75loxe&noteId=lyhS75loxe)][[arxiv link](https://arxiv.org/abs/2406.07549)][[code|official](https://github.com/changhaonan/A3VLM)][`SJTU + Shanghai AI Lab + Rutgers University + Yuandao AI + PKU + CUHK MMLab`]

* **Manipulate-Anything(CoRL2024)(arxiv2024.06)** Manipulate-Anything: Automating Real-World Robots using Vision-Language Models [[openreview link](https://openreview.net/forum?id=2SYFDG4WRA)][[arxiv link](https://arxiv.org/abs/2406.18915)][[project link](https://robot-ma.github.io/)][`University of Washington + NVIDIA + Allen Institute for Artifical Intelligence + Universidad Católica San Pablo`; `Dieter Fox`]

* **LLARVA(CoRL2024)(arxiv2024.06)** LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning [[openreview link](https://openreview.net/forum?id=Q2lGXMZCv8)][[arxiv link](https://arxiv.org/abs/2406.11815)][[project link](https://llarva24.github.io/)][[code|official](https://github.com/Dantong88/LLARVA)][`Berkeley AI Research, UC Berkeley`]

* **VKT(CoRL2024)(arxiv2024.06)** Scaling Manipulation Learning with Visual Kinematic Chain Prediction [[openreview link](https://openreview.net/forum?id=Yw5QGNBkEN)][[arxiv link](https://arxiv.org/abs/2406.07837)][[project link](https://mlzxy.github.io/visual-kinetic-chain/)][[code|official](https://github.com/mlzxy/visual-kinetic-chain)][`Rutgers University`][The proposed `Visual Kinematics Transformer (VKT)` is a `convolution-free` architecture that supports an `arbitrary number of camera viewpoints`, and that is trained with a single objective of `forecasting kinematic structures` through optimal `point-set matching`.]

* 👍**RoboPoint(CoRL2024)(arxiv2024.06)** RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics [[openreview link](https://openreview.net/forum?id=GVX6jpZOhU)][[arxiv link](https://arxiv.org/abs/2406.10721)][[project link](https://robo-point.github.io/)][[code|official](https://github.com/wentaoyuan/RoboPoint)][`University of Washington + NVIDIA`; `Dieter Fox`][ROBOPOINT is a `VLM` that predicts `image keypoint affordances` given `language instruction`s.]

* 👍**RAM(CoRL2024, Oral)(arxiv2024.07)** RAM: Retrieval-Based Affordance Transfer for Generalizable Zero-Shot Robotic Manipulation [[openreview link](https://openreview.net/forum?id=8LPXeGhhbH)][[arxiv link](https://arxiv.org/abs/2407.04689)][[project link](https://yxkryptonite.github.io/RAM/)][[code|official](https://github.com/yxKryptonite/RAM_code)][`University of Southern California + Peking University + Stanford University`; `He Wang`]

* 👍**Im2Flow2Act(CoRL2024)(arxiv2024.07)** Flow as the Cross-domain Manipulation Interface [[openreview link](https://openreview.net/forum?id=cNI0ZkK1y)][[arxiv link](https://arxiv.org/abs/2407.15208)][[project link](https://im-flow-act.github.io/)][`Stanford University + Columbia University + JP Morgan AI Research + Carnegie Mellon University`; `Shuran Song`]

* 👍**EquiBot(CoRL2024)(arxiv2024.07)** EquiBot: SIM(3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning [[openreview link](https://openreview.net/forum?id=ueBmGhLOXP)][[arxiv link](https://arxiv.org/abs/2407.01479)][[project link](https://equi-bot.github.io/)][[code|official](https://github.com/yjy0625/equibot)][`Stanford University`][ This work is largely based on their previous work [(ICRA2024)(arxiv2023.10) EquivAct: SIM(3)-Equivariant Visuomotor Policies beyond Rigid Object Manipulation](https://arxiv.org/abs/2310.16050).][During the human demonstration processing stage, it used `Grounded Segment Anything Model with DEVA` [(ICCV2023)](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) as the `object detection and tracking model` and `HaMeR` [(CVPR2024)](https://github.com/geopavlakos/hamer) as the `hand detection model`.]

* **TieBot(CoRL2024, Oral)(arxiv2024.07)** TieBot: Learning to Knot a Tie from Visual Demonstration through a Real-to-Sim-to-Real Approach [[openreview link](https://openreview.net/forum?id=Si2krRESZb)][[arxiv link](https://arxiv.org/abs/2407.03245)][[project link](https://tiebots.github.io/)][`National University of Singapore +  Shanghai Jiao Tong University + Nanjing University`; `Cewu Lu`][`Learning from action labels free human videos`]

* **Theia(CoRL2024)(arxiv2024.07)** Theia: Distilling Diverse Vision Foundation Models for Robot Learning [[openreview link](https://openreview.net/forum?id=ylZHvlwUcI)][[arxiv link](https://arxiv.org/abs/2407.20179)][[project link](https://theia.theaiinstitute.com/)][[blog weixin](https://mp.weixin.qq.com/s/183HUrtP8Tyru_-akw5y_Q)][[code|official](https://github.com/bdaiinstitute/theia)][`The AI Institute + Stony Brook University`]

* **Open-TeleVision(CoRL2024)(arxiv2024.07)** Open-TeleVision: Teleoperation with Immersive Active Visual Feedback [[openreview link](https://openreview.net/forum?id=Yce2jeILGt)][[arxiv link](https://arxiv.org/abs/2407.01512)][[project link](https://robot-tv.github.io/)][[code|official](https://github.com/OpenTeleVision/TeleVision)][`UC San Diego + MIT`; `Xiaolong Wang`]

* **EquiDiff(CoRL2024, Outstanding Paper Award Finalist)(arxiv2024.07)** Equivariant Diffusion Policy [[openreview link](https://openreview.net/forum?id=wD2kUVLT1g)][[arxiv link](https://arxiv.org/pdf/2407.01812)][[project link](https://equidiff.github.io/)][[code|official](https://github.com/pointW/equidiff)][`Northeastern University + Boston Dynamics AI Institute`]

* **Maniwhere(CoRL2024)(arxiv2024.07)** Learning to Manipulate Anywhere: A Visual Generalizable Framework For Reinforcement Learning [[openreview link](https://openreview.net/forum?id=jart4nhCQr)][[arxiv link](https://arxiv.org/abs/2407.15815)][[project link](https://gemcollector.github.io/maniwhere/)][`THU + SJTU + HKU + PKU +  Shanghai Qi Zhi Institute + Shanghai AI Lab`; `Huaze Xu`]

* **GaussianGBND(CoRL2024)(arxiv2024.08)** Dynamic 3D Gaussian Tracking for Graph-Based Neural Dynamics Modeling [[openreview link](https://openreview.net/forum?id=itKJ5uu1gW)][[project link](https://gaussian-gbnd.github.io/)]

* **ReMix(CoRL2024)(arxiv2024.08)** ReMix: Optimizing Data Mixtures for Large Scale Imitation Learning [[openreview link](https://openreview.net/forum?id=fIj88Tn3fc)][[arxiv link](https://arxiv.org/abs/2408.14037)][[code|official](https://github.com/jhejna/remix)][`Stanford + UC Berkeley`]

* **ACE/ACETeleop(CoRL2024)(arxiv2024.08)** ACE: A Cross-platform Visual-Exoskeletons for Low-Cost Dexterous Teleoperation [[openreview link](https://openreview.net/forum?id=7ddT4eklmQ)][[arxiv link](https://arxiv.org/abs/2408.11805)][[project link](https://ace-teleop.github.io/)][[code|official](https://github.com/ACETeleop/ACETeleop)][`UC San Diego`; `Xiaolong Wang`][Cross-Platform Teleoperation + Autonomous Skills]

* 👍**VISTA(CoRL2024)(arxiv2024.09)** View-Invariant Policy Learning via Zero-Shot Novel View Synthesis [[openreview link](https://openreview.net/forum?id=tqsQGrmVEu)][[arxiv link](https://arxiv.org/abs/2409.03685)][[project link](https://s-tian.github.io/projects/vista/)][[code|official](https://github.com/s-tian/VISTA)][`Stanford University + Toyota Research Institute`; `Jiajun Wu`]

* **ALOHA Unleashed(CoRL2024)(2024.09)** ALOHA Unleashed: A Simple Recipe for Robot Dexterity [[openreview link](https://openreview.net/forum?id=gvdXE7ikHI)][[pdf link](https://aloha-unleashed.github.io/assets/aloha_unleashed.pdf)][[project link](https://aloha-unleashed.github.io/)][[official blog (Robotics team)](https://deepmind.google/discover/blog/advances-in-robot-dexterity/)][`Google DeepMind`; `Chelsea Finn`][using their [ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual Teleoperation](https://aloha-2.github.io/) to operate experiments.]

* **RoVi-Aug(CoRL2024, Oral)(arxiv2024.09)** RoVi-Aug: Robot and Viewpoint Augmentation for Cross-Embodiment Robot Learning [[openreview link](https://openreview.net/forum?id=ctzBccpolr)][[arxiv link](https://arxiv.org/abs/2409.03403)][[project link](https://rovi-aug.github.io/)][`University of California, Berkeley + Toyota Research Institute + Physical Intelligence`]

* **RSRD(CoRL2024, Oral)(arxiv2024.09)** Robot See Robot Do: Imitating Articulated Object Manipulation with Monocular 4D Reconstruction [[openreview link](https://openreview.net/forum?id=2LLu3gavF1)][[arxiv link](https://arxiv.org/abs/2409.18121)][[project link](https://robot-see-robot-do.github.io/)][[code|official](https://github.com/kerrj/rsrd)][`UC Berkeley`]

* **D3RoMa(CoRL2024)(arxiv2024.09)** D3RoMa: Disparity Diffusion-based Depth Sensing for Material-Agnostic Robotic Manipulation [[openreview link](https://openreview.net/forum?id=7E3JAys1xO)][[arxiv link](https://arxiv.org/abs/2409.14365)][[project link](https://pku-epic.github.io/D3RoMa/)][`Peking University + University of California, Berkeley + Stanford University + Galbot + University of Chinese Academy of Sciences + Beijing Academy of Artificial Intelligence`; `He Wang`][It used the `left-right stereo image pair` as input.]

* **ReKep(CoRL2024)(arxiv2024.09)** ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation [[openreview link](https://openreview.net/forum?id=9iG3SEbMnL)][[arxiv link](https://arxiv.org/abs/2409.01652)][[project link](https://rekep-robot.github.io/)][[code|official](https://github.com/huangwl18/ReKep)][`Stanford University + Columbia University`; `Li Fei-Fei`]

* **MILES(CoRL2024)(arxiv2024.10)** MILES: Making Imitation Learning Easy with Self-Supervision [[openreview link](https://openreview.net/forum?id=y8XkuQIrvI)][[arxiv link](https://arxiv.org/abs/2410.19693)][[project link](https://www.robot-learning.uk/miles)][[code|official](https://github.com/gpapagiannis/miles-imitation)][`The Robot Learning Lab, Imperial College London, UK`; `Edward Johns`]

* **GenDP(CoRL2024, Oral)(arxiv2024.10)** GenDP: 3D Semantic Fields for Category-Level Generalizable Diffusion Policy [[openreview link](https://openreview.net/forum?id=7wMlwhCvjS)][[arxiv link](https://arxiv.org/abs/2410.17488)][[project link](https://robopil.github.io/GenDP/)][[code|official](https://github.com/WangYixuan12/gendp)][`Columbia University + University of Illinois Urbana-Champaign + Boston Dynamics AI Institute`][This work is based on `Diffusion Policy`, `robomimic` and `D3Fields`]

* 👍👍**DRRobot(CoRL2024, Oral)(arxiv2024.10)** Differentiable Robot Rendering [[openreview link](https://openreview.net/forum?id=lt0Yf8Wh5O)][[arxiv link](https://arxiv.org/abs/2410.13851)][[project link](https://drrobot.cs.columbia.edu/)][[code|official](https://github.com/cvlab-columbia/drrobot)][`Columbia University + Stanford University`; `Shuran Song`]

* 👍👍**ACDC(CoRL2024)(arxiv2024.10)** ACDC: Automated Creation of Digital Cousins for Robust Policy Learning [[openreview link](https://openreview.net/forum?id=7c5rAY8oU3)][[[arxiv link](https://arxiv.org/abs/2410.07408)][[project link](https://digital-cousins.github.io/)][[code|official](https://github.com/cremebrule/digital-cousins)][`Stanford University`; `Jiajun Wu + Li Fei-Fei`][`Digital Cousins`]

* **OKAMI(CoRL2024, Oral)(arxiv2024.10)** OKAMI: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation [[openreview link](https://openreview.net/forum?id=URj5TQTAXM)][[arxiv link](https://arxiv.org/abs/2410.11792)][[project link](https://sites.google.com/view/okami-corl2024)][`UT Austin + NVIDIA Research`; `Yuke Zhu`][It enables a humanoid robot to imitate manipulation skills from `a single human video demonstration`.]

* **ObjDex(CoRL2024)** Object-Centric Dexterous Manipulation from Human Motion Data [[openreview link](https://openreview.net/forum?id=KAzku0Uyh1)][[project link](https://sites.google.com/view/obj-dex)][`Stanford University + Peking University`]

* 👍**SplatSim(CoRL Workshop)(arxiv2024.09)** SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting [[arxiv link](https://arxiv.org/abs/2409.10161)][[project link](https://splatsim.github.io/)][`CMU`][`Use Gaussian Splatting as a Renderer over Existing Simulators`]


* **HPT(NIPS2024, Spotlight)(arxiv2024.09)** Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers [[openreview link](https://openreview.net/forum?id=Pf7kdIjHRf)][[arxiv link](https://arxiv.org/abs/2409.20537)][[project link](https://liruiw.github.io/hpt/)][[code|official](https://github.com/liruiw/HPT)][`MIT CSAIL + FAIR`; `Kaiming He`]

* **CLOVER(NIPS2024)(arxiv2024.09)** Closed-Loop Visuomotor Control with Generative Expectation for Robotic Manipulation [[openreview link](https://openreview.net/forum?id=1ptdkwZbMG)][[arxiv link](https://arxiv.org/abs/2409.09016)][[project link]()][[code|official](https://github.com/OpenDriveLab/CLOVER)][`Shanghai AI Lab + Shanghai Jiao Tong University + HKU + Tsinghua University`][It followed the methods `AVDC` and `RoboFlamingo`]

* **Any2Policy(NIPS2024)** Any2Policy: Learning Visuomotor Policy with Any-Modality [[openreview link](https://openreview.net/forum?id=8lcW9ltJx9)][`Midea Group`]


* **GeneralFlow(arxiv2024.01)** General Flow as Foundation Affordance for Scalable Robot Learning [[arxiv link](https://arxiv.org/abs/2401.11439)][[project link](https://general-flow.github.io/)][[code|official](https://github.com/michaelyuancb/general_flow)][`Tsinghua University + Shanghai Artificial Intelligence Laboratory + Shanghai Qi Zhi Institute`; `Yang Gao`]

* **Kinematic-Motion-Retargeting(arxiv2024.02)** Kinematic Motion Retargeting for Contact-Rich Anthropomorphic Manipulations [[arxiv link](https://arxiv.org/abs/2402.04820)][`Carnegie Mellon University + Boston Dynamics AI Institute + FAIR at Meta`][Hand motion capture and retargeting]

* **RT-H(arxiv2024.03)** RT-H: Action Hierarchies using Language [[arxiv link](https://arxiv.org/abs/2403.01823)][[project link](https://rt-hierarchy.github.io/)][[blog|weixin](https://mp.weixin.qq.com/s/4eXibz3dOSec1jtaJzP3Mw )][by `Google DeepMind` and `Stanford University`][Its insight is to teach the robot the `language of actions`]

* **VIHE(arxiv2024.03)** VIHE: Virtual In-Hand Eye Transformer for 3D Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2403.11461)][[project link](https://vihe-3d.github.io/)][[code|official](https://github.com/doublelei/VIHE)][`Baidu RAL + Johns Hopkins University`][It has cited `3D Diffuser Actor`, but not compared with it in `RLBench`]

* **DNAct(arxiv2024.03)** DNAct: Diffusion Guided Multi-Task 3D Policy Learning [[arxiv link](https://arxiv.org/abs/2403.04115)][[project link](https://dnact.github.io/)][`UC San Diego`; a work by the `Xiaolong Wang` group][It leverages `neural rendering` to distill `2D semantic features` from foundation models such as `Stable Diffusion` to a `3D space`, which provides a comprehensive semantic understanding regarding the scene.]

* **RiEMann(arxiv2024.03)** RiEMann: Near Real-Time SE(3)-Equivariant Robot Manipulation without Point Cloud Segmentation [[arxiv link](https://arxiv.org/abs/2403.19460)][[project link](https://riemann-web.github.io/)][[code|official](https://github.com/HeegerGao/RiEMann)][`NUS + THU + Shanghai AI Lab + Shanghai Qizhi Institute`; `Huazhe Xu`]

* **LCB(arxiv2024.05)** From LLMs to Actions: Latent Codes as Bridges in Hierarchical Robot Control [[arxiv link](https://arxiv.org/abs/2405.04798)][[project link](https://fredshentu.github.io/LCB_site/)][`University of California Berkeley`][It is tested on benchmarks `LangTable` and `CALVIN`]

* **ReAd(arxiv2024.05)** Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration [[arxiv link](https://arxiv.org/abs/2405.14314)][[project link](https://read-llm.github.io/)][`THU + Shanghai AI Lab + Northwestern Polytechnical University + ZJU`; `Multi-Agent Collaboration`][Reinforced Advantage feedback]

* **ORION(arxiv2024.05)** Vision-based Manipulation from Single Human Video with Open-World Object Graphs [[arxiv link](https://arxiv.org/abs/2405.20321)][[project link](https://ut-austin-rpl.github.io/ORION-release/)][`The University of Texas at Austin + Sony AI`; `Yuke Zhu`][`Learning from action labels free human videos`][We investigate the problem of `imitating robot manipulation` from `a single human video` in the `open-world setting`, where `a robot must learn to manipulate novel objects from one video demonstration`.]

* 👍**ManiCM(arxiv2024.06)** ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2406.01586)][[project link](https://manicm-fast.github.io/)][[code|official](https://github.com/ManiCM-fast/ManiCM)][`THU-SZ + Shanghai AI Lab + CMU`][It is based on `3D Diffusion Policy` and is much better, where DP3 is accelerated via `consistency model`.][It did not conduct experiments on benchmarks `RLBench` and `CALVIN`]

* **OpenVLA(arxiv2024.06)** OpenVLA: An Open-Source Vision-Language-Action Model [[arxiv link](https://arxiv.org/abs/2406.09246)][[project link](https://openvla.github.io/)][[code|official](https://github.com/openvla/openvla)][[SimplerEnv-OpenVLA (not officially)](https://github.com/DelinQu/SimplerEnv-OpenVLA)][`Stanford University + UC Berkeley + Toyota Research Institute + Google DeepMind + Physical Intelligence + MIT`][It has better performance than `RT-1/2/H/X` and `Octo`]

* **SigmaAgent(arxiv2024.06)** Contrastive Imitation Learning for Language-guided Multi-Task Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2406.09738)][[project link](https://teleema.github.io/projects/Sigma_Agent/)][`HKUST-GZ`][This work is partly based on the `RVT`; Sigma-Agent incorporates `contrastive Imitation Learning (contrastive IL)` modules to strengthen `vision-language` and `current-future` representations.]

* **LLaRA(arxiv2024.06)** LLaRA: Supercharging Robot Learning Data for Vision-Language Policy [[arxiv link](https://arxiv.org/abs/2406.20095)][[code|official](https://github.com/LostXine/LLaRA)][`Stony Brook University + University of Wisconsin-Madison`]

* **HumanRobotAlign(arxiv2024.06)** Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2406.14235)][[project link](https://jiaming-zhou.github.io/projects/HumanRobotAlign/)][`HKUST-GZ`]

* **RoboUniView(arxiv2024.06)** RoboUniView: Visual-Language Model with Unified View Representation for Robotic Manipulaiton [[arxiv link](https://arxiv.org/abs/2406.18977)][[project link](https://liufanfanlff.github.io/RoboUniview.github.io/)][[code|official](https://github.com/liufanfanlff/RoboUniview)][`Meituan`][This method is only trained and tested on `CALVIN`, and did not conduct `real robot experiments`.]

* **SpatialBot(arxiv2024.06)** SpatialBot: Precise Spatial Understanding with Vision Language Models [[arxiv link](https://arxiv.org/abs/2406.13642)][[SpatialBench link](https://huggingface.co/datasets/RussRobin/SpatialBench)][[weixin blog](https://mp.weixin.qq.com/s/KL0w16aFycW7OeBV2eKy-Q)][[code|official](https://github.com/BAAI-DCAI/SpatialBot)][`SJTU + Stanford + BAAI + PKU + Oxford + SEU`]

* **Streaming-DP(arxiv2024.06)** Streaming Diffusion Policy: Fast Policy Synthesis with Variable Noise Diffusion Models [[arxiv link](https://arxiv.org/abs/2406.04806)][[project link](https://streaming-diffusion-policy.github.io/)][[code|official](https://github.com/Streaming-Diffusion-Policy/streaming_diffusion_policy)][`Norwegian University of Science and Technology + Harvard University`]

* **R+X(arxiv2024.07)** R+X: Retrieval and Execution from Everyday Human Videos [[arxiv link](https://arxiv.org/abs/2407.12957)][[project link](https://www.robot-learning.uk/r-plus-x)][`The Robot Learning Lab at Imperial College London`; `Edward Johns`][`Learning from action labels free human videos`]

* **NeuralJacobianFields(arxiv2024.07)** Unifying 3D Representation and Control of Diverse Robots with a Single Camera [[arxiv link](https://arxiv.org/abs/2407.08722v1)][[project link](https://sizhe-li.github.io/publication/neural_jacobian_field/)][[code|official](https://github.com/sizhe-li/neural-jacobian-field)][`CSAIL, MIT`]

* **PerAct2(arxiv2024.07)** PerAct2: Benchmarking and Learning for Robotic Bimanual Manipulation Tasks [[arxiv link](https://arxiv.org/abs/2407.00278)][[project link](https://bimanual.github.io/)][[code|official](https://github.com/markusgrotz/peract_bimanual)][[dataset link](https://dataset.cs.washington.edu/fox/bimanual/)][`University of Washington`; `Dieter Fox`][This work extends previous work `PerAct` as well as `RLBench` for `bimanual manipulation` tasks.]

* **Points2Plans(arxiv2024.08)** Points2Plans: From Point Clouds to Long-Horizon Plans with Composable Relational Dynamics [[arxiv link](https://arxiv.org/abs/2408.14769)][[project link](https://sites.google.com/stanford.edu/points2plans)][[code|official](https://github.com/yixuanhuang98/Points2Plans)][`Stanford University + University of Utah + Princeton University + NVIDIA Research`][using `Issac Gym`]

* **BiD-Diffusion(arxiv2024.08)** Bidirectional Decoding: Improving Action Chunking via Closed-Loop Resampling [[arxiv link](https://arxiv.org/abs/2408.17355)][[project link](https://bid-robot.github.io/)][[code|official](https://github.com/YuejiangLIU/bid_diffusion)][`Stanford University`; `Chelsea Finn`]

* **MIFAG(arxiv2024.08)** Learning 2D Invariant Affordance Knowledge for 3D Affordance Grounding [[arxiv link](https://arxiv.org/abs/2408.13024)][[project link](https://goxq.github.io/mifag/)][[code|official](https://github.com/goxq/MIFAG-code)][`University of Science and Technology of China + Shanghai AI Laboratory + Northwestern Polytechnical University + TeleAI, China Telecom Corp Ltd`][`Multi-Image Guided Invariant-Feature-Aware 3D Affordance Grounding (MIFAG)`]


* **DemoStart(arxiv2024.09)** DemoStart: Demonstration-led auto-curriculum applied to sim-to-real with multi-fingered robots [[arxiv link](https://arxiv.org/abs/2409.06613)][[project link](https://sites.google.com/view/demostart)][[official blog (Robotics team)](https://deepmind.google/discover/blog/advances-in-robot-dexterity/)][`Google DeepMind`][`sim-to-real`, `multi-fingers`]

* 👍**RUM(arxiv2024.09)** Robot Utility Models: General Policies for Zero-Shot Deployment in New Environments [[arxiv link](https://arxiv.org/abs/2409.05865)][[project link](https://robotutilitymodels.com/)][[code|official](https://github.com/haritheja-e/robot-utility-models/)][`New York University + Hello Robot Inc + Meta Inc`]

* **DexSim2Real2(arxiv2024.09)** DexSim2Real2: Building Explicit World Model for Precise Articulated Object Dexterous Manipulation [[arxiv link](https://arxiv.org/abs/2409.08750)][[project link](https://jiangtaoran.github.io/dexsim2real2_website/)][`Tsinghua University + JD Explore Academy`]

* **NeuralMP(arxiv2024.09)** Neural MP: A Generalist Neural Motion Planner [[arxiv link](https://arxiv.org/abs/2409.05864)][[project link](https://mihdalal.github.io/neuralmotionplanner/)][[code|official](https://github.com/mihdalal/neuralmotionplanner)][`CMU`]

* **AHA(arxiv2024.09)** AHA: A Vision-Language-Model for Detecting and Reasoning over Failures in Robotic Manipulation [[pdf link](https://robofail-vlm.github.io/Aha_paper.pdf)][[project link](https://robofail-vlm.github.io/)][`NVIDIA + University of Washington + Universidad Católica San Pablo + MIT + Nanyang Technological University + Allen Institute for Artificial Intelligence`; `Dieter Fox`]

* **TinyVLA(arxiv2024.09)** TinyVLA : Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2409.12514)][[project link](https://tiny-vla.github.io/)][`Midea Group + East China Normal University + Shanghai University + Syracuse University + Beijing Innovation Center of Humanoid Robotics`]

* **AxisEst(arxiv2024.09)** Articulated Object Manipulation using Online Axis Estimation with SAM2-Based Tracking [[arxiv link](https://arxiv.org/abs/2409.16287)][[project link](https://hytidel.github.io/video-tracking-for-axis-estimation/)][[code|official](https://github.com/TianxingChen/VideoTracking-For-AxisEst)][`University of Hong Kong + Shenzhen University + Shanghai Jiaotong University + Southern University of Science and Technology`][`Articulated Object Manipulation`]

* **Catch_It(arxiv2024.09)** Catch It! Learning to Catch in Flight with Mobile Dexterous Hands [[arxiv link](https://arxiv.org/abs/2409.10319)][[project link](https://mobile-dex-catch.github.io/)][[code|official](https://github.com/hang0610/Catch_It)][`Shanghai Qi Zhi Institute + Tsinghua University + Shanghai AI Lab + Georgia Institute of Technology + Stanford University`]

* **PRESTO(arxiv2024.09)** PRESTO: Fast motion planning using diffusion models based on key-configuration environment representation [[arxiv link](https://arxiv.org/abs/2409.16012)][[project link](https://kiwi-sherbet.github.io/PRESTO/)][`UT Austin + KAIST`; `Yuke Zhu`]

* **DIAL-MPC(arxiv2024.09)** Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing [[arxiv link](https://arxiv.org/abs/2409.15610)][[project link](https://lecar-lab.github.io/dial-mpc/)][[code|official](https://github.com/LeCAR-Lab/dial-mpc)][`Carnegie Mellon University`][`DIAL-MPC: Diffusion-Inspired Annealing For Legged MPC`]

* **DiscretePolicy(arxiv2024.09)** Discrete Policy: Learning Disentangled Action Space for Multi-Task Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2409.18707)][`Syracuse University`]

* **VLM-Floorplan(arxiv2024.09)** Vision Language Models Can Parse Floor Plan Maps [[arxiv link](https://arxiv.org/abs/2409.12842)][[project link](https://sites.google.com/view/vlm-floorplan/home)][[code|official](https://github.com/bu-air-lab/floor_plan_VLM)][`Binghamton University`]

* **FlowMatchingPolicy(arxiv2024.09)** Affordance-based Robot Manipulation with Flow Matching [[arxiv link](https://arxiv.org/abs/2409.01083)][[project link](https://hri-eu.github.io/flow-matching-policy/)][`Honda Research Institute EU`]

* **MatchPolicy(arxiv2024.09)** Match Policy: A Simple Pipeline from Point Cloud Registration to Manipulation Policies [[arxiv link](https://arxiv.org/abs/2409.15517)][[project link](https://haojhuang.github.io/match_page/)][`Northeastern Univeristy + Worcester Polytechnic Institute`]

* 👍**RACER(arxiv2024.09)** RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning [[arxiv link](https://arxiv.org/abs/2409.14674)][[project link](https://rich-language-failure-recovery.github.io/)][[code|official](https://github.com/sled-group/RACER)][`University of Michigan`]

* **UniAff(arxiv2024.09)** UniAff: A Unified Representation of Affordances for Tool Usage and Articulation with Vision-Language Models [[arxiv link](https://arxiv.org/abs/2409.20551)][[project link](https://sites.google.com/view/uni-aff/home)][`SJTU + HKUST + NUS + Rutgers University`; `Cewu Lu`]

* **flow-matching-policy(arxiv2024.09)** Affordance-based Robot Manipulation with Flow Matching [[arxiv link](https://arxiv.org/abs/2409.01083)][[project link](https://hri-eu.github.io/flow-matching-policy/)][[code|official](https://github.com/HRI-EU/flow-matching-policy)][`Honda Research Institute EU`]

* **Scalable-DP(arxiv2024.09)** Scalable Diffusion Policy: Scale Up Diffusion Policy via Transformers for Visuomotor Learning [[arxiv link](https://arxiv.org/abs/2409.14411)][[project link](https://scaling-diffusion-policy.github.io/)][[code|official]()][`Midea Group + East China Normal University + Standford University, + Shanghai University`]


* **BiDexHD(arxiv2024.10)** Learning Diverse Bimanual Dexterous Manipulation Skills from Human Demonstrations [[arxiv link](https://arxiv.org/abs/2410.02477)][[project link](https://sites.google.com/view/bidexhd)][`Peking University`][It is based on the `TACO Dataset` and `Isaac Gym`]

* **GEMBench(arxiv2024.10)** Towards Generalizable Vision-Language Robotic Manipulation: A Benchmark and LLM-guided 3D Policy [[arxiv link](https://arxiv.org/abs/2410.01345)][[project link](https://www.di.ens.fr/willow/research/gembench/)][[code|official](https://github.com/vlc-robot/robot-3dlotus/)][`CNRS, PSL Research University`, `Shizhe Chen`][It is still based on the `RLBench`]

* **GR-2(arxiv2024.10)** GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation [[arxiv link](https://arxiv.org/abs/2410.06158)][[project link](https://gr2-manipulation.github.io/)][`Robotics Research Team, ByteDance Research`]

* **LADEV(arxiv2024.10)** LADEV: A Language-Driven Testing and Evaluation Platform for Vision-Language-Action Models in Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2410.05191)][[project link](https://sites.google.com/view/ladev)][`University of Alberta, Edmonto + University of Tokyo`]

* **BUMBLE(arxiv2024.10)** BUMBLE: Unifying Reasoning and Acting with Vision-Language Models for Building-wide Mobile Manipulation [[arxiv link](https://arxiv.org/abs/2410.06237)][[project link](https://robin-lab.cs.utexas.edu/BUMBLE/)][[code|official](https://github.com/UT-Austin-RobIn/BUMBLE)][`The University of Texas at Austin`; `Yuke Zhu`]

* **SBAMs(Humanoids 2024)(arxiv2024.10)** Learning Spatial Bimanual Action Models Based on Affordance Regions and Human Demonstrations [[arxiv link](https://arxiv.org/abs/2410.08848)][`Karlsruhe Institute of Technology`][`Spatial Bimanual Action Models`]

* **ARCap(arxiv2024.10)** ARCap: Collecting High-quality Human Demonstrations for Robot Learning with Augmented Reality Feedback [[arxiv link](https://arxiv.org/abs/2410.08464)][[project link](https://stanford-tml.github.io/ARCap/)][[code|official](https://github.com/Ericcsr/ARCap)][`Stanford University`; `Li Fei-Fei`][This is a portable data collection system that provides visual feedback through `augmented reality (AR)` and `haptic warnings` to guide users in collecting high-quality demonstrations.]

* 👍**iDP3(arxiv2024.10)** Generalizable Humanoid Manipulation with Improved 3D Diffusion Policies [[arxiv link](https://arxiv.org/abs/2410.10803)][[project link](https://humanoid-manipulation.github.io/)][[code|official](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy)][`Stanford University + Simon Fraser University + UPenn + UIUC + CMU`; `Jiajun Wu`]

* **RDT-1B(arxiv2024.10)** RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation [[arxiv link](https://arxiv.org/abs/2410.07864)][[project link](https://rdt-robotics.github.io/rdt-robotics/)][[code|official](https://github.com/thu-ml/RoboticsDiffusionTransformer)][`Tsinghua University`; `Jun Zhu`]

* **Shortcut-models(arxiv2024.10)** One Step Diffusion via Shortcut Models [[arxiv link](https://arxiv.org/abs/2410.12557)][[project link](https://kvfrans.com/shortcut-models/)][[code|official](https://github.com/kvfrans/shortcut-models)][`UC Berkeley`; `Sergey Levine + Pieter Abeel`]

* **affordance-policy(arxiv2024.10)** Affordance-Centric Policy Learning: Sample Efficient and Generalisable Robot Policy Learning using Affordance-Centric Task Frames [[arxiv link](https://arxiv.org/abs/2410.12124)][[project link](https://affordance-policy-policy.github.io/)][`QUT Centre for Robotics + University of Adelaide`]

* **Data-Scaling-Laws(arxiv2024.10)** Data Scaling Laws in Imitation Learning for Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2410.18647)][[project link](https://data-scaling-laws.github.io/)][[code|official](https://github.com/Fanqi-Lin/Data-Scaling-Laws)][`Tsinghua University + Shanghai Qi Zhi Institute + Shanghai Artificial Intelligence Laboratory`]

* 👍**DiT-Block-Policy(arxiv2024.10)** The Ingredients for Robotic Diffusion Transformers [[arxiv link](https://arxiv.org/pdf/2410.10088)][[project link](https://dit-policy.github.io/)][[code|official](https://github.com/sudeepdasari/dit-policy)][`Carnegie Mellon University + UC Berkeley`; `Sergey Levine`]

* **robots-pretrain-robots(arxiv2024.10)** Robots Pre-Train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets [[arxiv link](https://arxiv.org/abs/2410.22325)][[project link](https://robots-pretrain-robots.github.io/)][[code|official](https://github.com/luccachiang/robots-pretrain-robots)][`UC San Diego + Tongji University + Shanghai Jiao Tong University + University of Maryland + Tsinghua University`; `Huazhe Xu`]

* **CAGE(arxiv2024.10)** CAGE: Causal Attention Enables Data-Efficient Generalizable Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2410.14974)][[project link](http://cage-policy.github.io/)][[code|official](https://github.com/cage-policy/CAGE)][`Shanghai Jiao Tong University + Shanghai Artificial Intelligence Laboratory`; `Cewu Lu + Hao-Shu Fang`]

* **HuDOR(arxiv2024.10)** HuDOR: Bridging the Human to Robot Dexterity Gap through Object-Oriented Rewards [[arxiv link](https://arxiv.org/abs/2410.23289)][[project link](https://object-rewards.github.io/)][`New York University`]

* **HIL-SERL(arxiv2024.10)** Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning [[arxiv link](https://arxiv.org/abs/2410.21845)][[project link](https://hil-serl.github.io/)][[code|official](https://github.com/rail-berkeley/hil-serl)][`University of California, Berkeley`; `Sergey Levine`]

* 👍**π0(arxiv2024.10)** π0: A Vision-Language-Action Flow Model for General Robot Control [[pdf link](https://www.physicalintelligence.company/download/pi0.pdf)][[arxiv link](https://arxiv.org/abs/2410.24164)][[project link](https://www.physicalintelligence.company/blog/pi0)][[`Physical Intelligence (π)`](https://www.physicalintelligence.company/); `Chelsea Finn` + `Sergey Levine`]

* **VIRT(arxiv2024.10)** VIRT: Vision Instructed Robotic Transformer for Manipulation Learning [[arxiv link](https://arxiv.org/abs/2410.07169)][[project link](https://lizhuoling.github.io/VIRT_webpage/)][[code|official](https://github.com/Lizhuoling/VIRT)][`HKU + CVTE + HUST`]


* **DexH2R(arxiv2024.11)** DexH2R: Task-oriented Dexterous Manipulation from Human to Robots [[arxiv link](https://arxiv.org/abs/2411.04428)][` University of California, Berkeley`]

* **LEGATO(arxiv2024.11)** LEGATO: Cross-Embodiment Imitation Using a Grasping Tool [[arxiv link](http://arxiv.org/abs/2411.03682)][[project link](https://ut-hcrl.github.io/LEGATO/)][[code|official](https://github.com/UT-HCRL/LEGATO)][`1The University of Texas at Austin + The AI Institute`; `Yuke Zhu`]

* **RT-Affordance(arxiv2024.11)** RT-Affordance: Affordances are Versatile Intermediate Representations for Robot Manipulation [[arxiv link](https://arxiv.org/abs/2411.02704)][[project link](https://snasiriany.me/rt-affordance)][`Google DeepMind + The University of Austin at Texas`; `Yuke Zhu`]

* **CogACT(arxiv2024.11)** CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2411.19650)][[project link](https://cogact.github.io/)][[code|official](https://github.com/microsoft/CogACT)][`Microsoft Research Asia + Tsinghua University + USTC + Institute of Microelectronics, CAS`]

* **DexDiffuser(arxiv2024.11)** DexDiffuser: Interaction-aware Diffusion Planning for Adaptive Dexterous Manipulation [[arxiv link](https://arxiv.org/abs/2411.18562)][[project link](https://dexdiffuser.github.io/)][`The University of Hong Kong + UC Berkeley`; `Ping Luo`]

* **Spatially-Visual-Perception(arxiv2024.11)** Spatially Visual Perception for End-to-End Robotic Learning [[arxiv link](https://arxiv.org/abs/2411.17458)][`ZhiCheng AI + Peking University + Harvard University + Zhejiang University`]


* **CARP(arxiv2024.12)** CARP: Visuomotor Policy Learning via Coarse-to-Fine Autoregressive Prediction [[arxiv link](https://arxiv.org/abs/2412.06782)][[project link](https://carp-robot.github.io/)][`Westlake University + Zhejiang University + Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing`]

* **RAPL(arxiv2024.12)** Maximizing Alignment with Minimal Feedback: Efficiently Learning Rewards for Visuomotor Robot Policy Alignment [[arxiv link](https://arxiv.org/abs/2412.04835)][`UC Berkeley + Carnegie Mellon University`][`Representation-Aligned Preference-based Learning (RAPL)`]

* **BimArt(arxiv2024.12)** BimArt: A Unified Approach for the Synthesis of 3D Bimanual Interaction with Articulated Objects [[arxiv link](https://arxiv.org/abs/2412.05066)][`MPII + Google`]


* **** [[openreview link]()][[paper link]()][[arxiv link]()][[project link]()][[code|official]()]




