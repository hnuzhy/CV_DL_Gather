# ⭐Vision Language Action
*this is the most popluar paradigm for achieving `robot manipulation`, also similar to `image-to-action policy models,` and `state-to-action mappings`*

## ▶Materials

* [**website (CCF)** 具身智能 | CCF专家谈术语 (Cewu Lu)](https://www.ccf.org.cn/Media_list/gzwyh/jsjsysdwyh/2023-07-22/794317.shtml)
* [**website** GraspNet通用物体抓取(GraspNet-1Billion + AnyGrasp + SuctionNet-1Billion + TransCG)](https://graspnet.net/index.html)
* [**Github** Recent LLM-based CV and related works.](https://github.com/DirtyHarryLYL/LLM-in-Vision)
* [**Github** Must-read Papers on Large Language Model(LLM) Agents.](https://github.com/zjunlp/LLMAgentPapers)
* [**Github** Awesome-Robotics-Foundation-Models](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models) [[the survey paper](https://arxiv.org/abs/2312.07843)]
* [**Github** Survey Paper of foundation models for robotics](https://github.com/JeffreyYH/robotics-fm-surve) [[the survey paper](https://arxiv.org/abs/2312.08782)]
* [**Github** SAPIEN Manipulation Skill Framework, a GPU parallelized robotics simulator and benchmark](https://github.com/haosulab/ManiSkill) [[ManiSkill readthedocs](https://maniskill.readthedocs.io/en/latest/index.html)]
* [**Github** Robotic Grasping Papers and Codes - Grasp Detection](https://github.com/rhett-chen/Robotic-grasping-papers?tab=readme-ov-file#3-grasp-detection)
* [**Github** CV & Geometry-based 6DOF Robotic Grasping - 6D Grasp Pose Detection](https://github.com/kidpaul94/My-Robotic-Grasping?tab=readme-ov-file#6d-grasp-pose-detection)


***

## ▶Datasets

### ※ For Robot Grasp Pose

* 👍**GraspNet-1Billion(CVPR2020)** GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html)][[project link](www.graspnet.net)][[code|official](https://github.com/graspnet/graspnet-baseline)][`SJTU`]

* 👍**ACRONYM(ICRA2021)(arxiv2020.11)** ACRONYM: A Large-Scale Grasp Dataset Based on Simulation [[paper link](https://ieeexplore.ieee.org/abstract/document/9560844/)][[arxiv link](https://arxiv.org/abs/2011.09584)][[project link](https://sites.google.com/nvidia.com/graspdataset)][[code|official](https://github.com/NVlabs/acronym)][`NVIDIA + University of Washington`]
 
* **SuctionNet-1Billion(RAL2021)(arxiv2021.03)** SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping [[paper link](https://ieeexplore.ieee.org/abstract/document/9547830/)][[arxiv link](https://arxiv.org/abs/2103.12311)][[project link](https://graspnet.net/suction)][[code|official](https://github.com/graspnet/suctionnet-baseline)][`SJTU`]
   
* **REGRAD(RAL2022)(arxiv2021.04)** REGRAD: A Large-Scale Relational Grasp Dataset for Safe and Object-Specific Robotic Grasping in Clutter
  [[paper link](https://ieeexplore.ieee.org/abstract/document/9681218/)][[arxiv link](https://arxiv.org/abs/2104.14118)][[dataset link](https://stuxjtueducn-my.sharepoint.com/personal/chaser123_stu_xjtu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchaser123%5Fstu%5Fxjtu%5Fedu%5Fcn%2FDocuments%2FREGRAD%5Fv1&ga=1)][[code|official](https://github.com/poisonwine/REGRAD)][`XJTU`; `RElational GRAsps Dataset (REGRAD)`]

* **TransCG(RAL2022)(arxiv2022.02)** TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and a Grasping Baseline [[paper link](https://ieeexplore.ieee.org/abstract/document/9796631)][[arxiv link](https://arxiv.org/abs/2202.08471)][[project link](http://www.graspnet.net/transcg)][[code|official](https://github.com/galaxies99/TransCG)][`SJTU`]

* **DA2-Dataset(RAL2022)(arxiv2022.08)** DA2 Dataset: Toward Dexterity-Aware Dual-Arm Grasping [[paper link](https://ieeexplore.ieee.org/abstract/document/9826816/)][[arxiv link](https://arxiv.org/abs/2208.00408)][[project link](https://sites.google.com/view/da2dataset)][[dataset link](https://sites.google.com/view/da2dataset/dataset)][[code|official](https://sites.google.com/view/da2dataset/code)][`Technical University of Munich +  Tencent Robotics X + Imperial College London + ZJU + JHU`]
 
* 👍**OCRTOC(RAL2022)(arxiv2021.04)** OCRTOC: A Cloud-Based Competition and Benchmark for Robotic Grasping and Manipulation [[paper link](https://ieeexplore.ieee.org/abstract/document/9619915/)][[arxiv link](https://arxiv.org/abs/2104.11446)][[project link](https://www.ocrtoc.org/)][[code|official](https://github.com/OCRTOC/OCRTOC_software_package)][`Alibaba AI Labs + UC San Diego + University of Edinburgh + German Aerospace Center`]
   
* **ClothObjectSet(RAL2022)(arxiv2021.11)** Household Cloth Object Set: Fostering Benchmarking in Deformable Object Manipulation [[paper link](https://ieeexplore.ieee.org/abstract/document/9732698/)][[arxiv link](https://arxiv.org/abs/2111.01527)][[project link](https://www.iri.upc.edu/groups/perception/ClothObjectSet/)][`Spain`]

* 👍**DexGraspNet(ICRA2023)(arxiv2022.10)** DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation [[paper link](https://ieeexplore.ieee.org/abstract/document/10160982/)][[arxiv link](https://arxiv.org/abs/2210.02697)][[project link](https://pku-epic.github.io/DexGraspNet/)][[code|official](https://github.com/PKU-EPIC/DexGraspNet)][`PKU`]

* **MetaGraspNetV2(TASE2023)** MetaGraspNetV2: All-in-One Dataset Enabling Fast and Reliable Robotic Bin Picking via Object Relationship Reasoning and Dexterous Grasping [[paper link](https://ieeexplore.ieee.org/abstract/document/10309974/)][[code|official](https://github.com/maximiliangilles/MetaGraspNet)][`Karlsruher Institut für Technologie (KIT)`]
 
* 👍**Sim-Suction(TRO2023)(arxiv2023.05)** Sim-Suction: Learning a Suction Grasp Policy for Cluttered Environments Using a Synthetic Benchmark [[paper link](https://ieeexplore.ieee.org/abstract/document/10314015)][[arxiv link](https://arxiv.org/abs/2305.16378)][[project link](https://junchengli1.github.io/Sim-Suction/)][[code|official](https://github.com/junchengli1/Sim-Suction-API)][`Purdue University`]
   
* **CoAS-Net(RAL2024)** CoAS-Net: Context-Aware Suction Network With a Large-Scale Domain Randomized Synthetic Dataset [[paper link](https://ieeexplore.ieee.org/abstract/document/10333337)][[code|official](https://github.com/SonYeongGwang/CoAS-Net.git)][`Sungkyunkwan University`]

* 👍**Grasp-Anything(ICRA2024)(arxiv2023.09)** Grasp-Anything: Large-scale Grasp Dataset from Foundation Models [[arxiv link](https://arxiv.org/abs/2309.09818)][[project link](https://grasp-anything-2023.github.io/)][`FPT Software - AIC Lab (Hanoi, Vietnam)`]


### ※ For Robot Manipulation

* 👍**RLBench(RAL2020)(arxiv2019.09)** RLBench: The Robot Learning Benchmark & Learning Environment [[paper link](https://ieeexplore.ieee.org/abstract/document/9001253)][[arxiv link](https://arxiv.org/abs/1909.12271)][[project link](https://sites.google.com/view/rlbench)][[code|official](https://github.com/stepjam/RLBench)][`Dyson Robotics Lab, Imperial College London`]

* **Ravens(TransporterNets)(CoRL2020)(arxiv2020.10)** Transporter Networks: Rearranging the Visual World for Robotic Manipulation [[paper link](https://proceedings.mlr.press/v155/zeng21a.html)][[arxiv link](https://arxiv.org/abs/2010.14406)][[project link](https://transporternets.github.io/)][[code|official](https://github.com/google-research/ravens)][`Robotics at Google`][It trained robotic agents to learn `pick` and `place` with deep learning for `vision-based manipulation` in `PyBullet`.]

* 👍**CALVIN(RAL2022)(Best Paper Award)(arxiv2021.12)** Calvin: A Benchmark for Language-conditioned Policy Learning for Long-horizon Robot Manipulation Tasks [[paper link](https://ieeexplore.ieee.org/abstract/document/9788026/)][[arxiv link](https://arxiv.org/abs/2112.03227)][[project link](http://calvin.cs.uni-freiburg.de/)][[code|official](https://github.com/mees/calvin)][`University of Freiburg, Germany`]

* **VLMbench(NIPS2022 Datasets and Benchmarks)(arxiv2022.06)** VLMbench: A Compositional Benchmark for Vision-and-Language Manipulation [[paper link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/04543a88eae2683133c1acbef5a6bf77-Abstract-Datasets_and_Benchmarks.html)][[arxiv link](https://arxiv.org/abs/2206.08522)][[project link](https://sites.google.com/ucsc.edu/vlmbench/home)][[code|official](https://github.com/eric-ai-lab/vlmbench)][`University of California + University of Michigan`, It proposed the baseline method named `6D-CLIPort`]

* **RT-1(arxiv2022.12)** RT-1: Robotics Transformer for Real-World Control at Scale [[arxiv link](https://arxiv.org/abs/2212.06817)][[project link](https://robotics-transformer1.github.io/)][[released datasets](https://console.cloud.google.com/storage/browser/gresearch/rt-1-data-release)][[code|official](https://github.com/google-research/robotics_transformer)][by `Google DeepMind`]

* 👍**ARNOLD(ICCV2023)(arxiv2023.04)** ARNOLD: A Benchmark for Language-Grounded Task Learning with Continuous States in Realistic 3D Scenes [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Gong_ARNOLD_A_Benchmark_for_Language-Grounded_Task_Learning_with_Continuous_States_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2304.04321)][[project link](https://arnold-benchmark.github.io/)][[code|official](https://github.com/arnold-benchmark/arnold)][[dataset|official](https://drive.google.com/drive/folders/1yaEItqU9_MdFVQmkKA6qSvfXy_cPnKGA)][[challenges|official](https://sites.google.com/view/arnoldchallenge/)][`UCLA + PKU + THU + Columbia University + BIGAI`]

* **RH20T(CoRLW2023)(arxiv2023.07)** RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot [[paper link](https://openreview.net/forum?id=Sg9qzrodL9)][[arxiv link](https://arxiv.org/abs/2307.00595)][[project link](https://rh20t.github.io/)][`SJTU`]

* 👍**BridgeData-V2(CoRL2023)(arxiv2023.08)** BridgeData V2: A Dataset for Robot Learning at Scale [[openreview link](https://openreview.net/forum?id=f55MlAT1Lu)][[paper link](https://proceedings.mlr.press/v229/walke23a.html)][[arxiv link](https://arxiv.org/abs/2308.12952)][[project link](https://rail-berkeley.github.io/bridgedata/)][[code|official](https://github.com/rail-berkeley/bridge_data_v2)][`UC Berkeley + Stanford + Google DeepMind + CMU`][It is based on the `(arxiv2021.09) Bridge data: Boosting generalization of robotic skills with cross-domain datasets` with [[arxiv link](https://arxiv.org/abs/2109.13396)] and [[project link](https://sites.google.com/view/bridgedata)]]

* **LoHoRavens(arxiv2023.10)** LoHoRavens: A Long-Horizon Language-Conditioned Benchmark for Robotic Tabletop Manipulation [[arxiv link](https://arxiv.org/abs/2310.12020)][[project link](https://cisnlp.github.io/lohoravens-webpage/)][[code|official](https://github.com/Shengqiang-Zhang/LoHo-Ravens)][`LMU Munich + TUM`][The code is largely based on method `CLIPort-batchify(CoRL2021)(arxiv2021.09)` and dataset `Ravens(TransporterNets)(CoRL2020)`]

* **D4PAS(arxiv2023.12)** Multi-level Reasoning for Robotic Assembly: From Sequence Inference to Contact Selection [[arxiv link](https://arxiv.org/abs/2312.10571)][`UC Berkeley`; It is a large-scale `Dataset for Part Assembly Sequences (D4PAS)`]

* **Safety-Gymnasium(NIPS2023 Datasets and Benchmarks)(arxiv2023.10)** Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark [[openreview link](https://openreview.net/forum?id=WZmlxIuIGR)][[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html)][[arxiv link](https://arxiv.org/abs/2310.12567)][[project link](https://sites.google.com/view/safety-gymnasium)][[code|official](https://github.com/PKU-Alignment/safety-gymnasium)][`PKU`, Safety-Gymnasium is a `highly scalable` and `customizable` Safe Reinforcement Learning (`SafeRL`) library.]

* 👍**DROID(arxiv2024.03)** DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset [[arxiv link](https://arxiv.org/abs/2403.12945)][[project link](https://droid-dataset.github.io/)][[dataset visualizer](https://droid-dataset.github.io/visualizer/)][[code|official](https://github.com/droid-dataset/droid_policy_learning)][`Stanford + Berkeley + Toyota` and many other universities; It used the [`diffusion policy`](https://diffusion-policy.cs.columbia.edu/) for policy learning]

* 👍**RoboCasa(RSS2024)(arxiv2024.06)** RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots [[arxiv link](https://arxiv.org/pdf/2406.02523)][[project link](https://robocasa.ai/)][[weixin blog](https://mp.weixin.qq.com/s/PPXPbJYru1ZOxgJaMtzDjg)][[zhihu blog](https://zhuanlan.zhihu.com/p/701052987)][[code|official](https://github.com/robocasa/robocasa)][`The University of Texas at Austin + NVIDIA Research`; Real2Sim2Real]

* **Open6DOR(CVPRW2024 Oral)** Open6DOR: Benchmarking Open-instruction 6-DoF Object Rearrangement and A VLM-based Approach [[openreview link](https://openreview.net/forum?id=RclUiexKMt)][[project link](https://pku-epic.github.io/Open6DOR)][`PKU`, by the [`He Wang`](https://hughw19.github.io/) group][This is a work published in the `First Vision and Language for Autonomous Driving and Robotics Workshop`]

* **BRMData(arxiv2024.05)** Empowering Embodied Manipulation: A Bimanual-Mobile Robot Manipulation Dataset for Household Tasks [[arxiv link](https://arxiv.org/abs/2405.18860)][[project link](https://embodiedrobot.github.io/)][[dataset link](http://box.jd.com/sharedInfo/1147DC284DDAEE91DC759E209F58DD60)][`JD Explore Academy`][It proposed `BRMData`, a `Bimanual-mobile Robot Manipulation Dataset` specifically designed for `household applications`.]

* 👍**SimplerEnv(arxiv2024.05)** Evaluating Real-World Robot Manipulation Policies in Simulation [[arxiv link](https://arxiv.org/abs/2405.05941)][[project link](https://simpler-env.github.io/)][[code|official](https://github.com/simpler-env/SimplerEnv)][`UC San Diego + Stanford University + UC Berkeley + Google DeepMind`][Evaluating and reproducing real-world robot manipulation policies (e.g., `RT-1, RT-1-X, Octo`) in simulation under common setups (e.g., `Google Robot, WidowX+Bridge`)]


***

## ▶Papers

### ※ Survey

* **Survey(arxiv2024.05)** A Survey on Vision-Language-Action Models for Embodied AI [[arxiv link](https://arxiv.org/abs/2405.14093)][`CUHK +  Huawei Noah’s Ark Lab`]

* **Survey(arxiv2024.05)** Neural Scaling Laws for Embodied AI [[arxiv link](https://arxiv.org/abs/2405.14005)][`TUM + MIT`][This paper presents the first study to `quantify scaling laws` for `Robot Foundation Models (RFMs)` and the use of `LLMs` in `robotics` tasks.]

* **** [[openreview link]()][[paper link]()][[arxiv link]()][[project link]()][[code|official]()]

### ※ Conference

* **CLIPort(CoRL2021)(arxiv2021.09)** CLIPort: What and Where Pathways for Robotic Manipulation [[openreview link](https://openreview.net/forum?id=9uFiX_HRsIL)][[paper link](https://proceedings.mlr.press/v164/shridhar22a.html)][[arxiv link](https://arxiv.org/abs/2109.12098)][[project link](https://cliport.github.io/)][[code|official](https://github.com/cliport/cliport)][[code|not official - CLIPort-Batchify](github.com/ChenWu98/cliport-batchify)][`University of Washington + NVIDIA`]

* ❤**PerAct(CoRL2022)(arxiv2022.09)** Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation [[openreview link](https://openreview.net/forum?id=PS_eCS_WCvD)][[paper link](https://proceedings.mlr.press/v205/shridhar23a.html)][[arxiv link](https://arxiv.org/abs/2209.05451)][[project link](https://peract.github.io/)][[code|official](https://github.com/peract/peract)][`University of Washington + NVIDIA`][It proposed a 3D policy that `voxelizes the workspace` and detects the `next voxel action` through `global self-attention`.]

* 👍**CoP(ICRA2023)(arxiv2022.09)** Code as Policies: Language Model Programs for Embodied Control [[paper link](https://ieeexplore.ieee.org/abstract/document/10160591)][[arxiv link](https://arxiv.org/abs/2209.07753)][[project link](https://code-as-policies.github.io/)][[code|official](https://github.com/google-research/google-research/tree/master/code_as_policies)][`Robotics at Google`]

* 👍**ProgPrompt(ICRA2023)(arxiv2022.09)** ProgPrompt: Generating Situated Robot Task Plans using Large Language Models [[paper link](https://ieeexplore.ieee.org/abstract/document/10161317)][[arxiv link](https://arxiv.org/abs/2209.11302)][[project link](https://progprompt.github.io/)][[code|official](https://github.com/NVlabs/progprompt-vh)][`University of Southern California + NVIDIA`; It has released code for replicating the results on the `VirtualHome` dataset.]

* **RT-1(arxiv2022.12)** RT-1: Robotics Transformer for Real-World Control at Scale [[arxiv link](https://arxiv.org/abs/2212.06817)][[project link](https://robotics-transformer1.github.io/)][[code|official](https://github.com/google-research/robotics_transformer)][by `Google DeepMind`]

* 👍❤**DiffusionPolicy(RSS2023)(arxiv2023.03)** Diffusion Policy: Visuomotor Policy Learning via Action Diffusion [[paper link](https://www.roboticsproceedings.org/rss19/p026.pdf)][[arxiv link](https://arxiv.org/abs/2303.04137)][[project link](https://diffusion-policy.cs.columbia.edu/)][[code|official](https://github.com/real-stanford/diffusion_policy)][`Columbia University + Toyota Research Institute + MIT`][Did not conduct experiments on benchmarks `RLBench` and `CALVIN`]

* **MV-MWM(ICML2023)(arxiv2023.02)** Multi-View Masked World Models for Visual Robotic Manipulation [[paper link](https://proceedings.mlr.press/v202/seo23a.html)][[arxiv link](https://arxiv.org/abs/2302.02408)][[project link](https://sites.google.com/view/mv-mwm)][[code|official](https://github.com/younggyoseo/MV-MWM)][`KAIST  + Dyson Robot Learning Lab + Google Research + UC Berkeley`; It used the less-popular `TensorFlow 2`]

* **LLM-MCTS(NIPS2023)(arxiv2023.05)** Large Language Models as Commonsense Knowledge for Large-Scale Task Planning [[openreview link](https://openreview.net/forum?id=Wjp1AYB8lH)][[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/65a39213d7d0e1eb5d192aa77e77eeb7-Abstract-Conference.html)][[arxiv link](https://arxiv.org/abs/2305.14078)][[project link](https://llm-mcts.github.io/)][[code|official](https://github.com/1989Ryan/llm-mcts)][`National University of Singapore`][It used `Large Language Models` as both the `commonsense world model` and the `heuristic policy` within the `Monte Carlo Tree Search` framework, enabling better-reasoned `decision-making` for daily tasks.]

* **L2M / Learning-to-Modulate(NIPS2023)(arxiv2023.06)** Learning to Modulate pre-trained Models in RL [[openreview link](https://openreview.net/forum?id=aIpGtPwXny)][[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/77e59fafe99e94f822e79bf9308ec377-Abstract-Conference.html)][[arxiv link](https://arxiv.org/abs/2306.14884)][[code|official](https://github.com/ml-jku/L2M)][`Johannes Kepler University Linz, Austria + Google DeepMind + UCL`; tyr to `adapt` the already trained `RL` models.]

* ❤**ChainedDiffuser(CoRL2023)** ChainedDiffuser: Unifying Trajectory Diffusion and Keypose Prediction for Robotic Manipulation  [[openreview link](https://openreview.net/forum?id=W0zgY2mBTA8)][[paper link](https://proceedings.mlr.press/v229/xian23a.html)][[project link](https://chained-diffuser.github.io/)][[code|official](https://github.com/zhouxian/act3d-chained-diffuser)][`CMU`, using the `Diffusion`; the first authors [`Zhou Xian`](https://www.zhou-xian.com/) and [`Nikolaos Gkanatsios`](https://nickgkan.github.io/)][It proposed to replace `motion planners`, commonly used for keypose to keypose linking, with a `trajectory diffusion model` that conditions on the `3D scene feature cloud` and the `predicted target 3D keypose` to denoise a trajectory from the current to the target keypose.]

* **MOO(CoRL2023)(arxiv2023.03)** Open-World Object Manipulation using Pre-trained Vision-Language Models [[openreview link](https://openreview.net/forum?id=9al6taqfTzr)][[paper link](https://proceedings.mlr.press/v229/stone23a.html)][[arxiv link]()][[project link](https://robot-moo.github.io/)][`Robotics at Google`]

* **RVT(CoRL2023 Oral)(arxiv2023.06)** RVT: Robotic View Transformer for 3D Object Manipulation [[openreview link](https://openreview.net/forum?id=0hPkttoGAf)][[paper link](https://proceedings.mlr.press/v229/goyal23a.html)][[arxiv link](https://arxiv.org/abs/2306.14896)][[project link](https://robotic-view-transformer.github.io/)][[code|official](https://github.com/nvlabs/rvt)][`NVIDIA`][It `re-projects` the input `RGB-D` image to alternative image views, featurizes those and `lifts` the predictions to 3D to `infer 3D locations` for the robot’s end-effector.][It proposed a `3D policy` that deploys a `multi-view transformer` to predict actions and fuses those across views by `back-projecting` to 3D.]

* ❤**Act3D(CoRL2023)(arxiv2023.06)** Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation [[openreview link](https://openreview.net/forum?id=-HFJuX1uqs)][[paper link](https://proceedings.mlr.press/v229/gervet23a.html)][[arxiv link](https://arxiv.org/abs/2306.17817)][[project link](https://act3d.github.io/)][[code|official](https://github.com/zhouxian/act3d-chained-diffuser)][`CMU`, using the `Diffusion`; the first authors [`Theophile Gervet`](https://theophilegervet.github.io/) and [`Zhou Xian`](https://www.zhou-xian.com/) and [`Nikolaos Gkanatsios`](https://nickgkan.github.io/)][It proposed a 3D policy that featurizes the robot’s `3D workspace` using `coarse-to-fine sampling` and `featurization`.]

* **RT-2(CoRL2023)(arxiv2023.07)** RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control [[openreview link](https://openreview.net/forum?id=XMQgwiJ7KSX)][[paper link](https://proceedings.mlr.press/v229/zitkovich23a.html)][[arxiv link](https://arxiv.org/abs/2307.15818)][[project link](https://robotics-transformer2.github.io/)][[code|not official](https://github.com/kyegomez/RT-2)][by `Google DeepMind`; based on `RT-1`; it is trained on top of [`PaLM-E (12B)`](https://palm-e.github.io/); it is also trained on top of [`PaLI-X (55B)`](https://arxiv.org/abs/2305.18565); it plans to use more powerful `VLMs`, such as [`LLaVA (Large Language and Vision Assistant)`](https://llava-vl.github.io/) and `LLaVA-1.5`]

* **GRIF(CoRL2023)(arxiv2023.07)** Goal Representations for Instruction Following: A Semi-Supervised Language Interface to Control [[openreview link](https://openreview.net/forum?id=0bZaUfELuW)][[paper link](https://proceedings.mlr.press/v229/myers23a.html)][[arxiv link](https://arxiv.org/abs/2307.00117)][[project link](https://rail-berkeley.github.io/grif/)][[code|official](https://github.com/rail-berkeley/grif_release)][`University of California Berkeley + Microsoft Research`; It used `Semi-Supervised Learning` and `Contrastive Learning`, but it also used the less-popular `TensorFlow`]

* **GROOT(CoRL2023)(arxiv2023.10)** Learning Generalizable Manipulation Policies with Object-Centric 3D Representations [[openreview link](https://openreview.net/forum?id=9SM6l0HyY_)][[paper link](https://proceedings.mlr.press/v229/zhu23b.html)][[arxiv link](https://arxiv.org/abs/2310.14386)][[project link](https://ut-austin-rpl.github.io/GROOT/)][[code|official](https://github.com/UT-Austin-RPL/GROOT)][`The University of Texas, Austin + Sony AI`; It used the `SAM` for segmenting out target objects.]

* **MimicPlay(CoRL2023 Oral)(arxiv2023.02)** MimicPlay: Long-Horizon Imitation Learning by Watching Human Play [[openreview link](https://openreview.net/forum?id=hRZ1YjDZmTo)][[paper link](https://proceedings.mlr.press/v229/wang23a.html)][[arxiv link](https://arxiv.org/abs/2302.12422)][[project link](https://mimic-play.github.io/)][[code|official](https://github.com/j96w/MimicPlay)][`Stanford + NVIDIA + Georgia Tech + UT Austin + Caltech`, by `Stanford Fei-Fei Li`]

* **VoxPoser(CoRL2023 Oral)(arxiv2023.07)** VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models [[paper link](https://proceedings.mlr.press/v229/huang23b.html)][[arxiv link](https://arxiv.org/abs/2307.05973)][[project link](https://voxposer.github.io/)][[code|official](https://github.com/huangwl18/VoxPoser)][by `Stanford Fei-Fei Li`; It extracts `affordances` and `constraints` from large language models (`LLMs`) and vision-language  models (`VLMs`) to compose `3D value maps`; It needs `Detector+Segmentor+Tracker` and thus is very `slow`]

* **GNFactor(CoRL2023 Oral)(arxiv2023.08)** GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields [[openreview link](https://openreview.net/forum?id=b1tl3aOt2R2)][[paper link](https://proceedings.mlr.press/v229/ze23a.html)][[arxiv link](https://arxiv.org/abs/2308.16891)][[project link](https://yanjieze.com/GNFactor/)][[code|official](https://github.com/YanjieZe/GNFactor)][`SJTU + UC San Diego + `; a work by the `Xiaolong Wang` group][It proposed a 3D policy that co-optimizes a `neural field` for reconstructing the `3D voxels` of the input scene and a `PerAct` module for predicting actions based on `voxel representations`.]

* **RoboCat(TMLR2023)(arxiv2023.06)** RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation [[openreview link](https://openreview.net/forum?id=vsCpILiWHu)][[arxiv link](https://arxiv.org/abs/2306.11706)][`Google DeepMind`]

* **Open X-Embodiment(RT-2-X)(arxiv2023.10)** Open X-Embodiment: Robotic Learning Datasets and RT-X Models [[arxiv link](https://arxiv.org/abs/2310.08864)][[project link](https://robotics-transformer-x.github.io/)][[code|official](https://github.com/google-deepmind/open_x_embodiment)][by `Google DeepMind`]

* **LLM-RL / LLaRP(ICLR2024)(arxiv2023.10)** Large Language Models as Generalizable Policies for Embodied Tasks [[openreview link](https://openreview.net/forum?id=u6imHU4Ebu)][[arxiv link](https://arxiv.org/abs/2310.17722)][[project link](https://llm-rl.github.io/)][[code|official](https://github.com/apple/ml-llarp)][`Apple`; `Large LAnguage model Reinforcement Learning Policy (LLaRP)`]

* 👍**RoboFlamingo(ICLR2024 Spotlight)(arxiv2023.11)** Vision-Language Foundation Models as Effective Robot Imitators [[openreview link](https://openreview.net/forum?id=lFYj0oibGR)][[arxiv link](https://arxiv.org/abs/2311.01378)][[project link](https://roboflamingo.github.io/)][[code|official](https://github.com/RoboFlamingo/RoboFlamingo)][`ByteDance + THU + SJTU 
`; based on the `OpenFlamingo`, and tested on the dataset `CALVIN`]

* 👍**SuSIE(ICLR2024)(arxiv2023.11)** Zero-Shot Robotic Manipulation with Pre-Trained Image-Editing Diffusion Models [[openreview link](https://openreview.net/forum?id=c0chJTSbci)][[arxiv link](https://arxiv.org/abs/2310.10639)][[project link](https://rail-berkeley.github.io/susie/)][[code|official](https://github.com/kvablack/susie)][`UCB + Stanford+ Google
`; using the `InstructPix2Pix` to predict future frames; using the `Diffusion` to predict action; it has beated the previous SOTA `RT-2-X`]

* 👍**GR-1(ICLR2024)(arxiv2023.12)** Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation [[openreview link](https://openreview.net/forum?id=NxoFmGgWC9)][[arxiv link](https://arxiv.org/abs/2312.13139)][[project link](https://gr1-manipulation.github.io/)][[code|official](https://github.com/bytedance/GR-1)][`ByteDance`; it adopted the `GPT-style Transformers (GPT-1)`; it adopted the released `CLIP` and `MAE`; it is pretrained on the large video dataset `Ego4D(CVPR2022)`]

* 👍👍**3D Diffuser Actor(arxiv2024.02)** 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations [[arxiv link](https://arxiv.org/abs/2402.10885)][[project link](https://3d-diffuser-actor.github.io/)][[code|official](https://github.com/nickgkan/3d_diffuser_actor)][`CMU`, using the `Diffusion`; the first authors [`Tsung-Wei Ke`](https://twke18.github.io/) and [`Nikolaos Gkanatsios`](https://nickgkan.github.io/)][This work is largely based on their previous work `Actor3D` and `ChainedDiffuser`, and also closely related with methods `PerAct`, `DiffusionPolicy`, `RVT` and `GNFactor`][It used `rotary positional embeddings` proposed by [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) 
 to bulid the `3D Relative Position Denoising Transformer` module.][Comparing to `ChainedDiffuser`, It instead predicts the `next 3D keypose` for the robot’s end-effector alongside the `linking trajectory`, which is a much harder task than linking two given keyposes.]

* **RT-H(arxiv2024.03)** RT-H: Action Hierarchies using Language [[arxiv link](https://arxiv.org/abs/2403.01823)][[project link](https://rt-hierarchy.github.io/)][[blog|weixin](https://mp.weixin.qq.com/s/4eXibz3dOSec1jtaJzP3Mw )][by `Google DeepMind` and `Stanford University`][Its insight is to teach the robot the `language of actions`]

* **VIHE(arxiv2024.03)** VIHE: Virtual In-Hand Eye Transformer for 3D Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2403.11461)][[project link](https://vihe-3d.github.io/)][[code|official](https://github.com/doublelei/VIHE)][`Baidu RAL + Johns Hopkins University`; It has cited `3D Diffuser Actor`, but not compared with it in `RLBench`]

* **DNAct(arxiv2024.03)** DNAct: Diffusion Guided Multi-Task 3D Policy Learning [[arxiv link](https://arxiv.org/abs/2403.04115)][[project link](https://dnact.github.io/)][`UC San Diego`; a work by the `Xiaolong Wang` group][It leverages `neural rendering` to distill `2D semantic features` from foundation models such as `Stable Diffusion` to a `3D space`, which provides a comprehensive semantic understanding regarding the scene.]

* **ManiGaussian(arxiv2024.03)** ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2403.08321)][[project link](https://guanxinglu.github.io/ManiGaussian/)][[code|official](https://github.com/GuanxingLu/ManiGaussian)][[weixin blogs](https://mp.weixin.qq.com/s/HFaEoJFSkiECwsqLcJVbwg)][`PKU-SZ + CMU + PKU`][largely based on `PerAct`, `GNFactor`, and many `3DGS` projects]

* **LegManip(arxiv2024.03)** Learning Visual Quadrupedal Loco-Manipulation from Demonstrations [[arxiv link](https://arxiv.org/abs/2403.20328)][[project link](https://zhengmaohe.github.io/leg-manip/)][`Shanghai Qi Zhi Institute + University of California, Berkeley + Tsinghua University + HKUST-GZ`][It aims to empower a `quadruped robot` to execute real-world manipulation tasks `using only its legs`; It used `3D Diffusion Policy` as the `high-level planner`.]

* ❤**HDP(CVPR2024)(arxiv2024.03)** Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2403.03890)][[project link](https://yusufma03.github.io/projects/hdp/)][[code|official](https://github.com/dyson-ai/hdp)][`Dyson Robot Learning Lab`][It uses `PerAct` as the `high-level agent`]

* **SUGAR(CVPR2024)(arxiv2024.04)** SUGAR: Pre-training 3D Visual Representations for Robotics [[arxiv link](https://arxiv.org/abs/2404.01491)][[project link](https://cshizhe.github.io/projects/robot_sugar)][[code|official (not available)]()][`INRIA`; the first author [`Shizhe Chen`](https://cshizhe.github.io/); `3D Vision-Language-Action`]

* **SAM-E(ICML2024)** SAM-E: Leveraging Visual Foundation Model with Sequence Imitation for Embodied Manipulation [[paper link](https://sam-embodied.github.io/static/SAM-E.pdf)][[arxiv link]()][[project link](https://sam-embodied.github.io/)][[weixin blog](https://mp.weixin.qq.com/s/bLqyLHzFoBrRBT0jgkmZMw)][[code|official](https://github.com/pipixiaqishi1/SAM-E)][`THU + Shanghai AI Lab + HKUST`][only tested on the dataset `RLBench`, and obtained inferior results than `3D Diffuser Actor`]

* **RISE(arxiv2024.04)** RISE: 3D Perception Makes Real-World Robot Imitation Simple and Effective [[arxiv link](https://arxiv.org/abs/2404.12281)][[project link](https://rise-policy.github.io/)][[code|official](https://github.com/rise-policy/RISE)][`SJTU`; proposed by authors [`Chenxi Wang`](https://github.com/chenxi-wang), [`Hongjie Fang`](https://tonyfang.net/), [`Hao-Shu Fang`](https://fang-haoshu.github.io/), and [`Cewu Lu`](https://www.mvig.org/)][Did not conduct experiments on benchmarks `RLBench` and `CALVIN`, and compared to various baselines (2D: [`ACT`](https://tonyzhaozh.github.io/aloha/) and [`Diffusion Policy`](https://diffusion-policy.cs.columbia.edu/); 3D: [`Act3D`](https://act3d.github.io/) and [`DP3`](https://3d-diffusion-policy.github.io/)) on many tasks][It is an `end-to-end` baseline for real-world imitation learning, which `predicts continuous actions` directly from `single-view point clouds`. ]

* **Octo(arxiv2024.05)** Octo: An Open-Source Generalist Robot Policy [[arxiv link](https://arxiv.org/abs/2405.12213)][[project link](https://octo-models.github.io/)][[code|official](https://github.com/octo-models/octo)][`UC Berkeley + Stanford + CMU + Google DeepMind`][based on `RT-1-X` and `RT-2-X`; the low-level action policy is based on `Diffusion Policy`]

* **LCB(arxiv2024.05)** From LLMs to Actions: Latent Codes as Bridges in Hierarchical Robot Control [[arxiv link](https://arxiv.org/abs/2405.04798)][[project link](https://fredshentu.github.io/LCB_site/)][`University of California Berkeley`][It is tested on benchmarks `LangTable` and `CALVIN`]

* **ReAd / Reinforced Advantage feedback(arxiv2024.05)** Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration [[arxiv link](https://arxiv.org/abs/2405.14314)][[project link](https://read-llm.github.io/)][`THU + Shanghai AI Lab + Northwestern Polytechnical University + ZJU`; `Multi-Agent Collaboration`]

* ❤**DP3(RSS2024)(arxiv2024.03)** 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations [[arxiv link](https://arxiv.org/abs/2403.03954)][[project link](https://3d-diffusion-policy.github.io/)][[code|official](https://github.com/YanjieZe/3D-Diffusion-Policy)][`Shanghai Qizhi + SJTU + THU + Shanghai AI Lab`][This work is also published on [`IEEE 2024 ICRA Workshop 3D Manipulation`](https://openreview.net/forum?id=Xjvcxow3sM).][Did not conduct experiments on benchmarks `RLBench` and `CALVIN`]

* **R&D / Render&Diffuse(RSS2024)(arxiv2024.05)** Render and Diffuse: Aligning Image and Action Spaces for Diffusion-based Behaviour Cloning [[arxiv link](https://arxiv.org/abs/2405.18196)][[project link](https://vv19.github.io/render-and-diffuse/)][`Dyson Robot Learning Lab + Imperial College London`][It compared to methods [`ACT`](https://tonyzhaozh.github.io/aloha/) and [`Diffusion Policy`](https://diffusion-policy.cs.columbia.edu/) on `RLBench`; It did not consider adding the 3D information into inputs.]
 
* **IMOP(RSS2024)(arxiv2024.05)** One-Shot Imitation Learning with Invariance Matching for Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2405.13178)][[project link](https://mlzxy.github.io/imop/)][`Rutgers University`, `Invariance-Matching One-shot Policy Learning (IMOP)`][only tested on the dataset `RLBench`, and obtained inferior results than `3D Diffuser Actor`]

* ❤**ConsistencyPolicy(RSS2024)(arxiv2024.05)** Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation [[arxiv link](https://arxiv.org/abs/2405.07503)][[project link](https://consistency-policy.github.io/)][[code|official](https://github.com/Aaditya-Prasad/Consistency-Policy/)][`Stanford University + Princeton University`; `Consistency Policy` accelerates `Diffusion Policy `for real time inference on compute constrained robotics platforms.`]

* **MPI(RSS2024)(arxiv2024.06)** Learning Manipulation by Predicting Interaction [[arxiv link](https://arxiv.org/abs/2406.00439)][[project link](https://opendrivelab.com/MPI/)][[code|official](https://github.com/OpenDriveLab/MPI)][`Shanghai AI Lab + SJTU + Renmin University of China + PKU + Northwestern Polytechnical University`][It is tested on the benchmark `Franka Kitchen`][Given a pair of `keyframes` representing the `initial and final states`, along with `language instructions`, our algorithm `predicts the transition frame` and `detects the interaction object`, respectively. ]

* **RVT-2(RSS2024)(arxiv2024.06)** RVT-2: Learning Precise Manipulation from Few Examples [[arxiv link]()][[project link](https://robotic-view-transformer-2.github.io/)][[code|official](https://github.com/nvlabs/rvt)][`NVIDIA`][It is largely based on their predecessor [`RVT`](https://robotic-view-transformer.github.io/) to make it more `performant`, `precise` and `fast`.]

* 👍**ManiCM(arxiv2024.06)** ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation [[arxiv link](https://arxiv.org/abs/2406.01586)][[project link](https://manicm-fast.github.io/)][[code|official](https://github.com/ManiCM-fast/ManiCM)][`THU-SZ + Shanghai AI Lab + CMU`][It is based on `3D Diffusion Policy` and is much better, where DP3 is accelerated via `consistency model`.][It did not conduct experiments on benchmarks `RLBench` and `CALVIN`]

* **** [[openreview link]()][[paper link]()][[arxiv link]()][[project link]()][[code|official]()]


