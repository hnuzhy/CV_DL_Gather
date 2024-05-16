# ⭐Vision Language Action
*this is the most popluar paradigm for achieving `robot manipulation`, also similar to `image-to-action policy models,` and `state-to-action mappings`*

## ▶Materials

* [**website (CCF)** 具身智能 | CCF专家谈术语 (Cewu Lu)](https://www.ccf.org.cn/Media_list/gzwyh/jsjsysdwyh/2023-07-22/794317.shtml)
* [**website** GraspNet通用物体抓取(GraspNet-1Billion + AnyGrasp + SuctionNet-1Billion + TransCG)](https://graspnet.net/index.html)

***

## ▶Datasets

* **RLBench(RAL2020)(arxiv2019.09)** RLBench: The Robot Learning Benchmark & Learning Environment [[paper link](https://ieeexplore.ieee.org/abstract/document/9001253)][[arxiv link](https://arxiv.org/abs/1909.12271)][[project link](https://sites.google.com/view/rlbench)][[code|official](https://github.com/stepjam/RLBench)][`Dyson Robotics Lab, Imperial College London`]

* **CALVIN(RAL2022)(Best Paper Award)(arxiv2021.12)** Calvin: A Benchmark for Language-conditioned Policy Learning for Long-horizon Robot Manipulation Tasks [[paper link](https://ieeexplore.ieee.org/abstract/document/9788026/)][[arxiv link](https://arxiv.org/abs/2112.03227)][[project link](http://calvin.cs.uni-freiburg.de/)][[code|official](https://github.com/mees/calvin)][`University of Freiburg, Germany`]

* **ARNOLD(ICCV2023)(arxiv2023.04)** ARNOLD: A Benchmark for Language-Grounded Task Learning with Continuous States in Realistic 3D Scenes [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Gong_ARNOLD_A_Benchmark_for_Language-Grounded_Task_Learning_with_Continuous_States_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2304.04321)][[project link](https://arnold-benchmark.github.io/)][[code|official](https://github.com/arnold-benchmark/arnold)][[dataset|official](https://drive.google.com/drive/folders/1yaEItqU9_MdFVQmkKA6qSvfXy_cPnKGA)][[challenges|official](https://sites.google.com/view/arnoldchallenge/)][`UCLA + PKU + THU + Columbia University + BIGAI`]

* **LoHoRavens(arxiv2023.10)** LoHoRavens: A Long-Horizon Language-Conditioned Benchmark for Robotic Tabletop Manipulation [[arxiv link](https://arxiv.org/abs/2310.12020)][[project link](https://cisnlp.github.io/lohoravens-webpage/)][[code|official](https://github.com/Shengqiang-Zhang/LoHo-Ravens)][`LMU Munich +  TUM`]

***

## ▶Papers

### ※2D Vision-Language-Action

* **RT-1(arxiv2022.12)** RT-1: Robotics Transformer for Real-World Control at Scale [[arxiv link](https://arxiv.org/abs/2212.06817)][[project link](https://robotics-transformer1.github.io/)][[code|official](https://github.com/google-research/robotics_transformer)][by `Google DeepMind`]

* **ChainedDiffuser(CoRL2023)** ChainedDiffuser: Unifying Trajectory Diffusion and Keypose Prediction for Robotic Manipulation  [[openreview link](https://openreview.net/forum?id=W0zgY2mBTA8)][[paper link](https://proceedings.mlr.press/v229/xian23a.html)][[project link](https://chained-diffuser.github.io/)][[code|official](https://github.com/zhouxian/act3d-chained-diffuser)][`CMU`, using the `Diffusion`; the first authors [`Zhou Xian`](https://www.zhou-xian.com/) and [`Nikolaos Gkanatsios`](https://nickgkan.github.io/)]

* **Act3D(CoRL2023)(arxiv2023.06)** Act3D: 3D Feature Field Transformers for Multi-Task Robotic Manipulation [[openreview link](https://openreview.net/forum?id=-HFJuX1uqs)][[paper link](https://proceedings.mlr.press/v229/gervet23a.html)][[arxiv link](https://arxiv.org/abs/2306.17817)][[project link](https://act3d.github.io/)][[code|official](https://github.com/zhouxian/act3d-chained-diffuser)][`CMU`, using the `Diffusion`; the first authors [`Theophile Gervet`](https://theophilegervet.github.io/) and [`Zhou Xian`](https://www.zhou-xian.com/) and [`Nikolaos Gkanatsios`](https://nickgkan.github.io/)]

* **RT-2(arxiv2023.07)** RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control [[arxiv link](https://arxiv.org/abs/2307.15818)][[project link](https://robotics-transformer2.github.io/)][[code|not official](https://github.com/kyegomez/RT-2)][by `Google DeepMind`; based on `RT-1`; it is trained on top of [`PaLM-E (12B)`](https://palm-e.github.io/); it is also trained on top of [`PaLI-X (55B)`](https://arxiv.org/abs/2305.18565); it plans to use more powerful `VLMs`, such as [`LLaVA (Large Language and Vision Assistant)`](https://llava-vl.github.io/) and `LLaVA-1.5`]

* **VoxPoser(CoRL2023 Oral)(arxiv2023.07)** VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models [[paper link](https://proceedings.mlr.press/v229/huang23b.html)][[arxiv link](https://arxiv.org/abs/2307.05973)][[project link](https://voxposer.github.io/)][[code|official](https://github.com/huangwl18/VoxPoser)][by `Stanford Fei-Fei Li`; It extracts `affordances` and `constraints` from large language models (`LLMs`) and vision-language  models (`VLMs`) to compose `3D value maps`; It needs `Detector+Segmentor+Tracker` and thus is very `slow`]

* **Open X-Embodiment(RT-2-X)(arxiv2023.10)** Open X-Embodiment: Robotic Learning Datasets and RT-X Models [[arxiv link](https://arxiv.org/abs/2310.08864)][[project link](https://robotics-transformer-x.github.io/)][[code|official](https://github.com/google-deepmind/open_x_embodiment)][by `Google DeepMind`]

* 👍**RoboFlamingo(ICLR2024 Spotlight)(arxiv2023.11)** Vision-Language Foundation Models as Effective Robot Imitators [[openreview link](https://openreview.net/forum?id=lFYj0oibGR)][[arxiv link](https://arxiv.org/abs/2311.01378)][[project link](https://roboflamingo.github.io/)][[code|official](https://github.com/RoboFlamingo/RoboFlamingo)][`ByteDance + THU + SJTU 
`; based on the `OpenFlamingo`, and tested on the dataset `CALVIN`]

* 👍**SuSIE(ICLR2024)(arxiv2023.11)** Zero-Shot Robotic Manipulation with Pre-Trained Image-Editing Diffusion Models [[openreview link](https://openreview.net/forum?id=c0chJTSbci)][[arxiv link](https://arxiv.org/abs/2310.10639)][[project link](https://rail-berkeley.github.io/susie/)][[code|official](https://github.com/kvablack/susie)][`UCB + Stanford+ Google
`; using the `InstructPix2Pix` to predict future frames; using the `Diffusion` to predict action; it has beated the previous SOTA `RT-2-X`]

* 👍**GR-1(ICLR2024)(arxiv2023.12)** Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation [[openreview link](https://openreview.net/forum?id=NxoFmGgWC9)][[arxiv link](https://arxiv.org/abs/2312.13139)][[project link](https://gr1-manipulation.github.io/)][[code|official](https://github.com/bytedance/GR-1)][`ByteDance`; it adopted the `GPT-style Transformers (GPT-1)`; it adopted the released `CLIP` and `MAE`; it is pretrained on the large video dataset `Ego4D(CVPR2022)`]

* 👍**3D Diffuser Actor(arxiv2024.02)** 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations [[arxiv link](https://arxiv.org/abs/2402.10885)][[project link](https://3d-diffuser-actor.github.io/)][[code|official](https://github.com/nickgkan/3d_diffuser_actor)][`CMU`, using the `Diffusion`; the first authors [`Tsung-Wei Ke`](https://twke18.github.io/) and [`Nikolaos Gkanatsios`](https://nickgkan.github.io/); largely based on their previous work `Actor3D` and `ChainedDiffuser`]

* **RT-H(arxiv2024.03)** RT-H: Action Hierarchies using Language [[arxiv link](https://arxiv.org/abs/2403.01823)][[project link](https://rt-hierarchy.github.io/)][[blog|weixin](https://mp.weixin.qq.com/s/4eXibz3dOSec1jtaJzP3Mw )][by `Google DeepMind` and `Stanford University`; Its insight is to teach the robot the `language of actions`]

* **** [[openreview link]()][[paper link]()][[arxiv link]()][[project link]()][[code|official]()]


### ※3D Vision-Language-Action

* **SUGAR(CVPR2024)(arxiv2024.04)** SUGAR: Pre-training 3D Visual Representations for Robotics [[paper link]()][[arxiv link](https://arxiv.org/abs/2404.01491)][[project link](https://cshizhe.github.io/projects/robot_sugar)][[code|official (not available)]()][`INRIA`; the first author [`Shizhe Chen`](https://cshizhe.github.io/)]

