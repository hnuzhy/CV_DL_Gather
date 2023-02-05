# ⭐Generative Adversarial Network
also named ***Deep Generative Framework***

## Materials

* [(blog) 生成对抗网络 – Generative Adversarial Networks | GAN](https://easyai.tech/ai-definition/gan/)
* [(blog) Test and Train CycleGAN](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb#scrollTo=OzSKIPUByfiN)
* [(CSDNblog) CycleGAN论文的阅读与翻译，无监督风格迁移](https://zhuanlan.zhihu.com/p/45394148)
* [(CSDNblog) 生成对抗网络(四)CycleGAN讲解](https://blog.csdn.net/qq_40520596/article/details/104714762)
* [(blog) What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)


## Other Closely Related Paper

* **VAE(ICLR2014)** Auto-Encoding Variational Bayes [[paper link](https://arxiv.org/abs/1312.6114)][`Auto-Encoders`, `Gaussian Prior`]
* **AAE(ICLR2016)** Adversarial Autoencoders [[paper link](https://arxiv.org/abs/1511.05644)][`Auto-Encoders`, `Gaussian Prior`]
* **WAE(ICLR2018)** Wasserstein Auto-Encoders [[paper link](https://arxiv.org/abs/1711.01558)][`Auto-Encoders`, `Gaussian Prior`]


## Surveys

* **(arxiv2022)** Synthetic Data in Human Analysis: A Survey [[paper link](https://arxiv.org/abs/2208.09191)][`Synthetic Data usually needs GAN`]


## Papers

### ▲ GAN-based

* 👍**GAN(NIPS2014)** Generative Adversarial Networks [[paper link](https://arxiv.org/abs/1406.2661)]

* ❤ **CycleGAN(ICCV2017)** Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [[arxiv link](https://arxiv.org/pdf/1703.10593.pdf)][[project link](https://junyanz.github.io/CycleGAN/)][[Codes|PyTorch(official)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]

* **StyleGAN(CVPR2019)** A Style-Based Generator Architecture for Generative Adversarial Networks [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html)][[codes|official TensorFlow](https://github.com/NVlabs/stylegan)][`NVIDIA`]

* ❤ **CUT(ECCV2020)** Contrastive Learning for Unpaired Image-to-Image Translation [[arxiv link](https://arxiv.org/abs/2007.15651)][[project link](http://taesung.me/ContrastiveUnpairedTranslation/)][[Codes|PyTorch(official)](https://github.com/taesungp/contrastive-unpaired-translation)]

* **StyleGAN2(CVPR2020)** Analyzing and Improving the Image Quality of StyleGAN [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html)][[codes|official TensorFlow](https://github.com/NVlabs/stylegan2)][[codes|unofficial PyTorch 1](https://github.com/rosinality/stylegan2-pytorch)][[codes|unofficial PyTorch 2](https://github.com/lucidrains/stylegan2-pytorch)][`NVIDIA`]

* **StyleGAN2-ADA(NIPS2020)** Training Generative Adversarial Networks with Limited Data [[paper link](https://arxiv.org/abs/2006.06676)][[code|official PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch)][[code|official TensorFlow](https://github.com/NVlabs/stylegan2-ada/)][`NVIDIA`]

* **StyleGAN3(NIPS2021)** Alias-Free Generative Adversarial Networks [[paper link](https://proceedings.neurips.cc/paper/2021/hash/076ccd93ad68be51f23707988e934906-Abstract.html)][[code|official PyTorch](https://github.com/NVlabs/stylegan3)][`NVIDIA`]

* **DepthGAN(ECCV2022 Oral)** 3D-Aware Indoor Scene Synthesis with Depth Priors [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19787-1_23)][[project link](https://vivianszf.github.io/depthgan/)]

* **Long-RangeGAN(arxiv2022)** Lightweight Long-Range Generative Adversarial Networks [[paper link](https://arxiv.org/abs/2209.03793)]

* **StyleGAN-Human(arxiv2022)** StyleGAN-Human: A Data-Centric Odyssey of Human Generation [[paper link](https://arxiv.org/abs/2204.11823)][[project link](https://stylegan-human.github.io/)][[code|official PyTorch](https://github.com/stylegan-human/StyleGAN-Human)]


### ▲ Diffusion-based

* ❤ **GET3D(NIPS2022)** GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images [[paper link](https://nv-tlabs.github.io/GET3D/assets/paper.pdf)][[project link](https://nv-tlabs.github.io/GET3D/)][[codes|official PyTorch](https://github.com/nv-tlabs/GET3D)][`NVIDIA`]

* ❤ **SCAM(ECCV2022)** SCAM! Transferring humans between images with Semantic Cross Attention Modulation [[paper link](https://arxiv.org/abs/2210.04883)][[project link](https://imagine.enpc.fr/~dufourn/publications/scam.html)][[codes|official PyTorch](https://github.com/nicolas-dufour/SCAM)]

* **SDEdit(ICLR2022)** SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations [[paper link](https://arxiv.org/abs/2108.01073)][[project link](https://sde-image-editing.github.io/)][`Partial StyleGAN`]

* **HumanDiffusion(arxiv2022)** HumanDiffusion: a Coarse-to-Fine Alignment Diffusion Framework for Controllable Text-Driven Person Image Generation [[paper link](https://arxiv.org/abs/2211.06235)][`Human related image generation`]

* 👍**DreamFusion(arxiv2022)** DreamFusion: Text-to-3D using 2D Diffusion [[paper link](https://arxiv.org/abs/2209.14988)][[project link](https://dreamfusion3d.github.io/)]

* **Dream3D(arxiv2022)** Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models [[paper link](https://arxiv.org/abs/2212.14704)][[project link](https://bluestyle97.github.io/dream3d/)]


### ▲ NeRF-based

* 👍**NeRF(ECCV2020)** NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis [[paper link](https://dl.acm.org/doi/abs/10.1007/978-3-030-58452-8_24)]

* **NerfCap(TVCG2022)** NerfCap: Human Performance Capture With Dynamic Neural Radiance Fields [[paper link](https://ieeexplore.ieee.org/abstract/document/9870173)]

* **HumanNeRF(CVPR2022)** HumanNeRF: Efficiently Generated Human Radiance Field from Sparse Inputs [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_HumanNeRF_Efficiently_Generated_Human_Radiance_Field_From_Sparse_Inputs_CVPR_2022_paper.html)][`Human related image generation`]

* 👍**Humannerf(CVPR2022 Oral)** HumanNeRF: Free-Viewpoint Rendering of Moving People From Monocular Video [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Weng_HumanNeRF_Free-Viewpoint_Rendering_of_Moving_People_From_Monocular_Video_CVPR_2022_paper.html)][[project link](https://grail.cs.washington.edu/projects/humannerf/)][[code|official](https://github.com/chungyiweng/humannerf)][`Human related image generation`]

* **Block-NeRF(CVPR2022)** Block-NeRF: Scalable Large Scene Neural View Synthesis [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Tancik_Block-NeRF_Scalable_Large_Scene_Neural_View_Synthesis_CVPR_2022_paper.html)][[project link](waymo.com/research/block-nerf)]

* 👍**NeuMan(ECCV2022)** NeuMan: Neural Human Radiance Field from a Single Video [[paper link](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920400.pdf)][[code|official](https://github.com/apple/ml-neuman)][`Human related image generation`]

* **Neural-Sim(ECCV2022)** Neural-Sim: Learning to Generate Training Data with NeRF [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20050-2_28)][[code|official](https://github.com/gyhandy/Neural-Sim-NeRF)]

* ⭐**MoFaNeRF(ECCV2022)** MoFaNeRF:Morphable Facial Neural Radiance Field [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20062-5_16)][[code|official](https://github.com/zhuhao-nju/mofanerf)][`Face or head related NeRF`]

* **headshot(arxiv2022)** Novel View Synthesis for High-fidelity Headshot Scenes [[paper link](https://arxiv.org/abs/2205.15595)][[code|official](https://github.com/showlab/headshot)][`Face or head related NeRF`]

* **FLNeRF(arxiv2022)** FLNeRF: 3D Facial Landmarks Estimation in Neural Radiance Fields [[paper link](https://arxiv.org/abs/2211.11202)][[project link](https://github.com/ZHANG1023/FLNeRF)][`Face or head related NeRF`]

* **HexPlane(arxiv2023)** HexPlane: A Fast Representation for Dynamic Scenes [[paper link](https://arxiv.org/abs/2301.09632)][[project link](https://caoang327.github.io/HexPlane)]

* **K-Planes(arxiv2023)** K-Planes: Explicit Radiance Fields in Space, Time, and Appearance  [[paper link](https://arxiv.org/abs/2301.10241)][[project link](https://sarafridov.github.io/K-Planes/)]

* **MAV3D(Make-A-Video3D)(arxiv2023)** Text-To-4D Dynamic Scene Generation [[paper link](https://arxiv.org/abs/2301.11280)][[project link](https://make-a-video3d.github.io/)]

