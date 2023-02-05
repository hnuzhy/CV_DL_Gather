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
* **(arxiv2022)** NeRF: Neural Radiance Field in 3D Vision, A Comprehensive Review [[paper link](https://arxiv.org/abs/2210.00379)][`Neural Radiance Field`]

## Papers

### ▲ GAN-based
[Generative Adversarial Network (GAN): collections in paperswithcode wedsite](https://paperswithcode.com/method/gan)

* 👍**GAN(NIPS2014)** Generative Adversarial Networks [[paper link](https://arxiv.org/abs/1406.2661)][`seminal work`, `pioneering work`, `generator and discriminator`]

* **DC-GAN(ICLR2016)** Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks [[paper link](https://arxiv.org/abs/1511.06434)][`noise-to-image`]

* **WGAN(ICML2017)** Wasserstein Generative Adversarial Networks [[paper link](https://proceedings.mlr.press/v70/arjovsky17a.html)][`noise-to-image`]

* **pix2pix(CVPR2017)** Image-To-Image Translation With Conditional Adversarial Networks [[paper link](https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html)][[project link](https://phillipi.github.io/pix2pix/)][[code|official PyTorch](https://github.com/phillipi/pix2pix)][`image-to-image`]

* ❤ **CycleGAN(ICCV2017)** Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [[arxiv link](https://arxiv.org/pdf/1703.10593.pdf)][[project link](https://junyanz.github.io/CycleGAN/)][[Codes|PyTorch(official)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)][`image-to-image`]

* **pix2pixHD(CVPR2018)** High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs [[paper link](https://arxiv.org/pdf/1711.11585.pdf)][[project link](https://tcwang0509.github.io/pix2pixHD/)][[Codes|PyTorch(official)](https://github.com/NVIDIA/pix2pixHD)][`image-to-image`, `NVIDIA`]

* **StyleGAN(CVPR2019)** A Style-Based Generator Architecture for Generative Adversarial Networks [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html)][[codes|official TensorFlow](https://github.com/NVlabs/stylegan)][`image-to-image`, `NVIDIA`]

* ❤ **CUT(ECCV2020)** Contrastive Learning for Unpaired Image-to-Image Translation [[arxiv link](https://arxiv.org/abs/2007.15651)][[project link](http://taesung.me/ContrastiveUnpairedTranslation/)][[Codes|PyTorch(official)](https://github.com/taesungp/contrastive-unpaired-translation)][`image-to-image`]

* **Rotate-and-Render(CVPR2020)** Rotate-and-Render: Unsupervised Photorealistic Face Rotation From Single-View Images [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_Rotate-and-Render_Unsupervised_Photorealistic_Face_Rotation_From_Single-View_Images_CVPR_2020_paper.html)][[code|official PyTorch](https://github.com/Hangz-nju-cuhk/Rotate-and-Render)][`self-supervision`, `consistency regularization`, `profile face frontalization`]

* **StyleGAN2(CVPR2020)** Analyzing and Improving the Image Quality of StyleGAN [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html)][[codes|official TensorFlow](https://github.com/NVlabs/stylegan2)][[codes|unofficial PyTorch 1](https://github.com/rosinality/stylegan2-pytorch)][[codes|unofficial PyTorch 2](https://github.com/lucidrains/stylegan2-pytorch)][`image-to-image`, `NVIDIA`, `Face Image Generation`]

* **StyleGAN2-ADA(NIPS2020)** Training Generative Adversarial Networks with Limited Data [[paper link](https://arxiv.org/abs/2006.06676)][[code|official PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch)][[code|official TensorFlow](https://github.com/NVlabs/stylegan2-ada/)][`NVIDIA`, `image-to-image`]

* **StyleGAN3(NIPS2021)** Alias-Free Generative Adversarial Networks [[paper link](https://proceedings.neurips.cc/paper/2021/hash/076ccd93ad68be51f23707988e934906-Abstract.html)][[code|official PyTorch](https://github.com/NVlabs/stylegan3)][`NVIDIA`, `image-to-image`, `Face Generation`]

* **DepthGAN(ECCV2022 Oral)** 3D-Aware Indoor Scene Synthesis with Depth Priors [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19787-1_23)][[project link](https://vivianszf.github.io/depthgan/)]

* ❤ **EG3D(CVPR2022)** Efficient Geometry-Aware 3D Generative Adversarial Networks [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_Efficient_Geometry-Aware_3D_Generative_Adversarial_Networks_CVPR_2022_paper.html)][[code|official PyTorch](https://github.com/NVlabs/eg3d)][`NVIDIA`, `3D Face Image Generation`]

* **Long-RangeGAN(arxiv2022)** Lightweight Long-Range Generative Adversarial Networks [[paper link](https://arxiv.org/abs/2209.03793)]

* **StyleGAN-Human(arxiv2022)** StyleGAN-Human: A Data-Centric Odyssey of Human Generation [[paper link](https://arxiv.org/abs/2204.11823)][[project link](https://stylegan-human.github.io/)][[code|official PyTorch](https://github.com/stylegan-human/StyleGAN-Human)]


### ▲ Diffusion-based
[Diffusion model: collections in paperswithcode wedsite](https://paperswithcode.com/method/diffusion)

* **DDPM(NIPS2020)** Denoising Diffusion Probabilistic Models [[paper link](https://arxiv.org/abs/2006.11239v2)][[project link](https://hojonathanho.github.io/diffusion/)][[code|official TensorFlow](https://github.com/hojonathanho/diffusion)][[[code|unofficial PyTorch](https://github.com/huggingface/diffusers)][[tutorial unofficial](https://nn.labml.ai/diffusion/ddpm/index.html)][`pioneering work`]

* ❤ **GET3D(NIPS2022)** GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images [[paper link](https://nv-tlabs.github.io/GET3D/assets/paper.pdf)][[project link](https://nv-tlabs.github.io/GET3D/)][[codes|official PyTorch](https://github.com/nv-tlabs/GET3D)][`NVIDIA`]

* ❤ **SCAM(ECCV2022)** SCAM! Transferring humans between images with Semantic Cross Attention Modulation [[paper link](https://arxiv.org/abs/2210.04883)][[project link](https://imagine.enpc.fr/~dufourn/publications/scam.html)][[codes|official PyTorch](https://github.com/nicolas-dufour/SCAM)]

* **SDEdit(ICLR2022)** SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations [[paper link](https://arxiv.org/abs/2108.01073)][[project link](https://sde-image-editing.github.io/)][`Partial StyleGAN`]

* **HumanDiffusion(arxiv2022)** HumanDiffusion: a Coarse-to-Fine Alignment Diffusion Framework for Controllable Text-Driven Person Image Generation [[paper link](https://arxiv.org/abs/2211.06235)][`Human related image generation`]

* 👍**DreamFusion(arxiv2022)** DreamFusion: Text-to-3D using 2D Diffusion [[paper link](https://arxiv.org/abs/2209.14988)][[project link](https://dreamfusion3d.github.io/)]

* **Dream3D(arxiv2022)** Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models [[paper link](https://arxiv.org/abs/2212.14704)][[project link](https://bluestyle97.github.io/dream3d/)]


### ▲ NeRF-based
[Neural Radiance Field (NeRF): collections in paperswithcode wedsite](https://paperswithcode.com/method/nerf)

* 👍**NeRF(ECCV2020)** NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis [[paper link](https://dl.acm.org/doi/abs/10.1007/978-3-030-58452-8_24)][`pioneering work`]

* **HyperNeRF(SIGGRAPH2021)** HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields [[paper link](https://arxiv.org/abs/2106.13228)][[project link](https://hypernerf.github.io/)][[code|official](https://github.com/google/hypernerf)][`human face`]

* **Nerfies(ICCV2021)** Nerfies: Deformable Neural Radiance Fields [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Park_Nerfies_Deformable_Neural_Radiance_Fields_ICCV_2021_paper.html)][[project link](https://nerfies.github.io/)][[code|official](https://github.com/google/nerfies)][`human face`]

* **AnimatableNeRF(ICCV2021)** Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Animatable_Neural_Radiance_Fields_for_Modeling_Dynamic_Human_Bodies_ICCV_2021_paper.html?ref=https://githubhelp.com)][[project link](https://zju3dv.github.io/animatable_nerf/)][[code|official](https://github.com/zju3dv/animatable_nerf)][`human body`]

* **NeuralBody(CVPR2021 Best Paper Candidate)** Neural Body: Implicit Neural Representations With Structured Latent Codes for Novel View Synthesis of Dynamic Humans [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Peng_Neural_Body_Implicit_Neural_Representations_With_Structured_Latent_Codes_for_CVPR_2021_paper.html)][[project link](https://zju3dv.github.io/neuralbody/)][[code|official](https://github.com/zju3dv/neuralbody)][`human body`]

* **NerfCap(TVCG2022)** NerfCap: Human Performance Capture With Dynamic Neural Radiance Fields [[paper link](https://ieeexplore.ieee.org/abstract/document/9870173)]

* **SL-NeRF(CVPR2022)** Structured Local Radiance Fields for Human Avatar Modeling [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Structured_Local_Radiance_Fields_for_Human_Avatar_Modeling_CVPR_2022_paper.html)][`human body`]

* **DoubleField(CVPR2022)** DoubleField: Bridging the Neural Surface and Radiance Fields for High-Fidelity Human Reconstruction and Rendering [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Shao_DoubleField_Bridging_the_Neural_Surface_and_Radiance_Fields_for_High-Fidelity_CVPR_2022_paper.html)][`human body`]

* **Block-NeRF(CVPR2022)** Block-NeRF: Scalable Large Scene Neural View Synthesis [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Tancik_Block-NeRF_Scalable_Large_Scene_Neural_View_Synthesis_CVPR_2022_paper.html)][[project link](waymo.com/research/block-nerf)]

* **HumanNeRF(CVPR2022)** HumanNeRF: Efficiently Generated Human Radiance Field from Sparse Inputs [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_HumanNeRF_Efficiently_Generated_Human_Radiance_Field_From_Sparse_Inputs_CVPR_2022_paper.html)][`Human related image generation`]

* 👍**HumanNeRF(CVPR2022 Oral)** HumanNeRF: Free-Viewpoint Rendering of Moving People From Monocular Video [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Weng_HumanNeRF_Free-Viewpoint_Rendering_of_Moving_People_From_Monocular_Video_CVPR_2022_paper.html)][[project link](https://grail.cs.washington.edu/projects/humannerf/)][[code|official](https://github.com/chungyiweng/humannerf)][`Human related image generation`]

* **HeadNeRF(CVPR2022)** HeadNeRF: A Real-time NeRF-based Parametric Head Model [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Hong_HeadNeRF_A_Real-Time_NeRF-Based_Parametric_Head_Model_CVPR_2022_paper.html)][[code|official](https://github.com/CrisHY1995/headnerf)][`human face`]

* 👍**NeuMan(ECCV2022)** NeuMan: Neural Human Radiance Field from a Single Video [[paper link](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920400.pdf)][[code|official](https://github.com/apple/ml-neuman)][`Human related image generation`]

* **Neural-Sim(ECCV2022)** Neural-Sim: Learning to Generate Training Data with NeRF [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20050-2_28)][[code|official](https://github.com/gyhandy/Neural-Sim-NeRF)]

* ⭐**MoFaNeRF(ECCV2022)** MoFaNeRF:Morphable Facial Neural Radiance Field [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20062-5_16)][[code|official](https://github.com/zhuhao-nju/mofanerf)][`Face or head related NeRF`]

* **headshot(arxiv2022)** Novel View Synthesis for High-fidelity Headshot Scenes [[paper link](https://arxiv.org/abs/2205.15595)][[code|official](https://github.com/showlab/headshot)][`Face or head related NeRF`]

* **FLNeRF(arxiv2022)** FLNeRF: 3D Facial Landmarks Estimation in Neural Radiance Fields [[paper link](https://arxiv.org/abs/2211.11202)][[project link](https://github.com/ZHANG1023/FLNeRF)][`Face or head related NeRF`]

* **HexPlane(arxiv2023)** HexPlane: A Fast Representation for Dynamic Scenes [[paper link](https://arxiv.org/abs/2301.09632)][[project link](https://caoang327.github.io/HexPlane)]

* **K-Planes(arxiv2023)** K-Planes: Explicit Radiance Fields in Space, Time, and Appearance  [[paper link](https://arxiv.org/abs/2301.10241)][[project link](https://sarafridov.github.io/K-Planes/)]

* **MAV3D(Make-A-Video3D)(arxiv2023)** Text-To-4D Dynamic Scene Generation [[paper link](https://arxiv.org/abs/2301.11280)][[project link](https://make-a-video3d.github.io/)]

