# ⭐Generative Adversarial Network
also named ***Deep Generative Framework***

# Contents

* **[1) Materials](#1-Materials)**
* **[2) Other Closely Related Paper](#2-Other-Closely-Related-Paper)**
* **[3) Surveys](#3-Surveys)**
* **[4) Papers](#4-Papers)**
  * **[▲ GAN-based](#-GAN-based)**
  * **[▲ Diffusion-based](#-Diffusion-based)**
    * **[▶ Basic Theories](#-Basic-Theories)**
    * **[▶ Direct Image Diffusion](#-Direct-Image-Diffusion)** 
    * **[▶ Text-to-Image Diffusion](#-Text-to-Image-Diffusion)** 
    * **[▶ Text-to-3D Diffusion](#-Text-to-3D-Diffusion)** 
  * **[▲ NeRF-based](#-NeRF-based)**
  * **[▲ Others-based](#-Others-based)**

## 1) Materials

* [(blog) 生成对抗网络 – Generative Adversarial Networks | GAN](https://easyai.tech/ai-definition/gan/)
* [(blog) Test and Train CycleGAN](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb#scrollTo=OzSKIPUByfiN)
* [(CSDNblog) CycleGAN论文的阅读与翻译，无监督风格迁移](https://zhuanlan.zhihu.com/p/45394148)
* [(CSDNblog) 生成对抗网络(四)CycleGAN讲解](https://blog.csdn.net/qq_40520596/article/details/104714762)
* [(blog) What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)


## 2) Other Closely Related Paper

* **VAE(ICLR2014)** Auto-Encoding Variational Bayes [[paper link](https://arxiv.org/abs/1312.6114)][`Auto-Encoders`, `Gaussian Prior`]
* **AAE(ICLR2016)** Adversarial Autoencoders [[paper link](https://arxiv.org/abs/1511.05644)][`Auto-Encoders`, `Gaussian Prior`]
* **WAE(ICLR2018)** Wasserstein Auto-Encoders [[paper link](https://arxiv.org/abs/1711.01558)][`Auto-Encoders`, `Gaussian Prior`]


## 3) Surveys

* **(arxiv2022)** Synthetic Data in Human Analysis: A Survey [[paper link](https://arxiv.org/abs/2208.09191)][`Synthetic Data usually needs GAN`]
* **(arxiv2022)** NeRF: Neural Radiance Field in 3D Vision, A Comprehensive Review [[paper link](https://arxiv.org/abs/2210.00379)][`Neural Radiance Field`]

## 4) Papers

### ▲ GAN-based
[Generative Adversarial Network (GAN): collections in paperswithcode wedsite](https://paperswithcode.com/method/gan)

* 👍**GAN(NIPS2014)** Generative Adversarial Networks [[paper link](https://arxiv.org/abs/1406.2661)][`seminal work`, `pioneering work`, `generator and discriminator`]

* **DC-GAN(ICLR2016)** Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks [[paper link](https://arxiv.org/abs/1511.06434)][`noise-to-image`]

* **WGAN(ICML2017)** Wasserstein Generative Adversarial Networks [[paper link](https://proceedings.mlr.press/v70/arjovsky17a.html)][`noise-to-image`]

* **pix2pix(CVPR2017)** Image-To-Image Translation With Conditional Adversarial Networks [[paper link](https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html)][[project link](https://phillipi.github.io/pix2pix/)][[code|official PyTorch](https://github.com/phillipi/pix2pix)][`image-to-image`]

* ❤**CycleGAN(ICCV2017)** Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [[arxiv link](https://arxiv.org/pdf/1703.10593.pdf)][[project link](https://junyanz.github.io/CycleGAN/)][[Codes|PyTorch(official)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)][`image-to-image`]

* **pix2pixHD(CVPR2018)** High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs [[paper link](https://arxiv.org/pdf/1711.11585.pdf)][[project link](https://tcwang0509.github.io/pix2pixHD/)][[Codes|PyTorch(official)](https://github.com/NVIDIA/pix2pixHD)][`image-to-image`, `NVIDIA`]

* **StyleGAN(CVPR2019)** A Style-Based Generator Architecture for Generative Adversarial Networks [[paper link](https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html)][[codes|official TensorFlow](https://github.com/NVlabs/stylegan)][`image-to-image`, `NVIDIA`]

* ❤**CUT(ECCV2020)** Contrastive Learning for Unpaired Image-to-Image Translation [[arxiv link](https://arxiv.org/abs/2007.15651)][[project link](http://taesung.me/ContrastiveUnpairedTranslation/)][[Codes|PyTorch(official)](https://github.com/taesungp/contrastive-unpaired-translation)][`image-to-image`]

* **Rotate-and-Render(CVPR2020)** Rotate-and-Render: Unsupervised Photorealistic Face Rotation From Single-View Images [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_Rotate-and-Render_Unsupervised_Photorealistic_Face_Rotation_From_Single-View_Images_CVPR_2020_paper.html)][[code|official PyTorch](https://github.com/Hangz-nju-cuhk/Rotate-and-Render)][`self-supervision`, `consistency regularization`, `profile face frontalization`]

* **StyleGAN2(CVPR2020)** Analyzing and Improving the Image Quality of StyleGAN [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html)][[codes|official TensorFlow](https://github.com/NVlabs/stylegan2)][[codes|unofficial PyTorch 1](https://github.com/rosinality/stylegan2-pytorch)][[codes|unofficial PyTorch 2](https://github.com/lucidrains/stylegan2-pytorch)][`image-to-image`, `NVIDIA`, `Face Image Generation`]

* **StyleGAN2-ADA(NIPS2020)** Training Generative Adversarial Networks with Limited Data [[paper link](https://arxiv.org/abs/2006.06676)][[code|official PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch)][[code|official TensorFlow](https://github.com/NVlabs/stylegan2-ada/)][`NVIDIA`, `image-to-image`]

* **CIPS-3D(arxiv2021)** CIPS-3D: A 3D-Aware Generator of GANs Based on Conditionally-Independent Pixel Synthesis [[paper link](https://arxiv.org/abs/2110.09788)][[code|official PyTorch](https://github.com/PeterouZh/CIPS-3D)][`3D Face Image Generation`, `3D-aware GANs based on NeRF`]

* **StyleGAN3(NIPS2021)** Alias-Free Generative Adversarial Networks [[paper link](https://proceedings.neurips.cc/paper/2021/hash/076ccd93ad68be51f23707988e934906-Abstract.html)][[code|official PyTorch](https://github.com/NVlabs/stylegan3)][`NVIDIA`, `image-to-image`, `Face Generation`]

* **DepthGAN(ECCV2022 Oral)** 3D-Aware Indoor Scene Synthesis with Depth Priors [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-19787-1_23)][[project link](https://vivianszf.github.io/depthgan/)]

* ❤**EG3D(CVPR2022)** Efficient Geometry-Aware 3D Generative Adversarial Networks [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_Efficient_Geometry-Aware_3D_Generative_Adversarial_Networks_CVPR_2022_paper.html)][[code|official PyTorch](https://github.com/NVlabs/eg3d)][`NVIDIA`, `3D Face Image Generation`]

* ❤**StyleNeRF(ICLR2022)** StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis [[paper link](https://arxiv.org/abs/2110.08985)][[project link](https://jiataogu.me/style_nerf/)][[code|official PyTorch](https://github.com/facebookresearch/StyleNeRF)][`3D Face Image Generation`, `3D-aware GANs based on NeRF`]

* **Long-RangeGAN(arxiv2022)** Lightweight Long-Range Generative Adversarial Networks [[paper link](https://arxiv.org/abs/2209.03793)]

* **StyleGAN-Human(arxiv2022)** StyleGAN-Human: A Data-Centric Odyssey of Human Generation [[paper link](https://arxiv.org/abs/2204.11823)][[project link](https://stylegan-human.github.io/)][[code|official PyTorch](https://github.com/stylegan-human/StyleGAN-Human)]


### ▲ Diffusion-based
[Diffusion model: collections in paperswithcode wedsite](https://paperswithcode.com/method/diffusion)

#### ▶ Basic Theories

* **DPM(ICML2015)** Deep Unsupervised Learning using Nonequilibrium Thermodynamics [[paper link](http://proceedings.mlr.press/v37/sohl-dickstein15.html)][[code|official](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models)][`pioneering work`][The initial `diffusion probabilistic model`]

* **DDPM(NIPS2020)** Denoising Diffusion Probabilistic Models [[paper link](https://arxiv.org/abs/2006.11239v2)][[project link](https://hojonathanho.github.io/diffusion/)][[code|official TensorFlow](https://github.com/hojonathanho/diffusion)][[[code|unofficial PyTorch](https://github.com/huggingface/diffusers)][[tutorial unofficial](https://nn.labml.ai/diffusion/ddpm/index.html)][`improved DPM`]

* **DDIM(ICLR2021)** Denoising Diffusion Implicit Models [[paper link](https://arxiv.org/abs/2010.02502)][`improved DPM`][`considering strategies to save computation powers when handling high-resolution images`]

* **Score-based Diffusion(ICLR2021 oral)** Score-Based Generative Modeling through Stochastic Differential Equations [[paper link](https://arxiv.org/abs/2011.13456)][`improved DPM`]


#### ▶ Direct Image Diffusion
`Image diffusion methods can directly use pixel colors as training data`.

* **(NIPS2021)** Variational Diffusion Models [[paper link](https://proceedings.neurips.cc/paper/2021/hash/b578f2a52a0229873fefc2a4b06377fa-Abstract.html)][`small-scale image generation using original DPM`]

* **(NIPS2021)** Diffusion Models Beat GANs on Image Synthesis [[paper link](https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html)][`large-scale image generation using original DPM`][[Disco Diffusion](https://github.com/alembics/disco-diffusion) is `a clip-guided implementation of [Diffusion Models Beat GANs on Image Synthesis] to process text prompts`]]

* **FastDPM(ICMLW2021)** On Fast Sampling of Diffusion Probabilistic Models [[paper link](https://openreview.net/forum?id=agj4cdOfrAP)][[arxiv link](https://arxiv.org/abs/2106.00132)][[project link](https://fastdpm.github.io/)][[code|official](https://github.com/FengNiMa/FastDPM_pytorch)][`considering strategies to save computation powers when handling high-resolution images`]

* ❤**GET3D(NIPS2022)** GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images [[paper link](https://nv-tlabs.github.io/GET3D/assets/paper.pdf)][[project link](https://nv-tlabs.github.io/GET3D/)][[codes|official PyTorch](https://github.com/nv-tlabs/GET3D)][`NVIDIA`]

* 👍**LatentDiffusion(CVPR2022 oral)** High-Resolution Image Synthesis With Latent Diffusion Models [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)][[project homepage](https://ommer-lab.com/research/latent-diffusion-models/)][[code|official (latent-diffusion)](https://github.com/CompVis/latent-diffusion)][[code|official (stable-diffusion)](https://github.com/CompVis/stable-diffusion)][from `Latent Diffusion Model (LDM)` further extended to `Stable Diffusion Model (SDM)`][`SDM` is a large scale implementation of `LDM` to achieve text-to-image generation][[Stable Diffusion web UI (unofficial)](https://github.com/AUTOMATIC1111/stable-diffusion-webui)]

* ❤**SCAM(ECCV2022)** SCAM! Transferring humans between images with Semantic Cross Attention Modulation [[paper link](https://arxiv.org/abs/2210.04883)][[project link](https://imagine.enpc.fr/~dufourn/publications/scam.html)][[codes|official PyTorch](https://github.com/nicolas-dufour/SCAM)]

* **CascadedDiffusion(JMLR2022)** Cascaded Diffusion Models for High Fidelity Image Generation [[paper link](https://www.jmlr.org/papers/v23/21-0635.html)][`directly use pyramid-based  methods`]

* **Palette(SIGGRAPH2022)** Palette: Image-to-Image Diffusion Models [[paper link](https://dl.acm.org/doi/abs/10.1145/3528233.3530757)][[arxiv link](https://arxiv.org/abs/2111.05826)][[project link](https://diffusion-palette.github.io/)][`an unified diffusion-based image-to-image translation framework`]

* **PITI(arxiv2022)** Pretraining is All You Need for Image-to-Image Translation [[arxiv link](https://arxiv.org/abs/2205.12952)][[project link](https://tengfei-wang.github.io/PITI/index.html)][[code|official](https://github.com/PITI-Synthesis/PITI)][`a diffusion-based image-to-image translation method that utilizes large-scale pretraining as a way to improve the quality of generated results`]


#### ▶ Text-to-Image Diffusion
`Text-to-image generation is often achieved by encoding text inputs into latent vectors using pretrained language models like [CLIP](http://proceedings.mlr.press/v139/radford21a) (ICML2021 Learning Transferable Visual Models From Natural Language Supervision)`.

* **GLIDE(arxiv2021)(ICML2022)** GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models [[paper link](https://proceedings.mlr.press/v162/nichol22a.html)][[arxiv link](https://arxiv.org/abs/2112.10741)][[codes|official](https://github.com/openai/glide-text2im)][`a text-guided diffusion models supporting both image generating and editing`]

* **Imagen(NIPS2022)** Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding [[paper link](https://openreview.net/forum?id=08Yk-n5l2Al)][[arxiv link](https://arxiv.org/abs/2205.11487)][[project link](https://imagen.research.google/)][`a text-to-image structure that does not use latent images and directly diffuse pixels using a pyramid structure`][`google`]

* 👍**SDEdit(ICLR2022)** SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations [[paper link](https://arxiv.org/abs/2108.01073)][[project link](https://sde-image-editing.github.io/)][`Partial StyleGAN`][`The community of Stable Diffusion call this method img2img`]

* **DiffusionCLIP(CVPR2022)** DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.html)][[code|official](https://github.com/gwang-kim/DiffusionCLIP)]

* **BlendedDiffusion(CVPR2022)** Blended Diffusion for Text-Driven Editing of Natural Images [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Avrahami_Blended_Diffusion_for_Text-Driven_Editing_of_Natural_Images_CVPR_2022_paper.html)][[project link](https://omriavrahami.com/blended-diffusion-page/)]

* **InstructPix2Pix(arxiv2022)** InstructPix2Pix: Learning to Follow Image Editing Instructions [[arxiv link](https://arxiv.org/abs/2211.09800)][[project link](https://www.timothybrooks.com/instruct-pix2pix)]

* **Imagic(arxiv2022)** Imagic: Text-Based Real Image Editing with Diffusion Models [[paper link](https://arxiv.org/abs/2210.09276)][[project link](https://imagic-editing.github.io/)]

* **Prompt-to-Prompt(arxiv2022)** Prompt-to-Prompt Image Editing with Cross Attention Control [[arxiv link](https://arxiv.org/abs/2208.01626)][[project link](https://prompt-to-prompt.github.io/)][[codes|official](https://github.com/google/prompt-to-prompt)][`google`]

* **unCLIP(arxiv2022)** Hierarchical Text-Conditional Image Generation with CLIP Latents [[arxiv link](https://arxiv.org/abs/2204.06125)][`directly use multiple-stage methods`]

* **Textual Inversion(arxiv2022)** An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion [[arxiv link](https://arxiv.org/abs/2208.01618)][[project link](https://textual-inversion.github.io/)][`NVIDIA`, `customize (or personalize) the contents in the generated results`]

* **DreamBooth(arxiv2022)** DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation [[arxiv link](https://arxiv.org/abs/2208.12242)][[project link](https://dreambooth.github.io/)][`google`, `customize (or personalize) the contents in the generated results`]

* **HumanDiffusion(arxiv2022)** HumanDiffusion: a Coarse-to-Fine Alignment Diffusion Framework for Controllable Text-Driven Person Image Generation [[paper link](https://arxiv.org/abs/2211.06235)][`Human related image generation`]

* **SketchGuidedDiffusion(arxiv2022)** Sketch-Guided Text-to-Image Diffusion Models [[arxiv link](https://arxiv.org/abs/2211.13752)][[project link](https://sketch-guided-diffusion.github.io/)]

* ❤**styleganfusio(arxiv2022)** Diffusion Guided Image Generator Domain Adaptation [[arxiv link](https://arxiv.org/abs/2212.04473)][[project link](https://styleganfusion.github.io/)][`StyleGAN` + `Diffusion`, `Domain Adaptation`]

* 👍**T2I-Adapter(arxiv2023)** T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models [[arxiv link](https://arxiv.org/abs/2302.08453)][[codes|official](https://github.com/TencentARC/T2I-Adapter)][`Tencent`]

* 👍**ControlNet(arxiv2023)** Adding Conditional Control to Text-to-Image Diffusion Models [[arxiv link](https://arxiv.org/abs/2302.05543)][[codes|official](https://github.com/lllyasviel/ControlNet)][`an excellent work`] [[sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet): An extension for AUTOMATIC1111's [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)] [[ControlNet-for-Diffusers by haofanwang](https://github.com/haofanwang/ControlNet-for-Diffusers)]


#### ▶ Text-to-3D Diffusion

* 👍**DreamFusion(arxiv2022)** DreamFusion: Text-to-3D using 2D Diffusion [[paper link](https://arxiv.org/abs/2209.14988)][[project link](https://dreamfusion3d.github.io/)]

* **Dream3D(arxiv2022)** Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models [[paper link](https://arxiv.org/abs/2212.14704)][[project link](https://bluestyle97.github.io/dream3d/)]

* **MAV3D(Make-A-Video3D)(arxiv2023)** Text-To-4D Dynamic Scene Generation [[paper link](https://arxiv.org/abs/2301.11280)][[project link](https://make-a-video3d.github.io/)][`Meta AI`, `4D dynamic Neural Radiance Field (NeRF)`, `Diffusion`]

* **RealFusion(arxiv2023)** RealFusion: 360° Reconstruction of Any Object from a Single Image [[arxiv link](https://arxiv.org/abs/2302.10663)][[project link](https://lukemelas.github.io/realfusion/)][[codes|official](https://github.com/lukemelas/realfusion)][inspired by `DreamFields` and `DreamFusion`]



### ▲ NeRF-based
[Neural Radiance Field (NeRF): collections in paperswithcode wedsite](https://paperswithcode.com/method/nerf)

* 👍**NeRF(ECCV2020)** NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis [[paper link](https://dl.acm.org/doi/abs/10.1007/978-3-030-58452-8_24)][[code|official Tensorflow](https://github.com/bmild/nerf)][`pioneering work`]

* **NeRF-W(CVPR2021)** NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Martin-Brualla_NeRF_in_the_Wild_Neural_Radiance_Fields_for_Unconstrained_Photo_CVPR_2021_paper.html)]

* **HyperNeRF(SIGGRAPH2021)** HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields [[paper link](https://arxiv.org/abs/2106.13228)][[project link](https://hypernerf.github.io/)][[code|official](https://github.com/google/hypernerf)][`human face`]

* **Nerfies(ICCV2021)** Nerfies: Deformable Neural Radiance Fields [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Park_Nerfies_Deformable_Neural_Radiance_Fields_ICCV_2021_paper.html)][[project link](https://nerfies.github.io/)][[code|official](https://github.com/google/nerfies)][`human face`]

* **AnimatableNeRF(ICCV2021)** Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Animatable_Neural_Radiance_Fields_for_Modeling_Dynamic_Human_Bodies_ICCV_2021_paper.html?ref=https://githubhelp.com)][[project link](https://zju3dv.github.io/animatable_nerf/)][[code|official](https://github.com/zju3dv/animatable_nerf)][`human body`]

* **4D-Facial-Avatars(CVPR2021 Oral)** Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Gafni_Dynamic_Neural_Radiance_Fields_for_Monocular_4D_Facial_Avatar_Reconstruction_CVPR_2021_paper.html)][[](https://gafniguy.github.io/4D-Facial-Avatars/)][[code|official](https://github.com/gafniguy/4D-Facial-Avatars)][`human face`]

* **HybridNeRF(CVPR2021 Oral)** Learning Compositional Radiance Fields of Dynamic Human Heads [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Learning_Compositional_Radiance_Fields_of_Dynamic_Human_Heads_CVPR_2021_paper.html)][[project link](https://ziyanw1.github.io/hybrid_nerf/)][`human face`]

* **NeuralBody(CVPR2021 Best Paper Candidate)** Neural Body: Implicit Neural Representations With Structured Latent Codes for Novel View Synthesis of Dynamic Humans [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Peng_Neural_Body_Implicit_Neural_Representations_With_Structured_Latent_Codes_for_CVPR_2021_paper.html)][[project link](https://zju3dv.github.io/neuralbody/)][[code|official](https://github.com/zju3dv/neuralbody)][`human body`]

* **NerfCap(TVCG2022)** NerfCap: Human Performance Capture With Dynamic Neural Radiance Fields [[paper link](https://ieeexplore.ieee.org/abstract/document/9870173)]

* **SL-NeRF(CVPR2022)** Structured Local Radiance Fields for Human Avatar Modeling [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Structured_Local_Radiance_Fields_for_Human_Avatar_Modeling_CVPR_2022_paper.html)][`human body`]

* **DoubleField(CVPR2022)** DoubleField: Bridging the Neural Surface and Radiance Fields for High-Fidelity Human Reconstruction and Rendering [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Shao_DoubleField_Bridging_the_Neural_Surface_and_Radiance_Fields_for_High-Fidelity_CVPR_2022_paper.html)][`human body`]

* **Block-NeRF(CVPR2022)** Block-NeRF: Scalable Large Scene Neural View Synthesis [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Tancik_Block-NeRF_Scalable_Large_Scene_Neural_View_Synthesis_CVPR_2022_paper.html)][[project link](waymo.com/research/block-nerf)]

* **HumanNeRF(CVPR2022)** HumanNeRF: Efficiently Generated Human Radiance Field from Sparse Inputs [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_HumanNeRF_Efficiently_Generated_Human_Radiance_Field_From_Sparse_Inputs_CVPR_2022_paper.html)][`Human related image generation`]

* 👍**HumanNeRF(CVPR2022 Oral)** HumanNeRF: Free-Viewpoint Rendering of Moving People From Monocular Video [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Weng_HumanNeRF_Free-Viewpoint_Rendering_of_Moving_People_From_Monocular_Video_CVPR_2022_paper.html)][[project link](https://grail.cs.washington.edu/projects/humannerf/)][[code|official](https://github.com/chungyiweng/humannerf)][`Human related image generation`]

* **DVGO(CVPR2022 Oral)** Direct Voxel Grid Optimization: Super-Fast Convergence for Radiance Fields Reconstruction [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_Direct_Voxel_Grid_Optimization_Super-Fast_Convergence_for_Radiance_Fields_Reconstruction_CVPR_2022_paper.html)][[code|official](https://github.com/sunset1995/DirectVoxGO)][[project link](https://sunset1995.github.io/dvgo/)][`Super-Fast`, `Coarse-to-Fine`]

* **HeadNeRF(CVPR2022)** HeadNeRF: A Real-time NeRF-based Parametric Head Model [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Hong_HeadNeRF_A_Real-Time_NeRF-Based_Parametric_Head_Model_CVPR_2022_paper.html)][[code|official](https://github.com/CrisHY1995/headnerf)][`human face`]

* 👍**NeuMan(ECCV2022)** NeuMan: Neural Human Radiance Field from a Single Video [[paper link](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920400.pdf)][[code|official](https://github.com/apple/ml-neuman)][`Human related image generation`]

* **Neural-Sim(ECCV2022)** Neural-Sim: Learning to Generate Training Data with NeRF [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20050-2_28)][[code|official](https://github.com/gyhandy/Neural-Sim-NeRF)]

* ⭐**MoFaNeRF(ECCV2022)** MoFaNeRF:Morphable Facial Neural Radiance Field [[paper link](https://link.springer.com/chapter/10.1007/978-3-031-20062-5_16)][[code|official](https://github.com/zhuhao-nju/mofanerf)][`Face or head related NeRF`, `3DMM + NeRF`]

* **headshot(arxiv2022)** Novel View Synthesis for High-fidelity Headshot Scenes [[paper link](https://arxiv.org/abs/2205.15595)][[code|official](https://github.com/showlab/headshot)][`Face or head related NeRF`]

* **FLNeRF(arxiv2022)** FLNeRF: 3D Facial Landmarks Estimation in Neural Radiance Fields [[paper link](https://arxiv.org/abs/2211.11202)][[project link](https://github.com/ZHANG1023/FLNeRF)][`Face or head related NeRF`]

* **HexPlane(arxiv2023)** HexPlane: A Fast Representation for Dynamic Scenes [[paper link](https://arxiv.org/abs/2301.09632)][[project link](https://caoang327.github.io/HexPlane)]

* **K-Planes(arxiv2023)** K-Planes: Explicit Radiance Fields in Space, Time, and Appearance  [[paper link](https://arxiv.org/abs/2301.10241)][[project link](https://sarafridov.github.io/K-Planes/)]

* **MAV3D(Make-A-Video3D)(arxiv2023)** Text-To-4D Dynamic Scene Generation [[paper link](https://arxiv.org/abs/2301.11280)][[project link](https://make-a-video3d.github.io/)][`Meta AI`, `4D dynamic Neural Radiance Field (NeRF)`, `Diffusion`]


### ▲ Others-based

* **Taming Transformer (a.k.a VQ-GAN)(CVPR2021 oral)** Taming Transformers for High-Resolution Image Synthesis [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.html?ref=https://githubhelp.com)][[project link](https://compvis.github.io/taming-transformers/)][`image-to-image translation`, `Vision Transformers (ViTs) based`, `with the capability to both generate images and perform image-to-image translations`]




