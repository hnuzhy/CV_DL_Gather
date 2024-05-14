# Large Language Model (LLM) or Large Model

## Materials

* [**(github)** Luotuo-Chinese-LLM: 骆驼(Luotuo): Open Sourced Chinese Language Models](https://github.com/LC1332/Luotuo-Chinese-LLM)
* [**(zhihu)** NLP（九）：LLaMA, Alpaca, ColossalChat 系列模型研究](https://zhuanlan.zhihu.com/p/618695885)
* [**(foundation work)** (Transformers)(NIPS2017) Attention is All you Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
* [**(github)** Segment-anything related awesome extensions/projects/repos](https://github.com/JerryX1110/awesome-segment-anything-extensions)
* [**(github)** Tracking and collecting papers/projects/others related to Segment Anything](https://github.com/Hedlen/awesome-segment-anything)

## Papers

### ▶ NLP (Neural Language Processing)

* **GPT(generative pre-training)(2018)** Improving Language Understanding by Generative Pre-Training [[paper link](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)][`An autoregressive language model`, `pre-training model in NLP`, `OpenAI`]

* **GPT-2(2019)** Language Models are Unsupervised Multitask Learners [[[paper link](https://cs.brown.edu/courses/csci1460/assets/papers/language_models_are_unsupervised_multitask_learners.pdf)][`An autoregressive language model`, `pre-training model in NLP`, `OpenAI`]

* 👍**BERT(NAACL2019)** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [[paper link](https://arxiv.org/abs/1810.04805)][`An autoregressive language model`, `pre-training model in NLP`, `It uses masked language modeling (MLM) and next sentence prediction (NSP) for pre-training`]

* **GPT-3(NIPS2020)** Language Models are Few-Shot Learners [[paper link](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)][`An autoregressive language model`, `pre-training model in NLP`]


### ▶ CV (Computer Vision)

* **iGPT(ICML2020)** Generative Pretraining From Pixels [[paper link](http://proceedings.mlr.press/v119/chen20s.html)][`It operates on sequences of pixels and predicts unknown pixels`]

* 👍**ViT(Vision Transformer)(ICLR2021)** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [[paper link](https://arxiv.org/abs/2010.11929)][[code|official](https://github.com/google-research/vision_transformer)][`Google`, `It studies masked patch prediction for self-supervised learning`][`ViT-Tiny / Small / Base / Large / Huge`]

* **DeiT(Data-efficient image Transformer)(ICML2021)** Training data-efficient image transformers & distillation through attention [[paper link](https://proceedings.mlr.press/v139/touvron21a)][[arxiv link](https://arxiv.org/abs/2012.12877)]

* **ViTAE(NIPS2021)** ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias [[paper link](https://proceedings.neurips.cc/paper/2021/hash/efb76cff97aaf057654ef2f38cd77d73-Abstract.html)][[code|official](https://github.com/Annbless/ViTAE)][`ViT-based`, `self-supervised pre-training`, `Tao Dacheng`]

* **BEiT(arxiv2021.06)(ICLR2022 Oral)** BEiT: BERT Pre-Training of Image Transformers [[paper link](https://arxiv.org/abs/2106.08254)][[code|official](https://github.com/microsoft/unilm/tree/master/beit)][`ViT-based`, `self-supervised pre-training`, `MicroSoft`, `It proposes to predict discrete tokens`]

* **ViTAEv2(IJCV2023)** ViTAEv2: Vision Transformer Advanced by Exploring Inductive Bias for Image Recognition and Beyond [[paper link](https://link.springer.com/article/10.1007/s11263-022-01739-w)][[arxiv link](https://arxiv.org/abs/2202.10108)][`ViT-based`, `self-supervised pre-training`, `Tao Dacheng`]


### ▶ AIGC (Artificial Intelligence Generated Content) 

* **Survey(arxiv2023.03)** A Comprehensive Survey of AI-Generated Content (AIGC): A History of Generative AI from GAN to ChatGPT [[arxiv link](https://arxiv.org/abs/2303.04226)][[blog|zhihu](https://zhuanlan.zhihu.com/p/615522634)][`CMU`]

* **Survey(arxiv2023.01)** ChatGPT is not all you need. A State of the Art Review of large Generative AI models [[arxiv link](https://arxiv.org/abs/2301.04655)]


## ▶ Super Stars

### ⭐Segment Anything

#### Materials

* [**(github)** Awesome-Segment-Anything: the first comprehensive survey on Meta AI's Segment Anything Model (SAM).](https://github.com/liliu-avril/Awesome-Segment-Anything)

#### Papers

* 👍**SAM(arxiv2023.04)(ICCV2023 Best Paper)** Segment Anything [[paper link](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2304.02643)][[project homepage](https://segment-anything.com/)][[publication link](https://ai.facebook.com/research/publications/segment-anything/)][[blogs](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)][[code|official](https://github.com/facebookresearch/segment-anything)][`Meta AI`, `Facebook`]

* **SSA(2023.04)** Semantic Segment Anything [[demo link](https://replicate.com/cjwbw/semantic-segment-anything)][[code|official](https://github.com/fudan-zvg/Semantic-Segment-Anything)][`Fudan`]

* **Grounded-SAM(2023.04)** Grounded Segment Anything [[code|official](https://github.com/IDEA-Research/Grounded-Segment-Anything)][`IDEA-Research`]

* 👍**Anything-3D(2023.04)** Segment-Anything + 3D. Let's lift anything to 3D [[code|official](https://github.com/Anything-of-anything/Anything-3D)][`Anything-3D-Objects`, `Anything-3DNovel-View`, `Anything-NeRF`, `Any-3DFace`][`Any-3DFace` is based on SAM and [HRN (CVPR2023)](https://younglbw.github.io/HRN-homepage/)]

* 👍**3D-Box-Segment-Anything(2023.04)** 3D-Box via Segment Anything [[code|official](https://github.com/dvlab-research/3D-Box-Segment-Anything)][`It extends Segment Anything to 3D perception by combining it with [VoxelNeXt (CVPR2023)](https://github.com/dvlab-research/VoxelNeXt)`]

* **SALT(2023.04)** Segment Anything Labelling Tool (SALT) [[code|official](https://github.com/anuragxel/salt)][`Uses the Segment-Anything Model By Meta AI and adds a barebones interface to label images and saves the masks in the COCO format`]

* **SA3D(2023.04)** Segment Anything in 3D with NeRFs [[arxiv link](https://arxiv.org/abs/2304.12308)][[project link](https://jumpat.github.io/SA3D/)][[code|official](https://github.com/Jumpat/SegmentAnythingin3D)]

* 👍**Inpaint-Anything(arxiv2023.04)** Inpaint Anything: Segment Anything Meets Image Inpainting [[arxiv link](https://arxiv.org/abs/2304.06790)][[code|official](https://github.com/geekyutao/Inpaint-Anything)][[HuggingFace link](https://huggingface.co/spaces/InpaintAI/Inpaint-Anything)]

* **SAM-Survey(arxiv2023.05)** A Comprehensive Survey on Segment Anything Model for Vision and Beyond [[arxiv link](https://arxiv.org/abs/2305.08196)]

* **SAM3D(ICCVW2023)(arxiv2023.06)** SAM3D: Segment Anything in 3D Scenes [[arxiv link](https://arxiv.org/abs/2306.03908)][[code|official](https://github.com/Pointcept/SegmentAnything3D)][`University of Hong Kong`]

* **HQ-SAM(NIPS2023)(arxiv2023.06)** Segment Anything in High Quality [[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f828e38160f31935cfe9f67503ad17c-Abstract-Conference.html)][[arxiv link](https://arxiv.org/abs/2306.01567)][[code|official](https://github.com/SysCV/SAM-HQ)][`ETH Zurich & HKUST`, it proposes HQ-SAM to upgrade SAM for high-quality zero-shot segmentation.]

* **Semantic-SAM(arxiv2023.07)** Semantic-SAM: Segment and Recognize Anything at Any Granularity [[arxiv link](https://arxiv.org/abs/2307.04767)][[code|official](https://github.com/UX-Decoder/Semantic-SAM)][`HKUST`,  It introduces a universal image segmentation model to enable segment and recognize anything at any desired granularity. The authors have `trained on the whole SA-1B dataset` and the model can reproduce SAM and beyond it.]

* **PseCo(CVPR2024)(arxiv2023.11)** Point, Segment and Count: A Generalized Framework for Object Counting [[arxiv link](https://arxiv.org/abs/2311.12386)][[code|official](https://github.com/Hzzone/PseCo)][`Fudan University`, `few-shot/zero-shot object counting/detection`]

* **SAM-6D(CVPR2024)(arxiv2023.11)** SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation [[arxiv link](https://arxiv.org/abs/2311.15707)][[code|official](https://github.com/JiehongLin/SAM-6D)][`South China University of Technology`, the first author [`Jiehong Lin (林杰鸿)`](https://jiehonglin.github.io/)]

* **SAI3D(CVPR2024)(arxiv2023.12)** SAI3D: Segment Any Instance in 3D Scenes [[arxiv link](https://arxiv.org/abs/2312.11557)][[project link](https://yd-yin.github.io/SAI3D/)][[code|official](https://github.com/yd-yin/SAI3D)][`Peking University`, the first author [`Yingda Yin 尹英达`](https://yd-yin.github.io/)]


