# ⭐Vision Foundation Model
*currently, we treat `SAM` as the only Vision Foundation Model and collect related works*

## ⭐Segment Anything Series
*for the `instance segmentation` task*

### Materials

* [**(github)** Awesome-Segment-Anything: the first comprehensive survey on Meta AI's Segment Anything Model (SAM).](https://github.com/liliu-avril/Awesome-Segment-Anything)
* [**(github)** Segment-anything related awesome extensions/projects/repos](https://github.com/JerryX1110/awesome-segment-anything-extensions)
* [**(github)** Tracking and collecting papers/projects/others related to Segment Anything](https://github.com/Hedlen/awesome-segment-anything)

### Papers

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

***
***

## ⭐Depth Anything Series
*for the `monocular depth estimation` task*

### Papers

* **DepthAnything(CVPR2024)(arxiv2024.01)** Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data [[paper link](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Depth_Anything_Unleashing_the_Power_of_Large-Scale_Unlabeled_Data_CVPR_2024_paper.html)][[arxiv link](https://arxiv.org/abs/2401.10891)][[project link](https://depth-anything.github.io/)][[code|official](https://github.com/LiheYoung/Depth-Anything)][`HKU + TikTok + CUHK + ZJU`][the first author is [`Lihe Yang`](https://liheyoung.github.io/)][two keys factors: following the `scaling law of large dataset` and the `semi-supervised learning` technique][It harnesses large-scale unlabeled data to speed up data scaling-up and increase the data coverage]
  
* **DepthAnythingV2(NIPS2025)(arxiv2024.06)** Depth Anything V2 [[arxiv link](https://arxiv.org/abs/2406.09414)][[project link](https://depth-anything-v2.github.io/)][[code|official](https://github.com/DepthAnything/Depth-Anything-V2)][`HKU + TikTok`][the first author is [`Lihe Yang`](https://liheyoung.github.io/)][two keys factors: following the `scaling law of large dataset` and the `semi-supervised learning` technique][It demonstrates “precise synthetic data + pseudo-labeled real data” is a more promising roadmap than labeled real data]


***
***

## ⭐Mesh Anything Series
*for the `3D mesh generation` task*

* **MeshAnything(arxiv2024.06)** MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers [[arxiv link](https://arxiv.org/abs/2406.10163)][[project link](https://buaacyw.github.io/mesh-anything/)][[blog link](https://zhuanlan.zhihu.com/p/706166825)][[code|official](https://github.com/buaacyw/MeshAnything)][`S-Lab, Nanyang Technological University, + others`]

* **MeshAnythingV2(arxiv2024.08)** MeshAnything V2: Artist-Created Mesh Generation With Adjacent Mesh Tokenization [[arxiv link](https://arxiv.org/abs/2408.02555)][[project link](https://buaacyw.github.io/meshanything-v2/)][[blog link](https://baijiahao.baidu.com/s?id=1807065134602050319)][[code|official](https://github.com/buaacyw/MeshAnythingV2)][`S-Lab, Nanyang Technological University, + others`]


* **** [[paper link]()][[arxiv link]()][[project link]()][[code|official]()]
