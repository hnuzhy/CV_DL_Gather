# ⭐3D Body Model Regression
Also named ***3D Human Pose and Shape Regression*** or ***3D Human Pose and Shape Estimation*** or ***Human Mesh Recovery (HMS)***

## Materials

* **(blogs) OBJ Files** [[Everything You Need to Know About Using OBJ Files](https://www.marxentlabs.com/obj-files/)]
* **(blogs) OBJ Files** [[6 Best Free OBJ Editor Software For Windows](https://listoffreeware.com/free-obj-editor-software-windows/)]
* **(models) SMPL family, i.e. SMPL, SMPL+H, SMPL-X** [[codes|official github](https://github.com/vchoutas/smplx/tree/main/transfer_model)]


## Papers

### Journals

* **(survey)(arxiv2022.03)(TPAMI2023) Recovering 3D Human Mesh from Monocular Images: A Survey** [[paper link](https://arxiv.org/abs/2203.01923)] [[project link](https://github.com/tinatiansjz/hmr-survey)] [[CVPR 2022 related works](https://github.com/tinatiansjz/hmr-survey/issues/1)]

### Conferences

* **SMPL(SIGGRAPH2015)** SMPL: A Skinned Multi-Person Linear Model [[paper link](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf)][[project link](https://smpl.is.tue.mpg.de/)][`MPII 马普所`]

* **SMPL-X(CVPR2019)** Expressive Body Capture: 3D Hands, Face, and Body from a Single Image [[paper link](https://ps.is.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf)][[project link](https://smpl-x.is.tue.mpg.de/)][[codes|official](https://github.com/vchoutas/smplify-x)][`MPII 马普所`]

* **PARE(ICCV2019)** PARE: Part Attention Regressor for 3D Human Body Estimation [[paper link](https://arxiv.org/abs/2104.08527)][[project link](https://github.com/mkocabas/PARE)][[codes|official](https://pare.is.tue.mpg.de/)][`MPII 马普所`]

* **SPIN(ICCV2019)** Learning to Reconstruct 3D Human Pose and Shape via Model-Fitting in the Loop [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.html)][[project link](https://www.seas.upenn.edu/~nkolot/projects/spin/)][[codes|official](https://github.com/nkolot/SPIN)][`MPII 马普所`]

* **STAR(ECCV2020)** STAR: A Sparse Trained Articulated Human Body Regressor [[paper link](https://ps.is.mpg.de/uploads_file/attachment/attachment/618/star_paper.pdf)][[project link](https://star.is.tue.mpg.de/)][[codes|official](https://github.com/ahmedosman/STAR)][`MPII 马普所`]

* 👍**ExPose(ECCV2020)** Monocular Expressive Body Regression through Body-driven Attention [[paper linkl](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_2)][[project link](https://expose.is.tue.mpg.de/)][[codes|official](https://github.com/vchoutas/expose)][`MPII 马普所`][`the pioneering work (regression-based method) for the full-body mesh recovery task`][`Whole Body Recovery with Hands and Face`][`It directly predicts hands, face, and body parameters in the SMPL-X format and utilizes the body estimation to localize the face and hands regions and crop them from the high-resolution inputs for refinement. It learns part-specific knowledge from existing faceand hand-only datasets to improve performance.`]

* **DeepCap(CVPR2020)** DeepCap: Monocular Human Performance Capture Using Weak Supervision [[paper link](https://openaccess.thecvf.com/content_CVPR_2020/html/Habermann_DeepCap_Monocular_Human_Performance_Capture_Using_Weak_Supervision_CVPR_2020_paper.html)][[arxiv link](https://arxiv.org/abs/2003.08325)][`MPII`, `a pose estimation step + a non-rigid surface deformation step`, `SMPL based`]

* **PIXIE(3DV2021)** Collaborative Regression of Expressive Bodies using Moderation [[paper link](https://ieeexplore.ieee.org/abstract/document/9665886)][[arxiv link](https://arxiv.org/abs/2105.05301)][[project link](pixie.is.tue.mpg.de)][[codes|official](https://github.com/YadiraF/PIXIE)][`MPII 马普所`][`Whole Body Recovery with Hands and Face`][[[Yao Feng](https://is.mpg.de/person/yfeng)] `It estimates the confidence of partspecific features and fuses the face-body and hand-body features weighted according to moderators. The fused features are fed to the independent regressors for robust regression.`]

* **GTRS(ACMMM2021)** A Lightweight Graph Transformer Network for Human Mesh Reconstruction from 2D Human Pose [[paper link](https://arxiv.org/pdf/2111.12696.pdf)][[code|official](https://github.com/zczcwh/GTRS)]

* **BMP(CVPR2021)** Body Meshes as Points [[paper link](https://arxiv.org/abs/2105.02467)][[code|official](https://github.com/jfzhang95/BMP)][[Jianfeng Zhang](https://jeff95.me/)]

* **DetNet(CVPR2021)** Monocular Real-Time Full Body Capture With Inter-Part Correlations [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Monocular_Real-Time_Full_Body_Capture_With_Inter-Part_Correlations_CVPR_2021_paper.html)][`no official code`]

* **(CVPR2021)** Monocular Real-time Full Body Capture with Inter-part Correlations [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Monocular_Real-Time_Full_Body_Capture_With_Inter-Part_Correlations_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2012.06087)][`Whole Body Recovery with Hands and Face`][[[Yuxiao Zhou](https://calciferzh.github.io/)] `THU + MPII`, `It is a real-time method that captures body, hands, and face with competitive accuracy by exploiting the interpart relationship between body and hands. SMPL+H and 3DMM are used to represent the body+hands and face.`]

* **(CVPR2021)** Probabilistic 3D Human Shape and Pose Estimation From Multiple Unconstrained Images in the Wild [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Sengupta_Probabilistic_3D_Human_Shape_and_Pose_Estimation_From_Multiple_Unconstrained_CVPR_2021_paper.html)][[arxiv link](https://arxiv.org/abs/2103.10978)][[[Akash Sengupta](https://akashsengupta1997.github.io/)] `Regression-based Paradigm with Probabilistic Output`, `multi-image shape prediction`, `It assumes simple multivariate Gaussian distributions over SMPL pose parameters θ and let the network to predict µθ(I) and δθ(I).`, The same authors of `HierarchicalProbabilistic3DHuman(ICCV2021)`]

* **FrankMocap(ICCVW2021)** FrankMocap: A monocular 3D whole-body pose estimation system via regression and integration [[paper link](https://openaccess.thecvf.com/content/ICCV2021W/ACVR/html/Rong_FrankMocap_A_Monocular_3D_Whole-Body_Pose_Estimation_System_via_Regression_ICCVW_2021_paper.html)][[codes|official](https://github.com/facebookresearch/frankmocap)][`facebookresearch`]

* 👍**HierarchicalProbabilistic3DHuman(ICCV2021)** Hierarchical Kinematic Probability Distributions for 3D Human Shape and Pose Estimation From Images in the Wild [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Sengupta_Hierarchical_Kinematic_Probability_Distributions_for_3D_Human_Shape_and_Pose_ICCV_2021_paper.html)][[arxiv link](http://arxiv.org/abs/2110.00990)][[code|official](https://github.com/akashsengupta1997/HierarchicalProbabilistic3DHuman)][[[Akash Sengupta](https://akashsengupta1997.github.io/)] `Regression-based Paradigm with Probabilistic Output`, `multi-image shape prediction`, `It estimates a hierarchical matrix-Fisher distribution over the relative 3D rotation matrix of each joint. This probability density function is conditioned on the parent joint along with the body’s kinematic tree structure. The shape is still based on a Gaussian distribution.`]

* **LightweightMHMS(ICCV2021)** Lightweight Multi-Person Total Motion Capture Using Sparse Multi-View Cameras [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Lightweight_Multi-Person_Total_Motion_Capture_Using_Sparse_Multi-View_Cameras_ICCV_2021_paper.html)][`taking multi-view RGB sequences and body estimation results as inputs`, `using full-body model SMPL-X`, `Openpose + FaceAlignment + SRHandNet + HandHMR`]

* ❤**ROMP(ICCV2021)** Monocular, One-stage, Regression of Multiple 3D People [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Sun_Monocular_One-Stage_Regression_of_Multiple_3D_People_ICCV_2021_paper.html)][[codes|official](https://github.com/Arthur151/ROMP)][`related with MPII 马普所`][[Yu Sun](https://www.yusun.work/)]

* **PyMAF(ICCV2021 Oral)** PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop [[paper link](https://arxiv.org/pdf/2103.16507.pdf)][[project link](https://hongwenzhang.github.io/pymaf/)][[codes|official](https://github.com/HongwenZhang/PyMAF)]

* **(ICASSP2022)** Learning Monocular Mesh Recovery of Multiple Body Parts Via Synthesis [[paper link](https://ieeexplore.ieee.org/abstract/document/9747426)][`Whole Body Recovery with Hands and Face`][[[Yu Sun](https://www.yusun.work/)] `HIT`, `It predicts hands, and face parameters based on detected wholebody 2D keypoints, making it feasible to take advantage of synthetic data during training.`]

* ❤**PyMAF-X(arxiv2022)(TPAMI2023)** PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images [[paper link](https://arxiv.org/pdf/2207.06400.pdf)][[project link](https://www.liuyebin.com/pymaf-x/)][[codes|official](https://github.com/HongwenZhang/PyMAF)]

* 👍**Hand4Whole(CVPRW2022)** Accurate 3D Hand Pose Estimation for Whole-body 3D Human Mesh Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022W/ABAW/html/Moon_Accurate_3D_Hand_Pose_Estimation_for_Whole-Body_3D_Human_Mesh_CVPRW_2022_paper.html)][[codes|official](https://github.com/mks0601/Hand4Whole_RELEASE)][`Whole Body Recovery with Hands and Face`][`It obtains the joint-level features from feature maps, and regresses the 3D body/hand joint rotations from them. `]

* ❤**BEV(CVPR2022)** Putting People in their Place: Monocular Regression of 3D People in Depth [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_Putting_People_in_Their_Place_Monocular_Regression_of_3D_People_CVPR_2022_paper.html)][[project link](https://arthur151.github.io/BEV/BEV.html)][[codes|official](https://github.com/Arthur151/ROMP)][[Relative Human dataset](https://github.com/Arthur151/Relative_Human)][`related with MPII 马普所`][[Yu Sun](https://www.yusun.work/)]

* ❤**hmr-benchmarks(NIPS2022)** Benchmarking and Analyzing 3D Human Pose and Shape Estimation Beyond Algorithms [[paper link](https://openreview.net/forum?id=rjBYortWdRV)][[codes|official](https://github.com/smplbody/hmr-benchmarks)]

* **PSVT(CVPR2023)** PSVT: End-to-End Multi-person 3D Pose and Shape Estimation with Progressive Video Transformers [[paper link](https://arxiv.org/abs/2303.09187)][Datasets: `RH`, `AGORA`, `CMU Panoptic`, `3DPW`]

* 👍**BSTRO(Body-Scene contact TRansfOrmer)(CVPR2022)** Capturing and Inferring Dense Full-Body Human-Scene Contact [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Capturing_and_Inferring_Dense_Full-Body_Human-Scene_Contact_CVPR_2022_paper.html)][[project link](https://rich.is.tue.mpg.de/)][[code|official](https://github.com/paulchhuang/bstro)][dataset `RICH`, `Interaction-Contact-Humans`, `MPII`, `single-person`]

* **EgoEgo(CVPR2023)** Ego-Body Pose Estimation via Ego-Head Pose Estimation [[arxiv link](https://arxiv.org/abs/2212.04636)][[project link](https://lijiaman.github.io/projects/egoego/)][`predict head poses and estimate full-body poses`]

* **HuManiFlow(CVPR2023)** HuManiFlow: Ancestor-Conditioned Normalising Flows on SO(3) Manifolds for Human Pose and Shape Distribution Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Sengupta_HuManiFlow_Ancestor-Conditioned_Normalising_Flows_on_SO3_Manifolds_for_Human_Pose_CVPR_2023_paper.html)][[arxiv link](https://arxiv.org/abs/2305.06968)][[code|official](https://github.com/akashsengupta1997/HuManiFlow)][[[Akash Sengupta](https://akashsengupta1997.github.io/)] `Regression-based Paradigm with Probabilistic Output`, `multi-image shape prediction`, `It improves the consistency and diversity of predictions by modeling the ancestor-conditioned perbody-part pose distributions in an autoregressive manner.`]

