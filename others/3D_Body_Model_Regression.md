# ⭐3D Body Model Regression
Also named ***3D Human Pose and Shape Regression*** or ***3D Human Pose and Shape Estimation*** or ***Human Mesh Recovery (HMS)***

## Materials

* **(blogs) OBJ Files** [[Everything You Need to Know About Using OBJ Files](https://www.marxentlabs.com/obj-files/)]
* **(blogs) OBJ Files** [[6 Best Free OBJ Editor Software For Windows](https://listoffreeware.com/free-obj-editor-software-windows/)]
* **(models) SMPL family, i.e. SMPL, SMPL+H, SMPL-X** [[codes|official github](https://github.com/vchoutas/smplx/tree/main/transfer_model)]
* **(survey)(arxiv2022) Recovering 3D Human Mesh from Monocular Images: A Survey** [[paper link](https://arxiv.org/abs/2203.01923)] [[project link](https://github.com/tinatiansjz/hmr-survey)] [[CVPR 2022 related works](https://github.com/tinatiansjz/hmr-survey/issues/1)]
 

## Papers

* **SMPL(SIGGRAPH2015)** SMPL: A Skinned Multi-Person Linear Model [[paper link](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf)][[project link](https://smpl.is.tue.mpg.de/)][`MPII 马普所`]

* **SMPL-X(CVPR2019)** Expressive Body Capture: 3D Hands, Face, and Body from a Single Image [[paper link](https://ps.is.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf)][[project link](https://smpl-x.is.tue.mpg.de/)][[codes|official](https://github.com/vchoutas/smplify-x)][`MPII 马普所`]

* **SPIN(ICCV2019)** Learning to Reconstruct 3D Human Pose and Shape via Model-Fitting in the Loop [[paper link](https://openaccess.thecvf.com/content_ICCV_2019/html/Kolotouros_Learning_to_Reconstruct_3D_Human_Pose_and_Shape_via_Model-Fitting_ICCV_2019_paper.html)][[project link](https://www.seas.upenn.edu/~nkolot/projects/spin/)][[codes|official](https://github.com/nkolot/SPIN)][`MPII 马普所`]

* **STAR(ECCV2020)** STAR: A Sparse Trained Articulated Human Body Regressor [[paper link](https://ps.is.mpg.de/uploads_file/attachment/attachment/618/star_paper.pdf)][[project link](https://star.is.tue.mpg.de/)][[codes|official](https://github.com/ahmedosman/STAR)][`MPII 马普所`]

* **ExPose(ECCV2020)** Monocular Expressive Body Regression through Body-driven Attention [[paper linkl](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_2)][[project link](https://expose.is.tue.mpg.de/)][[codes|official](https://github.com/vchoutas/expose)][`MPII 马普所`][`the pioneering work (regression-based method) for the full-body mesh recovery task`]

* **GTRS(ACMMM2021)** A Lightweight Graph Transformer Network for Human Mesh Reconstruction from 2D Human Pose [[paper link](https://arxiv.org/pdf/2111.12696.pdf)][[code|official](https://github.com/zczcwh/GTRS)]

* **DetNet(CVPR2021)** Monocular Real-Time Full Body Capture With Inter-Part Correlations [[paper link](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Monocular_Real-Time_Full_Body_Capture_With_Inter-Part_Correlations_CVPR_2021_paper.html)][`no official code`]

* **PIXIE(3DV2021)** Collaborative regression of expressive bodies using moderation [[paper link](https://ps.is.mpg.de/uploads_file/attachment/attachment/667/PIXIE_3DV_CR.pdf)][[project link](https://pixie.is.tue.mpg.de/)][[codes|official](https://github.com/YadiraF/PIXIE)][`MPII 马普所`]

* **FrankMocap(ICCVW2021)** FrankMocap: A monocular 3D whole-body pose estimation system via regression and integration [[paper link](https://openaccess.thecvf.com/content/ICCV2021W/ACVR/html/Rong_FrankMocap_A_Monocular_3D_Whole-Body_Pose_Estimation_System_via_Regression_ICCVW_2021_paper.html)][[codes|official](https://github.com/facebookresearch/frankmocap)][`facebookresearch`]

* **LightweightMHMS(ICCV2021)** Lightweight Multi-Person Total Motion Capture Using Sparse Multi-View Cameras [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Lightweight_Multi-Person_Total_Motion_Capture_Using_Sparse_Multi-View_Cameras_ICCV_2021_paper.html)][`taking multi-view RGB sequences and body estimation results as inputs`, `using full-body model SMPL-X`, `Openpose + FaceAlignment + SRHandNet + HandHMR`]

* ❤**ROMP(ICCV2021)** Monocular, One-stage, Regression of Multiple 3D People [[paper link](https://openaccess.thecvf.com/content/ICCV2021/html/Sun_Monocular_One-Stage_Regression_of_Multiple_3D_People_ICCV_2021_paper.html)][[codes|official](https://github.com/Arthur151/ROMP)][`related with MPII 马普所`]

* **PyMAF(ICCV2021 Oral)** PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop [[paper link](https://arxiv.org/pdf/2103.16507.pdf)][[project link](https://hongwenzhang.github.io/pymaf/)][[codes|official](https://github.com/HongwenZhang/PyMAF)]

* ❤**PyMAF-X(arxiv2022)** PyMAF-X: Towards Well-aligned Full-body Model Regression from Monocular Images [[paper link](https://arxiv.org/pdf/2207.06400.pdf)][[project link](https://www.liuyebin.com/pymaf-x/)][[codes|official](https://github.com/HongwenZhang/PyMAF)]

* **Hand4Whole(CVPRW2022)** Accurate 3D Hand Pose Estimation for Whole-body 3D Human Mesh Estimation [[paper link](https://openaccess.thecvf.com/content/CVPR2022W/ABAW/html/Moon_Accurate_3D_Hand_Pose_Estimation_for_Whole-Body_3D_Human_Mesh_CVPRW_2022_paper.html)][[codes|official](https://github.com/mks0601/Hand4Whole_RELEASE)]

* ❤**BEV(CVPR2022)** Putting People in their Place: Monocular Regression of 3D People in Depth [[paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_Putting_People_in_Their_Place_Monocular_Regression_of_3D_People_CVPR_2022_paper.html)][[project link](https://arthur151.github.io/BEV/BEV.html)][[codes|official](https://github.com/Arthur151/ROMP)][[Relative Human dataset](https://github.com/Arthur151/Relative_Human)][`related with MPII 马普所`]

* ❤**hmr-benchmarks(NIPS2022)** Benchmarking and Analyzing 3D Human Pose and Shape Estimation Beyond Algorithms [[paper link](https://openreview.net/forum?id=rjBYortWdRV)][[codes|official](https://github.com/smplbody/hmr-benchmarks)]
