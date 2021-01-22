#  List for public implementation of various algorithms

## 1) Pubilc Datasets and Challenges

* [The KTH Dataset: Recognition of human actions (year 2004)](https://www.csc.kth.se/cvap/actions/)
* [The Weizmann Dataset (year 2005)](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html)
* [HMDB: a large human motion database (year 2011)](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#dataset)
* [Moments in Time Dataset](http://moments.csail.mit.edu/)
* [Charades Challenge (Recognize and locate activities taking place in a video)](http://vuchallenge.org/charades.html)
* [ActivityNet: A Large-Scale Video Benchmark for Human Activity Understanding](http://activity-net.org/)
* [Kinetics Dataset](https://deepmind.com/research/open-source/kinetics) [[paper link: The Kinetics Human Action Video Dataset](https://arxiv.org/abs/1705.06950)]
* [UCF101 (year 2012)](https://www.crcv.ucf.edu/data/UCF101.php) [[blog introduction](https://www.dazhuanlan.com/2019/10/16/5da6679ab4a42/)][[blog CSDN](https://blog.csdn.net/hehuaiyuyu/article/details/107052599)]


## 2) Pioneers and Experts




## 3) Blogs and Videos

* [(B站) [ValseWebinar]视频行为识别 Action Recognition](https://www.bilibili.com/video/BV1yE411x7mw/?spm_id_from=trigger_reload)
* [(B站) 人工智能 | 基于人体骨架的行为识别](https://www.bilibili.com/video/BV1wt411p7Ut/?spm_id_from=333.788.videocard.0)
* [(知乎) 计算机视觉技术深度解读之视频动作识别](https://zhuanlan.zhihu.com/p/90041025)
* [(github) Awesome Action Recognition](https://github.com/jinwchoi/awesome-action-recognition)
* [(CSDN blog) Kinetics-600 dataset介绍(包括ActivityNet)](https://blog.csdn.net/liuxiao214/article/details/80144375)
* [(CSDN blog) 计算机视觉技术深度解读之视频动作识别](https://baijiahao.baidu.com/s?id=1649249453982510365&wfr=spider&for=pc)
* [(CSDN blog) 视频行为识别检测综述 IDT TSN CNN-LSTM C3D CDC R-C3D](https://blog.csdn.net/xiaoxiaowenqiang/article/details/80752849)
* [(CSDN blog) 行为识别数据集汇总](https://blog.csdn.net/u012507022/article/details/52876179)
* [(CSDN blog) CVPR 2020 论文大盘点-动作识别篇](https://blog.csdn.net/moxibingdao/article/details/107329002)


## 4) Papers and Sources Codes

### ▶ Datasets Papers

* **KTH(ICPR2004)** Recognizing human actions: a local SVM approach [[paper link](https://www.researchgate.net/profile/Christian_Schueldt/publication/4090526_Recognizing_human_actions_A_local_SVM_approach/links/0912f5066c8adcddf0000000)]

* **Weizmann(ICCV2005)** Actions as space-time shapes [[paper link](https://www.researchgate.net/profile/Lena_Gorelick/publication/4193986_Action_as_space-time_shapes/links/02e7e5231c496913a4000000)]

* **UCF101(arxiv2012)** UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild [[arxiv link](http://export.arxiv.org/pdf/1212.0402)]

* **Kinetics(arxiv2017)** The Kinetics Human Action Video Dataset [[arxiv link](https://arxiv.org/pdf/1705.06950.pdf)]


### ▶ Technique Papers

#### 1) 基于人工特征(Manual-Features)

* **梯度直方图HOG(CVPR2005)** Histograms of Oriented Gradients for Human Detection [[paper link](https://www.cse.unr.edu/~bebis/CS474/StudentPaperPresentations/HOG.pdf)]

* **时空兴趣点检测(IJCV2005)** On Space-Time Interest Points [[paper link](http://read.pudn.com/downloads142/doc/614011/2005_ijcv_laptev.pdf)]

* **光流直方图(CVPR2008)** Learning Realistic Human Actions from Movies [[paper link](https://www2.cs.sfu.ca/~mori/courses/cmpt888/summer10/papers/laptev_cvpr08.pdf)]

* **密集轨迹特征DT(CVPR2011)** Action Recognition by Dense Trajectories [[paper link](http://www.nlpr.ia.ac.cn/2011papers/gjhy/gh37.pdf)][[project link](https://lear.inrialpes.fr/people/wang/improved_trajectories)][[Codes|offical C++](https://github.com/chensun11/dtfv)]

* **密集轨迹特征iDT(ICCV2013)** Action Recognition with Improved Trajectories [[paper link](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Wang_Action_Recognition_with_2013_ICCV_paper.pdf)][[CSDN blog1](https://blog.csdn.net/wzmsltw/article/details/53023363)][[CSDN blog2](https://blog.csdn.net/wzmsltw/article/details/53221179)]

* **RepresentationFlows(CVPR2019)** Representation Flow for Action Recognition [[arxiv link](https://arxiv.org/abs/1810.01455)][[project link](https://piergiaj.github.io/rep-flow-site/)][[Codes|PyTorch(offical)](https://github.com/piergiaj/representation-flow-cvpr19)]


#### 2) 基于时空双流神经网络(Two-Stream)

* **Two-Stream(NIPS2014)** Two-Stream Convolutional Networks for Action Recognition in Videos [[arxiv link](http://de.arxiv.org/pdf/1406.2199)]

* **two-stream+LSTM(CVPR2015)** Long-term Recurrent Convolutional Networks for Visual Recognition and Description [[arxiv link](https://arxiv.org/abs/1411.4389)][[project link](http://jeffdonahue.com/lrcn/)][[Codes|offical](https://github.com/woodfrog/ActionRecognition)]

* **two-stream+LSTM(CVPR2015)** Beyond short snippets: Deep networks for video classification [[paper link](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Ng_Beyond_Short_Snippets_2015_CVPR_paper.html)]

* **two-stream fusion(CVPR2016)** Convolutional Two-Stream Network Fusion for Video Action Recognition [[arxiv link](https://arxiv.org/abs/1604.06573)][[Codes|offical Matlab MatConvNet](https://github.com/feichtenhofer/twostreamfusion)]

* **TSN(ECCV2016)** Temporal Segment Networks: Towards Good Practices for Deep Action Recognition [[arxiv link](https://arxiv.org/abs/1608.00859)][[project link](http://yjxiong.me/others/tsn/)][[Codes|PyTorch(offical)](https://github.com/yjxiong/temporal-segment-networks)]

* **Co-occurrence+LSTM(+pose)(AAAI2016)** Co-occurrence Feature Learning for Skeleton based Action Recognition using Regularized Deep LSTM Networks [[arxiv link](https://arxiv.org/abs/1603.07772)]

* **RNN-based(+pose)(ECCV2016)** Online Human Action Detection using Joint Classification-Regression Recurrent Neural Networks [[arxiv link](https://arxiv.org/abs/1604.05633)]

* **TSN-based improved 1(CVPR2017)** Deep Local Video Feature for Action Recognition [[arxiv link](https://arxiv.org/abs/1701.07368)]

* **ST+Attention+LSTM(+pose)(AAAI2017)** An End-to-End Spatio-Temporal Attention Model for Human Action Recognition from Skeleton Data [[arxiv link](https://arxiv.org/abs/1611.06067)]

* **TRN(TSN-based improved 2)(ECCV2018)** Temporal Relational Reasoning in Videos [[arxiv link](https://arxiv.org/pdf/1711.08496.pdf)]

* **ST-GCN(+openpose)(AAAI2018)** Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition [[arxiv link](https://arxiv.org/abs/1801.07455)]

* **密集扩张网络(TIP2019)** Dense Dilated Network for Video Action Recognition [[paper link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8720204)]


#### 3) 基于三维卷积的神经网络(3D-ConvNet)

* **C3D(ICCV2015)** Learning Spatiotemporal Features with 3D Convolutional Networks [[arxiv link](https://arxiv.org/pdf/1412.0767.pdf)][[paper link](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html)][[project link](https://vlg.cs.dartmouth.edu/c3d/)][[Codes|offical caffe](https://github.com/facebookarchive/C3D)]

* **3D-ResNets(CVPR2018)** Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? [[arxiv link](https://arxiv.org/abs/1711.09577)][[paper link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.pdf)][[Codes|PyTorch(offical)](https://github.com/kenshohara/3D-ResNets-PyTorch)]

* **I3D(Facebook, use inception-V1)(CVPR2017)** Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset [[arxiv link](https://arxiv.org/abs/1705.07750)][[Codes|Tensorflow(offical)](https://github.com/deepmind/kinetics-i3d)][[Codes|PyTorch(unoffical v1)](https://github.com/piergiaj/pytorch-i3d)][[Codes|PyTorch(unoffical v2)](https://github.com/hassony2/kinetics_i3d_pytorch)]

* **T3D(CVPR2017)** Temporal 3D ConvNets: New Architecture and Transfer Learning for Video Classification [[arxiv link](https://arxiv.org/abs/1711.08200)][[Codes|offical PyTorch](https://github.com/MohsenFayyaz89/T3D)]

* **P3D(MSRA)(ICCV2017)** Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks [[arxiv link](https://arxiv.org/abs/1711.10305)][[CSDN blog](https://blog.csdn.net/u014380165/article/details/78986416)]

* **TPC(based on CDC)(AAAI2018)** Exploring Temporal Preservation Networks for Precise Temporal Action Localization [[arxiv link](https://arxiv.org/abs/1708.03280)]

* **3D-ResNets(arxiv2020)** Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs? [[arxiv link](https://arxiv.org/abs/2004.04968)][[Codes|PyTorch(offical)](https://github.com/kenshohara/3D-ResNets-PyTorch)]




#### 4) 基于长短记忆网络(LSTM)

* **two-stream+LSTM(CVPR2015)** Long-term Recurrent Convolutional Networks for Visual Recognition and Description [[arxiv link](https://arxiv.org/abs/1411.4389)][[project link](http://jeffdonahue.com/lrcn/)][[Codes|offical](https://github.com/woodfrog/ActionRecognition)]



#### 5) 基于对抗神经网络(GAN)

* **GAN-based(IJCAI2018)** Exploiting Images for Video Recognition with Hierarchical Generative Adversarial Networks [[arxiv link](https://arxiv.org/abs/1805.04384)]












