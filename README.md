# Subjects / Topics / Fields

## Action Recognition
<details>
<summary>Click Here to Show All</summary>
  
* **[Definition]** Action recognition is to classify short video clips that have been pre-segmented (Actually, videos in real environments are generally not pre-segmented and contain a lot of irrelevant information).
* **[Supplement]** The category of action recognition is usually all kinds of human actions, so it can also be called **Human Action Recognition**. However, most of the algorithms developed based on this problem are not specific to people, and can also be used for other types of video classification. In addition, the task of action recognition generally does not include the problem of **Temporal Action Detection** in videos.
* *Refer to branch [action_recognition](./action_recognition) for detailed introduction.*

</details>

## Affective Computing
<details>
<summary>Click Here to Show All</summary>
  
* **[Definition]** Affective computing is to establish a harmonious human-computer environment by giving the computer the ability to recognize, understand, express and adapt to human affections, and make the computer have higher and comprehensive intelligence.
* **[Supplement]** In computer vision field, the current affective computing mainly refers to **Facial Expression Recognition**, which includes three directions: universal expression recognition, **Facial Action Coding System (FACS)**, and continuous expression model **Valence-Arousal**. In addition, the input of accurate affective computing should be multimodal, including facial expression, voice and intonation, text information, body behavior and even electroencephalogram (EEG) signal. The definition of affection is always ambiguous for its multi-interdisciplinary character. So affective computing is rather difficult and has a long way to go.
* *Refer to branch [affective_computing](./affective_computing) for detailed introduction.*

</details>

## Face Head Related
<details>
<summary>Click Here to Show All</summary>

* **[Definition]** Here, we will collect some algorithms and materials about people face and head including detection, estimation, reconstruction and recognition in Computer Vision. The input is mostly a 2D image (may also having RGB-D or RGB-times). The outputs are points, bounding-boxes, Euler angles and so on.
* **[Supplement]** Various tasks in deep learning based Computer Vision have strong relation with person's face, sometimes the head. Therefore, we do not discuss them separately. Especially, the face related works or fileds include **Face Detection**, **Face Alignment**, **Face Reconstruction (3D)**, **Face Recognition**, **Beautify Face** and so on. The head related tasks are mainly about **Head Detector** and **Head/Hand Pose Estimation**. We will also introduce some other interesting related topics as long as we have found thems.
* *Refer to branch [face_head_related](./face_head_related) for detailed introduction.*

</details>

## Model Compression
<details>
<summary>Click Here to Show All</summary>
  
* **[Definition]** Model compression is to minimize the consumption of storage space, computing space and time of deep models, and is also committed to accelerating the training and inference of the model.
* **[Supplement]** Deep learning makes the performance of many computer vision tasks reach an unprecedented height. Although the complex model has better performance, the high storage space and computing resource consumption are the important reasons that make it difficult to effectively apply in various hardware platforms. Therefore, model compression is essential. To solve these problems, it is necessary to cut in from many aspects, including **machine learning algorithm**, **optimization theory**, **computer architecture**, **data compression**, **index compilation** and **hardware design**. The methods of model compression can be roughly divided into: **low rank approximation**, **network pruning**, **network quantification**, **knowledge distillation** and **compact network design**. This is the last battlefield of large-scale application of computer vision.
* *Refer to branch [model_compression](./model_compression) for detailed introduction.*

</details>

## Object Detection
<details>
<summary>Click Here to Show All</summary>
  
* **[Definition]** The task of object detection includes the location and classification of the pre-defined objects in the image.
* **[Supplement]** Object detection is always one of the basic, popular and important tasks in the field of computer vision. Especially after the great breakthrough of deep learning technology in **Image Classification** task (in 2012), the development of object detection is rather rapid, even now it is still the most active topic. Many technologies, ideas and innovations based on object detection have great significance for reference and promotion in other fields. Object detection is absolutely a bright pearl in the CV crown.
* *Refer to branch [object_detection](./object_detection) for detailed introduction.*

</details>

## Pose Estimation
<details>
<summary>Click Here to Show All</summary>
  
* **[Definition]** Given a 2D RGB image, the traditional task of pose estimation is to predict and output all the keypoints of human bodies contained in it, and connect them into independent skeletons.
* **[Supplement]** On this basis, there are other pose estimation tasks under different types of input. Such as **3D Pose Estimation** (RGBD), video based **Pose Estimation and Tracking** (videos), **WiFi Reflection Based Pose Estimation** (WiFi info) and **Human Body Pose and Shape Estimation** (3D motion data). Pose estimation can be used not only for practical applications directly, but also to tackle other computer vision tasks.
* *Refer to branch [pose_estimation](./pose_estimation) for detailed introduction.*

</details>

## Reinforcement Learning
<details>
<summary>Click Here to Show All</summary>
  
* **[Definition]** Reinforcement learning (RL) is used to describe and solve the problem that agents use learning strategies to take actions to maximize reward or achieve specific goals in the process of interaction with the environment.

* **[Supplement]** The common model of RL is standard **Markov Decision Process** (MDP). According to the given conditions, RL can be divided into **model-based RL** and **model-free RL**. The algorithms used to solve RL problems can be divided into strategy search algorithm and value function algorithm. Deep learning model can be used in RL to form deep reinforcement learning. Inspired by **behaviorist psychology**, RL focuses on online learning and tries to maintain a balance between exploration and exploitation. Unlike **supervised learning** and **unsupervised learning**, RL does not require any given data in advance, but obtains learning information and updates model parameters by receiving reward (feedback) from environment.
  
  RL has been discussed in the fields of **information theory**, **game theory** and **automatic control**. It is used to explain the **equilibrium state under bounded rationality**, design **recommendation system** and robot interaction system. Some complex RL algorithms have general intelligence to solve complex problems to a certain extent, which can reach the human level in go and electronic games. The learning cost and training cost of RL are very high.
* *Refer to branch [reinforcement_learning](./reinforcement_learning) for detailed introduction.*

</details>

## Super Resolution
<details>
<summary>Click Here to Show All</summary>
  
* **[Definition]** Image super resolution refers to the restoration of high-resolution image from a low-resolution image or image sequence.
* **[Supplement]** Image super-resolution technology is divided into super resolution restoration and super resolution reconstruction. At present, image super resolution research can be divided into three main categories: interpolation based, reconstruction based and learning based methods. Not surprisingly, the involvement of deep learning technology makes a great breakthrough and progress in this low level visual task. 
* *Refer to branch [super_resolution](./super_resolution) for detailed introduction.*

</details>

## Visual Tracking
<details>
<summary>Click Here to Show All</summary>
  
* **[Definition]** Visual tracking refers to the detection, extraction, recognition and tracking of moving objects in image sequences.
* **[Supplement]** The purpose of visual tracking is to obtain the motion parameters of the moving object, such as position, velocity, acceleration and trajectory, so as to process and analyze the next step, realize the behavior understanding of the moving object, and complete the higher level visual task. Visual tracking can be divided into two branches, CNN and correlation filtering. At present, CNN has not fully occupied the field of tracking. On the one hand, CNN model has poor real-time processing of images, on the other hand, there is a lot of redundant information in video. Visual tracking is the most significant and challenging task in the CV field.
* *Refer to branch [visual_tracking](./visual_tracking) for detailed introduction.*

</details>

## Other Collection Depository
* **[(website) Computer Vision Foundation open access](https://openaccess.thecvf.com/menu)**
* **[(github) daily-paper-computer-vision](https://github.com/amusi/daily-paper-computer-vision)**
* **[(github) ECCV2018-Papers-with-Code](https://github.com/amusi/ECCV2022-Papers-with-Code) [[Official Website](https://eccv2018.org/)][[ECCV Conference Papers 2018](https://www.ecva.net/papers.php)][[ECCV 2018 open access](https://openaccess.thecvf.com/ECCV2018)]**
* **[(github) ECCV2020-Papers-with-Code](https://github.com/amusi/ECCV2022-Papers-with-Code) [[Official Website](https://eccv2020.eu/)][[ECCV Conference Papers 2020](https://www.ecva.net/papers.php)]**
* **[(github) CVPR2021-Papers-with-Code-Demo](https://github.com/DWCTOD/CVPR2022-Papers-with-Code-Demo/blob/main/CVPR2021.md) [[Official Website](https://cvpr2021.thecvf.com/)][[(zhihu)CVPR2021的最新论文汇总](https://zhuanlan.zhihu.com/p/354043252?ivk_sa=1024320u)][[Paper Search](https://public.tableau.com/app/profile/joshpreston/viz/CVPR2021/Dashboard1)]**
* **[(github) ICCV2021-Papers-with-Code-Demo](https://github.com/DWCTOD/ICCV2021-Papers-with-Code-Demo) [[Official Website](http://iccv2021.thecvf.com/home)][[(zhihu)ICCV2021的最新论文汇总](https://zhuanlan.zhihu.com/p/392575669)]**
* **[(github) CVPR2022-Papers-with-Code-Demo](https://github.com/DWCTOD/CVPR2022-Papers-with-Code-Demo) [[Official Website](https://cvpr2022.thecvf.com/)][[CVPR 2022 open access](https://openaccess.thecvf.com/CVPR2022)][[Paper-Interaction-Tool](https://public.tableau.com/views/CVPR2022/Dashboard1)]**
* **[(github) ECCV2022-Papers-with-Code-Demo](https://github.com/DWCTOD/ECCV2022-Papers-with-Code-Demo) [[Official Website](https://eccv2022.ecva.net/)]**


