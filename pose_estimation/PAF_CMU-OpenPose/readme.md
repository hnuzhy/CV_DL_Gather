# PAF (or named CMU-OpenPose)
This code repo is a PyTorch version of *Realtime Multi-Person Pose Estimation*. It only inlcudes improved and reorganized test code rather than train code. For details about training your own model, please reference the origin code [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

## 1) Introduction
This PAF algorithm is from an Oral paper 2017 CVPR: **Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields**. It is a milestone in the bottom-up branch. Thanks to the universality of method design and the thoroughness of open source, it is still the most widely used method even today, including practical applications and other complex visual tasks research. 

The deep model part of it uses VGGNet to extract image features, and then divides into two branchs and multiple stages to predict the heatmaps of keypoints and the auxiliary vector PAFs used to help connect limbs, respectively. These iterations draw lessons from single person pose estimation CPM(CVPR2016). It's released source codes are here [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release). The Network Architecture for predicting heatmaps and PAFs is as follows. After that, it needs to use faster greedy matching to decode heatmaps and PAFs to get keypoints and limbs connection for every person pose.

![img1](./readme/PAF_NetworkArchitecture.jpg)

## 2) Environment
Codes are only tested Under Python2.7. Besides, you need to install [Pytorch](http://pytorch.org/) and OpenCV with pip

## 3) Testing
- Download test codes in your machine. But you have no need to clone other unrelated branch projects, and just pull `Pose_Estimation_Depository/PAF_CMU-OpenPose` as follows. **Do not use Windows PowerShell**, it's not supported. Please use `git bash` or Linux.
  ```
  git clone -n https://github.com/hnuzhy/Pose_Estimation_Depository.git
  cd Pose_Estimation_Depository
  git config core.sparsecheckout true
  echo PAF_CMU-OpenPose >> .git/info/sparse-checkout
  git checkout master
  ```
- `cd Pose_Estimation_Depository/PAF_CMU-OpenPose && cd model` and download [pre-trained model](https://pan.baidu.com/s/1jCBY8xef4-Pg1Ug9wrPs8g) to put it under this folder. PassWord is：**2t0f**
- `python eval_images.py` to run test script. It will predict images under *./test_imgs/classroom_test/*. You can change `folderList` in `eval_images.py` to test your own images.
- `cd src && vim config_file` You can edit *config_file* to change config for trade-off between detection accuracy and inference latency. `scale_search = 0.5, 1, 1.2, 1.5` is for image multi-scale detection. `scale_ratio = 0.5` will resize original image smaller to reduce time. `pk_mode = 'fullKP'` changes in ['fullKP', 'PK12'] for faster run. `'PK12'` mode predicts keypoints about upper body as below. 

![keypoints](./readme/COCO_keypoints_PK12.jpg)

## 4) Results

Below three images are detected with config: `scale_search = 0.5, 1, 1.2, 1.5` , `scale_ratio = 0.5` and `pk_mode = 'fullKP'`. The average time consumed is about 2.5 seconds. You can change `scale_ratio` back into `1.0` to detect more students.

**NOTE：`plot_pose(all_keypoints, canvas, imageName, resultSavePath, save_joints_info=True)` in `eval_images.py` should be comment out when computing time.**  

The first two 1K (1920x1080) images are from classroom.

![img2](./test_imgs/classroom_test_result/01_0207_Student_pose.jpg)
![img3](./test_imgs/classroom_test_result/07_0100_pose.jpg)

The last one is from lobby 4K (3840x2160) camera. It is cropped to 1800x1200 covering only key areas with students.

![img4](./test_imgs/classroom_test_result/022_ch40_2655_pose.jpg)

## 5) Future
By comparison, the test code runs faster and works better than the original code in our classroom scenario. But it can be found that PAF is still not good enough for students with **low resolution** or in **crowded situations**. Also, some other implementations of PAF may have better performance [other-implementations](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation#other-implementations). 
You can refer to them if need.

The next step is to investigate the use of more advanced algorithms. [PersonLab(ECCV2018)](../PersonLab/) and [PifPaf(CVPR2019)](../PifPaf/)

## 6) Citation
Please cite the paper in your publications if it helps your research:
```
@inproceedings{cao2017realtime,
  author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
  booktitle = {CVPR},
  title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  year = {2017}
  }
  
@inproceedings{wei2016cpm,
  author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
  booktitle = {CVPR},
  title = {Convolutional pose machines},
  year = {2016}
  }
```
