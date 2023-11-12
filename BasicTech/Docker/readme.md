# Contents

* **[1) Introduction](#1-Introduction)**
* **[2) Installition](#2-Installition)**
  * **[2.0 common operations](#20-common-operations)**
  * **[2.1 uninstall old version](#21-uninstall-old-version)**
  * **[2.2 install new version](#22-install-new-version)**
  * **[2.3 install nvidia-docker2](#23-install-nvidia-docker2)**
  * **[2.4 install docker-compose](#24-install-docker-compose)**
* **[3) Usage](#3-Usage)**
  * **[3.1 explore the docker-hub](#31-explore-the-docker-hub)**
  * **[3.2 write your Dockerfile](#32-write-your-dockerfile)**
  * **[3.3 build docker image](#33-build-docker-image)**
  * **[3.4 create docker container](#34-create-docker-container)**
  * **[3.5 manage containers and images](#35-manage-containers-and-images)**
  * **[3.6 test docker connection](#36-test-docker-connection)**
  * **[3.7 import/export docker tar](#37-importexport-docker-tar)**


# 1) Introduction

* [Docker](https://www.docker.com/): Empowering App Development for Developers | Docker
* [Dockerhub](https://hub.docker.com/): Docker Hub Container Image Library | App Containerization
* [docker-tutorial](https://www.runoob.com/docker/docker-tutorial.html): Docker Tutorial from Chinese website RUNOOD.COM

# 2) Installition

## 2.0 common operations
```
# 取消用户必须使用sudo调用docker的限制
$ sudo setfacl -m user:$USER:rw /var/run/docker.sock

# 查看docker容器中的logs
$ sudo docker logs container_name  # 查看全部logs
$ sudo docker logs container_name -f --tail 100  # 查看部分logs
```

## 2.1 uninstall old version
```
$ sudo apt-get autoremove docker docker-ce docker-engine docker.io containerd runc
$ dpkg -l | grep docker  # 查看docker是否卸载干净
$ dpkg -l | grep ^rc|awk '{print $2}' |sudo xargs dpkg -P  # 删除无用的相关的配置文件
$ sudo apt-get autoremove docker-ce-*  # 删除没有删除的相关插件
$ sudo rm -rf /etc/systemd/system/docker.service.d  # 删除docker的相关配置&目录
$ sudo rm -rf /var/lib/docker
$ docker --version  # 确定docker卸载完毕
```

## 2.2 install new version
```
# update source data, and install libs
$ sudo apt-get update
$ sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common

# install GPG
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# choose one source data from two options 
[official]
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
or [aliyun]
$ sudo add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"

# install docker engine
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io

# config of docker
$ sudo systemctl daemon-reload
$ service docker restart
$ sudo docker info
```

## 2.3 install nvidia-docker2
```
# 设置变量
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

# 安装公钥
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

# 获取list
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  
# 更新, 并安装nvidia-docker2
$ sudo apt-get update
$ sudo apt-get install -y nvidia-docker2

# 重启
$ sudo systemctl restart docker
```

## 2.4 install docker-compose
```
$ curl -L https://get.daocloud.io/docker/compose/releases/download/1.25.4/docker-compose-`uname -s`-`uname -m` -o ./docker-compose
possible result
===============================================================================================
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   423  100   423    0     0    171      0  0:00:02  0:00:02 --:--:--   171
100 16.3M  100 16.3M    0     0  4044k      0  0:00:04  0:00:04 --:--:-- 11.2M
===============================================================================================

$ sudo mv ./docker-compose /usr/local/bin/docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose
$ docker-compose info
```


# 3) Usage

## 3.1 explore the docker-hub
登录docker-hub官网 https://hub.docker.com/ ，在悬浮栏上方找到Explore搜索项，然后在其中输入想要搜索的关键词，查找一些基础的Docker镜像，并拉取作为自己Docker镜像的基本子系统。

**Case 1**：如果想要找到Ubuntu系统下的Python3.8环境，作为基本镜像，可输入关键词 python3.8，并在左边Operating Systems中选择Linux；

**Case 2**：如果想要在Docker环境中使用显卡，运行深度模型，则最好选择包含有cuda和cudnn驱动的基本镜像，比如输入关键词 cuda11.0 cudnn8.0，一般会有一些使用量不多的第三方匹配项；

**Case 3**：进一步地，如果Docker环境中运行的深度模型框架为常用的PyTorch和Tensorflow，则可以在关键词中加入PyTorch或Tensorflow的字样及版本，可以更精准地搜索到目标镜像

## 3.2 write your Dockerfile
waiting ...

## 3.3 build docker image
以下是几个例子
```
$ sudo docker build -f Dockerfile_CPU . -t asr_v1_cast
$ sudo docker build -f Dockerfile_CPU . -t nlp_v1_cast
$ sudo docker build -f Dockerfile_3090 . -t behavior_detect_v2
$ sudo docker build -f Dockerfile_3090 . -t pose_student_v1
```
其中，`-f`参数指定Dockerfile文件的相对位置；`-t`参数表示编译后docker镜像的代称；注意必须要有一个 `.` 在编译命令中

## 3.4 create docker container
开启docker容器分两种情况，一种是不使用GPU，一种是使用GPU：

**不使用GPU**
```
$ sudo docker run -d -p 5021:5000 --name p21_asr -v /datasda/docker_data:/data asr_v1_cast
$ sudo docker run -d -p 5022:5000 --name p22_nlp -v /datasda/docker_data:/data nlp_v1_cast
```
其中，`-d`表示在后台运行容器；`-p`参数表示端口映射，5000是docker容器中开启的微服务默认端口，5021/5022等是映射后本地服务器中的端口；`--name`参数表示docker容器的代称，命名方式以方便记忆为主；`-v`参数表示路径映射，/data表示docker容器中的虚拟地址，/datasda/docker_data表示本地服务器中的真实地址；最后，需要加上与新开启docker容器相关联的docker镜像代称

**使用GPU**
```
$ sudo docker run --gpus all -d -p 5001:5000 --name det_puyang -v /datasdb/data/upload_videos:/data behavior_detect_v2
$ sudo docker run --gpus all -d -p 5004:5000 --name pose_puyang -v /datasdb/data/upload_videos:/data pose_student_v1
```
其中，大部分参数与不使用GPU中的解释是一样的；`--gpus`参数表示允许docker容器能够访问到的服务器GPU的id号码，除了`all`以外，还可以指定具体那些GPU可以被使用

## 3.5 manage containers and images
```
# 查看
$ sudo docker images  # 查看现有镜像
$ sudo docker ps  # 查看在运行的所有容器
$ sudo docker ps -a  # 查看所有容器，包括停止的容器
$ sudo docker exec -it container_id /bin/bash  # 通过exec命令对指定的容器执行bash
$ sudo docker container top container_id  # 查看docker容器的PID和PPID信息

# 执行
$ sudo docker start xxxx  # 重新启动容器，xxxx表示容器ID的前四位编码
$ sudo docker start xxxx -a  # 重新启动容器，并展示debug信息，适合排查异常容器的bug
$ sudo docker stop xxxx  # 关闭在运行的容器，xxxx表示容器ID的前四位编码

# 清除
$ sudo docker rmi xxxx  # 删除指定的docker镜像，xxxx表示镜像ID的前四位编码
$ sudo docker rmi xxxx --force # 强制删除指定的docker镜像，慎用
$ sudo docker rm xxxx  # 删除指定的docker容器，xxxx表示容器ID的前四位编码
$ sudo docker container prune  # 大规模删除不在运行中的容器，执行后会让用户选择yes/no，慎用

# 复制
$ sudo docker cp container_id:/container/file/path /local/file/path  # 容器到本地，编辑指定的容器内文件
$ sudo docker cp /local/file/path container_id:/container/file/path  # 本地到容器，记得关闭再启动容器，才能生效
```

## 3.6 test docker connection
注意，以下各种测试函数的名称（`ping`，`teacher_detect`，`positive_behavior_detect`，`negative_behavior_detect`，`pose_student`等），需要运行的微服务程序内，有对应已经写好的函数

**简单情况：**
```
$ curl http://localhost:5001/ping  # 测试微服务程序连通性

# 测试teacher_detect函数，输入为一张图像，协议方式为GET
$ curl http://localhost:5001/teacher_detect?path=./testimg/000112.jpg
 # 测试positive_behavior_detect函数，输入为一张图像，协议方式为GET
$ curl http://localhost:5001/positive_behavior_detect?path=./testimg/000112.jpg
 # 测试negative_behavior_detect函数，输入为一张图像，协议方式为GET
$ curl http://localhost:5001/negative_behavior_detect?path=./testimg/000112.jpg

# 测试pose_student函数，输入为一张图像，协议方式为GET
$ curl http://localhost:5004/pose_student?path=./image/2.jpg
# 测试pose_student函数，输入为一张图像，协议方式为GET
$ curl http://localhost:5004/pose_student?path=./image/000134.jpg
```
**复杂情况：**
```
$ curl http://localhost:5031/refresh_photo_ids and post a json file (e.g. using Postman)
example_json_dict = {
        "folder_path": "./test_files/id_img",
        "aligned_path": "./test_files/id_img_aligned",
        "names": ["Chandler", "Joey", "Monica", "Phoebe", "Rachel", "Ross",
    "sjtu_jiangfei", "sjtu_sijiaxin", "sjtu_yutian", "sjtu_zhouhuayi"]
}
# 测试refresh_photo_ids函数，输入为一个JSON文件，协议方式为POST。可以借助Postman等工具传输JSON大文件
    
$ curl http://localhost:5031/refresh_photo_ids -X POST \
 --data '{"folder_path": "./test_files/id_img", \
 "aligned_path": "./test_files/id_img_aligned", \
 "names": ["Chandler", "Joey", "Monica", "Phoebe", "Rachel", "Ross", \
 "sjtu_jiangfei", "sjtu_sijiaxin", "sjtu_yutian", "sjtu_zhouhuayi"]}'
# 或直接在一串命令行中，包含JSON文件中所有内容。
# -X参数指定协议为POST；--data参数表示JSON文件内容，运行时请删除换行符 \
``` 

## 3.7 import/export docker tar
为了方便在配置大致相同的同类型机器间，快速复制部署成熟的docker镜像，需要了解关于docker镜像的导出和导入
```
$ sudo docker save -o asr_v1_cast_final.tar asr_v1_cast:latest  # 将docker镜像打包成tar文件存入本地
$ sudo docker save -o nlp_v1_cast_final.tar nlp_v1_cast:latest  # 将docker镜像打包成tar文件存入本地

$ sudo chmod a+rwx ./nlp_v1_cast_final.tar  # 改变docker镜像的访问权限
$ sudo chown username:username ./nlp_v1_cast_final.tar  # 改变docker镜像的所有权

$ sudo docker load -i asr_v1_cast_final.tar  # 从本地解压tar文件为docker镜像
$ sudo docker load -i nlp_v1_cast_final.tar  # 从本地解压tar文件为docker镜像
```

