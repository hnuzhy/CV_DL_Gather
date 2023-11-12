# CUDA

# Introduction

* [CUDA (Compute Unified Device Architecture)](https://developer.nvidia.com/)
* [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

# Anaconda3
Login official website https://www.anaconda.com/ to choose suitable version to download
```bash
$ wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
$ sh ./Anaconda3-2021.05-Linux-x86_64.sh
```


# CUDA Installition

Login official website https://developer.nvidia.com/cuda-toolkit-archive to choose suitable version to download
```bash
# e.g., cuda_11.2.1 for GTX3080Ti and GTX3090
$ wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda_11.2.1_460.32.03_linux.run
$ sudo sh cuda_11.2.1_460.32.03_linux.run
```

Possible logs
```
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-11.2/
Samples:  Installed in /home/gdp/, but missing recommended libraries

Please make sure that
 -   PATH includes /usr/local/cuda-11.2/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-11.2/lib64, or, add /usr/local/cuda-11.2/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.2/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 460.00 is required for CUDA 11.2 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log
```
config yourself CUDA path
```bash
$ sudo vim ~/.bashrc
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
export CUDA_HOME=/usr/local/cuda-11.2${CUDA_HOME:+:${CUDA_HOME}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
$ source ~/.bashrc
```
check if installed successfully
```
$ nvcc -V
or
$ cat /usr/local/cuda/version.json
```


# CUDNN Installition

Login official website https://developer.nvidia.com/rdp/cudnn-archive to choose suitable version to download
```bash
# e.g., cudnn-11.2 for GTX3080Ti and GTX3090
$ wget https://developer.download.nvidia.cn/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz
```
unzip and install
```bash
$ tar -xzvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
$ mkdir cudnn-11.2-linux-x64-v8.1.1.33
$ mv cuda cudnn-11.2-linux-x64-v8.1.1.33

$ sudo cp cuda/include/* /usr/local/cuda/include/
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/

$ sudo chmod a+r /usr/local/cuda/include/* /usr/local/cuda/lib64/libcudnn*
```


# NCCL Installition

Login official website https://developer.nvidia.com/nccl/nccl-legacy-downloads to choose suitable version to download

e.g., download NCCL 2.8.4, for CUDA 11.2, February 03,2021
```bash
# Network Installer for Ubuntu20.04
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
$ sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
$ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
$ sudo apt-get update

# Network Installer for Ubuntu18.04
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
$ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
$ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
$ sudo apt-get update
```

Then, run the following command to installer NCCL:
```bash
# For Ubuntu: 
$ sudo apt install libnccl2=2.8.4-1+cuda11.2 libnccl-dev=2.8.4-1+cuda11.2
# For RHEL/Centos: 
$ sudo yum install libnccl-2.8.4-1+cuda11.2 libnccl-devel-2.8.4-1+cuda11.2 libnccl-static-2.8.4-1+cuda11.2
```

