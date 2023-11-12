# Common Command Line and Script in Linux

* 系统相关
```
Ctrl+Alt+F3  # Ubuntu开机过程中操作，进入命令行交互界面
uname -a  # 查看当前操作系统内核信息
ubuntu-drivers devices  # 查看显卡型号和推荐驱动版本
sudo dmidecode | grep "Product Name"  # 查看机器型号
sudo dmidecode | grep -A16 "Memory Device$"|grep Size  # 查看机器有几个内存插槽及已使用几个

data -R  # 显示系统当前日期及时间，例如 Mon, 20 Dec 2021 19:00:11 +0800

sudo vim /etc/crontab  # 编辑该文本，设置定时重启机器。参考配置https://www.cnblogs.com/zdyang/p/13856459.html
00 08 * * * root /sbin/reboot now  # 每天早上8点钟，重启机器
00 08 * * 1 root /sbin/reboot now  # 或者，每周一早上8点钟，重启机器
00 08 */3 * * root /sbin/reboot now  # 或者，每隔3天早上8点钟，重启机器
sudo /etc/init.d/cron restart  # 重启cron，更新修改文本

sudo apt install at  # Linux/Ubuntu 延时执行命令at。参考https://www.cnblogs.com/VCplus/p/11542402.html
at 15:30  # 回车之后，需要在at后输入指令，表示在15:30要执行的指令, 在at后添加命令完毕后，选择ctrl+D退出
at 15:30 tomorrow  # 在明天15:30执行
at 15:30 01/10/22  # 在2022年01月10日15:30执行
at now +30 minutes  # 在30分钟后执行
atq  # 列出等待中的延时任务
atrm  # 撤销延时任务. 如 atrm 1 撤销编号为1的延时任务
date && sleep 1m && date  # sleep命令也可延迟当前命令的执行，参考https://www.linuxcool.com/sleep
```


* 用户管理
```
# 【方式一】【推荐】按提示，为系统添加新用户，并依次按需设置登录密码等信息
sudo adduser username
# 【方式二】【不推荐】按参数，为系统添加新用户，建立的账号保存在/etc/passwd文件中
# -m可保证添加后的用户主文件夹在/home下，-r指创建系统账户，-p设置密码
sudo useradd username -m -r -p passwd

sudo passwd username  # 为系统的新用户设置用户登陆密码，这一步骤可紧跟添加新用户之后
su username  # 切换到某个用户账号下，需要输入该用户密码
sudo userdel -r username  # 删除某个用户，加上-r可以删除/home/路径下的用户文件夹，否则不能
sudo chown -R username:username /path/to/data/username  # 将某个目录下的访问权限绑定到某个用户
grep bash /etc/passwd  # 查看系统下所有存在的用户名（/home目录下不一定真实）

sudo vim /etc/sudoers  # 为用户添加sudo权限
username ALL=(ALL)ALL  # 找到【root ALL=(ALL)ALL】行，紧随其后添加新用户sudo权限
```


* 硬盘存储
```
sudo mkfs -t ext4 /dev/sda      # 新的硬盘分区需要创建文件系统
sudo mkdir /datasda             # 在主机上创建新的root文件夹
sudo mount /dev/sda /datasda    # 挂载命令，从实际位置到挂载位置
sudo vim /etc/fstab              # 将自动挂载硬盘的信息写入相关文件，下次开机无需再次挂载
<file system> <mount point>   <type>  <options>       <dump>  <pass>    # /etc/fstab文件表头
/dev/sdb        /datasdb        ext4    defaults        0   0           # 写入/etc/fstab文件中的示例信息
```


* 网卡
```
dmesg | grep -i eth  # 查看网卡信息
lspci | egrep -i eth  # 查看有几块网卡

ifconfig  # 查看网络配置信息
ifconfig eth0 down && ifconfig eth0 up  # 方法1，将网卡网络服务禁用或开启，eth0为网卡名
ifdown eth0 && ifup eth0  # 方法2，将网卡网络服务禁用或开启，eth0为网卡名，需已安装apt install ifupdown2

sudo vim /etc/network/interfaces  # 查看网卡信息，并为网卡配置静态IP地址（旧的Ubuntu系统），需要root用户；退出并保存("Esc"+":wq")
# 编辑<interfaces>文件的一个模板
    # interfaces(5) file used by ifup(8) and ifdown(8)
    auto lo
    iface lo inet loopback
    auto eno1
    iface eno1 inet static
    address 202.120.57.94
    netmask 255.255.255.0
    gateway 202.120.57.254
    dns-nameservers 8.8.8.8
sudo service networking restart  # Linux中重新启动网卡网络服务，但Ubuntu下可能不可用
sudo systemctl restart networking  # Ubuntu下重新启动网卡网络服务，一种选择
sudo service NetworkManager restart  # Ubuntu下重新启动网卡网络服务，使用ifconfig亦可

# Ubuntu环境中，当机器同时连接内外两根网线时，如何同时兼顾两个网络？
# 解决方法：把内网所有网段加入到外网路由中即可，例如192.168.6.0表示内网192.168.6所有网段
sudo route add -net 192.168.6.0 netmask 255.255.255.0 gw 192.168.6.1
# 操作后，可以使用sudo route查看是否已添加，再先后重启（或拔掉网线）外网和内网
```


* 修改默认端口号
```
sudo vim /etc/ssh/sshd_config /etc/ssh/sshd_config.bak  # 备份一下相关系统配置文件
sudo vim /etc/ssh/sshd_config  # 修改系统配置文件，选择性注释掉“Port 22”，添加新的端口号“Port 2222”
iptables -A INPUT -p tcp --dport 2222 -j ACCEPT  # 添加到路由表，将修改内容执行生效
sudo /etc/init.d/sshd restart  # 重启sshd文件生效
sudo service ssh restart  # 如果上面的命令行不通，试一下这种方式重启sshd文件生效
netstat -an | grep "LISTEN "  # 查看端口状态
```


* 配置静态IP地址
```
sudo vim /etc/network/interfaces  # 只适用于系统版本低于Ubuntu18.04的Linux操作系统
sudo service NetworkManager restart

cd /etc/netplan  # 适用于系统版本为Ubuntu18.04，及其以上的Linux/Ubuntu操作系统
sudo vim 01-network-manager-all.yaml  # 若没有该文件，自行复制存在的.yaml文件，或新建
    # 编辑<01-network-manager-all.yaml>文件的一个模板
    network:
      ethernets:
              eno1:
                      addresses: [202.120.57.94/24]
                      gateway4: 202.120.57.254
                      #dhcp: true
                      nameservers:
                              addresses:
                              - 8.8.8.8
      version: 2
      #renderer: NetworkManager
sudo netplan apply

netstat -rn  # 查看gateway4等配置是否正确
telnet 192.168.1.11 2222  # 测试新配置的对应的IP地址和端口号是否可正常的访问
```


* CPU相关操作
```
top  # 查看内存及CPU使用情况
htop  # 查看内存及CPU使用更详细情况, sudo apt-get install htop
free -m  # 单独查看内存使用情况
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c  # 查看CPU基本信息（逻辑CPU、型号、频率等）
cat /proc/cpuinfo | grep physical | uniq -c  # 查看CPU实际的物理核数
cat /proc/meminfo  # 查看内存详细信息
```

* GPU相关操作
```
lspci | grep -i vga  # 查看电脑上的显卡硬件信息
nvidia-smi  # 查看GPU占用情况
watch -n 3 nvidia-smi  # 每隔n秒显示刷新一次GPU详情
cat /usr/local/cuda/version.txt  # 查看CUDA版本
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2  # 查看cudnn版本

# 清除GPU占用，特别是Ctrl+C终止运行后，GPU存储没有及时释放，需要手动清空
torch.cuda.empty_cache()  # 适合在PyTorch内部使用
ps aux | grep python  # 使用ps在命令行按照关键词如python，找到程序的PID，再使用kill结束该进程 kill -9 [pid]
nvidia-smi --gpu-reset -i [gpu_id]  # 直接重置没有被清空的 GPU
```

* 文件压缩与解压缩
```
# .tar 文件
tar -xvf filename.tar  # 解压tar
tar -cvf filename.tar dirname  # 将dirname和其下所有文件（夹）打包

# .tar.gz / .tgz 文件
tar -zxvf filename.tar.gz  # 解压tar.gz
tar -C dirname -zxvf filename.tar.gz  # 将文件解压到目标路径dirname
tar -zcvf filename.tar.gz dirname  # 将dirname和其下所有文件（夹）压缩

# .tar.bz2文件
tar -jxvf filename.tar.bz2  # 解压tar.bz2
tar -jcvf filename.tar.bz2 dirname  # 将dirname和其下所有文件（夹）压缩

# .Z文件
uncompress filename.Z  #解压
compress filename  #压缩

# .rar 文件
rar x filename.rar                     # 解压
rar a filename.rar dirname             # 压缩

# .zip 文件
unzip -O cp936 filename.zip [-d destpath]   # 解压（cp936 可防止乱码）
zip filename.zip dirname                    # 将dirname文件本身尽心压缩，不进行递归处理
zip -r filename.zip dirname                 # 递归处理压缩，将指定目录下的所有文件和子目录一并压缩

cat test.zip* > ./test.zip      # 解压分段压缩的文件test.zip.001, .002, and .003，需要想将其合并
unzip test.zip                  # 再解压合并后的整体文件test.zip 
```

* 文件复制
```
cp -r /folder/source /folder/target   # 文件夹使用-r --recursive

# 远程从某台服务器上拷贝文件。-r针对文件夹；-v动态显示信息；如果该服务器端口不是默认的22，使用-P修改
scp -r -v -P 2345 username@server_IP:/folder/source /folder/target

# 远程同步服务器文件，比scp更强大，不修改文件创建日期，可以不覆盖式传输，支持断点传输。
# -e后的引号内表示修改端口号，-avzut表示的内容，请自行使用rsync -h查看。
# 关于大量文件（文件小而多，传输极为耗时）的远传，需要自行写多线程传输的逻辑代码。
rsync -avzut -e 'ssh -p 2345' username@hostname:SourceFile DestFile  
```

* 文件查找、统计、计数及删除
```
du -sh *  # 显示文件夹下所有文件大小
df -h  # 显示机器上整个文件系统的使用情况
du -B G --max-depth=1  # 以GB为单位显示深度为1的当前各文件夹下的所占存储大小
ls -l ./your/path/ | grep "-" | wc -l  # 查看某文件夹下文件的个数(不包含子文件)
ls -lR ./your/path/ | grep "-" | wc -l  # 查看某文件夹下文件的个数(包含子文件)

find /your/path -name "*.mp4" -exec rm -rf {} \;  # 搜查并删除某个文件夹中的所有.mp4文件，注意不要遗漏 \;
```


* Ubuntu系统设置VPN
```
# [VPN类型]: L2TP/IPsec;  
# [参考链接]: ubuntu安装l2tp/ipsec, https://blog.csdn.net/huaxiao137/artile/details/88715427
# [Ununtu]: 图形界面，可满足在界面上配置“客户端”, 安装完成后，在右上角的网络连接处添加VPN连接，选择ipsec连接
sudo add-apt-repository ppa:nm-l2tp/network-manager-l2tp  # 添加个人软件源
sudo apt-get install network-manager-l2tp network-manager-l2tp-gnome  # 安装l2tp客户端软件  
```
