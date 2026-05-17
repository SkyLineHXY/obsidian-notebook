---
title: "Ubuntu安装实时内核和显卡驱动"
source: "https://blog.csdn.net/qq_59001382/article/details/142938618?fromshare=blogdetail&sharetype=blogdetail&sharerId=142938618&sharerefer=PC&sharesource=qq_59001382&sharefrom=from_link"
author:
  - "[[qq_59001382]]"
published: 2024-10-15
created: 2026-04-18
description: "文章浏览阅读6.3k次，点赞37次，收藏59次。"
tags:
  - "clippings"
---
## 一、检查电脑系统环境配置（以下是笔者系统配置）

1、系统版本： Ubuntu 20.04.6 LTS

打开设置，点击关于，查看版本号

2、内核：Ubuntu，Linux 5.15.0-122-generic

打开终端，输入uname -a命令，可查看内核版本号

```bash
uname -a
```

## 二、安装实时内核

### 1、下载内核文件与补丁文件

> https://www.kernel.org/pub/linux/kernel/  
> https://www.kernel.org/pub/linux/kernel/projects/rt/

在以上两个官网搜索与系统内核版本相近的版本下载，笔者选择如下:

内核文件：linux-5.15.167.tar.gz

补丁文件：patch-5.15.167-rt79.patch.gz

### 2、解压内核与补丁文件

①将内核与补丁文件拷贝至主目录（最好这样）

②解压内核文件

```cobol
tar -xzvf linux-5.15.167.tar.gz
```

③进入该内核文件夹下

```bash
cd linux-5.15.167
```

④解压补丁文件，并应用与当前文件夹下

```cobol
gzip -cd ../patch-5.15.167-rt79.patch.gz | patch -p1 --verbose
```

### 3、安装相关依赖

```cobol
sudo apt-get install autoconf bison build-essential dkms dwarves fakeroot flex libelf-dev libiberty-dev libidn11 libidn11-dev libncurses-dev libpci-dev libssl-dev libudev-dev minizip openssl zlibc zstd
```

### 4、对内核配置进行设置

①复制当前的内核配置文件

```bash
cp /boot/config-$(uname -r) .config
```

②进入内核配置config交互界面

```
make menuconfig
```

③在config界面下修改如下配置

<table><tbody><tr><td>General Setup</td><td>Timers subsystem</td><td>Timer tick handling</td><td>设置为Full dynticks system</td></tr><tr><td>General Setup</td><td>Timers subsystem</td><td colspan="2" rowspan="1">开启High Resolution Timer Support</td></tr><tr><td>General Setup</td><td>Preemption Model</td><td colspan="2" rowspan="1">设置为 Fully Preemptible Kernel(RT)</td></tr><tr><td colspan="2" rowspan="1">Processor type and features</td><td>Timer frequency</td><td>设置为 1000 HZ</td></tr></tbody></table>

④编辑.config文件内容

```lua
sudo gedit .config
```

修改内容如下：

> CONFIG\_SYSTEM\_TRUSTED\_KEYS=""  
> CONFIG\_SYSTEM\_REVOCATION\_KEYS=""

### 5、编译内核（多线程，编译时间漫长），且安装内核

```crystal
make -j$(nproc) deb-pkg -j 32
```
```cobol
sudo dpkg -i ../*.deb
```

### 6、查看实时内核是否安装成功

①查看系统内所有内核，检查是否安装成功

```cobol
grep "menuentry '" /boot/grub/grub.cfg
```

②检查当前内核版本号

```bash
uname -r
```
```bash
uname -a
```

如下是笔者成功安装的示例：

> menuentry ' **Ubuntu，Linux 5.15.167-rt79** ' --class ubuntu --class gnu-linux --class gnu --class os $menuentry\_id\_option 'gnulinux-5.15.167-rt79-advanced-1d60b691-c660-40ef-915a-2d88dbf7d0d7' {  
> menuentry ' **Ubuntu, with Linux 5.15.167-rt79** (recovery mode)' --class ubuntu --class gnu-linux --class gnu --class os $menuentry\_id\_option 'gnulinux-5.15.167-rt79-recovery-1d60b691-c660-40ef-915a-2d88dbf7d0d7' {

## 三、安装实时内核显卡驱动

①首先禁用nouveau，nouveau是默认显卡驱动，但它是核显

```cobol
sudo gedit /etc/modprobe.d/blacklist.conf
```

修改如下内容：

> blacklist nouveau  
> options nouveau modeset=0

更新系统配置，重启验证是否禁用成功

```perl
sudo update-initramfs -u   

sudo reboot

lsmod | grep nouveau
```

②下载适合的显卡驱动版本

查看非实时内核适配的显卡驱动版本

```
ubuntu-drivers devices
```

进入NVIDIA官网下载电脑对应的显卡驱动，并放至主目录下。笔者电脑适配的显卡驱动如下：

> NVIDIA-Linux-x86\_64-535.183.01.run

③创建显卡驱动启动脚本文件

在主目录下创建install-nvidia.sh文件

```
sudo gedit install-nvidia.sh
```

将install-nvidia.sh文件添加如下内容，且修改NV\_FILE的对应文件

```cobol
#!/bin/bash

 

BUILD_BASE=\`pwd\`

NV_FILE="NVIDIA-Linux-x86_64-xxx.run"  # 这里改成自己下载的.run文件名（之前安装过的）

#NV_URL="https://us.download.nvidia.cn/XFree86/Linux-x86_64/430.50/${NV_FILE}"  # 之前已经下好了显卡驱动.run文件，就不用从网上下载了，直接注释掉，而且如果要下载的话，则慢

NEED_TO_COMPILE_NV_KO=1

 

function clean_env() {

 

    [ -d ./${NV_DIR} ] && rm -rf ./${NV_DIR}

}

 

function check_env() {

    

    # check if in rt kernel

    uname -r | grep rt 1>/dev/null 2>&1 

    if [ $? -ne 0 ]

    then

        echo "Not in rt kernel, Please install apollo kernel and reboot machine first."

        exit 2

    fi

 

    # check if nv ko already in kernel

    if [ ! -f /lib/modules/\`uname -r\`/kernel/drivers/video/nvidia.ko ]

    then

        export NEED_TO_COMPILE_NV_KO=1

    fi

}

 

function prepare_nv() {

 

    ## download nv install file from nvidia home page 

    #if [ ! -f ./${NV_FILE} ]

    #then

    #   echo "Downloading ${NV_FILE} from nvidia website..."

    #    wget ${NV_URL} -O ${NV_FILE}

    #    if [ $? -ne 0 ]

    #    then

    #        echo "Downloading ${NV_FILE} failed, please check your network connection!"

    #        rm -rf ./${NV_FILE}

    #        exit 1

    #    fi

    #fi

    ###########上面是下载驱动的代码，我们已经提前下载好了，就不需要这段代码了，直接注释掉############

 

    # +x 

    chmod +x ./${NV_FILE}

    echo "Extracting nvidia install run file..."

    ./${NV_FILE} -x 1>/dev/null 2>&1

    NV_DIR="\`echo ${NV_FILE} | awk -F '.run' '{print $1}'\`"

    NV_VERSION="\`echo ${NV_FILE} | awk -F '-' '{print $4}' | awk -F '.run' '{print $1}'\`"

 

    export NV_DIR

    export NV_VERSION

    export NVIDIA_SOURCE="${NV_DIR}/kernel"

}

 

function install_lib() {

   

    NV_LIB_OUTPUT_PATH="/usr/lib/x86_64-linux-gnu/"

    NV_BIN_OUTPUT_PATH="/usr/bin/"

 

    [ -f ./${NV_DIR}/libnvidia-ml.so.${NV_VERSION} ] && /bin/cp -f ./${NV_DIR}/libnvidia-ml.so.${NV_VERSION} ${NV_LIB_OUTPUT_PATH}

    [ -f ./${NV_DIR}/libnvidia-fatbinaryloader.so.${NV_VERSION} ] && /bin/cp -f ./${NV_DIR}/libnvidia-fatbinaryloader.so.${NV_VERSION} ${NV_LIB_OUTPUT_PATH}

    [ -f ./${NV_DIR}/libnvidia-ptxjitcompiler.so.${NV_VERSION} ] && /bin/cp -f ./${NV_DIR}/libnvidia-ptxjitcompiler.so.${NV_VERSION} ${NV_LIB_OUTPUT_PATH}

    [ -f ./${NV_DIR}/libcuda.so.${NV_VERSION} ] && /bin/cp -f ./${NV_DIR}/libcuda.so.${NV_VERSION} ${NV_LIB_OUTPUT_PATH}

    [ -f ./${NV_DIR}/nvidia-modprobe ] && /bin/cp -f ./${NV_DIR}/nvidia-modprobe ${NV_BIN_OUTPUT_PATH}

    [ -f ./${NV_DIR}/nvidia-smi ] && /bin/cp -f ./${NV_DIR}/nvidia-smi ${NV_BIN_OUTPUT_PATH}

 

    chmod +x /usr/bin/nvidia*

    chmod +s /usr/bin/nvidia-modprobe

 

    # link for nvidia

    /bin/rm -rf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so

    /bin/ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.${NV_VERSION} /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1

    /bin/ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so

 

    /bin/rm -rf /usr/lib/x86_64-linux-gnu/libcuda.so  /usr/lib/x86_64-linux-gnu/libcuda.so.1

    /bin/ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.${NV_VERSION} /usr/lib/x86_64-linux-gnu/libcuda.so.1

    /bin/ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so

 

    # take effect

    /sbin/ldconfig 1>/dev/null 2>&1

}

 

function build_nv() {

 

    if [ ${NEED_TO_COMPILE_NV_KO} == 0 ]

    then

        return

    fi

 

    NVIDIA_MOD_REL_PATH='kernel/drivers/video'

    NVIDIA_OUTPUT_PATH="/lib/modules/\`uname -r\`/${NVIDIA_MOD_REL_PATH}"

    CPUNUM=\`cat /proc/cpuinfo | grep processor | wc | awk -F " " '{print $1}'\`

 

    export IGNORE_PREEMPT_RT_PRESENCE=true

    cd ${NVIDIA_SOURCE} && make -j ${CPUNUM} module

    cd ${BUILD_BASE}

 

    unset IGNORE_PREEMPT_RT_PRESENCE

 

    mkdir -p ${NVIDIA_OUTPUT_PATH}

 

    [ -f ${NVIDIA_SOURCE}/nvidia.ko ] && cp ${NVIDIA_SOURCE}/nvidia.ko ${NVIDIA_OUTPUT_PATH}

    [ -f ${NVIDIA_SOURCE}/nvidia-modeset.ko ] && cp ${NVIDIA_SOURCE}/nvidia-modeset.ko ${NVIDIA_OUTPUT_PATH}

    [ -f ${NVIDIA_SOURCE}/nvidia-drm.ko ] && cp ${NVIDIA_SOURCE}/nvidia-drm.ko ${NVIDIA_OUTPUT_PATH}

    [ -f ${NVIDIA_SOURCE}/nvidia-uvm.ko ] && cp ${NVIDIA_SOURCE}/nvidia-uvm.ko ${NVIDIA_OUTPUT_PATH}

 

    depmod -a

}

 

# check environment

check_env

 

# prepare for nvidia

prepare_nv

 

# build nvidia.ko

build_nv

 

# install user lib

install_lib

 

# clean environment

clean_env

 

echo "Done to install nvidia kernel driver and user libraries."
```

④运行显卡驱动脚本，检查实时内核下显卡驱动是否安装成功

添加权限

```bash
chmod +x ./install-nvidia.sh
```

运行脚本，启动驱动

```bash
sudo ./install-nvidia.sh
```

显卡驱动脚本运行无误后，重启

```
nvidia-smi
```