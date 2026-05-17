---
title: 复现Diffusion-policy模型
source: https://blog.csdn.net/qq_59001382/article/details/144341544?spm=1001.2014.3001.5501
author:
  - "[[qq_59001382]]"
published: 2024-12-09
created: 2026-04-18
description: 文章浏览阅读4.7k次，点赞42次，收藏66次。
tags:
  - clippings
---

### 0 引言

本文将介绍扩散模型算法在Franka机械臂上的部署与应用，包括环境配置、空间鼠标控制操作、数据采集以及数据训练等环节的详细说明。

### 1 实验条件

电脑配置：拯救者Y7000P笔记本、显卡 4060、CPU 14700  
机械臂硬件：Franka机械臂（Franka Research 3）机器人系统版本5.6.0  
系统环境：ubuntu20.04、实时内核PREEMPT\_RT、 anaconda 环境（ubuntu22.04也可以）

系统版本： Ubuntu 20.04.6 LTS打开设置，点击关于，查看版本号2、内核：Ubuntu，Linux 5.15.0-122-generic打开终端，输入uname -a命令，可查看内核版本号uname -a。_ubuntu安装实时内核。将diffusion-policy算法部署至franka机械臂上的实验环境包括ubuntu系统与实时内核环境、polymetis环境(基于pytorch的franka实时控制器)和robodiff环境(diffusion-policy模型依赖环境)。_diffusion policy论文复现

### 2 实验环境配置

将diffusion-policy算法部署至franka机械臂上的实验环境包括ubuntu系统与实时内核环境、polymetis环境(基于 pytorch 的franka实时控制器)和robodiff环境(diffusion-policy模型依赖环境)。

#### 2.1 ubuntu20.04系统与实时内核环境安装

ubuntu20.04系统安装可参考笔者博客[[安装Ubuntu双系统]]。
实时内核与显卡驱动安装可参考笔者博客[[Ubuntu安装实时内核和显卡驱动]]。

#### 2.2 polymetis环境安装

polymetis是基于pytorch的franka实时控制器，核心控制franka机械臂依旧依托libfranka底层代码来开发机械臂的接口，能够完成对franka的实时控制策略。polymetis环境安装的步骤如下：

①在终端下通过git方式克隆github远程仓库的源码(一定要通过git的方式将源码拷贝下来)

```bash
git clone git@github.com:facebookresearch/fairobash
```

---

**注意：** 在git克隆的过程中可能出现无法读取远程仓库问题，原因可能是密钥设置不正确。解决方法可以参考其他博客。

[github 推送出现“git@github.com: Permission denied (publickey). fatal: 无法读取远程仓库。请确认您有正确的访问权限并且仓库存在。” 问题\_fatal: 无法读取远程仓库。 请确认您有正确的访问权限并且仓库存在。-CSDN博客文章浏览阅读3.5w次，点赞51次，收藏129次。今天想把vuepress开发的博客推送到github，bash deploy.sh出现了如下错误git@github.com: Permission denied (publickey). fatal: 无法读取远程仓库。请确认您有正确的访问权限并且仓库存在。上网查询，发现是密钥设置的不对解决办法：1.首先检查有没有在GitHub的https://github.com/settings...\_fatal: 无法读取远程仓库。 请确认您有正确的访问权限并且仓库存在。https://blog.csdn.net/iiiliuyang/article/details/104203197](https://blog.csdn.net/iiiliuyang/article/details/104203197 "github 推送出现“git@github.com: Permission denied (publickey). fatal: 无法读取远程仓库。请确认您有正确的访问权限并且仓库存在。” 问题_fatal: 无法读取远程仓库。 请确认您有正确的访问权限并且仓库存在。-CSDN博客")

---

②进入fairo/polymetis目录下，通过项目作者提供的环境yml文件配置polymetis环境。

```bash
cd fairo/polymetis

conda env create -f ./polymetis/environment.yml

conda activate polymetis-local
bash
```

③成功创建conda虚拟环境后，在虚拟环境下pip 下载 polymetis包。

```bash
pip install -e ./polymetisbash
```

④通过脚本安装libfranka的依赖，这是驱动franka机械臂硬件的关键。但脚本安装下来的libfranka是0.9.0版本的，此版本显然与Franka Research 3(5.6.0)所需版本不匹配，后续将重新安装正确的libfranka 0.13.3版本。

```bash
./scripts/build_libfranka.shbash
```

⑤对polymetis项目进行编译，生成可执行文件。

```bash
mkdir -p ./polymetis/build

cd ./polymetis/build

cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=[OFF/ON] -DBUILD_TESTS=[OFF/ON] -DBUILD_DOCS=[OFF/ON]

make -j
bash
```

---

**注意：** make-j是可能会出现报错：

> libtorch\_cpu.so: undefined reference to \`iJIT\_IsProfilingActive'  
> libtorch\_cpu.so: undefined reference to \`iJIT\_GetNewMethodID'
> 
> libtorch\_cpu.so: undefined reference to \`iJIT\_NotifyEvent'

原因可能是pytorch版本与conda版本存在冲突，重新pip安装torch依赖。

```bash
pip install --force-reinstall torch==1.13.1 --index-url https://download.pytorch.org/whl/cpubash
```

---

按照上述①～⑤步后就完成在anaconda下创建了一个polymetis-local虚拟环境，并且下载了需要的依赖。

---

**注意：** 连接机械臂硬件通过launch\_robot.py启动franka控制服务器会出现如下问题：

> libfranka: Incompatible library version (server version: x, library version: x)

此问题出现的原因是libfranka的版本与机械臂版本不配对，Franka Research 3(5.6.0)对应是libfranka 0.13.3版本。解决方法重新安装libfranka，步骤如下：

①进入libfranka源码的目录,将0.9.0版本的libfranka删除或者重命名。

> /home/xx/fairo/polymetis/polymetis/src/clients/franka\_panda\_client/third\_party

②在third\_party文件夹下git克隆libfranka0.13.3版本或在github上手动下载。

```bash
git clone --recursive https://github.com/frankaemika/libfranka --branch 0.13.3bash
```

③根据如下操作完成libfranka 0.13.3版本安装。

```bash
cd libfranka

mkdir build

cd build

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..

cmake --build .
bash
```

④成功安装libfranka 0.13.3源码后，重新编译polymetis项目，参考前面的步骤⑤。  

---

#### 2.3 robodiff环境安装

robodiff环境是 diffusion-policy 项目运行所需要的环境，主要参考github社区diffusion-policy项目的README。 **非常感谢大佬们的开源。**

①克隆或手动下载Diffusion\_Policy \_vila项目源码。  
[GitHub - Yingdong-Hu/diffusion\_policy\_vila: \[RSS 2023\] Diffusion Policy Visuomotor Policy Learning via Action Diffusion\[RSS 2023\] Diffusion Policy Visuomotor Policy Learning via Action Diffusion - Yingdong-Hu/diffusion\_policy\_vilahttps://github.com/Yingdong-Hu/diffusion\_policy\_vila](https://github.com/Yingdong-Hu/diffusion_policy_vila "GitHub - Yingdong-Hu/diffusion_policy_vila: [RSS 2023] Diffusion Policy Visuomotor Policy Learning via Action Diffusion") ②安装Mujoco 机器人模拟库所需的依赖。

```bash
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 patchelfbash
```

③安装realsense的SDK，可参考其他博客。

[【视觉传感器系列】—— Intel RealSense D455深度相机驱动安装与运行\_realsense驱动-CSDN博客文章浏览阅读3.5k次，点赞31次，收藏59次。Intel RealSense D455是一款双摄像头深度相机，具备高精度深度感知技术和激光投影器，可实时捕捉场景的深度信息和RGB图像。其双摄像头设计和激光投影技术确保在各种环境下都能以高精度和高分辨率获取场景深度，保证可靠性和稳定性。此外，D455具有灵活性和易用性，支持多种操作系统和开发平台，广泛应用于机器人导航、自动驾驶、增强现实、虚拟现实等领域，为各种智能系统提供关键的感知能力。\_realsense驱动https://blog.csdn.net/2401\_82458959/article/details/138764398?fromshare=blogdetail&sharetype=blogdetail&sharerId=138764398&sharerefer=PC&sharesource=qq\_59001382&sharefrom=from\_link](https://blog.csdn.net/2401_82458959/article/details/138764398?fromshare=blogdetail&sharetype=blogdetail&sharerId=138764398&sharerefer=PC&sharesource=qq_59001382&sharefrom=from_link "【视觉传感器系列】—— Intel RealSense D455深度相机驱动安装与运行_realsense驱动-CSDN博客")

④根据作者提供的yml文件创建conda虚拟环境并且下载相应的依赖,适用于真实机械臂环境。

```bash
conda env create -f conda_environment_real.yaml
bash
```

⑤安装spacemouse相关依赖。

```bash
sudo apt install libspnav-dev spacenavd; sudo systemctl start spacenavdbash
```

### 3 实验操作

实验操作部分将记录笔者和伙伴们在franka硬件平台复现Diffusion-policy模型所完成的实验。

#### 3.1 （实验一）使用polymetis控制器控制franka机械臂

①模拟机械臂平台

进入launch\_robot.py文件所在的目录下,打开终端，并进入polymetis-local虚拟环境。

> launch\_robot.py文件目录：/home/xx/fairo/polymetis/polymetis/python/scripts

在终端内运行 python 文件，并且添加相关配置文件。运行后，机械臂服务器正常启动，并且会弹出pybullet物理实时模拟平台，franka机械臂模型会出现在界面内。

```bash
launch_robot.py robot_client=franka_sim use_real_time=false gui=truebash
```

启动模拟机器人服务器后，运行polymetis包含的运动例程，例如执行1\_set\_ee\_pose.py文件。

运动例程所在的路径：

> /home/zh/fairo/polymetis/examples

打开终端进入上述路径，运行机器人运动python文件。

```bash
python 1_set_ee_pose.pybash
```

---

**注意：** 运行后可能出现port已经被占用的问题，解决方法是释放默认的port(50051)占用。

第一步：找到占用端口的进程ID。

```bash
sudo lsof -i :50051bash
```

第二步：杀掉占用端口的进程。

```bash
sudo kill -9 PIDbash
```

第三步：如果端口被防火墙占用，可以关闭防火墙。

```bash
sudo ufw disablebash
```

如果上述方法不行，可以重新启动计算机清除存在的进程占用。

---

②真实机械臂平台

与上述模拟平台一样，进入launch\_robot.py文件所在的目录下,打开终端，并进入polymetis-local虚拟环境。在终端内运行python文件，并且添加相关配置文件。运行后，机械臂服务器正常启动，终端中会显示已经连接。

```bash
launch_robot.py robot_client=franka_hardwarebash
```

启动真实机器人服务器后，运行polymetis包含的运动例程，例如执行1\_set\_ee\_pose.py文件。

运动例程所在的路径：

> /home/xx/fairo/polymetis/examples

打开终端进入上述路径，运行机器人运动python文件，机器人开始运动。

```bash
python 1_set_ee_pose.pybash
```

#### 3.2（实验二）使用spacemouse来控制franka机械臂

实验条件：两台PC电脑，一台必须包含polymetis环境，另一台必须包含robodiff环境。  
按照如图关系连接两个PC电脑与franka机械臂控制柜。

![](https://i-blog.csdnimg.cn/direct/9aeb550d64824ac2b5ef50fec23e2ad5.png)

①PC电脑一

搭建好硬件平台后，首先在polymetis环境下启动机械臂的服务器。

```bash
launch_robot.py robot_client=franka_hardwarebash
```

然后依旧在polymetis环境下进入scripts\_real目录下启动FrankaInterface 服务器。

> /home/xx/diffusion\_policy\_vila/scripts\_real

```bash
python scripts_real/launch_franka_interface_server.pybash
```

②PC电脑二

启动spacemouse控制机械臂相关文件，就能使用空间鼠标来控制机械臂。

```bash
python demo_real_franka.py -o data/demo_pusht_real --robot_ip 10.7.5.2bash
```

**spacemouse操作提醒：**
只使用spacemose的遥感旋钮可以实现X Y轴平面下的移动；
长按MENU按钮，能够允许Z轴运动，再遥感旋钮上拉或下拉，控制Z轴方向的运动；
长按EFI按钮，是进行旋转运动，再遥感旋钮可以更改机械臂末端姿态。
