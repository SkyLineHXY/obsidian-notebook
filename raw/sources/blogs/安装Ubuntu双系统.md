---
title: "安装Ubuntu双系统"
source: "https://blog.csdn.net/qq_59001382/article/details/142951549?fromshare=blogdetail&sharetype=blogdetail&sharerId=142951549&sharerefer=PC&sharesource=qq_59001382&sharefrom=from_link"
author:
  - "[[qq_59001382]]"
published: 2024-10-15
created: 2026-04-18
description: "文章浏览阅读2.3w次，点赞101次，收藏291次。"
tags:
  - "clippings"
---
## 前言

笔者笔记本电脑配置：

①电脑名：拯救者Y7000P（2024款）

②CPU：Intel(R) Core(TM) i7-14700HX

③显卡：RXT 4060

④系统环境：Windows 11 家庭 中文 版

⑤BIOS模式（ UEFI ）和磁盘分区方式（GUID）

## 一、检查电脑类型

1、win+r打开运行，输入msinfo32查看电脑BIOS模式（UEFI和MBR）

```cobol
msinfo32
```

**笔者电脑BIOS模式：UEFI**

2、菜单开始栏鼠标右击，选择磁盘管理，确定ubuntu系统待装在哪块磁盘里，按照图中①~③进行查看磁盘分区类型。

![](https://i-blog.csdnimg.cn/direct/0c8a4f446fff40d28e8203f6469aaac7.png)

**笔者电脑磁盘分区类型：GUID**

## 二、关闭BitLocker 加密

**注意：** 关闭BitLocker加密非常重要，否则导致电脑无法正常进入windows系统

**途径：** 打开设置，搜索BitLocker加密，进行关闭即可

确保如图小红框处无 **BitLocker已加密**

![](https://i-blog.csdnimg.cn/direct/77fe30956b174c24ad803f24088180de.png)

## 三、制作ubuntu安装盘

1、准备一个内存8G以上的U盘

2、下载Ventoy软件，将U盘制作为启动盘

①Ventoy软件下载链接： [https://www.ventoy.net/cn/](https://www.ventoy.net/cn/ "https://www.ventoy.net/cn/")

②打开Ventoy2Disk.exe，插入准备的U盘，点击安装

![](https://i-blog.csdnimg.cn/direct/0ade57aa5d324d589ac0b927251fbd43.png)

③安装完成后，重新插拔U盘，电脑识别Ventoy盘

## 四、下载 Ubuntu 安装映像文件

1、Ubuntu安装映像网址： [https://repo.huaweicloud.com/ubuntu-releases/](https://repo.huaweicloud.com/ubuntu-releases/ "https://repo.huaweicloud.com/ubuntu-releases/")

2、选择 **ubuntu-xx-xx-xx-desktop-amd64.iso** 文件下载

笔者选择Ubuntu安装映像文件： **ubuntu-20.04.6-desktop-amd64.iso**

需要安装Ubuntu22.04，选择Ubuntu安装映像文件： **ubuntu-22.04.4-desktop-amd64.iso**

（Ubuntu20.04和Ubuntu22.04安装方法一样）

3、将下载好的Ubuntu安装映像文件拷贝至启动U盘中

## 五、压缩windows分区

**建议：** 装双系统建议为笔记电脑扩容，加装一块固态硬盘

1、菜单开始栏鼠标右击，选择磁盘管理，压缩一块足够的空闲区域

①选择一个具备足够容量的卷，鼠标右击，选择压缩卷

![](https://i-blog.csdnimg.cn/direct/82205d0202434b78accd13336b0c45eb.png)

②输入待压缩的空间量，点击压缩

③磁盘管理中出现一块空闲区域，即为ubuntu系统待安装区域

## 六、进入BIOS，修改配置

**BIOS启动按键**

| 联想笔记本 | F2或F12 |
| --- | --- |
| 华硕笔记本 | Esc |
| 华为笔记本 | F12 |
| 惠普笔记本 | F9 |

1、关闭显卡直连（windows内也可以关闭）

2、关闭Secure Boot（重要）

3、关闭Intel RST

4、在BIOS内设置U盘引导为首启动（ **前提：u盘已插入笔记本电脑** ）

## 七、安装Ubuntu系统

完成BIOS设置后，进行保存与退出，开始进入U盘引导启动

1、进入ubuntu安装界面前的选择

①选择待安装的ISO系统 （ **ubuntu-20.04.6-desktop-amd64.iso）**

②选择Boot in normal mode（默认第一个）

③选择Try or Install Ubuntu（默认第一个）

2、进入ubuntu安装界面 **（安装步骤）**

①界面1：系统语言选择，Chinese或English，点击Install Ubuntu

②界面2：键盘布局，设置语言

③界面3：选择正常安装，也可勾选为图形或无线硬件，以及其他媒体格式安装第三方软件

**注意：安装ubuntu时下载更新不要勾选，否则安装时间漫长**

④界面4：安装类型

类型一：可以覆盖原来的安装的Ubuntu系统重新安装

类型二：选择其他选项，自定义安装（ **推荐，新人首选** ）

**关键：（选择其他选项后）**

第一步：寻找设备中在windows下压缩的空闲区域（可以根据内存大小判别）并且选中

第二步：点击左下角的+号，挂载点选择 /

第三步：为引导器选择安装位置，选中windows Boot Manager这个efi分区设备号，然后在（安装启动引导器的设备）下拉菜单里找到带有windows Boot Manager的引导器设备，点击现在安装

⑤界面5：选择地区

⑥界面6：设置信息（计算机名称、密码等）

等待安装成功

## 八、设置Ubuntu系统引导参数

①重启电脑，在系统引导菜单下选择ubuntu,按下E

②在Linux行的$vt\_handoff后面添加nomodeset

③点击F10，保存与退出

注意：系统引导参数一致生效，需要进入Ubuntu系统，修改grub内容

```cobol
sudo gedit /etc/default/grub
```

在quiet splash添加nomodeset（无引导启动）

```cobol
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nomodeset"
```

## 九、卸载Ubuntu系统（UEFI引导模式）

1、安装DiskGenius软件 [https://www.diskgenius.cn/download.php](https://www.diskgenius.cn/download.php "https://www.diskgenius.cn/download.php")

2、打开DiskGenius.exe

3、选中Ubuntu安装的分区，鼠标右键删除分区，点击保存更改

![](https://i-blog.csdnimg.cn/direct/3464b57399324fe491c2ea93121ae3c1.png)

4、删除Ubuntu引导项

①选择SYSTEM\_DRV(0)

②选择EFI

③选择ubuntu

④选中ubunt内的全部文件，彻底删除，点击保存更改

**注意：** 如果不删除引导ubuntu引导项，会导致重新启动计算机会出现grud命令栏，输入exit可进入windows。如果还不行可以进入BIOS更改启动项