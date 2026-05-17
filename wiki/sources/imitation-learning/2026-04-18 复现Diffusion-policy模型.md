---
type: source
tags: [Diffusion Policy, Franka, polymetis, robodiff, 机器人, 模仿学习, 实时控制]
sources: [raw/sources/blogs/复现Diffusion-policy模型.md]
created: 2026-04-18
updated: 2026-04-18
---

[[raw/sources/blogs/复现Diffusion-policy模型.md]]

# 复现 Diffusion Policy 模型（Franka Research 3）

> **原文**：[CSDN 博客 - 复现Diffusion-policy模型](https://blog.csdn.net/qq_59001382/article/details/144341544)
> **作者**：[[qq_59001382]] | **发布**：2024-12-09

---

## 核心要点

### 实验硬件
- PC：联想拯救者 Y7000P，RTX 4060，Intel i7-14700HX
- 机械臂：[[wiki/entities/hardware/Franka Research 3]]（系统版本 5.6.0）

### 软件栈三层架构
```
Ubuntu 20.04 + PREEMPT_RT 实时内核
        ↓
polymetis（基于 PyTorch 的 Franka 实时控制器）
        ↓
robodiff（Diffusion Policy 模型运行环境）
```
- 系统基础：[[wiki/entities/hardware/Ubuntu 20.04]] + [[wiki/concepts/infrastructure/PREEMPT_RT实时内核]]
- 策略框架：[[wiki/concepts/generative-models/Diffusion Policy]]

---

## 环境安装要点

### polymetis 安装

**关键陷阱：libfranka 版本不匹配**
- polymetis 脚本默认安装 libfranka **0.9.0**
- Franka Research 3（系统 5.6.0）需要 libfranka **0.13.3**
- 解决：手动替换 `third_party/` 目录下的 libfranka，clone 0.13.3 版本重新编译

```bash
git clone --recursive https://github.com/frankaemika/libfranka --branch 0.13.3
cd libfranka && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
cmake --build .
```

**编译报错处理（`iJIT_IsProfilingActive` 未定义）**：
- 原因：PyTorch 版本与 conda 冲突
- 解决：`pip install --force-reinstall torch==1.13.1`

### robodiff 安装
- 使用项目：[diffusion_policy_vila](https://github.com/Yingdong-Hu/diffusion_policy_vila)（RSS 2023）
- 依赖安装：`conda env create -f conda_environment_real.yaml`
- 额外依赖：MuJoCo、Intel RealSense D455 SDK、spacemouse（`libspnav-dev spacenavd`）

---

## 实验操作

### 双 PC 架构（真实机械臂实验必须）
```
PC1（polymetis 环境）          PC2（robodiff 环境）
  ↓                                  ↓
launch_robot.py               demo_real_franka.py
  ↓                                  ↓
launch_franka_interface_server.py
        ↓
  Franka Research 3 机械臂
```

### Spacemouse 操控说明
| 操作             | 效果      |
| -------------- | ------- |
| 遥感旋钮           | XY 平面移动 |
| 长按 MENU + 遥感上下 | Z 轴运动   |
| 长按 EFI + 遥感    | 末端姿态旋转  |

---

## 常见问题
- **端口占用**（50051）：`sudo lsof -i :50051` 找进程 → `sudo kill -9 PID`
- **SSH 推送失败**：检查 GitHub 公钥配置
- **libfranka 版本错误**：按上述步骤替换为 0.13.3

---

## 关联知识
- 前置：[[wiki/sources/infrastructure/2026-04-18 安装Ubuntu双系统]] → [[wiki/sources/infrastructure/2026-04-18 Ubuntu安装实时内核和显卡驱动]]
- 算法改进：[[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]] 使用 RL 对类似流程训练的策略进行在线微调
