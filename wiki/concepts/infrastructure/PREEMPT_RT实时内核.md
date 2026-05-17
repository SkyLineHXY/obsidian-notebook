---
type: concept
tags: [Linux内核, 实时系统, 机器人控制, PREEMPT_RT, 系统配置]
sources: [raw/sources/blogs/Ubuntu安装实时内核和显卡驱动.md, raw/sources/blogs/复现Diffusion-policy模型.md]
created: 2026-04-18
updated: 2026-04-18
---

# PREEMPT_RT 实时内核

**PREEMPT_RT**（Fully Preemptible Kernel）是 Linux 内核的实时补丁，通过将大多数内核代码路径设置为可抢占，大幅降低中断响应延迟，实现**硬实时（Hard Real-Time）**特性。

## 核心特性
- **完全可抢占**：内核临界区可被高优先级任务打断
- **高分辨率定时器**：支持微秒级定时精度
- **动态时钟节拍（Full Dynticks）**：减少不必要的定时器中断，降低延迟抖动
- 典型延迟：从标准内核的毫秒级降至 **< 100 微秒**

## 为什么机器人控制需要实时内核？
控制 Franka 等工业机械臂需要在**严格时间约束**内完成力矩命令：
- polymetis（Franka 的 PyTorch 控制器）要求系统具备实时能力
- 非实时系统可能导致控制信号延迟 → 机械臂抖动或急停保护触发

## 安装方法（来自知识库）
详见：[[wiki/sources/infrastructure/2026-04-18 Ubuntu安装实时内核和显卡驱动]]

关键 menuconfig 参数：

| 配置项                   | 值                                 |
| --------------------- | --------------------------------- |
| Preemption Model      | **Fully Preemptible Kernel (RT)** |
| Timer tick handling   | Full dynticks system              |
| High Resolution Timer | 开启                                |
| Timer frequency       | **1000 HZ**                       |

## 主要陷阱
1. **NVIDIA 驱动不兼容**：标准安装程序拒绝在 RT 内核下安装，需设置 `IGNORE_PREEMPT_RT_PRESENCE=true` 手动编译 `nvidia.ko`
2. **内核签名**：必须清空 `CONFIG_SYSTEM_TRUSTED_KEYS` 避免证书错误
3. **版本匹配**：内核版本与 RT 补丁版本必须严格一致（如 5.15.167 对应 rt79）

## 在知识库中的出现
| 来源                                               | 角色                                |
| :----------------------------------------------- | :-------------------------------- |
| [[wiki/sources/infrastructure/2026-04-18 Ubuntu安装实时内核和显卡驱动]]    | 主题：编译安装 RT 内核的完整教程                |
| [[wiki/sources/imitation-learning/2026-04-18 复现Diffusion-policy模型]] | 前置条件：polymetis 控制 Franka 机械臂的必要环境 |

## 相关实体
- 应用平台：[[wiki/entities/hardware/Franka Research 3]]
- 基础系统：[[wiki/entities/hardware/Ubuntu 20.04]]
