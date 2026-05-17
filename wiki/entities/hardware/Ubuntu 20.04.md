---
type: entity
tags: [操作系统, Linux, Ubuntu, 机器人, 系统环境]
sources: [raw/sources/blogs/安装Ubuntu双系统.md, raw/sources/blogs/Ubuntu安装实时内核和显卡驱动.md, raw/sources/blogs/复现Diffusion-policy模型.md]
created: 2026-04-18
updated: 2026-04-18
---

# Ubuntu 20.04

**Ubuntu 20.04.6 LTS**（Long Term Support）是当前知识库中机器人系统环境配置的**标准操作系统**。

## 在知识库中的角色

| 来源 | 用途 |
|------|------|
| [[wiki/sources/infrastructure/2026-04-18 安装Ubuntu双系统]] | 目标系统（与 Windows 11 双引导） |
| [[wiki/sources/infrastructure/2026-04-18 Ubuntu安装实时内核和显卡驱动]] | 基础系统，基础内核 5.15.0-122-generic |
| [[wiki/sources/imitation-learning/2026-04-18 复现Diffusion-policy模型]] | Franka 机械臂控制所需的系统环境 |

## 关键配置细节
- **检查内核**：`uname -a`（默认 5.15.x 系列）
- **RTX 40 系 GPU**：需在 GRUB 添加 `nomodeset` 参数才能正常显示
- **与实时内核的关系**：Ubuntu 20.04 标准内核可编译并替换为 [[wiki/concepts/infrastructure/PREEMPT_RT实时内核]]
- **与 Ubuntu 22.04 的关系**：双系统安装方法完全相同，polymetis 也支持 22.04

## 注意事项
- 安装前必须关闭 Windows BitLocker 加密
- BIOS 需为 UEFI 模式（GUID 分区）才支持双系统与 Secure Boot 配置
