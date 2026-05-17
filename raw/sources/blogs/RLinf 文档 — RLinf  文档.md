---
title: "RLinf 文档 — RLinf  文档"
source: "https://rlinf.readthedocs.io/zh-cn/latest/"
author:
published:
created: 2026-04-24
description:
tags:
  - "clippings"
---
## RLinf 文档

## 欢迎来到 RLinf！

RLinf 是一个灵活且可扩展的开源基础架构，专为通过强化学习对基础模型进行后训练而设计。名称中的 "inf" 代表 Infrastructure（基础架构），强调其作为新一代训练强大支撑系统的角色；同时也代表 Infinite（无限），象征该系统支持开放式学习、持续泛化和智能发展的无限可能性。

---

[![_images/overview.svg](https://rlinf.readthedocs.io/zh-cn/latest/_images/overview.svg)](https://rlinf.readthedocs.io/zh-cn/latest/_images/overview.svg)

---

**RLinf 的独特之处在于：**

- 宏观到微观流程（Macro-to-Micro Flow）：一种新范式 M2Flow，通过微观级的执行流程完成宏观级的逻辑流程， **解耦逻辑工作流构建（可编程）与物理通信调度（高效执行）** 。
- 灵活的执行模式
	- **共享式** ：所有任务共享全部 GPU。
		- **分离式** ：支持细粒度流水线。
		- **混合式** ：可定制的混合部署，结合了共享式和分离式两种模式。
- 自动调度策略
	- **动态调度** ：动态调度资源分配,最大化资源利用率。
		- **静态调度** ：根据训练任务自动选择最合适的执行模式，无需手动资源分配。
- 具身智能支持
	- 快速适配主流 VLA 模型： [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [π₀](https://github.com/Physical-Intelligence/openpi), [GR00T-N1.5](https://github.com/NVIDIA/Isaac-GR00T.git)
		- 通过标准化 RL 接口支持主流基于 CPU 和 GPU 的模拟器： [ManiSkill3](https://github.com/haosulab/ManiSkill) 、 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 、 [IsaacLab](https://github.com/isaac-sim/IsaacLab)
		- 支持 π₀ 模型族首次基于 flow-matching 动作专家进行的强化学习微调。

**RLinf 拥有出色的训练速度：**

- 结合细粒度流水线的混合式：相比其他框架， **吞吐率提升超过 120%** 。
- 自动在线扩缩策略：训练资源动态扩展，GPU 切换只需数秒， **进一步提高效率 20–40%** ，同时保持 RL 算法的 on-policy 特性。

**RLinf 同时兼具灵活性与易用性：**

- 多种后端集成支持
	- 统一接口可驱动两种互补的后端，无需修改代码即可无缝切换。
		- **FSDP + Hugging Face** ：快速适配新模型与算法，适合初学者与快速原型开发。
		- **Megatron + SGLang** ：优化大规模训练效率，适用于对性能要求极高的专家用户。
- 通过异步通信通道实现自适应通信
- 内建对多种主流强化学习方法的支持，包括 [PPO](https://arxiv.org/abs/1707.06347) 、 [GRPO](https://arxiv.org/abs/2402.03300) 、 [DAPO](https://arxiv.org/abs/2503.14476) 、 [Reinforce++](https://arxiv.org/abs/2501.03262) 等。

---

- [快速开始](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/index.html)
- [SOTA 强化学习复现](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/index.html#sota)
	- [安装说明](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/installation.html)
		- [快速上手 1：在 Maniskill3 上使用 PPO 训练 VLA 模型](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/vla.html)
		- [快速上手 2：使用 GRPO 训练 LLM 进行 MATH 推理](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/llm.html)
		- [多节点训练](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/distribute.html)
		- [评估教程1：具身VLA](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/vla-eval.html)
		- [评估教程2：数学推理LLM](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/start/llm-eval.html)

---

- [教程](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/index.html)
	- [统一编程接口](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/user/index.html)
		- [YAML 配置](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/user/yaml.html)
				- [基于 Worker 的编程接口](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/user/worker.html)
				- [Worker 放置策略](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/user/placement.html)
				- [基于 Ray 的集群启动](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/user/cluster.html)
				- [高层次编程流程概览](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/user/flow.html)
		- [灵活的执行模式](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/mode/index.html)
		- [共享式模式](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/mode/collocated.html)
				- [分离式模式](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/mode/disaggregated.html)
				- [混合式模式](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/mode/hybrid.html)
		- [自动调度](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/scheduler/index.html)
		- [自动扩缩机制](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/scheduler/online-scaling.html)
				- [动态调度](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/scheduler/dynamic-scheduling.html)
				- [自动放置](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/scheduler/auto-placement.html)
		- [弹性通信](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/communication/index.html)
		- [自适应点对点通信](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/communication/collective.html)
				- [流水线中的 Channel 队列](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/communication/channel.html)
		- [算法组件](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/components/index.html)
		- [Replay Buffer 使用教程](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/components/replay_buffer.html)
				- [数据采集](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/components/data_collection.html)
		- [高级特性](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/advance/index.html)
		- [5D 并行配置](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/advance/5D.html)
				- [LoRA 集成](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/advance/lora.html)
				- [切换 SGLang 版本](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/advance/version.html)
				- [检查点恢复](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/advance/resume.html)
				- [Checkpoint 转换](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/advance/convertor.html)
				- [异构软硬件集群配置](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/advance/hetero.html)
				- [云边协同训练配置](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/advance/cloud-edge.html)
				- [训练可视化](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/advance/logger.html)
		- [支持的强化学习算法](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/index.html)
		- [近端策略优化 (PPO)](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/ppo.html)
				- [组相对策略优化 (GRPO)](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/grpo.html)
				- [解耦裁剪与动态采样策略优化 (DAPO)](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/dapo.html)
				- [REINFORCE++](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/reinforce.html)
				- [软演员评论家 (SAC)](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/sac.html)
				- [Cross-Q](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/crossq.html)
				- [结合先验数据的强化学习 (RLPD)](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/rlpd.html)
				- [隐式Q学习 (IQL)](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/iql.html)
				- [异步近端策略优化 (Async PPO)](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/rlalg/async_ppo.html)
		- [扩展框架](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/extend/index.html)
		- [添加新环境](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/extend/new_env.html)
				- [使用 FSDP+HuggingFace 添加新模型](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/extend/new_model_fsdp.html)
				- [使用 Megatron+SGLang 添加新模型](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/extend/new_model_megatron.html)
				- [使用 FSDP+HuggingFace 添加新模型 SFT 训练](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/extend/new_model_sft.html)
				- [Reward Model 使用指南](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/extend/reward_model.html)
				- [Reward Model 使用指南（真机）](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/extend/reward_model_realworld.html)
		- [版本说明](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/tutorials/release.html)

---

- [示例库](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/index.html)
	- [具身智能场景](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/index.html)
		- [智能体场景](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/agentic/index.html)
		- [系统级优化](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/system/index.html)

---

- [论文](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/publications/index.html)
	- [RLinf-USER: Unified System for Real-world Online Policy Learning](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/publications/rlinf_user.html)
		- [RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/publications/rlinf_vla.html)
		- [Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/publications/rlinf_co.html)
		- [RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/publications/rlinf_system.html)
		- [πRL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/publications/pi_rl.html)
		- [WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/publications/wovr.html)
		- [WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/publications/wideseek_r1.html)

---

- [API手册](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/apis/index.html)
	- [Worker 接口](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/apis/worker.html)
		- [Placement 接口](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/apis/placement.html)
		- [Cluster 接口](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/apis/cluster.html)
		- [Channel 接口](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/apis/channel.html)
		- [Actor 接口](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/apis/actor.html)
		- [Rollout 接口](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/apis/rollout.html)
		- [Environment 接口](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/apis/env.html)
		- [Data 接口](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/apis/data.html)
		- [Embodied Data 接口](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/apis/embodied_data.html)
		- [Replay Buffer](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/apis/replay_buffer.html)

---

- [常见问题](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/faq.html)