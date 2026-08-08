---
type: source
tags:
  - VLA
  - FlowMatching
  - RealTimeControl
  - Inpainting
  - ActionChunking
  - LeRobot
sources:
  - raw/assets/papers/VLA/Black 等 - 2025 - Real-Time Execution of Action Chunking Flow Policies.pdf
created: 2026-07-02
updated: 2026-07-02
---

# Real-Time Execution of Action Chunking Flow Policies (RTC)

**arXiv**: 2506.07339
**作者**: Kevin Black, Manuel Y. Galliker, Sergey Levine（Physical Intelligence & UC Berkeley）
**发表**: NeurIPS 2025
**代码（官方 Kinetix）**: https://github.com/Physical-Intelligence/real-time-chunking-kinetix
**代码（LeRobot 生产实现）**: https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/rtc
**项目主页**: https://pi.website/research/real_time_chunking
**摄取日期**: 2026-07-02
**摄取来源**: 用户提供 PDF + MinerU 转换；LeRobot 仓库 sparse clone 源码分析

[[raw/assets/papers/VLA/Black 等 - 2025 - Real-Time Execution of Action Chunking Flow Policies.pdf]]
[[raw/sources/papers/VLA/Black 等 - 2025 - Real-Time Execution of Action Chunking Flow Policies/Black 等 - 2025 - Real-Time Execution of Action Chunking Flow Policies.md]]

---

## 一句话摘要

RTC 是一个**免训练的推理时算法**，把"边执行当前动作块、边生成下一个动作块"的异步执行问题重构为**流匹配 inpainting 问题**：冻结那些必然会被执行的动作（前缀），用带软掩码的 ΠGDM 引导把剩余动作"填补"成与前缀连续的新块，从而让任意 diffusion/flow VLA 在数百毫秒推理延迟下仍能平滑、实时、无停顿地运行。

---

## 核心背景与动机

### 问题：动作分块（action chunking）无法解决延迟问题

动作分块策略 $\pi(\mathbf{A}_t \mid \mathbf{o}_t)$ 每次推理输出一段未来动作 $\mathbf{A}_t = [\mathbf{a}_t, \dots, \mathbf{a}_{t+H-1}]$（$H$ 为**预测视界 prediction horizon**），实际只执行前 $s \le H$ 个（$s$ 为**执行视界 execution horizon**，通常 $s \approx H/2$）。

- **同步推理（synchronous）**：在执行视界末尾才启动推理并"等待"，当模型延迟 $\delta > \Delta t$（控制周期）时产生**可见停顿**，改变机器人动力学，造成训练/部署的分布偏移。
- **朴素异步（naive async）**：提前启动推理、并行执行，但由于生成 $\mathbf{A}_{s-d}$ 时无法预知 $s-d$ 到 $s$ 之间发生了什么，新旧块在衔接点 $\mathbf{a}_{s-1|0} \to \mathbf{a}_{s|s-d}$ 可能任意跳变（mode-jumping），产生 out-of-distribution 的高加速度冲击。
- **时序集成（Temporal Ensembling, TE）**：对多个块的预测取平均，但**多个有效动作的平均不一定有效**（多模态分布），反而更糟。

延迟到底有多大？论文给出实测：RTX 4090 上 3B 的 $\pi_0$ 仅 KV cache prefill 就 46ms，而 50Hz 控制要求 $\Delta t = 20$ms；远程推理再加 13–20ms 网络延迟；OpenVLA(7B) 优化后在 A100 上仍需 321ms。因此 $\delta \le \Delta t$ 在现代 VLA 上几乎不可能，**异步是必需的**。

![[raw/sources/papers/VLA/Black 等 - 2025 - Real-Time Execution of Action Chunking Flow Policies/images/4ef40a751a91d0eecc9bb5e97cf3d538679a72a5aaae973f8037f5dad03ea080.jpg]]
*图 2：相邻块之间的典型分叉。旧块（黑）计划从障碍上方走，新块（红）走下方，但新块要 $d=7$ 步后才可用。朴素异步会从 $a_{10}$ 跳到 $a_{11}'$ 产生极高 OOD 加速度；TE 插值虽降低加速度但产生无效动作。*

### 关键洞察

> 把实时分块视作 **inpainting（图像修补）** 问题。新块必须与旧块"兼容"：前 $d$ 个动作（$d$ = 推理延迟对应的控制步数）在新块可用前就已执行完毕，因此把它们**冻结**为已知会执行的值；剩余动作则以此冻结前缀为约束"填补"出来——正如修补一张被抠掉一块的图像。

关键定义：$d := \lceil \delta / \Delta t \rceil$ 为**推理延迟（inference delay）**，即从收到 $\mathbf{o}_t$ 到 $\mathbf{A}_t$ 可用之间的控制步数。实时约束在 $d \le H - s$ 时可满足。

---

## 方法：Real-Time Chunking (RTC)

### 前置：流匹配采样

策略用 conditional flow matching 训练；从高斯噪声 $\mathbf{A}_t^0$ 出发，按学到的速度场 $\mathbf{v}_\pi$ 积分 $\tau: 0 \to 1$：

$$
\mathbf{A}_t^{\tau + \frac{1}{n}} = \mathbf{A}_t^{\tau} + \frac{1}{n}\,\mathbf{v}_\pi(\mathbf{A}_t^{\tau}, \mathbf{o}_t, \tau)
$$

其中 $n$ 为去噪步数（控制任务常用极小值，如 $n=5$）。

### 3.1 推理时 inpainting（ΠGDM 引导）

基于 Pokle 等的 training-free 图像逆问题算法（源自 pseudoinverse guidance / ΠGDM）。在每个去噪步给速度场加一个基于梯度的引导项，使最终生成逼近目标 $\mathbf{Y}$（= 被掩码破坏的期望结果，这里即旧块的剩余动作）：

$$
\mathbf{v}_{\Pi GDM}(\mathbf{A}_t^{\tau}, \mathbf{o}_t, \tau) = \mathbf{v}(\mathbf{A}_t^{\tau}, \mathbf{o}_t, \tau) + \min\!\left(\beta, \frac{1-\tau}{\tau \cdot r_\tau^2}\right)\left(\mathbf{Y} - \widehat{\mathbf{A}_t^1}\right)^{\!\top}\!\operatorname{diag}(\mathbf{W})\,\frac{\partial \widehat{\mathbf{A}_t^1}}{\partial \mathbf{A}_t^{\tau}}
$$

$$
\widehat{\mathbf{A}_t^1} = \mathbf{A}_t^{\tau} + (1-\tau)\,\mathbf{v}(\mathbf{A}_t^{\tau}, \mathbf{o}_t, \tau), \qquad r_\tau^2 = \frac{(1-\tau)^2}{\tau^2 + (1-\tau)^2}
$$

- $\widehat{\mathbf{A}_t^1}$：当前对**最终去噪结果**的一步估计（clean sample estimate）。
- 引导项是一个 **vector-Jacobian product**，可用反向自动微分（对 $\mathbf{A}_t^\tau$ 求导）计算。
- $\beta$ 是作者新增的**引导权重裁剪（guidance weight clipping）**：在 $\tau=0$ 处 $\frac{1-\tau}{\tau r_\tau^2} \to \infty$，且控制任务去噪步数极少（$n=5$）时高权重会导致动作块发散，故裁剪。消融显示 $\beta=5$ 即足够（A.2）。

### 3.2 软掩码（Soft Masking）—— 跨块连续性的关键

只用前 $d$ 个重叠动作做硬掩码时，$d$ 较小则引导信号弱，新块仍可能切换策略产生不连续。解决方案：利用**全部 $H-s$ 个重叠动作**，把掩码 $\mathbf{W}$ 设为实值而非 0/1：

$$
\mathbf{W}_i = \begin{cases}
1 & \text{if } i < d \\
c_i\,\dfrac{e^{c_i}-1}{e-1} & \text{if } d \le i < H-s \\
0 & \text{if } i \ge H-s
\end{cases}
\quad\text{where } c_i = \frac{H-s-i}{H-s-d+1},\; i \in \{0,\dots,H-1\}
$$

即：前 $d$ 个（冻结区）权重为 1；末 $s$ 个（无重叠，需全新生成）权重为 0；中间按指数从 1 衰减到 0——越远的未来动作越不确定、权重越低。$\mathbf{W}$ 直观上调节对旧块每个动作的"注意力"。消融（A.4）：指数衰减 > 线性衰减 > 硬掩码。

![[raw/sources/papers/VLA/Black 等 - 2025 - Real-Time Execution of Action Chunking Flow Policies/images/9949c786822576f84b180dda50cc458f657924a5acaeaf2333a2a461b78c9b28.jpg]]
*图 3：RTC 注意力示意。若推理延迟 $d=4$，则 $a_{0:3}$ 被冻结（引导权重 1）；中间区 $a_{4:10}$ 权重指数衰减；末尾 $s$ 个全新生成（权重 0）。执行视界约束为 $d \le s \le H-d$。*

### 3.3 完整系统（Algorithm 1）

- **GETACTION**($\mathbf{o}_{next}$)：控制器每 $\Delta t$ 调用一次，消费 $\mathbf{a}_{t-1}$、提供新观测、返回下一动作。
- **INFERENCELOOP**：后台线程持续生成下一块，保证动作永远可用。用**过去延迟的缓冲队列**保守预测下一次延迟 $d = \max(Q)$。执行视界逐块自适应：$s = \max(d, s_{\min})$。
- **GUIDEDINFERENCE**：用 Eq.5 算 $\mathbf{W}$，把 $\mathbf{A}_{prev}$ 右填充到长度 $H$，从噪声出发按 Eq.2 逐步引导去噪。

---

## LeRobot 代码实现详解（`src/lerobot/policies/rtc`）

LeRobot 把 RTC 拆成 6 个模块，`RTCProcessor` 是引导核心，`ActionQueue` 负责异步队列，`LatencyTracker` 估计延迟。

### 文件结构

| 文件                                         | 职责                                                   |
| ------------------------------------------ | ---------------------------------------------------- |
| `modeling_rtc.py`                          | `RTCProcessor`：ΠGDM 引导去噪步 + 软掩码权重生成（核心）              |
| `configuration_rtc.py`                     | `RTCConfig` dataclass：调度类型、$\beta$、执行视界              |
| `action_queue.py`                          | `ActionQueue`：线程安全队列，RTC 模式下按延迟丢弃前缀、替换队列             |
| `latency_tracker.py`                       | `LatencyTracker`：滑动窗口内的 max/p95 延迟估计（对应 $d=\max(Q)$） |
| `relative.py`                              | 相对动作策略的前缀重锚（把绝对坐标前缀转回模型空间）                           |
| `debug_tracker.py` / `debug_visualizer.py` | 去噪轨迹调试可视化                                            |

### 核心 1：`RTCProcessor.denoise_step`（对应论文 Eq.2–3）

关键代码（`modeling_rtc.py:212–229`）：

```python
tau = 1 - time                      # LeRobot 时间 1→0，需反转得到论文的 τ:0→1
...
with torch.enable_grad():
    v_t = original_denoise_step_partial(x_t)   # 基础速度场 v_π
    x_t.requires_grad_(True)
    x1_t = x_t - time * v_t                     # clean sample 估计 Â¹ (Eq.3)
    err  = (prev_chunk_left_over - x1_t) * weights   # 加权误差 (Y - Â¹)·diag(W)
    grad_outputs = err.clone().detach()
    correction = torch.autograd.grad(x1_t, x_t, grad_outputs)[0]  # VJP ∂Â¹/∂x_t

# 引导权重 min(β, (1-τ)/(τ·r_τ²))
inv_r2 = ((1-tau)**2 + tau**2) / (1-tau)**2
c = torch.nan_to_num((1 - tau) / tau, posinf=max_guidance_weight)
guidance_weight = torch.minimum(c * inv_r2, max_guidance_weight)

result = v_t - guidance_weight * correction     # 减号：LeRobot 速度方向相反
```

**与论文的三处符号对应**（务必注意方向约定差异）：

1. **时间反转**：论文 $\tau: 0\to1$；LeRobot `time: 1→0`（`dt = -1/n`，从噪声 `time=1` 积分到数据 `time=0`），故 `tau = 1 - time`。
2. **clean 估计符号**：论文 $\widehat{\mathbf{A}_t^1} = \mathbf{A}^\tau + (1-\tau)v$；LeRobot 写 `x1_t = x_t - time * v_t`。因为 $(1-\tau) = \text{time}$，且 LeRobot 的 $v$ 指向"时间减小"方向（数据→噪声反向），两个符号翻转恰好抵消 → 等价。
3. **引导符号**：论文对速度场做 $v + w\cdot g$（加）；LeRobot 是 `v_t - guidance_weight * correction`（减），同样源于速度方向相反。
4. **引导权重恒等式**：`c * inv_r2 = [(1-τ)/τ]·[(τ²+(1-τ)²)/(1-τ)²] = (τ²+(1-τ)²)/(τ(1-τ)) = (1-τ)/(τ·r_τ²)`，与论文 Eq.2 完全一致，最后由 `max_guidance_weight`（即 $\beta$）裁剪。

### 核心 2：软掩码 `get_prefix_weights`（对应论文 Eq.5）

```python
def get_prefix_weights(self, start, end, total):   # start=inference_delay(d), end=execution_horizon
    start = min(start, end)
    if schedule == ZEROS:   # 硬掩码：只有前 d 个为 1
        weights = torch.zeros(total); weights[:start] = 1.0
    elif schedule == ONES:  # 前 end 个全 1
        weights = torch.ones(total);  weights[end:] = 0.0
    elif schedule == LINEAR:
        lin = self._linweights(start, end, total)          # start..end 线性 1→0
        weights = self._add_leading_ones(self._add_trailing_zeros(lin, total, end), start, total)
    elif schedule == EXP:
        lin = self._linweights(start, end, total)
        lin = lin * torch.expm1(lin).div(math.e - 1)       # 论文 Eq.5 的 c·(e^c-1)/(e-1)
        weights = self._add_leading_ones(self._add_trailing_zeros(lin, total, end), start, total)
```

四种调度：`ZEROS`=硬掩码，`ONES`=不衰减，`LINEAR`=线性衰减，`EXP`=论文的指数衰减（`expm1` 实现 $\frac{e^{c}-1}{e-1}$）。**注意**：默认 `RTCConfig.prefix_attention_schedule = LINEAR`（代码里有 `# Todo change to exp` 注释，论文推荐 EXP）。另一处工程简化：LeRobot 用 `execution_horizon` 作为衰减终点 `end`，而论文公式用 $H-s$；两者语义等价（重叠区终点），但参数化方式不同。

### 核心 3：异步队列 `ActionQueue` + 延迟估计（对应 Algorithm 1）

- `get_left_over()`：返回旧块**未消费**的原始动作，作为下一次推理的 `prev_chunk_left_over`（即论文的 $\mathbf{A}_{prev}$）。
- `merge(..., real_delay)` → `_replace_actions_queue`：RTC 模式下**丢弃新块前 `clamped_delay` 个动作**（推理期间已被执行的部分），再整块替换队列并重置 `last_index=0`——对应 Algorithm 1 里 `t = t - s` 的索引重置。
- `LatencyTracker.max()` / `p95()`：滑动窗口内保守估计延迟，对应论文 $d = \max(Q)$。
- `RTCConfig.execution_horizon` 默认 10，`max_guidance_weight` 默认 10.0（$\beta$）。

### 核心 4：与流策略的集成（`modeling_smolvla.py` / `modeling_pi0.py`）

`RTCProcessor` 通过依赖注入插入采样循环（`modeling_smolvla.py:sample_actions`）：

```python
for step in range(num_steps):
    time = 1.0 + step * dt                       # dt = -1/num_steps
    def denoise_step_partial_call(input_x_t):    # 只吃 x_t、返回 v_t 的闭包
        return self.denoise_step(x_t=input_x_t, prefix_pad_masks=..., past_key_values=..., timestep=...)
    if self._rtc_enabled():
        v_t = self.rtc_processor.denoise_step(
            x_t=x_t, prev_chunk_left_over=kwargs["prev_chunk_left_over"],
            inference_delay=kwargs["inference_delay"], time=time,
            original_denoise_step_partial=denoise_step_partial_call,
            execution_horizon=kwargs["execution_horizon"])
    else:
        v_t = denoise_step_partial_call(x_t)
    x_t = x_t + dt * v_t
```

**设计要点**：RTC 是一层**装饰器/包裹（wrapper）**——把原策略的单步去噪函数 `denoise_step_partial_call` 作为 `original_denoise_step_partial` 传入，RTC 在其外部叠加引导修正。因此它对 $\pi_0$、SmolVLA 等任意 flow VLA 即插即用，无需改训练，与论文"out of the box, no re-training"的主张一致。首步 `prev_chunk_left_over is None` 时直接返回 `v_t`（无引导）。

---

## 实验结果

### 仿真：Kinetix 12 个高动态任务（$H=8$, 4 层 MLP-Mixer）

- **延迟鲁棒性**：RTC 对推理延迟最鲁棒，随 $d$ 增大对 BID 的优势扩大；TE 全程表现差（印证多模态平均无效）。
- **计算量对比**：BID 采样 64 个动作块（32 强 + 32 弱），计算量远大于 RTC。
- **软掩码有效性**：硬掩码在 $d$ 较小时明显弱于软掩码。
- **闭环收益**：得益于跨块连续性，RTC 在执行视界减小时性能**单调提升**（更充分利用闭环校正）。

![[raw/sources/papers/VLA/Black 等 - 2025 - Real-Time Execution of Action Chunking Flow Policies/images/15fb3044be2e0fbff389bd4cbbbd1c204b0bb7ab6ff810285fd2cf1b5e6e7f18.jpg]]
*图 4：硬掩码 vs 软掩码——硬掩码与冻结区匹配差、方向变化更急。*

### 真实世界：$\pi_{0.5}$ VLA，6 个双臂灵巧任务

设置：$H=50$, $\Delta t=20$ms, $n=5$；基线延迟 76ms，RTC 97ms（因反传每个去噪步 2.5×，见 Table 3）；LAN 远程推理起始 $d\approx6$，并注入 +100ms（$d\approx11$）、+200ms（$d\approx16$）。

- **平均吞吐（throughput）**：RTC 在所有延迟下最优，+100/+200ms 时统计显著；**RTC 对注入延迟完全无退化**，而同步推理线性退化，两种 TE 因剧烈震荡触发机器人保护性急停、根本无法运行。
- **速度**：相同运动 RTC 比同步推理快 20%（图 1）。
- **精度任务**：点火柴（唯一不可重试、最需精度的任务）RTC 最终分数大幅领先；铺床（最难任务）RTC 也有强效果。
- **延迟分解**（Table 3）：RTC 相对 vanilla $\pi_{0.5}$ 每去噪步 14ms→35ms（2.5×），总 76ms→97ms。BID(N=16, full) 达 223ms（2.3× RTC）。

---

## 局限性与未来工作

作者自陈（Sec.6）：
1. **计算开销大**：相比直接采样基础策略，需反传每个去噪步。
2. **仅适用于 diffusion/flow 策略**（不适用自回归 VLA）。
3. **实验场景仍偏操作类**：更动态的设定（如腿式运动）仅在仿真验证、未做真机。

---

## 与已有方法的关系

- **[[wiki/concepts/generative-models/Flow Matching]]**：RTC 完全建立在 flow matching 采样之上，引导项作用于速度场 $\mathbf{v}_\pi$。
- **[[wiki/entities/models/π₀.₅]]**：真机实验的基础策略。
- **[[wiki/sources/frameworks/2026-04-19 LeRobot]]**：RTC 已作为生产级模块进入 LeRobot（`policies/rtc`），可挂载到 SmolVLA / π₀。
- **[[wiki/concepts/imitation-learning/ACT]]**：ACT/ALOHA 提出的 Temporal Ensembling 是 RTC 的主要对比基线（论文证明其在多模态下失效）。
- **Bidirectional Decoding (BID)**：最相关工作，用拒绝采样保跨块连续；RTC 用引导式 inpainting，性能更好且计算更省。

---

## 新概念追踪

**首次出现，追踪中**：
- **ΠGDM (Pseudoinverse-Guided Diffusion Models) 引导**：inpainting 的梯度引导核心；出现于 Pokle 2023、Song 2023 与本文；若再遇独立来源可升格概念页。
- **Real-Time Chunking / 异步动作块执行**：本源独有；若 LeRobot 之外再有独立实现可升格。
- **Guidance Weight Clipping ($\beta$)**：少步去噪下稳定 ΠGDM 的关键 trick；仅本来源。

---

## 关联页面

- [[wiki/concepts/generative-models/Flow Matching]] — 采样与速度场基础
- [[wiki/analyses/Flow Matching 完整数学推导]] — 数学背景
- [[wiki/entities/models/π₀.₅]] — 真机基础策略
- [[wiki/sources/frameworks/2026-04-19 LeRobot]] — 生产实现载体
- [[wiki/sources/vla/2026-04-29 π₀.₇]] — 同门 π 系列 VLA
