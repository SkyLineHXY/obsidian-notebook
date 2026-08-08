---
type: source
tags: [Data Collection, Active Perception, VR Teleoperation, Imitation Learning, Bimanual Manipulation, Egocentric Vision, In-the-Wild, VLA]
sources: [Zeng 等 - 2025 - ActiveUMI Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations.md]
created: 2026-08-08
updated: 2026-08-08
---

[[raw/sources/papers/VLA/Zeng 等 - 2025 - ActiveUMI Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations/Zeng 等 - 2025 - ActiveUMI Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations.md]]

# ActiveUMI: Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations (Zeng 2025)

> **一句话**：把 [[wiki/entities/hardware/UMI]] 的「手持夹爪 + GoPro + SLAM」换成「Meta Quest 3S 头显 + 改装控制器」，用 VR inside-out 追踪替代视觉 SLAM，并顺势把**操作员的头部 6-DoF 也录进动作空间**，让策略学会自己控制视角。

**论文**：Zeng 等, *"ActiveUMI: Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations"*, arXiv:2510.01607, 2025-10-02
**机构**：上海大学 / Stanford / 美的集团（Midea Group）
**项目**：https://activeumi.github.io

![[raw/sources/papers/VLA/Zeng 等 - 2025 - ActiveUMI Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations/images/fig1-teaser.jpg]]
*Figure 1: ActiveUMI 采集的人类演示（左）与主动感知带来的能力（右）——长程、抗遮挡、高精度、可形变/铰接物体操作。*

---

## 核心论点

现有 in-the-wild 采集方案（UMI / DexUMI / FastUMI / AirExo / DexCap）几乎都**只有腕部相机**。腕相机随手臂运动，其视角由**操作需求**决定，而非由**感知需求**决定。这带来三个后果：

1. 长程任务中无法「回头确认」全局状态；
2. 遮挡发生时无法主动绕开；
3. 与真正采用头部相机的机器人平台（humanoid、带云台的双臂）**观测分布不匹配**。

人类做操作时会**主动转头**来管理遮挡与获取上下文。ActiveUMI 的主张是：**learning how to look is as important as learning what to do**——视角选择本身应当是任务的一部分，被记录、被学习、被预测。

---

## 系统设计

![[raw/sources/papers/VLA/Zeng 等 - 2025 - ActiveUMI Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations/images/fig2-hardware.jpg]]
*Figure 2: ActiveUMI 硬件。改装的 Quest 3S 控制器（挂载目标机器人夹爪副本 + 微电机 + 鱼眼相机）与自足式背包（电池 + PC）。*

### 1. VR 夹爪控制器

- **基座**：改装 Meta Quest 3S 商用控制器，利用头显 inside-out 追踪——头显机载摄像头持续三角测量控制器上的红外 LED 阵列，实时解算 6-DoF 位姿 $(x,y,z,\text{roll},\text{pitch},\text{yaw})$。
- **非侵入式夹爪**：**不替换**机器人原装夹爪，而是复制一份**相同的**夹爪挂到操作员控制器上。这与 UMI「机器人必须换装 UMI 同款夹爪」形成关键区别——ActiveUMI 可部署到大量 stock robot 上而无需机械改造。
- **夹爪驱动**：控制器上集成微电机直驱夹爪开合，操作员直觉控制抓握。
- **腕部鱼眼相机**：每个控制器加装鱼眼相机，提供局部操作细节，作为头部第一人称视角的**补充**（而非 UMI 中的唯一视觉源）。

### 2. 头显（HMD）的双重角色

- **高精度定位中枢**：Quest 3S 的 SLAM 系统建立统一世界坐标系，**同时**追踪头部与两个控制器的 6-DoF 位姿。一个消费级设备解决三点追踪。
- **动态顶部相机**：头显前置彩色相机作为「头相机」，其视角**内在耦合于操作员的视线**。这正是 active perception 的数据来源。

### 3. 沉浸式采集（Immersive Rendering）

VR 环境内实时渲染机器人手臂 3D 模型，与操作员手持控制器精确对齐。操作员**能看见机器人的运动学**，从而在采集时就自然规避不可达/奇异构型。

> 对比：UMI 是「盲采」——人按人的习惯运动，事后再用机器人 URDF 做运动学可行性过滤，把不可行的轨迹**丢弃**（浪费采集成本）。ActiveUMI 把这个约束从**事后过滤**前移到**事前对齐**。

### 4. 可穿戴计算单元

背包内含电池 + 小型 PC，操作员完全脱离固定工作站，可在任意环境自由移动采集。

### 5. 标定三件套

ActiveUMI 记录**绝对坐标**（相对于初始标定建立的统一世界系），因此标定质量直接决定数据质量：

| 方法 | 机制 | 作用 |
|---|---|---|
| **In-Situ 环境设置** | 按控制器 'B' 键重置 6-DoF 零点；坐标轴在头显内实时渲染 | 任意环境快速起采，操作员可直觉对齐虚拟参考系与物理工作空间 |
| **夹爪 Placeholder** | 物理定位夹具（docking station），控制器插入后相对位姿固定为预定义状态；按键即完成标定 | 保证每次 session 起始位姿一致 |
| **触觉零点反馈** | 夹爪进入零点 3 cm 内时控制器马达高频振动 | 无需读数值即可确认对齐，大幅提速标定 |

---

## 主动感知的学习机制

![[raw/sources/papers/VLA/Zeng 等 - 2025 - ActiveUMI Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations/images/fig3-system-overview.jpg]]
*Figure 3: 左为采集流程与数据配置（6-DoF×3 + 1-DoF×2，30 Hz，三路相机）；右为部署推理——策略预测头部位姿主动搜索「红色罐子在哪」。*

**训练时**：把操作员 HMD 的实时 6-DoF 位姿作为**额外的动作维度**记录进数据集。模型因此能学到「头部运动（视觉注意）↔ 手部动作」之间的相关性。

**部署时**：策略预测机器人头部的 6-DoF 目标位姿，由低层控制器执行，使机器人**主动模仿操作员学到的注意力模式**——动态调整视角、绕开遮挡。

**动作空间**：$6\text{-DoF} \times 3$（左夹爪 + 右夹爪 + 头）$+\ 1\text{-DoF} \times 2$（两夹爪开合）$=$ **20 DoF**，30 Hz。

**策略网络**：π0（flow-matching VLA），微调 50k iterations，cosine LR scheduler，语言条件（如 "Take the red drink out of the backpack."）。

**硬件平台**：3 × ARX R5 六自由度机械臂——两臂做双手操作（各带腕部鱼眼相机），**第三臂充当「脖子」**承载头相机。

---

## 实验

![[raw/sources/papers/VLA/Zeng 等 - 2025 - ActiveUMI Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations/images/fig5-tasks.jpg]]
*Figure 5: 六个评测任务——积木拆分（精密）、衬衫折叠（可形变）、绳子入盒（长程）、工具箱清理（铰接）、瓶子放置（位置随机化鲁棒性）。*

### 主结果：主动感知的价值（每任务 10 trials）

| 相机配置 | 瓶子放置 | 绳子入盒 | 衬衫折叠 | 积木拆分 | 包中取饮料 | **平均** |
|---|---|---|---|---|---|---|
| Wrist-only（UMI 设定） | 60% | 20% | 10% | 0% | 40% | **26%** |
| + 固定顶部头相机 | 60% | 40% | 40% | 20% | 50% | **42%** |
| **ActiveUMI（主动头）** | **90%** | **70%** | **80%** | **30%** | **80%** | **70%** |

### 泛化：新环境 / 新物体

| 相机配置 | 瓶子放置 | 绳子入盒 | 衬衫折叠 | 积木拆分 | 包中取饮料 | **平均** |
|---|---|---|---|---|---|---|
| Wrist-only（UMI 设定） | 30% | 0% | 0% | 0% | 0% | **6%** |
| + 固定顶部头相机 | 30% | 10% | 20% | 0% | 20% | **16%** |
| **ActiveUMI** | **70%** | **50%** | **80%** | **30%** | **50%** | **56%** |

> **关键读数**：域内差距 26% → 70%（+44），新环境差距 6% → 56%（**+50，相对差距进一步拉大**）。视角受限的策略在视觉分布偏移下**近乎完全崩溃**，而主动视角提供了对视觉漂移的弹性。

作者给出的两条机理假设：
1. In-the-wild 采集时演示者会移动头部和身体；主动相机让策略能**补偿**这种运动，而不是把它当作观测噪声；
2. 主动视角选择让策略能**按需获取任务关键信息**（如验证抓取是否成功）。

### 混合训练：少量遥操作数据的锚定作用

衬衫折叠任务，20 trials：

| 遥操作数据占比 | 10% | **1%** | 0% |
|---|---|---|---|
| 平均成功率 | 90% | **95%** | 80% |

结论：ActiveUMI 数据高度 sample-efficient，仅需**极少量**真机遥操作数据即可显著提升——与「大规模仿真 + 少量真机」的既有结论一致。这意味着可以用低成本 in-the-wild 数据承担主体，真机数据只做域锚定。

### 采集吞吐与精度

![[raw/sources/papers/VLA/Zeng 等 - 2025 - ActiveUMI Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations/images/fig6-data-comparison.jpg]]
*Figure 6(a)-(c): 三种采集方式对比——ActiveUMI 演示、真机遥操作、裸手演示（绳子入盒 / 衬衫折叠）。*

**吞吐**（相对裸手演示的耗时倍率，越小越好）：

| 任务 | ActiveUMI | 真机遥操作 |
|---|---|---|
| 绳子入盒 | 2.06× | 3.27× |
| 衬衫折叠 | **1.49×** | 2.63× |

ActiveUMI 处于裸手与遥操作之间，稳定快于遥操作。

**精度**：Relative Pose Error 对比 UMI —— **ActiveUMI 4.0 mm vs UMI 10.1 mm（约 2.5× 提升）**。测量方法：操作员持控制器夹爪在卷尺两端定位（标称距离 100 cm 递减至 10 cm，共 10 个点），记录 6-DoF 位姿序列后在真机上回放，用同一卷尺量取两夹爪内侧实际距离。

---

## 与 UMI 的详细对比

见独立对比页：[[wiki/comparisons/UMI 系列采集接口对比]]

浓缩版：

| 维度 | UMI (Chi 2024) | ActiveUMI (Zeng 2025) |
|---|---|---|
| 位姿来源 | ORB-SLAM3 视觉惯性里程计（GoPro 鱼眼 + IMU） | VR inside-out 追踪（头显三角测量控制器 IR LED） |
| 追踪精度 | MoCap 基准 < 1 cm / < 4° | RPE 4.0 mm（UMI 同实验 10.1 mm） |
| 视觉源 | 腕部鱼眼 ×1/臂（155° + 侧面镜子伪立体） | 腕部鱼眼 ×2 + **HMD 头相机**（可动） |
| 视角控制 | 被动，末端执行器中心 | **主动**，策略预测头部 6-DoF |
| 动作表示 | **相对** EE 轨迹 $\Delta T_t = T_{t_0}^{-1}T_t$ | **绝对**坐标（统一世界系） |
| 标定需求 | 几乎无（相对轨迹天然抗漂移/抗标定误差） | **强依赖**：零点重置 + placeholder 夹具 + 触觉确认 |
| 夹爪耦合 | 强绑定 UMI 专用平行夹爪，机器人须换装 | **非侵入**：复制机器人原装夹爪挂到控制器 |
| 操作员反馈 | 无 | VR 内渲染机器人手臂，沉浸式对齐 |
| 运动学约束 | 事后 URDF 过滤（丢数据） | 事前可视化对齐 |
| 策略 | Diffusion Policy（DDPM U-Net）+ CLIP ViT | π0（flow VLA），语言条件 |
| 机器人侧要求 | 任意臂 + UMI 夹爪 | 双臂 + **第三臂作为主动头**（20 DoF） |
| 穿戴负担 | 手持一个夹爪，极轻 | 头显 + 背包（电池 + PC） |
| 采集速度 | 接近裸手 | 裸手的 1.49–2.06× |

---

## 批判性阅读（Caveats）

在引用本文数据前需注意以下问题：

1. **「UMI」基线并非真正的 UMI**。Table 1/2 中的 "UMI" 行实际是**在 ActiveUMI 自己的数据与 π0 架构下做的 wrist-camera-only 消融**（缺失第三视角的 visual token 用 padding 填充）。真正的 UMI 系统包含相对轨迹动作表示、侧面镜子伪立体、延迟匹配、Diffusion Policy——这些**一个都没有复现**。因此 26% vs 70% **不能读作「ActiveUMI 打败 UMI」**，只能读作「在本文设定下移除头相机会掉 44 个点」。这是本文最主要的比较公平性问题。
2. **RPE 定义自相矛盾**。论文公式 $\Delta L = |L_\text{replay} - L_\text{measure}|$，$\text{RPE} = \frac{\Delta L}{L_\text{measure}} \times 100\%$ 定义的是**百分比**，但 Figure 6(e) 报告的单位是 **mm**。此外正文写「the RPE of UMI is 2.5x smaller than UMI」——主语写错，应为 ActiveUMI。
3. **任务名称前后不一致**。4.2 节讨论 "PourWater task" 并称高出固定相机 30%、高出腕相机 60%，但任务列表与所有表格中均无 PourWater，对应位置是 "Take Drink from Bag"（80% vs 50% vs 40%，恰好是 +30/+40）。数字也对不上。
4. **3.1 节整段文字重复了第 2 节 Related Work**（"Data collection is a central pillar of modern deep learning..." 一整段），明显的写作/排版事故。
5. **统计功效不足**。主表每格仅 10 trials，混合训练表 20 trials。1% vs 10% 遥操作数据的 95% vs 90% 仅相差 1 次成功，**统计上不显著**，不应据此断言「1% 是最优配比」。
6. **未报告数据规模**。论文没有给出采集了多少条演示轨迹、多少小时、多少场景——这对一篇以「可扩展数据采集」为卖点的工作是重要缺失，也使 in-the-wild 泛化结论难以复现。
7. **「in-the-wild」名不副实**。所有图示与评测均在桌面工作台完成；UMI 原文在 30 个不同物理位置（家庭/办公室/餐厅/室外）采集 1,400 条演示。ActiveUMI 的「新环境」泛化实验只换了一个环境。

---

## 深度分析：VR 设备作为 UMI 的特点

### 优势

**1. 把最脆弱的一环换成工业级成熟方案**
UMI 管线中最易失效的是 ORB-SLAM3：需预先扫描建图、无纹理场景（白墙、纯色桌面）崩溃、长时间遮挡跟丢（FastUMI 正是为此改用 RealSense T265）、单目 SfM 存在尺度模糊（UMI 靠 IMU 恢复公制尺度）、且必须离线后处理。VR inside-out 追踪是消费级量产验证过的方案：实时、低延迟、公制尺度天然正确、开箱即用。**精度提升 2.5×（10.1 mm → 4.0 mm）**。

**2. 一个设备天然提供多点追踪 → 使能主动感知**
头显同时解算头部 + 双控制器共三个 6-DoF 位姿。这不是附加功能，而是 VR 架构的**固有产物**。UMI 架构里根本没有「头」这个概念——要加一路可动的第三人称视角，需要额外的定位系统。VR 方案是「免费」拿到的。

**3. 双向交互通道（UMI 完全没有）**
- **输出**：头显显示 → 渲染机器人手臂做 embodiment 对齐；渲染坐标轴做标定对齐。
- **输入**：控制器按键 → In-Situ 零点重置；控制器马达 → 触觉标定确认。

这把采集从「单向录制」变成「闭环人机交互」，直接降低操作员认知负荷与废数据率。

**4. Embodiment 对齐从事后过滤前移到事前引导**
UMI 采完再用 URDF 过滤掉不可达轨迹是纯损耗。ActiveUMI 让操作员实时看见机器人构型，在采集瞬间就自然避开——这在长程任务上省下的成本是复利的。

**5. 硬件解耦方向相反但更彻底**
FastUMI 的解耦思路是做**标准化插拔指尖**去适配 90% 的夹爪型号；ActiveUMI 的思路更简单粗暴——**直接复制目标机器人的原装夹爪**挂到控制器上。前者要设计通用适配层，后者只需 3D 打印一个副本。对「部署到 stock robot」而言，后者摩擦更小。

### 代价与固有局限

**1. 绝对坐标 vs 相对轨迹：一次有代价的哲学倒退**
这是最深刻的分歧。UMI 论文用消融**明确证明**：改用绝对坐标动作，Cup Arrangement 成功率从 100% 崩到 **25%**（标定误差主导）。相对轨迹 $\Delta T_t = T_{t_0}^{-1}T_t$ 天然抗 SLAM 漂移、抗相机位移、无需绝对标定——这正是 UMI 敢称「in-the-wild」的技术根基。

ActiveUMI 反其道而行，用绝对世界坐标，靠 VR 追踪精度 + 三重标定流程硬扛。**但这不是设计失误，而是主动感知强加的约束**：头部位姿与双手位姿必须表达在**同一个世界系**里，策略才能学到「头看向哪里」与「手在哪里」的空间关系。相对轨迹表达不了这种跨肢体的空间对齐。

代价是系统从 **stateless 变成 stateful**：UMI 拿起夹爪就能采，ActiveUMI 每个 session 都要 dock 校准、重置零点。讽刺的是——ActiveUMI 在 "in-the-wild 随意性" 这个 UMI 的立身之本上是**退步**的，三个标定方法的存在本身就是这一退步的证据。

**2. VR 追踪的失效域与 SLAM 正好互补（不是包含）**
Inside-out 追踪依赖头显摄像头**看得见**控制器的 IR LED。失效场景：
- 手伸出头显视野（背后、身侧、头顶、桌面下方）
- 身体/物体遮挡控制器
- **强光/阳光下红外受干扰** → 户外场景是硬伤

此时系统退化为 IMU 推算，快速漂移。而 UMI 的 SLAM 依赖**环境**特征：户外纹理丰富时反而更稳，白墙/纯色场景崩溃。

> **结论：两者不是「VR 全面优于 SLAM」，而是失效域互补。** ActiveUMI 的 4.0 mm RPE 是在桌面卷尺实验这一**对 VR 最有利**的条件下测得的（控制器全程在头显视野正前方）。

**3. 穿戴负担直接换掉了 UMI 的核心卖点**
UMI 的核心论证是「便宜（$371）+ 轻便 → 数据可规模化」。ActiveUMI 是头显 + 背包（电池 + PC）：热、重、晕动、续航受限、单人采集时长受限。论文自报比裸手慢 1.49–2.06×，而 UMI 手持夹爪几乎接近裸手速度。

**在「数据吞吐」这个 UMI 最本质的指标上，ActiveUMI 是净退步**。它换来的是数据维度（+头部 6-DoF）与位姿精度。这是一次明确的 trade-off，不是帕累托改进。

**4. 部署侧要求反而更苛刻**
UMI 策略可部署到任何装了同款夹爪的机械臂（论文演示了 UR5e → Franka FR2 跨机器人迁移，90% 成功率）。ActiveUMI 要求机器人**必须有一个主动头**——本文用了整条第三个 6-DoF 机械臂当脖子，总计 20 DoF。

所以「兼容 stock robot」的说法**只在夹爪层面成立，在头部层面反而更挑平台**。没有可动头的机器人无法执行 ActiveUMI 策略的头部动作维度。不过这一点也可看作前瞻性：humanoid 和带云台的双臂平台正在成为主流形态，ActiveUMI 的观测-动作结构与它们天然对齐（论文在 Introduction 中明确以此为动机）。

**5. 腕部相机的设计负担被卸载**
UMI 为了在腕相机里塞进足够上下文，堆了两个 hack：155° 超广角鱼眼（消融显示降到 69° FoV 会让成功率从 100% 掉到 55%）和**物理侧面镜子**做伪立体深度（去掉掉 15 个点）。ActiveUMI 把「全局上下文」的职责交给头相机后，腕相机回归纯粹的局部细节角色，这些 hack 不再必需。这是架构层面的简化。

### 一句话定位

> **VR 头显本质上是「消费级多点 6-DoF 动捕系统 + 显示器 + 触觉输入设备」四合一。**用它做 UMI，是把 UMI 最脆弱的视觉 SLAM 换成成熟方案，白送头部自由度和操作员反馈通道；代价是系统从无标定、无状态、随手可采，变成有世界坐标系、需 session 级标定、需穿戴负重。

**适用边界**：
- **ActiveUMI 更适合**：固定工作台、长程、双臂协同、需要频繁切换视角/抗遮挡的任务（叠衣、整理收纳、装配、实验室操作）；目标平台是 humanoid 或带主动头的双臂机器人。
- **UMI 更适合**：移动大范围、户外、动态高速（投掷）、多地点海量采集；目标平台是任意标准机械臂。

---

## Related Concepts & Entities

- [[wiki/entities/hardware/UMI]]：直接前身与主要对比基准
- [[wiki/sources/data-collection/2026-04-24 UMI (Chi 2024)]]：原始 UMI 论文
- [[wiki/sources/data-collection/2026-04-24 FastUMI (Zhaxizhuoma 2025)]]：另一条改进路线（硬件解耦 + T265 替换 VIO）
- [[wiki/comparisons/UMI 系列采集接口对比]]：UMI / FastUMI / ActiveUMI 三方对比
- [[wiki/concepts/generative-models/Diffusion Policy]]：UMI 使用的策略主干
- [[wiki/concepts/vla/Vision-Language-Action 模型]]：ActiveUMI 使用 π0 作为策略
- [[wiki/entities/models/π₀.₅]]：π0 系列后续版本
- 相关工作：Vision-in-Action (Xiong 2025) —— 最接近的主动感知工作，但属于**遥操作**系统；DexUMI（灵巧手扩展）；AirExo / DexCap / Dexop / NuEXO（外骨骼与手套路线）
