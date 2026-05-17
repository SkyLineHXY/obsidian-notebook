---
type: analysis
tags: [DDPM, DDIM, 扩散模型, 生成模型, 数学推导, 变分推断, ELBO, 重参数化, ODE, 加速采样]
sources: [raw/sources/papers/Generative Model/Ho 等 - 2020 - Denoising Diffusion Probabilistic Models.pdf]
created: 2026-04-23
updated: 2026-04-23
---

# DDPM & DDIM 完整数学推导

> **定位**：本页面是 [[wiki/concepts/DDPM]] 的零基础完整数学展开，并系统推导其加速采样变体 DDIM（Song et al., 2020）。目标是让你不仅会用公式，还能从最基础的概率论出发，理解每一步推导从哪里来、为什么这样设计。
>
> 关联页面：[[wiki/concepts/DDPM]] · [[wiki/concepts/Diffusion Policy]] · [[wiki/concepts/DPPO]] · [[wiki/analyses/DPPO 完整数学推导]] · [[wiki/analyses/Flow Matching 完整数学推导]]

---

## 0. 符号定义

在阅读任何推导前，先把所有变量的含义固定下来：

| 符号                                          | 含义                                                     |     |
| ------------------------------------------- | ------------------------------------------------------ | --- |
| $x_0 \sim q(x_0)$                           | 真实数据（如一张图片），来自未知的真实数据分布                                |     |
| $x_1, x_2, \ldots, x_T$                     | 前向过程产生的中间隐变量，维度与 $x_0$ 完全相同                            |     |
| $T$                                         | 总扩散步数，DDPM 原文取 $T=1000$                                |     |
| $t \in \{1,\ldots,T\}$                      | 当前时间步索引                                                |     |
| $\beta_t \in (0, 1)$                        | 第 $t$ 步的**噪声调度系数**（Variance Schedule），预设的超参数递增序列       |     |
| $\alpha_t = 1 - \beta_t$                    | 第 $t$ 步的**信号保留比例**                                     |     |
| $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ | 前 $t$ 步的**累积信号保留比例**                                   |     |
| $\epsilon \sim \mathcal{N}(0, I)$           | 标准正态噪声（各维独立，与 $x_0$ 同维度）                               |     |
| $\epsilon_\theta(x_t, t)$                   | 神经网络（U-Net 或 Transformer），以加噪图 $x_t$ 和时间步 $t$ 为输入，预测噪声 |     |
| $\mu_\theta(x_t, t)$                        | 神经网络参数化的逆向均值                                           |     |
| $\sigma_t^2$                                | 逆向过程的方差（可学习或固定）                                        |     |
| $\tilde{\mu}_t(x_t, x_0)$                   | 前向后验 $q(x_{t-1}\mid x_t,x_0)$ 的**真实均值**                |     |
| $\tilde{\beta}_t$                           | 前向后验 $q(x_{t-1}\mid x_t,x_0)$ 的**真实方差**                |     |
| $D_{\mathrm{KL}}(p \| q)$                   | 从分布 $p$ 到分布 $q$ 的 KL 散度                                |     |
| $\mathcal{N}(x;\,\mu,\,\Sigma)$             | 均值 $\mu$、协方差 $\Sigma$ 的多元高斯密度                          |     |

---

## 第一部分：DDPM（去噪扩散概率模型）

### 1. 核心框架：两个过程

DDPM 的思路可以用一句话概括：**先用固定规则把数据加噪成纯噪声（前向），再训练神经网络学会把纯噪声逆向恢复成数据（逆向）**。

```
数据 x_0 ──[前向：逐步加噪]──► 纯噪声 x_T ~ N(0, I)
纯噪声 x_T ──[逆向：逐步去噪]──► 新数据 x̂_0
```

---

### 2. 前向过程（Forward Process）

#### 2.1 逐步加噪的数学定义

前向过程是一条**固定的、不含可学习参数的马尔可夫链**，每步向数据中注入少量高斯噪声：

$$
q(x_{1:T} \mid x_0) \;:=\; \prod_{t=1}^{T} q(x_t \mid x_{t-1})
$$

$$
q(x_t \mid x_{t-1}) \;:=\; \mathcal{N}\!\left(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t\, I\right)
$$

**物理意义**：每步把上一时刻的信号缩小到 $\sqrt{1-\beta_t}$ 倍，再叠加标准差为 $\sqrt{\beta_t}$ 的高斯噪声。

等价地，可以写成重参数化形式：

$$
x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_{t}, \qquad \epsilon_{t} \sim \mathcal{N}(0, I)
$$

随着 $t$ 增大，$\beta_t$ 逐渐增大（线性或余弦调度），信号被逐步湮没，最终 $x_T \approx \mathcal{N}(0, I)$。

---

#### 2.2 关键推导：任意时刻的闭式解

**目标**：不逐步迭代，直接从 $x_0$ 采样任意时刻 $x_t$。

**工具**：高斯分布的**线性变换的可加性**——若 $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ 与 $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ 独立，则 $aX + bY \sim \mathcal{N}(a\mu_1 + b\mu_2,\, a^2\sigma_1^2 + b^2\sigma_2^2)$。

**步骤一**：利用重参数化展开 $x_t$：

$$
x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\, \epsilon_{t-1}, \quad \alpha_t = 1-\beta_t
$$

再展开 $x_{t-1}$：

$$
x_{t-1} = \sqrt{\alpha_{t-1}}\, x_{t-2} + \sqrt{1-\alpha_{t-1}}\, \epsilon_{t-2}
$$

代入得：

$$
x_t = \sqrt{\alpha_t\alpha_{t-1}}\, x_{t-2}
      + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\, \epsilon_{t-2} + \sqrt{1-\alpha_t}\, \epsilon_{t-1}}_{\text{两个独立高斯之和}}
$$

**步骤二**：合并两个独立高斯：

$$
\text{方差} = \alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = 1 - \alpha_t\alpha_{t-1}
$$

因此合并后服从 $\mathcal{N}(0,\; (1-\alpha_t\alpha_{t-1})I)$，即：

$$
x_t = \sqrt{\alpha_t\alpha_{t-1}}\, x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\, \bar{\epsilon}_{t-2}, \quad \bar{\epsilon}_{t-2} \sim \mathcal{N}(0,I)
$$

**步骤三**：一直展开到 $x_0$（数学归纳法），最终得到：

$$
\boxed{
  x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
}
$$

等价地，闭式条件分布为：

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t)\, I\right)
$$

> **直觉**：信号以 $\sqrt{\bar{\alpha}_t}$ 的速率衰减，噪声以 $\sqrt{1-\bar{\alpha}_t}$ 的幅度增长，两者的平方和恒为 1——这是能量守恒的体现。当 $t \to T$ 时，$\bar{\alpha}_t \to 0$，信号完全消失，$x_T \approx \mathcal{N}(0,I)$。

---

### 3. 逆向过程（Reverse Process）

#### 3.1 目标与困难

推理时，我们希望从 $x_T \sim \mathcal{N}(0,I)$ 出发，通过逐步去噪恢复 $x_0$。这需要知道**逆向条件分布**：

$$
q(x_{t-1} \mid x_t)
$$

用贝叶斯定理可以写出：

$$
q(x_{t-1} \mid x_t) = \frac{q(x_t \mid x_{t-1})\, q(x_{t-1})}{q(x_t)}
$$

**困难**：分母 $q(x_t) = \int q(x_t \mid x_0)\, q(x_0)\, dx_0$ 需要对整个数据分布 $q(x_0)$ 积分——这是**不可解析计算的**。

**解决方案**：用神经网络参数化的 $p_\theta(x_{t-1} \mid x_t)$ 来近似它，并通过最大化对数似然（ELBO）来训练。

---

#### 3.2 条件后验的可解析性（关键技巧）

虽然 $q(x_{t-1} \mid x_t)$ 不可解析，但**以 $x_0$ 为条件的后验** $q(x_{t-1} \mid x_t, x_0)$ 却是可解析的高斯分布！

**推导**：利用贝叶斯定理：

$$
q(x_{t-1} \mid x_t, x_0)
= \frac{q(x_t \mid x_{t-1},x_0)\, q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
= \frac{q(x_t \mid x_{t-1})\, q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
$$

（第二个等号利用了马尔可夫性：$x_t$ 给定 $x_{t-1}$ 后与 $x_0$ 条件独立。）

三个分布均为高斯，取对数后对 $x_{t-1}$ 配方。设 $c$ 为与 $x_{t-1}$ 无关的常数：

$$
\log q(x_{t-1} \mid x_t, x_0) = \log q(x_t \mid x_{t-1}) + \log q(x_{t-1} \mid x_0) + c
$$

展开各项（利用 $\alpha_t = 1-\beta_t$，$\bar\alpha_t = \prod \alpha_s$）：

$$
= -\frac{1}{2}\!\left[\frac{(x_t - \sqrt{\alpha_t}\, x_{t-1})^2}{1-\alpha_t} + \frac{(x_{t-1} - \sqrt{\bar\alpha_{t-1}}\, x_0)^2}{1-\bar\alpha_{t-1}}\right] + c
$$

展开并**收集 $x_{t-1}$ 的二次项系数**：

$$
-\frac{1}{2}\!\left[\frac{\alpha_t}{1-\alpha_t} + \frac{1}{1-\bar\alpha_{t-1}}\right] x_{t-1}^2
= -\frac{1}{2}\cdot\frac{\alpha_t(1-\bar\alpha_{t-1}) + (1-\alpha_t)}{(1-\alpha_t)(1-\bar\alpha_{t-1})}\, x_{t-1}^2
$$

化简分子：$\alpha_t - \alpha_t\bar\alpha_{t-1} + 1 - \alpha_t = 1 - \bar\alpha_t$

因此**后验方差**为：

$$
\boxed{
  \tilde\beta_t = \frac{(1-\alpha_t)(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} = \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
}
$$

**收集 $x_{t-1}$ 的一次项系数**（即 $2\cdot[\text{...}]\cdot x_{t-1}$）：

$$
\frac{\sqrt{\alpha_t}}{1-\alpha_t}\, x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\, x_0
$$

**后验均值** $= \tilde\beta_t \times$ 上式，即：

$$
\boxed{
  \tilde\mu_t(x_t, x_0)
  = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\, x_0
  + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x_t
}
$$

因此：

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\; \tilde\mu_t(x_t, x_0),\; \tilde\beta_t\, I\right)
$$

> **直觉**：后验均值是 $x_0$（干净数据）和 $x_t$（当前噪声图）的加权平均——告诉我们"应该往哪里去噪"。

---

### 4. 训练目标：ELBO 推导

#### 4.1 从对数似然到变分下界

**目标**：最大化训练数据的对数似然 $\log p_\theta(x_0)$。

直接计算 $p_\theta(x_0) = \int p_\theta(x_{0:T})\, dx_{1:T}$ 的积分不可解析。使用**变分推断**：

$$
\log p_\theta(x_0)
= \log \int \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\, q(x_{1:T}\mid x_0)\, dx_{1:T}
= \log \mathbb{E}_{q}\!\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]
$$

由 **Jensen 不等式**（$\log$ 是凹函数，$\log \mathbb{E}[\cdot] \geq \mathbb{E}[\log\cdot]$）：

$$
\log p_\theta(x_0)
\;\geq\; \mathbb{E}_{q}\!\left[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]
\;=:\; \text{ELBO}
$$

**展开 ELBO**：

$$
\text{ELBO}
= \mathbb{E}_q\!\left[\log p_\theta(x_{0:T}) - \log q(x_{1:T}\mid x_0)\right]
$$

利用各自的马尔可夫分解：

$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t)
\qquad
q(x_{1:T}\mid x_0) = \prod_{t=1}^T q(x_t\mid x_{t-1})
$$

代入并整理（详见 Ho et al. 2020 附录 A），可将 ELBO 分解为三类项：

$$
\text{ELBO}
= \underbrace{-D_{\mathrm{KL}}\!\left(q(x_T\mid x_0)\,\|\,p(x_T)\right)}_{\mathcal{L}_T}
+ \sum_{t=2}^{T}\underbrace{-D_{\mathrm{KL}}\!\left(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\right)}_{\mathcal{L}_{t-1}}
+ \underbrace{\mathbb{E}_q\!\left[\log p_\theta(x_0\mid x_1)\right]}_{\mathcal{L}_0}
$$

**各项物理意义**：

| 项                                  | 含义                                                      |
| ---------------------------------- | ------------------------------------------------------- |
| $\mathcal{L}_T$                    | $x_T$ 与先验 $\mathcal{N}(0,I)$ 之差（无参数，训练时为常数）             |
| $\mathcal{L}_{t-1}$（$t=2\ldots T$） | 逆向预测 $p_\theta(x_{t-1}\mid x_t)$ 与前向真实后验的 KL 散度（主要训练目标） |
| $\mathcal{L}_0$                    | 最后一步重建损失                                                |

---

#### 4.2 化简 $\mathcal{L}_{t-1}$：两个高斯的 KL 散度

设 $p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t,t),\, \sigma_t^2 I)$（方差固定）。

两个同方差高斯之间的 KL 散度为：

$$
D_{\mathrm{KL}}\!\left(\mathcal{N}(\mu_1,\sigma^2 I)\,\|\,\mathcal{N}(\mu_2,\sigma^2 I)\right)
= \frac{\|\mu_1 - \mu_2\|^2}{2\sigma^2}
$$

因此：

$$
\mathcal{L}_{t-1}
= -\mathbb{E}_{x_0,\epsilon}\!\left[\frac{\left\|\tilde\mu_t(x_t,x_0) - \mu_\theta(x_t,t)\right\|^2}{2\sigma_t^2}\right] + c
$$

**核心结论**：训练目标等价于让神经网络预测的均值 $\mu_\theta$ 匹配真实后验均值 $\tilde\mu_t$。

---

#### 4.3 ε-预测参数化（关键技巧）

直接预测均值在数值上不稳定。注意到 $\tilde\mu_t$ 可以用**闭式解中的噪声** $\epsilon$ 来表达：

将 $x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}$ 代入 $\tilde\mu_t$ 的表达式：

**第一步**：代入 $x_0$：

$$
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t} \cdot \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}
            + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x_t
$$

**第二步**：合并 $x_t$ 系数（记 $\beta_t = 1-\alpha_t$）：

$$
\text{$x_t$ 系数} = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}} + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
= \frac{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}{(1-\bar\alpha_t)\sqrt{\alpha_t}}
= \frac{1-\bar\alpha_t}{(1-\bar\alpha_t)\sqrt{\alpha_t}}
= \frac{1}{\sqrt{\alpha_t}}
$$

**第三步**：得到 $\epsilon$ 系数：

$$
-\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}} \cdot \sqrt{1-\bar\alpha_t}
= -\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar\alpha_t)} \cdot \sqrt{1-\bar\alpha_t}
= -\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}
$$

因此真实后验均值为：

$$
\boxed{
  \tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right)
}
$$

**这给出了神经网络的参数化方式**：令

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\right)
$$

则 $\mathcal{L}_{t-1}$ 化简为：

$$
\mathcal{L}_{t-1} \propto \mathbb{E}_{x_0,\epsilon}\!\left[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t(1-\bar\alpha_t)}\,\left\|\epsilon - \epsilon_\theta\!\left(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\right)\right\|^2\right]
$$

> **等价关系**：这正是**去噪分数匹配（Denoising Score Matching，DSM）**目标——网络学会预测噪声，等价于估计对数密度的梯度（Score Function）$\nabla_{x_t}\log q(x_t)$。

---

#### 4.4 简化训练目标 $\mathcal{L}_\text{simple}$

Ho et al. 2020 发现，丢弃时间步依赖的加权系数，直接用等权 MSE，实验效果更好：

$$
\boxed{
  \mathcal{L}_\text{simple}(\theta)
  := \mathbb{E}_{t \sim \mathcal{U}[1,T],\; x_0 \sim q,\; \epsilon \sim \mathcal{N}(0,I)}\!\left[\left\|\epsilon - \epsilon_\theta\!\left(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\right)\right\|^2\right]
}
$$

**为什么 $\mathcal{L}_\text{simple}$ 更好？**

原始加权系数 $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$ 在小 $t$（接近原始数据）时极大，使网络把大量容量浪费在"几乎无噪声"的简单任务上。$\mathcal{L}_\text{simple}$ 等权处理所有时间步，让网络更专注于大噪声的困难情况，从而产生更高质量的样本。

---

### 5. DDPM 采样算法

**训练完成后**，从 $x_T \sim \mathcal{N}(0,I)$ 出发，执行以下循环：

$$
\boxed{
  x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\right) + \underbrace{\sigma_t\, z}_{\text{随机扰动}}, \quad z \sim \mathcal{N}(0,I)
}
$$

其中 $\sigma_t$ 通常取 $\sqrt{\beta_t}$ 或 $\sqrt{\tilde\beta_t}$，对应不同的方差假设。

**算法流程**：

```
输入：训练好的 ε_θ
x_T ~ N(0, I)

for t = T, T-1, ..., 1:
    z ~ N(0, I)  if t > 1  else  z = 0
    x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z

return x_0
```

**缺点**：需要完整的 $T=1000$ 步，每步都要调用神经网络，推理极慢。

---

### 6. DDPM 与 Score Matching 的等价关系（补充）

郎之万动力学（Langevin Dynamics）的迭代更新为：

$$
x_{t-1} = x_t + \frac{\delta}{2}\nabla_{x_t}\log p(x_t) + \sqrt{\delta}\, z
$$

对比 DDPM 采样步骤，令 $\frac{\delta}{2} \approx \frac{\beta_t}{1-\bar\alpha_t}$（退火），则 $\epsilon_\theta(x_t,t) \approx -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log q(x_t)$。

这表明：**预测噪声的网络 $\epsilon_\theta$，本质上是在估计数据分布的 Score Function**，两者在数学上等价。

---

## 第二部分：DDIM（去噪扩散隐式模型）

### 7. 动机：DDPM 为何慢？

DDPM 的采样需要 $T=1000$ 步，每步调用一次神经网络。在生成一张 $256\times256$ 的图片时，这意味着 $\sim$ 1000 次前向传播，极其耗时。

**能否跳步？** 直接从 $x_T$ 跳到 $x_{T/2}$ 再到 $x_0$？DDPM 是马尔可夫链，跳步会破坏分布一致性。

**DDIM 的洞察**（Song et al., ICLR 2021）：

> DDPM 的训练目标 $\mathcal{L}_\text{simple}$ 只约束了**边缘分布** $q(x_t \mid x_0)$，并未约束**联合分布** $q(x_{1:T} \mid x_0)$。因此，可以构造一族不同的联合分布，只要它们与 DDPM 有**相同的边缘分布**，就能复用同一个训练好的 $\epsilon_\theta$，同时赋予采样过程全新的性质。

---

### 8. DDIM 的非马尔可夫前向过程

DDIM 定义了一个**广义的、以 $x_0$ 为条件的后验分布**，带有可调节的随机性参数 $\sigma_t \geq 0$：

$$
\boxed{
  q_\sigma(x_{t-1} \mid x_t, x_0)
  = \mathcal{N}\!\left(x_{t-1};\;
    \underbrace{\sqrt{\bar\alpha_{t-1}}\, x_0}_{\text{去噪方向}}
    + \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\frac{x_t - \sqrt{\bar\alpha_t}\, x_0}{\sqrt{1-\bar\alpha_t}}}_{\text{指向 }x_t\text{ 的方向}},\;
    \sigma_t^2 I\right)
}
$$

#### 8.1 证明边缘分布一致性

**命题**：若 $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I)$，则对任意 $\sigma_t$，

$$
q_\sigma(x_{t-1} \mid x_0) := \int q_\sigma(x_{t-1}\mid x_t, x_0)\, q(x_t \mid x_0)\, dx_t = \mathcal{N}\!\left(\sqrt{\bar\alpha_{t-1}}\,x_0,\,(1-\bar\alpha_{t-1})I\right)
$$

**证明**（利用高斯线性变换）：

设 $B = \frac{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}}{\sqrt{1-\bar\alpha_t}}$，后验均值可以写成关于 $x_t$ 的线性函数：

$$
x_{t-1} = \underbrace{\left(\sqrt{\bar\alpha_{t-1}} - B\sqrt{\bar\alpha_t}\right)}_A x_0 + B\, x_t + \sigma_t\, z
$$

其中 $z \sim \mathcal{N}(0,I)$。

**计算边缘均值**：

$$
\mathbb{E}[x_{t-1} \mid x_0]
= A\, x_0 + B\, \mathbb{E}[x_t \mid x_0]
= A\, x_0 + B\sqrt{\bar\alpha_t}\, x_0
= \left(A + B\sqrt{\bar\alpha_t}\right) x_0
= \sqrt{\bar\alpha_{t-1}}\, x_0 \;\checkmark
$$

**计算边缘方差**：

$$
\mathrm{Var}[x_{t-1} \mid x_0]
= B^2\,\mathrm{Var}[x_t \mid x_0] + \sigma_t^2
= \frac{1-\bar\alpha_{t-1}-\sigma_t^2}{1-\bar\alpha_t}\cdot(1-\bar\alpha_t) + \sigma_t^2
= 1-\bar\alpha_{t-1}-\sigma_t^2 + \sigma_t^2
= 1-\bar\alpha_{t-1} \;\checkmark
$$

**结论**：无论 $\sigma_t$ 取何值，DDIM 的边缘分布与 DDPM **完全相同**，因此可以直接使用 DDPM 训练好的 $\epsilon_\theta$。

---

### 9. DDIM 采样公式推导

在推理时，$x_0$ 未知，用神经网络**预测的 $x_0$** 来替代：

$$
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
$$

**代入 $q_\sigma(x_{t-1} \mid x_t, x_0)$ 的均值公式**：

注意到 $x_t - \sqrt{\bar\alpha_t}\,\hat{x}_0 = \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)$，因此：

$$
\frac{x_t - \sqrt{\bar\alpha_t}\,\hat{x}_0}{\sqrt{1-\bar\alpha_t}} = \epsilon_\theta(x_t,t)
$$

代入后，DDIM 的采样更新规则为：

$$
\boxed{
  x_{t-1}
  = \underbrace{\sqrt{\bar\alpha_{t-1}}\,\hat{x}_0}_{\text{①去噪分量}}
  + \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\epsilon_\theta(x_t,t)}_{\text{②指向 }x_t\text{ 的分量}}
  + \underbrace{\sigma_t\,\epsilon_t}_{\text{③随机扰动}}, \quad \epsilon_t \sim \mathcal{N}(0,I)
}
$$

将 $\hat{x}_0$ 展开，完整形式为：

$$
x_{t-1}
= \sqrt{\bar\alpha_{t-1}}\cdot\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
+ \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\epsilon_\theta(x_t,t)
+ \sigma_t\,\epsilon_t
$$

---

### 10. 两个极端情况

#### 10.1 $\sigma_t = \tilde\beta_t$：退化为 DDPM

当 $\sigma_t = \sqrt{\frac{(1-\bar\alpha_{t-1})\beta_t}{1-\bar\alpha_t}}$（即前向后验的真实方差），DDIM 完全等价于 DDPM 的随机采样。

#### 10.2 $\sigma_t = 0$：确定性 DDIM 采样

当所有时间步均取 $\sigma_t = 0$，随机项消失，采样变为**完全确定性的 ODE 轨迹**：

$$
\boxed{
  x_{t-1}
  = \sqrt{\bar\alpha_{t-1}}\cdot\underbrace{\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}}_{\hat{x}_0}
  + \sqrt{1-\bar\alpha_{t-1}}\cdot\epsilon_\theta(x_t,t)
}
$$

**性质**：给定同一个初始噪声 $x_T$，每次采样产生**完全相同的** $x_0$。这使得 $x_T$ 空间成为有意义的隐空间。

---

### 11. 与 Neural ODE 的联系

将确定性 DDIM 采样步骤写成连续极限（时间步 $t \to t - dt$）：

$$
dx = \left[\frac{x}{2} - \frac{1}{2}\,\frac{\epsilon_\theta(x,t)}{\sqrt{1-\bar\alpha_t}}\right] d\log\bar\alpha_t
$$

这是一个**常微分ODE）**！在时间轴上对这个 ODE 数值积分，等价于运行 DDIM 采样器：
方程（

| 概念 | DDIM 中的对应物 |
|---|---|
| 状态 $x(t)$ | 采样轨迹 $x_t$ |
| 向量场 $f(x,t)$ | $\frac{x}{2} - \frac{\epsilon_\theta}{\sqrt{2(1-\bar\alpha_t)}}$ |
| ODE 积分 | DDIM 的逐步更新 |

**加速采样的原理**：ODE 可以用高阶数值积分方法（如 Heun 方法、PLMS 方法）以**更大的步长**高精度积分，从而大幅减少函数评估次数。这就是为何 DDIM 可以从 1000 步缩减到 50 步甚至 10 步。

---

### 12. 加速采样：子序列采样

**DDIM 的核心加速技巧**：将时间步从完整序列 $\{1,\ldots,T\}$ 替换为一个**稀疏子序列** $\{\tau_1, \tau_2, \ldots, \tau_S\} \subset \{1,\ldots,T\}$，其中 $S \ll T$（如 $S=50$）。

只需将上述采样公式中的相邻时间步从 $(t, t-1)$ 替换为 $(\tau_i, \tau_{i-1})$，公式结构不变：

$$
x_{\tau_{i-1}}
= \sqrt{\bar\alpha_{\tau_{i-1}}}\cdot\hat{x}_0
+ \sqrt{1-\bar\alpha_{\tau_{i-1}}-\sigma_{\tau_i}^2}\cdot\epsilon_\theta(x_{\tau_i}, \tau_i)
+ \sigma_{\tau_i}\,\epsilon
$$

**为什么 DDPM 不能跳步而 DDIM 可以？**

| | DDPM | DDIM |
|---|---|---|
| 过程类型 | 马尔可夫链，$x_{t-1}$ 只依赖 $x_t$ | 非马尔可夫，$x_{t-1}$ 依赖 $x_t$ 和 $\hat{x}_0$ |
| 跳步后的分布 | 不等于正确的边缘分布 | 通过调整 $\bar\alpha$ 的映射，分布保持一致 |
| 必要步数 | $\sim 1000$ | $10 \sim 50$（效果接近） |

---

### 13. 隐空间结构与插值

由于 $\sigma_t = 0$ 时过程完全确定，存在**双射**：

$$
x_T \xrightarrow{\text{DDIM 采样（确定性）}} x_0
\qquad
x_0 \xrightarrow{\text{DDIM 逆向（deterministic inversion）}} x_T
$$

这使得 $x_T$ 空间具有语义结构，支持线性插值：

$$
x_T^{(\lambda)} = \text{slerp}\!\left(x_T^{(1)},\, x_T^{(2)},\, \lambda\right), \quad \lambda \in [0,1]
$$

（球面线性插值，$\text{slerp}$，比普通线性插值更合适，因为 $x_T$ 分布于高维球面附近。）

---

## 第三部分：综合对比

### 14. DDPM vs DDIM 全维度对比

| 维度             | DDPM                                    | DDIM ($\sigma_t=0$)                     |
| -------------- | --------------------------------------- | --------------------------------------- |
| **本质**         | 随机马尔可夫链                                 | 确定性 ODE 积分                              |
| **训练**         | $\mathcal{L}_\text{simple}$，与 DDIM 完全相同 | 复用 DDPM 的 $\epsilon_\theta$，无需额外训练      |
| **采样步数**       | $T=1000$，不可减少                           | $S=10\sim50$，可自由设置                      |
| **给定相同 $x_T$** | 每次结果不同（随机）                              | 每次结果相同（确定性）                             |
| **隐空间**        | $x_T$ 无意义，不可插值                          | $x_T$ 对应唯一的 $x_0$，可插值                   |
| **DDPM 退化**    | —                                       | 令 $\sigma_t = \tilde\beta_t$ 即可退化为 DDPM |
| **理论基础**       | 变分推断 + ELBO                             | ODE 连续极限 + 数值积分                         |

### 15. 参数 $\sigma_t$ 的作用

$\sigma_t$ 控制采样过程中随机性的大小，可在 DDIM 和 DDPM 之间平滑插值：

$$
\sigma_t = \eta \cdot \sqrt{\frac{(1-\bar\alpha_{t-1})\beta_t}{1-\bar\alpha_t}}, \quad \eta \in [0, 1]
$$

- $\eta = 0$：完全确定性（DDIM）
- $\eta = 1$：等价于 DDPM 后验方差（接近 DDPM）

---

## 第四部分：算法总结

### 16. DDPM 训练算法

```
训练阶段：
repeat:
  1. x_0 ~ q(x_0)          ← 从真实数据中采样
  2. t ~ Uniform[1, T]      ← 随机采样时间步
  3. ε ~ N(0, I)            ← 采样噪声
  4. x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε   ← 前向过程闭式解
  5. 计算损失 L = ‖ε - ε_θ(x_t, t)‖²
  6. 梯度下降更新 θ
until 收敛
```

### 17. DDPM 采样算法

```
采样阶段：
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
  z ~ N(0, I)  if t > 1  else  z = 0
  x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t)) + √β_t * z
return x_0
```

### 18. DDIM 采样算法（加速版）

```
采样阶段（给定子序列 τ₁ < τ₂ < ... < τ_S = T，step 数 S << T）：
x_{τ_S} ~ N(0, I)
for i = S, S-1, ..., 1:
  ε_pred = ε_θ(x_{τ_i}, τ_i)             ← 一次神经网络调用
  x̂_0 = (x_{τ_i} - √(1-ᾱ_{τ_i}) * ε_pred) / √ᾱ_{τ_i}
  x_{τ_{i-1}} = √ᾱ_{τ_{i-1}} * x̂_0
               + √(1-ᾱ_{τ_{i-1}} - σ_{τ_i}²) * ε_pred
               + σ_{τ_i} * z       (z ~ N(0,I)，若 σ=0 则省略)
return x_0
```

---

## 延伸阅读

| 主题 | 相关页面 |
|---|---|
| 将 DDPM 用于机器人动作生成 | [[wiki/concepts/Diffusion Policy]] |
| 用 PPO 在线微调扩散策略 | [[wiki/concepts/DPPO]] · [[wiki/analyses/DPPO 完整数学推导]] |
| DDPM 的替代路线（ODE 流） | [[wiki/concepts/Flow Matching]] · [[wiki/analyses/Flow Matching 完整数学推导]] |
| DDPM 概念总览 | [[wiki/concepts/DDPM]] |
