# SA-VLA: Spatially-Aware Flow-Matching for Vision-Language-Action Reinforcement Learning

Preprint, compiled February 3, 2026

Xu Pan ID 1,2, Zhenglin Wan3, Xingrui Yu ID 2∗, Xianwei Zheng ID 1, Youkai Ke1, Ming Sun4, Rui Wang4, Ziwei Wang ID 5, and Ivor Tsang ID 2

1State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University, Wuhan, P.R. China   
2Centre for Frontier AI Research (CFAR), Institute of High Performance Computing (IHPC), Agency for Science, Technology and Research (A\*STAR), Singapore   
3Department of Computer Science, National University of Singapore, Singapore   
4Institute of Information Engineering, Chinese Academy of Sciences, Beijing, P.R. China   
5School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore

# Abstract

Vision-Language-Action (VLA) models exhibit strong generalization in robotic manipulation, yet reinforcement learning (RL) fine-tuning often degrades robustness under spatial distribution shifts. For flow-matching VLA policies, this degradation is closely associated with the erosion of spatial inductive bias during RL adaptation, as sparse rewards and spatially agnostic exploration increasingly favor short-horizon visual cues. To address this issue, we propose SA-VLA, a spatially-aware RL adaptation framework that preserves spatial grounding during policy optimization by aligning representation learning, reward design, and exploration with task geometry. SA-VLA fuses implicit spatial representations with visual tokens, provides dense rewards that reflect geometric progress, and employs SCAN, a spatially-conditioned annealed exploration strategy tailored to flow-matching dynamics. Across challenging multi-object and cluttered manipulation benchmarks, SA-VLA enables stable RL fine-tuning and improves zero-shot spatial generalization, yielding more robust and transferable behaviors. Code and project page are available at https://xupan.top/Projects/savla

# 1 Introduction

Enabling robots to perform complex manipulation tasks in the physical world remains a central challenge in embodied artificial intelligence [1]. Robotic manipulation requires not only perceptual recognition but also precise reasoning over object geometry, spatial relations, and contact dynamics to generate reliable actions [2]. Recent Vision-Language-Action (VLA) models demonstrate that large multimodal policies can generalize across diverse tasks by jointly modeling visual observations, language instructions, and action generation [3, 4]. In particular, diffusion and flow-matching policies have emerged as a powerful paradigm for continuous, high-dimensional control, enabling multimodal action generation with fine-grained dexterity [5]. Prominent flow-based VLAs such as 0 further highlight the πpotential of foundation models for robotic manipulation under long-horizon and language-conditioned settings [6].

Despite this progress, adapting pretrained VLA policies via reinforcement learning (RL) remains fragile. Although RL finetuning [7] can improve in-distribution performance, it often undermines robustness under spatial distribution shifts, such as large viewpoint changes or increased environmental clutter [8]. As shown in Fig. 1, naive RL adaptation can induce phase-inconsistent spatial behavior within a single execution, causing the policy to miss target objects or placement regions under shifted observations. These failures arise not solely from the representational limits of visual tokens. High-variance updates during on-policy RL can overwrite pretrained geometric regularities. Meanwhile, sparse or weakly shaped rewards encourage overfitting to spurious reward–action correlations along limited trajectories, reinforcing short-horizon visual cues that fail to generalize across viewpoints or layouts[9, 10]. While spatially agnostic exploration and ambiguous credit assignment are common challenges in RL-based adaptation, their impact is particularly pronounced for flow-matching VLA policies [11]. Due to the continuous-time, noise-driven formulation of flow matching, spatial behaviors are maintained largely through implicit geometric priors, which can be gradually eroded under unstable credit assignment, leading to spatial overfitting and reduced geometric consistency under distribution shifts [12].

![](images/24c44719a2cf904b7257c5e5a517086c74bec04d81e503713055a0e55441c742.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Decisions"] --> B["Drastic Spatial Distribution Drift"]
    B --> C["Naive RL fine-tuning"]
    B --> D["SA-VLA"]
    C --> E["Spatial Perception"]
    C --> F["Dense Rewarding"]
    C --> G["SCAN Exploration"]
    D --> H["PRESERVING SPATIAL INDUCTIVE"]
    I["Spacely Grounded ADAPTATION"] --> J["Robust under Viewpoint and Clutter"]
    K["Spatial OVERFITTING"] --> L["Loss of Geometric Consistency"]
    M[" rewards 0.000 "] --> N[" terminations False "]
    M --> O[" task pick up the black bowl between the plate and the namekin and place it on the plate view 0.0190.0.0 initstate 49.1 "]
    N --> O
```
</details>

Figure 1: Illustration of spatial inductive bias collapse during naive RL fine-tuning (left) and preserved spatial grounding with SA-VLA (right) under the same task and identical spatial perturbations. For each method, end-effector poses from three temporal phases of a single execution trajectory are rendered as semi-transparent red, green, and blue masks and overlaid to visualize how spatial behavior evolves over time.

To address these challenges, we propose SA-VLA, a spatiallyaware framework for RL adaptation of flow-matching Vision-Language-Action policies (Fig. 1). SA-VLA stabilizes geometry-aware decision making during RL by injecting spatial structure into state representations, reward signals, and exploration noise. Here, implicit spatial representations refer to geometric cues derived from 2D visual tokens that support 3D spatial reasoning without requiring 3D reconstruction. Steplevel dense rewards evaluate geometric progress at each interaction, mitigating shortcut learning and improving credit assignment. Exploration is guided by SCAN (Spatially-Conditioned Annealed Noise), which modulates action-space noise based on spatial representations while retaining an annealed isotropic diffusion as the underlying reference process. By combining these elements, SA-VLA reduces catastrophic forgetting and reward-correlation overfitting, ensuring that RL fine-tuning preserves both semantic and spatial consistency under distribution shifts.

The main contributions of this work are as follows:

• We introduce SA-VLA, a spatially-aware RL adaptation framework that integrates implicit spatial representations with 2D visual tokens and step-level dense rewards for robust flow-matching VLA policies.   
• We propose SCAN, a spatially-conditioned exploration strategy that injects geometric understanding into noisedriven policy exploration, improving credit assignment and preserving spatial consistency during RL finetuning.   
• We demonstrate that aligning spatial representations, step-level rewards, and spatially-conditioned exploration yields stable RL fine-tuning and improved generalization under spatial distribution shifts.

# 2 Related Works

Vision-Language-Action and Spatial Representations. Vision-Language-Action (VLA) models aim to unify visual perception, language instruction, and action generation within a single framework [13, 14]. By leveraging large-scale visionlanguage pretraining and robot demonstrations, these models perform long-horizon, language-conditioned manipulation more generally than task-specific policies [15, 16]. Most VLA methods use 2D appearance cues or shallow depth priors and do not explicitly model multi-view or implicit 3D geometry, limiting geometric continuity and spatial reasoning [17, 14, 18]. Recent efforts such as BridgeVLA [19] demonstrate strategies for incorporating 3D structure into VLM-based manipulation learning, highlighting the importance of spatial priors in efficient and generalizable policies. StereoVLA [20] further exploits binocular geometric cues to improve spatial perception and robustness under viewpoint variations. A recent survey provides a broader landscape of VLA models and their integration of spatial information. [21]

Explicit 3D representations such as point clouds, voxels, or meshes enable metric reasoning but suffer from discretization artifacts and poor scalability in dynamic or cluttered environments [22, 23, 24, 25]. Recent transformer-based backbones such as VGGT [26] infer high-fidelity implicit spatial tokens directly from multi-view imagery, capturing depth continuity, occlusion, and object topology. These implicit spatial representations provide a strong spatial inductive bias beyond 2D visual tokens, enhancing the policy’s ability to reason about spatial relationships, object geometry, and contact dynamics in manipulation tasks.

Multimodal Alignment and Flow-Based Policy Learning. Large vision-language models (VLMs) excel at grounding visual perception to semantic concepts through multimodal pretraining [27, 28, 29], yet their embeddings remain largely 2D, lacking consistent spatial structure. Efforts incorporating multiview or depth cues [18, 30] often rely on shallow projections, failing to preserve geometric continuity across viewpoints.

Reinforcement and imitation learning have advanced continuous control and dexterous manipulation [31, 32], but policies often overfit to frequent or easy states due to sparse task-level rewards. Intrinsic motivation, self-supervised objectives [33, 34], and curriculum or difficulty-aware strategies [35, 36] partially mitigate this issue, yet most still neglect geometric priors or local spatial difficulty. Flow-matching-based policy optimization [37, 38, 39] improves stability and gradient consistency by matching the gradient flow of optimal trajectories, and recent benchmarks show competitive or superior performance relative to diffusion or RL baselines across manipulation tasks [40]. Methods such as Rein-Flow further integrate flow matching with online reinforcement learning for stable fine-tuning of continuous control policies. Existing methods, however, operate at the trajectory level, are reward-driven, and do not explicitly model step-level progress or leverage implicit spatial representations. These limitations motivate our approach, which integrates implicit spatial representations, step-level dense rewards, and spatially-conditioned exploration to robustly learn manipulation policies in cluttered and partially observable environments.

# 3 Method

We propose SA-VLA, a spatially-aware adaptation framework that fuses implicit spatial representations with visual tokens and stabilizes RL via step-level dense supervision and spatiallyconditioned exploration (Fig. 2). Below, we formalize the problem and describe its three core components: spatial token fusion, step-level dense rewards, and SCAN exploration.

# 3.1 Problem Definition

We consider adapting pretrained VLA policies to cluttered environments with multiple objects, frequent occlusions, and complex spatial interactions. At timestep t, the agent observes $s _ { t } ~ = ~ ( x _ { t } , o _ { t } )$ , where $x _ { t }$ is visual input and $o _ { t }$ includes proprioception or task descriptors. Actions $a _ { t }$ are sampled from a policy $\pi _ { \theta } ( a _ { t } \mid s _ { t } )$ .

![](images/49a2f8d266f65a93be1cbe383064124a05b363b53d46e1a4c0f8cd8def1f6e32.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Extract Implicit Representation"] --> B["Spatial Token"]
    B --> C["Visual Token"]
    C --> D["View IDs"]
    D --> E["Fuser Visual × Spatial"]
    E --> F["Rollout"]
    F --> G["Reach"]
    F --> H["Place"]
    F --> I["Leave"]
    F --> J["Dense Rewards"]
    J --> K["Actions"]
    K --> L["VO"]
    L --> M["Actor"]
    M --> N["NOISE Injection"]
    N --> O["SCAN Spatially Conditioned Exploration"]
    O --> P["Exploration"]
    P --> Q["Action"]
    Q --> R["VLM"]
    R --> S["With Other Embs"]
    S --> T["Visual Token"]
    T --> U["Visual Token"]
    U --> V["Extract Implicit Representation"]
```
</details>

Figure 2: Overview of SA-VLA. Visual and spatial tokens are fused into geometry-aware embeddings, which are optimized via step-level dense rewards and spatially-conditioned exploration (SCAN) for robust RL adaptation.

Expert demonstrations $\mathcal { D } = \{ ( x _ { t } , o _ { t } , a _ { t } ) \} _ { t = } ^ { N }$ 1 may be available, but , , adaptation relies on RL to maximize expected return:

$$
J (\theta) = \mathbb {E} _ {\pi_ {\theta}} \Big [ \sum_ {t = 1} ^ {T} r (s _ {t}, a _ {t}) \Big ], \tag {1}
$$

where $r ( s _ { t } , a _ { t } )$ may provide step-level spatial supervision. Naive ,fine-tuning often degrades pretrained geometric priors, causing brittle behavior under spatial shifts.

# 3.2 Method Overview

SA-VLA grounds policy updates in geometry. Visual tokens xt are augmented with implicit spatial tokens $\mathbf { z } _ { t } ,$ encoding object layout and relative geometry, and fused via

$$
\mathbf {h} _ {t} = f _ {\phi} (\mathbf {x} _ {t}, \mathbf {z} _ {t}), \tag {2}
$$

which conditions the policy $\pi _ { \theta } ( a _ { t } \mid \mathbf { h } _ { t } )$ .

Policy optimization uses step-level dense rewards $r _ { t } \colon$

$$
\mathcal {L} _ {\mathrm{RL}} (\theta) = - \mathbb {E} _ {\pi_ {\theta}} \Big [ \sum_ {t} r _ {t} \Big ], \tag {3}
$$

and SCAN exploration injects spatially-aware noise $\epsilon _ { t } ^ { \operatorname { S C A N } }$ , producing actions $a _ { t } = \pi _ { \theta } ( \mathbf h _ { t } ) + \epsilon _ { t } ^ { \mathrm { S C A N } }$ .

The following sections detail spatial token fusion, dense rewards, and SCAN-based exploration.

# 3.3 Spatial Token Fusion

Figure 3 illustrates the spatial token fusion module. To achieve robust manipulation in spatially complex environments, state representations must capture both fine-grained visual details and underlying 3D structure. Visual tokens $\mathbf { x } _ { t } \in \mathbb { R } ^ { L \times C }$ encode local semantics but are sensitive to occlusion, lighting, and viewpoint changes. Complementary geometric cues are provided by implicit spatial tokens $\mathbf { z } _ { t } \in \mathbb { R } ^ { \check { L } \times \check { C } _ { s } }$ derived from multi-view features [26], which capture scene layout and coarse spatial structure. SA-VLA fuses these modalities into unified geometryaware embeddings for downstream policy conditioning.

![](images/e11d462f10dfc58ed9630dca438e5399f3680218647d3a04462e466e0a77a3e5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["z(B,L,Cv)"] --> B["w"]
    B --> C["⊕"]
    C --> D["LN"]
    D --> E["ŷ(B,L,C)"]
    F["x(B,L,C)"] --> G["q"]
    F --> H["k"]
    F --> I["v"]
    G --> J["Cross Attention"]
    H --> J
    I --> J
    J --> K["LN"]
    K --> L["⊕"]
    L --> M["LN"]
    M --> N["MLP"]
    N --> O["⊕"]
    O --> P["h"]
    Q["PE p2D"] --> C
    R["VE pview"] --> D
    S["g"] --> K
```
</details>

Figure 3: Spatial Token Fusion. Visual semantic tokens x attend to spatial tokens z augmented with positional and view embeddings via unidirectional cross-attention. A learnable channel-wise gate g modulates spatial contributions, followed by a residual MLP to produce fused embeddings for flow-matching policies.

Spatial tokens are projected into the visual embedding space and augmented with positional and view encodings:

$$
\tilde {\mathbf {z}} _ {t} = \text { LayerNorm } (\mathbf {z} _ {t} \mathbf {W} _ {\text { proj }} + \mathbf {p} _ {2 \mathrm{D}} + \mathbf {p} _ {\text { view }}), \tag {4}
$$

where ${ \bf p } _ { \mathrm { 2 D } }$ and $\mathbf { p } _ { \mathrm { v i e w } }$ encode spatial layout and camera biases.

Visual tokens attend to spatial tokens via unidirectional crossattention:

$$
\mathbf {a} _ {t} = \text { CrossAttn } (\mathbf {x} _ {t}, \tilde {\mathbf {z}} _ {t}, \tilde {\mathbf {z}} _ {t}), \tag {5}
$$

, , ,followed by a learnable channel-wise gate g:

$$
\mathbf {h} _ {t} = \mathbf {x} _ {t} + \tanh (\mathbf {g}) \odot \text { LayerNorm } (\mathbf {a} _ {t}), \tag {6}
$$

which stabilizes early RL updates while allowing geometric cues to propagate in occluded regions. A residual MLP further refines $\mathbf { h } _ { t } \mathbf { \bar { \Omega } } \mathbf { h } _ { t }  \mathbf { h } _ { t } + \mathbf { M } \mathbf { L } \mathbf { P } ( \mathbf { I }$ LayerNorm(ht)), preserving pretrained features and producing geometry-aware embeddings.

The resulting ht integrates semantic, positional, and multi-view spatial information, forming a robust input for action selection. By combining cross-modal attention, adaptive gating, and residual refinement, the fuser mitigates limitations of 2D-only or spatial-only representations. These fused embeddings serve as a foundation for the dense reward design described in the next section.

# 3.4 Spatially-Aware Step-Level Dense Reward

![](images/0e666e3b1c3c471080599eefdcfac7dd287d4a552f4317fa52c6b17ded59ba1c.jpg)

<details>
<summary>text_image</summary>

rewards 0.000
terminations False
task pick up the black
bowl between the
plates and the ramwick
and place it on me
plates with 1200.00
initiate 45T
</details>

![](images/fbb09ed4839a1c36ee7147d3753a386db234aff5cb3f07aa126aa2b02f4318ee.jpg)

<details>
<summary>text_image</summary>

rewards 0.001
terminators false
task pickup the black
bowl between them
plants and frame run
and place it on their
plants w/0 100.00
initiate run
</details>

![](images/68493a6c6e2c0e7d429cbab75f2dc1e81b8b5ff813c8dd20bb3ef6f9a1f2c986.jpg)

<details>
<summary>text_image</summary>

rewards 0.000
terminations True
task pick a game black
bowl between the
plate and the start view
and position this
plate view 0.100.0
initstate 491
</details>

Figure 4: Phase-consistent geometric progress used for steplevel dense rewards, decomposing manipulation into Reach, Place, and Leave phases.

Effective RL in spatially complex manipulation requires feedback that is both dense and geometrically meaningful. Sparse, episode-level rewards fail to convey intermediate progress or encode spatial relations critical for robust generalization. To address this, SA-VLA introduces a step-level dense reward that provides phase-consistent geometric feedback at each interaction step, explicitly aligned with task structure and manipulation dynamics (Fig. 4).

We decompose a manipulation episode into three semantically distinct phases: Reach, Place, and Leave. These phases correspond to generic geometric objectives that naturally arise in object-centric manipulation: approaching the target object, transporting it toward the destination, and disengaging the endeffector after successful placement. Unlike manually scheduled trajectory segmentation, this structure enables dense supervision while preserving temporal flexibility during policy execution.

At each step t, we compute two normalized geometric distances. Let $\mathbf { p } _ { \mathrm { e e f } } , \mathbf { p } _ { \mathrm { o b j } } ,$ and $\mathbf { p } _ { \mathrm { d e s t } }$ denote the end-effector, target object, and destination positions, respectively. We define:

$$
d _ {r o} ^ {(t)} = \frac {\left\| \mathbf {p} _ {\text { eef }} ^ {(t)} - \mathbf {p} _ {\text { obj }} ^ {(t)} \right\|}{d _ {r o} ^ {\text { ref }}}, \quad d _ {o d} ^ {(t)} = \frac {\left\| \mathbf {p} _ {\text { obj }} ^ {(t)} - \mathbf {p} _ {\text { dest }} ^ {(t)} \right\|}{d _ {o d} ^ {\text { ref }}}, \tag {7}
$$

where $d _ { r o } ^ { \mathrm { r e f } }$ and $d _ { o d } ^ { \mathrm { r e f } }$ od are reference distances measured at the first valid interaction step and used to normalize subsequent measurements. Distances are clipped to [0 1] to ensure numerical stability.

The dense reward is formulated as a phase-specific temporal progress signal, computed from the signed change of the relevant normalized distance:

$$
r _ {t} ^ {\text { Reach }} = \lambda (d _ {r o} ^ {(t - 1)} - d _ {r o} ^ {(t)}), \tag {8}
$$

$$
r _ {t} ^ {\text { Place }} = \lambda (d _ {o d} ^ {(t - 1)} - d _ {o d} ^ {(t)}), \tag {9}
$$

$$
r _ {t} ^ {\text { Leave }} = \lambda (d _ {r o} ^ {(t)} - d _ {r o} ^ {(t - 1)}), \tag {10}
$$

where  is a scaling coefficient. Reach and Place reward reλductions in end-effector-object and object-destination distances, while Leave rewards increasing separation between the endeffector and object to discourage post-placement interference.

Phase assignment is inferred online from low-level interaction signals rather than predefined temporal boundaries. Transitions are triggered based on gripper state stability and relative object poses across consecutive steps, combined with task-relevant thresholds such as object-destination proximity. A prioritybased rule enforces mutually exclusive phase selection: Leave overrides Place once the object is released and stably positioned, while rapid oscillations between phases are prevented. This stability-driven inference ensures consistent phase attribution without relying on annotations or fixed temporal segments.

By converting geometric task structure into immediate, phaseconsistent learning signals, step-level dense rewards embed manipulation priors directly into the RL objective. While the Reach-Place-Leave decomposition reflects common manipulation semantics, the reward formulation itself depends only on relative geometric progress and online phase inference, making it adaptable to different objects and task variants. Combined with fused spatial-visual embeddings, this reward design enables stable fine-tuning of flow-matching VLA policies, preserves zero-shot generalization, and enforces both local and global spatial consistency in complex, cluttered environments.

# 3.5 Spatially-Conditioned Annealed Noise (SCAN)

The preceding sections establish two key ingredients for robust RL adaptation in spatially complex manipulation: geometryaware state representations via spatial token fusion and informative learning signals via step-level dense rewards. However, effective exploitation of these signals critically depends on exploration. Without sufficiently diverse and geometry-consistent trajectories, even well-shaped rewards provide limited supervision, particularly under occlusion, contact uncertainty, and multi-modal action distributions. This motivates a spatiallyaware exploration mechanism that is aligned with the same geometric structure encoded in the policy input and reward design.

To this end, we introduce SCAN (Spatially-Conditioned Annealed Noise), an exploration strategy that injects stochasticity conditioned on spatial embeddings while enforcing an annealed minimum noise floor. SCAN ensures persistent exploration throughout fine-tuning, avoids premature collapse to deterministic behaviors, and respects the local geometry of the manipulation space.

We treat the flow-matching policy as defining a stochastic action distribution that is fully compatible with PPO’s surrogate objective. We consider a standard Markov decision process $( S , \mathcal { R } , P , r , \gamma )$ , where $s _ { t } \in S$ denotes the state, $a _ { t } \in \mathcal { A }$ the action,

Algorithm 1 SCAN: Spatially-Conditioned Annealed Noise   
Input: policy $\pi_{\theta}$ , dataset of states $s_{t}$ , annealing schedule $\sigma_{\min}(t)$ for each training step t do
    Extract spatial features $x_{t}$ from $s_{t}$ Predict learned noise $\sigma_{\text{learned}}(x_{t})$ Compute total noise: $\sigma_{t}(x_{t}) = \sigma_{\min}(t) + h(\sigma_{\text{learned}}(x_{t}) - \sigma_{\min}(t))$ Sample $\epsilon_{t} \sim \mathcal{N}(0, \sigma_{t}^{2}(x_{t}))$ Inject noise: $x_{t+\delta} = x_{t} + \delta f_{\theta}(x_{t}, t) + g_{t}(x_{t}) \epsilon_{t}$ Update policy via PPO using perturbed trajectories
end for

P the transition kernel, r the reward function, and  the discount γfactor. Policy optimization aims to maximize the expected return

$$
J (\theta) = \mathbb {E} _ {\pi_ {\theta}} \Big [ \sum_ {t = 0} ^ {T} \gamma^ {t} r (s _ {t}, a _ {t}) \Big ]. \tag {11}
$$

Flow-matching VLA policies model continuous, highdimensional control dynamics as

$$
x _ {t + \delta} = x _ {t} + \delta f _ {\theta} (x _ {t}, t) + g _ {t} (x _ {t}) \epsilon , \quad \epsilon \sim \mathcal {N} (0, I), \tag {12}
$$

where $f _ { \theta }$ predicts the deterministic flow and $g _ { t } ( x _ { t } )$ scales the θstochastic component. In practice, unconstrained learning of $g _ { t } ( x _ { t } )$ often leads to noise collapse in low-gradient or locally stable regions, yielding nearly deterministic trajectories that hinder exploration. Conversely, fixed isotropic noise ignores spatial structure and may induce inefficient or unsafe behaviors. SCAN addresses this trade-off by coupling learned, geometryaware noise with an annealed lower bound.

Specifically, SCAN defines the effective noise scale as

$$
\sigma_ {t} (x _ {t}) = \sigma_ {\min} (t) + h (\sigma_ {\text { learned }} (x _ {t}) - \sigma_ {\min} (t)), \tag {13}
$$

where $\sigma _ { \mathrm { l e a r n e d } } ( x _ { t } )$ is predicted from the fused visual-spatial emσbedding ${ \bf h } _ { t } , h ( \cdot ) = \log ( 1 + \exp ( \cdot ) )$ is a smooth monotonic function, and $\sigma _ { \mathrm { m i n } } ( t )$ is an annealed minimum floor that prevents σvanishing stochasticity. The floor is scheduled as

$$
\sigma_ {\min} (t) = \alpha (t) \sqrt {\frac {t}{1 - t}}, \tag {14}
$$

$$
\alpha (t) = \alpha_ {0} + (\alpha_ {1} - \alpha_ {0}) \frac {\min (k , K)}{K}, \tag {15}
$$

where k denotes the global training step and K the annealing horizon. This construction enforces strictly positive noise throughout training, effectively maintaining a lower bound on policy entropy and avoiding premature convergence.

Formally, SCAN induces a spatially-adaptive stochastic process

$$
d x = f _ {\theta} (x, t) d t + \Sigma (x, t) d W _ {t}, \tag {16}
$$

$$
\Sigma (x, t) = \mathrm{diag} (\sigma_ {t} ^ {2} (x)), \tag {17}
$$

$$
\Sigma (x, t) \succeq \Sigma_ {\min} (t), \tag {18}
$$

where $d W _ { t }$ denotes a Wiener process [41]. The learned component $\sigma _ { \mathrm { l e a r n e d } } ( x _ { t } )$ captures local geometric and perceptual uncerσtainty, assigning higher variance in regions of contact sensitivity, spatial ambiguity, or incomplete observation, while allowing near-deterministic behavior in well-explored states. The annealed floor $\sigma _ { \mathrm { m i n } } ( t )$ encourages broad exploration early and σgradually shifts toward fine-grained refinement, implementing a coarse-to-fine homotopy over the spatial action landscape.

By conditioning exploration noise on the same geometry-aware embeddings used for policy prediction and by enforcing an annealed minimum, SCAN aligns exploration with step-level dense rewards. This synergy increases the likelihood of visiting geometrically meaningful states where phase-wise rewards provide informative gradients, improving credit assignment across manipulation phases. When integrated with PPO, SCAN balances exploration and exploitation in complex, cluttered environments, stabilizes online fine-tuning, and reinforces behaviors that generalize under viewpoint shifts and occlusion.

Algorithm 1 summarizes the SCAN procedure, highlighting spatial feature extraction, noise computation, and policy updates with spatially-conditioned stochasticity.

# 4 Experiments

This section evaluates SA-VLA under spatial distribution shifts, with a focus on robustness and training stability during reinforcement learning adaptation. Rather than only measuring performance gains, we aim to attribute robustness improvements to specific design choices in representation, reward design, and exploration.

Accordingly, we structure the experimental analysis around three research questions:

• RQ1: Does explicit spatial token fusion improve zeroshot spatial generalization before RL adaptation?   
• RQ2: How does reward density influence robustness and optimization stability under noise-injected RL?   
• RQ3: Does spatially-conditioned exploration provide additional benefits beyond dense rewards when spatial coverage is limited?

Each question isolates a distinct component of SA-VLA, enabling a component-wise analysis of spatial inductive bias across representation learning, reward shaping, and exploration.

# 4.1 Datasets

We conduct experiments on LIBERO [42] and its robustness extension LIBERO-PLUS [43]. LIBERO consists of languageconditioned manipulation tasks with visual and proprioceptive observations, while LIBERO-PLUS introduces systematic perturbations in camera pose and initial state. These perturbations evaluate whether a policy preserves consistent spatial reasoning under changes in viewpoint and configuration, a regime where RL fine-tuning has been observed to degrade spatial inductive bias (Sec. 1).

For few-shot RL, we construct a sparse spatial subset by uniformly sampling three trajectories per task and perturbation pair, resulting in 60 pairs. Evaluation is performed using 80 parallel environments over five non-repeating sampling epochs to reduce variance and ensure stable comparison across methods.

Table 1: Zero-shot success rate (%) on the spatial-perturbation subset of LIBERO-PLUS. All models are trained on LIBERO and evaluated without RL adaptation. 

<table><tr><td>Method</td><td>View</td><td>Init State</td><td>Total</td></tr><tr><td> $\pi_{0.5}$  (baseline)</td><td>78.47</td><td>83.77</td><td>81.00</td></tr><tr><td>Ours (w/ Spatial)</td><td>82.30</td><td>84.29</td><td>83.25 (+2.25)</td></tr></table>

# 4.2 Implementation Details

All experiments initialize from a pretrained flow matching VLA policy with a frozen visual language backbone. Implicit spatial tokens are extracted using a pretrained multi view spatial encoder and fused with visual tokens as described in Sec. 3.3. The resulting embeddings ht condition the policy during action generation.

Policy optimization uses actor-critic PPO with GAE. The learning rates are $5 \times 1 0 ^ { - 6 }$ for the policy and $1 \times 1 0 ^ { - 4 }$ for the value function, with a clip ratio of 0.2, discount factor $\gamma = 0 . 9 9$ , and γ .GAE  = 0 95. We apply gradient clipping at 1.0 and use a λ .global batch size of 1024 aggregated from 64 parallel environments over eight rollout epochs. The maximum episode length is 240 steps, and training runs for 100 total steps, requiring approximately 38.5 hours on four H800 GPUs.

Sparse reward agents receive only terminal success signals. Dense reward agents additionally use step-level spatial rewards defined in Sec. 3.4, with the dense reward coefficient set to $\lambda _ { \mathrm { d e n s e } } ~ = ~ 0 . 3$ . Exploration noise is injected using SDE [39], λ .learnable flow noise [37], or the proposed SCAN described in Sec. 3.5, with an annealing threshold of 80 steps.

Evaluation is conducted using 80 parallel environments over five non repeating sampling epochs, with two random seeds per setting. We found trends to be consistent across seeds, and report standard deviations to reflect remaining variance. All metrics report mean and standard deviation.

# 4.3 Zero-Shot Spatial Generalization Enabled by Spatial Token Fusion

We first address RQ1 by isolating the effect of spatial representations before RL adaptation. All models are trained on the full set of LIBERO tasks using supervised learning and evaluated zero-shot on the spatial-perturbation subset of LIBERO-PLUS. This evaluation subset includes variations in camera viewpoint and object initial configurations, testing the ability of policies to generalize without additional training.

Table 1 shows that incorporating spatial token fusion improves success rates across all perturbation types. The overall gain of +2 25% demonstrates that explicitly encoding geometric .structure strengthens task-relevant spatial reasoning under distribution shifts. The improvement is more pronounced under camera-view perturbations (+3 83%) than under initial-state .perturbations (+0 52%), indicating that multi-view spatial cues .enhance robustness to observation-level changes while offering limited benefit for initial-state variations.

These results establish a controlled baseline before RL adaptation, isolating the contribution of spatial token fusion. They provide a reference point for evaluating how reward design and exploration strategies further improve robustness under spatial distribution shifts.

![](images/c77468f21530f3ad990a92782b5f0ae1487e54ce537d5061ed0a6766e3e725a6.jpg)

<details>
<summary>line</summary>

| Step | few-shot: dense reward | few-shot: sparse reward | zero-shot: dense reward | zero-shot: sparse reward |
| ---- | ------------------------ | ------------------------ | ------------------------ | ------------------------- |
| 10   | 81.5                     | 82.5                     | 83.0                     | 81.0                      |
| 20   | 81.0                     | 82.0                     | 82.5                     | 81.5                      |
| 30   | 81.5                     | 81.5                     | 82.0                     | 80.5                      |
| 40   | 81.5                     | 81.0                     | 81.5                     | 79.5                      |
| 50   | 81.5                     | 80.5                     | 81.0                     | 78.0                      |
| 60   | 82.0                     | 80.5                     | 81.5                     | 77.0                      |
| 70   | 82.5                     | 80.5                     | 82.0                     | 76.5                      |
| 80   | 83.0                     | 80.5                     | 82.5                     | 76.5                      |
| 90   | 83.5                     | 80.5                     | 82.5                     | 76.0                      |
| 100  | 84.0                     | 80.5                     | 82.5                     | 76.0                      |
</details>

Figure 5: Training dynamics on the LIBERO-PLUS spatialperturbation subset. Success rates are evaluated using SDEbased policy checkpoints saved every 10 training steps. Solid curves denote few-shot RL, and dashed curves denote zero-shot evaluation. Zero-shot evaluation uses 8 environments with a global batch size of 384, while few-shot RL uses 64 environments with a global batch size of 2048.

# 4.4 Effect of Reward Density on Spatial Generalization

We next examine RQ2, studying how reward density influences spatial generalization during noise-injected RL fine-tuning. This evaluation addresses whether optimization stability under spatial perturbations depends on supervision granularity. All experiments use a fixed spatial-perturbation subset of LIBERO-PLUS and apply the same SDE-based stochasticity for policy exploration.

Figure 5 compares sparse and dense reward signals under zeroshot and few-shot evaluation. Dense rewards consistently lead to faster convergence and smoother training dynamics across all settings. Notably, this effect is observed even in zero-shot evaluation, where no task-specific adaptation occurs, indicating that dense rewards stabilize learning rather than solely improving final performance.

The difference arises from how rewards constrain noise-induced exploration. Sparse rewards provide weak guidance for credit assignment, leading to high-variance gradients and unstable updates. Dense rewards supply step-level supervision aligned with intermediate spatial progress, guiding exploration toward meaningful states and regularizing policy updates under stochastic perturbations.

This stabilizing effect is especially pronounced in few-shot settings, where limited trajectory coverage amplifies optimization noise. Dense supervision prevents overfitting to incidental behaviors from stochastic exploration and preserves consistent spatial reasoning across perturbations.

Overall, these results show that reward density is a key factor for robust RL adaptation under spatial distribution shifts, complementing spatial representations by improving optimization stability rather than only enhancing final success rates.

![](images/33474eecdeedd4163114ba66080cdcf91c667b2d6f1dcdaf1dc76e0d1e3b7434.jpg)

<details>
<summary>line</summary>

| Step | scan_d mean | scan_d ±1 std | noise_d mean | noise_d ±1 std | noise_s mean | noise_s ±1 std |
|------|-------------|---------------|--------------|----------------|--------------|----------------|
| 0    | 0.87        | 0.87          | 0.87         | 0.87           | 0.87         | 0.87           |
| 20   | 0.88        | 0.88          | 0.89         | 0.89           | 0.89         | 0.89           |
| 40   | 0.90        | 0.90          | 0.90         | 0.90           | 0.90         | 0.90           |
| 60   | 0.91        | 0.91          | 0.91         | 0.91           | 0.91         | 0.91           |
| 80   | 0.91        | 0.91          | 0.91         | 0.91           | 0.91         | 0.91           |
| 100  | 0.91        | 0.91          | 0.91         | 0.91           | 0.91         | 0.91           |
</details>

Figure 6: Training dynamics under limited spatial coverage. Dense rewards stabilize RL optimization, while combining dense rewards with SCAN further improves final success rate. Shaded regions denote one standard deviation over two seeds.

# 4.5 Spatially-Conditioned Exploration Beyond Dense Rewards

While step-level dense rewards stabilize RL optimization (Sec. 4.4), limited spatial coverage can still hinder the policy from encountering key geometric configurations. This motivates RQ3, assessing whether spatially-conditioned exploration via SCAN further preserves spatial inductive bias during fine-tuning.

We evaluate three regimes: (1) flow\_noise [37] with sparse rewards, (2) flow\_noise with dense rewards, and (3) dense rewards combined with SCAN. Figure 6 illustrates training dynamics under few-shot RL on spatially perturbed environments.

Sparse rewards alone lead to unstable updates and local convergence, as the policy receives weak guidance on intermediate geometric progress. Dense rewards reduce gradient variance and stabilize optimization, but in settings with limited trajectory coverage, the agent may fail to visit critical states, resulting in partial spatial exploration and suboptimal generalization. In contrast, SCAN injects spatially-conditioned stochasticity aligned with the fused visual-spatial embeddings, actively guiding the agent to under-sampled geometric regions while maintaining consistency with phase-specific dense rewards.

The combination of dense rewards and SCAN yields three key effects: (1) gradients are well-aligned with geometric progress, preventing reward-action shortcuts; (2) exploration covers spatially relevant states that are otherwise missed, improving state visitation and credit assignment; (3) stochasticity preserves controlled diversity, reflected in higher variance that corresponds to targeted, spatially-aware exploration rather than training instability.

As a result, SCAN enables policies to maintain robust spatial reasoning under distribution shifts, achieving higher final success rates than dense rewards alone. These observations support our central claim that coupling geometry-aware dense rewards with spatially conditioned exploration preserves the spatial inductive bias of flow-matching VLA policies during RL adaptation, leading to stable and transferable behaviors under spatial perturbations.

# 4.6 Ablation Study

We perform a cumulative ablation study to quantify the contribution of each spatial component in SA-VLA. All experiments are evaluated on the LIBERO-PLUS spatial-perturbation subset, and unless noted otherwise, all variants are fine-tuned following the RL protocol in Sec. 4.5.

Table 2 reports few-shot success rates as components are progressively removed from the full model. Each row cumulatively removes modules to isolate the effects of spatially conditioned exploration (SCAN), step-level dense rewards (DR), and spatial token fusion.

Removing SCAN leads to a clear performance drop, showing that spatially conditioned exploration is critical for robustness after RL adaptation. Further removing dense rewards results in the lowest success among RL-finetuned variants, consistent with prior results on reward density. Spatial token fusion alone yields modest gains but cannot ensure robustness without reward shaping and exploration.

The final row shows a non-RL baseline, which lacks task-specific adaptation. While it performs better than sparsely rewarded RL fine-tuning, it is not directly comparable to RL-adapted variants.

Table 2: Cumulative ablation of spatial components on fewshot success rate (%) on LIBERO-PLUS. The bottom row is evaluated without RL adaptation, while all other rows report RL-finetuned policies obtained in Sec. 4.5. 

<table><tr><td>Method</td><td>SCAN</td><td>DR</td><td>Fusion</td><td>SR (%)</td></tr><tr><td>SA-VLA</td><td>√</td><td>√</td><td>√</td><td>83.75</td></tr><tr><td>w/o SCAN</td><td>×</td><td>√</td><td>√</td><td>83.00</td></tr><tr><td>w/o DR</td><td>×</td><td>×</td><td>√</td><td>77.50</td></tr><tr><td>w/o Fusion</td><td>×</td><td>×</td><td>×</td><td>81.00</td></tr></table>

Overall, the ablation confirms that SCAN, dense rewards, and spatial fusion provide complementary benefits. The strongest performance is achieved when all components are enabled, while removing any subset during RL adaptation systematically degrades spatial robustness.

# 5 Conclusion

We identify that the degradation of flow-matching Vision-Language-Action policies under spatial distribution shifts stems from a collapse of spatial inductive bias during RL fine-tuning. To address this, we propose SA-VLA, which preserves spatial awareness through aligned representation learning, reward design, and exploration. Across diverse manipulation tasks, this spatially grounded adaptation improves robustness and transferability, highlighting the importance of preserving inductive structure when extending flow-based VLA policies beyond supervised pretraining.

# References

[1] Charles C Kemp, Aaron Edsinger, and Eduardo Torres-Jara. Challenges for robot manipulation in human environments [grand challenges of robotics]. IEEE Robot. Autom. Mag., 14(1):20–29, 2007.   
[2] JJ Gibson. The ecological approach to visual perception: classic edition, 2014.   
[3] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.   
[4] Kento Kawaharazuka, Jihoon Oh, Jun Yamada, Ingmar Posner, and Yuke Zhu. Vision-language-action models for robotics: A review towards real-world applications. IEEE Access, 2025.   
[5] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. Int. J. Robotics Res., 44(10-11):1684– 1704, 2025. doi: 10.1177/02783649241273668.   
[6] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. $\pi _ { 0 . 5 } \colon$ a vision-language-action model with openπ .world generalization, 2025.   
[7] Kang Chen, Zhihao Liu, Tonghe Zhang, Zhen Guo, Si Xu, Hao Lin, Hongzhi Zang, Quanlu Zhang, Zhaofei Yu, Guoliang Fan, et al. $\pi _ { \mathrm { r 1 } }            \mathrm { : }$ Online rl fine-tuning for flowπbased vision-language-action models. arXiv preprint arXiv:2510.25889, 2025.   
[8] Hongyin Zhang, Shiyuan Zhang, Junxi Jin, Qixin Zeng, Yifan Qiao, Hongchao Lu, and Donglin Wang. Balancing signal and variance: Adaptive offline rl post-training for vla flow models. arXiv preprint arXiv:2509.04063, 2025.   
[9] Guozheng Ma, Zhen Wang, Zhecheng Yuan, Xueqian Wang, Bo Yuan, and Dacheng Tao. A comprehensive survey of data augmentation in visual reinforcement learning. Int. J. Comput. Vis., 133(10):7368–7405, 2025.   
[10] Leitian Tao, Ilia Kulikov, Swarnadeep Saha, Tianlu Wang, Jing Xu, Sharon Li, Jason E Weston, and Ping Yu. Hybrid reinforcement: When reward is sparse, it’s better to be dense. arXiv preprint arXiv:2510.07242, 2025.   
[11] Cheng Yin, Yankai Lin, Wang Xu, Sikyuen Tam, Xiangrui Zeng, Zhiyuan Liu, and Zhouping Yin. Deepthinkvla: Enhancing reasoning capability of vision-language-action models. arXiv preprint arXiv:2511.15669, 2025.   
[12] Mingyang Lyu, Yinqian Sun, Erliang Lin, Huangrui Li, Ruolin Chen, Feifei Zhao, and Yi Zeng. Reinforcement

fine-tuning of flow-matching policies for vision-languageaction models. arXiv preprint arXiv:2510.09976, 2025.   
[13] Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. A survey on vision-language-action models for embodied ai. arXiv preprint arXiv:2405.14093, 2024.   
[14] Zhijie Wang, Zhehua Zhou, Jiayang Song, Yuheng Huang, Zhan Shu, and Lei Ma. Vlatest: Testing and evaluating vision-language-action models for robotic manipulation. Proc. ACM Softw. Eng., 2(FSE):1615–1638, 2025.   
[15] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.   
[16] Scott E. Reed, Konrad Zolna, Emilio Parisotto, Sergio Gómez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, Tom Eccles, Jake Bruce, Ali Razavi, Ashley Edwards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol Vinyals, Mahyar Bordbar, and Nando de Freitas. A generalist agent. Trans. Mach. Learn. Res., 2022, 2022.   
[17] Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, et al. Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robot. Autom. Lett., 2025.   
[18] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In IEEE Conf. Comput. Vis. Pattern Recog., pages 815–824, 2023.   
[19] Peiyan Li, Yixiang Chen, Hongtao Wu, Xiao Ma, Xiangnan Wu, Yan Huang, Liang Wang, Tao Kong, and Tieniu Tan. Bridgevla: Input-output alignment for efficient 3d manipulation learning with vision-language models. arXiv preprint arXiv:2506.07961, 2025.   
[20] Shengliang Deng, Mi Yan, Yixin Zheng, Jiayi Su, Wenhao Zhang, Xiaoguang Zhao, Heming Cui, Zhizheng Zhang, and He Wang. Stereovla: Enhancing visionlanguage-action models with stereo vision. arXiv preprint arXiv:2512.21970, 2025.   
[21] Rui Shao, Wei Li, Lingsen Zhang, Renshan Zhang, Zhiyang Liu, Ran Chen, and Liqiang Nie. Large vlm-based vision-language-action models for robotic manipulation: A survey. arXiv preprint arXiv:2508.13073, 2025.   
[22] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In IEEE Conf. Comput. Vis. Pattern Recog., pages 652–660, 2017.   
[23] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In IEEE Conf. Comput. Vis. Pattern Recog., 2017.   
[24] Xian-Feng Han, Hamid Laga, and Mohammed Bennamoun. Image-based 3d object reconstruction: Stateof-the-art and trends in the deep learning era. IEEE Trans. Pattern Anal. Mach. Intell., 43(5):1578–1604, 2019.

[25] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973, 2023.   
[26] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In IEEE Conf. Comput. Vis. Pattern Recog., pages 5294–5306, 2025.   
[27] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Adv. Neural Inform. Process. Syst., 35:23716–23736, 2022.   
[28] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, et al. Palm-e: An embodied multimodal language model. 2023.   
[29] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Int. Conf. Comput. Vis., pages 11975–11986, 2023.   
[30] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. Advances in Neural Information Processing Systems, 36:1363–1389, 2023.   
[31] Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. J. Mach. Learn. Res., 17(39):1–40, 2016.   
[32] Ilija Radosavovic, Tete Xiao, Stephen James, Pieter Abbeel, Jitendra Malik, and Trevor Darrell. Real-world robot learning with masked visual pre-training. In Conf. Robot. Learn., pages 416–426. PMLR, 2023.   
[33] Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by selfsupervised prediction. In Int. Conf. Mach. Learn., pages 2778–2787. PMLR, 2017.   
[34] Max Schwarzer, Nitarshan Rajkumar, Michael Noukhovitch, Ankesh Anand, Laurent Charlin, R Devon Hjelm, Philip Bachman, and Aaron C Courville. Pretraining representations for data-efficient reinforcement learning. Adv. Neural Inform. Process. Syst., 34: 12686–12699, 2021.   
[35] Rémy Portelas, Cédric Colas, Katja Hofmann, and Pierre-Yves Oudeyer. Teacher algorithms for curriculum learning of deep rl in continuously parameterized environments. In Conf. Robot. Learn., pages 835–853. PMLR, 2020.   
[36] Sebastien Racaniere, Andrew Lampinen, Adam Santoro, David Reichert, Vlad Firoiu, and Timothy Lillicrap. Automated curriculum generation through setter-solver interactions. In Int. Conf. Learn. Represent., 2020.   
[37] Tonghe Zhang, Chao Yu, Sichang Su, and Yu Wang. Reinflow: Fine-tuning flow matching policy with online reinforcement learning. arXiv preprint arXiv:2505.22094, 2025.   
[38] Samuel Pfrommer, Yixiao Huang, and Somayeh Sojoudi. Reinforcement learning for flow-matching policies. arXiv preprint arXiv:2507.15073, 2025.

[39] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.   
[40] Dechen Gao, Boqi Zhao, Andrew Lee, Ian Chuang, Hanchu Zhou, Hang Wang, Zhe Zhao, Junshan Zhang, and Iman Soltani. Vita: Vision-to-action flow matching policy. arXiv preprint arXiv:2507.13231, 2025.   
[41] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.   
[42] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Adv. Neural Inform. Process. Syst., 36:44776–44791, 2023.   
[43] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. Liberoplus: In-depth robustness analysis of vision-languageaction models. arXiv preprint arXiv:2510.13626, 2025.

# A Additional Methodological Insights

This appendix provides complementary methodological details that clarify several design and implementation choices underlying our approach. Rather than repeating standard optimization settings described in the main text, we focus on aspects that are either implicit in the algorithmic formulation or tightly coupled to the stability of reinforcement learning fine-tuning under spatial distribution shifts. These details aim to improve clarity and reproducibility while preserving the conceptual flow of the main paper.

# A.1 Why Implicit Spatial Tokens Instead of Explicit 3D Representations

A natural alternative to spatial tokens is to explicitly reconstruct 3D geometry using point clouds, voxel grids, or signed distance fields. However, under reinforcement learning fine-tuning of pretrained policies, explicit 3D representations are poorly matched to the optimization objective.

Explicit reconstruction pipelines typically rely on auxiliary supervision or reconstruction losses. When combined with policy optimization, this introduces competing gradient signals that are not directly aligned with the reinforcement learning objective,

$$
\nabla_ {\theta} \mathbb {E} \left[ \sum_ {t} r _ {t} \right], \tag {19}
$$

and can destabilize policy updates. Moreover, discretized 3D representations are sensitive to partial observability, sensor noise, and viewpoint sparsity, which are exacerbated during online data collection. These factors jointly lead to high-variance gradients and brittle adaptation behavior.

In contrast, implicit spatial tokens encode geometric structure directly within a learned feature space derived from multi-view observations. They preserve continuous spatial information while remaining fully differentiable and directly coupled to the policy backbone. As a result, spatial reasoning is injected into the policy through feature modulation rather than through an explicit reconstruction objective. This allows geometric cues to influence action selection while remaining fully aligned with the reinforcement learning loss.

We therefore treat spatial tokens as complementary geometric context rather than explicit scene reconstructions. This design preserves the simplicity of the policy learning pipeline while enabling stable and geometry-aware adaptation under spatial distribution shifts.

# A.2 Directional Design of Spatial Token Fusion

Spatial token fusion is implemented using unidirectional cross-attention from visual tokens to spatial tokens. This asymmetric design reflects the distinct roles of the two representations. Visual tokens encode task-relevant semantics and fine-grained appearance cues, but are sensitive to occlusion and viewpoint variation. Spatial tokens encode coarser but more stable geometric structure derived from multi-view aggregation.

Unidirectional attention allows visual tokens to selectively query geometric context that is relevant for action generation. Formally, spatial tokens serve as keys and values that condition visual features, while remaining invariant to gradients originating from appearance noise. In contrast, bidirectional attention would allow noisy visual updates to directly modify spatial representations, gradually eroding their geometric consistency during reinforcement learning.

By keeping spatial tokens as a read-only context during fusion, the model preserves a stable geometric anchor throughout fine-tuning. This asymmetric interaction improves robustness under partial observability and prevents destructive interference between appearance-driven and geometry-driven signals.

# A.3 Design Rationale of Channel-wise Gating

Channel-wise gating regulates how spatial information modulates visual representations during policy adaptation. Unlike a simple residual addition or a scalar gate, channel-wise modulation allows the model to selectively amplify or suppress geometric cues along different semantic dimensions. This is critical because only a subset of visual channels benefit from spatial conditioning at any given state.

The gating function is bounded using a hyperbolic tangent activation, which constrains feature modulation to a fixed range. This prevents excessive amplification of spatial features during early stages of reinforcement learning, where policy gradients can be highly unstable. At the same time, bounded modulation still permits strong geometric influence in regions affected by occlusion, clutter, or viewpoint changes.

Empirically, this design stabilizes policy optimization while preserving the expressive capacity of the fused representation. It allows spatial cues to guide action selection without overwhelming pretrained visual semantics.

# A.4 Why Reach-Place-Leave Decomposition

The Reach-Place-Leave decomposition is not a task-specific script, but a minimal abstraction of object-centric manipulation geometry. Across a wide range of manipulation tasks, successful execution requires reducing the distance between the end-effector and the target object, transporting the object toward a goal region, and increasing separation after placement. These stages correspond to distinct geometric relations that can be measured using relative distances.

Phase assignment is inferred online from interaction signals rather than predefined temporal schedules. This enables policies to transition between phases based on execution dynamics rather than fixed time steps. As a result, dense supervision is provided without constraining the temporal structure of the policy.

Importantly, the reward formulation depends only on signed changes in relative geometric distances within each phase. It does not rely on absolute thresholds or hand-designed trajectories. This preserves flexibility while embedding minimal manipulation priors into the reinforcement learning objective.

# A.5 Why Spatially-Conditioned Exploration Beyond Dense Rewards

Dense rewards shape the direction of policy gradients but do not directly control state visitation. In spatially complex environments, policy optimization can converge to locally consistent but globally suboptimal behaviors when exploration fails to cover critical geometric configurations.

Spatially-conditioned exploration addresses this limitation by explicitly biasing action sampling toward spatially meaningful regions of the state space. While dense rewards encode task progress, exploration governs which geometric interactions are experienced during training. These two mechanisms therefore operate at different levels of the learning process.

By combining phase-consistent dense rewards with spatially-conditioned exploration, SA-VLA improves both gradient quality and state coverage. This combination is essential for stable fine-tuning under spatial distribution shifts.

# B Detailed Algorithmic and Implementation Details

This section provides additional algorithmic and implementation details that complement, rather than replicate, the main method description. While the core model design and learning objectives are presented in Secs. 3.3-3.5, several practical choices are difficult to fully articulate in the main text without interrupting the methodological narrative.

We therefore focus on implementation-level considerations that are critical for training stability, numerical robustness, and reproducibility, including token handling strategies, phase inference logic, reward normalization, and exploration scheduling. These details reflect constraints encountered during online reinforcement learning and are essential for faithfully reproducing the reported results.

# B.1 Choice and Analysis of Exploration Noise

The parameterization of exploration noise, in particular whether stochasticity is modeled as an intrinsic component of the policy or introduced externally, plays a critical role in both the theoretical validity of PPO optimization and the interpretation of experimental results. Accordingly, we employ different exploration noise mechanisms across experiments to isolate specific factors while maintaining conceptual clarity.

• SDE-based isotropic noise (SDE): used in Sec. 4.4 to isolate the effect of reward density. This geometry-agnostic noise serves as a controlled baseline, enabling a fair comparison between sparse and dense rewards without introducing policy-dependent exploration.   
• Flow\_noise: used in Sec. 4.5 to evaluate spatially-conditioned exploration. This learned noise head generates geometryconsistent stochasticity from spatial tokens and is fully parameterized as part of the flow-matching policy, ensuring compatibility with PPO optimization.   
• SCAN (Spatially-Conditioned Annealed Noise): combines a learned, geometry-consistent noise head with a temporarily injected isotropic SDE floor. The annealing schedule ensures that exploration noise converges to a policy-dependent distribution, reconciling early-stage exploration with theoretical consistency.

Mathematical analysis under PPO We formalize the distinction by considering a Gaussian policy

$$
\pi_ {\theta} (x _ {t - \Delta} \mid x _ {t}, s) = \mathcal {N} (\mu_ {\theta} (x _ {t}, s), \sigma_ {\theta} (x _ {t}, s) ^ {2}),
$$

where $x _ { t - \Delta } = \mu _ { \theta } ( x _ { t } , s ) + \sigma _ { \theta } ( x _ { t } , s ) \cdot \epsilon { \mathrm { ~ a n d ~ } } \epsilon \sim N ( 0 , 1 ) .$

1. SDE-based isotropic noise injects stochasticity via

$$
x _ {t - \Delta} = \mu_ {\theta} (x _ {t}) + \sigma (t) \sqrt {\Delta} \epsilon , d x = v _ {\theta} (x, t) d t + \sigma (t) d W _ {t},
$$

where $d W _ { t }$ is a Brownian increment and $\sigma ( t )$ is independent of policy parameters . As a result:

• The injected noise is external to the policy $\pi _ { \theta } ,$ and therefore not reflected in the likelihood ratio optimized by PPO.   
πθ• This leads to an objective mismatch, as the effective action distribution used for exploration differs from the distribution assumed in PPO gradient estimation.   
• Consequently, variance cannot adapt to the reward signal; only the mean is optimized, limiting the explorationexploitation trade-off and potentially amplifying instability under sparse or long-horizon rewards.

# 2. Flow\_noise instead defines

$$
x _ {t - \Delta} = \mu_ {\theta} (x _ {t}, s) + \sigma_ {\theta} (x _ {t}, s) \cdot \epsilon ,
$$

where $\sigma _ { \theta }$ is learned from spatial tokens and explicitly parameterized as part of the policy. This ensures that both the mean σθand variance contribute to the likelihood ratio $r _ { t } ( \theta )$ , making surrogate objectives such as PPO’s clipped loss

$$
L ^ {\mathrm{CLIP}} (\theta) = \mathbb {E} _ {t} \left[ \min \left(r _ {t} (\theta) \hat {A} _ {t}, \operatorname{clip} \left(r _ {t} (\theta), 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {t}\right) \right]
$$

theoretically consistent.

3. SCAN temporarily injects SDE-style noise to encourage early exploration, but anneals it toward the learned $\sigma _ { \theta } ( x _ { t } , s )$ . After σθ ,annealing, all stochasticity is policy-dependent, guaranteeing that the final optimization objective remains mathematically aligned with PPO.

Empirical comparison Figure 7 reports few-shot evaluation on LIBERO-PLUS under dense and sparse rewards. Notably, the advantage of learned noise is consistent across reward regimes, indicating that policy-dependent stochasticity improves exploration robustness rather than merely amplifying reward density.

![](images/2460919e8c21aeb24bb3104bf06aff55ce0ae006870c115f208d85ac68ad734f.jpg)

<details>
<summary>line</summary>

| Step | noise: dense reward | noise: sparse reward | sde: dense reward | sde: sparse reward |
| ---- | ------------------- | -------------------- | ----------------- | ------------------ |
| 10   | 82.0                | 83.5                 | 81.8              | 83.5               |
| 20   | 82.2                | 83.2                 | 81.5              | 82.8               |
| 30   | 82.5                | 82.8                 | 81.0              | 82.5               |
| 40   | 82.5                | 82.0                 | 81.5              | 81.5               |
| 50   | 82.8                | 81.5                 | 82.0              | 81.0               |
| 60   | 82.8                | 81.0                 | 82.5              | 80.5               |
| 70   | 83.5                | 80.5                 | 83.0              | 80.0               |
| 80   | 83.5                | 80.5                 | 83.5              | 80.5               |
| 90   | 83.5                | 81.0                 | 84.0              | 80.5               |
| 100  | 83.0                | 81.5                 | 84.0              | 80.0               |
</details>

Figure 7: Few-shot evaluation on LIBERO-PLUS comparing SDE-based and learned exploration noise. Across both sparse and dense reward settings, learned noise consistently outperforms SDE-based noise with lower variance and more stable performance, highlighting the robustness of policy-dependent exploration.

# Discussion

• Flow\_noise and SCAN exploit spatial priors by learning exploration variance from fused visual-spatial tokens, enabling the policy to modulate stochasticity based on geometry-aware state representations.   
• SDE-based noise is suitable for isolating reward-density effects, but is theoretically inconsistent with PPO due to its policy-independent variance.   
• SCAN reconciles effective early-stage exploration with mathematical rigor by annealing external noise toward a policydependent distribution.   
• By conditioning exploration noise on fused spatial tokens, learned noise preserves spatial inductive bias during RL adaptation, mitigating shortcut learning that arises from geometry-agnostic stochasticity.   
• Empirically, learned noise yields more stable and consistent behavior, supporting the importance of policy-dependent, geometry-aware stochasticity.

Overall, this analysis clarifies that comparisons across reward density and exploration strategies are only meaningful when the exploration noise is parameterized consistently with the underlying policy, thereby justifying our experimental design choices.

These observations justify our experimental design: SDE for reward density analysis, and flow\_noise / SCAN for evaluating spatially-conditioned exploration beyond dense rewards.

![](images/709961f8bb7d46b7a8d6b1fc37eec001e7bec671c3940de01be9154bb136cb02.jpg)  
Figure 8: Phase-wise dense reward visualization. Shown are changes in $d _ { \mathrm { r o } } , d _ { \mathrm { o d } } .$ , gripper opening angle, and the corresponding dense reward throughout task execution.

# B.2 Implementation Details of Spatial Token Fusion

This subsection clarifies several implementation-level design choices in the spatial token fusion module that are not explicitly discussed in the main text.

Separation of grid and global spatial tokens. The spatial encoder produces two types of tokens: dense grid tokens encoding spatial layout, and a small set of global tokens summarizing scene-level geometry. Only grid tokens participate in cross-attention with visual tokens. Global tokens are concatenated after fusion and bypass the attention mechanism.

This separation is intentional. Grid tokens provide localized geometric cues suitable for selective querying by visual semantics, while global tokens encode coarse context that does not benefit from fine-grained alignment. Allowing global tokens to enter cross-attention was empirically observed to introduce noisy gradients and degrade training stability.

Read-only treatment of spatial tokens. Spatial tokens are frozen during reinforcement learning and serve as keys and values in cross-attention. Gradients from policy updates do not modify spatial features, preventing appearance-driven noise from corrupting geometric structure. This design preserves the role of spatial tokens as a stable geometric reference throughout fine-tuning.

# B.3 Stability-Oriented Phase Inference

Phase inference is implemented using stability-based criteria rather than distance thresholds. This choice avoids premature phase transitions caused by noisy observations or transient contacts.

Specifically, phase transitions depend on short-horizon consistency of relative poses and gripper state, rather than instantaneous distance values. This prevents oscillations between Reach and Place when the end-effector briefly contacts the object, and avoids false Leave transitions due to transient object motion.

When a phase switch occurs, the reference distance for the newly active phase is reinitialized. This prevents spurious reward spikes at transition boundaries and ensures that dense rewards reflect local geometric progress within each phase.

# B.4 Reward Scaling and Numerical Stability

Dense rewards are computed as temporal differences of normalized distances. Normalization by the initial valid distance removes dependence on absolute scene scale and yields comparable reward magnitudes across tasks.

We additionally clip normalized distances to the unit interval and optionally clip reward values. These operations prevent large gradients caused by sudden distance changes, especially during early exploration or failed grasps.

Empirically, we observe that performance is robust to moderate variations of the scaling coefficient. This indicates that dense rewards primarily act as a directional geometric bias rather than a finely tuned optimization objective.

# C Additional Experimental Results

To further illustrate the efficacy of SA-VLA, we provide additional qualitative and interpretative analyses that complement the quantitative results presented in the main text. These results highlight the robustness of the policy under spatial perturbations and the interpretability of the step-level dense rewards.

# C.1 Phase-wise Reward Visualization

To illustrate the behavior of the dense reward, Fig. 8 closely tracks geometric progress: it increases when the end-effector approaches or moves the object toward the target, and decreases when actions deviate from the intended motion. The reward dynamics qualitatively reflect the underlying manipulation phases, providing interpretable learning signals aligned with task progression.

# C.2 Qualitative Results Under Spatial Perturbations

![](images/e0b958da0cc5daa4bb56084ca4e528291631586885cf2ef7c07d608362709ab9.jpg)  
Figure 9: Representative task executions under spatial perturbations. Keyframes are uniformly sampled (8 per rollout) from successful episodes. Despite variations in observation geometry, the policy consistently achieves task completion.

We visualize representative rollouts under diverse spatial perturbations in Fig. 9, sampling 8 frames uniformly along each episode horizon. All perturbations are applied at evaluation time. Across settings, the policy reliably progresses toward the target and completes the task, supporting the robustness trends observed in quantitative evaluations.

# D Limitations and Discussion

While SA-VLA generalizes across a range of object-centric manipulation tasks, several limitations remain. The Reach-Place-Leave abstraction captures common interactions but may require adaptation for tasks involving deformable objects or extended contact. The quality of spatial tokens is dependent on the pretrained spatial encoder, which can constrain performance in visually ambiguous or cluttered scenes. Future work could explore alternative interaction modalities, richer geometric representations, and more flexible phase abstractions to extend applicability and robustness.