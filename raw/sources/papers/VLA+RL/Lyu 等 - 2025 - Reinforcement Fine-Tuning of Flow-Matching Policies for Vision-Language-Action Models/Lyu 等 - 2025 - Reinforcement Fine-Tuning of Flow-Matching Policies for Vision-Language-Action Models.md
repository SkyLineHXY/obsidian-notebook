# Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models

Mingyang Lyua,b,1, Yinqian Suna,d,1, Erliang Lina, Huangrui Lib, Ruolin Chena,b, Feifei Zhaoa,c,d,2, Yi Zenga,b,c,d,2

Abstract— Vision-Language-Action (VLA) models such as OpenVLA, Octo, and π0 have shown strong generalization by leveraging large-scale demonstrations, yet their performance is still fundamentally constrained by the quality and coverage of supervised data. Reinforcement learning (RL) provides a promising path for improving and fine-tuning VLAs through online interaction. However, conventional policy gradient methods are computationally infeasible in the context of flow-matching based models due to the intractability of the importance sampling process, which requires explicit computation of policy ratios. To overcome this limitation, we propose Flow Policy Optimization (FPO) algorithm, which reformulates importance sampling by leveraging per-sample changes in the conditional flow-matching objective. Furthermore, FPO achieves stable and scalable online reinforcement fine-tuning of the π0 model by integrating structure-aware credit assignment to enhance gradient efficiency, clipped surrogate objectives to stabilize optimization, multi-step latent exploration to encourage diverse policy updates, and a Q-ensemble mechanism to provide robust value estimation. We evaluate FPO on the LIBERO benchmark and the ALOHA simulation task against supervised, preferencealigned, diffusion-based, autoregressive online RL, and π0-FAST baselines, observing consistent improvements over the imitation prior and strong alternatives with stable learning under sparse rewards. In addition, ablation studies and analyses of the latent space dynamics further highlight the contributions of individual components within FPO, validating the effectiveness of the proposed computational modules and the stable convergence of the conditional flow-matching objective during online RL.

# I. INTRODUCTION

The pursuit of generalist robots capable of executing a diverse array of physical tasks has advanced significantly with the emergence of Vision-Language-Action (VLA) models. Recent architectures, such as OpenVLA [1] and Octo [2], have demonstrated that policies pre-trained on large-scale datasets of human demonstrations can acquire broad semantic understanding and effectively execute a wide spectrum of instructions. Notably, the $\pi _ { 0 }$ [3] model implements action generation via a flow-matching technique [4], [5]. This method confers a unique advantage: it enables the generation of smooth, temporally coherent, high-frequency action segments, which is essential for achieving dexterous and long-horizon

Author affiliations: a Brain-inspired Cognitive AI Lab, Institute of Automation, Chinese Academy of Sciences, Beijing, China.; b University of Chinese Academy of Sciences (UCAS), Beijing, China.; c State Key Laboratory of Brain Cognition and Brain-inspired Intelligence Technology, Shanghai, China.; d Long-term AI, Beijing, China.

1 Co-first authors. 2 Co-corresponding authors. Correspondence: zhaofeifei2014@ia.ac.cn (Feifei Zhao), yi.zeng@ia.ac.cn (Yi Zeng). Additional contact: lvmingyang2024@ia.ac.cn (Mingyang Lyu).

manipulation tasks that require more than isolated, single-step action predictions.

Drawing inspiration from the remarkable progress of Reinforcement Learning (RL) in enhancing Large Language Models (LLMs) beyond their supervised fine-tuning (SFT) performance [15]–[17], there is a growing trend to apply RL for post-training of embodied VLA models. Approaches such as online RL for auto-regressive VLAs [6], iterative RL+SL stabilization for large VLAs [7], and policy-gradient fine-tuning of diffusion/flow-matching policies [8], [9] have demonstrated that robotic agents can leverage online interaction to refine skills and discover strategies superior to those available in initial imitation datasets. This paradigm allows agents to overcome the inherent limitations of offline demonstration data quality and coverage, pushing performance beyond the imitation ceiling.

However, a core technical incompatibility arises when applying conventional RL techniques to flow-matching-based VLA models such as $\pi _ { 0 }$ [3]. Commonly used policy gradient methods for reinforcement fine-tuning of VLA models, such as PPO [12] and TRPO [18], require importance sampling, i.e., the explicit computation of policy ratios. For flow-matching models, this computation is analytically intractable. It necessitates solving an underlying ordinary differential equation [19] and integrating a computationally prohibitive Jacobian trace term along the generation path [20]. This renders such methods computationally infeasible for the demands of online fine-tuning. While reward-weighted supervised learning approaches exist, they typically struggle with active exploration and the discovery of novel, out-of-distribution behaviors. These combined challenges have largely precluded the effective application of online RL for fine-tuning flowmatching generative policy-based VLA models.

In this paper, we introduce Flow Policy Optimization (FPO), a method designed to overcome the incompatibility between flow-matching policies and PPO-style updates by constructing a likelihood-free policy ratio based on persample changes in the conditional flow-matching objective. This formulation eliminates the need for explicit action likelihoods and ODE–Jacobian computations while preserving consistency with the policy’s generative structure. Furthermore, we provide structure-aware credit assignment in the latent space by leveraging the model’s training objective as a per-sample improvement signal. Combined with a clipped surrogate objective, multi-step latent exploration, and a Q-ensemble, FPO enables stable and efficient learning even in environments with sparse rewards and contactrich dynamics. The proposed method successfully facilitates online reinforcement fine-tuning of the $\pi _ { 0 }$ model, with its effectiveness and superiority empirically validated on the LIBERO benchmark and the ALOHA Transfer Cube task. In summary, the main contributions of this work are summarized as follows:

• We propose FPO, a practical policy optimization framework that bridges flow-matching policies and PPO-style updates by introducing a likelihood-free policy ratio derived from per-sample changes in the conditional flow-matching objective. This formulation avoids explicit density estimation and complex Jacobian computations, while retaining structural consistency with the generative policy.   
• We develop an online reinforcement fine-tuning algorithm for the $\pi _ { 0 }$ model by integrating structure-aware credit assignment in the latent space with key RL components including a clipped surrogate objective, multi-step latent exploration, and a Q-ensemble. This combination ensures stable and efficient learning in challenging environments with sparse rewards and contactrich dynamics.   
• Extensive experiments on the LIBERO benchmark and the ALOHA Transfer Cube task demonstrate the superior performance of $\pi _ { 0 } { \mathrm { - F P O } }$ over six strong baselines such as OpenVLA, Octo, Diffusion Policy, GRAPE, and $\pi _ { 0 } -$ FAST, achieving an average success rate of 87.2% on LIBERO, 65.3% on LIBERO-Long, and more than 1.5× the baseline success rate on ALOHA-sim. Ablation studies validate the contribution of each component, while qualitative and latent-space analyses highlight improved correction of recurrent failure modes.

# II. RELATED WORK

# A. Vision-Language-Action Models

VLA models are commonly trained via behavioral cloning on large-scale human demonstrations, coupling language and vision with end-to-end control [14]. Early systems predominantly used autoregressive, token-based action decoders, recent policies replace discrete heads with diffusion- or flow-matching controllers [40], [41], capturing continuous, multi-modal action distributions and enabling smooth, highfrequency control for dexterous manipulation. Representative systems include Octo and $\pi _ { 0 }$ [2], [3], whose generative action heads are grounded in the modeling literature and applied to visuomotor control [4], [5], [28]. In parallel, OpenVLA provides an open-source generalist policy that scales across embodiments and supports efficient fine-tuning [1], orthogonal advances include frequency-space action tokenization (FAST) for high-rate control [29], BC-Z for zero-shot generalization from language goals [52], and preference alignment via GRAPE [30], DPO [43] and learning from human preferences [44]. Together, these developments situate diffusion- and flow-matching decoders within a broader VLA trajectory [14] and motivate online updates that improve beyond the imitation prior.

# B. Reinforcement Learning for VLA Policies

Policy-gradient fine-tuning of generative policies faces two obstacles: intractable (or costly) likelihoods along generative trajectories and long-horizon credit assignment. For diffusion policies, PPO-style surrogates have been adapted to the denoising process with architecture-aware designs (DPPO) [9]. For flow-matching policies, recent work follows two lines. One line performs reward-weighted supervised updates that avoid explicit likelihoods by biasing training toward high-return samples (RWFM and variants) [36]. The other introduces stochastic relaxations or noise injection to enable sampling-based ratios and on-policy updates (Flow-GRPO [37], ReinFlow [8]). Related gradient estimators for flow models have also been explored from a policygradient perspective [23]. In VLA settings with autoregressive heads, VLA-RL reports procedures for scaling online RL with trajectory-level optimization [6], while preference-based tuning connects to advances in RL for large models [15]–[17]. A persistent difficulty for flow-based actors is that exact policy ratios generally require solving probability-flow ODEs with Jacobian-trace terms [41], rendering likelihoods and ratios expensive or intractable for online control [19], [20]. We address this by constructing a likelihood-free ratio from persample changes in the conditional flow-matching objective and performing clipped PPO-style updates aligned with the model’s generative structure.

# III. METHOD

The proposed FPO is an actor–critic framework that enables online fine-tuning of pretrained conditional flow-matching policies without requiring tractable action likelihoods. The central idea is to reformulate importance sampling by exploiting per-sample changes in the CFM objective as a structure-aligned signal, which is mapped to a likelihoodfree ratio and used within a PPO-style clipped surrogate. To obtain stable and efficient updates, FPO integrates (i) structure-aware credit assignment in the action latent space, (ii) clipped surrogates for trust-region control, (iii) multistep latent (Euler) exploration to produce smooth, temporally correlated perturbations, and (iv) a critic ensemble that supplies robust value estimates. Training alternates between rollout and update (Fig. 1b,c, Algorithm 1): rollout logs transitions and per-sample CFM losses into a small slidingwindow buffer; updates recompute losses under the current actor, map their differences to a clipped surrogate, and train the critic ensemble; the updated actor is then used for the next rollout. Formal details appear in Sec. III-A and Sec. III-B.

# A. FPO Pipeline and Problem Formulation

FPO steers a frozen base policy $\pi _ { 0 } ( a \mid s , x )$ with a flowbased actor $\pi _ { \boldsymbol { \theta } } ( \cdot \mid s )$ operating in the action latent space. Let $x ( u ; s )$ denote the actor’s latent at flow time $u \in [ 0 , 1 ]$ for state s, and write $x _ { t } : = x ( 1 ; s _ { t } )$ for the latent produced at environment step t. A frozen encoder maps observations to $s _ { t } \in \mathbb { R } ^ { d } ;$ ; the actor samples $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { D }$ ; the base decodes $( s _ { t } , x _ { t } )$ to an action $a _ { t } ;$ the environment yields reward $r _ { t }$ and next state $s _ { t + 1 } ~ ( \mathrm { F i g . ~ 1 } )$ . The optimization objective is

(a)Task Episode Timeline   
![](images/9ab5b7dd6d98b23efeb6324175b5555468a20bff1d5c59903a8526bba35b0106.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (b) ROLLOUT PHASE
        A["Observate"] --> B["Approach"] --> C["Align"] --> D["Grasp"] --> E["Finish"]
        F["Observations RGB"] --> G["Frozen Vision Encoder"]
        H["Proprioceptive Sensors"] --> I["Grab something and put it in a container"]
        J["Frozen Base Policy Decoder"] --> K["Robot Action"]
        L["Environment"] --> M["LIBERO&ALOHA TASKS"]
        N["Update"] --> O["Latent action chunk"]
        P["Latent action chunk"] --> Q["FPO Actor π₀"]
        Q --> R["Frozen Vision Encoder"]
        S["Latent action chunk"] --> T["Frozen Base Policy Decoder"]
        U["Latent action chunk"] --> V["Frozen Base Policy Decoder"]
        W["Latent action chunk"] --> X["Frozen Base Policy Decoder"]
        Y["Latent action chunk"] --> Z["Frozen Base Policy Decoder"]
        AA["Latent action chunk"] --> AB["Frozen Base Policy Decoder"]
        AC["Latent action chunk"] --> AD["Frozen Base Policy Decoder"]
        AE["Latent action chunk"] --> AF["Frozen Base Policy Decoder"]
        AG["Latent action chunk"] --> AH["Frozen Base Policy Decoder"]
        AI["Latent action chunk"] --> AJ["Frozen Base Policy Decoder"]
        AK["Latent action chunk"] --> AL["Frozen Base Policy Decoder"]
        AM["Latent action chunk"] --> AN["Frozen Base Policy Decoder"]
        AO["Latent action chunk"] --> AP["Frozen Base Policy Decoder"]
        AQ["Latent action chunk"] --> AR["Frozen Base Policy Decoder"]
        AS["Latent action chunk"] --> AT["Frozen Base Policy Decoder"]
        AU["Latent action chunk"] --> AV["Frozen Base Policy Decoder"]
        AW["Latent action chunk"] --> AX["Frozen Base Policy Decoder"]
        AY["Latent action chunk"] --> AZ["Frozen Base Policy Decoder"]
        BA["Latent action chunk"] --> BB["Frozen Base Policy Decoder"]
        BC["Latent action chunk"] --> BD["Frozen Base Policy Decoder"]
        BE["Latent action chunk"] --> BF["Frozen Base Policy Decoder"]
        BG["Latent action chunk"] --> BH["Frozen Base Policy Decoder"]
        BI["Latent action chunk"] --> BJ["Frozen Base Policy Decoder"]
        BK["Latent action chunk"] --> BL["Frozen Base Policy Decoder"]
        BM["Latent action chunk"] --> BN["Frozen Base Policy Decoder"]
        BO["Latent action chunk"] --> BP["Frozen Base Policy Decoder"]
        BQ["Latent action chunk"] --> BR["Frozen Base Policy Decoder"]
        BS["Latent action chunk"] --> BT["Frozen Base Policy Decoder"]
        BU["Latent action chunk"] --> BV["Frozen Base Policy Decoder"]
        BW["Latent action chunk"] --> BX["Frozen Base Policy Decoder"]
        BY["Latent action chunk"] --> BZ["Frozen Base Policy Decoder"]
        CA["Latent action chunk"] --> CB["Frozen Base Policy Decoder"]
        CC["Latent action chunk"] --> CD["Frozen Base Policy Decoder"]
        DD["Latent action chunk"] --> DE["Frozen Base Policy Decoder"]
        DF["Latent action chunk"] --> DG["Frozen Base Policy Decoder"]
        DH["Latent action chunk"] --> DI["Frozen Base Policy Decoder"]
        DJ["Latent action chunk"] --> DK["Frozen Base Policy Decoder"]
        DL["Latent action chunk"] --> DN["Frozen Base Policy Decoder"]
        DB["Latent action chunk"] --> DC["Frozen Base Policy Decoder"]
        DV["Latent action chunk"] --> DW["Frozen Base Policy Decoder"]
        DX["Latent action chunk"] --> DXB["Frozen Base Policy Decoder"]
        DXB --> DXC["Frozen Base Policy Decoder"]
    end

    subgraph (c) UPDATE PHASE
        AB["Loss Merger"] --> AC["PPO Clipper"]
        AC --> AD["CFM Loss Recomputation"]
        AD --> AE["Input: θ, x₁, s, ||vθ(xₜ, s, t) - (x₁ - ε)||t ~ U(0,1), ℓ_curr, ε ~ N(0,1)"]
        AE --> AF
        AF --> AG["Loss Differential"]
        AG --> AH["Ratio Transform ρ = exp β(δ - μ)/σ Ratio Transform"]
        AH --> AI["minibatches"]
        AI --> AJ["(s₁, x₁, r₁, s₁₋₁)"]
        AJ --> AK["Q-Ensemble Critics Q(s,a) min Qₐ(s,a) → reduced overestimation"]
        AK --> AL["Ahat Advantage Estimates"]
    end

    subgraph (d) Performance Analysis
        AM["Training Dynamics on ALOHA Transfer 65% Success Rate"]
        AN["Latent Action Space Evolution (t-SNE) t=0 Exploration t=50k Convergence t=160k Exploitation t=160k OpenVLA Octo GRAPE πo-FAST FPO (Ours) 53.7% 51.1% 55.8% 60.2% 65.3% LIBERO Benchmark Performacne"]
    end
```
</details>

Fig. 1. Overview of Flow Policy Optimization (FPO). (a) Task episode timeline. (b) Rollout phase: a frozen encoder produces state s, the actor πθ outputs a latent chunk $x _ { 1 } ,$ the frozen base policy $\pi _ { 0 }$ decodes $( s , x _ { 1 } )$ to control a, yielding $( r , s ^ { \prime } )$ . We store the transition and cache the initial CFM loss in a near sliding-window trajectory buffer. (c) Update phase: Batch of trajectories are sampled, the CFM loss is recomputed to form a loss differential, which is mapped to a likelihood-free ratio. A Q-ensemble supplies advantages, and the actor is updated with a clipped surrogate. The updated policy feeds back to rollout. (d) Performance panels: example training curve, latent-space evolution (t-SNE), and LIBERO success rates.

$$
J (\theta) = \mathbb {E} \left[ \sum_ {t = 0} ^ {T} \gamma^ {t} r _ {t} \right], \quad \gamma \in (0, 1). \tag {1}
$$

Because the actor is a conditional flow model, log $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { t } \mid$ $s _ { t } )$ is generally intractable, precluding direct policy-ratio computation.

a) Rollout and Data Recording: Training proceeds in alternating rollout and update phases (Fig. 1b,c; Algorithm 1). During interaction, a frozen rollout copy $\theta _ { \mathrm { o l d } }$ is used to generate experience so that logged quantities remain consistent with the data-collecting policy. At each step, the encoder produces $s _ { t } ,$ the actor samples a latent chunk $x _ { t }$ (with optional short Euler perturbations in latent space for exploration), and the frozen base policy $\pi _ { 0 }$ decodes $( s _ { t } , x _ { t } )$ ) to low-level control $a _ { t }$ that is executed in the environment. The system records $\left( { { s _ { t } } , { x _ { t } } , { a _ { t } } , { r _ { t } } , { s _ { t + 1 } } } \right)$ and caches the persample CFM loss $\ell _ { \mathrm { c f m } } ( x _ { t } \mid s _ { t } ; \theta _ { \mathrm { o l d } } )$ attached to the exact latent used for control. Transitions are stored in a small slidingwindow trajectory buffer that retains only recent rollouts. This design preserves the linkage between cached losses and their originating policy, and bounds the distributional gap between the data-collecting policy and the subsequently updated actor.

b) Update Cycle: During updates, data are drawn from the trajectory buffer and the CFM loss is re-evaluated under the current actor θ on the same $( s _ { t } , x _ { t } )$ pairs. The resulting per-sample loss differential is converted, via batchstandardization and a monotone mapping, into a likelihoodfree ratio proxy that serves as the multiplicative factor in a PPO-style clipped surrogate. Advantages are supplied by a critic ensemble queried in latent space. Actor and critics are optimized for several SGD epochs per interaction batch while continuously evicting older trajectories to keep the training distribution close to recent behavior. Target networks for the critics are updated by Polyak averaging to stabilize bootstrapped targets. After the update cycle, parameters are synchronized by setting $\theta _ { \mathrm { o l d } }  \theta$ before the next interaction phase. This schedule closes the interaction–update loop, maintains a tight coupling between the ratio proxy and the data-collecting policy, and yields steady improvement without requiring tractable action likelihoods.

# B. Structure-Aligned Policy Update and Training Components

a) Likelihood-Free Ratio from CFM Loss: Because $\log \pi _ { \theta } ( x _ { t } \ \mid \ s _ { t } )$ is intractable for flow-based actors, FPO uses the actor’s CFM objective as the update signal. Let $\ell _ { \mathrm { c f m } } ( x _ { t } \mid s _ { t } ; \boldsymbol { \theta } )$ denote the per-sample CFM loss [4], [5]. For each stored pair $( s _ { t } , x _ { t } )$ , the loss reduction is:

![](images/2eaa9685355951fd569b4d65509c376dec7b011237da5c71f4a1036c57b53284.jpg)

<details>
<summary>text_image</summary>

"AlohaTransferCube-v0"
"Aloha
"pick up the black bowl on the ramekin and place it on the plate"
Libero
</details>

![](images/c6f017e1e7558cfcf376edbbc9470c289bd240ad844ca24fc9d983c178b59019.jpg)

<details>
<summary>text_image</summary>

"pick up the orange juice and place it in the basket"
"put both the alphabet soup and the cream cheese box in the basket"
</details>

Fig. 2. An overview of the challenging visuomotor control environments used in our evaluation: the bimanual ALOHA Transfer Cube task and several multi-object manipulation tasks from the LIBERO suite. These environments require a combination of long-horizon reasoning, precise control, and generalization across different objects and initial conditions.

Algorithm 1 Flow Policy Optimization (FPO)   
1: Input: frozen base policy $\pi_{0}$ , actor $\pi_{\theta}$ , critic ensemble $\{Q_{\phi_{i}}\}_{i=1}^{M}$ , target critics $\{Q_{\bar{\phi}_{i}}\}_{i=1}^{M}$ , buffer B
2: for each iteration do
3: $\theta_{old} \leftarrow \theta$ {Freeze actor for loss caching}
4: // Rollout phase
5: for $t = 0 \ldots T_{rollout} - 1$ do
6: observe $s_{t}$ 7: sample latent $x_{t} \sim \pi_{\theta}(\cdot \mid s_{t})$ 8: decode action $a_{t} \sim \pi_{0}(\cdot \mid s_{t}, x_{t})$ 9: step env, obtain $(r_{t}, s_{t+1})$ 10: cache $\ell_{init,t} \leftarrow \ell_{cfm}(x_{t} \mid s_{t}; \theta_{old})$ 11: push $(s_{t}, x_{t}, a_{t}, r_{t}, s_{t+1}, \ell_{init,t})$ into B
12: end for
13: // Update phase
14: for $k = 1 \ldots K_{update}$ do
15: sample batch $M \subset B$ 16: // Critic update (Eqs.5,6)
17: for each $(s_{t}, x_{t}, r_{t}, s_{t+1}) \in M$ :
18: sample $x'_{t+1} \sim \pi_{\theta}(\cdot \mid s_{t+1})$ and set
19: $y_{t} \leftarrow r_{t} + \gamma \min_{i} Q_{\bar{\phi}_{i}}(s_{t+1}, x'_{t+1})$ 20: update $\{\phi_{i}\}$ by minimizing $L_{critic}(\phi)$ ;
21: Polyak update targets $Q_{\bar{\phi}_{i}}$ 22: // Actor update (Eqs.2,3,4)
23: compute $\Delta\ell_{cfm,t} \leftarrow \ell_{cfm}(x_{t} \mid s_{t}; \theta_{old}) \ell_{cfm}(x_{t} \mid s_{t}; \theta)$ 24: standardize $z_{t} \leftarrow standardize(\Delta\ell_{cfm,t})$ 25: map ratio proxy $\rho_{t} \leftarrow \exp(\beta z_{t})$ 26: compute advantages $\hat{A}_{t}$ from the critic ensemble (e.g., GAE)
27: update $\theta$ by minimizing $L_{actor}(\theta)$ 28: end for
29: end for

$$
\Delta \ell_ {\mathrm{cfm}, t} = \ell_ {\mathrm{cfm}} (x _ {t} \mid s _ {t}; \theta_ {\mathrm{old}}) - \ell_ {\mathrm{cfm}} (x _ {t} \mid s _ {t}; \theta) \tag {2}
$$

which measures improvement on the same sample relative to the rollout actor. Under a mild local monotonicity assumption—that per-sample CFM loss decreases coincide with increases in the actor’s conditional density—we treat $\Delta \ell _ { \mathrm { c f m } , t }$ as an order-preserving surrogate of the intractable importance ratio $\pi _ { \boldsymbol { \theta } } ( x _ { t } | \boldsymbol { s } _ { t } ) / \pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } ( x _ { t } | \boldsymbol { s } _ { t } )$ . And the $\Delta \ell _ { \mathrm { c f m } , t }$ is normalized with

$$
z _ {t} = \frac {\Delta \ell_ {\mathrm{cfm} , t} - \mu_ {\Delta}}{\sigma_ {\Delta}}, \quad \rho_ {t} = \exp (\beta z _ {t}) \tag {3}
$$

where $( \mu _ { \Delta } , \sigma _ { \Delta } )$ are mean and standard deviation as batch statistics and $\beta > 0$ controls the sharpness of the mapping.

b) Clipped Surrogate and Actor Update: The actor is optimized with a PPO-style clipped surrogate [12] using advantages ${ \hat { A } } _ { t } \mathbf { : }$

$$
\mathcal {L} _ {\text { actor }} (\theta) = - \mathbb {E} _ {t} \left[ \min \left(\rho_ {t} \hat {A} _ {t}, \operatorname{clip} \left(\rho_ {t}, 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {t}\right) \right] \tag {4}
$$

where $\epsilon > 0$ is the clipping parameter. This construction regulates update magnitude while preserving alignment with the actor’s generative structure.In practice, we standardize $\hat { A } _ { t }$ within each minibatch and stop gradients through $\rho _ { t }$ to reduce variance and avoid feedback instabilities.

c) Critic Ensemble and Advantage Estimation: We employ an ensemble of action–value functions $\{ Q _ { \phi _ { i } } ( s , x ) \} _ { i = 1 } ^ { M }$ to reduce overestimation and stabilize advantage estimates. Target critics $Q _ { \bar { \phi } _ { i } }$ are updated by Polyak averaging once per gradient step. For a transition $\left( { { s _ { t } } , { x _ { t } } , { r _ { t } } , { s _ { t + 1 } } } \right)$ , the temporaldifference target is:

$$
y _ {t} = r _ {t} + \gamma \min _ {i} Q _ {\bar {\phi} _ {i}} (s _ {t + 1}, x _ {t + 1} ^ {\prime}), \quad x _ {t + 1} ^ {\prime} \sim \pi_ {\theta} (\cdot | s _ {t + 1}) \tag {5}
$$

where the operation of minimizing introduces a conservative target that empirically curbs optimistic bias. For terminal $s _ { t + 1 }$ the bootstrap term is masked out. The critic loss is the squared TD error:

$$
\mathcal {L} _ {\text { critic }} (\phi) = \mathbb {E} \left[ \left(Q _ {\phi} (s _ {t}, x _ {t}) - y _ {t}\right) ^ {2} \right] \tag {6}
$$

Advantages are computed with generalized advantage estimation (GAE) [33]. The value baseline $V ( s )$ is taken as a conservative estimate from the ensemble (the minimum across members). We reuse stored latents when available; otherwise a fresh latent is sampled from $\pi _ { \boldsymbol { \theta } } ( \cdot \mid s )$ .

d) Latent-Space Exploration and Data Handling: Exploration is induced by multi-step Euler integration in the actor’s latent dynamics. Starting from a sampled latent $x _ { t } ^ { ( 0 ) } \sim \pi _ { \theta } ( \cdot \mid s _ { t } )$ xt and the CFM velocity field $v _ { \theta } ,$ we apply K short steps:

$$
x _ {t} ^ {(k + 1)} = x _ {t} ^ {(k)} + \eta v _ {\theta} \left(x _ {t} ^ {(k)}, \tau^ {(k)} \mid s _ {t}\right), \quad k = 0, \dots , K - 1 \tag {7}
$$

where $\{ \tau ^ { ( k ) } \}$ is a discretization of the flow time and $\eta > 0$ is a small step size. The final $x _ { t } ^ { ( K ) }$ is decoded by the frozen base. This procedure yields smooth, temporally correlated perturbations that remain aligned with the actor’s generative field. Transitions are stored in a compact sliding-window trajectory buffer B that retains only recent rollouts. Each update draws sample batches from B and performs several SGD epochs while evicting older entries, which limits distributional drift between the update policy and the datacollecting policy and keeps the loss differential $\Delta \ell _ { \mathrm { c f m } , t }$ (Eq. 2) evaluated under a distribution close to behavior, stabilizing the ratio mapping in Eq. 3.

TABLE I   
OVERALL SUCCESS RATE (SR, %) AND PER-SUITE RANK ON LIBERO BENCHMARKS. RANKS ARE COMPUTED AMONG BASELINE METHODS ONLY(EXCLUDING π0-FAST). AVG RANK IS THE MEAN OF PER-SUITE RANKS FOR EACH BASELINE. BEST IN BOLD.

<table><tr><td rowspan="2">Method</td><td colspan="2">LIBERO-Spatial</td><td colspan="2">LIBERO-Object</td><td colspan="2">LIBERO-Goal</td><td colspan="2">LIBERO-Long</td><td colspan="2">Average</td></tr><tr><td>SR (%)</td><td>Rank</td><td>SR (%)</td><td>Rank</td><td>SR (%)</td><td>Rank</td><td>SR (%)</td><td>Rank</td><td>SR (%)</td><td>Avg Rank</td></tr><tr><td>Diffusion Policy [28]</td><td>78.3</td><td>6</td><td>92.5</td><td>2</td><td>68.3</td><td>6</td><td>50.5</td><td>6</td><td>72.4</td><td>5.0</td></tr><tr><td>GRAPE (DPO) [30]</td><td>87.6</td><td>3</td><td>91.2</td><td>4</td><td>82.2</td><td>3</td><td>55.8</td><td>3</td><td>79.2</td><td>3.3</td></tr><tr><td>Octo (SFT) [2]</td><td>78.9</td><td>5</td><td>85.7</td><td>6</td><td>84.6</td><td>2</td><td>51.1</td><td>5</td><td>75.1</td><td>4.5</td></tr><tr><td>OpenVLA (SFT) [1]</td><td>84.7</td><td>4</td><td>88.4</td><td>5</td><td>79.2</td><td>5</td><td>53.7</td><td>4</td><td>76.5</td><td>4.5</td></tr><tr><td>VLA-RL [6]</td><td>90.2</td><td>2</td><td>91.8</td><td>3</td><td>82.2</td><td>3</td><td>59.8</td><td>2</td><td>81.0</td><td>2.5</td></tr><tr><td> $\pi_0$ -FAST [3]</td><td>96.4</td><td>-</td><td>96.8</td><td>-</td><td>88.6</td><td>-</td><td>60.2</td><td>-</td><td>85.5</td><td>-</td></tr><tr><td> $\pi_0$ -FPO (Ours)</td><td>97.2</td><td>1</td><td>97.3</td><td>1</td><td>89.4</td><td>1</td><td>65.3</td><td>1</td><td>87.2</td><td>1</td></tr></table>

(a) Success Rate   
![](images/4906e378d7d049b5673ccb592a87a8a3d55b8610546dacc81645f6d0b3dc70d2.jpg)

<details>
<summary>line</summary>

| Training Steps (K) | Smoothed | SFT Baseline | Raw Data |
| ------------------ | -------- | ------------ | -------- |
| 0                  | 10       | 60           | 10       |
| 50                 | 25       | 60           | 30       |
| 100                | 20       | 60           | 40       |
| 150                | 35       | 60           | 50       |
| 200                | 40       | 60           | 60       |
| 250                | 55       | 60           | 70       |
| 300                | 50       | 60           | 65       |
| 350                | 60       | 60           | 90       |
| 400                | 45       | 60           | 80       |
| 450                | 65       | 60           | 85       |
| 500                | 65       | 60           | 85       |
</details>

(b) Average Return   
![](images/727361aac462b13072a18610c0ab25d0489ec888cf2b6d6c36ac16270419922c.jpg)

<details>
<summary>line</summary>

| Training Steps (K) | Smoothed | Raw Data |
| ------------------ | -------- | -------- |
| 0                  | 0.0      | 0.1      |
| 50                 | 0.2      | 0.3      |
| 100                | 0.4      | 0.4      |
| 150                | 0.4      | 0.5      |
| 200                | 0.5      | 0.4      |
| 250                | 0.6      | 0.5      |
| 300                | 0.5      | 0.4      |
| 350                | 0.6      | 0.5      |
| 400                | 0.4      | 0.3      |
| 450                | 0.6      | 0.5      |
| 500                | 0.6      | 0.6      |
</details>

(c) Episode Length   
![](images/8897efc03cd5bfb3a46fb9dccc97c32f8450b7c8d5a99992fab267dd2b4eee0b.jpg)

<details>
<summary>line</summary>

| Training Steps (K) | Smoothed | Raw Data |
| ------------------ | -------- | -------- |
| 0                  | 400      | 400      |
| 50                 | 380      | 360      |
| 100                | 370      | 380      |
| 150                | 360      | 370      |
| 200                | 350      | 360      |
| 250                | 340      | 350      |
| 300                | 330      | 340      |
| 350                | 320      | 330      |
| 400                | 310      | 320      |
| 450                | 300      | 310      |
| 500                | 290      | 300      |
</details>

Fig. 3. LIBERO-Long simulation: online fine-tuning curves on a representative task.

# IV. EXPERIMENTS

In this section, we empirically evaluate FPO along three axes: (i) final performance on standard manipulation benchmarks relative to strong baselines. (ii) learning dynamics under online interaction (improvement curves and exploration behavior). and (iii) ablations that isolate the contribution of each component.

# A. Experimental Setup

a) Tasks and Evaluation: The proposed FPO algorithm was evaluated on two simulated visuomotor benchmarks: LIBERO [21] and ALOHA Transfer Cube [22] (as shown in Fig. 2). LIBERO [21] comprises four sub-suites—Spatial, Object, Goal, and LIBERO-Long. ALOHA Transfer Cube [22] is a bimanual manipulation task with contact-rich dynamics. We follow the official success criteria and report per-suite success rate (SR, %).   
b) Baselines and Protocol: We compare against π0- FAST [29], GRAPE [30], Diffusion Policy [28], Open-VLA [1], Octo [2], and VLA-RL [6], covering supervised

steering of $\pi _ { 0 } ,$ preference alignment, diffusion-based control, large-scale SFT VLAs, and online RL with autoregressive heads. Evaluations follow the official success metrics and protocols. We use public checkpoints when available, otherwise authors’ reference implementations with reported settings. Task definitions, observation/action interfaces, and evaluation seeds are matched across methods. Our runs initialize from the released $\pi _ { 0 }$ checkpoint, keep the $\pi _ { 0 }$ decoder frozen, and update only the flow actor and an ensemble critic online.

# B. Performance Evaluation and Analysis

a) Performance Advantages of FPO on the LIBERO Benchmark: FPO (denoted as $\pi _ { 0 } { \bf - F P 0 ) }$ achieves state-of-theart performance across all four task suites of the LIBERO benchmark, as shown in Table I. It attains suite-leading success rates with an overall average of 87.2%, outperforming all baseline methods. On the LIBERO-Long suite, FPO attains 65.3% SR. This corresponds to improvements of +5.5 percentage points over the RL baseline VLA-RL (59.8%), +9.5 over GRAPE (55.8%), and +5.1 over $\pi _ { 0 } { \mathrm { - F A S T } }$ (60.2%), as reported in Table I. These comparisons situate FPO ahead of both online RL and offline-trained baselines under the same evaluation protocol.

This improvement indicates that even foundation models trained on large-scale offline datasets retain unused capacity. FPO leverages an online, reward-driven update mechanism to exploit this capacity, addressing the lack of adaptability and exploration inherent to purely offline training. These results demonstrate that FPO can effectively learn from sparse rewards and refine goal-directed behaviors, extending performance beyond the limits of imitation-based methods.

![](images/46fd06907fd2521afc9b7b393dacf039eafda9fc7e65ca342437bef71ea42f47.jpg)

<details>
<summary>line</summary>

| Training Steps | Approach | Grasp | Outcome |
| -------------- | -------- | ----- | ------- |
| 1.6e6          |          |       |         |
| 0.8e6          |          |       |         |
| 0              |          |       |         |
</details>

(a)Policy Evolution

![](images/762ac5e6861a974b3fcd6a2d29de5187027ac50860df02816e60f777d30ce793.jpg)

<details>
<summary>line</summary>

| Training Steps | Smoothed Curve | Baseline (0.4) | Raw Data |
| -------------- | -------------- | -------------- | -------- |
| 0.0            | 0.35           | 0.4            | 0.3      |
| 0.2            | 0.3            | 0.4            | 0.4      |
| 0.4            | 0.35           | 0.4            | 0.5      |
| 0.6            | 0.3            | 0.4            | 0.6      |
| 0.8            | 0.4            | 0.4            | 0.7      |
| 1.0            | 0.5            | 0.4            | 0.6      |
| 1.2            | 0.55           | 0.4            | 0.7      |
| 1.4            | 0.6            | 0.4            | 0.8      |
| 1.6            | 0.65           | 0.4            | 0.9      |
</details>

(b） Result of Learning  
Fig. 4. FPO online learning on the ALOHA Transfer Cube task. (a) Policy evolution at 0/0.8M/1.6M training steps: the baseline side-grasp failure mode is corrected to a robust top-down grasp that consistently completes the task. (b) Success rate (SR) curve: the smoothed trajectory (purple) steadily improves, surpassing the 40% baseline (red dashed) and reaching 65%, mirroring the behavioural change in (a).

b) FPO Enables Stable and Efficient Online Learning: To further assess the effectiveness of FPO in improving model performance through online reinforcement learning, we examine its learning dynamics on the LIBERO benchmark and the ALOHA Transfer Cube task. Starting from an SFT baseline, FPO consistently increases both success rate and average return throughout training, as shown in Fig. 3. The return curve exhibits a similar upward trend, while episode length remains steady downward trend, indicating that gains arise from discovering more direct and efficient strategies rather than extending trial duration.

Comparable behavior is observed in the ALOHA Transfer Cube task (Fig. 4). Beginning with the $\pi _ { 0 }$ model at ∼ 40% success, FPO exceeds 65% after a comparable training budget. The smoothed trajectory improves monotonically and avoids the instability typically seen in online RL under sparse rewards. Together, these results indicate stable online learning across distinct manipulation domains, supporting FPO as an effective fine-tuning framework for VLA policies.

# C. Analysis of FPO’s Internal Learning Mechanism

To analyze how FPO achieves its performance, we examined the evolution of its internal behavior during training by visualizing the distribution of latent action chunks across different stages, shown as Fig. 5. Using t-SNE for dimensionality reduction, the results reveal a clear trajectory from broad exploration to focused exploitation in the latent space.

In the initial phase (Fig. 5(a)), the policy, guided by the imitation prior, explores a wide region of the latent space, enabling the discovery of rewarding areas beyond the baseline policy. During the breakthrough phase (Fig. 5(b)), exploration becomes more structured and concentrated around successful action sequences, indicating prioritization of high-value regions. At convergence period (Fig. 5(c)), the distribution narrows into a low-variance cluster, reflecting efficient exploitation of the optimal region. The bar chart in Fig. 5(d) quantifies this transition, showing a marked reduction in dispersion and variance over training. These results demonstrate that FPO supports gradient-driven exploration beyond imitation priors and enables stable convergence to efficient task-solving behaviors.

![](images/abe52eeeda6709fb095b60b09f52f9d36a4f34e00d3c46e2034355952231b4e3.jpg)

<details>
<summary>contour</summary>

| t-SNE Component 1 | t-SNE Component 2 |
| ----------------- | ----------------- |
| -100              | 0                 |
| -50               | 0                 |
| 0                 | 0                 |
| 50                | 0                 |
| 100               | 0                 |
</details>

![](images/f008e58e4d36301d8e6d68e3f4ab68ab660135b59a7669df739c9891171dbe6d.jpg)

<details>
<summary>contour</summary>

| t-SNE Component 1 | t-SNE Component 2 | Density |
| ----------------- | ----------------- | ------- |
| -100              | 100               | Low     |
| -50               | 50                | Medium  |
| 0                 | 0                 | High    |
| 50                | -50               | Medium  |
| 100               | -100              | Low     |
</details>

12 10 8 4
6
8
10 6 24   
![](images/aea04635b33db38d3fc593ab5698c136bd474c49555f94d0574a59deb6eb0897.jpg)

<details>
<summary>contour</summary>

| t-SNE Component 1 | t-SNE Component 2 | Point Density |
| ----------------- | ----------------- | ------------- |
| -100              | -100              | 2             |
| -50               | -50               | 4             |
| 0                 | 0                 | 6             |
| 50                | 50                | 8             |
| 100               | 100               | 10            |
</details>

![](images/2c1bf0dddbd83f8b74a084a84c4e26266f52c055769f93bf9d667962ceb64fc6.jpg)

<details>
<summary>bar</summary>

(d) Exploration range comparison
| Training methods | Exploring range (standard deviation mean) | Conting area (trained ball) | Overall dispersion |
|---|---|---|---|
| Extended Baseline (0-10K) | 52.56 | 39400 | 52.6 |
| FPO Breakthrough (80-100K) | 43.62 | 38630 | 43.6 |
| FPO Late-Training Phase (130K+) | 40.52 | 37077 | 40.5 |
</details>

Fig. 5. FPO latent action space evolution. Visualized via t-SNE, this figure shows the policy’s latent action distribution transitioning from broad exploration to focused exploitation across training stages. (a) Initial policy: wide, high-variance exploration. (b) Breakthrough phase: distribution concentrates around successful sequences. (c) Late-Training Phase: highly focused, low-variance exploitation of optimal regions. (d) Bar chart: quantifies reduced exploration range and dispersion, confirming convergence to refined behaviors.

In addition, the Fig. 6 visually demonstrates FPO’s ability to resolve specific, recurring failure modes. The pre-trained $\pi _ { 0 }$ policy often fails in a representative LIBERO task by attempting a suboptimal side grasp, leading to object instability (Fig. 6, top). FPO, after online fine-tuning, fundamentally alters this approach, consistently executing a robust top-down grasp from the same initial state that previously led to failure (Fig. 6, bottom). This illustrates FPO’s capacity to learn physically grounded, effective trajectories through active online interaction, fixing nuanced, contact-rich errors that are challenging for offline methods alone.

![](images/e8eb95940868f93f668eed6143c6936161dd0cc2d48f350f37fc9588e20fec58.jpg)

<details>
<summary>natural_image</summary>

3D rendering of a robotic arm in motion, showing multiple frames and control panels (no text or symbols visible)
</details>

![](images/e7a8c8004ae829e0124bbc6edac23d6ffaba41c3421150888d874a9c5ec73173.jpg)

<details>
<summary>text_image</summary>

Ours
</details>

Fig. 6. FPO’s ability to correct suboptimal behaviors. (Top) The SFT baseline policy consistently fails the task due to a suboptimal grasping approach inherited from the imitation prior. (Bottom) After online finetuning with FPO, the policy discovers a novel and successful trajectory from the same initial state, showcasing effective online correction.

# D. Ablation Studies

To assess which components drive FPO’s performance, we conduct ablations on the LIBERO-90 task pick up the butter and put it in the tray. Table II reports final success rate (%) after training. Each variant disables exactly one component while keeping the training budget, architectures, and hyperparameters fixed. We consider four interventions: substituting the CFM-based ratio with an SAC-style latent-space update, removing PPO-style clipping, reducing exploration to singlestep integration, and replacing the critic ensemble with a single critic. All ablations degrade performance relative to the complete method: replacing the CFM-based ratio causes the most substantial drop, removing clipping also leads to a marked reduction, limiting exploration depth yields a smaller but consistent decline, and using a single critic has the least impact yet remains non-negligible. Taken together, these results indicate that all components are consequential—the structure-aligned ratio and trust-region control account for a large portion of the gains, while exploration depth and value ensembling contribute additional stability and data efficiency.

TABLE II ABLATIONS ON LIBERO-90 TASK pick up the butter and put it in the tray. NUMBERS ARE FINAL SUCCESS RATE (%). EACH ROW REMOVES ONE COMPONENT FROM FPO. 

<table><tr><td>Method</td><td>Success Rate (%)</td></tr><tr><td>FPO (complete)</td><td>78.5</td></tr><tr><td>- without CFM ratio proxy</td><td>32.4</td></tr><tr><td>- without PPO clipping</td><td>45.1</td></tr><tr><td>- single-step integration (K=1)</td><td>61.7</td></tr><tr><td>- single critic (without ensemble)</td><td>71.2</td></tr></table>

a) Importance of the CFM-based Ratio Proxy.: Replacing the CFM-based policy ratio proxy with standard Soft Actor-Critic (SAC) in the latent space significantly reduced the success rate from 78.5% to 32.4%. This indicates that FPO’s performance depends critically on the structurally-aware update rule that leverages the generative loss. Conventional RL in latent space is insufficient to achieve comparable results.   
b) Effect of PPO-style Clipping.: Removing PPO-style clipping caused instability and reduced success to 45.1% (- 33.4%). This confirms the role of clipping as a trust-region mechanism [12], preventing uncontrolled updates from the strong CFM signal and avoiding policy collapse.   
c) Role of Multi-step Exploration.: Disabling multi-step Euler integration (K = 1) lowered success to 61.7% (-16.8%). This shows the benefit of generating temporally correlated latent trajectories, which improve the discovery of viable action sequences and enable stable execution in contact-rich tasks.   
d) Contribution of the Q-Ensemble.: Using a single critic instead of a Q-ensemble reduced success to 71.2%. Although the effect is smaller (-7.2%), the ensemble improves stability by providing more reliable advantage estimates, which is particularly useful in sparse-reward settings.

# E. Latent Space Characteristics of Successful Policies

To examine latent space characteristics distinguishing successful from failed policy executions, we conducted a statistical analysis of latent action chunks from the untrained imitation prior, failed rollouts after partial training, and successful rollouts after extended training. Fig. 7 summarizes the statistical differences across these groups. The t-SNE and PCA projections (Fig. 7a,b) show that successful trajectories occupy a compact and well-defined region of the latent space, whereas actions from the initial policy and failed rollouts are more dispersed. This separation indicates that FPO guides the policy toward a higher-performing latent subspace.

Analysis of action magnitudes (Fig. 7c) reveals that successful trajectories fall within a narrower range, suggesting avoidance of extreme actions and convergence toward an effective magnitude. The per-dimension variance plot (Fig. 7d) further shows reduced variance across most dimensions, confirming that successful policies act with greater precision by suppressing unnecessary exploratory noise and focusing capacity on reliable execution. This transition reflects effective skill acquisition in the latent space.

Quantitatively, successful rollouts exhibit higher silhouette scores and larger Mahalanobis distance to the success centroid, alongside lower within-cluster variance and reduced first-order temporal differences, indicating concentration of probability mass in a stable high-value latent region.

![](images/733b5c69d9e0976ab01e8675b7c7baa9e08df48939fd917586b3483485ed986b.jpg)  
Fig. 7. Latent space analysis on initial policy, failed and successful Trajectories. The t-SNE and PCA plots (top row) demonstrate that successful trajectories (green) converge to a distinct, highly structured region of the latent space, sharply contrasting with the diffuse distributions of the initial policy (blue) and failed attempts (red). The action magnitude distribution (bottom-left) shows successful trajectories favoring a narrower, optimal range of action norms. Critically, the per-dimension variance plot (bottomright) reveals significantly lower variance across most latent dimensions for successful trajectories, indicating learned precision and intentionality.

# V. CONCLUSION

This work introduced a FPO algorithm for online finetuning of flow-matching VLA policies. FPO resolves the incompatibility with conventional policy-gradient methods by replacing explicit likelihood ratios with a likelihood-free proxy derived from per-sample changes in the conditional flow-matching objective, thereby enabling PPO-style clipped updates without Jacobian or density evaluation. The method incorporates structure-aware credit assignment in the latent space, a clipped surrogate objective, multi-step latent exploration, and a Q-ensemble, enabling stable and efficient optimization in sparse-reward and contact-rich environments. Experiments on the LIBERO benchmark and the ALOHA Transfer Cube task demonstrate that $\pi _ { 0 } { \mathrm { - F P O } }$ consistently outperforms imitation-trained priors and strong baselines, including OpenVLA, Octo, Diffusion Policy, GRAPE, and $\pi _ { 0 } { \mathrm { - F A S T } }$ . Ablation experiments and latent space dynamics analysis not only confirm the efficacy of individual FPO components, but also demonstrate enhanced mitigation of recurring failure patterns through qualitative assessment. In the future, we will further enhance the few-shot adaptation capability based on limited online interactions, enabling faster learning and transfer while minimizing additional data requirements.

# ACKNOWLEDGMENT

This study is supported by the Strategic Priority Research Program of the Chinese Academy of Sciences (Grant No. XDB1010302), the State Key Laboratory of Brain Cognition and Brain-inspired Intelligence Technology (Grant No. JS202401), the funding from Institute of Automation, Chinese Academy of Sciences (Grant No. E411230101), the Postdoctoral Fellowship Program of CPSF (Grant No. GZC20232994), and the Beijing Natural Science Foundation (Grant No. 4252052).

# REFERENCES

[1] A. S. Xie et al., “OpenVLA: An Open-Source Vision-Language-Action Policy,” arXiv:2406.09246, 2024.   
[2] Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al., “Octo: An Open-Source Generalist Robot Policy,” arXiv:2405.12213, 2024.   
[3] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al., “π0: A Vision-Language-Action Flow Model for General Robot Control,” arXiv:2410.24164, 2024.   
[4] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow Matching for Generative Modeling,” arXiv:2210.02747, 2022.   
[5] A. Tong, E. Chatzipantazis, F. Lindsten, J. Umenberger, M. Zwicker, A. Taghvaei, and C. J. Maddison, “Improving and Generalizing Flow-based Generative Models with Conditional Flow Matching,” arXiv:2501.03192, 2025.   
[6] G. Lu, W. Guo, C. Zhang, Y. Zhou, H. Jiang, Z. Gao, Y. Tang, and Z. Wang, “VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning,” arXiv:2505.18719, 2025.   
[7] Y. Guo, J. Zhang, X. Chen, X. Ji, Y.-J. Wang, Y. Hu, and J. Chen, “Improving Vision-Language-Action Model with Online Reinforcement Learning,” ICRA, 2025.arXiv:2501.16664   
[8] T. Zhang, C. Yu, S. Su, and Y. Wang, “ReinFlow: Finetuning Flow Matching Policy with Online Reinforcement Learning,” arXiv:2505.22094, 2025.   
[9] A. Z. Ren, J. Lidard, L. L. Ankile, A. Simeonov, P. Agrawal, A. Majumdar, B. Burchfiel, H. Dai, and M. Simchowitz, “Diffusion Policy Policy Optimization,” arXiv:2409.00588, 2024.   
[10] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Dream to Control: Learning Behaviors by Latent Imagination,” arXiv:1912.01603, 2019.   
[11] A. Yarats, R. Ferm, D. Hafner, and L. Zisserman, “DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations,” arXiv:2110.14565, 2021.   
[12] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” arXiv:1707.06347, 2017.   
[13] L. Chen, R. Paleja, and M. Gombolay, “Learning from Suboptimal Demonstration via Self-Supervised Reward Regression,” arXiv:2010.11723, 2020.   
[14] M. U. D. Waseem Akram, L. S. Saoud, J. Rosell, and I. Hussain, “Vision Language Action Models in Robotic Manipulation: A Systematic Review,” arXiv:2507.10672, 2025.   
[15] L. Ouyang et al., “Training Language Models to Follow Instructions with Human Feedback,” arXiv:2203.02155, 2022.

[16] Y. Bai et al., “Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback,” arXiv:2204.05862, 2022.   
[17] N. Stiennon et al., “Learning to Summarize from Human Feedback,” NeurIPS, 2020.arXiv:2009.01325   
[18] J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel, “Trust Region Policy Optimization,” Proceedings of the 32nd International Conference on Machine Learning (ICML), PMLR 37:1889–1897, 2015.   
[19] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, “Neural Ordinary Differential Equations,” in Advances in Neural Information Processing Systems (NeurIPS), 2018.   
[20] W. Grathwohl, R. T. Q. Chen, J. Bettencourt, I. Sutskever, and D. Duvenaud, “FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models,” arXiv:1810.01367, 2018.   
[21] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning,” in NeurIPS Datasets & Benchmarks Track, 2023.   
[22] T. Z. Zhao, J. Tompson, D. Driess, P. Florence, K. Ghasemipour, C. Finn, and A. Wahid, “ALOHA Unleashed: A Simple Recipe for Robot Dexterity,” arXiv:2410.13126, 2024.   
[23] David McAllister et al., “Flow Matching Policy Gradients,” arXiv:2507.21053, 2025.   
[24] A. Wagenmaker , “Steering Your Diffusion Policy with Latent Space Reinforcement Learning,” in CoRL, 2025.   
[25] S. H. Lee, M. Salimans, Y. Guo, et al., “Stochastic Latent Actor-Critic: Deep RL in the Model’s Learned Latent Space,” in NeurIPS, 2020.   
[26] H. Wang, S. Lin, and J. Zhang, “Adaptive Ensemble Q-learning: Minimizing Estimation Bias via Error Feedback,” in NeurIPS, 2021.   
[27] A. An , “Q-Ensemble for Offline RL: Don’t Scale the Ensemble,” in NeurIPS Workshop on Offline RL, 2022.   
[28] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,” in Robotics: Science and Systems (RSS), 2023.   
[29] K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine, “FAST: Efficient Action Tokenization for Vision-Language-Action Models,” arXiv:2501.09747, 2025.   
[30] Z. Zhang, K. Zheng, Z. Chen, J. Jang, Y. Li, S. Han, C. Wang, M. Ding, D. Fox, and H. Yao, “GRAPE: Generalizing Robot Policy via Preference Alignment,” arXiv:2411.19309, 2024.   
[31] Y. Wang, H. He, C. Wen, X. Tan,et al., “Truly Proximal Policy Optimization,” in UAI (PMLR), 2020.   
[32] G. Chen, Y. Peng, and M. Zhang, “An Adaptive Clipping Approach for Proximal Policy Optimization,” arXiv:1804.06461, 2018.   
[33] J. Schulman, P. Moritz, S. Levine, M. I. Jordan, and P. Abbeel, “High-Dimensional Continuous Control Using Generalized Advantage Estimation,” arXiv:1506.02438, 2015.   
[34] R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, and M. G. Bellemare, “Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress,” in NeurIPS 2022, pp. 28955–28971.   
[35] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,” in Proceedings of the 35th International Conference on Machine Learning (ICML), 2018.   
[36] S. Pfrommer, Y. Huang, and S. Sojoudi, “Reinforcement Learning for Flow-Matching Policies,” arXiv:2507.15073, 2025.   
[37] J. Liu, G. Liu, J. Liang, et al., “Flow-GRPO: Training Flow Matching Models via Online RL,” arXiv:2505.05470, 2025. First work integrating GRPO into flow matching via ODE-to-SDE conversion   
[38] A. Wagenmaker, M. Nakamoto, Y. Zhang, S. Park, W. Yagoub, A. Nagabandi, A. Gupta, and S. Levine, “Steering Your Diffusion Policy with Latent Space Reinforcement Learning (DSRL),” CoRL, 2025. Enables sample-efficient steering of diffusion policies via RL over latent noise   
[39] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-Rank Adaptation of Large Language Models,” in ICLR, 2022. Introduced LoRA for parameter-efficient fine-tuning in large pretrained models.   
[40] J. Ho and T. Salimans, “Denoising Diffusion Probabilistic Models,” in Advances in Neural Information Processing Systems (NeurIPS), 2020.   
[41] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-Based Generative Modeling through Stochastic Differential Equations,” in International Conference on Learning Representations (ICLR), 2021.

[42] Open X-Embodiment Collaboration, “Open X-Embodiment: Robotic Learning Datasets and RT-X Models,” arXiv:2310.08864, 2023.   
[43] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct Preference Optimization: Your Language Model is Secretly a Reward Model,” in Advances in Neural Information Processing Systems (NeurIPS), 2023.   
[44] P. F. Christiano, J. Leike, T. B. Brown, M. Martic, S. Legg, and D. Amodei, “Deep Reinforcement Learning from Human Preferences,” arXiv:1706.03741, 2017.   
[45] B. T. Polyak and A. B. Juditsky, “Acceleration of Stochastic Approximation by Averaging,” SIAM Journal on Control and Optimization, 30(4):838–855, 1992.   
[46] T. P. Lillicrap, J. J. Hunt, A. Pritzel, et al., “Continuous Control with Deep Reinforcement Learning,” arXiv:1509.02971, 2015.   
[47] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing Function Approximation Error in Actor-Critic Methods,” in Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80:1587–1596, 2018.   
[48] I. Osband, C. Blundell, A. Pritzel, and B. Van Roy, “Deep Exploration via Bootstrapped DQN,” in Advances in Neural Information Processing Systems (NeurIPS), 2016.   
[49] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles,” in Advances in Neural Information Processing Systems (NeurIPS), 2017.   
[50] A. Brohan et al., “RT-1: Robotics Transformer for Real-World Control at Scale,” arXiv:2212.06817, 2022.   
[51] A. Brohan et al., “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control,” arXiv:2307.15818, 2023.   
[52] E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn, “BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning,” in CoRL, 2022.