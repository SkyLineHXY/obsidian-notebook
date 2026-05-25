# DECOUPLED Q-CHUNKING

# Qiyang Li

UC Berkeley

qcli@berkeley.edu

# Seohong Park

UC Berkeley

seohong@berkeley.edu

# Sergey Levine

UC Berkeley

svlevine@berkeley.edu

# ABSTRACT

Temporal-difference (TD) methods learn state and action values efficiently by bootstrapping from their own future value predictions, but such a self-bootstrapping mechanism is prone to bootstrapping bias, where the errors in the value targets accumulate across steps and result in biased value estimates. Recent work has proposed to use chunked critics, which estimate the value of short action sequences (“chunks”) rather than individual actions, speeding up value backup. However, extracting policies from chunked critics is challenging: policies must output the entire action chunk open-loop, which can be sub-optimal for environments that require policy reactivity and also challenging to model especially when the chunk length grows. Our key insight is to decouple the chunk length of the critic from that of the policy, allowing the policy to operate over shorter action chunks. We propose a novel algorithm that achieves this by optimizing the policy against a distilled critic for partial action chunks, constructed by optimistically backing up from the original chunked critic to approximate the maximum value achievable when a partial action chunk is extended to a complete one. This design retains the benefits of multi-step value propagation while sidestepping both the open-loop sub-optimality and the difficulty of learning action chunking policies for long action chunks. We evaluate our method on challenging, long-horizon offline goalconditioned tasks and show that it reliably outperforms prior methods.

Code: github.com/ColinQiyangLi/dqc.

![](images/c7c8ffc6eddb79b0e766cf855df50a954aca8f4693bbc426c8bc9e21cfd2bb04.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["n-step value target V(□)"] --> B["backup"]
    B --> C["q state action chunk"]
    C --> D["distill"]
    D --> E["q partial chunk"]
    E --> F["extract"]
    F --> G["π(□)"]
    
    H["large critic chunk"] --> I["Actions"]
    I --> J["full chunk"]
    J --> K["States"]
    K --> L["... Time ..."]
    L --> M["Actions"]
    M --> N["partial chunk"]
    N --> O["Decoupled Chunking"]
    O --> P["small policy chunk"]
    
    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style E fill:#ccf,stroke:#333
    style F fill:#ccf,stroke:#333
    style G fill:#ccf,stroke:#333
    style H fill:#dfd,stroke:#333
    style I fill:#dfd,stroke:#333
    style J fill:#dfd,stroke:#333
    style K fill:#dfd,stroke:#333
    style L fill:#dfd,stroke:#333
    style M fill:#dfd,stroke:#333
    style N fill:#dfd,stroke:#333
    style O fill:#dfd,stroke:#333
```
</details>

![](images/54e236554276a709044809432cc8fdc9f70156103a67c56755098c25e13ff0ab.jpg)

<details>
<summary>bar</summary>

| Category | Aggregated Score |
| -------- | ---------------- |
| DQC      | 82               |
| QC       | 25               |
| NS       | 68               |
| SHARSA   | 44               |
| HIQL     | 18               |
</details>

Figure 1: Decoupled Q-chunking (DQC). Left: The key idea of our method is to ‘decouple’ the action chunk size of the critic Q from that of the policy π. A large critic chunk size allows for efficient value learning while a small policy chunk size makes policy learning more tractable and allows for better policy reactivity. Right: Our method outperforms all baselines on six hardest environments on OGBench, an offline goal-conditioned RL benchmark with challenging long-horizon tasks.

# 1 INTRODUCTION

Temporal-difference (TD) methods are powerful reinforcement learning (RL) techniques that can directly learn from off-policy prior data without requiring an explicit dynamics model, making them well-suited for offline RL (Levine et al., 2020) and sample-efficient online RL (Chen et al., 2021).

Despite their successes, a key challenge remains: bootstrapping bias (Jaakkola et al., 1993; Sutton et al., 1998; De Asis et al., 2018; Park et al., 2025b). This bias stems from the core design of TD updates, where the value at the current state is learned by regressing towards the learner’s own predictions at the next time step. As a result, any prediction error is compounded backward across steps, making learning particularly challenging in long-horizon, sparse-reward tasks.

Multi-step return backups (Sutton et al., 1998) can alleviate bootstrapping bias by shifting the regression target further into the future and effectively reducing the time horizon. However, naïvely applying them introduces additional biases because computing the target involves summing rewards along off-policy trajectories that may deviate from the actions that the agent would take. Although importance sampling can in principle correct such off-policy biases by reweighting the off-policy trajectories (Munos et al., 2016), it often suffers from high variance and thus requires truncation and other heuristics for numerical stability, making it difficult to tune in practice. Recent works (Seo & Abbeel, 2025; Li et al., 2025a; Tian et al., 2025; Li et al., 2025b) leverage chunked value functions, which estimate the value of short action sequences (“chunks”) rather than a single action. This formulation allows n-step return backup without the pessimistic bias (under some condition we formalize in Section 4). However, theoretical guarantees of action chunking Q-learning, especially on arbitrary off-policy data, are still an open problem as existing analysis (e.g., in Li et al. (2025b)) only considers the case where the data is collected by an action chunking policy. Moreover, on the empirical side, directly optimizing a policy over full action chunks is difficult, particularly as the chunk size grows, and it is still unclear how to best extract a policy from a chunked critic.

In this work, we lay the theoretical foundation of action chunking Q-learning where we identify the key open-loop consistency condition (Definition 2) under which Q-learning with action chunking critic is guaranteed to produce a near-optimal action chunking policy. On top of it, we characterize the condition when closed-loop execution (i.e., only executing the first action in the predicted action chunk) of such action chunking policy is expected to be even close to the optimal closed-loop policy. Motivated by our analysis, we develop a simple practical algorithm that builds on top of the idea of closed-loop execution of action chunking policies to address the action chunking policy learning challenge. The key insight is that we can avoid training the policy to predict the full action chunks and instead to only predict shorter, partial action chunks against the chunked critic. To achieve this, we use a ‘distilled’ chunked critic with a chunk size that matches the policy: it optimistically regresses to the original chunked critic to approximate the maximum value that the partial action chunk can achieve after being extended into a full action chunk. Conceptually, while the action optimization is still done for the longer, complete action chunks, the policy network is only trained to output the partial action chunk of an optimized complete action chunk. This way, the policy only needs to predict a much shorter action chunk (e.g., in the extreme case, only one action), which often admits a much simpler distribution, while enjoying the value learning benefits from the use of chunked critics.

Our main contributions are two-fold. On the theoretical side, we provide the first formal analysis of Q-learning with action chunking, focusing on characterizing the value learning bias of the bellman backup of action chunking critic. Specifically, we introduce the open-loop consistency condition under which we exactly characterize the worst-case value estimation bias (Theorems 1 and 2) and suboptimality gap (Theorems 3 and 4) at the fixed point of the bellman optimality equations. Moreover, we characterize (i) the conditions under which action chunking critic backup is preferable over n-step return backup with a single-step critic (Proposition 2), and (ii) the conditions under which closedloop execution of the action chunking policy further mitigates the open-loop bias (Theorem 5). On the empirical side, we propose a new technique, Decoupled Q-chunking (DQC), that addresses the policy learning challenge in action chunking Q-learning by decoupling the policy chunk size from the critic chunk size. DQC trains a policy to only predict a partial action chunk, significantly reducing the policy learning challenge, while retaining the value learning benefits of the chunked critic. We instantiate this technique as a practical offline RL algorithm that outperforms the previous state-ofthe-art method on the hardest set of environments in OGBench (Park et al., 2025a), a challenging, long-horizon goal-conditioned RL benchmark.

# 2 RELATED WORK

Theory of action chunking. Existing analyses for action chunking focus exclusively on the imitation learning setting (Tu et al., 2022; Simchowitz et al., 2025). While they laid out the theoretical foundation of action chunking policies for imitation learning, formal guarantees of action chunking RL are still an open problem. In the adjacent field of stochastic optimal control (SOC), action chunking is related to control under intermittent observations where the observation inputs to the controller are either unreliable (e.g., with a Poissonian model (Wang, 2001; Dupuis & Wang, 2002)), or partially missing (Mishra et al., 2020; Yan et al., 2022; Noba & Yamazaki, 2022; Bayer et al., 2024). While conceptually related, these analyses are in the continuous-time setting in contrast to discrete-time transitions. To the best of our knowledge, we are the first to provide a formal analysis of action chunking in Q-learning. In particular, we identify the key open-loop consistency condition under which we quantify the exact worst-case Q-learning sub-optimality.

Offline and offline-to-online reinforcement learning methods assume access to an offline dataset to learn a policy without interactions with the environment (offline) (Kumar et al., 2020; Kostrikov et al., 2022; Tarasov et al., 2024) or with as little online interaction with the environment as possible (offlineto-online) (Lee et al., 2022; Ball et al., 2023; Nakamoto et al., 2024). Q-learning or TD-based RL algorithms have been a popular choice for these problem settings as they naturally handle off-policy data without the need for on-policy rollouts, and also exhibit great online sample-efficiency (Chen et al., 2021; D’Oro et al., 2023). A large body of literature in these two problem settings has been focusing on tackling the distribution shift challenge by appropriately constraining the policies with respect to the prior offline data, and most of them use the standard 1-step TD backup for Q-learning, which has been known to suffer from the bootstrapping bias problem in the RL literature (Jaakkola et al., 1993; Sutton et al., 1998). To tackle this, recent work (Jeong et al., 2023; Park & Lee, 2025; Park et al., 2025b; Li et al., 2025b) has shown that multi-step return backups are effective for improving offline/offline-to-online Q-learning agents. These methods either use a standard single-step critic network (Park et al., 2025b) that suffers from the off-policy bias, or use a ‘chunked,’ multi-step critic network (Li et al., 2025b) that does not have such bias but poses a huge policy learning challenge when the chunk size is too large. Our method brings the best of both worlds—it uses critic chunking to avoid the off-policy bias while simultaneously avoiding the policy learning challenge by extracting a simpler policy that extracts a shorter action chunk from the full-chunk critic.

Multi-step return backups are computed with multi-step off-policy rewards that can lead to systematic value underestimation (Sutton et al., 1998; Peng & Williams, 1994; Konidaris et al., 2011; Thomas et al., 2015), and there has been a rich literature (Precup et al., 2000; Munos et al., 2016; Rowland et al., 2020) dedicated to fix these biases via importance sampling (Kloek & Van Dijk, 1978) with truncation (Ionides, 2008). These approaches often require a careful balance between bias and variance that can be tricky to tune. More recently, Seo & Abbeel (2025); Li et al. (2025a); Tian et al. (2025); Li et al. (2025b) group temporally extended sequences of actions as chunks and directly estimate the value of an action chunk rather than a single action. Such a formulation allows the value backup to operate directly in the chunk space, which allows multi-step return backup without the systematic biases from the sub-optimal off-policy data. Despite their empirical success, we still lack a good theoretical understanding of the convergence of TD-learning with ‘chunked’ critics, as well as when it should be preferred over the standard n-step returns. Our work lays out the theoretical foundation for Q-learning with critic chunking, and identifies an important yet subtle, often overlooked bias in the chunked TDbackup. We quantify such bias and provide the condition under which TD backup using critic chunking is guaranteed to perform better than the standard n-step return backup with a single-step critic.

See additional discussions of related work on hierarchical reinforcement learning and theoretical analysis under confounding variables in Section D.

# 3 PRELIMINARIES

Reinforcement learning can be formalized as a Markov decision process, $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , T , r , \rho , \gamma )$ , where S is the state space, A is the action space, $T : \mathcal { S } \times \mathcal { A }  \Delta _ { \mathcal { S } }$ is the transition kernel that defines the next state distribution conditioned on the current state and the current action (e.g., $s ^ { \prime } \sim T ( \cdot \mid s , a ) ) , r : \mathcal { S } \times \mathcal { A }  [ 0 , 1 ]$ is the reward function, $\rho \in \Delta _ { S }$ is the initial state distribution, and $\gamma \in [ 0 , 1 )$ is the discount factor. We also assume we have access to a prior offline dataset $D = \{ ( s _ { 0 } ^ { i } , a _ { 0 } ^ { i } , r _ { 0 } ^ { i } , s _ { 1 } ^ { i } , a _ { 1 } ^ { i } , r _ { 1 } ^ { i } , \cdot \cdot \cdot , s _ { H } ^ { i } ) \} _ { i = 1 } ^ { | D | }$ where the goal is to learn a policy, ∞ $\pi : { \mathcal { S } }  \Delta _ { { \mathcal { A } } }$ that maximizes its return, $\begin{array} { r } { \eta ( \pi ) = \mathbb { E } _ { s _ { t + 1 } \sim T ( \cdot | s _ { t } , a _ { t } ) , a _ { t } \sim \pi ( \cdot | s _ { t } ) , s _ { 0 } \sim \rho } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) \right] } \end{array}$ . We call a policy that attains the maximum return as an optimal policy, π⋆.

Temporal difference learning. Modern value-based reinforcement learning methods often learn a critic network, $Q : S \times \mathcal { A }  \mathbb { R }$ parameterized by $\phi$ to approximate the expected return starting from state s and action a, and ϕ is often trained using the temporal-difference (TD) loss:

$$
L (\phi) = \mathbb {E} _ {s, a, s ^ {\prime} \sim \mathcal {D}} \left[ (Q _ {\phi} (s, a) - r (s, a) - \gamma \max _ {a ^ {\prime}} Q _ {\bar {\phi}} (s ^ {\prime}, a ^ {\prime})) ^ {2} \right], \tag {1}
$$

where $\bar { \phi }$ is often set to be an exponential moving average of $\phi .$ .

Implicit value backup. Instead of using maxa′ $Q ( s ^ { \prime } , a ^ { \prime } )$ as the TD target, we can use an implicit maximization loss function $f _ { \mathrm { i m p } }$ to learn $V _ { \xi } ( s )$ to approximate it (Kostrikov et al., 2022):

$$
L (\xi) = \mathbb {E} _ {s, a \sim \mathcal {D}} \left[ f _ {\mathrm{imp}} ^ {\kappa} (\bar {Q} (s, a) - V _ {\xi} (s)) \right]. \tag {2}
$$

Two popular choices of $f _ { \mathrm { i m p } } ^ { \kappa }$ are (1) expectile: $f _ { \mathrm { e x p e c t i l e } } ^ { \kappa } ( c ) ~ = ~ | \kappa - \mathbb { I } _ { c < 0 } | c ^ { 2 }$ , and (2) quantile: $f _ { \mathrm { q u a n t i l e } } ^ { \kappa } ( c ) ~ = ~ | \kappa - \mathbb { I } _ { c < 0 } | | c | ,$ for any real value $\kappa \in [ 0 . 5 , 1 )$ . At the optimum of $L ( \xi ) , V _ { \xi } ( s )$ approximates the κ-expectile/quantile of the distribution of the TD target for $Q ( s , a )$ , induced by the data distribution D. With such a technique, we no longer need to explicitly find the action a that maximizes $Q ( s , a )$ and can use $V _ { \xi } ( s )$ as the backup target:

$$
L (\phi) = \mathbb {E} _ {s, a, s ^ {\prime} \sim \mathcal {D}} \left[ (Q _ {\phi} (s, a) - r (s, a) - \gamma V _ {\xi} (s ^ {\prime})) ^ {2} \right]. \tag {3}
$$

Multi-step return backup. TD learning can sometimes struggle with long-horizon tasks due to the bootstrapping bias problem, where regressing the value network towards its own potentially inaccurate value estimates amplifies the value estimation errors further. To tackle this challenge, we can instead sample a trajectory segment, $( s _ { t } , a _ { t } , s _ { t + 1 } , \cdots , a _ { t + n - 1 } , s _ { t + n } )$ , to construct an n-step return backup target from states h steps ahead:

$$
L _ {\mathrm{ns}} (\phi) = \mathbb {E} _ {s _ {t}, a _ {t}, \dots , s _ {t + n}} \left[ \left(Q _ {\phi} (s _ {t}, a _ {t}) - R _ {t: t + n} - \gamma^ {n} \bar {Q} (s _ {t + n}, a _ {t + n} ^ {\star})\right) ^ {2} \right], \tag {4}
$$

where $\begin{array} { r } { a _ { t + n } ^ { \star } = \arg \operatorname* { m a x } _ { a _ { t + n } } Q ( s _ { t + n } , a _ { t + n } ) , R _ { t : t + n } : = \sum _ { t ^ { \prime } = t } ^ { t + n - 1 } \gamma ^ { t ^ { \prime } - t } r ( s _ { t ^ { \prime } } , a _ { t ^ { \prime } } ) } \end{array}$ . The n-step return backup reduces the effective horizon by a factor of $n ,$ alleviating the bootstrapping bias problem. However, such a value estimate is always biased towards the off-policy data distribution, and is also commonly referred to as the uncorrected n-step return estimator (Fedus et al., 2020; Kozuno et al., 2021). While there are ways to correct this value estimator via importance sampling (Precup et al., 2000; Munos et al., 2016; Rowland et al., 2020), they require additional tricks (e.g., importance ratio truncation) for numerical stability and re-introduce biases into the estimator, resulting in a delicate trade-off between variances and biases that needs to be carefully balanced.

Action chunking critic. Alternatively, one may learn an action chunking critic to estimate the value of a short sequence of actions (an action chunk), $a _ { t : t + h } : = ( a _ { t } , a _ { t + 1 } , \cdot \cdot \cdot , a _ { t + h - 1 } )$ instead: $Q ( s _ { t } , a _ { t : t + h } )$ (Seo & Abbeel, 2025; Li et al., 2025a; Tian et al., 2025; Li et al., 2025b). The TD backup loss for such a critic is naturally multi-step:

$$
L _ {\mathrm{QC}} (\phi) = \mathbb {E} _ {s _ {t: t + h + 1}, a _ {t: t + h}} \left[ \left(Q _ {\phi} (s _ {t}, a _ {t: t + h}) - R _ {t: t + h} - \gamma^ {h} \bar {Q} (s _ {t + h}, a _ {t + h: t + 2 h} ^ {\star})\right) ^ {2} \right], \tag {5}
$$

where again $\boldsymbol a _ { t + h : t + 2 h } ^ { \star } = \arg \operatorname* { m a x } _ { \boldsymbol a _ { t + h : t + 2 h } } Q ( \boldsymbol s _ { t + h } , \boldsymbol a _ { t + h : t + 2 h } )$ . On the one hand, unlike n-step return estimate for single-action critic that is pessimistic, the n-step return estimate (with $n = h )$ for the action chunking critic is unbiased as long as the action chunk $a _ { t : t + h }$ is independent of the intermediate states $s _ { t + 1 : t + h + 1 }$ , while enjoying the reduction in effective horizon (Li et al., 2025a;b). On the other hand, action chunking critic implicitly imposes a constraint on the policy that the actions are predicted and executed in chunks. As a result, the policy extracted from the action chunking critic needs to predict the entire action chunk all at once, posing a learning challenge, especially for environments with complex transition dynamics.

# 4 WHEN AND HOW SHOULD WE USE ACTION CHUNKING FOR Q-LEARNING?

In this section, we build a theoretical foundation for Q-learning with action chunking critic functions. We start by formalizing the setup of our analysis in Section 4.1, providing a formal definition of our key open-loop consistent condition in Section 4.2, quantifying the value estimation bias incurred from backing up on non-action chunking data (Theorems 1 and 2) and the optimality of action chunking policy (Theorems 3 and 4) using this condition in Section 4.3. Leveraging these results, we derive the conditions when we prefer action chunking Q-learning over the standard n-step return learning (Proposition 2) in Section 4.4. Finally, we characterize the key optimality variability conditions under which the closed-loop execution of a learned action chunking policy is close to the optimal closed-loop policy (Theorems 5 and 6) in Section 4.5. A brief summary of the key results is available in Table 1.

<table><tr><td>Assumptions</td><td>Value Estimation Error $\left| \hat{V}_{\text{ac}} - V_{\text{ac}} \right|$ </td><td>AC Optimality $V^{\star} - V_{\text{ac}}^{+}$ </td><td>Closed-loop AC Optimality $V^{\star} - V^{\bullet}$ </td></tr><tr><td>Weak  $\varepsilon_h$ -OLC</td><td> $\Theta(\varepsilon_h H \bar{H})$  (Theorems 1 and 2)</td><td> $\Omega(H)$  (Proposition 1)</td><td>-</td></tr><tr><td>Strong  $\varepsilon_h$ -OLC</td><td> $O(\varepsilon_h H \bar{H})$  (Theorem 1)</td><td> $\Theta(\varepsilon_h H \bar{H})$  (Theorems 3 and 4)</td><td> $O(\varepsilon_h H^2 \bar{H})$  (Proposition 3)</td></tr><tr><td> $(\vartheta_h^L, \vartheta_h^G)$ -BOV</td><td>-</td><td> $\Omega(H)$  (Theorem 6)</td><td> $\Theta(\vartheta_h^L H + \vartheta_h^G H \bar{H})$  (Theorems 5 and 6)</td></tr></table>

Table 1: Summary of our main theoretical contributions. In this work, we introduce open-loop consistency (OLC: Definition 2) and bounded optimality variability (BOV: Definition 4). Weak OLC provides guarantees on the value estimation error of action chunking critic but not the optimality of the learned action chunking policy. Strong OLC provides guarantees on the optimality of the learned action chunking policy and its closed-loop execution performance. BOV is an alternative condition to provide guarantees on the closed-loop execution performance. $\Omega ( H )$ means that a constant factor of the maximum value gap can be achieved in the worst case.

# 4.1 ASSUMPTIONS AND NOTATIONS

To build the foundation of our analysis, we start by describing the trajectory data distribution that we use for Q-learning and the trajectory distribution induced by an action chunking policy. In particular, we assume that the trajectory data distribution obeys the transition dynamics $T \colon$

Assumption 1 (Data Obeys the Transition Dynamics) $\mathcal { D } \in \Delta _ { \mathcal { T } }$ is a trajectory distribution generated by rolling out a behavior policy from a distribution of $s _ { t } \sim \mu$ . The behavior policy can be non-Markovian $( i . e . , \pi _ { \beta } ( a _ { t + k } \mid s _ { t : t + k + 1 } , a _ { t : t + k } ) )$ ). Each subsequent state is generated according to the dynamics of the $\mathrm { M D P } \mathrm { \mathcal { M } } \colon s _ { t + k + 1 } \sim \dot { T } ( \cdot \mid s _ { t + k } , a _ { t + k } ) , \mathsf { \bar { v } } k \in \{ 0 , 1 , \cdot \cdot \cdot , h - 1 \}$ . The resulting trajectory is $\left\{ s _ { t } , s _ { t + 1 } , \cdot \cdot \cdot , s _ { t + h } , a _ { t } , a _ { t + 1 } , \cdot \cdot \cdot , a _ { t + h } \right\} \in { \mathcal { T } } = S ^ { h } \times { \mathcal { A } } ^ { h }$ .

Next, we formally define the open-loop trajectory distribution that we would obtain if we take the same actions in the data and roll them out open-loop for h steps in the MDP.

Definition 1 (Open-loop Trajectory) From any data distribution D, we use $\pi _ { \mathcal { D } } ^ { \circ } : \mathcal { S }  \Delta _ { \mathcal { A } ^ { h } }$ to denote an action chunking policy that admits the same marginal distribution as D:

$$
\pi_ {\mathcal {D}} ^ {\circ} (a _ {t: t + h} \mid s _ {t}) := P _ {\mathcal {D}} (a _ {t: t + h} \mid s _ {t}). \tag {6}
$$

Rolling out this action chunking policy by carrying out actions in chunks induces a trajectory distribution $P _ { \mathcal { D } } ^ { \circ } \in \Delta _ { S ^ { h + 1 } , A ^ { h } }$ that is generally different from $P _ { \mathcal { D } } \mathrm { : }$ :

$$
P _ {\mathcal {D}} ^ {\circ} (s _ {t + 1: t + h + 1}, a _ {t: t + h} \mid s _ {t}) := \pi_ {\mathcal {D}} ^ {\circ} (a _ {t: t + k} \mid s _ {t}) \prod_ {k = 0} ^ {h - 1} T (s _ {t + k + 1} \mid s _ {t + k}, a _ {t + k}). \tag {7}
$$

Next, we introduce a set of notations and conventions that we use in our theoretical analysis. We use $a _ { t : t + h }$ to denote an action chunk of length h: $( a _ { t } , a _ { t + 1 } , \cdot \cdot \cdot , a _ { t + h - 1 } )$ (not including $a _ { t + h } )$ . We use the subscript $[ \cdot ] _ { \mathrm { a c } }$ for all action chunking policies or value functions, ˆ[·] to denote the nominal (i.e., estimated) value (in contrast to the actual without the $\cdot \bf \hat { \mu } )$ , and $[ \cdot ] ^ { + }$ to denote something that is learned from the data (usually defaults to D). For example, $\hat { V } _ { \mathrm { a c } } ^ { + } : \mathcal { S } \to [ 0 , 1 / ( 1 - \gamma ) ]$ is the nominal value $( i . e .$ , expected discounted return) of an action chunking policy $\pi _ { \mathrm { a c } } ^ { + }$ learned from D, whereas $V _ { \mathrm { a c } } ^ { + }$ is the actual value of the same action chunking policy (where the value is obtained by rolling the policy out in the MDP with open-loop action chunks). As we will elucidate in the next section, the nominal value and the actual value of a policy are usually different, and hence making this differentiation critical in our analysis. We also use $\check { [ \cdot ] } ^ { \star }$ to denote the optimal policy or value function under the constraint of the policy class $( e . g . , \pi _ { \mathrm { a c } } ^ { \star }$ for the optimal action chunking policy and $\pi ^ { \star }$ for the optimal closed-loop 1-step policy). Finally, we use $\bar { H } \overset { \cdot } { = } 1 / ( 1 - \gamma ) , \bar { H } = 1 \breve { / } ( \bar { 1 } - \bar { \gamma } ^ { h } )$ to denote the effective horizon for 1-step TD backup and h-step TD backup respectively.

# 4.2 WEAK AND STRONG OPEN-LOOP CONSISTENCY (OLC)

From the definition above, we have demonstrated that replaying the actions from the trajectory data distribution $P _ { \mathcal { D } }$ in an open-loop manner may result in a different trajectory distribution, $P _ { \mathcal { D } } ^ { \circ }$ . This discrepancy between $\dot { P } _ { \mathcal { D } } ^ { \circ }$ and $P _ { D }$ has not been carefully analyzed by prior work but can play a huge role in the optimal policy that action chunking Q-learning converges to. To characterize this discrepancy, we use a notion of consistency as defined below.

Definition 2 (Open-Loop Consistency) D is weakly $\varepsilon _ { h } .$ -open-loop consistent if for every $s _ { t } \in S$ with $\begin{array} { r } { P _ { \mathcal { D } } ( \tilde { \boldsymbol { s } } _ { t } ) > 0 \ \dot { ( } i . e . , \boldsymbol { s } _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } } ( \boldsymbol { s } _ { t } ) ) ) } \end{array}$ ,

$$
D _ {\mathrm{TV}} (P _ {\mathcal {D}} ^ {\circ} (s _ {t + h ^ {\prime}}, a _ {t + h ^ {\prime}} \mid s _ {t}) \parallel P _ {\mathcal {D}} (s _ {t + h ^ {\prime}}, a _ {t + h ^ {\prime}} \mid s _ {t})) \leq \varepsilon_ {h}, \forall h ^ {\prime} \in \{1, 2, \dots , h - 1 \}, \tag {8}
$$

$$
D _ {\mathrm{TV}} (P _ {\mathcal {D}} ^ {\circ} (s _ {t + h} \mid s _ {t}) \parallel P _ {\mathcal {D}} (s _ {t + h} \mid s _ {t})) \leq \varepsilon_ {h}. \tag {9}
$$

D is strongly $\varepsilon _ { h } .$ -open-loop consistent if additionally for every $a _ { t : t + h } \in \mathrm { s u p p } ( P _ { \mathcal { D } } ( a _ { t : t + h } \mid s _ { t } ) )$ ,

$$
D _ {\mathrm{TV}} (T (s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h ^ {\prime}}) \parallel P _ {\mathcal {D}} (s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h})) \leq \varepsilon_ {h}, \forall h ^ {\prime} \in \{1, 2, \dots , h \}, \tag {10}
$$

where we use $T ( s _ { t + h ^ { \prime } } \mid s _ { t } , a _ { t : t + h ^ { \prime } } )$ to denote the distribution of the future state $s _ { t + h ^ { \prime } }$ after carrying out the action sequence $a _ { t : t + h ^ { \prime } }$ in the environment open-loop from the current state $s _ { t } .$

Intuitively, D is $\varepsilon _ { h }$ -open-loop consistent if, when executing the same sequence of actions from it open-loop from $s _ { t } ,$ , the resulting marginal distribution of the state-action h steps into the future $( i . e . ,$ $s _ { t + h } )$ deviates from the corresponding distribution in the dataset by at most $\varepsilon _ { h }$ in total variation distance. The strong version (Equation (10)) requires the total variation distance bound to hold for every action sequence in the support, whereas the weak version (Equations (8) and (9)) only requires the bound to hold in expectation. See Section E.1 for examples of weakly open-loop consistent data.

# 4.3 VALUE LEARNING BIAS OF ACTION CHUNKING Q-LEARNING

Next, we show that the weak open-loop consistency of D alone is sufficient to show that behavior value iteration of an action chunking critic results in a nominal value function $( i . e . , \hat { V } _ { \mathrm { a c } } )$ with a bounded bias from the true value $( i . e . , V _ { \mathrm { a c } } )$ of the behavior cloning action chunking policy $\tilde { \pi } _ { \mathrm { a c } } \mathrm { : }$

Theorem 1 (AC Value Bias) Let $\hat { V } _ { \mathrm { a c } } : S  [ 0 , 1 / ( 1 - \gamma ) ]$ be a solution of

$$
\hat {V} _ {\mathrm{ac}} (s _ {t}) = \mathbb {E} _ {s _ {t + 1: t + h + 1}, a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t})} \left[ R _ {t: t + h} + \gamma^ {h} \hat {V} _ {\mathrm{ac}} (s _ {t + h}) \right], \tag {11}
$$

with $\begin{array} { r } { R _ { t : t + h } = \sum _ { t ^ { \prime } = t } ^ { t + h } \gamma ^ { t ^ { \prime } - t } r ( s _ { t ^ { \prime } } , a _ { t ^ { \prime } } ) } \end{array}$ and $V _ { \mathrm { a c } }$ is the true value of $\tilde { \pi } _ { \mathrm { a c } } : s _ { t } \mapsto P _ { \mathcal { D } } \big ( a _ { t : t + h } ~ | ~ s _ { t } \big )$ If D is weakly εh-open-loop consistent, then for all $s _ { t } \in \mathrm { s u p p } ( P _ { D } ( s _ { t } ) )$ ,

$$
\left| V _ {\mathrm{ac}} (s _ {t}) - \hat {V} _ {\mathrm{ac}} (s _ {t}) \right| \leq \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})} \leq \varepsilon_ {h} H \bar {H}. \tag {12}
$$

Furthermore, we show that this bound is tight for any value of $h > 1 , \gamma \in [ 0 , 1 )$ and $\begin{array} { r } { 0 \leq \varepsilon _ { h } \leq \frac { 1 } { 2 } ; } \end{array}$

Theorem 2 (Worst-case AC Value Bias) For any $h > 1 , \gamma \in [ 0 , 1 ) , \varepsilon _ { h } \in [ 0 , 1 / 2 ]$ , there exists an MDP M and a weakly εh-open-loop consistent D such that for some $s _ { t } \in \dot { \mathrm { s u p p } } ( P _ { \mathcal { D } } ( s _ { t } ) )$ ,

$$
V _ {\mathrm{ac}} (s _ {t}) - \hat {V} _ {\mathrm{ac}} (s _ {t}) = \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}. \tag {13}
$$

Similarly, there exists M and $\varepsilon _ { h }$ -open-loop consistent D such that for some $s _ { t } \in$ supp $\left( P _ { \mathcal { D } } ( s _ { t } ) \right)$

$$
\hat {V} _ {\mathrm{ac}} (s _ {t}) - V _ {\mathrm{ac}} (s _ {t}) = \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}. \tag {14}
$$

The proofs can be found in Section F.2 and Section F.3. A direct consequence of these results is that the true value of the optimal action chunking policy is close to that of the optimal closed-loop policy:

Corollary 1 (Optimality Gap for AC Policy) Let ${ \mathcal { D } } ^ { \star }$ be the data collected by any optimal policy $\pi ^ { \star }$ . If $\mathcal { D } ^ { \star }$ is weakly εh-open-loop consistent, then for all $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } \bar { ( } s _ { t } ) ) $ ,

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {\star} (s _ {t}) \leq V ^ {\star} (s _ {t}) - \tilde {V} _ {\mathrm{ac}} (s _ {t}) \leq \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})} \leq \varepsilon_ {h} H \bar {H}, \tag {15}
$$

where $V ^ { \star }$ is the value of the optimal policy $\pi ^ { \star } , V _ { \mathrm { a c } } ^ { \star }$ is the true value of the optimal action chunking policy, and $\tilde { V } _ { \mathrm { a c } }$ is the true value of the action chunking policy from cloning the data $\mathcal { D } ^ { \star }$ :

$$
\tilde {\pi} _ {\mathrm{ac}} ^ {\mathcal {D} ^ {\star}} (a _ {t: t + h} \mid s _ {t}): s _ {t} \mapsto P _ {\mathcal {D} ^ {\star}} (\cdot \mid s _ {t}). \tag {16}
$$

We show that his bound is also tight. The proofs can be found in Section F.4 and Section F.5.

Corollary 2 (Worse-case Optimality Gap for Action Chunking Policy) For any $h > 1 , \gamma \in$ $[ 0 , 1 ) , \varepsilon _ { h } \stackrel { \cdot } { \in } [ 0 , 1 / 2 ]$ , there exists an MDP M whose optimal policy $\pi ^ { \star }$ induces a data distribution $\dot { \mathcal { D } } ^ { \star }$ that is weakly $\varepsilon _ { h }$ -open-loop consistent, such that for some $\dot { s _ { t } } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {\star} (s _ {t}) = \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}. \tag {17}
$$

The key observation that enables these results is that $\hat { V } _ { \mathrm { a c } }$ obtained from value iteration on ${ \mathcal { D } } ^ { \star }$ (data collected by an optimal policy) recovers the value of the optimal policy $V ^ { \star }$ . This allows us to use Theorem 1 to directly obtain a bound on the optimality gap for action chunking policies.

Next, we analyze the performance of the action chunking policy obtained by Q-learning. In particular, we analyze the Q-function obtained as a solution of the bellman optimality equation under supp(D):

$$
\hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h}) = \mathbb {E} _ {s _ {t + 1: t + h + 1} \sim P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h})} \left[ R _ {t: t + h} + \gamma^ {h} \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t + h}, \pi_ {\mathrm{ac}} ^ {+} (s _ {t + h})) \right], \tag {18}
$$

where $\pi _ { \mathrm { a c } } ^ { + }$ is defined as follows:

$$
\pi_ {\mathrm{ac}} ^ {+}: s _ {t} \mapsto \arg \max _ {a _ {t: t + h} \in \operatorname{supp} (P _ {\mathcal {D}} (a _ {t: t + h} | s _ {t}))} \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h}). \tag {19}
$$

With only the weak open-loop consistency condition, the worst-case performance of the action chunking policy may be arbitrarily low, as formalized below (proof available in Section F.6).

Proposition 1 (AC Q-Learning under Weak OLC) For any $h > 1 , \gamma \in [ 0 , 1 ) , c \in [ 0 , 1 / 2 )$ , $\varepsilon _ { h } \ \in \ ( 0 , 1 / 2 )$ , there exists an MDP M, a weakly εh-open-loop consistent D and $\mathcal { \dot { D } } ^ { \star }$ with supp $( P _ { \mathcal D } ( \dot { s } _ { t } , a _ { t : t + h } ) ) \supseteq \operatorname { s u p p } ( P _ { \mathcal D ^ { \star } } ( s _ { t } , a _ { t : t + h } ) )$ , such that for some $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) = V _ {\mathrm{ac}} ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) = \frac {\gamma c}{1 - \gamma}. \tag {20}
$$

Intuitively, the chunked critic $Q ( s _ { t } , a _ { t : t + h } )$ has no way of differentiating a low-probability, ‘lucky’ success from a closed-loop, high-probability success. This can cause the learned policy $\pi _ { \mathrm { a c } } ^ { + }$ to erroneously prefer very low-value action chunks even when the optimal action chunks are available in the data distribution. With Proposition 1, we conclude that the weak open-loop consistency is insufficient for effectively bounding the sub-optimality of action chunking Q-learning. Fortunately, the strong open-loop consistency (Equation (10)) is sufficient as quantified by the following bound:

Theorem 3 (AC Q-Learning under Strong OLC) If D and $\mathcal { D } ^ { \star }$ are strongly εh-open-loop consistent and supp $( P _ { \mathcal D } ( s _ { t } , a _ { t : t + h } ) ) \supseteq \mathrm { s u p p } ( P _ { \mathcal D ^ { \star } } ( s _ { t } , a _ { t : t + h } ) )$ , then for all $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) \leq \frac {\varepsilon_ {h} \gamma}{1 - \gamma} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right] \leq 3 \varepsilon_ {h} H \bar {H}, \tag {21}
$$

where $V ^ { \star }$ is the value of a closed-loop optimal policy and $V _ { \mathrm { a c } } ^ { + }$ is the true value of $\pi _ { \mathrm { a c } } ^ { + } .$

Theorem 3 (proof in Section F.7) shows that as long as both D and ${ \mathcal { D } } ^ { \star }$ satisfy the strongly openloop consistency condition and D contains the behavior in ${ \mathcal { D } } ^ { \star }$ , Q-learning with action chunking is guaranteed to converge to a near-optimal action chunking policy regardless of how sub-optimal the data D might be. Also, we show this bound is tight (proof in Section F.8):

Theorem 4 (Worst-case Analysis of Q-Learning with Action Chunking Policy on Off-policy Data) For any $h > 1 , \gamma \in ( \bar { 0 } , 1 ) , \varepsilon _ { h } \overset { \cdot } { \in } ( 0 , 1 / \bar { 5 } ) , c _ { 1 } \in ( 0 , \varepsilon _ { h } / 2 )$ , and $c _ { 2 } \in ( 0 , 2 \varepsilon _ { h } \gamma )$ , there exists an MDP M and strongly $\varepsilon _ { h } \mathrm { - o p e n - l o o p }$ consistent data distributions $\mathcal { D }$ and D⋆ with supp $( P _ { \mathcal D } ( s _ { t } , a _ { t : t + h } ) ) \supseteq \mathrm { s u p p } ( \bar { P } _ { \mathcal D ^ { \star } } ( s _ { t } , a _ { t : t + h } ) )$ , such that for some $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) = \frac {2 \varepsilon_ {h} \gamma - c _ {2}}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})} + \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - \varepsilon_ {h} - c _ {1}) \gamma^ {h})}, \tag {22}
$$

where $V ^ { \star }$ is the value of an optimal policy and $V _ { \mathrm { a c } } ^ { + }$ is the true value of $\pi _ { \mathrm { a c } } ^ { + } . \mathrm { A s } c _ { 1 } , c _ { 2 }  0 ,$

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) \rightarrow \frac {\varepsilon_ {h} \gamma}{1 - \gamma} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right]. \tag {23}
$$

Up to now, none of the bounds that we have shown so far depend on the sub-optimality of the data. Indeed, we can make the data arbitrarily sub-optimal while the action chunking policy learning is still guaranteed to be near optimal. As we will show in the following section, this is in contrast to n-step return policy where its performance depends on the sub-optimality of the data.

# 4.4 COMPARING TO n-STEP RETURN Q-LEARNING

We now characterize the condition when action chunking Q-learning should be preferred over the standard n-step return backup. We start by introducing a notion of sub-optimality:

Definition 3 (Sub-optimal Data) D is $\delta _ { n }$ -sub-optimal for a backup horizon length of n $> 1$ if

$$
Q ^ {\star} (s _ {t}, a _ {t}) - \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t})} \left[ R _ {t: t + n} + \gamma^ {n} V ^ {\star} (s _ {t + n}) \right] \geq \delta_ {n}, \forall s _ {t}, a _ {t} \in \mathrm{supp} (P _ {\mathcal {D}} (s _ {t}, a _ {t})). \tag {24}
$$

Intuitively, $\delta _ { n }$ captures how much worse the n-step return policy can get compared to the optimal policy incurred by the backup bias. Under such condition, we can show that the action chunking policy is provably better than the n-step return policy as long as $\delta _ { n }$ is large.

Proposition 2 (Comparing action chunking backup and n-step return backup) Let D be strongly εh-open-loop consistent and $\delta _ { n }$ -sub-optimal, and supp $( P _ { \mathcal { D } } ( s _ { t } ) ) \supseteq \operatorname { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ . Let $\pi _ { n } ^ { + } : s _ { t } \mapsto \arg \operatorname* { m a x } _ { a _ { t } } \hat { Q } _ { n } ^ { + } ( s _ { t } , a _ { t } )$ be the policy learned from $\mathcal { D } ,$ via n-step return backup:

$$
\hat {Q} _ {n} ^ {+} (s _ {t}, a _ {t}) = \mathbb {E} \left[ R _ {t: t + n} + \gamma^ {n} \hat {Q} _ {n} ^ {+} (s _ {t + n}, \pi_ {n} ^ {+} (s _ {t + n})) \right]. \tag {25}
$$

Then, for all $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ (and with $\bar { H } _ { n } = 1 / ( 1 - \gamma ^ { n } ) )$ ,

$$
\begin{array}{l} V _ {\mathrm{ac}} ^ {+} (s _ {t}) - \hat {V} _ {n} ^ {+} (s _ {t}) \geq \frac {\delta_ {n}}{1 - \gamma^ {n}} - \frac {\varepsilon_ {h} \gamma}{1 - \gamma} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right], \tag {26} \\ \geq \delta_ {n} \bar {H} _ {n} - 3 \varepsilon_ {h} H \bar {H}. \\ \end{array}
$$

The proof of Proposition 2 is available in Section F.10. Notably, for $n = h ,$ , as long as D is more than $( 3 \varepsilon _ { h } \bar { H } )$ -sub-optimal, the value of the action chunking policy is provably better than the value of the n-step return policy. It is worth noting that Proposition 2 uses the nominal value of the n-step return, which may be lower than its actual value. We refer the readers to Section E.2 for examples where the n-step return policy is provably worse than the action chunking policy.

Up to now, we have characterized the conditions under which action chunking policies are better than n-step return policies. However, action chunking policies are still fundamentally limited when subject to poor open-loop consistent data. To tackle this challenge, we explore closed-loop execution of an action chunking policy $( i . e .$ , carrying out the first action of the full action chunk at every step). While this has been explored in robotic applications (Zhao et al., 2023; Chi et al., 2023; Lin et al., 2025; Black et al., 2025) to reduce latency and improve smoothness, the theoretical property of closed-loop execution of action chunking policies is not well-understood, especially in the context of Q-learning.

# 4.5 CLOSED-LOOP EXECUTION OF ACTION CHUNKING POLICY

If we reuse the same strongly $\varepsilon _ { h }$ -open-loop consistency assumption, we can guarantee that closedloop execution of the action chunking policy is also near-optimal. The intuition is that in order for action chunking policy to be near-optimal, the first action in the chunk cannot be too sub-optimal:

Proposition 3 (Optimality of Closed-loop Execution of Action Chunking Policy) Let $V ^ { \bullet }$ be the value of the one-step policy, $\pi ^ { \bullet }$ , as a result of the closed-loop execution of the action chunking policy $\pi _ { \mathrm { a c } } ^ { + }$ learned from $\mathrm { \bar { \it { D } } }$ . That is, for each $s _ { t } \in \mathrm { s u p p } ( P _ { D } ( s _ { t } ) )$ ,

$$
\pi^ {\bullet} (s _ {t}) = a _ {t} ^ {+}, \quad \text { where } a _ {t: t + h} ^ {+} = \pi_ {\mathrm{ac}} ^ {+} (s _ {t}). \tag {27}
$$

If D and ${ \mathcal { D } } ^ { \star }$ are both strongly εh-open-loop consistent and supp $\left( P _ { \mathcal { D } } ( s _ { t } , a _ { t : t + h } ) \right)$ ⊇ supp $\left( P _ { \mathcal { D } ^ { \star } } ( s _ { t } , a _ { t : t + h } ) \right)$ , then for all $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V ^ {\bullet} (s _ {t}) \leq \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) ^ {2}} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right] \leq 3 \varepsilon_ {h} H ^ {2} \bar {H}. \tag {28}
$$

The proof is available in Section F.9. This result demonstrates that closed-loop execution is also nearoptimal as long as the action chunking policy is near-optimal, though we might have to pay up to a horizon factor H $( i . e . , 1 / ( 1 - \gamma ) )$ in sub-optimality gap in the worst case. Can we do better than this?

In practical applications, the data distributions that we are dealing with often have more structure. For example, it is common to have a dataset consisting of multiple sources where each data source is collected by either a human expert or a scripted policy that exhibits a somewhat predictable behavior (e.g., after a robot arm picks up a cube, it will always move up rather than dropping it right away). We formalize this kind of structures as a notion of optimality variability as follows:

Definition 4 (Optimality Variability) D exhibits $\vartheta _ { h }$ -bounded variability in optimality conditioned on an event X if

$$
\max _ {\operatorname{supp} \left(P _ {\mathcal {D}} (\cdot | X)\right)} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} \left(s _ {t + h}\right) \right] - \min _ {\operatorname{supp} \left(P _ {\mathcal {D}} (\cdot | X)\right)} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} \left(s _ {t + h}\right) \right] \leq \vartheta_ {h}. \tag {29}
$$

If we pick X to be the current state and the current action, a bounded optimality variability subject to such conditioning means that as long as we observe the initial action, the optimality of the outcome after h-steps does not vary too much. It turns out that if (1) the data distribution is a mixture of a bunch of data sources where the optimality variability conditioned on the current actions is bounded within each data source, and additionally (2) the optimality variability conditioned on the current action chunks is bounded globally across mixture, we can form a much stronger bound on the optimality of $\pi ^ { \bullet }$ . It is worth noting that the second optimality variability condition is much weaker than the first one because it is conditioned on the event where we observe the state $s _ { t }$ and the entire action chunk $a _ { t : t + h }$ (rather than only the first action $a _ { t } )$ . We now state our theorem as follows:

Theorem 5 (Closed-loop AC Policy under Bounded OV) Let ${ \mathcal { D } } ^ { \star }$ be the data distribution collected by an optimal policy. Assume $\mathcal { D }$ can be decomposed into a mixture of data distributions $\{ \mathcal { D } ^ { \star } , \mathcal { D } _ { 1 } , \dot { \mathcal { D } } _ { 2 } , \cdot \cdot \dot { \cdot } \mathcal { D } _ { M } \}$ such that each data distribution component satisfies Assumption 1 and for some $\vartheta _ { h } ^ { L } , \vartheta _ { h } ^ { G } \geq 0 ,$ they satisfy the following two conditions:

1. Locally bounded optimality variability condition: every $\mathcal { D } _ { i }$ (including $\mathcal { D } ^ { \star } )$ exhibits $\vartheta _ { h } ^ { L }$ bounded variability in optimality conditioned on $s _ { t } , a _ { t }$ for all $( s _ { t } , a _ { t } ) \in \mathrm { s u p p } ( P _ { \mathcal { D } _ { i } } ( s _ { t } , a _ { t } ) )$ , and   
2. Globally bounded optimality variability condition: D as a whole exhibits $\vartheta _ { h } ^ { G }$ -variability in optimality conditioned on $s _ { t } , a _ { t : t + h }$ for all $( s _ { t } , a _ { t : t + h } ) \in \operatorname { s u p p } ( P _ { \mathcal { D } } ( s _ { t } , a _ { t : t + h } ) )$ .

Then for all $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V ^ {\bullet} (s _ {t}) \leq \frac {\vartheta_ {h} ^ {L}}{1 - \gamma} + \frac {\vartheta_ {h} ^ {G} + \gamma^ {h} \min (\vartheta_ {h} ^ {L} , \vartheta_ {h} ^ {G})}{(1 - \gamma) (1 - \gamma^ {h})} \leq \vartheta_ {h} ^ {L} H + 2 \vartheta_ {h} ^ {G} H \bar {H}. \tag {30}
$$

The proof of Theorem 5 (available in Section F.12) is made possible by observing that $V ^ { \star } ( s _ { t } ) - \hat { V } _ { \mathrm { a c } } ^ { + } ( s _ { t } )$ and $\hat { V } _ { \mathrm { a c } } ^ { + } ( s _ { t } ) - Q ^ { \star } ( s _ { t } , a _ { t } ^ { + } )$ are bounded by $\vartheta _ { h } ^ { G } / ( 1 - \gamma ^ { h } )$ and $\vartheta _ { h } ^ { L } / ( 1 - \gamma ^ { h } )$ respectively. Combining these two bounds naïvely already allows us to derive a relatively loose bound $V ^ { \star } ( s _ { t } ) - Q ^ { \star } ( s _ { t } , a _ { t } ^ { + } ) \leq$ $( \vartheta _ { h } ^ { L } + \vartheta _ { h } ^ { G } ) / ( 1 - \gamma ^ { h } )$ which leads to $V ^ { \star } ( s _ { t } ) - V ^ { \bullet } ( s _ { t } ) \leq ( \vartheta _ { h } ^ { \bar { L } } + \vartheta _ { h } ^ { G } ) / ( 1 - \gamma ^ { h } ) / ( \bar { 1 } - \bar { \gamma } )$ . To obtain the tight bound in Theorem 5, we leverage a key insight that the amount of overestimation in $V _ { \mathrm { a c } } ^ { + }$ can never exceed $\begin{array} { r } { \vartheta _ { h } ^ { L } + \frac { \vartheta ^ { G } } { 1 - \gamma ^ { h } } } \end{array}$ ϑG as otherwise the nominal value of the action chunking policy h-step into the future, $\hat { V } _ { \mathrm { a c } } ^ { + } ( s _ { t + h } )$ , would have an optimality gap higher than $\vartheta _ { h } ^ { G } / ( 1 - \gamma ^ { h } )$ , which is impossible under the global optimality variability condition. Forming this tight bound is important because it effectively shaves off a factor of $\bar { H } \overset { \cdot } { = } 1 / ( 1 - \gamma ^ { h } )$ from the $\vartheta _ { h } ^ { L }$ term (the stronger local condition) and only bumps up a factor of $\approx 2$ to the $\vartheta _ { h } ^ { G }$ term (the weaker global condition).

It is worth noting that although the global optimality variability condition looks similar to the strong open-loop consistency condition, they have completely different properties. For instance, a nearly strong open-loop consistent data distribution D can have unbounded global optimality variability and a data distribution that exhibits zero optimality variability can also have large open-loop inconsistency. The implication of this is that while the closed-loop execution of an action chunking policy can be near-optimal, the same action chunking policy executed in chunks can be sub-optimal. We formalize this intuition as the worse-case result below:

Theorem 6 (Worst-case Closed-loop AC Policy under BOV) For any $h \ > \ 1 , \gamma \in$ $\begin{array} { r } { ( 0 , 1 ) , \vartheta _ { h } ^ { G } , \vartheta _ { h } ^ { L } \in \left( 0 , \frac { \gamma - \gamma ^ { h } } { 4 ( 1 - \gamma ) } \right] , c \in \left[ 0 , \frac { \gamma - \gamma ^ { h } } { 4 ( 1 - \gamma ^ { h } ) } \right) , \sigma \in \left( 0 , \frac { \operatorname* { m i n } ( \vartheta _ { h } ^ { G } , \vartheta _ { h } ^ { L } ) } { 1 - \gamma } \right) } \end{array}$ , there exists M and D satisfying the assumptions in Theorem 5 such that there exists $s _ { t } \in \mathrm { s u p p } ( P _ { { D ^ { \star } } } ( s _ { t } ) )$ , where

$$
V ^ {\star} (s _ {t}) - V ^ {\bullet} (s _ {t}) = \frac {\vartheta_ {h} ^ {L}}{1 - \gamma} + \frac {\vartheta_ {h} ^ {G} + \gamma^ {h} \min (\vartheta_ {h} ^ {L} , \vartheta_ {h} ^ {G})}{(1 - \gamma) (1 - \gamma^ {h})} - \sigma , V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) \geq \frac {c}{1 - \gamma}. (3 1)
$$

The examples in the proof of Theorem 6 (available in Section F.13) serve as a dual purpose—they not only show that our upper-bound in Theorem 5 is $t i g h t$ (since we can make $\sigma  0 )$ , but also show that the sub-optimality of the action chunking policy can be made arbitrarily large. Furthermore, both the local optimality $\dot { ( \vartheta _ { h } ^ { L } ) }$ condition and the global optimality $( \vartheta _ { h } ^ { G } )$ are necessary to guarantee $\pi ^ { \bullet }$ being near-optimal. When any of them is large, Theorem 6 implies that there exists an MDP where $\pi ^ { \bullet }$ is sub-optimal. As a side note, we can guarantee $\pi ^ { \bullet }$ to be near-optimal with an alternative ‘stochastic shortcut’ assumption (a weaker form of the global optimality variability assumption) and a slightly stronger data mixing assumption. We refer the readers to Section E.3 for the formal results under this alternative assumption.

Overall, combining Theorem 5 and Proposition 3 shows that, compared to executing the action chunking policy in open-loop chunks, closed-loop execution attains a similar bound under the strongly εh-open-loop consistent assumption, and excels under the bounded optimality variability assumptions. Conceptually, closed-loop execution of the learned action chunking policy decouples the open-loop execution horizon (policy chunk length) from the value-learning horizon (critic chunk length). Such decoupling inherits the strength of action chunking TD and 1-step TD: (1) the value learning speedup of action chunking Q-learning, and (2) the reactivity of a standard, single-step policy. Furthermore, executing the first action (or more generally a partial chunk) of the original action chunk also brings practical benefits: it removes the need to explicitly train a policy to predict the full action chunk all at once, which can be especially challenging when the chunk size grows big. Can we develop a practical method that realizes such potential?

# 5 DECOUPLED Q-CHUNKING

In this section, we propose a new algorithm that enjoys the benefits of value backup speedup of critic chunking while avoiding the difficulty of learning an open-loop action chunking policy with a large chunk size. As we have elucidated in the previous section, our core idea is to decouple the chunk size of the critic from that of the policy where the policy only predicts a partial action chunk. In particular, we train a policy $\pi ( a _ { t : t + h _ { a } } \mid s _ { t } )$ to output an action chunk (with a size of $h _ { a } \ll h )$ using the following objective:

$$
L (\pi) := - \mathbb {E} _ {a _ {t: t + h _ {a}} \sim \pi (\cdot | s _ {t})} [ Q _ {\phi} (s _ {t}, [ a _ {t: t + h _ {a}}, a _ {t + h _ {a}: t + h} ^ {\star} ]) ], \tag {32}
$$

where $[ a _ { t : t + h _ { a } } , a _ { t + h _ { a } : t + h } ^ { \star } ]$ represents the concatenation of two partial action chunks (size $h _ { a }$ and size $h - h _ { a } )$ a  into a full action chunk $a _ { t : t + h }$ of size h, and $a _ { t + h _ { a } : t + h } ^ { \star }$ is the best ‘second-half’ of the action chunk that maximizes the critic value under $Q _ { \phi } \mathrm { . }$ :

$$
a _ {t + h _ {a}: t + h} ^ {\star} := \arg \max _ {a _ {t + h _ {a}: t + h}} Q _ {\phi} (s _ {t}, [ a _ {t: t + h _ {a}}, a _ {t + h _ {a}: t + h} ]). \tag {33}
$$

Essentially, we want our policy to predict the partial action chunk (of size $h _ { a } )$ within an optimal action chunk of size $h ,$ rather than the entire optimal action chunk. This lowers the policy expressivity requirement and hence the learning challenges associated with it.

However, directly optimizing the objective in Equation (32) does not lead to a new algorithm because taking the maximization over $a _ { t + h _ { a } : t + h }$ seemingly requires us to learn a policy of the original chunk size anyways. To address this issue, we learn a separate partial critic $\dot { Q _ { \psi } ^ { P } }$ , which only takes in the partial action chunk (of size $h _ { a } )$ as input, to approximate the maximum value this partial action chunk can achieve when it is extended to the full action chunk (of size h):

$$
Q _ {\psi} ^ {P} (s _ {t}, a _ {t: t + h _ {a}}) \approx Q _ {\phi} (s _ {t}, [ a _ {t: t + h _ {a}}, a _ {t + h _ {a}: t + h} ^ {\star} ]). \tag {34}
$$

To train $Q _ { \psi } ^ { P }$ , we can use an implicit maximization loss function (as described in Equation (2)):

$$
L (\psi) := f _ {\text { imp }} ^ {\kappa_ {d}} \left(\bar {Q} _ {\phi} \left(s _ {t}, a _ {t: t + h}\right) - Q _ {\psi} ^ {P} \left(s _ {t}, a _ {t: t + h _ {a}}\right)\right), \tag {35}
$$

where $s _ { t } , a _ { t : t + h }$ are sampled from the offline dataset D. As a result, the partial critic, $Q _ { \psi } ^ { P }$ , is distilled from the original critic via an optimistic regression, where its optimum $Q _ { \psi } ^ { \star } \big ( s _ { t } , a _ { t : t + h _ { a } } \big )$ approximates $Q _ { \phi } \big ( s _ { t } , [ a _ { t : t + h _ { a } } , a _ { t + h _ { a } : t + h } ^ { \star } ] \big )$ in Equation (32), conveniently removing the need for training a policy to predict the whole optimal action chunk entirely. This allows us to simplify the policy objective as

$$
L (\pi) := - \mathbb {E} _ {a _ {t: t + h _ {a}} \sim \pi (\cdot | s _ {t})} \left[ Q _ {\psi} ^ {P} (s _ {t}, a _ {t: t + h _ {a}}) \right]. \tag {36}
$$

In summary, DQC trains a policy to predict a partial chunk, $a _ { t : t + h _ { c } }$ a (of size $h _ { a } )$ , by hill climbing the value of a partial critic $\ u { Q _ { \psi } ^ { P } } \overset { \cdot } { ( } s _ { t } , \grave { a _ { t : t + h _ { a } } } \grave { ) }$ that is distilled from the original chunked critic $Q _ { \phi } \big ( s _ { t } , a _ { t : t + h } \big )$ via an implicit maximization loss. This allows our policy to fully leverage the chunked critic $Q _ { \phi }$ (and thus the value speedup benefits associated with Q-chunking) without the need to predict the full action chunk (of size h), mitigating the learning challenge of an action chunking policy.

Practical considerations for offline RL. Finally, we describe several implementation details that we find to work well in the offline RL setting, which our experiments focus on. Our implementation draws inspiration from a prior method, IDQL (Hansen-Estruch et al., 2023).

We first train a behavior cloning flow policy $\pi _ { \beta }$ using a standard flow-matching objective (Liu et al., 2023) on the offline dataset D. Then, we approximate the policy optimization objective in Equation (36) using best-of-N sampling without explicitly modeling π:

$$
a _ {t: t + h _ {a}} ^ {\star} \leftarrow \arg \max _ {\left\{a _ {t: t + h _ {a}} ^ {i} \right\} _ {i = 1} ^ {N}} Q _ {\psi} ^ {P} (s _ {t}, a _ {t: t + h _ {a}}), \quad \text { where } a _ {t: t + h _ {a}} ^ {1}, \dots , a _ {t: t + h _ {a}} ^ {N} \sim \pi_ {\beta} (\cdot \mid s _ {t}), \tag {37}
$$

and $a _ { t : t + h _ { a } } ^ { \star }$ is output of the policy that we extract from $Q _ { \psi } ^ { P }$ for state $s _ { t } .$ . Essentially, this sampling procedure is a test-time approximation of the objective in Equation (36), where it outputs an action (chunk) that maximizes $\stackrel { \triangledown } { Q } _ { \psi } ^ { P }$ , subject to the behavior prior, as modeled by $\pi _ { \beta }$ .

For TD learning of $Q _ { \phi }$ , directly computing the TD backup target from either $Q _ { \bar { \phi } }$ or $Q _ { \hat { \psi } } ^ { P }$ is computationally expensive, as either requires samples from the current policy, which is approximated via the bestof-N sampling procedure as described above. Instead, we use the implicit value backup (Kostrikov et al., 2022) (i.e., as described in Equation (2)) to approximate the target:

$$
L (\xi) = f _ {\text { quantile }} ^ {\kappa_ {b}} \left(\bar {Q} _ {\psi} ^ {P} \left(s _ {t}, a _ {t: t + h _ {a}}\right) - V _ {\xi} \left(s _ {t}\right)\right), \tag {38}
$$

where we pick the quantile regression loss as the implicit maximization loss function. This is because the Q-value obtained from best-of-N sampling can be seen as the largest order statistic of a random batch (of size N ) of the behavior Q-values. Such statistic estimates the behavior Q-value distribution’s $\textstyle { \frac { N - 1 } { N } }$ -quantile, which is the same as $V _ { \xi } ( s )$ at the optimum of $L ( \xi )$ if we set $\kappa _ { b } \overset { \sim } { = } \frac { N - 1 } { N }$ . In practice, we use a smaller $\kappa _ { b }$ for numerical stability (see Table 8).

Finally, we pick the expectile regression loss for training the distilled partial critic $Q _ { \psi } ^ { P }$ because prior work has found it to work the best among all implicit maximization loss functions (Hansen-Estruch et al., 2023). A summary of the algorithm is available in Algorithm 1.

Algorithm 1 Decoupled Q-chunking (DQC).   
Given: $D, Q_{\phi}(s_t, a_{t:t+h}), Q_{\psi}^P(s_t, a_{t:t+h_a}), V_{\xi}(s_t), \pi_{\beta}(a_{t:t+h_a} \mid s_t)$

# 1. Agent Update:

$(s_{t:t + h + 1},a_{t:t + h},r_{t:t + h})\sim D.$ ▷ sample trajectory chunk from the offline dataset

Optimize $Q_{\phi}$ with $L(\phi) = \left(Q_{\phi}(s_t, a_{t:t+h}) - \sum_{k=0}^{h-1} \gamma^k r_{t+k} - \gamma^h \bar{V}_{\xi}(s_{t+h})\right)^2$ .

Optimize $Q_{\psi}^{P}$ with $L(\psi) = f_{\mathrm{expectile}}^{\kappa_{\mathrm{d}}}\left(\bar{Q}_{\phi}(s_t,a_{t:t + h}) - Q_{\psi}^{P}(s_t,a_{t:t + h_a})\right)$ .

Optimize $V_{\xi}$ with $L(\xi) = f_{\mathrm{quantile}}^{\kappa_{\mathrm{b}}}(\bar{Q}_{\psi}^{P}(s_{t},a_{t:t + h_{a}}) - V_{\xi}(s_{t}))$

# 2. Policy Extraction:

$a_{t:t+h_a}^1, a_{t:t+h_a}^2, \cdots, a_{t:t+h_a}^N \sim \pi_\beta(\cdot \mid s_t) \triangleright sample N actions from behavior policy$

$a_{t:t+h_a}^{\star} \leftarrow \arg\max_{\{a_{t:t+h_a}^i\}_{i=1}^N} Q_\psi^P(s_t, a_{t:t+h_a}) \quad \triangleright take the action with the highest Q-value$

![](images/a58623efae416def610a96c8fdcbc83d7bcb709507f615d5c145db2c2df67e0b.jpg)

<details>
<summary>bar</summary>

| Category | Aggregated Score |
| :--- | :--- |
| FBC | 14 |
| HFBC | 21 |
| IQL | 24 |
| HIQL | 18 |
| SHARSA | 44 |
| OS | 14 |
| NS | 68 |
| QC | 25 |
| DQC-naïve | 36 |
| DQC | 82 |
</details>

Figure 2: Aggregated score across six hardest OGBench environments (10 seeds): cube-{triple/quadruple/octuple}, humanoidmaze-giant, and puzzle-{4x5,4x6}.

# 6 EXPERIMENTAL SETUP

We conduct experiments to evaluate the benefits of decoupling the policy chunk size and the critic chunk size on OGBench (Park et al., 2025a)—a challenging long-horizon, goal-conditioned offline RL benchmark consisting of a diverse set of environments (from manipulation to locomotion). In particular, we use the more difficult environments introduced by Park et al. (2025b) (Figure 7), where multi-step return backups are crucial. These environments require highly complex, long-horizon reasoning. For example, the puzzle tasks require stitching up to 24 atomic motions to solve a combinatorial puzzle with a robot arm, and the humanoidmaze tasks require controlling a highdimensional humanoid robot over 3000 environment steps to navigate a maze. These environments serve as an ideal testbed for our algorithm, which improves upon n-step returns and Q-chunking. We now describe our main comparisons. To start with, we consider several direct ablation baselines where the same algorithm backbone is being used (i.e., implicit value backup and best-of-N sampling):

QC (Q-chunking (Li et al., 2025b)) uses a single critic that has the same chunk length as that of the policy $( i . e . , h = \mathsf { \bar { h } } _ { a } )$ . This baseline tests whether having decoupled chunk sizes is important.

DQC-naïve is a naïve attempt at decoupling the critic chunk size from the policy chunk size, where it takes the QC policy to predict full action chunks of size h but only execute the first $h _ { a }$ actions.

NS: n-step return TD backup. This baseline uses a single one-step critic $( i . e . , Q ( s _ { t } , a _ { t } ) )$ . Compared to DQC with h = n and $h _ { a } = 1$ , this baseline tests whether using a chunked critic is important.

OS: Standard 1-step TD backup. This is the same as NS but with $n = 1$ .

Beyond the ablation baselines, we also consider the following strong goal-conditioned baselines from prior work:

FBC/HFBC: Goal-conditioned and hierarchical goal-conditioned flow behavior cloning baselines considered in Park et al. (2025b).

![](images/f6a73355a42ec60e788aafdbf3134ffde4aa38a65a71c6e8ef3272994c0b9362.jpg)

Figure 3: Offline goal-conditioned RL results (10 seeds). Our method (DQC) uses decoupled critic and policy chunk sizes. QC: Q-chunking (Li et al., 2025b); NS: n-step return backup; OS: 1-step TD-backup; DQC-naïve: same as QC but executes a partial action chunk. 

<table><tr><td>Task</td><td>FBC</td><td>HFBC</td><td>IQL</td><td>HIQL</td><td>SHARSA</td><td>OS</td><td>NS</td><td>QC</td><td>DQC-naïve</td><td>DQC</td></tr><tr><td>cube-triple-100M</td><td>54[51,56]</td><td>56[53,59]</td><td>66[63,67]</td><td>35[31,39]</td><td>83[81,85]</td><td>47[41,53]</td><td>93[91,94]</td><td>20[7,36]</td><td>27[18,38]</td><td>98[98,99]</td></tr><tr><td>cube-quadruple-100M</td><td>34[32,37]</td><td>37[34,40]</td><td>53[52,55]</td><td>24[21,28]</td><td>64[62,68]</td><td>0[0,0]</td><td>27[11,43]</td><td>35[26,43]</td><td>40[29,49]</td><td>92[90,93]</td></tr><tr><td>cube-octuple-1B</td><td>0[0,0]</td><td>28[26,29]</td><td>0[0,0]</td><td>20[17,23]</td><td>34[31,36]</td><td>0[0,0]</td><td>9[6,12]</td><td>0[0,0]</td><td>3[1,5]</td><td>34[33,35]</td></tr><tr><td>humanoidmaze-giant</td><td>1[1,2]</td><td>6[4,8]</td><td>3[2,5]</td><td>24[22,26]</td><td>19[16,23]</td><td>0[0,0]</td><td>95[94,97]</td><td>48[45,52]</td><td>80[77,83]</td><td>92[90,94]</td></tr><tr><td>puzzle-4x5</td><td>0[0,0]</td><td>0[0,0]</td><td>20[19,20]</td><td>0[0,0]</td><td>1[1,2]</td><td>19[18,19]</td><td>93[91,95]</td><td>20[20,20]</td><td>33[29,37]</td><td>96[95,97]</td></tr><tr><td>puzzle-4x6-1B</td><td>1[0,1]</td><td>4[3,5]</td><td>6[3,9]</td><td>9[5,13]</td><td>64[60,68]</td><td>19[19,20]</td><td>91[86,94]</td><td>28[27,30]</td><td>33[28,38]</td><td>83[80,86]</td></tr></table>

Table 2: Comparisons with prior methods (10 seeds). Our method outperforms SHARSA (Park et al., 2025b) (the previous state-of-the-art method on this benchmark) on most environments.

IQL/HIQL (Kostrikov et al., 2022; Park et al., 2023): These are strong goal-conditioned RL methods that train a goal-conditioned value function with implicit value backups and extract a flat (IQL) or hierarchical (HIQL) policy from the value function.

SHARSA (Park et al., 2025b): The previous state-of-the-art method on the long-horizon environments that we evaluate on. The method uses a combination of n-step return and bi-level hierarchical policies.

In our ablation study, we also consider an additional baseline, QC-NS, that uses the idea of decoupled policy chunking and critic chunking $\left( h _ { a } \ < \ h \right)$ , but without using a distilled critic. This baseline simply uses n-step return targets to directly train a critic with a chunk size of $h _ { a }$ without implicit maximization (Equation (35)). The performance of this baseline helps determine how important it is to learn a separate distilled critic for partial action chunks with implicit maximization. We run 10 seeds for all methods, and report the means and the 95% confidence intervals.

# 7 RESULTS

In this section, we present our experimental results to answer the following three questions:

(Q1) Does DQC improve upon n-step return, Q-chunking? Figure 3 compares DQC (ours) to both n-step and QC across six challenging long-horizon GCRL environments, with our method performing on par or better across the board. Table 2 shows DQC also consistently outperforms the previous state-of-the-art method on this benchmark, SHARSA (Park et al., 2025b), on all environments. For each environment, we tune DQC (ours), QC, NS, and OS (see the tuning range in Table 9) and pick the best configuration (Table 7) for hyperparameters used in Figure 3 and Table 2. For all baselines from prior work (SHARSA, HIQL, IQL, HFBC, FBC), we directly use their tuned hyperparameters and run with the same batch size (i.e., 4096) as used in our method and other baselines. See the complete result table for all combinations of $h , n , h _ { a }$ in Section A.

![](images/ce2136e6ea827a6c6791ce4c6f60876c4bc89cd7315b5d2d9572730edfea7376.jpg)

Figure 4: Distilled critic ablations (10 seeds). Each group in the legend contains DQC and its nondistilled counterpart with the same configuration. Our method (DQC) performs on par or better than the non-distilled counterpart across all configurations.   
![](images/2bf0512cc625c55b2200db35d368a61230e80be7808b1ebb41e888819897bb93.jpg)  
Figure 5: Hyperparameter sensitivity analysis on cube-quadruple (10 seeds). Best-of-N: the number of action samples drawn from $\pi _ { \beta } ( \cdot \mid s )$ during policy evaluation; Implicit loss type: the implicit maximization loss function used for distillation and value backup; Batch size: the number of examples used in each gradient step.

(Q2) Is training a separate distilled critic $Q _ { \psi } ^ { P }$ necessary? In Figure 4, we compare DQC to DQC without using the distilled critic across three different $( h , h _ { a } )$ configurations: $( h = 2 5 , h _ { a } = 5 )$ , $( h = 2 5 , h _ { a } = 1 )$ , and $( h = 5 , h _ { a } = 1 )$ ). For configurations with $h _ { a } = 1$ , the baseline without using the distilled critic is the same as the n-step return baseline (with $n = h )$ and for the configuration with $h _ { a } = 5$ , it is the same as combining Q-chunking and n-step return (QC-NS). Across three configurations, DQC performs on par or better than its non-distilled counterpart. This highlights that the a separate distilled critic for the partial action chunk is necessary for the effectiveness of DQC.   
(Q3) How sensitive is DQC to its hyperparameters? Figure 5 shows that our method is not sensitive to the implicit backup method (quantile or expectile), and somewhat sensitive to the implicit parameters $\kappa _ { b } , \kappa _ { d }$ . In particular, DQC is still reasonably effective as long as some form of optimism is employed $( i . e .$ , either $\kappa _ { b } \neq 0 . 5 \mathrm { o r } \kappa _ { d } \neq 0 . 5 )$ . Using no optimism $( \kappa _ { b } = \kappa _ { d } = 0 . 5 )$ results in a big performance drop. The other important hyperparameters are N in the best-of-N policy extraction and the batch size. Having large enough batch size $( i . e . , 4 0 9 6 )$ and $N \left( e . g . , N = 3 2 \right)$ is crucial for good performance, although increasing N further $( e . g . , N = 1 2 8 )$ does not lead to better performance.

# 8 DISCUSSION

We provide a theoretical foundation for action chunking Q-learning and demonstrate how to effectively extract policies from chunked critics. Theoretically, we provide a formal analysis of action chunking Q-learning, identifying the TD backup bias that arises from open-loop inconsistency and characterizing the conditions under which action chunking Q-learning is preferred over n-step return learning and the conditions under which closed-loop execution of the action chunking policy is near-optimal. Empirically, we develop a new technique that enables effective policy extraction from chunked critics with long action chunks, scaling up action chunking Q-learning to much harder environments. Together, these contributions advance the goal of tackling bootstrapping bias in TDlearning. Several challenges remain, indicating promising avenues for future research. For example, our method relies on a fixed policy action chunk size $h _ { a }$ and critic action chunk size h across all states, even though the optimal action chunk size may vary by state. Developing practical methods that can support flexible, state-dependent chunk sizes would be a natural next step.

# ACKNOWLEDGMENTS

This work was supported by DARPA ANSR and ONR N00014-25-1-2060. This research used the Savio computational cluster resource provided by the Berkeley Research Computing program at UC Berkeley. We would like to thank William Chen for discussions and inspiration, especially on the proof for Proposition 4. We would also like to thank Andrew Wagenmaker for suggestions and feedback on the theory (Theorems 1 and 3 and Propositions 2 and 3). We would also like to thank Dibya Ghosh for feedback on an early version of the teaser figure and Ameesh Shah for writing feedback on an early draft of the paper.

# REFERENCES

Anurag Ajay, Aviral Kumar, Pulkit Agrawal, Sergey Levine, and Ofir Nachum. OPAL: Offline primitive discovery for accelerating offline reinforcement learning. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=V69LGwJ0lIN.   
Kamyar Azizzadenesheli, Alessandro Lazaric, and Animashree Anandkumar. Reinforcement learning of pomdps using spectral methods. In Conference on Learning Theory, pp. 193–256. PMLR, 2016.   
Pierre-Luc Bacon, Jean Harb, and Doina Precup. The option-critic architecture. In Proceedings of the AAAI conference on artificial intelligence, volume 31, 2017.   
Akhil Bagaria and George Konidaris. Option discovery using deep skill chaining. In International Conference on Learning Representations, 2019.   
Akhil Bagaria, Ben Abbatematteo, Omer Gottesman, Matt Corsaro, Sreehari Rammohan, and George Konidaris. Effectively learning initiation sets in hierarchical reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.   
Philip J Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. Efficient online reinforcement learning with offline data. In International Conference on Machine Learning, pp. 1577–1594. PMLR, 2023.   
Christian Bayer, Boualem Djehiche, Eliza Rezvanova, and Raul Fidel Tempone. Continuous time stochastic optimal control under discrete time partial observations. arXiv preprint arXiv:2407.18018, 2024.   
Andrew Bennett and Nathan Kallus. Proximal reinforcement learning: Efficient off-policy evaluation in partially observed markov decision processes. Operations Research, 72(3):1071–1086, 2024.   
Andrew Bennett, Nathan Kallus, Lihong Li, and Ali Mousavi. Off-policy evaluation in infinitehorizon reinforcement learning with latent confounders. In International Conference on Artificial Intelligence and Statistics, pp. 1999–2007. PMLR, 2021.   
Kevin Black, Manuel Y Galliker, and Sergey Levine. Real-time execution of action chunking flow policies. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=UkR2zO5uww.   
Boyuan Chen, Chuning Zhu, Pulkit Agrawal, Kaiqing Zhang, and Abhishek Gupta. Self-supervised reinforcement learning that transfers using random features. Advances in Neural Information Processing Systems, 36, 2024.   
Xinyue Chen, Che Wang, Zijian Zhou, and Keith W. Ross. Randomized ensembled double q-learning: Learning fast without a model. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=AY8zfZm0tDd.   
Nuttapong Chentanez, Andrew Barto, and Satinder Singh. Intrinsically motivated reinforcement learning. Advances in neural information processing systems, 17, 2004.   
Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, pp. 02783649241273668, 2023.   
Imre Csiszár. On information-type measure of difference of probability distributions and indirect observations. Studia Sci. Math. Hungar., 2:299–318, 1967.

Christian Daniel, Gerhard Neumann, Oliver Kroemer, and Jan Peters. Hierarchical relative entropy policy search. Journal of Machine Learning Research, 17(93):1–50, 2016.   
Peter Dayan and Geoffrey E Hinton. Feudal reinforcement learning. Advances in neural information processing systems, 5, 1992.   
Kristopher De Asis, J Hernandez-Garcia, G Holland, and Richard Sutton. Multi-step reinforcement learning: A unifying algorithm. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.   
Thomas G Dietterich. Hierarchical reinforcement learning with the maxq value function decomposition. Journal of artificial intelligence research, 13:227–303, 2000.   
Pierluca D’Oro, Max Schwarzer, Evgenii Nikishin, Pierre-Luc Bacon, Marc G Bellemare, and Aaron Courville. Sample-efficient reinforcement learning by breaking the replay ratio barrier. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=OpC-9aBBVJe.   
Paul Dupuis and Hui Wang. Optimal stopping with random intervention times. Advances in Applied probability, 34(1):141–157, 2002.   
Ishan P Durugkar, Clemens Rosenbaum, Stefan Dernbach, and Sridhar Mahadevan. Deep reinforcement learning with macro-actions. arXiv preprint arXiv:1606.04615, 2016.   
William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio, Hugo Larochelle, Mark Rowland, and Will Dabney. Revisiting fundamentals of experience replay. In International conference on machine learning, pp. 3061–3071. PMLR, 2020.   
Roy Fox, Sanjay Krishnan, Ion Stoica, and Ken Goldberg. Multi-level discovery of deep options. CoRR, abs/1703.08294, 2017. URL http://arxiv.org/abs/1703.08294.   
Kevin Frans, Seohong Park, Pieter Abbeel, and Sergey Levine. Unsupervised zero-shot reinforcement learning via functional reward encodings. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 13927–13942. PMLR, 21–27 Jul 2024. URL https://proceedings. mlr.press/v235/frans24a.html.   
Zuyue Fu, Zhengling Qi, Zhaoran Wang, Zhuoran Yang, Yanxun Xu, and Michael R Kosorok. Offline reinforcement learning with instrumental variables in confounded markov decision processes. arXiv preprint arXiv:2209.08666, 2022.   
Jonas Gehring, Gabriel Synnaeve, Andreas Krause, and Nicolas Usunier. Hierarchical skills for efficient exploration. Advances in Neural Information Processing Systems, 34:11553–11564, 2021.   
Philippe Hansen-Estruch, Ilya Kostrikov, Michael Janner, Jakub Grudzien Kuba, and Sergey Levine. IDQL: Implicit Q-learning as an actor-critic method with diffusion policies. arXiv preprint arXiv:2304.10573, 2023.   
Hao Hu, Yiqin Yang, Jianing Ye, Ziqing Mai, and Chongjie Zhang. Unsupervised behavior extraction via random intent priors. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=4vGVQVz5KG.   
Edward L Ionides. Truncated importance sampling. Journal of Computational and Graphical Statistics, 17(2):295–311, 2008.   
Tommi Jaakkola, Michael Jordan, and Satinder Singh. Convergence of stochastic iterative dynamic programming algorithms. Advances in neural information processing systems, 6, 1993.   
Jihwan Jeong, Xiaoyu Wang, Michael Gimelfarb, Hyunwoo Kim, Baher abdulhai, and Scott Sanner. Conservative bayesian model-based value expansion for offline policy optimization. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/ forum?id=dNqxZgyjcYA.

Nathan Kallus and Angela Zhou. Confounding-robust policy evaluation in infinite-horizon reinforcement learning. Advances in neural information processing systems, 33:22293–22304, 2020.   
Nathan Kallus and Angela Zhou. Minimax-optimal policy learning under unobserved confounding. Management Science, 67(5):2870–2890, 2021.   
Chinmaya Kausik, Yangyi Lu, Kevin Tan, Maggie Makar, Yixin Wang, and Ambuj Tewari. Offline policy evaluation and optimization under confounding. In International Conference on Artificial Intelligence and Statistics, pp. 1459–1467. PMLR, 2024.   
Teun Kloek and Herman K Van Dijk. Bayesian estimates of equation system parameters: an application of integration by monte carlo. Econometrica: Journal of the Econometric Society, pp. 1–19, 1978.   
Anita De Mello Koch, Akhil Bagaria, Bingnan Huo, Cameron Allen, Zhiyuan Zhou, and George Konidaris. Learning transferable sub-goals by hypothesizing generalizing features, 2025. URL https://openreview.net/forum?id=OvrmA3GMiX.   
George Konidaris, Scott Niekum, and Philip S Thomas. TDγ: Re-evaluating complex backups in temporal difference learning. Advances in Neural Information Processing Systems, 24, 2011.   
George Dimitri Konidaris. Autonomous robot skill acquisition. University of Massachusetts Amherst, 2011.   
Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit qlearning. In International Conference on Learning Representations, 2022. URL https:// openreview.net/forum?id=68n2s9ZJWF8.   
Tadashi Kozuno, Yunhao Tang, Mark Rowland, Rémi Munos, Steven Kapturowski, Will Dabney, Michal Valko, and David Abel. Revisiting Peng’s Q (λ) for modern reinforcement learning. In International Conference on Machine Learning, pp. 5794–5804. PMLR, 2021.   
Tejas D Kulkarni, Karthik Narasimhan, Ardavan Saeedi, and Josh Tenenbaum. Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation. Advances in neural information processing systems, 29, 2016.   
Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative Q-learning for offline reinforcement learning. Advances in Neural Information Processing Systems, 33:1179–1191, 2020.   
Seunghyun Lee, Younggyo Seo, Kimin Lee, Pieter Abbeel, and Jinwoo Shin. Offline-to-online reinforcement learning via balanced replay and pessimistic Q-ensemble. In Conference on Robot Learning, pp. 1702–1712. PMLR, 2022.   
Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.   
Ge Li, Dong Tian, Hongyi Zhou, Xinkai Jiang, Rudolf Lioutikov, and Gerhard Neumann. TOP-ERL: Transformer-based off-policy episodic reinforcement learning. In The Thirteenth International Conference on Learning Representations, 2025a. URL https://openreview.net/forum?id= N4NhVN30ph.   
Qiyang Li, Zhiyuan Zhou, and Sergey Levine. Reinforcement learning with action chunking. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b. URL https://openreview.net/forum?id=XUks1Y96NR.   
Toru Lin, Yu Zhang, Qiyang Li, Haozhi Qi, Brent Yi, Sergey Levine, and Jitendra Malik. Learning visuotactile skills with two multifingered hands. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 5637–5643. IEEE, 2025.   
Qinghua Liu, Alan Chung, Csaba Szepesvári, and Chi Jin. When is partially observable reinforcement learning not scary? In Conference on Learning Theory, pp. 5175–5220. PMLR, 2022.   
Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=XVjTT1nw5z.

Amy McGovern and Richard S Sutton. Macro-actions in reinforcement learning: An empirical analysis. 1998.   
Ishai Menache, Shie Mannor, and Nahum Shimkin. Q-cut—dynamic discovery of sub-goals in reinforcement learning. In Machine Learning: ECML 2002: 13th European Conference on Machine Learning Helsinki, Finland, August 19–23, 2002 Proceedings 13, pp. 295–306. Springer, 2002.   
Josh Merel, Leonard Hasenclever, Alexandre Galashov, Arun Ahuja, Vu Pham, Greg Wayne, Yee Whye Teh, and Nicolas Heess. Neural probabilistic motor primitives for humanoid control. In International Conference on Learning Representations, 2019. URL https://openreview.net/ forum?id=BJl6TjRcY7.   
Rui Miao, Zhengling Qi, and Xiaoke Zhang. Off-policy evaluation for episodic partially observable markov decision processes under non-parametric models. Advances in Neural Information Processing Systems, 35:593–606, 2022.   
Prabhat K Mishra, Debasish Chatterjee, and Daniel E Quevedo. Stochastic predictive control under intermittent observations and unreliable actions. Automatica, 118:109012, 2020.   
Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare. Safe and efficient off-policy reinforcement learning. Advances in neural information processing systems, 29, 2016.   
Ofir Nachum, Shixiang Shane Gu, Honglak Lee, and Sergey Levine. Data-efficient hierarchical reinforcement learning. Advances in neural information processing systems, 31, 2018.   
Mitsuhiko Nakamoto, Simon Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, and Sergey Levine. Cal-QL: Calibrated offline RL pre-training for efficient online finetuning. Advances in Neural Information Processing Systems, 36, 2024.   
Hongseok Namkoong, Ramtin Keramati, Steve Yadlowsky, and Emma Brunskill. Offpolicy policy evaluation for sequential decisions under unobserved confounding. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 18819–18831. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/file/ da21bae82c02d1e2b8168d57cd3fbab7-Paper.pdf.   
Soroush Nasiriany, Tian Gao, Ajay Mandlekar, and Yuke Zhu. Learning and retrieval from prior data for skill-based imitation learning. In Conference on Robot Learning, 2022.   
Kei Noba and Kazutoshi Yamazaki. On stochastic control under poisson observations: optimality of a barrier strategy in a general l\’evy model. arXiv preprint arXiv:2210.00501, 2022.   
Alexandros Paraschos, Christian Daniel, Jan R Peters, and Gerhard Neumann. Probabilistic movement primitives. Advances in neural information processing systems, 26, 2013.   
Kwanyoung Park and Youngwoon Lee. Model-based offline reinforcement learning with lower expectile q-learning. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=OATPSB5JK1.   
Seohong Park, Dibya Ghosh, Benjamin Eysenbach, and Sergey Levine. HIQL: Offline goalconditioned RL with latent states as actions. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=cLQCCtVDuW.   
Seohong Park, Tobias Kreiman, and Sergey Levine. Foundation policies with hilbert representations. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview. net/forum?id=LhNsSaAKub.   
Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. OGBench: Benchmarking offline goal-conditioned RL. In The Thirteenth International Conference on Learning Representations, 2025a. URL https://openreview.net/forum?id=M992mjgKzI.

Seohong Park, Kevin Frans, Deepinder Mann, Benjamin Eysenbach, Aviral Kumar, and Sergey Levine. Horizon reduction makes RL scalable. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b. URL https://openreview.net/forum?id= hguaupzLCU.   
Jing Peng and Ronald J Williams. Incremental multi-step Q-learning. In Machine Learning Proceedings 1994, pp. 226–232. Elsevier, 1994.   
Xue Bin Peng, Glen Berseth, KangKang Yin, and Michiel Van De Panne. Deeploco: Dynamic locomotion skills using hierarchical deep reinforcement learning. Acm transactions on graphics (tog), 36(4):1–13, 2017.   
Karl Pertsch, Youngwoon Lee, and Joseph Lim. Accelerating reinforcement learning with learned skill priors. In Conference on robot learning, pp. 188–204. PMLR, 2021.   
Doina Precup, Richard S Sutton, and Satinder Singh. Eligibility traces for off-policy policy evaluation. In ICML, volume 2000, pp. 759–766. Citeseer, 2000.   
Martin Riedmiller, Roland Hafner, Thomas Lampe, Michael Neunert, Jonas Degrave, Tom Wiele, Vlad Mnih, Nicolas Heess, and Jost Tobias Springenberg. Learning by playing solving sparse reward tasks from scratch. In International conference on machine learning, pp. 4344–4353. PMLR, 2018.   
Mark Rowland, Will Dabney, and Rémi Munos. Adaptive trade-offs in off-policy learning. In International Conference on Artificial Intelligence and Statistics, pp. 34–44. PMLR, 2020.   
Younggyo Seo and Pieter Abbeel. Coarse-to-fine q-network with action sequence for data-efficient reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=VoFXUNc9Zh.   
Younggyo Seo, Jafar Uruç, and Stephen James. Continuous control with coarse-to-fine reinforcement learning. In 8th Annual Conference on Robot Learning, 2024. URL https://openreview.net/ forum?id=WjDR48cL3O.   
Tanmay Shankar and Abhinav Gupta. Learning robot skills with temporal variational inference. In International Conference on Machine Learning, pp. 8624–8633. PMLR, 2020.   
Chengchun Shi, Masatoshi Uehara, Jiawei Huang, and Nan Jiang. A minimax learning approach to offpolicy evaluation in confounded partially observable markov decision processes. In International Conference on Machine Learning, pp. 20057–20094. PMLR, 2022.   
Chengchun Shi, Jin Zhu, Ye Shen, Shikai Luo, Hongtu Zhu, and Rui Song. Off-policy confidence interval estimation with confounded markov decision process. Journal of the American Statistical Association, 119(545):273–284, 2024.   
Max Simchowitz, Daniel Pfrommer, and Ali Jadbabaie. The pitfalls of imitation learning when actions are continuous. In Nika Haghtalab and Ankur Moitra (eds.), Proceedings of Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pp. 5248–5351. PMLR, 30 Jun–04 Jul 2025. URL https://proceedings.mlr.press/v291/ simchowitz25a.html.   
Özgür ¸Sim¸sek and Andrew G. Barto. Betweenness centrality as a basis for forming skills. Workingpaper, University of Massachusetts Amherst, April 2007.   
Aravind Srinivas, Ramnandan Krishnamurthy, Peeyush Kumar, and Balaraman Ravindran. Option discovery in hierarchical reinforcement learning using spatio-temporal clustering. arXiv preprint arXiv:1605.05359, 2016.   
Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.   
Richard S Sutton, Doina Precup, and Satinder Singh. Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artificial intelligence, 112(1-2):181–211, 1999.

Denis Tarasov, Vladislav Kurenkov, Alexander Nikulin, and Sergey Kolesnikov. Revisiting the minimalist approach to offline reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.   
Guy Tennenholtz, Uri Shalit, and Shie Mannor. Off-policy evaluation in partially observable environments. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 10276– 10283, 2020.   
Philip S Thomas, Scott Niekum, Georgios Theocharous, and George Konidaris. Policy evaluation using the Ω-return. Advances in Neural Information Processing Systems, 28, 2015.   
Dong Tian, Ge Li, Hongyi Zhou, Onur Celik, and Gerhard Neumann. Chunking the critic: A transformer-based soft actor-critic with N-step returns. arXiv preprint arXiv:2503.03660, 2025.   
Ahmed Touati, Jérémy Rapin, and Yann Ollivier. Does zero-shot reinforcement learning exist? In The Eleventh International Conference on Learning Representations, 2022.   
Stephen Tu, Alexander Robey, Tingnan Zhang, and Nikolai Matni. On the sample complexity of stability constrained imitation learning. In Learning for Dynamics and Control Conference, pp. 180–191. PMLR, 2022.   
Alexander Vezhnevets, Volodymyr Mnih, Simon Osindero, Alex Graves, Oriol Vinyals, John Agapiou, et al. Strategic attentive writer for learning macro-actions. Advances in neural information processing systems, 29, 2016.   
Alexander Sasha Vezhnevets, Simon Osindero, Tom Schaul, Nicolas Heess, Max Jaderberg, David Silver, and Koray Kavukcuoglu. Feudal networks for hierarchical reinforcement learning. In International conference on machine learning, pp. 3540–3549. PMLR, 2017.   
Hui Wang. Some control problems with random intervention times. Advances in Applied Probability, 33(2):404–422, 2001.   
Max Wilcoxson, Qiyang Li, Kevin Frans, and Sergey Levine. Leveraging skills from unlabeled prior data for efficient online exploration. In International Conference on Machine Learning (ICML), 2025.   
Yihong Wu. Lecture notes on information-theoretic methods for high-dimensional statistics. Lecture Notes for ECE598YW (UIUC), 16:15, 2017.   
Kevin Xie, Homanga Bharadhwaj, Danijar Hafner, Animesh Garg, and Florian Shkurti. Latent skill planning for exploration and transfer. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=jXe91kq3jAq.   
Shuhao Yan, Mark Cannon, and Paul J Goulart. Stochastic output feedback MPC with intermittent observations. Automatica, 141:110282, 2022.   
Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023. doi: 10.15607/RSS.2023.XIX.016.

# A FULL RESULTS

Table 3 reports the performance of our method (DQC) and baselines for all hyperparameter configurations. All of them use the same hyperparameters in Table 5 with the only exception that SHARSA, HIQL, IQL, FBC, and HFBC handle goal-sampling for training behavior cloning policies differently. We discuss this in more details in Section C.

<table><tr><td>Method</td><td></td><td></td><td>c3-100M</td><td>c4-100M</td><td>c8-1B</td><td>hg</td><td>p45</td><td>p46-1B</td></tr><tr><td>DQC</td><td> $h = 25$ </td><td> ${h}_{a} = 1$ </td><td>76[73,80]</td><td>45[41,49]</td><td>10[8,11]</td><td>92[90,94]</td><td>91[89,92]</td><td>83[80,86]</td></tr><tr><td>DQC</td><td> $h = 25$ </td><td> ${h}_{a} = 5$ </td><td>98[98,99]</td><td>92[90,93]</td><td>34[33,35]</td><td>51[48,54]</td><td>96[95,97]</td><td>68[66,71]</td></tr><tr><td>DQC</td><td> $h = 5$ </td><td> ${h}_{a} = 1$ </td><td>95[94,97]</td><td>84[83,86]</td><td>0[0,0]</td><td>19[15,22]</td><td>90[88,92]</td><td>44[42,47]</td></tr><tr><td>DQC-naïve</td><td> $h = 25$ </td><td> ${h}_{a} = 1$ </td><td>14[8,22]</td><td>16[9,23]</td><td>1[0,2]</td><td>22[20,24]</td><td>32[28,36]</td><td>33[29,37]</td></tr><tr><td>DQC-naïve</td><td> $h = 25$ </td><td> ${h}_{a} = 5$ </td><td>27[18,38]</td><td>27[15,39]</td><td>3[1,5]</td><td>0[0,1]</td><td>33[29,37]</td><td>33[28,38]</td></tr><tr><td>DQC-naïve</td><td> $h = 5$ </td><td> ${h}_{a} = 1$ </td><td>16[7,30]</td><td>40[29,49]</td><td>0[0,0]</td><td>80[77,83]</td><td>20[20,20]</td><td>26[25,28]</td></tr><tr><td>QC</td><td> $h = 25$ </td><td> ${h}_{a} = {25}$ </td><td>21[13,31]</td><td>12[7,18]</td><td>0[0,0]</td><td>0[0,0]</td><td>30[27,33]</td><td>37[33,42]</td></tr><tr><td>QC</td><td> $h = 5$ </td><td> ${h}_{a} = 5$ </td><td>20[7,36]</td><td>35[26,43]</td><td>0[0,0]</td><td>48[45,52]</td><td>20[20,20]</td><td>28[27,30]</td></tr><tr><td>QC-NS</td><td> $n = 25$ </td><td> ${h}_{a} = 5$ </td><td>51[22,80]</td><td>53[28,77]</td><td>18[10,25]</td><td>60[58,61]</td><td>95[94,96]</td><td>95[93,97]</td></tr><tr><td>NS</td><td> $n = 25$ </td><td> ${h}_{a} = 1$ </td><td>30[26,35]</td><td>19[11,28]</td><td>9[6,12]</td><td>95[94,97]</td><td>89[87,91]</td><td>91[86,94]</td></tr><tr><td>NS</td><td> $n = 5$ </td><td> ${h}_{a} = 1$ </td><td>93[91,94]</td><td>27[11,43]</td><td>1[0,3]</td><td>89[87,91]</td><td>93[91,95]</td><td>56[48,63]</td></tr><tr><td>OS</td><td> $n = 1$ </td><td> ${h}_{a} = 1$ </td><td>47[41,53]</td><td>0[0,0]</td><td>0[0,0]</td><td>0[0,0]</td><td>19[18,19]</td><td>19[19,20]</td></tr><tr><td>FBC</td><td></td><td></td><td>54[51,56]</td><td>34[32,37]</td><td>0[0,0]</td><td>1[1,2]</td><td>0[0,0]</td><td>1[0,1]</td></tr><tr><td>HFBC</td><td></td><td></td><td>56[53,59]</td><td>37[34,40]</td><td>28[26,29]</td><td>6[4,8]</td><td>0[0,0]</td><td>4[3,5]</td></tr><tr><td>IQL</td><td></td><td></td><td>66[63,67]</td><td>53[52,55]</td><td>0[0,0]</td><td>3[2,5]</td><td>20[19,20]</td><td>6[3,9]</td></tr><tr><td>HIQL</td><td></td><td></td><td>35[31,39]</td><td>24[21,28]</td><td>20[17,23]</td><td>24[22,26]</td><td>0[0,0]</td><td>9[5,13]</td></tr><tr><td>SHARSA</td><td></td><td></td><td>83[81,85]</td><td>64[62,68]</td><td>34[31,36]</td><td>19[16,23]</td><td>1[1,2]</td><td>64[60,68]</td></tr></table>

Table 3: Complete results for all hyperparameter configurations across different combinations of $h ,$ n and $h _ { a }$ (10 seeds). We adopt the following abbreviations: c3=cube-triple, c4=cube-quadruple, c8=cube-octuple, hg=humanoidmaze-giant, p45=puzzle-4x5, p46=puzzle-4x6. The hyperparameters used are specified in Tables 7 and 8.

![](images/3cc9eeef7b992617231a5bed163dae36fef2cb4e17e66c36eb6e5355daa5f872.jpg)  
Figure 6: Batch size sensitivity (10 seeds). Large batch size is crucial for DQC’s performance especially on hard tasks (e.g., cube-quadruple, cube-octuple, puzzle-4x5 and puzzle-4x6).

# B ENVIRONMENTS AND DATASETS

To evaluate our method, we consider 8 goal-conditioned environments in OGBench with varying difficulties (Figure 7). The dataset size, episode length, and the action dimension for each environment is available in Table 4. We describe each of the environments and the datasets we use as follows.

Environment cube-\*: We consider three cube environments (cube-triple, cube-quadruple, cube-octuple). As the names suggest, the goal of these environments involve using a robot arm to manipulate 3/4/8 cubes from some initial configuration to some specified goal configuration. We use the same five evaluation tasks used in OGBench (Park et al., 2025a) for cube-triple and cube-quadruple and the same five evaluation tasks used in Park et al. (2025b) for cube-octuple. We refer the environment detail to the corresponding references.

<table><tr><td>Environment</td><td>Dataset Size</td><td>Episode Length</td><td>Action Dimension (A)</td></tr><tr><td>cube-triple-100M</td><td>100M</td><td>1000</td><td>5</td></tr><tr><td>cube-quadruple-100M</td><td>100M</td><td>1000</td><td>5</td></tr><tr><td>cube-octuple-1B</td><td>1B</td><td>1500</td><td>5</td></tr><tr><td>humanoidmaze-giant</td><td>4M (default)</td><td>4000</td><td>21</td></tr><tr><td>puzzle-4x5</td><td>3M (default)</td><td>1000</td><td>5</td></tr><tr><td>puzzle-4x6-1B</td><td>1B</td><td>1000</td><td>5</td></tr></table>

Table 4: Environment metadata. For both humanoidmaze-giant and puzzle-4x5, we use the default dataset that is released in the original OGBench benchmark (Park et al., 2025a). For the other environments, we use larger datasets as we find them to be essential for achieving good performances on these environments.

Environment humanoidmaze-\*: We also consider the hardest locomotion environment available in OGBench. The goal of the environment is to control and navigate a humanoid agent from some initial location to some specified goal location in a 16 × 12 maze. This environment also has the longest episode length (4000, more than twice as long as the second longest episode length as used in cube-octuple). We refer the environment detail to Park et al. (2025a).

Environment puzzle-\*: Finally, we consider two environments that involve solving a combinatorial puzzle with a robot arm. The puzzle consists of a board of 4 × 5 or 4 × 6 buttons, organized as a regular grid (4 rows and 5 or 6 columns). Each button has a binary state. Whenever the end-effector of the arm touches a button, the button and all its adjacent four buttons (three or two if the button is on the edge of the grid or in the corner) flip its binary state. The goal of the environment is to transform the board from some initial state to some specified goal state. We refer the environment detail to Park et al. (2025b).

At the test-time/evaluation-time, the goal-conditioned agent is tested on five evaluation tasks for each of the six environments we consider. The overall success rate is the average over 5 tasks with 50 evaluation trials each. For the prior baselines, SHARSA, HIQL, IQL, HFBC and FBC, we run 15 evaluation trials for each task, following Park et al. (2025b).

Datasets. We use play datasets for all cube-\* and puzzle-\* environments and navigate dataset for humanoidmaze-\*. We use the original datasets available for humanoidmaze-giant and puzzle-4x5 because they are sufficient for solving the environments. Using larger datasets on these environments do not help differentiate among different methods/baselines. For each of the other environments, we use the largest dataset available from Park et al. (2025b) as we find it to be necessary to solve these environments (or achieve non-trivial performance on cube-octuple).

# C HYPERPARAMETERS AND IMPLEMENTATION DETAILS

Hyperparameters. Table 5 describes the common hyperparameters used in all our experiments. Tables 7 and 8 describe the environment-specific hyperparameters and Table 9 describes the range of hyperparameters we use for tuning each method.

Goal-conditioned RL implementation details. While we have described in the main body of the paper how DQC works as a general RL algorithm, we have not touched on how DQC and similarly all our baselines works with the goal-conditioned RL (GCRL) setting. We consider the setting where we have access to an oracle goal representation Ψ : S → G where G is the goal space (see Table 6 for the oracle goal representation description for each environment). The goal-conditioned reward function $r : ( s , g ) \mapsto \mathbb { I } _ { \Psi ( s ) = g }$ is a binary reward function where its output is 1 if the goal g is reached by the current state s. We can treat g as part of an extended state $\tilde { s } = [ s , g ] \in \tilde { S } = S \times \mathcal { G }$ and learn value functions $( e . g . , Q _ { \phi } ( \tilde { s } , a ) )$ normally with such extended state.

![](images/7a400c475f9aafebc0cc59704a8892a55bb3b3841f5815ae57237267ab8aed8d.jpg)

<details>
<summary>natural_image</summary>

3D rendering of a robotic arm in space with colored cubes on a dark floor (no text or symbols)
</details>

cube-triple

![](images/a5adaf2c08622dd111ee402fdde33d66ee24fbc6cdea5e897ad6dd8ada096179.jpg)

<details>
<summary>natural_image</summary>

3D rendering of a robotic arm interacting with colorful cubes against a starry space background (no text or symbols)
</details>

cube-quadruple

![](images/e604814f40e3dad21895a975a97c7708754a1e627cb398f2c8906ae37e80502b.jpg)

<details>
<summary>natural_image</summary>

3D rendering of a robotic arm interacting with colorful cubes against a starry background (no text or symbols)
</details>

cube-octuple

![](images/be12d7c615db9d4f68a3848f431ba2c630b5483a93e7385ffaa9187a7d232bcf.jpg)

<details>
<summary>natural_image</summary>

Abstract geometric pattern with interlocking lines forming a maze-like structure (no text or symbols)
</details>

humanoidmaze-giant

![](images/ab148f58ad41943a2f58c4c4852a2f081e36a0352118ec397c28117895368d90.jpg)

<details>
<summary>natural_image</summary>

3D illustration of a robotic arm interacting with a grid of colored dots (red, blue, white) on a dark background, no text or symbols present.
</details>

puzzle-4x5

![](images/a12b5b7677f58a9eabf82fcb8b7170c2ce14b37cdd63ec65641d8635506cd541.jpg)

<details>
<summary>natural_image</summary>

3D illustration of a robotic arm interacting with a grid of colored dots (no text or symbols)
</details>

puzzle-4x6   
Figure 7: Environments used in our experiments.

A common practical trick in the GCRL setting is goal relabeling. That is, during training for each (s, a) pair in the training batch, a goal g is sampled from some distribution $( i . e . , p ^ { \mathcal { D } } ( \cdot \mid \bar { s } , a ) )$ and the reward of the transition is relabeled with the goal-conditioned reward function. Following Park et al. (2025b), the goal distribution $P ^ { g } ( \cdot \mid s , a ) : \mathcal { S } \times \mathcal { A }  \Delta _ { \mathcal { G } }$ is a mixture of four distributions, conditioned on the training state-action example:

$$
P ^ {g} = w _ {\text { cur }} P _ {\text { cur }} ^ {g} + w _ {\text { geom }} P _ {\text { geom }} ^ {g} + w _ {\text { traj }} P _ {\text { traj }} ^ {g} + w _ {\text { rand }} P _ {\text { rand }} ^ {g}, \tag {39}
$$

where

1. $P _ { \mathrm { c u r } } ^ { g } ( \cdot \mid s , a ) = \delta _ { \Psi ( s ) } ;$ the goal is the same as the current state;   
2. $P _ { \mathrm { g e o m } } ^ { g } ( \cdot \mid s , a )$ : geometric distribution over the future states in the same trajectory that $( s , a )$   
3. $P _ { \mathrm { t r a i } } ^ { g } ( \cdot \mid s , a )$ : uniform distribution over the future states in the same trajectory that $( s , a )$ is from; and finally   
4. $P _ { \mathrm { r a n d } } ^ { g } ( \cdot \mid s , a ) = \Psi ( \mathcal { U } _ { D ( s ) } )$ : uniform distribution over the dataset $( \mathcal { D } ( s )$ is the distribution

and $w _ { \mathrm { c u r } } , w _ { \mathrm { g e o m } } , w _ { \mathrm { t r a j } } , w _ { \mathrm { r a n d } } > 0$ are the corresponding weights for each of the distribution components with $w _ { \mathrm { c u r } } + w _ { \mathrm { g e o m } } + w _ { \mathrm { t r a j } } + w _ { \mathrm { r a n d } } = 1$ .

In practice, it has been found to be beneficial to use a separate set of goal sampling weights for TD backup (Park et al., 2025a) $( i . e . , ( w _ { \mathrm { c u r } } ^ { \mathrm { v } } , w _ { \mathrm { g e o m } } ^ { \mathrm { v } } , w _ { \mathrm { t r a j } } ^ { \mathrm { v } } , w _ { \mathrm { r a n d } } ^ { \mathrm { v } } ) )$ and for policy learning $( i . e . ,$ $( w _ { \mathrm { c u r } } ^ { \mathrm { p } } , w _ { \mathrm { g e o m } } ^ { \mathrm { p } } , w _ { \mathrm { t r a j } } ^ { \mathrm { p } } , w _ { \mathrm { r a n d } } ^ { \mathrm { p } } ) )$ . However, in our implementation of DQC/QC/NS/OS, we do not train aour policy extraction is done entirely at test-time by best-of-N sampling from an unconditional (i.e., not goal-conditioned) behavior policy $\pi _ { \beta }$ . In particular, we use an unconditioned flow policy $\pi _ { \beta } ( \cdot \mid s )$ that is parameterized by a velocity field vβ : $S \times \mathbb { R } ^ { A } \times [ 0 , 1 ] \to \mathbb { R } ^ { A }$ that is trained with the standard flow-matching objective:

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Batch size</td><td>4096</td></tr><tr><td>Discount factor ( $\gamma$ )</td><td>0.999</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $3 \times 10^{-4}$ </td></tr><tr><td>Target network update rate ( $\lambda$ )</td><td> $5 \times 10^{-3}$ </td></tr><tr><td>Critic ensemble size ( $K$ )</td><td>2</td></tr><tr><td>Critic target</td><td> $\min(Q_1, Q_2)$  for cube- $*$  $(Q_1 + Q_2)/2$  for puzzle- $*$  and humanoid- $*$ </td></tr><tr><td>Value loss type</td><td>binary cross entropy</td></tr><tr><td>Best-of-N sampling ( $N$ )</td><td>32</td></tr><tr><td>Number of flow steps</td><td>10</td></tr><tr><td>Number of training steps</td><td> $10^6$ </td></tr><tr><td>Network width</td><td>1024</td></tr><tr><td>Network depth</td><td>4 hidden layers</td></tr><tr><td>Value goal sampling ( $w_{cur}^{v}, w_{geom}^{v}, w_{traj}^{v}, w_{rand}^{v}$ )</td><td>(0.2, 0, 0.5, 0.3)</td></tr><tr><td>Actor goal sampling ( $w_{cur}^{p}, w_{geom}^{p}, w_{traj}^{p}, w_{rand}^{p}$ )</td><td>DQC/QC/NS/OS:  $\pi_\beta$  is not goal-conditionedSHARSA (cube): (0, 1, 0, 0)SHARSA (puzzle): (0, 0, 1, 0)SHARSA (humanoidmaze): (0, 0, 1, 0)</td></tr></table>

Table 5: Common hyperparameters. For the GCRL goal-sampling distribution we follow the same hyperparameters used in Park et al. (2025b). 

<table><tr><td>Environment</td><td>Goal Representation (Ψ)</td><td>Goal Domain (G)</td></tr><tr><td>cube-triple</td><td>(x,y,z) of three cubes (rel. to center)</td><td> $\mathbb{R}^{9}$ </td></tr><tr><td>cube-quadruple</td><td>(x,y,z) of four cubes (rel. to center)</td><td> $\mathbb{R}^{12}$ </td></tr><tr><td>cube-octuple</td><td>(x,y,z) of eight cubes (rel. to center)</td><td> $\mathbb{R}^{24}$ </td></tr><tr><td>humanoidmaze-giant</td><td>(x,y) of the humanoid</td><td> $\mathbb{R}^{2}$ </td></tr><tr><td>puzzle-4x5</td><td>the binary state for each button</td><td> $\{0,1\}^{20}$ </td></tr><tr><td>puzzle-4x6</td><td>the binary state for each button</td><td> $\{0,1\}^{24}$ </td></tr></table>

Table 6: Oracle goal representation description for each environment. Following Park et al. (2025b), we assume access to an oracle goal representation for each environment. More detailed definition of these oracle goal representations is available in OGBench (Park et al., 2025a).

$$
L _ {\mathrm{FM}} (\beta) = \mathbb {E} _ {u \sim \mathcal {U} [ 0, 1 ], z \sim \mathcal {N}, (s, a) \sim \mathcal {D}} \left[ \| v _ {\beta} (s, (1 - u) z + u a, u) - a + z \| _ {2} ^ {2} \right] \tag {40}
$$

For SHARSA, we use the official implementation where both flow policies (high-level and lowlevel) are goal-conditioned (and thus are trained with the goal distribution mixture specified by $w _ { \mathrm { c u r } } ^ { \mathrm { p } } , w _ { \mathrm { g e o m } } ^ { \mathrm { p } } , w _ { \mathrm { t r a j } } ^ { \mathrm { p } } , w _ { \mathrm { r a n d } } ^ { \mathrm { p } } )$ , wp geom , w traj i,wPan wprand). The goal sampling distribution for training the value networks (for all methods) and the goal sampling distribution for the policy networks (for SHARSA only) are provided in Table 5.

<table><tr><td>Environment</td><td>DQC $(h, h_a, \kappa_b, \kappa_d)$ </td><td>DQC-naïve $(h, h_a, \kappa_b)$ </td><td>QC-NS $(h, h_a, \kappa_b)$ </td><td>QC $(h = h_a, \kappa_b)$ </td><td>NS $(n, \kappa_b)$ </td><td>OS $(\kappa_b)$ </td><td>SHARSA $(n)$ </td><td>HIQL $(h, \kappa, \alpha)$ </td><td>IQL $(\alpha)$ </td><td>HFBC $(h)$ </td></tr><tr><td>cube-triple-100M</td><td>(25, 5, 0.93, 0.8)</td><td>(25, 5, 0.93)</td><td>(25, 5, 0.93)</td><td>(5, 0.93)</td><td>(5, 0.5)</td><td>0.5</td><td>25</td><td>(25, 0.5, 10)</td><td>3</td><td>25</td></tr><tr><td>cube-quadruple-100M</td><td>(25, 5, 0.93, 0.8)</td><td>(5, 1, 0.93)</td><td>(25, 5, 0.93)</td><td>(5, 0.93)</td><td>(5, 0.7)</td><td>0.7</td><td>25</td><td>(25, 0.5, 10)</td><td>3</td><td>25</td></tr><tr><td>cube-octuple-1B</td><td>(25, 5, 0.93, 0.5)</td><td>(25, 5, 0.93)</td><td>(25, 5, 0.93)</td><td>(25, 0.93)</td><td>(25, 0.97)</td><td>0.7</td><td>25</td><td>(50, 0.5, 10)</td><td>10</td><td>50</td></tr><tr><td>humanoidmaze-giant</td><td>(25, 1, 0.5, 0.8)</td><td>(5, 1, 0.9)</td><td>(25, 5, 0.5)</td><td>(5, 0.5)</td><td>(25, 0.7)</td><td>0.5</td><td>50</td><td>(50, 0.5, 3)</td><td>0.3</td><td>50</td></tr><tr><td>puzzle-4x5</td><td>(25, 5, 0.9, 0.5)</td><td>(25, 5, 0.9)</td><td>(25, 5, 0.7)</td><td>(5, 0.9)</td><td>(25, 0.7)</td><td>0.7</td><td>50</td><td>(25, 0.7, 3)</td><td>1</td><td>25</td></tr><tr><td>puzzle-4x6-1B</td><td>(25, 1, 0.7, 0.5)</td><td>(25, 5, 0.7)</td><td>(25, 5, 0.5)</td><td>(5, 0.7)</td><td>(25, 0.5)</td><td>0.7</td><td>50</td><td>(25, 0.7, 3)</td><td>1</td><td>25</td></tr></table>

Table 7: Environment-specific hyperparameters for DQC, QC, NS, OS, SHARSA, HIQL, IQL, and HFBC. For SHARSA, HIQL, IQL, and HFBC, we follow the hyperparameters in the original paper (Park et al., 2025b).

<table><tr><td>Environment</td><td>DQC $h = 25, h_a = 5$  $(\kappa_b, \kappa_d)$ </td><td>DQC $h = 25, h_a = 1$  $(\kappa_b, \kappa_d)$ </td><td>DQC $h = 5, h_a = 1$  $(\kappa_b, \kappa_d)$ </td><td>QC-NS $n = 25, h_a = 5$  $\kappa_b$ </td><td>NS $n = 25$  $\kappa_b$ </td><td>NS $n = 5$  $\kappa_b$ </td><td>QC $h = 25$  $\kappa_b$ </td><td>QC $h = 5$  $\kappa_b$ </td><td>OS $\kappa_b$ </td></tr><tr><td>cube-triple-100M</td><td>(0.93, 0.8)</td><td>(0.93, 0.8)</td><td>(0.5, 0.8)</td><td>0.93</td><td>0.5</td><td>0.5</td><td>0.93</td><td>0.93</td><td>0.5</td></tr><tr><td>cube-quadruple-100M</td><td>(0.93, 0.8)</td><td>(0.93, 0.8)</td><td>(0.5, 0.8)</td><td>0.93</td><td>0.5</td><td>0.7</td><td>0.93</td><td>0.93</td><td>0.7</td></tr><tr><td>cube-octuple-1B</td><td>(0.93, 0.5)</td><td>(0.93, 0.5)</td><td>(0.93, 0.5)</td><td>0.93</td><td>0.97</td><td>0.5</td><td>0.93</td><td>0.93</td><td>0.7</td></tr><tr><td>humanoidmaze-giant</td><td>(0.5, 0.8)</td><td>(0.5, 0.8)</td><td>(0.5, 0.5)</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>puzzle-4x5</td><td>(0.9, 0.5)</td><td>(0.9, 0.5)</td><td>(0.5, 0.5)</td><td>0.7</td><td>0.7</td><td>0.5</td><td>0.9</td><td>0.9</td><td>0.7</td></tr><tr><td>puzzle-4x6-1B</td><td>(0.7, 0.5)</td><td>(0.7, 0.5)</td><td>(0.5, 0.5)</td><td>0.5</td><td>0.7</td><td>0.5</td><td>0.7</td><td>0.7</td><td>0.7</td></tr></table>

Table 8: Environment-specific hyperparameters under different $h , n , h _ { a }$ configurations for DQC, QC, NS, OS. For DQC-naïve, we use the same hyperparameter as the corresponding QC baseline.

# D ADDITIONAL RELATED WORK

Theoretical analysis for reinforcement learning under unobserved confounding variables. RL with action chunking policies can be seen as a special case of RL under unobserved confounding variables (Kallus & Zhou, 2021) as the action chunking policies ignore the intermediate states during the execution of an action chunk. Prior analyses are based off either causal-inference-inspired sensitivity models (Kallus & Zhou, 2020; Namkoong et al., 2020; Kausik et al., 2024), confounded MDP models (Bennett et al., 2021; Fu et al., 2022; Shi et al., 2024), or more general partially observable MDP (POMDP) models (Tennenholtz et al., 2020; Miao et al., 2022; Shi et al., 2022; Bennett & Kallus, 2024) where the confounding variables are modeled as part of the partially observable states. These models largely focus on characterizing either how much confounding variables affect the policy behavior (e.g., bounded odds-ratio between the policy with or without conditioning on the confounding variables (Kallus & Zhou, 2020)) or how much the observations reveal the confounding variables (e.g., the full-rank emission matrix assumption (Azizzadenesheli et al., 2016) and the weak revealing assumption (Liu et al., 2022) in POMDP). In contrast, our analysis specializes in action chunking policies where the unobserved variables are the intermediate states during an action chunk. This allows us to establish a more specialized (and thus distinct) openloop consistency condition under which we can identify the exact worst case bias (i.e., with matching lower and upper-bound to the exact value) for both behavioral value estimation and sub-optimality gap of the fixed-point for bellman optimality iteration, which are usually unknown under the more general models/assumptions in the literature.

Hierarchical reinforcement learning methods (Dayan & Hinton, 1992; Dietterich, 2000; Peng et al., 2017; Riedmiller et al., 2018; Shankar & Gupta, 2020; Pertsch et al., 2021; Gehring et al., 2021; Xie et al., 2021) solve tasks by typically leveraging a bi-level structure: a set of low-level/skill policies that directly interact with the environment and a high-level policy that selects among low-level policies. The low-level policies can also be learned via online RL (Kulkarni et al., 2016; Vezhnevets et al., 2016; 2017; Nachum et al., 2018) or offline pre-training on a prior dataset (Paraschos et al., 2013; Merel et al., 2019; Ajay et al., 2021; Pertsch et al., 2021; Touati et al., 2022; Nasiriany et al., 2022; Hu et al., 2023; Frans et al., 2024; Chen et al., 2024; Park et al., 2024). In the options framework, these low-level policies are often additionally associated with initiation and termination conditions that specify when and for how long these actions can be used (Sutton et al., 1999; Menache et al., 2002; Chentanez et al., 2004; ¸Sim¸sek & Barto, 2007; Konidaris, 2011; Daniel et al., 2016; Srinivas et al., 2016; Fox et al., 2017; Bacon et al., 2017; Bagaria & Konidaris, 2019; Bagaria et al., 2024; Koch et al., 2025). A long-lasting challenge in HRL is optimization stability because the highlevel policy needs to optimize for an objective that is shaped by the constantly changing low-level policies (Nachum et al., 2018). Prior work (Ajay et al., 2021; Pertsch et al., 2021; Wilcoxson et al., 2025) avoided this by first pre-training low-level policies and then keeping them frozen during the optimization of the high-level policy. Macro-actions (McGovern & Sutton, 1998; Durugkar et al., 2016), or action chunking (Zhao et al., 2023) is another form of temporally extended action, a special case of the low-level policies often considered in HRL, options literature, where a short horizon of actions is predicted all at once and executed in open loop. Such an approach collapses the bi-level structure, conveniently side-stepping optimization instability, and when combined with Q-learning, has shown great empirical successes in offline-to-online RL setting (Seo et al., 2024; Li et al., 2025b). Action chunking policies need to predict multiple actions open-loop, which can be difficult to learn and sacrifice reactivity. Our approach regains policy reactivity by predicting and executing only a partial action chunk, while still learning with the fully chunked critic for TD-backup. This design preserves the value propagation benefits of chunked critic without relying on fully open-loop action chunking policies, allowing our approach to work well on a wider range of tasks.

<table><tr><td>Environment</td><td>Backup Quantile $(\kappa_b)$ </td><td>Distillation Expectile $(\kappa_d)$ </td><td>Backup Horizon $(h)$  or  $(n)$ </td><td>Policy Chunk Size $(h_a)$ </td></tr><tr><td>cube-*</td><td>{0.5, 0.7, 0.9, 0.93, 0.95, 0.97, 0.99}</td><td>{0.5, 0.8}</td><td>{5, 25}</td><td>{1, 5, 25}</td></tr><tr><td>Others</td><td>{0.5, 0.7, 0.9}</td><td>{0.5, 0.8}</td><td>{5, 25}</td><td>{1, 5, 25}</td></tr></table>

Table 9: Hyperparameter tuning range for all methods. For NS, we only tune $\kappa _ { b }$ and n because the policy chunk size is always 1 and there is no distilled critic. Similarly, for $\mathbf { Q C } ,$ , we only tune $\kappa _ { b }$ and $h = h _ { a }$ because the policy chunk size is the same as the critic chunk size and there is no distilled critic. For OS, we only tune $\kappa _ { b }$ .

# E ADDITIONAL THEORETICAL RESULTS

# E.1 ε-DETERMINISTIC DYNAMICS IS WEAKLY OPEN-LOOP CONSISTENT

To provide some intuitions on what this open-loop consistency implies, we discuss a concrete family of MDPs where any data distribution from these MDPs is (weakly) εh-open-loop consistent (Proposition 4, with proof available in Section F.15).

Definition 5 (Near-deterministic Dynamics) A transition dynamics $T$ is ε-deterministic if there exists a deterministic transition dynamics represented by function $f : \mathcal { S } \times \mathcal { A }  \mathcal { S }$ and another arbitrary transition dynamics $\tilde { T } : \mathcal { S } \times \mathcal { A }  \Delta _ { \mathcal { S } }$ , and T is a combination of f and $\tilde { T } { : }$

$$
T (s ^ {\prime} \mid s, a) = (1 - \varepsilon) \delta_ {f (s, a)} (s ^ {\prime}) + \varepsilon \tilde {T} (s ^ {\prime} \mid s, a), \forall s, s ^ {\prime} \in \mathcal {S}, a \in \mathcal {A}. \tag {41}
$$

Proposition 4 (Deterministic Dynamics are Weakly Open-loop Consistent) If a transition dynamics M is ε-deterministic, then any data D collected from M is weakly εh-open-loop consistent with respect to M for any $h \in \mathbb { N } ^ { + }$ as long as $\varepsilon _ { h } \geq 3 ( 1 - ( 1 - \varepsilon ) ^ { h - 1 } )$ .

An ε-deterministic dynamics acts like a deterministic one most of the time (with 1 − ε probability) and a non-deterministic one occasionally (with ε probability). This bounded stochasticity allows the results of taking an action sequence (of length h) open-loop to be deterministically determined in the event that the deterministic dynamics is ‘triggered’ (with a joint $( 1 - \varepsilon ) ^ { h - 1 }$ probability across h time steps). It is clear that under such event, there is no gap between the ‘replayed’ open-loop data $P _ { \mathcal { D } } ^ { \circ }$ and the original data distribution $P _ { \mathcal { D } } .$ , and as result there is also no value estimation bias under this event, and thus intuitively we can bound the value estimation error by a function of the probability that the stochastic dynamics is ‘triggered’ (i.e., with $1 - ( 1 - \varepsilon ) ^ { h - 1 }$ probability).

# E.2 CONDITIONS WHEN n-STEP RETURN POLICIES ARE PROVABLY SUB-OPTIMAL

Definition 6 (Near Optimal Data) We say D is $\tilde { \delta } _ { n }$ -optimal for backup horizon length n $\in \mathbb { N } ^ { + }$ if

$$
Q ^ {\star} (s _ {t}, a _ {t}) - \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t})} \left[ R _ {t: t + n} + \gamma^ {n} V ^ {\star} (s _ {t + n}) \right] \leq \tilde {\delta} _ {n}, \forall s _ {t}, a _ {t} \in \operatorname{supp} (P _ {\mathcal {D}} (s _ {t}, a _ {t})). \tag {42}
$$

In Proposition 2, we have shown that the value of the learned action chunking policy is better than the nominal value of n-step return policy with a value gap of $3 \varepsilon _ { h } H$ . However, the actual value of the n-step return policy maybe better. Here, we analyze the worst-case performance of n-step return policies.

Proposition 5 (Worst-case analysis of n-step return backup) For any $n ~ \in ~ \mathbb { N } ^ { + } , ~ \tilde { \delta } _ { n } ~ \in$ $( 0 , \gamma - \gamma ^ { n } )$ and $\sigma \in \left( 0 , \tilde { \delta } _ { n } / ( 1 - \gamma ) \right)$ , there exists an MDP $\mathcal { M } ,$ and a $\tilde { \delta } _ { n }$ -optimal data distribution D with supp $( P _ { \mathcal { D } } ( s _ { t } , a _ { t } ) ) \supseteq \operatorname { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } , a _ { t } ) )$ such that for some $s \in \operatorname { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V _ {\mathrm{ac}} ^ {+} (s) - V _ {n} ^ {+} (s) = \frac {\tilde {\delta} _ {n}}{1 - \gamma} - \sigma , \tag {43}
$$

and for all $s \in \operatorname { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s) = V _ {\mathrm{ac}} ^ {+} (s). \tag {44}
$$

The proof (available in Section F.14) provides concrete examples where n-step return policies are worse than action chunking policies. The implication of this result is that the sub-optimality of the data distribution (as characterized by $\delta _ { n }$ and $\tilde { \delta } _ { n } )$ is generally independent from the open-loop consistency (as characterized by $\varepsilon _ { h } )$ .

# E.3 CLOSED-LOOP EXECUTION WITHOUT STOCHASTIC SHORTCUTS

In this section, we provide an alternative way of bounding the sub-optimality of $\pi ^ { \bullet }$ , the closed-loop execution of the learned action chunking policy $\pi _ { \mathrm { a c } } ^ { + }$ . In particular, we characterize two conditions when closed-loop execution of an action chunking policy can help mitigate open-loop biases.

Our first condition is based on the key observation that only a certain type of value overestimation is harmful for closed-loop execution of the action chunking policy. The source of this type of value overestimation comes from stochastic shortcuts:

Definition 7 (Stochastic Shortcuts) We say M is free of $\vartheta _ { h }$ -stochastic shortcuts for a horizon h if

$$
V ^ {\star} (s _ {t + h}) + R _ {t: t + h} - V ^ {\star} (s _ {t}) \leq \vartheta_ {h},
$$

$$
\forall s _ {t: t + h + 1}, a _ {t: t + h}: \prod_ {k = 0} ^ {h - 1} P (s _ {t + k + 1} \mid s _ {t + k}, a _ {t + k}) > 0, \tag {45}
$$

where $V ^ { \star }$ is the value function of optimal policy in M.

Intuitively, stochastic shortcuts are low-probability (but plausible) paths $( i . e . , s _ { t } , a _ { t } , \cdot \cdot \cdot , s _ { t + h } )$ in the MDP that lead to returns that are much higher than the optimal expected value $( i . e . , V ^ { \star } )$ . These stochastic shortcuts are particularly problematic for action chunking value backup because the chunked critic/Q-function cannot distinguish between a low-probability stochastic shortcut and an optimal (or near-optimal) closed-loop trajectory, leading it to erroneously favor the shortcut.

Our second condition is that our data distribution is a mixture of some data distribution that is collected by some optimal closed-loop policy $( \mathcal { D } ^ { \star } )$ and some data distribution that is collected by an open-loop policy $( \dot { \mathcal { D } } ^ { \circ }$ , and thus is open-loop consistent). Intuitively, this condition makes sure that any non-optimal trajectory can be accurately estimated by the action chunking value function $\hat { V } _ { \mathrm { a c } } ^ { + }$ and the bounded mixing ratio restricts the amount of bias that the $\hat { V } _ { \mathrm { a c } } ^ { + }$ has on the estimation of the optimal trajectories when the open-loop action chunks (e.g., in D◦) coincide with the action chunks in the optimal data (e.g., in D⋆). We formally define the second condition as follows:

Definition 8 (Open-loop Data Mix) We say D is α-open-loop mixed if for some $\beta \in [ 0 , 1 )$ , D can be decomposed into two data distributions $\mathcal { D } ^ { \star } , \mathcal { D } ^ { \bar { \circ } }$ as

$$
P _ {\mathcal {D}} (\cdot \mid s _ {t}) = \beta P _ {\mathcal {D} ^ {*}} (\cdot \mid s _ {t}) + (1 - \beta) P _ {\mathcal {D} ^ {\circ}} (\cdot \mid s _ {t}), \tag {46}
$$

where ${ \mathcal { D } } ^ { \star }$ is any data distribution collected by an optimal closed-loop policy $\pi ^ { \star }$ and $\mathcal { D } ^ { \circ }$ is any strongly open-loop consistent data distribution, and

$$
P _ {\mathcal {D} ^ {\circ}} \left[ a _ {t: t + h} \in \operatorname{supp} (P _ {\mathcal {D} ^ {\star}} (a _ {t: t + h} \mid s _ {t})) \mid s _ {t} \right] \leq \frac {\alpha \beta}{(1 - \alpha) (1 - \beta)}, \quad \forall s _ {t} \tag {47}
$$

With such data mixing assumption and in the absence of stochastic shortcuts, we can show that closedloop execution of the action chunking policy $( i . e .$ , only executing the first action of the action chunk) recovers a near-optimal closed-loop policy:

Theorem 7 (Closed-loop Execution in the Absence of Stochastic Shortcuts) D is α-open-loop mixed and M is free of $\vartheta _ { h }$ -stochastic shortcut, the value $( V ^ { \bullet } )$ of the one-step policy $( \pi ^ { \bullet } )$ as a result of the closed-loop execution of the action chunking policy $\pi _ { \mathrm { a c } } ^ { + }$ learned from $\mathcal { D }$ admits the following bound for all $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ :

$$
V ^ {\star} (s _ {t}) - V ^ {\bullet} (s _ {t}) \leq \frac {\alpha}{(1 - \gamma) ^ {2} (1 - \gamma^ {h} (1 - \alpha))} + \frac {\vartheta_ {h} \gamma^ {h}}{(1 - \gamma) (1 - \gamma^ {h})}. \tag {48}
$$

A proof is available in Section F.11. Intuitively, the second condition measures how much percentage of the open-loop data has overlapping support as the optimal data. With some algebraic manipulating, assuming the worst case of Equation (47), we can rewrite the data mixture as

$$
\mathcal {D} = \hat {\beta} [ (1 - \alpha) \mathcal {D} ^ {\star} + \alpha \mathcal {D} _ {\text { in }} ^ {\circ} ] + (1 - \hat {\beta}) \mathcal {D} _ {\text { out }} ^ {\circ}, \tag {49}
$$

where $\begin{array} { r } { \hat { \beta } = \frac { \beta } { 1 - \alpha } } \end{array}$ , supp $( P _ { \mathcal { D } _ { \mathrm { i n } } ^ { \circ } } ( \cdot \mid s _ { t } ) ) \subseteq$ supp $\left( P _ { \mathcal { D } ^ { \star } } ( \cdot \mid s _ { t } ) \right)$ and supp $( P _ { { D _ { \mathrm { o u t } } ^ { \circ } } } ( \cdot \mid s _ { t } ) )$ ∩ supp $( P _ { \mathbf { \delta D } ^ { \star } } ( \cdot \mid$ $s _ { t } ) ) = \emptyset$ . As the bound is independent of $\hat { \beta }$ , it becomes clear that $\mathcal { D } _ { \mathrm { o u t } } ^ { \circ }$ plays no contribution to the optimality of action chunking policy learning. The only harmful portion of the open-loop data distribution is $\mathcal { D } _ { \mathrm { i n } } ^ { \circ } .$ as the action chunking Q-function cannot differentiate these open-loop actions in $\mathcal { D } _ { \mathrm { i n } } ^ { \circ }$ from the closed-loop optimal actions in $\mathcal { D } ^ { \star }$ . This is reflected as the first term in our bound. The implication is that even if the data $\mathcal { D }$ is arbitrarily sub-optimal (with $\hat { \beta } \to 0 .$ , and hence arbitrarily bad for n-step return policies), $\pi ^ { \bullet }$ remains near-optimal as long as the ‘in-distribution’ open-loop data $\mathcal { D } _ { \mathrm { i n } } ^ { \circ }$ is relatively low in density compared to the optimal closed-loop data $\mathcal { D } ^ { \star } \left( i . e . , \alpha \right.$ is small).

Furthermore, our bound is independent of the open-loop consistency of the data D. As $\alpha , \vartheta  0$ , closed-loop execution of the action chunking policy exactly recovers the optimal policy. In contrast, even when $\alpha , \vartheta  0 .$ , open-loop execution of the original action chunking policy $( i . e . , \pi _ { \mathrm { a c } } ^ { + } )$ can suffer from the open-loop inconsistency of the data D: its value error can only be bounded by εh(1−γ)(1−γh) $\frac { \varepsilon _ { h } } { ( 1 - \gamma ) ( 1 - \gamma ^ { h } ) }$ Eh (as shown in Theorem 1), a function of $\varepsilon _ { h }$ (the strong open-loop consistency of D).

# F PROOFS OF MAIN RESULTS

# F.1 UTILITY LEMMATA

Lemma 1 (Mean value theorem for conditional probabilities) Let $P _ { 1 } , P _ { 2 } \in \Delta _ { { \mathcal { X } } \times { \mathcal { Y } } }$ and $P ( x , y ) : = \hat { \alpha } ( y ) P _ { 1 } ( x , y ) + ( 1 - \hat { \alpha } ( y ) ) P _ { 2 } ( x , y )$ and there exists $\alpha > 0$ such that $\hat { \alpha } ( y ) \perp$ $\alpha , \forall y \in \mathcal { V }$ . Then, there exists $y \in \mathcal { V }$ and $\tilde { \alpha } \leq \alpha$ such that

$$
P (\cdot \mid y) = \tilde {\alpha} P _ {1} (\cdot \mid y) + (1 - \tilde {\alpha}) P _ {2} (\cdot \mid y) \tag {50}
$$

Proof.

$$
\frac {P (x , y)}{P (y)} = \frac {\hat {\alpha} (y) P _ {1} (y) P _ {1} (x \mid y) + (1 - \hat {\alpha} (y)) P _ {2} (x \mid y)}{\hat {\alpha} (y) P _ {1} (y) + (1 - \hat {\alpha} (y)) P _ {2} (y)} \tag {51}
$$

$$
= \beta (y) P _ {1} (x \mid y) + (1 - \beta (y)) P _ {2} (x \mid y)
$$

where $\begin{array} { r } { \beta ( y ) : = \frac { \hat { \alpha } ( y ) P _ { 1 } ( y ) } { \hat { \alpha } ( y ) P _ { 1 } ( y ) + ( 1 - \hat { \alpha } ( y ) ) P _ { 2 } ( y ) } } \end{array}$ . We now prove $\exists y \in \mathcal { V } , \tilde { \alpha } \leq \alpha$ for Equation (50) to hold by

We first assume $\tilde { \alpha } = \beta ( y ) > \alpha , \forall y \in \mathcal { V }$ . Now, substitute $\beta ( y )$ in and integrate both side by y to obtain

$$
\hat {\alpha} (y) P _ {1} (y) > \alpha \hat {\alpha} (y) P _ {1} (y) + \alpha (1 - \hat {\alpha} (y)) P _ {2} (y) \tag {52}
$$

$$
\hat {\alpha} (y) > \alpha \hat {\alpha} (y) + \alpha - \alpha \hat {\alpha} (y) = \alpha , \tag {53}
$$

which is a contradiction to the condition $\hat { \alpha } ( y ) \le \alpha$ .

Therefore, there must exist $y \in \mathcal { V }$ with $\tilde { \alpha } \leq \alpha$ such that Equation (50) holds.

![](images/dd98a4fa4539251989aeb8d2e8d17ca5ef40a214b07033d0a2a134889e7a7a2b.jpg)

Lemma 2 (Expectation difference for bounded function and TV) For two distributions $P , Q \in$ $\Delta { _ { X } }$ and two bounded functions $f , g : \mathcal { X }  [ 0 , 1 ] .$ , if the TV distance between P and Q is no larger than ε and $\| f - g \| _ { \infty } \leq \delta$ under supp $( P ) \bar { \cap } \operatorname { s u p p } ( Q )$ , then

$$
\left| \mathbb {E} _ {x \sim P} [ f (x) ] - \mathbb {E} _ {x \sim Q} [ g (x) ] \right| \leq (1 - \varepsilon) \delta + \varepsilon . \tag {54}
$$

Proof. Let’s decompose the probability mass of $P$ and Q in terms of $d _ { P } , d _ { P Q } , d _ { Q } : \mathcal { X } $ R as the following:

$$
P (x) = d _ {P} (x) + d _ {P Q} (x), \tag {55}
$$

$$
Q (x) = d _ {P Q} (x) + d _ {Q} (x). \tag {56}
$$

The $\textstyle \int d _ { P } ( x ) \mathrm { d } { : }$ x maximizing solution is

$$
d _ {P} (x) = \max (P (x), Q (x)) - Q (x) \tag {57}
$$

$$
d _ {Q} (x) = \max (P (x), Q (x)) - P (x) \tag {58}
$$

$$
d _ {P Q} (x) = P (x) + Q (x) - \max (P (x), Q (x)). \tag {59}
$$

It is clear that under this decomposition,

$$
\int d _ {P} (x) \mathrm{d} x = \int d _ {Q} (x) \mathrm{d} x = \hat {\varepsilon} \leq \varepsilon , \tag {60}
$$

$$
\int d _ {P Q} (x) \mathrm{d} x = 1 - \hat {\varepsilon} \geq 1 - \varepsilon . \tag {61}
$$

Now we are ready to bound the expectation difference:

$$
\begin{array}{l} \left| \mathbb {E} _ {x \sim P} [ f (x) ] - \mathbb {E} _ {x \sim Q} [ g (x) ] \right| \\ = \left| \left(\int d _ {P} (x) f (x) \mathrm{d} x - \int d _ {Q} (x) g (x) \mathrm{d} x\right) + \left(\int d _ {P Q} (x) (f (x) - g (x)) \mathrm{d} x\right) \right| \\ \leq \left| \int d _ {P} (x) f (x) \mathrm{d} x - \int d _ {Q} (x) g (x) \mathrm{d} x \right| + \left| \int d _ {P Q} (x) (f (x) - g (x)) \mathrm{d} x \right| \\ \leq \max \left(\sup _ {x} f (x) \int d _ {P} (x) \mathrm{d} x - \inf _ {x} g (x) \int d _ {Q} (x) \mathrm{d} x, \sup _ {x} g (x) \int d _ {Q} (x) \mathrm{d} x - \inf _ {x} f (x) \int d _ {P} (x) \mathrm{d} x\right) \\ + \left| \left(\sup _ {x: d _ {P Q} (x) > 0} | f (x) - g (x) |\right) \int d _ {P Q} (x) \mathrm{d} x \right| \\ \leq \hat {\varepsilon} + \left(\sup _ {x \in \operatorname{supp} (P) \cap \operatorname{supp} (Q)} | f (x) - g (x) |\right) (1 - \hat {\varepsilon}) \\ = \hat {\varepsilon} + \| f - g \| _ {\infty} (1 - \hat {\varepsilon}) \\ \leq \hat {\varepsilon} (1 - \delta) + \delta \\ \leq \varepsilon (1 - \delta) + \delta \\ = (1 - \varepsilon) \delta + \varepsilon \\ \end{array}
$$

Lemma 3 (Total variation under event conditioning) For two random variables $X \in \Delta _ { \mathcal { X } }$ and $Y \in \Delta _ { \mathcal { Y } }$ and any $y \in \mathcal { V }$ ,

$$
D _ {\mathrm{TV}} (P (X \mid Y = y) \| P (X)) \leq 1 - P (Y = y) \tag {63}
$$

Proof. Let $p = P ( Y = y )$

$$
\begin{array}{l} D _ {\mathrm{TV}} (P (X \mid Y = y) \| P (X)) \\ = \frac {1}{2} \int | P (x) - P (x \mid y) | \mathrm{d} x \\ = \frac {1}{2} \int | P (x \mid Y = y) (P (Y = y) - 1) + P (x \mid Y \neq y) P (Y \neq y) | \mathrm{d} x \tag {64} \\ = \frac {1 - p}{2} \int | (P (x \mid Y \neq y) - P (x \mid Y = y)) | \mathrm{d} x \\ = (1 - p) D _ {\mathrm{TV}} \left(P (X \mid Y = y) \| P (X \mid Y \neq y)\right) \\ \leq 1 - p \\ \end{array}
$$

□

Lemma 4 (Data Processing Inequality for f-divergence (Csiszár, 1967)) For two random variables A, $B \in \Delta _ { \mathcal { X } }$ and a deterministic function ${ \bar { f } } : { \mathcal { X } }  { \mathcal { V } }$ , and $C : = g ( A ) , D : = g ( B )$

$$
D _ {f} (P _ {A} \parallel P _ {B}) \geq D _ {f} (P _ {C} \parallel P _ {D}). \tag {65}
$$

Since TV-distance is a f-divergence with $f = | x - 1 |$ , we have

$$
D _ {\mathrm{TV}} (P _ {A} \parallel P _ {B}) \geq D _ {\mathrm{TV}} (P _ {C} \parallel P _ {D}). \tag {66}
$$

Proof from Wu (2017).

$$
\begin{array}{l} D _ {f} \left(P _ {A} \| P _ {B}\right) = \mathbb {E} _ {x \sim P _ {B}} \left[ f \left(P _ {A} (x) / P _ {B} (x)\right) \right] \\ = \mathbb {E} _ {P _ {B D}} \left[ f (P _ {A C} / P _ {B D}) \right] \\ = \mathbb {E} _ {(x, y) \sim P _ {D}} \left[ \mathbb {E} _ {P _ {B | D}} \left[ f (P _ {A C} (x, y) / P _ {B D} (x, y)) \right] \right] \\ \geq \mathbb {E} _ {y \sim P _ {D}} \left[ f \left(\mathbb {E} _ {x \sim P _ {B | D = y}} \left[ P _ {A C} (x, y) / P _ {B D} (x, y) \right]\right) \right] \tag {67} \\ = \mathbb {E} _ {y \sim P _ {D}} \left[ f \left(\mathbb {E} _ {x \sim P _ {B | D = y}} \left[ P _ {C} (y) / P _ {D} (y) \right]\right) \right] \\ = \mathbb {E} _ {y \sim P _ {D}} \left[ f \left(P _ {C} (y) / P _ {D} (y)\right) \right] \\ = D _ {f} (P _ {C} \parallel P _ {D}). \\ \end{array}
$$

![](images/4fb7f35bc8d23cf02b4f871235142b3b07f51540d5dcd9d90f205aedebd9da31.jpg)

# F.2 PROOF OF THEOREM 1

Theorem 1 (AC Value Bias) Let $\hat { V } _ { \mathrm { a c } } : S  [ 0 , 1 / ( 1 - \gamma ) ]$ be a solution of

$$
\hat {V} _ {\mathrm{ac}} (s _ {t}) = \mathbb {E} _ {s _ {t + 1: t + h + 1}, a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t})} \left[ R _ {t: t + h} + \gamma^ {h} \hat {V} _ {\mathrm{ac}} (s _ {t + h}) \right], \tag {11}
$$

with If D $\begin{array} { r } { R _ { t : t + h } = \sum _ { t ^ { \prime } = t } ^ { t + h } \gamma ^ { t ^ { \prime } - t } r ( s _ { t ^ { \prime } } , a _ { t ^ { \prime } } ) } \end{array}$ and nt, th $V _ { \mathrm { a c } }$ is thfor all $\tilde { \pi } _ { \mathrm { a c } } : s _ { t } \mapsto P _ { \mathcal { D } } \big ( a _ { t : t + h } ~ | ~ s _ { t } \big )$ $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } } ( s _ { t } ) )$

$$
\left| V _ {\mathrm{ac}} (s _ {t}) - \hat {V} _ {\mathrm{ac}} (s _ {t}) \right| \leq \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})} \leq \varepsilon_ {h} H \bar {H}. \tag {12}
$$

Proof. Since D is $\varepsilon _ { h ^ { \prime } }$ -open-loop consistent in state-action for $h ^ { \prime } < h$ , the state-action distribution leading up to step h admits the following bound:

$$
D _ {\mathrm{TV}} (P _ {\mathcal {D}} (s _ {t + h}, a _ {t + h} \mid s _ {t}) \parallel P _ {\mathcal {D}} ^ {\circ} (s _ {t + h}, a _ {t + h} \mid s _ {t})) \leq \varepsilon_ {h} \tag {68}
$$

Let $\begin{array} { r } { R _ { t : t + h } = \sum _ { k = 0 } ^ { h - 1 } \gamma ^ { k } r ( s _ { t + k } , a _ { t + k } ) } \end{array}$ be the h-step reward distribution. Then the difference in h-step reward is bounded by

$$
\begin{array}{l} \left| \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t})} [ R _ {t: t + h} ] - \mathbb {E} _ {P _ {\mathcal {D}} ^ {\circ} (\cdot | s _ {t})} [ R _ {t: t + h} ] \right| \\ \leq \sum_ {h ^ {\prime} = 1} ^ {h - 1} \left[ \gamma^ {h ^ {\prime}} \mathbb {E} _ {P _ {\mathcal {D}} (s _ {t + h ^ {\prime}}, a _ {t + h ^ {\prime}} | s _ {t})} [ r (s _ {t + h ^ {\prime}}, a _ {t + h ^ {\prime}}) ] - \mathbb {E} _ {P _ {\mathcal {D}} ^ {\circ} (s _ {t + h ^ {\prime}}, a _ {t + h ^ {\prime}} | s _ {t})} [ r (s _ {t + h ^ {\prime}}, a _ {t + h ^ {\prime}}) ] \right] \tag {69} \\ \leq \sum_ {h ^ {\prime} = 1} ^ {h - 1} \gamma^ {h ^ {\prime}} \varepsilon_ {h}. \\ \end{array}
$$

where the first inequality uses Lemma 2 and the fact that TV distance is bounded (Equation (68)).

Since D is $\varepsilon _ { h }$ -open-loop consistent for h in state, we have

$$
D _ {\mathrm{TV}} (P _ {\mathcal {D}} (s _ {t + h} \mid s _ {t}) \parallel P _ {\mathcal {D}} ^ {\circ} (s _ {t + h} \mid s _ {t})) \leq \varepsilon_ {h}, \tag {70}
$$

which can then be used to bound the estimation error using Lemma 2:

$$
\begin{array}{l} \left| \mathbb {E} _ {s _ {t + h} \sim P _ {\mathcal {D}} (s _ {t + h} | s _ {t})} \left[ \hat {V} _ {\mathrm{ac}} (s _ {t + h}) \right] - \mathbb {E} _ {s _ {t + h} \sim P _ {\mathcal {D}} ^ {\circ} (s _ {t + h} | s _ {t})} \left[ V _ {\mathrm{ac}} (s _ {t + h}) \right] \right| \\ \leq \frac {\varepsilon_ {h}}{1 - \gamma} + (1 - \varepsilon_ {h}) \sup _ {s _ {t + h} \in \operatorname{supp} \left(P _ {\mathcal {D}} \left(s _ {t + h} \mid s _ {t}\right)\right)} \left[ | \hat {V} _ {\mathrm{ac}} \left(s _ {t + h}\right) - V _ {\mathrm{ac}} \left(s _ {t + h}\right) | \right] \tag {71} \\ \end{array}
$$

For all $s _ { t } \in \mathrm { s u p p } ( P _ { D } ( s _ { t } ) )$ ,

$$
\begin{array}{l} \left| \hat {V} _ {\mathrm{ac}} (s _ {t}) - V _ {\mathrm{ac}} (s _ {t}) \right| \\ \leq \left| \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t})} [ R _ {t: t + h} ] - \mathbb {E} _ {P _ {\mathcal {D}} ^ {\circ} (\cdot | s _ {t})} [ R _ {t: t + h} ] \right| \\ + \gamma^ {h} \left| \mathbb {E} _ {s _ {t + h} \sim P _ {\mathcal {D}} (s _ {t + h} | s _ {t})} \left[ \hat {V} _ {\mathrm{ac}} (s _ {t + h}) \right] - \mathbb {E} _ {s _ {t + h} \sim P _ {\mathcal {D}} ^ {\circ} (s _ {t + h} | s _ {t})} \left[ V _ {\mathrm{ac}} (s _ {t + h}) \right] \right| \tag {72} \\ \leq \sum_ {h ^ {\prime} = 0} ^ {h - 1} \left[ \gamma^ {h ^ {\prime}} \varepsilon_ {h} \right] + \frac {\gamma^ {h} \varepsilon_ {h}}{1 - \gamma} + \gamma^ {h} (1 - \varepsilon_ {h}) \sup _ {s _ {t + h} \in \operatorname{supp} (P _ {\mathcal {D}} (s _ {t + h} | s _ {t}))} \left[ | \hat {V} _ {\mathrm{ac}} (s _ {t + h}) - V _ {\mathrm{ac}} (s _ {t + h}) | \right]. \\ \end{array}
$$

Since the support of $s _ { t + h } \mid s _ { t }$ is a subset of the support for $s _ { t }$ by Assumption 1, we can recursively apply the inequality to obtain,

$$
\begin{array}{l} \left| \hat {V} _ {\mathrm{ac}} (s _ {t}) - V _ {\mathrm{ac}} (s _ {t}) \right| \leq \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \left(\sum_ {h ^ {\prime} = 1} ^ {h - 1} \left[ \gamma^ {h ^ {\prime}} \varepsilon_ {h} \right] + \frac {\gamma^ {h} \varepsilon_ {h}}{1 - \gamma}\right) \tag {73} \\ = \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}, \\ \end{array}
$$

as desired.

![](images/a28c2ac5fa2a5b7b5ec7acf5fe26c0228de3fd351530409ee80239bd391c1417.jpg)

![](images/854e715b5c4669e327df36f43870e6d50d1dfcf40d15e03ab95490a1cc84553d.jpg)  
Figure 8: A 2h-state MDP that is constructed to meet the upper-bound in Theorem 1. The data distribution D that achieves such an upper bound is collected by the optimal policy: $\pi ( X _ { i } ) =$ $1 , \pi ( \tilde { X } _ { i } ) = 0$ .

# F.3 PROOF OF THEOREM 2

Theorem 2 (Worst-case AC Value Bias) For any $h > 1 , \gamma \in [ 0 , 1 ) , \varepsilon _ { h } \in [ 0 , 1 / 2 ]$ , there exists an MDP M and a weakly $\varepsilon _ { h } .$ -open-loop consistent D such that for some $s _ { t } \in \mathrm { s u p p } ( P _ { D } ( s _ { t } ) )$ ,

$$
V _ {\mathrm{ac}} (s _ {t}) - \hat {V} _ {\mathrm{ac}} (s _ {t}) = \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}. \tag {13}
$$

Similarly, there exists M and $\varepsilon _ { h }$ -open-loop consistent D such that for some $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } } ( s _ { t } ) )$ ,

$$
\hat {V} _ {\mathrm{ac}} (s _ {t}) - V _ {\mathrm{ac}} (s _ {t}) = \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}. \tag {14}
$$

Proof. Let $\delta \in [ 0 , 1 ]$ be any value that satisfies $\varepsilon _ { h } = 2 \delta ( 1 - \delta ) .$ . δ must exist because $\varepsilon _ { h } \in [ 0 , 1 / 2 ]$ Let us define a MDP that has $S = 2 h$ states, $\mathcal { S } = \{ X _ { 0 } , X _ { 1 } , \tilde { X } _ { 1 } , \cdot \cdot \cdot , X _ { h - 1 } , \tilde { X } _ { h - 1 } , Z \}$ , and $A = 2$ actions, $\mathcal { A } = \{ 0 , 1 \}$ , and the following transition function $T$ and reward function r (see a diagram in Figure 8):

$$
T (\tilde {X} _ {i + 1} \mid X _ {i}, a) = T (\tilde {X} _ {i + 1} \mid \tilde {X} _ {i}, a) = \delta , \quad \forall a \in \{0, 1 \}, i \in \{1, \dots , h - 2 \}
$$

$$
T (X _ {i + 1} \mid X _ {i}, a) = T (X _ {i + 1} \mid \tilde {X} _ {i}, a) = 1 - \delta , \quad \forall a \in \{0, 1 \}, i \in \{0, \dots , h - 2 \}
$$

$$
T (Z \mid \tilde {X} _ {h - 1}, a = 1) = T (Z \mid X _ {h - 1}, a = 0) = 1
$$

$$
T \left(X _ {0} \mid \tilde {X} _ {h - 1}, a = 0\right) = T \left(X _ {0} \mid X _ {h - 1}, a = 1\right) = 1 \tag {74}
$$

$$
r (\tilde {X} _ {i}, a = 0) = r (X _ {i}, a = 1) = 1, \quad \forall i \in \{0, \dots , h - 1 \}
$$

$$
r (\tilde {X} _ {i}, a = 1) = r (X _ {i}, a = 0) = 0, \quad \forall i \in \{0, \dots , h - 1 \}
$$

$$
r (Z, a = 1) = r (Z, a = 0) = 0
$$

$$
T (Z \mid Z, a = 0) = T (Z \mid Z, a = 1) = 1
$$

Now, we assume that the data D is collected by the optimal closed-loop policy where

$$
\pi (X _ {i}) = 1, \pi (\tilde {X} _ {i}) = 0. \tag {75}
$$

First, we check D is $\varepsilon _ { h }$ -open-loop consistent.

We can show that by computing the distribution for $P _ { \mathcal { D } } ( s _ { t + i } , a _ { t + i } \mid s _ { t } = X _ { 0 } )$ and $P _ { \mathcal { D } } ^ { \circ } ( s _ { t + i } , a _ { t + i } \mid$ $s _ { t } = X _ { 0 } )$ as follows:

$$
\left[ \begin{array}{l l} P _ {\mathcal {D}} \left(s _ {t + i} = \tilde {X} _ {i}, a _ {t + i} = 0 \mid X _ {0}\right) & P _ {\mathcal {D}} \left(s _ {t + i} = \bar {X} _ {i}, a _ {t + i} = 1 \mid X _ {0}\right) \\ P _ {\mathcal {D}} \left(s _ {t + i} = X _ {i}, a _ {t + i} = 0 \mid X _ {0}\right) & P _ {\mathcal {D}} \left(s _ {t + i} = X _ {i}, a _ {t + i} = 1 \mid X _ {0}\right) \end{array} \right] = \left[ \begin{array}{c c} \delta & 0 \\ 0 & 1 - \delta \end{array} \right] \tag {76}
$$

$$
\left[ \begin{array}{c c} P _ {\mathcal {D}} ^ {\circ} (s _ {t + i} = \tilde {X} _ {i}, a _ {t + i} = 0 \mid X _ {0}) & P _ {\mathcal {D}} ^ {\circ} (s _ {t + i} = \tilde {X} _ {i}, a _ {t + i} = 1 \mid X _ {0}) \\ P _ {\mathcal {D}} ^ {\circ} (s _ {t + i} = X _ {i}, a _ {t + i} = 0 \mid X _ {0}) & P _ {\mathcal {D}} ^ {\circ} (s _ {t + i} = X _ {i}, a _ {t + i} = 1 \mid X _ {0}) \end{array} \right] = \left[ \begin{array}{c c} \delta^ {2} & (1 - \delta) \delta \\ \delta (1 - \delta) & (1 - \delta) ^ {2} \end{array} \right]
$$

From the calculation above, it is clear that

$$
D _ {\mathrm{TV}} (P _ {\mathcal {D}} ^ {\circ} (s _ {t + i}, a _ {t + i} \mid s _ {t}) \parallel P _ {\mathcal {D}} (s _ {t + i}, a _ {t + i} \mid s _ {t})) = \varepsilon_ {h}, \quad \forall i \in \{1, 2, \dots , h - 1 \}. \tag {77}
$$

From the computed values of $P _ { \mathcal { D } } ^ { \circ } ( s _ { t + h - 1 } , a _ { t + h - 1 } \mid s _ { t } )$ and $P _ { \mathcal { D } } ( s _ { t + h - 1 } , a _ { t + h - 1 } \mid s _ { t } )$ , we can derive

$$
\begin{array}{l} P _ {\mathcal {D}} \left(s _ {t + h} = Z \mid s _ {t} = X _ {0}\right) = 0, \\ P ^ {\circ} (s _ {t + h} = Z \mid s _ {t} = X _ {0}) = 2 (1 - f) f _ {t + h} \end{array} \tag {78}
$$

$$
P _ {\mathcal {D}} ^ {\circ} (s _ {t + h} = Z \mid s _ {t} = X _ {0}) = 2 (1 - \delta) \delta = \varepsilon_ {h}.
$$

From the calculation above, it is clear that

$$
D _ {\mathrm{TV}} (P _ {\mathcal {D}} ^ {\circ} (s _ {t + h} \mid s _ {t}) \parallel P _ {\mathcal {D}} (s _ {t + h} \mid s _ {t})) = \varepsilon_ {h}. \tag {79}
$$

Up to now, we have checked that D is $\varepsilon _ { h }$ -open-loop consistent. Now, we are left with analyzing $\hat { V } _ { \mathrm { a c } }$ and $V _ { \mathrm { a c } }$ . With some calculations, we can obtain the following:

$$
\begin{array}{c} \mathbb {E} _ {P _ {\mathcal {D}} ^ {\circ}} \left[ R _ {t: t + h} \right] = 1 + \frac {(1 - \varepsilon_ {h}) (\gamma - \gamma^ {h})}{1 - \gamma}, \\ \hat {V} _ {\mathrm{ac}} (X _ {0}) = \frac {1}{1 - \gamma}, \end{array} \tag {80}
$$

$$
V _ {\mathrm{ac}} (Z) = 0.
$$

Now, we are ready to compute $V _ { \mathrm { a c } } ( X _ { 0 } )$ :

$$
\begin{array}{l} V _ {\mathrm{ac}} \left(X _ {0}\right) = \frac {\left(1 - \gamma^ {h}\right) - \varepsilon_ {h} \left(\gamma - \gamma^ {h}\right)}{(1 - \gamma)} + \gamma^ {h} \left[ \left(1 - \varepsilon_ {h}\right) V _ {\mathrm{ac}} \left(X _ {0}\right) + \varepsilon_ {h} V _ {\mathrm{ac}} (Z) \right] \tag {81} \\ = \frac {1 - \gamma^ {h} - \varepsilon_ {h} (\gamma - \gamma^ {h})}{(1 - \gamma) (1 - \gamma^ {h} (1 - \varepsilon_ {h}))} \\ \end{array}
$$

Finally, with $X _ { 0 } \in \mathrm { s u p p } ( \mathcal { D } )$ , we obtain the desired value difference

$$
\hat {V} _ {\mathrm{ac}} (X _ {0}) - V _ {\mathrm{ac}} (X _ {0}) = \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) (1 - \gamma^ {h} (1 - \varepsilon_ {h}))}. \tag {82}
$$

By symmetry, we can flip the reward value $( i . e . , 0 \to 1$ and $1  0 )$ to construct the example such that

$$
V _ {\mathrm{ac}} (X _ {0}) - \hat {V} _ {\mathrm{ac}} (X _ {0}) = \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) (1 - \gamma^ {h} (1 - \varepsilon_ {h}))}. \tag {83}
$$

![](images/df07be2ab47aec4dd34e14a014a9fdd850aa5ea8eb57353e9b8d37c4ada9d2d9.jpg)

# F.4 PROOF OF COROLLARY 1

Corollary 1 (Optimality Gap for AC Policy) Let ${ \mathcal { D } } ^ { \star }$ be the data collected by any optimal policy $\pi ^ { \star }$ . If $\mathcal { D } ^ { \star }$ is weakly $\varepsilon _ { h } .$ -open-loop consistent, then for all $s _ { t } \in \mathrm { s u p p } ( P _ { { D ^ { \star } } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {\star} (s _ {t}) \leq V ^ {\star} (s _ {t}) - \tilde {V} _ {\mathrm{ac}} (s _ {t}) \leq \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})} \leq \varepsilon_ {h} H \bar {H}, \tag {15}
$$

where $V ^ { \star }$ is the value of the optimal policy $\pi ^ { \star } , V _ { \mathrm { a c } } ^ { \star }$ is the true value of the optimal action chunking policy, and $\tilde { V } _ { \mathrm { a c } }$ is the true value of the action chunking policy from cloning the data $\mathcal { D } ^ { \star }$ :

$$
\tilde {\pi} _ {a c} ^ {\mathcal {D} ^ {*}} (a _ {t: t + h} \mid s _ {t}): s _ {t} \mapsto P _ {\mathcal {D} ^ {*}} (\cdot \mid s _ {t}). \tag {16}
$$

Proof. Let $\hat { V } _ { \mathrm { a c } }$ be the fixed point of the following equation:

$$
\hat {V} _ {\mathrm{ac}} \left(s _ {t}\right) = \mathbb {E} _ {s _ {t + 1: t + h + 1}, a _ {t: t + h} \sim P _ {\mathcal {D} ^ {*}} (\cdot | s _ {t})} \left[ R _ {t: t + h} + \gamma^ {h} \hat {V} _ {\mathrm{ac}} \left(s _ {t + h}\right) \right] \tag {84}
$$

where again the followin $\begin{array} { r } { R _ { t : t + h } = \sum _ { t ^ { \prime } = t } ^ { t + h } \gamma ^ { t ^ { \prime } - t } r ( s _ { t ^ { \prime } } , a _ { t ^ { \prime } } ) } \end{array}$ . The value of the optimal policy is the fixed point of

$$
\begin{array}{l} V ^ {\star} (s _ {t}) = \mathbb {E} _ {s _ {t + 1}, a _ {t} \sim P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t})} [ r (s _ {t}, a _ {t}) + \gamma V ^ {\star} (s _ {t + 1}) ] \\ = \mathbb {E} _ {s _ {t: t + 2}, a _ {t: t + 1} \sim P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t})} [ r (s _ {t}, a _ {t}) + \gamma r (s _ {t + 1}, a _ {t + 1}) + \gamma V ^ {\star} (s _ {t + 2}) ] \tag {85} \\ \end{array}
$$

$$
\cdot \cdot \cdot
$$

$$
= \mathbb {E} _ {s _ {t + 1: t + h + 1}, a _ {t: t + h} \sim P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t})} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} (s _ {t + h}) \right]
$$

which is equivalent to fixed-point equation for $\hat { V } _ { \mathrm { a c } }$ . Therefore $\hat { V } _ { \mathrm { a c } } = V ^ { \star }$ . By Theorem 1, we know that the true value $V _ { \mathrm { a c } }$ of the action chunking policy $\tilde { \pi } _ { \mathrm { a c } }$ that clones ${ \mathcal { D } } ^ { \star }$ is close to $\hat { V } _ { \mathrm { a c } }$ . More specifically, for all $s _ { t } \in \operatorname { s u p p } ( \mathcal { D } ^ { \star } )$ ,

$$
\left| \hat {V} _ {\mathrm{ac}} (s _ {t}) - \tilde {V} _ {\mathrm{ac}} (s _ {t}) \right| \leq \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}, \tag {86}
$$

which means that

$$
V ^ {\star} (s _ {t}) - \tilde {V} _ {\mathrm{ac}} (s _ {t}) \leq \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}, \tag {87}
$$

where we can remove the absolute value operator because $V ^ { \star } ( s _ { t } )$ is by definition always at least as large as $\tilde { V } _ { \mathrm { a c } } ( s _ { t } )$ . Since the optimal action chunking policy, by definition, attains equally good or better values (over S) represented by $V _ { \mathrm { a c } }$ , and the optimal policy $\pi ^ { \star }$ also attains equally good or better value $( i . e . , V ^ { \star } )$ compared to that of the optimal action chunking policy $\pi _ { \mathrm { a c } } ^ { \star } ( i . { \bar { e } } _ { \cdot } , \dot { V _ { \mathrm { a c } } ^ { \star } } )$ , the following inequality holds for all $s _ { t } \in \mathrm { s u p p } ( \mathcal { D } ^ { \star } )$ :

$$
V ^ {\star} (s _ {t}) \geq V _ {\mathrm{ac}} ^ {\star} (s _ {t}) \geq \tilde {V} _ {\mathrm{ac}} (s _ {t}). \tag {88}
$$

Therefore,

$$
V _ {\mathrm{ac}} ^ {\star} (s _ {t}) - V ^ {\star} (s _ {t}) \leq \tilde {V} _ {\mathrm{ac}} (s _ {t}) - V ^ {\star} (s _ {t}) \leq \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}, \tag {89}
$$

as desired.

![](images/91be77af6f23eb6612335e7fb40cf9dc6c545d3cd87f8508d563aab87a5977ab.jpg)

# F.5 PROOF OF COROLLARY 2

Corollary 2 (Worse-case Optimality Gap for Action Chunking Policy) For any $h > 1 , \gamma \in$ $[ 0 , 1 ) , \varepsilon _ { h } \stackrel { \cdot } { \in } [ 0 , 1 / 2 ]$ , there exists an MDP M whose optimal policy $\pi ^ { \star }$ induces a data distribution $\bar { \mathcal { D } } ^ { \star }$ that is weakly εh-open-loop consistent, such that for some $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {\star} (s _ {t}) = \frac {\gamma \varepsilon_ {h}}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}. \tag {17}
$$

Proof. To show this, we need a slightly more complicated MDP (compared to the 2h-state MDP we use in the proof Section F.3). The MDP we construct for this proof is a (3h − 1)-state MDP as illustrated in Figure 9.

![](images/a95d405506c6f372c32829eca01741ce6304f8dc38bfbdcbfd0d78e19009a047.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X0 -->|a=1| r1["r=1"]
    X0 -->|a=0| r2["r=1"]
    r1 -->|δ/2| A1["A1"]
    r1 -->|δ/2| X1["X1"]
    r1 -->|δ/2| B1["B1"]
    r2 -->|δ/2| A1
    r2 -->|δ/2| X2["X2"]
    r2 -->|δ/2| B2["B2"]
    r1 -->|δ/2| B1
    r1 -->|δ/2| B2
    A1 -->|a=0| r0["r=0"]
    A1 -->|a=1| r1
    A2 -->|δ/2| A1
    A2 -->|δ/2| X0["X0"]
    A2 -->|δ/2| B2
    A2 -->|δ/2| Z["Z"]
    X0 -->|a=0| r0
    X0 -->|a=0| r1
    X1 -->|a=0| r1
    X1 -->|a=1| r2
    X2 -->|δ/2| A2
    X2 -->|δ/2| B2
    X2 -->|δ/2| Z
    Xh-1 -->|a=0| r0
    Xh-1 -->|a=0| r1
    Xh-1 -->|a=1| r2
    Xh-1 -->|a=1| r3
    Xh-1 -->|a=0| r4
    Xh-1 -->|a=0| r5
    Xh-1 -->|a=0| r6
    Xh-1 -->|a=0| r7
    Xh-1 -->|a=0| r8
    Xh-1 -->|a=0| r9
    Xh-1 -->|a=0| r10
    Z -->|r0| R0["R0"]
    Z -->|r0| R1["R1"]
    Z -->|r0| R2["R2"]
    Z -->|r0| R3["R3"]
    Z -->|r0| R4["R4"]
    Z -->|r0| R5["R5"]
    Z -->|r0| R6["R6"]
    Z -->|r0| R7["R7"]
    Z -->|r0| R8["R8"]
    Z -->|r0| R9["R9"]
```
</details>

Figure 9: A (3h − 1)-state MDP that is constructed to meet the upper-bound in Corollary 1.

The optimal policy we pick is described as follows:

$$
\pi^ {\star} (a = 0 \mid X _ {i}) = 1 / 2
$$

$$
\pi^ {\star} (a = 1 \mid X _ {i}) = 1 / 2 \tag {90}
$$

$$
\pi^ {\star} (a = 1 \mid A _ {i}) = 1
$$

$$
\pi^ {\star} (a = 0 \mid B _ {i}) = 1 / 2
$$

This induces the following state distribution,

$$
\begin{array}{l} P _ {\mathcal {D} ^ {\star}} \left(s _ {t + i} = A _ {i} \mid s _ {t} = X _ {0}\right) = P _ {\mathcal {D} ^ {\star}} \left(s _ {t + i} = B _ {i} \mid s _ {t} = X _ {0}\right) \\ = P _ {\mathcal {D} ^ {\star}} ^ {\circ} (s _ {t + i} = A _ {i} \mid s _ {t} = X _ {0}) = P _ {\mathcal {D} ^ {\star}} ^ {\circ} (s _ {t + i} = B _ {i} \mid s _ {t} = X _ {0}) = \delta / 2, \tag {91} \\ \end{array}
$$

$$
P _ {\mathcal {D} ^ {\star}} (s _ {t + i} = X _ {i} \mid s _ {t} = X _ {0}) = P _ {\mathcal {D} ^ {\star}} ^ {\circ} (s _ {t + i} = X _ {i} \mid s _ {t} = X _ {0}) = 1 - \delta ,
$$

and a fully factorized distribution for the action chunk,

$$
P _ {\mathcal {D} ^ {*}} ^ {\circ} (a _ {t + i} = 0 \mid s _ {t}) = P _ {\mathcal {D} ^ {*}} ^ {\circ} (a _ {t + i} = 0 \mid s _ {t}, a _ {t: t + i}) = \frac {1}{2} (\delta_ {a = 0} + \delta_ {a = 1}). \tag {92}
$$

Now, we derive the condition on $\delta$ when the optimal data ${ \mathcal { D } } ^ { \star }$ is $\varepsilon _ { h }$ -open-loop consistent. We start by calculating the TV distance discrepancy for the future state-action distribution:

$$
\begin{array}{l} D _ {\mathrm{TV}} (P _ {\mathcal {D} ^ {\star}} ^ {\mathrm{open}} (s _ {t + i}, a _ {t + i} \mid s _ {t}) \parallel P _ {\mathcal {D} ^ {\star}} (s _ {t + i}, a _ {t + i} \mid s _ {t})) \\ = \frac {1}{2} \left\| \left[ \begin{array}{c c} 0 & \delta / 2 \\ (1 - \delta) / 2 & (1 - \delta) / 2 \\ \delta / 2 & 0 \end{array} \right] - \left[ \begin{array}{c c} \delta / 4 & \delta / 4 \\ (1 - \delta) / 2 & (1 - \delta) / 2 \\ \delta / 4 & \delta / 4 \end{array} \right] \right\| _ {1, 1} \tag {93} \\ = \delta / 2. \\ \end{array}
$$

In the second line of the equations above, each row in the matrix corresponds to a distinct action $a _ { t + i } \in \{ 0 , 1 \}$ and each row in the matrix corresponds to a distinct state $s _ { t + i } \in \{ A _ { i } , X _ { i } , B _ { i } \}$ .

Next, we calculate the TV distance discrepancy for $s _ { t + h } .$ :

$$
\begin{array}{l} D _ {\mathrm{TV}} (P _ {\mathcal {D} ^ {\star}} ^ {\mathrm{open}} (s _ {t + h} \mid s _ {t}) \parallel P _ {\mathcal {D} ^ {\star}} (s _ {t + h} \mid s _ {t})) \\ = \frac {1}{2} \| [ 1 \quad 0 ] - [ 1 - \delta / 2 \quad \delta / 2 ] \| _ {1} \tag {94} \\ = \delta / 2. \\ \end{array}
$$

In the second line of the equations above, each element in the vector corresponds to a distinct state $s _ { t + h } \in \{ X _ { 0 } , Z \}$ . Up to now, we have concluded that ${ \mathcal { D } } ^ { \star }$ is $\left( \delta / 2 \right)$ -open-loop consistent.

Due to the symmetric structure of this MDP, it is clear that any action chunking policy $\pi _ { \mathrm { a c } } ( X _ { 0 } ) =$ $\scriptstyle a _ { t : t + h }$ with $a _ { t : t + h } \in \{ 0 , 1 \}$ is optimal and achieves the following value:

$$
\begin{array}{l} V _ {\mathrm{ac}} ^ {\star} \left(X _ {0}\right) = 1 + (1 - \delta / 2) \left[ \frac {\gamma - \gamma^ {h}}{1 - \gamma} + \gamma^ {h} V _ {\mathrm{ac}} ^ {\star} \left(X _ {0}\right) \right] \tag {95} \\ = \frac {(1 - \gamma) + (1 - \delta / 2) (\gamma - \gamma^ {h})}{(1 - \gamma) (1 - (1 - \delta / 2) \gamma^ {h})}. \\ \end{array}
$$

The optimal closed-loop policy can achieve the maximum possible return

$$
V ^ {\star} (X _ {0}) = \frac {1}{1 - \gamma}. \tag {96}
$$

Therefore, with $\varepsilon _ { h } = \delta / 2$ , the optimality gap achieved by this (3h − 1)-state MDP is

$$
V ^ {\star} (X _ {0}) - V _ {\mathrm{ac}} ^ {\star} (X _ {0}) = \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}, \tag {97}
$$

as desired.

![](images/6e49e15f22baa7855403851e4fb6117342549859cb993da6a2a62090d6b04eb9.jpg)

# F.6 PROOF OF PROPOSITION 1

Proposition 1 (AC Q-Learning under Weak OLC) For any $h > 1 , \gamma \in [ 0 , 1 ) , c \in [ 0 , 1 / 2 )$ $\varepsilon _ { h } \in ( 0 , 1 / 2 )$ , there exists an MDP M, a weakly $\varepsilon _ { h }$ -open-loop consistent D and ${ \mathcal { D } } ^ { \star }$ with supp $( P _ { \mathcal D } ( \dot { s } _ { t } , a _ { t : t + h } ) ) \supseteq \operatorname { s u p p } ( P _ { \mathcal D ^ { \star } } ( s _ { t } , a _ { t : t + h } ) )$ , such that for some $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) = V _ {\mathrm{ac}} ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) = \frac {\gamma c}{1 - \gamma}. \tag {20}
$$

Proof. To prove this theorem, we show an example where the optimal action chunking policy defined in Equation (19) can be highly sub-optimal in the absence of the strong open-loop consistency condition.

We define an MDP as follows. Let $S = \{ A , B , C , D , E , Z \}$ and $\mathcal { A } = \{ 0 , 1 \}$ . Define the transition dynamics and reward function as shown in the diagram below (Figure 10):

![](images/c27839afd939c5e3a48097cdbe5156dac5ec26a5b6b2f692d669fcadfc369af1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["A"] -->|a=0| B["B"]
    A -->|a=1| C["C"]
    B -->|a=0| D["D"]
    B -->|a=1| Z["Z"]
    C -->|a=0| Z
    C -->|a=1| E["E"]
    D -->|r=1| D
    D -->|r=0| Z
    E -->|r=c̃| E
    C -.->|δ| B
    C -.->|1-δ| C
```
</details>

Figure 10: A 6-state MDP that is constructed to illustrate the pathological failure mode of action chunking Q-learning under weak open-loop consistency.

where $\delta , \tilde { c } \in [ 0 , 1 )$ are real numbers and dotted lines denote stochastic transitions. The reward function depends only on states $( r ( A ) = r ( B ) = r ( C ) = r ( F ) = 0 , r ( D ) = 1$ , and $r ( G ) = c )$ Assume that the dataset is collected by a policy πD defined as $\pi _ { \mathcal { D } } ( A ) = 0$ (with probability θ) or 1 (with probability $( 1 - \theta ) ) , \pi _ { \mathcal { D } } ( B ) = 0$ (with probability 1), $\pi _ { \mathcal { D } } ( C ) = 1$ (with probability 1), and $\pi _ { \mathcal { D } } ( D ) = \pi _ { \mathcal { D } } ( Z ) = \pi _ { \mathcal { D } } ( E ) = 0$ (with probability 1).

Let $\pi ^ { \star }$ be the deterministic policy with $\pi ( A ) = 1 , \pi ( C ) = 1 , \pi ( E ) = 0$ . Then, it is clear that $\begin{array} { r } { \operatorname { s u p p } ( P _ { \mathcal { D } } ( s _ { t } , a _ { t : t + h } ) ) \supseteq \operatorname { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } , a _ { t : t + h } ) ) = \{ ( s _ { t } = A , a _ { t } = 1 , a _ { t + 1 } = 1 , a _ { t + 2 : t + h } = 0 ) \} } \end{array}$ . Now we are left to show that D is εh-open-loop consistent.

Since $D , E , Z$ are all self-loops, we only need to analyze the first two actions in the action chunk; the rest is all $a _ { t + 2 : t + h } = 0$ and does not affect the value.

We can compute the following:

$$
P _ {\mathcal {D}} (A, (0, 0)) = D, R (A, (0, 0)) = \gamma ,
$$

$$
P _ {\mathcal {D}} (A, (0, 1)) = E, R (A, (0, 1)) = 2 c \gamma , \tag {98}
$$

$$
P _ {\mathcal {D}} (A, (1, 1)) = E, R (A, (1, 1)) = 2 c \gamma ,
$$

where we denote action chunks as a tuple and slightly abuse notation to denote deterministic outputs of $P _ { \mathcal { D } } ( \cdot \mid s _ { 0 } , a _ { 0 : 2 } ) \left( e . g . , P _ { \mathcal { D } } ( A , ( 0 , 0 ) \right) ^ { - } = D$ indicates that all length-2 trajectories in D from state A with $a _ { 0 } = a _ { 1 } = 0$ have $s _ { 2 } = D$ with probability 1).

We can similarly compute the marginal state distributions as follows:

$$
\left[ \begin{array}{c c} P _ {\mathcal {D}} (D & A) \\ P _ {\mathcal {D}} (Z & A) \\ P _ {\mathcal {D}} (E & A) \end{array} \right] = \left[ \begin{array}{c} \theta \delta \\ 0 \\ 1 - \theta \delta \end{array} \right], \tag {99}
$$

and

$$
\left[ \begin{array}{l l} P _ {\mathcal {D}} (s _ {t + 1} = B, a _ {t} = 0 \mid s _ {t} = A) & P _ {\mathcal {D}} (s _ {t + 1} = B, a _ {t} = 1 \mid s _ {t} = A) \\ P _ {\mathcal {D}} (s _ {t + 1} = C, a _ {t} = 0 \mid s _ {t} = A) & P _ {\mathcal {D}} (s _ {t + 1} = C, a _ {t} = 1 \mid s _ {t} = A) \end{array} \right] = \left[ \begin{array}{c c} \theta \delta & 0 \\ 0 & 1 - \theta \delta \end{array} \right]. \tag {100}
$$

The marginal probability distribution of the action chunks is

$$
P _ {\mathcal {D}} (a _ {0, 1} = (0, 0) \mid A) = \theta \delta ,
$$

$$
P _ {\mathcal {D}} (a _ {0, 1} = (0, 1) \mid A) = (1 - \delta) \theta , \tag {101}
$$

$$
P _ {\mathcal {D}} (a _ {0, 1} = (1, 1) \mid A) = 1 - \theta .
$$

The induced $P _ { \mathcal { D } } ^ { \circ }$ is then

$$
\left[ \begin{array}{c} P _ {\mathcal {D}} ^ {\circ} (s _ {t + h} = D \mid s _ {t} = A) \\ P _ {\mathcal {D}} ^ {\circ} (s _ {t + h} = Z \mid s _ {t} = A) \\ P _ {\mathcal {D}} ^ {\circ} (s _ {t + h} = E \mid s _ {t} = A) \end{array} \right] = \left[ \begin{array}{c} \theta \delta^ {2} \\ (1 - \delta) \theta \delta \\ (1 - \delta) ^ {2} \theta + (1 - \theta) \end{array} \right], \tag {102}
$$

and

$$
\left[ \begin{array}{l l} P _ {\mathcal {D}} ^ {\circ} \left(s _ {t + 1} = B, a _ {t + 1} = 0 \mid s _ {t} = A\right) & P _ {\mathcal {D}} ^ {\circ} \left(s _ {t + 1} = B, a _ {t + 1} = 1 \mid s _ {t} = A\right) \\ P _ {\mathcal {D}} ^ {\circ} \left(s _ {t + 1} = C, a _ {t + 1} = 0 \mid s _ {t} = A\right) & P _ {\mathcal {D}} ^ {\circ} \left(s _ {t + 1} = C, a _ {t + 1} = 1 \mid s _ {t} = A\right) \end{array} \right] \tag {103}
$$

$$
= \left[ \begin{array}{c c} \delta^ {2} \theta & (1 - \delta) \delta \theta \\ \delta (1 - \delta) \theta & (1 - \delta) ^ {2} \theta + 1 - \theta \end{array} \right].
$$

We can then compute

$$
\begin{array}{l} D _ {\mathrm{TV}} \left(P _ {\mathcal {D}} ^ {\circ} \left(s _ {t + 1}, a _ {t + 1} \mid s _ {t} = A\right) \| P _ {\mathcal {D}} \left(s _ {t + 1}, a _ {t + 1} \mid s _ {t} = A\right)\right) = 2 \theta \delta (1 - \delta) (104) \\ D _ {\mathrm{TV}} (P _ {\mathcal {D}} ^ {\circ} (s _ {t + h} \mid s _ {t} = A) \parallel P _ {\mathcal {D}} (s _ {t + h} \mid s _ {t} = A)) = \frac {3}{2} \theta \delta (1 - \delta) (104) \\ \end{array}
$$

Note that we only need to check t + 1 time step for the state-action distribution because

$$
\begin{array}{l} \begin{array}{l} D _ {\mathrm{TV}} \left(P _ {\mathcal {D}} ^ {\circ} \left(s _ {t + h ^ {\prime}}, a _ {t + h ^ {\prime}} \mid s _ {t} = A\right) \| P _ {\mathcal {D}} \left(s _ {t + h ^ {\prime}}, a _ {t + h ^ {\prime}} \mid s _ {t} = A\right)\right) = \\ P _ {\text {二}} \left(P _ {\text {三}} ^ {\circ} \left(s _ {t + h ^ {\prime}}, a _ {t + h ^ {\prime}} \mid s _ {t} = A\right) \| P _ {\text {四}} \left(s _ {t + h ^ {\prime}}, a _ {t + h ^ {\prime}} \mid s _ {t} = A\right)\right) \times 1 / 2 ^ {\prime} = 1 \end{array} \tag {105} \\ D _ {\mathrm{TV}} (P _ {\mathcal {D}} ^ {\circ} (s _ {t + h} \mid s _ {t} = A) \parallel P _ {\mathcal {D}} (s _ {t + h} \mid s _ {t} = A)), \quad \forall h ^ {\prime} > 1, \\ \end{array}
$$

coming from the fact that the rest of the action chunks are all constant $0 \left( i . e . , a _ { t + 2 : t + h } = 0 \right)$ .

Now, we can set δ to be solution of $\varepsilon _ { h } = 2 ( 1 - \delta ) \delta \theta$ , such that D is εh-open-loop consistent as required by the assumption.

Next, we analyze the action chunking policy $\pi _ { \mathrm { a c } } ^ { + }$ learned from this εh-open-loop consistent data distribution. We start by calculating $\hat { Q } _ { \mathrm { a c } } ^ { + }$ as follows:

$$
\hat {Q} _ {\mathrm{ac}} ^ {+} (A, (0, 0)) = \frac {\gamma}{1 - \gamma},
$$

$$
\hat {Q} _ {\mathrm{ac}} ^ {+} (A, (0, 1)) = \frac {\tilde {c} \gamma}{1 - \gamma}, \tag {106}
$$

$$
\hat {Q} _ {\mathrm{ac}} ^ {+} (A, (1, 1)) = \frac {\tilde {c} \gamma}{1 - \gamma}.
$$

The optimal action chunking policy is then $\hat { \pi } _ { \mathrm { a c } } ^ { + } ( A ) = ( 0 , 0 )$ (Equation (19)) because $c < 1$ .

The true value of this action chunking policy is

$$
V _ {\mathrm{ac}} ^ {+} (A) = \frac {\delta \gamma}{1 - \gamma} \tag {107}
$$

As long as $\tilde { c } > \delta ,$ , the optimal strategy in this MDP is to always choose $( a _ { 0 } , a _ { 1 } ) = ( 1 , 1 )$ , in which case the agent receives a constant return:

$$
V ^ {\star} (A) = V _ {\mathrm{ac}} ^ {\star} (A) = \frac {\tilde {c} \gamma}{1 - \gamma}. \tag {108}
$$

The optimality gap in this example is therefore

$$
V ^ {\star} (A) - V _ {\mathrm{ac}} ^ {+} (A) = (\tilde {c} - \delta) \frac {\gamma}{1 - \gamma}. \tag {109}
$$

Finally, we solve $\varepsilon _ { h } = 2 ( 1 - \delta ) \delta \theta$ and pick the smaller solution:

$$
\delta = \frac {1 - \sqrt {1 - 2 \varepsilon_ {h} / \theta}}{2}. \tag {110}
$$

If we set $\theta = 2 \varepsilon _ { h }$ and $\begin{array} { r } { \tilde { c } = c + \frac { 1 } { 2 } } \end{array}$ , then we get

$$
V ^ {\star} (A) - V _ {\mathrm{ac}} ^ {+} (A) = \frac {c \gamma}{1 - \gamma}, \tag {111}
$$

as desired.

As a small extra note, the last step is where the assumption $\varepsilon _ { h } > 0$ becomes necessary, since otherwise the term $2 \varepsilon _ { h } / \theta$ (with $\theta = 2 \varepsilon _ { h } )$ would be undefined.

# F.7 PROOF OF THEOREM 3

Theorem 3 (AC Q-Learning under Strong OLC) If D and $\mathcal { D } ^ { \star }$ are strongly $\varepsilon _ { h }$ -open-loop consistent and supp $\left( P _ { \mathcal { D } } ( s _ { t } , a _ { t : t + h } ) \right) \supseteq \operatorname { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } , a _ { t : t + h } ) )$ , then for all $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) \leq \frac {\varepsilon_ {h} \gamma}{1 - \gamma} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right] \leq 3 \varepsilon_ {h} H \bar {H}, \tag {21}
$$

where $V ^ { \star }$ is the value of a closed-loop optimal policy and $V _ { \mathrm { a c } } ^ { + }$ is the true value of $\pi _ { \mathrm { a c } } ^ { + }$

Proof of Theorem 3. We start by constructing a bound between $\hat { Q } _ { \mathrm { a c } } ^ { + }$ and $Q _ { \mathrm { a c } } ^ { \star }$ , the solution of the following bellman equation:

$$
Q _ {\mathrm{ac}} ^ {\star} (s _ {t}, a _ {t: t + h}) = \mathbb {E} _ {s _ {t + 1: t + h + 1} \sim P _ {\mathcal {D}} ^ {\circ} (\cdot | s _ {t}, a _ {t: t + h})} \left[ R _ {t: t + h} + \gamma^ {h} \max _ {a _ {t + h: t + 2 h}} Q _ {\mathrm{ac}} ^ {\star} (s _ {t + h}, a _ {t + h: t + 2 h}) \right]. \tag {112}
$$

Intuitively, $Q _ { \mathrm { a c } } ^ { \star }$ is the Q-function of the optimal action chunking policy $\pi _ { \mathrm { a c } } ^ { \star }$ that can be learned from D. Because supp $\begin{array} { r } { \vartheta ) \supseteq \mathrm { s u p p } ( \mathcal { D } ^ { \star } ) , \pi _ { \mathrm { a c } } ^ { \star } } \end{array}$ is at least as good as $\tilde { \pi } _ { \mathrm { a c } } .$ , the action chunking policy obtained by behavior cloning ${ \mathcal { D } } ^ { \star }$ . Bounding the difference between $\hat { Q } _ { \mathrm { a c } } ^ { + }$ and $Q _ { \mathrm { a c } } ^ { \star }$ allows us to leverage the bound in Corollary 1 to form a bound between $\hat { V } _ { \mathrm { a c } } ^ { + }$ and $V ^ { \star }$ .

Since D is strongly εh-open-loop consistent,

$$
D _ {\mathrm{TV}} (T (s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h ^ {\prime}}) \| P _ {\mathcal {D}} (s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h})) \leq \varepsilon_ {h}, \forall h ^ {\prime} \in \{1, \dots , h - 1 \}. \tag {113}
$$

Since ${ \mathcal { D } } ^ { \star }$ is also strongly $\varepsilon _ { h }$ -open-loop consistent,

$$
D _ {\mathrm{TV}} (T (s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h ^ {\prime}}) \parallel P _ {\mathcal {D} ^ {*}} (s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h})) \leq \varepsilon_ {h}, \forall h ^ {\prime} \in \{1, \dots , h - 1 \}. \tag {114}
$$

Using the transitive property of TV distance, we have

$$
D _ {\mathrm{TV}} (P _ {\mathcal {D}} (s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h}) \parallel P _ {\mathcal {D} ^ {*}} (s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h})) \leq 2 \varepsilon_ {h}, \forall h ^ {\prime} \in \{1, \dots , h - 1 \}. \tag {115}
$$

Now, for the h-step reward, we have

$$
\begin{array}{l} \left| \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h})} [ R _ {t: t + h} ] - \mathbb {E} _ {P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t}, a _ {t: t + h})} [ R _ {t: t + h} ] \right| \\ \leq \sum_ {h ^ {\prime} = 1} ^ {h - 1} \left[ \gamma^ {h ^ {\prime}} D _ {\mathrm{TV}} (P _ {\mathcal {D}} (s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h}) \parallel P _ {\mathcal {D} ^ {*}} (s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h})) \right] \tag {116} \\ \leq \frac {2 (\gamma - \gamma^ {h}) \varepsilon_ {h}}{1 - \gamma}. \\ \end{array}
$$

Similarly, for the value h-step into the future, we can use Lemma 2 to obtain the following bound:

$$
\begin{array}{l} \left| \mathbb {E} _ {s _ {t + h} \sim P _ {\mathcal {D}} (s _ {t + h} | s _ {t})} \left[ V ^ {\star} (s _ {t + h}) \right] - \mathbb {E} _ {s _ {t + h} \sim P _ {\mathcal {D} ^ {\star}} (s _ {t + h} | s _ {t})} \left[ \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) \right] \right| \tag {117} \\ \leq 2 \varepsilon_ {h} + (1 - 2 \varepsilon_ {h}) \sup _ {s _ {t + h} \in \mathcal {D} ^ {\star}} \left| V ^ {\star} (s _ {t + h}) - \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) \right|. \\ \end{array}
$$

We define $Q ^ { \star } \big ( s _ { t } , a _ { t : t + h } \big )$ to be

$$
Q ^ {\star} (s _ {t}, a _ {t: t + h}) := \mathbb {E} _ {P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t}, a _ {t: t + h})} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} (s _ {t + h}) \right]. \tag {118}
$$

It is clear that

$$
V ^ {\star} (s _ {t}) = \mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D} ^ {\star}}} \left[ Q ^ {\star} (s _ {t}, a _ {t: t + h}) \right]. \tag {119}
$$

Combining the bound for the h-step reward and the bound on the value for $s _ { t + h }$ , for all $s _ { t } , a _ { t : t + h } \in$ supp( $P _ { \mathcal { D } ^ { \star } } \left( s _ { t } , a _ { t : t + h } \right) \big )$ ,

$$
\begin{array}{l} \Delta (s _ {t}, a _ {t: t + h}) = Q ^ {\star} (s _ {t}, a _ {t: t + h}) - \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h}) \\ \leq 2 \varepsilon_ {h} \gamma^ {h} + \frac {2 (\gamma - \gamma^ {h}) \varepsilon_ {h}}{1 - \gamma} + (1 - 2 \varepsilon_ {h}) \gamma^ {h} \left(V ^ {\star} (s _ {t + h}) - \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h})\right) \\ \leq \frac {2 \varepsilon_ {h} \gamma}{1 - \gamma} + (1 - 2 \varepsilon_ {h}) \gamma^ {h} \left(\mathbb {E} _ {P _ {\mathcal {D} ^ {\star}}} \left[ Q ^ {\star} (s _ {t + h}, a _ {t + h: t + 2 h}) \right] - \sup _ {a _ {t + h: t + 2 h}} \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t + h}, a _ {t + h: t + 2 h})\right) \\ \leq \frac {2 \varepsilon_ {h} \gamma}{1 - \gamma} + (1 - 2 \varepsilon_ {h}) \gamma^ {h} \left(\mathbb {E} _ {P _ {\mathcal {D} ^ {*}}} \left[ \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t + h}, a _ {t + h: t + 2 h}) + \Delta (s _ {t + h}, a _ {t + h: t + 2 h}) \right] - \sup _ {a _ {t + h: t + 2 h}} \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t + h}, a _ {t + h: t + 2 h})\right) \\ \leq \frac {2 \varepsilon_ {h} \gamma}{1 - \gamma} + (1 - 2 \varepsilon_ {h}) \gamma^ {h} \sup _ {s _ {t + h}, a _ {t + h: t + 2 h}} [ \Delta (s _ {t + h}, a _ {t + h: t + 2 h}) ], \tag {120} \\ \end{array}
$$

which can be recursively expanded to obtain

$$
V ^ {\star} (s _ {t}) - \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) \leq \frac {2 \varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})}. \tag {121}
$$

By Theorem 1, for all $s _ { t } \in$ supp(D),

$$
\left| \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) \right| \leq \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}. \tag {122}
$$

Combining the two inequalities above, for all $s _ { t } \in \mathrm { s u p p } ( \mathcal { D } ^ { \star } )$ ,

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) \leq \frac {\varepsilon_ {h} \gamma}{1 - \gamma} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right]. \tag {123}
$$

![](images/af0dc7c8825d1b6cc8dd9dd96557df27c43a3b8ac72ed0bd7b329973b68a083a.jpg)

# F.8 PROOF OF THEOREM 4

Theorem 4 (Worst-case Analysis of Q-Learning with Action Chunking Policy on Off-policy Data) For any $h > 1 , \gamma \in ( 0 , 1 ) , \varepsilon _ { h } \in ( 0 , 1 / \bar { 5 } ) , c _ { 1 } \in ( 0 , \varepsilon _ { h } / 2 )$ , and $c _ { 2 } \in ( 0 , 2 \varepsilon _ { h } \gamma )$ , there exists an MDP M and strongly εh-open-loop consistent data distributions D and $\dot { \mathcal { D } } ^ { \star }$ with supp $( P _ { \mathcal D } ( s _ { t } , a _ { t : t + h } ) ) \supseteq \mathrm { s u p p } \overline { { ( P _ { \mathcal D ^ { \star } } ( s _ { t } , a _ { t : t + h } ) } ) }$ , such that for some $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) = \frac {2 \varepsilon_ {h} \gamma - c _ {2}}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})} + \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - \varepsilon_ {h} - c _ {1}) \gamma^ {h})}, \tag {22}
$$

where $V ^ { \star }$ is the value of an optimal policy and $V _ { \mathrm { a c } } ^ { + }$ is the true value of $\pi _ { \mathrm { a c } } ^ { + } . \mathrm { A s } c _ { 1 } , c _ { 2 }  0 ,$

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) \rightarrow \frac {\varepsilon_ {h} \gamma}{1 - \gamma} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right]. \tag {23}
$$

The examples in the following proof of Theorem 4 (available in Section F.8) provide insights on the factor of 3 in $V ^ { \star } - V _ { \mathrm { a c } } ^ { + } \leq \bar { 3 } \varepsilon _ { h } H \bar { H }$ (with $H = 1 / ( 1 - \gamma ) , \bar { H } = 1 / ( 1 - \gamma ^ { \bar { h } } ) )$ is necessary. In particular, the worst case can be roughly seen as a combination of the two main results that we have presented so far:

1. $V ^ { \star } - V _ { \mathrm { a c } } ^ { \star }$ ≈ $\varepsilon _ { h } H \bar { H }$ (Corollary 1, Corollary 2): the optimal action chunking policy is $\left( \varepsilon _ { h } H ^ { 2 } \right)$ -sub-optimal due to its inability to react to environment stochasticity, quantified by the strongly-εh open-loop consistency of $\mathcal { D } ^ { \star }$ .   
2. $V _ { \mathrm { a c } } ^ { \star } - \hat { V } _ { \mathrm { a c } } ^ { + } \approx \varepsilon _ { h } H \bar { H }$ (a transformation of Theorem 1 and Theorem 2 on the optimal action chunking policy $\pi _ { \mathrm { a c } } ^ { \star } ) \colon$ : the value under-estimation bias can incur another factor of $\varepsilon _ { h } H \bar { H }$ bringing up the sub-optimality of $\hat { V } _ { \mathrm { a c } } ^ { + }$ to at most $2 \varepsilon _ { h } H \bar { H }$ , and finally,   
3. $\hat { V } _ { \mathrm { a c } } ^ { + } - V _ { \mathrm { a c } } ^ { + } \approx \varepsilon _ { h } H \bar { H }$ (Theorem 1, Theorem 2): the action chunking value functi+ prefer an overestimated action chunking policy $\pi _ { \mathrm { a c } } ^ { + }$ where its actual value is again $\varepsilon _ { h } H \bar { H }$ from its estimated value, resulting in a total sub-optimality of $3 \varepsilon _ { h } H \bar { H }$ .

Our construction (in the proof of Theorem 4) directly builds on the above insights by using a 2-part MDP where the first part corresponds to an $\left( \bar { \varepsilon } _ { h } H \bar { H } \right)$ -underestimated action chunking policy that has $\mathrm { ~ a ~ } ( \varepsilon _ { h } H \bar { H } )$ -optimality gap from the optimal closed-loop policy and the second part corresponds to an $( \varepsilon _ { h } H \bar { H } )$ )-overestimated action chunking policy that has a $( 3 \dot { \varepsilon } _ { h } H \bar { H } )$ -optimality gap that is preferred by the value function.

Before we start our main proof, we first introduce a Lemma that helps simplifies the inequalities.

Lemma 5 (Optimality gap comparator) For any $\tilde { \gamma } \in [ 0 , 1 )$ and $0 < \varepsilon _ { 1 } < \varepsilon _ { 2 } < 1$

$$
\frac {\varepsilon_ {1}}{1 - (1 - \varepsilon_ {1}) \tilde {\gamma}} <   \frac {\varepsilon_ {2}}{1 - (1 - \varepsilon_ {2}) \tilde {\gamma}}. \tag {124}
$$

Proof.

$$
\begin{array}{l} 0 <   (1 - \gamma) (\varepsilon_ {2} - \varepsilon_ {1}) \\ = \varepsilon_ {2} - \varepsilon_ {2} \tilde {\gamma} - \varepsilon_ {1} + \varepsilon_ {1} \tilde {\gamma} \tag {125} \\ = \varepsilon_ {2} - \varepsilon_ {2} \tilde {\gamma} + \varepsilon_ {1} \varepsilon_ {2} \tilde {\gamma} - \varepsilon_ {1} + \varepsilon_ {1} \tilde {\gamma} - \varepsilon_ {1} \varepsilon_ {2} \tilde {\gamma} \\ = \varepsilon_ {2} (1 - (1 - \varepsilon_ {1}) \tilde {\gamma}) - \varepsilon_ {1} (1 - (1 - \varepsilon_ {2}) \tilde {\gamma}) \\ \end{array}
$$

Since $1 - ( 1 - \varepsilon _ { 1 } ) \tilde { \gamma } > 0$ and $1 - ( 1 - \varepsilon _ { 2 } ) \tilde { \gamma } > 0 \nonumber$ , we can divide both sides by $( 1 - ( 1 - \varepsilon _ { 1 } ) \tilde { \gamma } ) ( 1 -$ $( 1 - \varepsilon _ { 2 } ) \tilde { \gamma } )$ to get

$$
0 <   \frac {\varepsilon_ {2}}{1 - (1 - \varepsilon_ {2}) \tilde {\gamma}} - \frac {\varepsilon_ {1}}{1 - (1 - \varepsilon_ {1}) \tilde {\gamma}}, \tag {126}
$$

as desired.

![](images/3a040a7e58dc5893fa03d1634be1d8b26cef5dfe2056d6a5e178258331748f5c.jpg)

Now, we begin the main proof as follows.

Proof of Theorem 4. We prove by constructing the following (2h + 4)-state MDP where the agent can take any of the three actions {0, 1, 2} at each state (see a diagram in Figure 11).

Notations: we start by introducing some abbreviations for all action chunks that appear in this proof:

$$
a _ {t: t + h} ^ {\star} = (0, 0, 0, \dots , 0)
$$

$$
a _ {t: t + h} ^ {\diamond} = (0, 1, 0, \dots , 0)
$$

$$
a _ {t: t + h} ^ {\bullet} = (0, 2, 0, \dots , 0)
$$

$$
a _ {t: t + h} ^ {\triangle} = (1, 1, 1, \dots , 1) \tag {127}
$$

$$
a _ {t: t + h} ^ {\circ} = (1, 0, 1, \dots , 1)
$$

$$
a _ {t: t + h} ^ {\times} = (1, 2, 1, \dots , 1)
$$

The first three action chunks $a _ { t : t + h } ^ { \star } , a _ { t : t + h } ^ { \diamond } , a _ { t : t + h } ^ { \bullet }$ are only possible in the top branch and the last three action chunks $a _ { t : t + h } ^ { \triangle } , a _ { t : t + h } ^ { \circ } , a _ { t : t + h } ^ { \times }$ △ are only possible in the bottom branch because the first action in the action chunk deterministically divides it into the two branches.

Among these action chunks, it is clear by inspection that $\pi _ { \mathrm { a c } } ( X _ { 0 } ) = ( 0 , 0 , \cdot \cdot \cdot , 0 ) $ is the optimal action chunking policy, and thus we directly use $\cdot _ { \star } \cdot$ to denote $a _ { t : t + h } ^ { \star } = ( 0 , 0 , \cdot \cdot \cdot , 0 ) . \ : a _ { t : t + h } ^ { \triangle }$ is also of great importance: as we will show later, $\pi _ { \mathrm { a c } } ^ { + } ( X _ { 0 } ) = a _ { t : t + h } ^ { \triangle }$ . The actual values and nominal/estimated values for these action chunks are $( V _ { \mathrm { a c } } ^ { \star } , V _ { \mathrm { a c } } ^ { \diamond } , V _ { \mathrm { a c } } ^ { \bullet } , V _ { \mathrm { a c } } ^ { \triangle } , V _ { \mathrm { a c } } ^ { \circ } , V _ { \mathrm { a c } } ^ { \times } )$ and $( \hat { V } _ { \mathrm { a c } } ^ { \star } , \hat { V } _ { \mathrm { a c } } ^ { \diamond } , \hat { V } _ { \mathrm { a c } } ^ { \bullet } , \hat { V } _ { \mathrm { a c } } ^ { \triangle } , \hat { V } _ { \mathrm { a c } } ^ { \circ } , \hat { V } _ { \mathrm { a c } } ^ { \times } )$ Vˆ ◦ , V respectively. Much of the focus of this proof is to calculate the optimality gap, which is the difference between the optimal closed-loop value and the action chunking policy value (either estimated or actual):

$$
\text { actual   optimality   gap: } \quad V ^ {\star} (X _ {0}) - V _ {\mathrm{ac}} ^ {[ \cdot ]} (X _ {0}) \tag {128}
$$

$$
\text { nominal   optimality   gap: } \quad V ^ {\star} (X _ {0}) - \hat {V} _ {\mathrm{ac}} ^ {[ \cdot ]} (X _ {0}) \tag {129}
$$

High-level proof sketch: The MDP contains two branches: a top branch where (as we will show) both the optimal policy $\pi ^ { \star }$ and the optimal action chunking policy $\pi _ { \mathrm { a c } } ^ { \star }$ take, and a bottom branch where (as we will also show) the learned action chunking policy $\pi _ { \mathrm { a c } } ^ { + }$ takes. The key idea of the construction is that for the top branch, we have

$$
V ^ {\star} (X _ {0}) - \hat {V} _ {\mathrm{ac}} ^ {\star} (X _ {0}) \approx \frac {2 \varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})}, \tag {130}
$$

and for the bottom branch, we have

$$
\hat {V} _ {\mathrm{ac}} ^ {\star} (X _ {0}) <   \hat {V} _ {\mathrm{ac}} ^ {+} (X _ {0}) \approx V _ {\mathrm{ac}} ^ {+} (X _ {0}) + \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - \varepsilon_ {h}) \gamma^ {h})}. \tag {131}
$$

Combining these two together gives

$$
V ^ {\star} (X _ {0}) - V _ {\mathrm{ac}} ^ {+} (X _ {0}) \approx \frac {\varepsilon_ {h} \gamma}{1 - \gamma} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right]. \tag {132}
$$

We use $\cdot \approx $ because the equalities are not strictly achievable but (as we will show) can be made arbitrarily close.

The proof can be roughly divided into the following steps (we use $\cdot _ { \approx } ,$ to help illustrate the high-level idea below and use more precise argument in the actual proof):

1. MDP description: we formally describe the transition dynamics T and the reward function r for each state-action pair for both the top and the bottom branches.   
2. Strong $\varepsilon _ { h } - \mathsf { o p e n - 1 0 } \mathsf { o p }$ consistency of $\mathcal { D } ^ { \star }$ : we then check the strong open-loop consistency assumption for ${ \mathcal { D } } ^ { \star }$ .   
3. Data distribution $\mathcal { D } _ { \mathrm { t o p } }$ for the top branch: we use a mixture data distribution from two policies to construct $\mathcal { D } _ { \mathrm { t o p } }$ .

4. Strong εh-open-loop consistency of $\mathcal { D } _ { \mathrm { t o p } } \colon$ we then check that the constructed data distribution of the top branch satisfies the strongly open-loop consistency assumption. Note that we can do so separately for the top and the bottom because these two distributions have non-overlapping support in $a _ { t : t + h }$ .   
5. The optimality gap and value estimation error for the top branch: we prove that $\begin{array} { r l r } { V ^ { \star } ( X _ { 0 } ) - V _ { \mathrm { a c } } ^ { \star } ( X _ { 0 } ) } & { = } & { { \frac { \varepsilon _ { h } \gamma } { ( 1 - \gamma ) ( 1 - ( 1 - 2 \varepsilon _ { h } ) \gamma ^ { h } ) } } } \end{array}$ and $V ^ { \star } ( X _ { 0 } ) ~ - ~ \hat { V } _ { \mathrm { a c } } ^ { \star } ( X _ { 0 } ) ~ =$ (1−γ)(1−(1−2εh)γh) and the other two possible action chunks a⋄t:t+h = (0, 1, 0, · · · ) and $\frac { 2 \varepsilon _ { h } \gamma } { ( 1 - \gamma ) ( 1 - ( 1 - 2 \varepsilon _ { h } ) \gamma ^ { h } ) }$ 2εhγ $a _ { t : t + h } ^ { \diamond } = ( 0 , 1 , 0 , \cdot \cdot \cdot )$ $a _ { t : t + h } ^ { \bullet } = ( 0 , 2 , 0 , \cdots )$ both admit lower estimated values compared to $a _ { t : t + h } ^ { \star } \colon \hat { V } _ { \mathrm { a c } } ^ { \diamond } ( X _ { 0 } ) <$ $\hat { V } _ { \mathrm { a c } } ^ { \star } ( X _ { 0 } )$ and $\hat { V } _ { \mathrm { a c } } ^ { \bullet } ( X _ { 0 } ) < \hat { V } _ { \mathrm { a c } } ^ { \star } ( X _ { 0 } )$ .   
6. Data distribution $\mathcal { D } _ { \mathrm { b o t t o m } }$ for the bottom branch: we again use a mixture data distribution from two different policies to construct $\mathcal { D } _ { \mathrm { b o t t o m } }$ .   
7. Strong εh-open-loop consistency of $\mathcal { D } _ { \mathrm { b o t t o m } } \colon$ we then check that the constructed data distribution of the bottom branch satisfies the strongly open-loop consistency assumption.   
8. The optimality gap and value estimation error for the bottom branch: we prove that $\begin{array} { r } { V ^ { \star } ( X _ { 0 } ) - \hat { V } _ { \mathrm { a c } } ^ { \bigtriangleup } ( X _ { 0 } ) \approx \frac { 2 \varepsilon _ { h } \gamma } { ( 1 - \gamma ) ( 1 - ( 1 - 2 \varepsilon _ { h } ) \gamma ^ { h } ) } } \end{array}$ 2εhγ(1−γ)(1−(1−2εh)γh) and Vˆ △ac (X0) − V △ac (X0) = $\hat { V } _ { \mathrm { a c } } ^ { \triangle } ( X _ { 0 } ) - V _ { \mathrm { a c } } ^ { \triangle } ( X _ { 0 } ) =$ $\frac { \varepsilon _ { h } \gamma } { ( 1 - \gamma ) ( 1 - ( 1 - \varepsilon _ { h } ) \gamma ^ { h } ) }$ , and the other two possible action chunks $a _ { t : t + h } ^ { \circ } ~ = ~ ( 1 , 0 , 0 , \cdots )$ and $a _ { t : t + h } ^ { \times } ~ = ~ ( 1 , 2 , 0 , \cdot \cdot \cdot )$ both admit lower estimated values compared to $a _ { t : t + h } ^ { \triangle } \colon$ $\hat { V } _ { \mathrm { a c } } ^ { \circ } ( X _ { 0 } ) < \hat { V } _ { \mathrm { a c } } ^ { \triangle } ( X _ { 0 } )$ and $\hat { V } _ { \mathrm { a c } } ^ { \times } ( X _ { 0 } ) < \hat { V } _ { \mathrm { a c } } ^ { \triangle } ( X _ { 0 } )$ . Moreover $a _ { t : t + h } ^ { \star }$ also admits a lower estimated value compared to $a _ { t : t + h } ^ { \triangle } \colon \hat { V } _ { \mathrm { a c } } ^ { \star } ( X _ { 0 } ) < \hat { V } _ { \mathrm { a c } } ^ { \triangle } ( X _ { 0 } )$ which proves $\pi _ { \mathrm { a c } } ^ { + } ( X _ { 0 } ) = a _ { t : t + h } ^ { \triangle }$ and thus concluding our proof: V ⋆(X0) − V +ac (X0) ≈ (1−γ)(1−(1−2εh)γh) + εhγ(1−γ)(1−(1−εh)γh) . $\begin{array} { r } { V ^ { \star } ( X _ { 0 } ) - V _ { \mathrm { a c } } ^ { + } ( X _ { 0 } ) \approx \frac { 2 \varepsilon _ { h } \gamma } { ( 1 - \gamma ) ( 1 - ( 1 - 2 \varepsilon _ { h } ) \gamma ^ { h } ) } + \frac { \varepsilon _ { h } \gamma } { ( 1 - \gamma ) ( 1 - ( 1 - \varepsilon _ { h } ) \gamma ^ { h } ) } } \end{array}$ 2εhγ

Now we begin our proof as follows.

Step 1. MDP description (Figure 11).

The transition function T of the MDP is defined as follows (from left to right):

$$
\begin{array}{l} T (Z \mid Z, a) = T (G \mid G, a) = 1, \quad \forall a, \\ T (Z \mid s, a = 2) = 1, \quad \forall a, \forall s: s \neq G \\ T (X _ {1} \mid X _ {0}, a = 0) = 1 - 2 \varepsilon_ {h} \\ T (\tilde {X} _ {1} \mid X _ {0}, a = 0) = \varepsilon_ {h} \\ T (C \mid X _ {0}, a = 0) = \varepsilon_ {h} \\ T (Y _ {1} \mid X _ {0}, a = 1) = 1 - \varepsilon_ {h} - c _ {1} \\ T (\tilde {Y} _ {1} \mid X _ {0}, a = 1) = \varepsilon_ {h} \\ T (G \mid X _ {0}, a = 1) = c _ {1} \\ T (X _ {2} \mid X _ {1}, a = 0) = 1 \\ \begin{array}{c} T (X _ {2} \mid \tilde {X} _ {1}, a = 1) = 1 \\ T (X _ {2} \mid C, a = 1) = 1 \end{array} \tag {133} \\ T (Z \mid X _ {1}, a = 1) = 1 \\ T (Z \mid C, a = 0) = 1 \\ T (G \mid \tilde {X} _ {1}, a = 0) = 1 \\ T (Y _ {2} \mid Y _ {1}, a = 1) = 1 \\ T \left(Y _ {2} \mid \tilde {Y} _ {1}, a = 0\right) = 1 \\ T (Z \mid Y _ {1}, a = 0) = 1 \\ T (Z \mid \tilde {Y} _ {1}, a = 1) = 1 \\ \end{array}
$$

$$
T \left(X _ {i + 1} \mid X _ {i}, a \in \{0, 1 \}\right) = T \left(Y _ {i + 1} \mid Y _ {i}, a \in \{0, 1 \}\right) = 1, \quad \forall i \in \{2, \dots , h - 2 \}
$$

$$
T (X _ {0} \mid X _ {h - 1}, a \in \{0, 1 \}) = T (Y _ {0} \mid Y _ {h - 1}, a \in \{0, 1 \}) = 1
$$

![](images/e63cfc12539764e62e497d48b1bd7bd86f922b91bdaa96fa6cad289962034175.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["X0"] --> B["a=0"]
    A --> C["r=1"]
    B --> D["C"]
    C --> E["X̃1"]
    D --> F["X1"]
    E --> G["X2"]
    F --> H["Xh-1"]
    G --> I["X0"]
    H --> J["X0"]
    K["Y1"] --> L["a=1"]
    K --> M["r=0"]
    L --> N["c1"]
    M --> O["c1"]
    N --> P["G"]
    O --> Q["G"]
    P --> R["Y2"]
    Q --> S["Y2"]
    T["Y1"] --> U["a=0"]
    T --> V["r=1-δ"]
    U --> W["Z"]
    V --> X["Z"]
    W --> Y["..."]
    X --> Z["..."]
    Y --> AA["..."]
    Z --> AB["..."]
    AA --> AC["..."]
    AB --> AD["..."]
    AC --> AE["..."]
    AD --> AF["..."]
    AE --> AG["..."]
    AF --> AH["..."]
    AG --> AI["..."]
    AH --> AJ["..."]
    AI --> AK["..."]
    AJ --> AL["..."]
    AK --> AM["..."]
    AL --> AN["..."]
    AM --> AO["..."]
    AN --> AP["..."]
    AO --> AQ["..."]
    AP --> AR["..."]
    AQ --> AS["..."]
    AR --> AT["..."]
```
</details>

Figure 11: A (2h + 4)-state MDP that is constructed to illustrate the MDP constructed to meet the exact upper-bound optimality gap in Theorem 3. We redraw the same states Z, G, $X _ { 0 }$ in multiple locations in the diagram above for better clarity.

The reward function is defined as

$$
r (Z, a) = 0, \quad \forall a
$$

$$
r (G, a) = 1, \quad \forall a
$$

$$
r (s, a = 2) = 0, \quad \forall s: s \neq G
$$

$$
r (X _ {0}, a = 0) = r (X _ {0}, a = 1) = 1,
$$

$$
\begin{array}{l} r (C, a = 1) = r \left(X _ {1}, a = 0\right) = r \left(\tilde {X} _ {1}, a \in \{0, 1 \}\right) = 1, \\ r (C, a = 0) = r \left(X _ {1}, a = 1\right) = 0, \tag {134} \\ \end{array}
$$

$$
r (Y _ {1}, a = 1) = r (\tilde {Y} _ {1}, a = 0) = 1 - \delta ,
$$

$$
r (Y _ {1}, a = 0) = r (\tilde {Y} _ {1}, a = 1) = 0,
$$

$$
r (X _ {i}, a \in \{0, 1 \}) = 1, \quad \forall i \in \{2, \dots , h - 1 \}
$$

$$
r \left(Y _ {i}, a \in \{0, 1 \}\right) = 1 - \delta , \quad \forall i \in \{2, \dots , h - 1 \}
$$

Notably, there are some special states:

• State Z: a self-looping “black hole” state that always gets 0 reward at each time step and thus has a constant value of 0.

• State $G \mathrm { : }$ a self-looping “black hole” state that always gets 1 reward at each time step and thus has a constant value of $1 / ( 1 - \gamma )$ .   
• State $X _ { 0 } { \mathrm { : } }$ the special state that branches out based on the action taken. The agent periodically visit this state every h steps unless it has been trapped in either $Z$ or $G$ . As we proceed in the proof, we will encounter factors in the form of $\frac { 1 } { 1 - b \gamma ^ { h } }$ in the calculation of the optimality gap. These factors come from the agent looping around and revisiting $X _ { 0 }$ with b-probability each cycle.

These two absorbing states are important because their values sit at the boundary of the value range of our value function $\breve { V } ( s ) \in [ 0 , 1 / ( 1 - \gamma ) ]$ ]. Shifting the reaching probability from $Z$ to $G$ or the other way around results in the biggest possible difference in the policy value. Our construction hinges on the constructing D such that

1. $P _ { \cal D } ( \cdot \mid s _ { t } , \pi ^ { \star } ( s _ { t } ) )$ and $\displaystyle T ( \cdot \mid s _ { t } , \pi ^ { \star } ( s _ { t } ) )$ differs by only $\varepsilon _ { h }$ (in TV distance as required by the strongly open-loop consistency assumption) where precisely $\varepsilon _ { h }$ probability mass is moved from reaching state $Z$ to reaching state G. This causes the $\hat { V } _ { \mathrm { a c } } ^ { \star }$ to precisely underestimates the value of $V _ { \mathrm { a c } } ^ { \star }$ by (1−γ)(1−(1−2εh)γh) . It is worth noting that we cannot make the 2εh in $\frac { \varepsilon _ { h } \gamma ^ { h } } { ( 1 - \gamma ) ( 1 - ( 1 - 2 \varepsilon _ { h } ) \gamma ^ { h } ) }$ εh γh $2 \varepsilon _ { h }$ the denominator $\varepsilon _ { h }$ because $\bar { V } _ { \mathrm { a c } } ^ { \star }$ needs to simultaneously maintain a value gap with $V ^ { \star }$ . If we were to construct an example where $\hat { V } _ { \mathrm { a c } } ^ { \star } ( X _ { 0 } ) - V _ { \mathrm { a c } } ^ { \star } ( X _ { 0 } ) = V _ { \mathrm { a c } } ^ { \star }$ be $\frac { \varepsilon _ { h } \gamma ^ { \prime \prime } } { ( 1 - \gamma ) ( 1 - ( 1 - \varepsilon _ { h } ) \gamma ^ { h } ) } .$ εh γh it would enforce $V _ { \mathrm { a c } } ^ { \star } ( X _ { 0 } ) = V ^ { \star } ( X _ { 0 } )$ because there would be no probability mass left to create the gap between $V ^ { \star }$ and $V _ { \mathrm { a c } } ^ { \star }$ . With an extra $\varepsilon _ { h }$ in the denominator, we can also make the optimality gap of $V _ { \mathrm { a c } } ^ { \star }$ precisel y (1−γ)(1−(1−2εh)γh) , bringing the combined value gap $\frac { \varepsilon _ { h } \gamma ^ { h } } { ( 1 - \gamma ) ( 1 - ( 1 - 2 \varepsilon _ { h } ) \gamma ^ { h } ) }$ εh γh (between V ⋆ and Vˆ ⋆ac) up to (1−γ)(1−(1−2εh)γh) . $V ^ { \star }$ $\hat { V } _ { \mathrm { a c } } ^ { \star } )$ $\frac { 2 \varepsilon _ { h } \gamma ^ { n } } { ( 1 - \gamma ) ( 1 - ( 1 - 2 \varepsilon _ { h } ) \gamma ^ { h } ) }$ 2εhγh

2. $P _ { D } ( \cdot \mid s _ { t } , \pi ^ { + } ( s _ { t } ) )$ and $T ( \cdot \mid s _ { t } , \pi ^ { + } ( s _ { t } ) )$ differs by only $\varepsilon _ { h }$ (again in TV distance as required by the strongly open-loop consistency assumption) where precisely $\varepsilon _ { h }$ probability mass is moved from reaching state $G$ to reaching state $Z$ . This causes the $\hat { V } _ { \mathrm { a c } } ^ { + }$ to precisely overestimates the value of V +ac by (1−γ)(1−(1−εh)γh) . $V _ { \mathrm { a c } } ^ { + }$ $\frac { \varepsilon _ { h } \gamma ^ { h } } { ( 1 - \gamma ) ( 1 - ( 1 - \varepsilon _ { h } ) \gamma ^ { h } ) }$ εh γh

We use a special action $a = 2$ where upon taking the action the agent immediately transitions to $Z$ and receives a reward of 0 (except in $G )$ . As we will see soon, this action is useful for constructing a data distribution with an easily ‘controllable’ probability of reaching $Z$ for the top branch and an easily ‘controllable’ probability of reaching G for the bottom branch. Before we start constructing D, we first check the condition that $\mathcal { D } ^ { \star }$ is strongly εh-open-loop consistent.

Step 2. Strong $\varepsilon _ { h } { - } \circ _ { \mathrm { P } } { \in } \mathrm { n } { - } 1$ oop consistency of $\mathcal { D } ^ { \star }$ : It is clear that one possible $\pi ^ { \star }$ that achieves $1 / ( 1 - \gamma )$ value is

$$
\pi^ {\star} (X _ {i}) = 0
$$

$$
\pi^ {\star} (C) = 1 \tag {135}
$$

$$
\pi^ {\star} (\tilde {X}) = 0
$$

We can easily check that $\mathcal { D } ^ { \star }$ collected by $\pi ^ { \star }$ is strongly $\varepsilon _ { h }$ -open-loop consistent by observing that the only path that $\pi ^ { \star }$ outputs $( 0 , 1 , 0 , 0 , \cdots )$ has $\varepsilon _ { h }$ probability, which causes the state distribution of $s _ { t + 1 }$ to differ by at most $\varepsilon _ { h }$ under the TV distance (subject to $a = ( 0 , 1 , 0 , 0 , \cdots )$ or $a = ( 0 , 0 , 0 , \cdots )$ conditioning). This concludes that ${ \mathcal { D } } ^ { \star }$ generated by $\pi ^ { \star }$ above is strongly εh-open-loop consistent.

Now, depending on the first action $a _ { t } .$ , the MDP can be decomposed into two parts: the top $( a = 0 )$ and the bottom $( a = 1 )$ . We construct the data distribution for each branch and analyze the actual and nominal optimality gap for each branch in the following steps.

Step 3. Data distribution $\mathcal { D } _ { \mathrm { t o p } }$ for the top branch: we use a mixture of the following two policies to construct a strongly εh-open-loop consistent $\mathcal { D } _ { \mathrm { t o p } }$ .

Policy π1top: $\pi _ { \mathrm { t o p } } ^ { 1 }$

$$
\pi_ {\text { top }} ^ {1} \left(X _ {0}\right) = \pi_ {\text { top }} ^ {1} (C) = \pi_ {\text { top }} ^ {1} (Z) = 0, \tag {136}
$$

$$
\pi_ {\mathrm{top}} ^ {1} (X _ {1}) = \pi_ {\mathrm{top}} ^ {1} (\tilde {X} _ {1}) = 2.
$$

$\pi _ { \mathrm { t o p } } ^ { 1 }$ always take $a = 2$ unless it is in state $X _ { 0 } , C$ or $Z$ where it always takes $a = 0$ . It is clear that this policy only produces two possible action chunks: $a _ { t : t + h } = ( 0 , 0 , \cdots , 0 )$ or $a _ { t : t + h } = a _ { t : t + h } ^ { \bullet } : =$ $( 0 , 2 , 0 , \cdots )$ . We note that the $\scriptstyle a _ { t : t + h }$ policy always leads to state Z:

$$
P _ {\mathcal {D} _ {\pi_ {\text { top }} ^ {1}}} (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} = 0) = 1. \tag {137}
$$

Policy π2top: $\pi _ { \mathrm { t o p } } ^ { 2 }$

$$
\pi_ {\mathrm{top}} ^ {2} (X _ {0}) = 0,
$$

$$
\pi_ {\text {top}} ^ {2} \left(\tilde {X} _ {1}\right) = \pi_ {\text {top}} ^ {2} (\tilde {C}) = 1, \tag {138}
$$

$$
\pi_ {\mathrm{top}} ^ {2} (a = 0 \mid X _ {1}) = 1 - \delta_ {G},
$$

$$
\pi_ {\mathrm{top}} ^ {2} (a = 1 \mid X _ {1}) = \delta_ {G},
$$

with some $\delta _ { G } \in ( 0 , 1 ) . \pi _ { \mathrm { t o p } } ^ { 2 }$ can also only produce two possible action chunks: $a _ { t : t + h } = ( 0 , 0 , \cdots , 0 )$ or $a _ { t : t + h } = a _ { t : t + h } ^ { \diamond } : = ( 0 , \bar { 1 } , 0 , \cdot \cdot \cdot , 0 )$ .

The distribution of $s _ { t + h }$ conditioned on $a _ { t : t + h } = 0$ is

$$
P _ {\mathcal {D} _ {\pi_ {\mathrm{top}} ^ {2}}} (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} = 0) = 0,
$$

$$
P _ {\mathcal {D} _ {\pi_ {\text { top }} ^ {2}}} \left(s _ {t + h} = G \mid s _ {t}, a _ {t: t + h} = 0\right) = 0, \tag {139}
$$

$$
P _ {\mathcal {D} _ {\pi_ {\mathrm{top}} ^ {2}}} (s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} = 0) = 1.
$$

Mixing $\pi _ { \mathrm { t o p } } ^ { 1 }$ and $\pi _ { \mathrm { t o p } } ^ { 2 } .$ : Let $\mathcal { D } _ { \mathrm { t o p } }$ be a mixture of $\mathcal { D } _ { \pi _ { \mathrm { t o p } } ^ { 1 } }$ 1t and $\mathcal { D } _ { \pi _ { \mathrm { t o p } } ^ { 2 } }$

$$
P _ {\mathcal {D} _ {\text { top }}} = (1 - \varsigma) P _ {\mathcal {D} _ {\text { top }} ^ {1}} + \varsigma P _ {\mathcal {D} _ {\text { top }} ^ {2}}, \tag {140}
$$

where

$$
\varsigma = \frac {1}{2 (1 - \delta_ {G}) + 1}. \tag {141}
$$

It is clear that $0 < \varsigma < 1$ (because $\delta _ { G } \in ( 0 , 1 ) )$ , making it valid mixing ratio.

We now compute the marginal state distribution of the mixture by first analyzing the action probability:

$$
\begin{array}{l} P _ {\mathcal {D} _ {\text {top}} ^ {1}} \left(a _ {t: t + h} ^ {\star} \mid s _ {t}\right) = \varepsilon_ {h}, \\ P _ {\text {top}} \left(\star_ {t - 1} \mid s _ {t}\right) = (1 - 2) (1 - s _ {t}) \end{array} \tag {142}
$$

$$
P _ {\mathcal {D} _ {\mathrm{top}} ^ {2}} (a _ {t: t + h} ^ {\star} \mid s _ {t}) = (1 - 2 \varepsilon_ {h}) (1 - \delta_ {G}).
$$

The state marginals are then

$$
\begin{array}{l} P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\star}) = \frac {P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + h} = Z , a _ {t : t + h} ^ {\star} \mid s _ {t})}{P _ {\mathcal {D} _ {\mathrm{top}}} (a _ {t : t + h} ^ {\star} \mid s _ {t})} \\ = \frac {(1 - \varsigma) P _ {\mathcal {D} _ {\text {top}} ^ {1}} (a _ {t : t + h} ^ {\star} \mid s _ {t})}{(1 - \varsigma) P _ {\mathcal {D} _ {\text {top}} ^ {1}} (a _ {t : t + h} ^ {\star} \mid s _ {t}) + \varsigma P _ {\mathcal {D} _ {\text {top}} ^ {2}} (a _ {t : t + h} ^ {\star} \mid s _ {t})} \tag {143} \\ = \frac {\varepsilon_ {h} (1 - \varsigma)}{\varepsilon_ {h} (1 - \varsigma) + (1 - 2 \varepsilon_ {h}) (1 - \delta_ {G}) \varsigma} \\ = 2 \varepsilon_ {h}. \\ \end{array}
$$

Therefore,

$$
P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\star}) = 2 \varepsilon_ {h},
$$

$$
P _ {\mathcal {D} _ {\text { top }}} (s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} ^ {\star}) = 1 - 2 \varepsilon_ {h}, \tag {144}
$$

$$
P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + h} = G \mid s _ {t}, a _ {t: t + h} ^ {\star}) = 0.
$$

Step 4. Strong $\varepsilon _ { h } { - } \circ _ { \mathrm { P } } { \mathrm { e n - 1 } }$ oop consistency of $\mathcal { D } _ { \mathrm { t o p } }$ : Now, we check for strong openloop consistency for the three possible action chunks on the top branch:

$$
\begin{array}{l} a _ {t: t + h} ^ {\star} = (0, 0, 0, \dots) \\ a _ {t: t + h} ^ {\diamond} = (0, 1, 0, \dots) \tag {145} \\ \end{array}
$$

$$
a _ {t: t + h} ^ {\bullet} = (0, 2, 0, \dots)
$$

For $a _ { t : t + h } ^ { \star } = 0$ , we can compute open-loop marginal state distribution as follows:

$$
T (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\star}) = \varepsilon_ {h},
$$

$$
T (s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} ^ {\star}) = 1 - 2 \varepsilon_ {h}, \tag {146}
$$

$$
T (s _ {t + h} = G \mid s _ {t}, a _ {t: t + h} ^ {\star}) = \varepsilon_ {h}.
$$

Combining this with the data distribution calculated in Equation (144), it is clear that

$$
D _ {\mathrm{TV}} \left(T (s _ {t + h} \mid s _ {t}, a _ {t: t + h} = 0) \parallel P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + h} \mid s _ {t}, a _ {t: t + h} = 0)\right) = \varepsilon_ {h}. \tag {147}
$$

We can repeat the same procedure to show that

$$
D _ {\mathrm{TV}} \left(T \left(s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h ^ {\prime}} = 0\right) \| P _ {\mathcal {D} _ {\text {top}}} \left(s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h} = 0\right)\right) = \varepsilon_ {h}, \quad \forall h ^ {\prime} \in \{1, \dots , h - 1 \} \tag {148}
$$

because the only difference in these distributions is that they occupy $s _ { t + h ^ { \prime } } = X _ { h ^ { \prime } }$ ′ with $2 \varepsilon _ { h }$ probability instead of $s _ { t + h } = X _ { 0 }$ with $2 \varepsilon _ { h }$ probability.

For $a _ { t : t + h } ^ { \bullet } = ( 0 , 2 , 0 , \cdots )$ , it is clear that

$$
D _ {\mathrm{TV}} \left(T \left(s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h} = a _ {t: t + h} ^ {\bullet}\right) \| P _ {\mathcal {D} _ {\text { top }}} \left(s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h} = a _ {t: t + h} ^ {\bullet}\right)\right) = \varepsilon_ {h} \tag {149}
$$

holds for any $h ^ { \prime } \in \{ 1 , 2 , \cdots , h \}$ since the only difference between these two distributions is the $\varepsilon _ { h } \cdot$ probability path $( i . e . , X _ { 0 }  C  Z$ where the probability is under $T ( \cdot \mid s _ { t } , a _ { t : t + h } ^ { \bullet } ) )$ .

For $a _ { t : t + h } ^ { \diamond } = ( 0 , 1 , 0 , \cdot \cdot \cdot )$ , we first compute the marginal state distributions:

$$
P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = \frac {(1 - 2 \varepsilon_ {h}) \delta_ {G}}{2 \varepsilon_ {h} + (1 - 2 \varepsilon_ {h}) \delta_ {G}},
$$

$$
P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = \frac {2 \varepsilon_ {h}}{2 \varepsilon_ {h} + (1 - 2 \varepsilon_ {h}) \delta_ {G}},
$$

$$
P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + h} = G \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = 0.
$$

$$
P _ {\mathcal {D} _ {\text { top }}} (s _ {t + 1} = X _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = \frac {(1 - 2 \varepsilon_ {h}) \delta_ {G}}{2 \varepsilon_ {h} + (1 - 2 \varepsilon_ {h}) \delta_ {G}}. \tag {150}
$$

$$
P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + 1} = \tilde {X} _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = \frac {\varepsilon_ {h}}{2 \varepsilon_ {h} + (1 - 2 \varepsilon_ {h}) \delta_ {G}}.
$$

$$
P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + 1} = C \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = \frac {\varepsilon_ {h}}{2 \varepsilon_ {h} + (1 - 2 \varepsilon_ {h}) \delta_ {G}}.
$$

We can also compute the open-loop marginal state distribution as follows:

$$
T (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = 1 - 2 \varepsilon_ {h}
$$

$$
T (s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = 2 \varepsilon_ {h}
$$

$$
T (s _ {t + h} = G \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = 0.
$$

$$
\begin{array}{l} T \left(s _ {t + h} - a ^ {\diamond} \mid s _ {t}, a _ {t: t + h}\right) = 0. \\ T \left(s _ {t + 1} = X _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\diamond}\right) = 1 - 2 \varepsilon_ {h}. \end{array} \tag {151}
$$

$$
T (s _ {t + 1} = \tilde {X} _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = \varepsilon_ {h}.
$$

$$
T (s _ {t + 1} = C \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = \varepsilon_ {h}.
$$

Let $c _ { 3 }$ be any value that satisfies $c _ { 3 } \in ( 0 , \varepsilon _ { h } / 2 )$ , we can set

$$
\delta_ {G} = \frac {\varepsilon_ {h} (1 - 2 \varepsilon_ {h} - 2 c _ {3})}{(\varepsilon_ {h} + c _ {3}) (1 - 2 \varepsilon_ {h})}, \tag {152}
$$

such that

$$
P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = 1 - 2 \varepsilon_ {h} - 2 c _ {3},
$$

$$
P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = 2 \varepsilon_ {h} + 2 c _ {3},
$$

$$
\begin{array}{l} P _ {\mathcal {D} _ {\text {top}}} \left(s _ {t + h} = G \mid s _ {t}, a _ {t: t + h} ^ {\diamond}\right) = 0, \\ P _ {\text {top}} \left(s _ {t + h} = Y _ {t} \mid s _ {t}, a _ {t: t + h} ^ {\diamond}\right) = 1, 2, \dots , 2. \end{array} \tag {153}
$$

$$
P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + 1} = X _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = 1 - 2 \varepsilon_ {h} - 2 c _ {3},
$$

$$
P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + 1} = \tilde {X} _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = \varepsilon_ {h} + c _ {3},
$$

$$
P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + 1} = C \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = \varepsilon_ {h} + c _ {3}.
$$

It is easy to check that $0 < \delta _ { G } < 1$ (a valid probability) because in Equation (152), each term in the numerator has a larger term in the denominator $( i . e . , \varepsilon _ { h } < \varepsilon _ { h } + c _ { 3 }$ and $1 - 2 \varepsilon _ { h } - 2 c _ { 3 } < 1 - 2 \varepsilon _ { h } )$ .

Now, for all $h ^ { \prime } \in \{ 1 , 2 , \cdots , h \}$ , using the values calculated in Equations (151) and (153), we have

$$
D _ {\mathrm{TV}} \left(T (s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h ^ {\prime}} = a _ {t: t + h ^ {\prime}} ^ {\diamond}) \right\| P _ {\mathcal {D} _ {\mathrm{top}}} (s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h} = a _ {t: t + h} ^ {\diamond}) \big) = 2 c _ {3}. \tag {154}
$$

Since $c _ { 3 } < \varepsilon _ { h } / 2 .$ , the strong open-loop consistency assumption holds for $a _ { t : t + h } ^ { \diamond }$ as well.

Step 5. The optimality gap and value estimation error for the top branch:

Now we can compute the optimality gap for the estimated value for $a _ { t : t + h } ^ { \diamond } \colon$

$$
V ^ {\star} (X _ {0}) - \hat {V} _ {\mathrm{ac}} ^ {\diamond} (X _ {0}) = \frac {(1 - 2 \varepsilon_ {h} - 2 c _ {3}) \gamma}{(1 - \gamma) (1 - 2 (\varepsilon_ {h} + c _ {3}) \gamma^ {h})}, \tag {155}
$$

where the h-step reward sub-optimality gap is due to the agent reaching Z with $\left( 1 - 2 \varepsilon _ { h } - 2 c _ { 3 } \right)$ probability, and the h-step distribution gap is reflected in the $( 1 - 2 ( \varepsilon _ { h } ^ { - } + c _ { 3 } ) \gamma ^ { h } )$ term at bottom because the probability of reaching $X _ { 0 }$ after h steps is $2 ( \varepsilon _ { h } + c _ { 3 } )$ .

Similarly, we can compute the optimality gap for $V _ { \mathrm { a c } } ^ { \star }$ and $\hat { V } _ { \mathrm { a c } } ^ { \star }$

$$
\begin{array}{l} V ^ {\star} \left(X _ {0}\right) - V _ {\mathrm{ac}} ^ {\star} \left(X _ {0}\right) = \varepsilon_ {h} \frac {\gamma - \gamma^ {h}}{1 - \gamma} + \frac {\varepsilon_ {h} \gamma^ {h}}{1 - \gamma} + \gamma^ {h} \left(1 - 2 \varepsilon_ {h}\right) \left(V ^ {\star} - \hat {V} _ {\mathrm{ac}}\right) \tag {156} \\ = \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})}, \\ \end{array}
$$

$$
V ^ {\star} \left(X _ {0}\right) - \hat {V} _ {\mathrm{ac}} ^ {\star} \left(X _ {0}\right) = \frac {2 \varepsilon_ {h} \left(\gamma - \gamma^ {h}\right)}{1 - \gamma} + \frac {2 \varepsilon_ {h} \gamma^ {h}}{1 - \gamma} \gamma^ {h} \left(1 - 2 \varepsilon_ {h}\right) \left(V ^ {\star} - \hat {V} _ {\mathrm{ac}}\right) \tag {157}
$$

$$
= \frac {2 \varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})}.
$$

Now, we observe that

$$
1 - 2 \varepsilon_ {h} - 2 c _ {3} > 1 - 3 \varepsilon_ {h} > 2 \varepsilon_ {h}, \tag {158}
$$

where the first inequality is due to $c _ { 3 } \in ( 0 , \varepsilon _ { h } / 2 )$ and the second inequality is due to $\varepsilon _ { h } \in ( 0 , 1 / 5 )$ in our assumption. This allows us to lower-bound the estimated optimality gap for $a _ { t : t + h } ^ { \diamond }$ as follows:

$$
\begin{array}{l} V ^ {\star} (X _ {0}) - \hat {V} _ {\mathrm{ac}} ^ {\diamond} (X _ {0}) = \frac {(1 - 2 \varepsilon_ {h} - 2 c _ {3}) \gamma}{(1 - \gamma) (1 - 2 (\varepsilon_ {h} + c _ {3}) \gamma^ {h})} \\ > \frac {2 \varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})} \tag {159} \\ = V ^ {\star} (X _ {0}) - \hat {V} _ {\mathrm{ac}} ^ {\star} (X _ {0}), \\ \end{array}
$$

where the inequality is obtained by triggering Lemma $5 ( e . g .$ , by setting $\varepsilon _ { 1 } = 2 \varepsilon _ { h } , \varepsilon _ { 2 } = ( 1 - 2 \varepsilon _ { h } -$ $2 c _ { 3 } ) , \tilde { \gamma } = \gamma ^ { h } )$ . The bound above rules out the possibility of $a _ { t : t + h } ^ { \flat }$ 1 h 2  being picked by $\hat { \pi } _ { \mathrm { a c } } ^ { + }$ h because it has a lower estimated value compared to a⋆t:t+h. $a _ { t : t + h } ^ { \star }$

Finally, for $a _ { t : t + h } ^ { \bullet }$ , since it is correlated with $s _ { t + h } = Z$ and receives no reward except the first step in $\mathcal { D } _ { \mathrm { t o p } } .$ the estimated value is just 1, being trivially smaller than $\hat { V } _ { \mathrm { a c } } ^ { \star } ( X _ { 0 } )$ and would never get picked by $\hat { \pi } _ { \mathrm { a c } } ^ { + }$ . Up to now, we have finished our data distribution construction and analysis for the top branch. We summarize the key intermediate results as the remark below:

Remark 1 (Intermediate results from Step 1-4) The optimal action chunk is $a _ { t : t + h } ^ { \star }$ and the estimated values for the two other possible action chunks $a _ { t : t + h } ^ { \bullet } , a _ { t : t + h } ^ { \circ }$ are smaller than that of a t:t+h: $a _ { t : t + h } ^ { \star } \colon $

$$
\hat {V} _ {\mathrm{ac}} ^ {\bullet} (X _ {0}) <   \hat {V} _ {\mathrm{ac}} ^ {\diamond} (X _ {0}) <   \hat {V} _ {\mathrm{ac}} ^ {\star} (X _ {0}) = V ^ {\star} (X _ {0}) - \frac {2 \varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})}. \tag {160}
$$

In addition, both $\mathcal { D } _ { \mathrm { t o p } }$ and ${ \mathcal { D } } ^ { \star }$ are strongly εh-open-loop consistent.

Next, we move on to the bottom branch.

Step 6. Data distribution $\mathcal { D } _ { \mathrm { b o t t o m } }$ for the bottom branch: For the bottom, we again use two policies.

Policy π1bottom:

$$
\pi_ {\text { bottom }} ^ {1} \left(X _ {0}\right) = \pi_ {\text { bottom }} ^ {1} (G) = \pi_ {\text { bottom }} ^ {1} (Z) = 1, \tag {161}
$$

$$
\pi_ {\mathrm{bottom}} ^ {1} (Y _ {1}) = \pi_ {\mathrm{bottom}} ^ {1} (\tilde {Y} _ {1}) = 2.
$$

$\pi _ { \mathrm { { b o t t o m } } } ^ { 1 }$ takes $a = 1$ at $X _ { 0 }$ and G and $Z ,$ and takes $a = 2$ otherwise (at $Y _ { 1 } , { \tilde { Y } } _ { 1 } )$ . It is clear that this policy only produces two possible action chunks: $a _ { t : t + h } ^ { \triangle } = ( 1 , 1 , 1 , \cdot \cdot \cdot ) \mathrm { o r } a _ { t : t + h } ^ { \times } = ( 1 , 2 , 1 , \cdot \cdot \cdot )$ .

Policy π2bottom: $P o l i c y \pi _ { \mathrm { b o t t o m } } ^ { 2 } \colon$

$$
\pi_ {\mathrm{bottom}} ^ {2} (X _ {0}) = 1,
$$

$$
\pi_ {\mathrm{bottom}} ^ {2} (a = 0 \mid Y _ {1}) = \delta_ {Z},
$$

$$
\pi_ {\text { bottom }} ^ {2} (a = 1 \mid Y _ {1}) = 1 - \delta_ {Z}, \tag {162}
$$

$$
\pi_ {\mathrm{bottom}} ^ {2} (\tilde {Y} _ {1}) = 0,
$$

$$
\pi_ {\mathrm{bottom}} ^ {2} (Y _ {i}) = 1, \quad \forall i \in \{2, \dots , h - 1 \},
$$

where $\delta _ { Z } \in ( 0 , 1 )$ and we shall specify the exact value of $\delta _ { Z }$ shortly.

π2bottom $\pi _ { \mathrm { b o t t o m } } ^ { 2 }$ takes $a = 1$ when it is at $Y _ { i }$ and takes either $a = 0$ (with $\delta _ { Z }$ probability) or $a = 1$ (with $1 - \delta _ { Z }$ probability) when it is at $\tilde { Y } _ { 1 }$ . It is clear that this policy only produces two possible action chunks: $a _ { t : t + h } ^ { \triangle } = ( 1 , 1 , 1 , \cdot \cdot \cdot )$ or $a _ { t : t + h } ^ { \circ } = ( 1 , 0 , 1 , \cdots )$ .

Now, we observe that the marginal state distributions for both policies conditioned on $a _ { t : t + h } ^ { \triangle }$ are independent of $c _ { 1 }$ and $\delta _ { Z }$ because the action chunk only appears when $\pi _ { \mathrm { b o t t o m } } ^ { 1 }$ reaches $G$ and when π2bottom $\pi _ { \mathrm { b o t t o m } } ^ { 2 }$ reaches $X _ { 0 }$ . More specifically,

$$
P _ {\mathcal {D} _ {\text { bottom }} ^ {1}} (s _ {t + 1} = G \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = P _ {\mathcal {D} _ {\text { bottom }} ^ {1}} (s _ {t + h} = G \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = 1, \tag {163}
$$

$$
P _ {\mathcal {D} _ {\text { bottom }} ^ {2}} \left(s _ {t + i} = X _ {i} \mid s _ {t}, a _ {t: t + h} ^ {\triangle}\right) = P _ {\mathcal {D} _ {\text { bottom }} ^ {2}} \left(s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} ^ {\triangle}\right) = 1, \forall i \in \{1, \dots , h - 1 \}. \tag {164}
$$

We can now mix D1botto $\mathcal { D } _ { \mathrm { b o t t o m } } ^ { 1 }$ m and D2bottom $\mathcal { D } _ { \mathrm { b o t t o m } } ^ { 2 }$ with an appropriate ratio to control the state marginals for $s _ { t : t + h } = G$ and $s _ { t : t + h } = X _ { 0 }$ arbitrarily $( s _ { t : t + h } = Z$ stays at 0 because none of the policies take/have taken a△t:t+h $a _ { t : t + h } ^ { \triangle }$ when they reach $Z )$ .

Mixing $\pi _ { \mathrm { b o t t o m } } ^ { 1 } { a n d \pi _ { \mathrm { b o t t o m } } ^ { 2 } }$ : Let $\mathcal { D } _ { \mathrm { b o t t o m } }$ be a mixture of $\mathcal { D } _ { \mathrm { b o t t o m } } ^ { 1 }$ and $ { \mathcal { D } _ { \mathrm { { b o t t o m } } } } ^ { 2 }$

$$
P _ {\mathcal {D} _ {\text { bottom }}} = (1 - \vartheta) P _ {\mathcal {D} _ {\text { bottom }} ^ {1}} + \vartheta P _ {\mathcal {D} _ {\text { bottom }} ^ {2}}, \tag {165}
$$

where we set the mixing ratio to be

$$
\vartheta = \frac {c _ {1}}{c _ {1} + (1 - \delta_ {Z}) (\varepsilon_ {h} + c _ {1})}. \tag {166}
$$

This mixing ratio helps the calculations to be simpler later on.

We can now compute the marginal state distribution of the mixture. We start by analyzing the action probability:

$$
P _ {\mathcal {D} _ {\text {bottom}} ^ {1}} \left(a _ {t: t + h} ^ {\triangle} \mid s _ {t}\right) = c _ {1}, \tag {167}
$$

$$
P _ {\mathcal {D} _ {\mathrm{bottom}} ^ {2}} (a _ {t: t + h} ^ {\triangle} \mid s _ {t}) = (1 - \varepsilon_ {h} - c _ {1}) (1 - \delta_ {Z}).
$$

The state marginal is then

$$
\begin{array}{l} P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = \frac {P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + h} = X _ {0} , a _ {t : t + h} ^ {\triangle} \mid s _ {t})}{P _ {\mathcal {D} _ {\mathrm{bottom}}} (a _ {t : t + h} ^ {\triangle} \mid s _ {t})} \\ = \frac {\vartheta P _ {\mathcal {D} _ {\text {bottom}} ^ {2}} (a _ {t : t + h} ^ {\triangle} \mid s _ {t})}{(1 - \vartheta) P _ {\mathcal {D} _ {\text {bottom}} ^ {1}} (a _ {t : t + h} ^ {\triangle} \mid s _ {t}) + \vartheta P _ {\mathcal {D} _ {\text {bottom}} ^ {2}} (a _ {t : t + h} ^ {\triangle} \mid s _ {t})} \\ = \frac {(1 - \varepsilon_ {h} - c _ {1}) (1 - \delta_ {Z}) \vartheta}{c _ {1} (1 - \vartheta) + (1 - \varepsilon_ {h} - c _ {1}) (1 - \delta_ {Z}) \vartheta} \\ = 1 - \varepsilon_ {h} - c _ {1}. \tag {168} \\ \end{array}
$$

We can use it to deduce the rest of the marginals as follows:

$$
P _ {\mathcal {D} _ {\text { bottom }}} (s _ {t + h} = G \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = \varepsilon_ {h} + c _ {1}, \quad \forall h ^ {\prime} \in \{1, \dots , h - 1 \},
$$

$$
P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = 1 - \varepsilon_ {h} - c _ {1},
$$

$$
P _ {\mathcal {D} _ {\text { bottom }}} (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = 0, \tag {169}
$$

$$
P _ {\mathcal {D} _ {\text {bottom}}} (s _ {t + h ^ {\prime}} = Y _ {h ^ {\prime}} \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = 1 - \varepsilon_ {h} - c _ {1}, \quad \forall h ^ {\prime} \in \{1, \dots , h - 2 \},
$$

$$
P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + 1} = \tilde {Y} _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = 0.
$$

Up to now, we have established $\mathcal { D } _ { \mathrm { b o t t o m } }$ and we are ready to check the strong open-loop consistency.

Step 7. Strong $\varepsilon _ { h } { - } \circ _ { \mathrm { P } } { \in } \mathrm { n } { - } 1$ oop consistency of $\mathcal { D } _ { \mathrm { b o t t o m } }$ :

$a _ { t : t + h } ^ { \triangle } = ( 1 , 1 , \cdot \cdot \cdot )$

$$
T (s _ {t + h ^ {\prime}} = G \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = c _ {1}, \quad \forall h ^ {\prime} \in \{1, \dots , h - 1 \},
$$

$$
T (s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = 1 - \varepsilon_ {h} - c _ {1},
$$

$$
T (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = \varepsilon_ {h}. \tag {170}
$$

$$
T (s _ {t + h ^ {\prime}} = Y _ {h ^ {\prime}} \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = 1 - \varepsilon_ {h} - c _ {1}, \quad \forall h ^ {\prime} \in \{1, \dots , h - 2 \}
$$

$$
T (s _ {t + 1} = \tilde {Y} _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\triangle}) = \varepsilon_ {h}.
$$

Combining it with the marginals calculated in Equation (169), it is clear that for all $h ^ { \prime } \in \{ 1 , \cdots , h -$ 1},

$$
D _ {\mathrm{TV}} \left(T \left(s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h ^ {\prime}} = a _ {t: t + h ^ {\prime}} ^ {+}\right) \| P _ {\mathcal {D} _ {\text { bottom }}} \left(s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h} = a _ {t: t + h} ^ {+}\right)\right) = \varepsilon_ {h}, \tag {171}
$$

satisfying the open-loop consistency.

For $a _ { t : t + h } ^ { \times } = ( 1 , 2 , 1 , \cdot \cdot \cdot )$ , the data and open-loop state marginals are

$$
P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\times}) = 1,
$$

$$
P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + 1} = Y _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\times}) = \frac {1 - \varepsilon_ {h} - c _ {1}}{1 - c _ {1}},
$$

$$
P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + 1} = \tilde {Y} _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\times}) = \frac {\varepsilon_ {h}}{1 - c _ {1}},
$$

$$
T (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\times}) = 1 - c _ {1}, \tag {172}
$$

$$
T (s _ {t + h} = G \mid s _ {t}, a _ {t: t + h} ^ {\times}) = c _ {1},
$$

$$
T (s _ {t + 1} = Y _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\times}) = 1 - \varepsilon_ {h} - c _ {1},
$$

$$
T (s _ {t + 1} = \tilde {Y} _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\times}) = \varepsilon_ {h},
$$

$$
T (s _ {t + 1} = G \mid s _ {t}, a _ {t: t + h} ^ {\times}) = c _ {1}.
$$

This allows us to bound the TV distance for all $h ^ { \prime } \in \{ 1 , \cdots , h - 1 \}$ as

$$
D _ {\mathrm{TV}} \left(T \left(s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h ^ {\prime}} = a _ {t: t + h ^ {\prime}} ^ {\times}\right) \| P _ {\mathcal {D} _ {\text {bottom}}} \left(s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h} = a _ {t: t + h} ^ {\times}\right)\right) \leq \frac {c _ {1}}{1 - c _ {1}}. \tag {173}
$$

Since $c _ { 1 } < \varepsilon _ { h } / 2 < 1 / 1 0 .$

$$
\frac {c _ {1}}{1 - c _ {1}} <   \frac {1 0}{9} c _ {1} <   5 \varepsilon_ {h} / 9 <   \varepsilon_ {h}, \tag {174}
$$

satisfying the strong open-loop consistency assumption.

For $a _ { t : t + h } ^ { \circ } = ( 1 , 0 , 1 , \cdots )$ , we first compute the state marginals in $\mathcal { D } _ { \mathrm { b o t t o m } }$ as follows:

$$
P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\circ}) = \frac {(1 - \varepsilon_ {h} - c _ {1}) \delta_ {Z}}{\varepsilon_ {h} + (1 - \varepsilon_ {h} - c _ {1}) \delta_ {Z}},
$$

$$
P _ {\mathcal {D} _ {\text {bottom}}} \left(s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} ^ {\circ}\right) = \frac {\varepsilon_ {h}}{\varepsilon_ {h} + \left(1 - \varepsilon_ {h} - c _ {1}\right) \delta_ {Z}}, \tag {175}
$$

$$
P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + 1} = Y _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\circ}) = \frac {(1 - \varepsilon_ {h} - c _ {1}) \delta_ {Z}}{\varepsilon_ {h} + (1 - \varepsilon_ {h} - c _ {1}) \delta_ {Z}}.
$$

$$
P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + 1} = \tilde {Y} _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\circ}) = \frac {\varepsilon_ {h}}{\varepsilon_ {h} + (1 - \varepsilon_ {h} - c _ {1}) \delta_ {Z}}.
$$

We can also compute the open-loop marginal state distribution as follows:

$$
T (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\circ}) = 1 - \varepsilon_ {h} - c _ {1},
$$

$$
T (s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} ^ {\circ}) = \varepsilon_ {h},
$$

$$
T (s _ {t + h} = G \mid s _ {t}, a _ {t: t + h} ^ {\circ}) = c _ {1},
$$

$$
T \left(s _ {t + 1} = Y _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\circ}\right) = 1 - \varepsilon_ {h} - c _ {1}, \tag {176}
$$

$$
T (s _ {t + 1} = \tilde {Y} _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\circ}) = \varepsilon_ {h},
$$

$$
T (s _ {t + 1} = G \mid s _ {t}, a _ {t: t + h} ^ {\circ}) = c _ {1}.
$$

Let $c _ { 4 } \in ( c _ { 1 } , \varepsilon _ { h } )$ , and we set

$$
\delta_ {Z} = \frac {\varepsilon_ {h} (1 - \varepsilon_ {h} - c _ {4})}{(\varepsilon_ {h} + c _ {4}) (1 - \varepsilon_ {h} - c _ {1})}. \tag {177}
$$

Then, we have

$$
P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + h} = Z \mid s _ {t}, a _ {t: t + h} ^ {\circ}) = 1 - \varepsilon_ {h} - c _ {4},
$$

$$
P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + h} = X _ {0} \mid s _ {t}, a _ {t: t + h} ^ {\circ}) = \varepsilon_ {h} + c _ {4},
$$

$$
P _ {\mathcal {D} _ {\text { bottom }}} (s _ {t + h} = G \mid s _ {t}, a _ {t: t + h} ^ {\circ}) = 0, \tag {178}
$$

$$
P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + 1} = Y _ {1} \mid s _ {t}, a _ {t: t + h} ^ {\circ}) = 1 - \varepsilon_ {h} - c _ {4},
$$

$$
P _ {\mathcal {D} _ {\mathrm{bottom}}} (s _ {t + 1} = \tilde {Y} _ {1} | s _ {t}, a _ {t: t + h} ^ {\circ}) = \varepsilon_ {h} + c _ {4}.
$$

The TV distance is then

$$
D _ {\mathrm{TV}} \left(T \left(s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h ^ {\prime}} = a _ {t: t + h ^ {\prime}} ^ {\circ}\right) \| P _ {\mathcal {D} _ {\text { bottom }}} \left(s _ {t + h ^ {\prime}} \mid s _ {t}, a _ {t: t + h} = a _ {t: t + h} ^ {\circ}\right)\right) = c _ {4}. \tag {179}
$$

Since $c _ { 4 } < \varepsilon _ { h }$ , the strong open-loop consistency is also satisfied for $a _ { t : t + h } ^ { \circ } .$

Up to now, we have checked that all three possible action chunks in the bottom branch satisfy the strong open-loop consistency assumption. Since $\mathcal { D } _ { \mathrm { t o p } }$ and $\mathcal { D } _ { \mathrm { b o t t o m } }$ have non-overlapping supports for $a _ { t : t + h }$ , and they are both strongly $\varepsilon _ { h }$ -open-loop consistent on their own, we can construct $\dot { \mathcal { D } }$ as

$$
P _ {\mathcal {D}} (\cdot \mid s _ {t}) = (1 - \varrho) P _ {\mathcal {D} _ {\text { top }}} (\cdot \mid s _ {t}) + \varrho P _ {\mathcal {D} _ {\text { bottom }}} (\cdot \mid s _ {t}), \tag {180}
$$

for any $\varrho \in ( 0 , 1 )$ , and conclude that

Remark 2 (Intermediate result from Step 5-7) D is strongly $\varepsilon _ { h } .$ -open-loop consistent.

Up to now, we have constructed and checked both D and $\mathcal { D } ^ { \star }$ are strongly $\varepsilon _ { h }$ -open-loop consistent.

As the final step, we calculate the optimality gap and value estimation error for these action chunks.

Step 8. The optimality gap and value estimation error for the bottom branch:

We first note that similar to $a _ { t : t + h } ^ { \bullet } , a _ { t : t + h } ^ { \times }$ is correlated with $s _ { t + h } = Z$ and always receives 0 reward except the first step in D. Thus, the estimated value $\hat { V } ^ { \times }$ is just 1, being trivially smaller than $\hat { V } _ { \mathrm { a c } } ^ { \star }$ and would never get picked by $\hat { \pi } _ { \mathrm { a c } } ^ { \bigtriangleup }$ . The only top contenders are $a _ { t : t + h } ^ { + } , a _ { t : t + h } ^ { \circ }$ and $a _ { t : t + h } ^ { \star }$ (which we already analyzed in Step 5 above).

We start with $a _ { t : t + h } ^ { \circ }$ where we can compute optimality gap as follows:

$$
V ^ {\star} (X _ {0}) - \hat {V} _ {\mathrm{ac}} ^ {\circ} (X _ {0}) = \frac {(1 - \varepsilon_ {h} - c _ {4}) \gamma + \delta (1 - \gamma) + (\varepsilon_ {h} + c _ {4}) \delta (\gamma - \gamma^ {h})}{(1 - \gamma) (1 - (\varepsilon_ {h} + c _ {4}) \gamma^ {h})}. \tag {181}
$$

Now, observe that

$$
\varepsilon_ {h} + c _ {4} <   2 \varepsilon_ {h} <   1 - 2 \varepsilon_ {h}, \tag {182}
$$

where again the last inequality comes from the fact that $\varepsilon _ { h } < 1 / 4$ .

We can now lower-bound the optimality gap as follows:

$$
\begin{array}{l} V ^ {\star} (X _ {0}) - \hat {V} _ {\mathrm{ac}} ^ {\circ} (X _ {0}) > \frac {2 \varepsilon_ {h} \gamma + \delta (1 - \gamma) + (\varepsilon_ {h} + c _ {4}) \delta (\gamma - \gamma^ {h})}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})} \\ > \frac {2 \varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})} \tag {183} \\ = V ^ {\star} (X _ {0}) - \hat {V} _ {\mathrm{ac}} ^ {\star} (X _ {0}). \\ \end{array}
$$

where the first inequality is obtained by triggering Lemma $5 \ ( e . g .$ ., by setting $\varepsilon _ { 1 } = 2 \varepsilon _ { h } , \varepsilon _ { 2 } =$ $( 1 - \varepsilon _ { h } - c _ { 4 } ) , \tilde { \gamma } = \bar { \gamma } ^ { h } )$ .

With this lower-bound, we can conclude that $a _ { t : t + h } ^ { \circ }$ would not be picked by $\pi _ { \mathrm { a c } } ^ { + }$ as well because $\hat { V } _ { \mathrm { a c } } ^ { \circ } ( X _ { 0 } ) < \hat { V } _ { \mathrm { a c } } ^ { \star } ( X _ { 0 } )$ .

Up to now, we have eliminated both $a _ { t : t + h } ^ { \circ }$ and $a _ { t : t + h } ^ { \times }$ (for the possibility of being picked by $\pi _ { \mathrm { a c } } ^ { + } )$ and the only remaining contender left is a△t:t+h. $a _ { t : t + h } ^ { \triangle } .$

We can also compute the estimated and the actual values for $a _ { t : t + h } = a _ { t : t + h } ^ { \triangle } = 1$ in terms of their optimality gaps:

$$
V ^ {\star} (X _ {0}) - \hat {V} _ {\mathrm{ac}} ^ {\triangle} (X _ {0}) = \frac {\delta (1 - \varepsilon_ {h} - c _ {1}) \gamma}{(1 - \gamma) (1 - (1 - \varepsilon_ {h} - c _ {1}) \gamma^ {h})}, \tag {184}
$$

$$
V ^ {\star} (X _ {0}) - V _ {\mathrm{ac}} ^ {\triangle} (X _ {0}) = \frac {[ \delta (1 - \varepsilon_ {h} - c _ {1}) + \varepsilon_ {h} ] \gamma}{(1 - \gamma) (1 - (1 - \varepsilon_ {h} - c _ {1}) \gamma^ {h})}. \tag {185}
$$

Let

$$
\delta = \frac {2 \varepsilon_ {h} \gamma - c _ {2}}{(1 - \varepsilon_ {h} - c _ {1}) \gamma} \frac {1 - (1 - \varepsilon_ {h} - c _ {1}) \gamma^ {h}}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}}. \tag {186}
$$

We first check $1 - \delta$ is a valid reward value (within $[ 0 , 1 ] )$ :

$$
\begin{array}{l} \delta <   \frac {2 \varepsilon_ {h}}{1 - \varepsilon_ {h} - c _ {1}} \frac {1 - (1 - \varepsilon_ {h} - c _ {1}) \gamma^ {h}}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} \\ <   \frac {2 \varepsilon_ {h}}{1 - 2 \varepsilon_ {h}} \frac {1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} \tag {187} \\ = \frac {2 \varepsilon_ {h}}{1 - 2 \varepsilon_ {h}} \\ \leq 1, \\ \end{array}
$$

where the first inequality is because $c _ { 2 } > 0$ , the second inequality is due to $c _ { 1 } < \varepsilon _ { h }$ , and the final inequality is due to $\varepsilon _ { h } < 1 / 4$ .

It is also clear that $\delta > 0$ because all terms are positive in the fraction (Equation (186)).

Next, we substitute δ in to obtain

$$
V ^ {\star} (X _ {0}) - \hat {V} _ {\mathrm{ac}} ^ {\triangle} (X _ {0}) = \frac {2 \varepsilon_ {h} \gamma - c _ {2}}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})}, \tag {188}
$$

$$
V ^ {\star} (X _ {0}) - V _ {\mathrm{ac}} ^ {\triangle} (X _ {0}) = \frac {2 \varepsilon_ {h} \gamma - c _ {2}}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})} + \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - \varepsilon_ {h} - c _ {1}) \gamma^ {h})}, \tag {189}
$$

where intuitively the second term in $V ^ { \star } ( X _ { 0 } ) - V _ { \mathrm { a c } } ^ { \triangle } ( X _ { 0 } )$ is due to the fact that from $P _ { \mathcal { D } } ( \cdot \mid s _ { t } , a _ { t : t + h } ^ { \triangle } )$ to $T ( \cdot \mid s _ { t } , a _ { t : t + h } ^ { \triangle } )$ , there is a shift in $\varepsilon _ { h }$ probability mass from $\boldsymbol { s } _ { t : t + h } = ( \boldsymbol { X } _ { 0 } , \boldsymbol { G } , \cdot \cdot \cdot )$ to $s _ { t : t + h } =$ $( X _ { 0 } , \tilde { Y } _ { 1 } , Z , \cdots )$ incurring an additional $\frac { \varepsilon _ { h } \gamma } { 1 - \gamma }$ sub-optimality in terms of the h-step reward, and then amplified by the value recursion by an additional factor of $\frac { 1 } { 1 - ( 1 - \varepsilon _ { h } - c _ { 1 } ) \gamma ^ { h } }$ (where $1 - \varepsilon _ { h } - c _ { 1 }$ is the probability that $a _ { t : t + h } ^ { \triangle }$ reaches $X _ { 0 }$ for the value recursion to occur).

Since $c _ { 2 } > 0$ , we can now show that $a _ { t : t + h } ^ { \triangle }$ achieves the highest estimated value among six possible action chunks:

$$
V ^ {\star} - \hat {V} _ {\mathrm{ac}} ^ {\triangle} <   V ^ {\star} - \hat {V} _ {\mathrm{ac}} ^ {\star} = \frac {2 \varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})}, \tag {190}
$$

$\pi _ { \mathrm { a c } } ^ { + } ( X _ { 0 } ) = a _ { t : t + h } ^ { \triangle } = ( 1 , 1 , \cdot \cdot \cdot )$ $\hat { V } _ { \mathrm { a c } } ^ { \triangle } = \hat { V } _ { \mathrm { a c } } ^ { + }$

Finally, putting everything together, we have

$$
V ^ {\star} (X _ {0}) - V _ {\mathrm{ac}} ^ {+} (X _ {0}) = \frac {2 \varepsilon_ {h} \gamma - c _ {2}}{(1 - \gamma) (1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h})} + \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) (1 - (1 - \varepsilon_ {h} - c _ {1}) \gamma^ {h})}, \tag {191}
$$

as desired.

![](images/d152f2d6b8b4fd03da5e528d7326798c2fa8b5520d9a1345011986a57dd9f077.jpg)

# F.9 PROOF OF PROPOSITION 3

Proposition 3 (Optimality of Closed-loop Execution of Action Chunking Policy) Let $V ^ { \bullet }$ be the value of the one-step policy, $\pi ^ { \bullet } ,$ as a result of the closed-loop execution of the action chunking policy $\pi _ { \mathrm { a c } } ^ { + }$ learned from D. That is, for each $s _ { t } \in \mathrm { s u p p } ( P _ { D } ( s _ { t } ) )$ ,

$$
\pi^ {\bullet} (s _ {t}) = a _ {t} ^ {+}, \quad \text { where } a _ {t: t + h} ^ {+} = \pi_ {\mathrm{ac}} ^ {+} (s _ {t}). \tag {27}
$$

If D and ${ \mathcal { D } } ^ { \star }$ are both strongly $\varepsilon _ { h } .$ -open-loop consistent and supp $\big ( P _ { \mathcal { D } } \big ( s _ { t } , a _ { t : t + h } \big ) \big ) \quad \supseteq$ supp $P _ { \cal D ^ { \star } } ( s _ { t } , a _ { t : t + h } ) )$ , then for all $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V ^ {\bullet} (s _ {t}) \leq \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) ^ {2}} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right] \leq 3 \varepsilon_ {h} H ^ {2} \bar {H}. \tag {28}
$$

Proof. We observe that

$$
V _ {\mathrm{ac}} ^ {+} \left(s _ {t}\right) = Q _ {\mathrm{ac}} ^ {+} \left(s _ {t}, a _ {t: t + h} ^ {+}\right) \tag {192}
$$

$$
\leq Q ^ {\star} (s _ {t}, a _ {t} ^ {+}).
$$

Combining this with Theorem 3, we get

$$
Q ^ {\star} (s _ {t}, a _ {t} ^ {+}) \geq V ^ {\star} (s _ {t}) - \Delta , \tag {193}
$$

where $\begin{array} { r } { \Delta = \frac { \varepsilon _ { h } \gamma } { 1 - \gamma } \left[ \frac { 2 } { 1 - ( 1 - 2 \varepsilon _ { h } ) \gamma ^ { h } } + \frac { 1 } { 1 - ( 1 - \varepsilon _ { h } ) \gamma ^ { h } } \right] } \end{array}$ .

Now, we can bound $V ^ { \bullet }$ as follows:

$$
\begin{array}{l} V ^ {\star} (s _ {t}) - V ^ {\bullet} (s _ {t}) \leq Q ^ {\star} (s _ {t}, a _ {t} ^ {+}) - Q ^ {\bullet} (s _ {t}, a _ {t} ^ {+}) + \Delta \\ \leq \gamma \mathbb {E} _ {T (\cdot | s _ {t}, a _ {t} ^ {+})} \left[ V ^ {\star} (s _ {t + 1}) - V ^ {\bullet} (s _ {t + 1}) \right] + \Delta \tag {194} \\ \leq \frac {\varepsilon_ {h} \gamma}{(1 - \gamma) ^ {2}} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right]. \\ \end{array}
$$

□

# F.10 PROOF OF PROPOSITION 2

Proposition 2 (Comparing action chunking backup and n-step return backup) Let D be strongly $\varepsilon _ { h } .$ -open-loop consistent and $\delta _ { n }$ -sub-optimal, and supp $P _ { \mathcal { D } } ( s _ { t } ) ) \supseteq \tilde { \mathrm { s u p p } } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ . Let $\pi _ { n } ^ { + } : s _ { t } \mapsto \arg \operatorname* { m a x } _ { a _ { t } } \hat { Q } _ { n } ^ { + } ( s _ { t } , a _ { t } )$ be the policy learned from $\mathcal { D } ,$ via n-step return backup:

$$
\hat {Q} _ {n} ^ {+} (s _ {t}, a _ {t}) = \mathbb {E} \left[ R _ {t: t + n} + \gamma^ {n} \hat {Q} _ {n} ^ {+} (s _ {t + n}, \pi_ {n} ^ {+} (s _ {t + n})) \right]. \tag {25}
$$

Then, for all $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ (and with $\bar { H } _ { n } = 1 / ( 1 - \gamma ^ { n } ) )$ ,

$$
V _ {\mathrm{ac}} ^ {+} (s _ {t}) - \hat {V} _ {n} ^ {+} (s _ {t}) \geq \frac {\delta_ {n}}{1 - \gamma^ {n}} - \frac {\varepsilon_ {h} \gamma}{1 - \gamma} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right], \tag {26}
$$

$$
\geq \delta_ {n} \bar {H} _ {n} - 3 \varepsilon_ {h} H \bar {H}.
$$

To prove Proposition 2, we first prove the following helper Lemma 6 to quantify sub-optimality for n-step return policy.

Lemma 6 Let $Q _ { n } ^ { \star }$ be the solution of the uncorrected n-step return backup equation:

$$
Q _ {n} ^ {\star} (s _ {t}, a _ {t}) = \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t})} \left[ R _ {t: t + n} + \gamma^ {n} \max _ {a _ {t + n}} Q _ {n} ^ {\star} (s _ {t + n}, a _ {t + n}) \right] \tag {195}
$$

The following inequality holds as long as $\mathcal { D }$ is $\delta _ { n }$ -sub-optimal:

$$
Q ^ {\star} (s _ {t}, a _ {t}) \geq Q _ {n} ^ {\star} (s _ {t}, a _ {t}) + \frac {\delta_ {n}}{1 - \gamma^ {n}}, \forall s _ {t} \in \mathcal {S}, a _ {t} \in \mathcal {A} \tag {196}
$$

where $Q ^ { \star }$ is the Q-function of the optimal policy in M. For the n-step return policy

$$
\pi_ {n} ^ {\star}: s _ {t} \mapsto \arg \max _ {a _ {t}} Q _ {n} ^ {\star} (s _ {t}, a _ {t}), \tag {197}
$$

its corresponding value admits a similar bound:

$$
V ^ {\star} (s _ {t}) \geq V _ {n} ^ {\star} (s _ {t}) + \frac {\delta_ {n}}{1 - \gamma^ {n}}, \forall s _ {t} \tag {198}
$$

Proof. Using the definition of sub-optimal data (Definition 3), we have

$$
Q _ {n} ^ {\star} \left(s _ {t}, a _ {t}\right) = \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t})} \left[ R _ {t: t + n} + \gamma^ {n} \max _ {a _ {t + n}} Q _ {n} ^ {\star} \left(s _ {t + n}, a _ {t + n}\right) \right] \tag {199}
$$

$$
\leq Q ^ {\star} (s _ {t}, a _ {t}) - \delta_ {n} + \gamma^ {h} \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t})} \left[ \max _ {a _ {t + n}} Q _ {n} ^ {\star} (s _ {t + n}, a _ {t + n}) - V ^ {\star} (s _ {t + h}) \right]
$$

Rearranging the inequality above yields

$$
Q _ {n} ^ {\star} (s _ {t}, a _ {t}) - Q ^ {\star} (s _ {t}, a _ {t}) \leq - \delta_ {n} + \gamma^ {n} \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t})} [ V _ {n} ^ {\star} (s _ {t + n}) - V ^ {\star} (s _ {t + n}) ], \forall s _ {t} \in \mathcal {S}, a _ {t} \in \mathcal {A} \tag {200}
$$

By recursively applying the inequality above, we have

$$
Q ^ {\star} (s _ {t}, a _ {t}) \geq Q _ {n} ^ {\star} (s _ {t}, a _ {t}) + \frac {\delta_ {n}}{1 - \gamma^ {n}}, \forall s _ {t} \in \mathcal {S}, a _ {t} \in \mathcal {A} \tag {201}
$$

By choosing $a _ { t } ^ { \star } = \pi _ { n } ^ { \star } ( s _ { t } )$ , we see that

$$
\begin{array}{l} V ^ {\star} (s _ {t}) \geq Q ^ {\star} (s _ {t}, a _ {t}) \\ \geq Q _ {n} ^ {\star} (s _ {t}, a _ {t} ^ {\star}) + \frac {\delta_ {n}}{1 - \gamma^ {n}} \tag {202} \\ = V _ {n} ^ {\star} (s _ {t}) + \frac {\delta_ {n}}{1 - \gamma^ {n}} \\ \end{array}
$$

□

Now we are ready to prove the main Proposition 2.

Proof of Proposition 2. From Lemma 6 and Theorem 3, we have

$$
V _ {n} ^ {\star} (s) + \frac {\delta_ {n}}{1 - \gamma^ {n}} \leq V ^ {\star} (s) \leq V _ {\mathrm{ac}} ^ {+} (s) + \frac {\varepsilon_ {h} \gamma}{1 - \gamma} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right]. \tag {203}
$$

Rearranging the terms give

$$
V _ {\mathrm{ac}} ^ {+} (s) - V _ {n} ^ {\star} (s) \geq \frac {\delta_ {n}}{1 - \gamma^ {n}} - \frac {\varepsilon_ {h} \gamma}{1 - \gamma} \left[ \frac {2}{1 - (1 - 2 \varepsilon_ {h}) \gamma^ {h}} + \frac {1}{1 - (1 - \varepsilon_ {h}) \gamma^ {h}} \right]. \tag {204}
$$

![](images/41c6158c5cfbc92cfa4287704737da200f0b3f734e4df2cf21473138cc97d48b.jpg)

# F.11 PROOF OF THEOREM 7

Theorem 7 (Closed-loop Execution in the Absence of Stochastic Shortcuts) D is α-open-loop mixed and M is free of $\vartheta _ { h }$ -stochastic shortcut, the value $( V ^ { \bullet } )$ of the one-step policy $( \pi ^ { \bullet } )$ as a result of the closed-loop execution of the action chunking policy $\pi _ { \mathrm { a c } } ^ { + }$ learned from D admits the following bound for all $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ :

$$
V ^ {\star} (s _ {t}) - V ^ {\bullet} (s _ {t}) \leq \frac {\alpha}{(1 - \gamma) ^ {2} (1 - \gamma^ {h} (1 - \alpha))} + \frac {\vartheta_ {h} \gamma^ {h}}{(1 - \gamma) (1 - \gamma^ {h})}. \tag {48}
$$

Before we start proving the main theorem, we first prove the following Lemma, which shows that the overestimation of $\hat { V } _ { \mathrm { a c } } ^ { + }$ is bounded when the advantage of the stochastic shortcut (as defined in Definition 7) is bounded.

Lemma 7 (Lack of stochastic shortcut bounds overestimation) If M is free of $\vartheta _ { h ^ { - } }$ advantageous stochastic shortcuts for a horizon $h ,$ then

$$
V _ {\mathrm{ac}} ^ {+} (s _ {t}) - V ^ {\star} (s _ {t}) \leq \frac {\vartheta_ {h}}{1 - \gamma^ {h}}, \tag {205}
$$

where $V _ { \mathrm { a c } } ^ { + }$ is the value function of the action chunking policy learned from any data distribution D from M and $V ^ { \star }$ is the optimal value function in M.

Proof.

$$
\begin{array}{l} \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) - V ^ {\star} (s _ {t}) = \mathbb {E} _ {P _ {\mathcal {D}}} \left[ R _ {t: t + h} + \gamma^ {h} \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) \right] - V ^ {\star} \\ \leq \mathbb {E} _ {P _ {\mathcal {D}}} \left[ \gamma^ {h} (\hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h})) \right] + \vartheta_ {h} \tag {206} \\ \leq \frac {\vartheta_ {h}}{1 - \gamma^ {h}}. \\ \end{array}
$$

![](images/33c8d258b17b1004d4ccf26f56fdf398b8d4f96e34a155741d4b48d1d1d88957.jpg)

Lemma 8 (Monotonicity of optimality) Let $\mathcal { D } ^ { \circ }$ be any data distribution that is collected by an open-loop policy. Then, for all $s _ { t } , a _ { t : t + h } ,$ ,

$$
V ^ {\star} (s _ {t}) \geq \mathbb {E} _ {s _ {t + 1} \sim T (\cdot | s _ {t}, a _ {t})} [ r _ {t} + \gamma V ^ {\star} (s _ {t + 1}) ] \geq \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}} (\cdot | s _ {t}, a _ {t: t + h})} [ R _ {t: t + h} + \gamma^ {h} V ^ {\star} (s _ {t + h}) ] \tag {207}
$$

Proof. The first inequality is clear from the definition of $Q ^ { \star } ( s _ { t } , a _ { t } )$ and $V ^ { \star } ( s _ { t } ) \colon$ :

$$
Q ^ {\star} (s _ {t}, a _ {t}) := \mathbb {E} _ {s _ {t + 1} \sim T (\cdot | s _ {t}, a _ {t})} [ r _ {t} + \gamma V ^ {\star} (s _ {t + 1}) ] \tag {208}
$$

$$
V ^ {\star} (s _ {t}) := \max _ {a _ {t} ^ {\star}} Q ^ {\star} (s _ {t}, a _ {t} ^ {\star}) \geq Q ^ {\star} (s _ {t}, a _ {t}) \tag {209}
$$

For the second inequality, we observe that

$$
\begin{array}{l} \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}} (\cdot | s _ {t}, a _ {t: t + h - 1})} \left[ R _ {t: t + h - 1} + \gamma^ {h - 1} V ^ {\star} (s _ {t + h - 1}) \right] \\ = \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}} (\cdot | s _ {t}, a _ {t: t + h - 1})} \left[ R _ {t: t + h - 1} + \gamma^ {h - 1} \max _ {a _ {t + h - 1} ^ {\star}} Q ^ {\star} (s _ {t + h - 1}, a _ {t + h - 1} ^ {\star}) \right] (210) \\ \geq \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}} (\cdot | s _ {t}, a _ {t: t + h - 1})} \left[ R _ {t: t + h - 1} + \gamma^ {h - 1} Q ^ {\star} (s _ {t + h - 1}, a _ {t + h - 1}) \right] (210) \\ = \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}} (\cdot | s _ {t}, a _ {t: t + h})} \left[ R _ {t: t + h - 1} + \gamma^ {h - 1} \left[ r (s _ {t + h - 1}, a _ {t + h - 1}) + \gamma V ^ {\star} (s _ {t + h}) \right] \right] \\ = \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}} (\cdot | s _ {t}, a _ {t: t + h})} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} (s _ {t + h}) \right] \\ \end{array}
$$

By induction,

$$
\mathbb {E} _ {T (\cdot | s _ {t}, a _ {t})} \left[ r _ {t} + \gamma V ^ {\star} (s _ {t + 1}) \right] \geq \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}} (\cdot | s _ {t}, a _ {t: t + h})} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} (s _ {t + h}) \right], \tag {211}
$$

as desired.

![](images/3333f6aeadc8ad4ca31a2590ad4435f39a082c98cfc58b8feacb7cb431d0621c.jpg)

With these two lemmata, we are now ready to present our main proof.

Proof of Theorem 7. The key idea of our proof is to analyze the action chunk taken by the policy $\pi _ { \mathrm { a c } } ^ { + }$ $( i . e . , \pi _ { \mathrm { a c } } ^ { + } : s _ { t } \mapsto$ arg max $\mathbf { \Phi } _ { a } \hat { Q } ( s _ { t } , a _ { t : t + h } ) )$ . Due to our construction of $\mathcal { D } _ { : }$ the action chunk learned by ac distributions, $\pi _ { \mathrm { a c } } ^ { + }$ either comes from $\mathcal { D } _ { \mathrm { i n } } ^ { \circ }$ and $\mathcal { D } _ { \mathrm { o u t } } ^ { \circ } \mathrm { i }$ ${ \mathcal { D } } ^ { \star }$ : or $\mathcal { D } ^ { \circ }$ . We can $\mathcal { D } ^ { \circ }$ express as the aggregation of two open-loop data

$$
P _ {\mathcal {D} ^ {\circ}} (\cdot \mid s _ {t}) := \hat {\alpha} P _ {\mathcal {D} _ {\text { in }} ^ {\circ}} (\cdot \mid s _ {t}) + (1 - \hat {\alpha}) P _ {\mathcal {D} _ {\text { out }} ^ {\circ}} (\cdot \mid s _ {t}) \tag {212}
$$

where supp $( P _ { \mathcal { D } _ { \mathrm { o u t } } ^ { \circ } } ( a _ { t : t + h } \mid s _ { t } ) ) \cap \operatorname { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( a _ { t : t + h } \mid s _ { t } ) ) = \emptyset$ and

$$
\hat {\alpha} \leq \frac {\alpha \beta}{(1 - \alpha) (1 - \beta)}. \tag {213}
$$

Case 1, from ${ \mathcal { D } } ^ { \star } \colon \operatorname { I f } \pi _ { \mathrm { a c } } ^ { + } ( \cdot \mid s _ { t } ) \cap \operatorname { s u p p } ( P _ { { \mathcal { D } } ^ { \circ } } ( a _ { t : t + h } \mid s _ { t } ) ) = \emptyset$ , then we know that

$$
a _ {t: t + h} ^ {\diamond} \in \operatorname{supp} (P _ {\mathcal {D} ^ {*}} (a _ {t: t + h} \mid s _ {t})), \tag {214}
$$

for any $a _ { t : t + h } ^ { \diamond } \in \pi _ { \mathrm { a c } } ^ { + }$ Thus, the closed-loop execution policy $\pi ^ { \bullet }$ takes the optimal action at $s _ { t } .$ . This leads to the following equalities:

$$
\begin{array}{l} V ^ {\star} \left(s _ {t}\right) - V ^ {\bullet} \left(s _ {t}\right) = V ^ {\star} \left(s _ {t}\right) - Q ^ {\bullet} \left(s _ {t}, a _ {t} ^ {\diamond}\right) \tag {215} \\ = \gamma \mathbb {E} _ {T (\cdot | s _ {t}, a _ {t} ^ {\diamond})} \left[ \left(V ^ {\star} (s _ {t + 1}) - V ^ {\bullet} (s _ {t + 1})\right) \right] \\ \end{array}
$$

$$
\hat {V} _ {\mathrm{ac}} ^ {+} \left(s _ {t}\right) - V ^ {\star} \left(s _ {t}\right) = \mathbb {E} _ {P _ {\mathcal {D} ^ {\star}}} \left[ \hat {Q} _ {\mathrm{ac}} ^ {+} \left(s _ {t}, a _ {t: t + h} ^ {\diamond}\right) \right] - V ^ {\star} \left(s _ {t}\right) \tag {216}
$$

$$
= \gamma^ {h} \mathbb {E} _ {P _ {\mathcal {D} ^ {\star}}} \left[ \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h}) \right]
$$

Case 2, from $\mathcal { D } ^ { \circ }$ : If $\pi _ { \mathrm { a c } } ^ { + } ( \cdot \mid s _ { t } ) \cap \operatorname { s u p p } ( P _ { \mathcal { D } ^ { \circ } } ( a _ { t : t + h } \mid s _ { t } ) ) \neq \emptyset$ , then there exists $a _ { t : t + h } ^ { \circ } \in \pi _ { \mathrm { a c } } ^ { + } ( \cdot \mid s _ { t } )$ such that

$$
a _ {t: t + h} ^ {\circ} \in \operatorname{supp} (P _ {\mathcal {D} ^ {\circ}} (a _ {t: t + h} \mid s _ {t})). \tag {217}
$$

For any $a _ { t : t + h } ^ { + } \in \pi _ { \mathrm { a c } } ^ { + } ( \cdot \mid s _ { t } )$ such that $a _ { t : t + h } ^ { + } \not \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \circ } } ( a _ { t : t + h } \mid s _ { t } ) )$ , we know from the previous case that $V ^ { \star } ( s _ { t } ) - Q ^ { \bullet } ( s _ { t } , a _ { t } ^ { + } ) = \gamma \mathbb { E } _ { T ( \cdot | s _ { t } , a _ { t } ^ { \diamond } ) } \left[ \left( V ^ { \star } ( s _ { t + 1 } ) - V ^ { \bullet } ( s _ { t + 1 } ) \right) \right]$ . We are left with analyzing $V ^ { \star } - Q ^ { \bullet } ( s _ { t } , a _ { t } ^ { \circ } )$ for the second case.

Since $P _ { \mathcal { D } } ( \cdot \mid s _ { t } ) = \beta P _ { \mathcal { D } ^ { \star } } ( \cdot \mid s _ { t } ) + ( 1 - \beta ) \hat { \alpha } P _ { \mathcal { D } _ { \mathrm { i n } } ^ { \diamond } } ( \cdot \mid s _ { t } ) + ( 1 - \beta ) ( 1 - \hat { \alpha } ) P _ { \mathcal { D } _ { \mathrm { o u t } } ^ { \diamond } } ( \cdot \mid s _ { t } )$ with supp $( P _ { \mathcal { D } _ { \mathrm { o u t } } ^ { \circ } } ( a _ { t : t + h } \mid s _ { t } ) ) \cap \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( a _ { t : t + h } \mid s _ { t } ) ) = \emptyset$ , we can isolate the contribution of $\mathcal { D } _ { \mathrm { o u t } } ^ { \circ }$ in $\mathcal { D } ^ { \circ }$ by an event I that is 1 when $a _ { t : t + h }$ ∈ supp( $P _ { \mathcal { D } ^ { \star } } ( a _ { t : t + h } \mid s _ { t } ) \big )$ and 0 otherwise. Now, we can remove $\mathcal { D } _ { \mathrm { o u t } } ^ { \circ }$ when conditioned on $I = 1$ as follows:

$$
P _ {\mathcal {D}} (\cdot \mid s _ {t}, I = 1) = \frac {\beta}{\beta + (1 - \beta) \hat {\alpha}} P _ {\mathcal {D} ^ {*}} (\cdot \mid s _ {t}, I = 1) + \frac {(1 - \beta) \hat {\alpha}}{\beta + (1 - \beta) \hat {\alpha}} P _ {\mathcal {D} ^ {\circ}} (\cdot \mid s _ {t}, I = 1) \tag {218}
$$

$$
= (1 - \bar {\alpha}) P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t}, I = 1) + \bar {\alpha} P _ {\mathcal {D} ^ {\circ}} (\cdot | s _ {t}, I = 1)
$$

Since $\begin{array} { r } { \hat { \alpha } \le \frac { \alpha \beta } { ( 1 - \alpha ) ( 1 - \beta ) } } \end{array}$ , with some algebraic manipulation, we can obtain $\bar { \alpha } \leq \alpha$

By Lemma 1, there exists $a _ { t : t + h } ^ { \diamond } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( a _ { t : t + h } \mid s _ { t } ) )$ such that

$$
P _ {\mathcal {D}} (\cdot \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) = (1 - \tilde {\alpha}) P _ {\mathcal {D} ^ {*}} (\cdot \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) + \tilde {\alpha} P _ {\mathcal {D} ^ {\circ}} (\cdot \mid s _ {t}, a _ {t: t + h} ^ {\diamond}) \tag {219}
$$

with $\tilde { \alpha } \le \bar { \alpha } \le \alpha$

We are now ready to bound $\hat { V } _ { \mathrm { a c } } ^ { + }$ as follows:

$$
\begin{array}{l} \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) - V ^ {\star} (s _ {t}) \geq \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h} ^ {\diamond}) - V ^ {\star} (s _ {t}) \\ = (1 - \tilde {\alpha}) \mathbb {E} _ {P _ {\mathcal {D} ^ {\star}}} \left[ \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h} ^ {\diamond}) \right] + \tilde {\alpha} \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}}} \left[ \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h} ^ {\diamond}) \right] - V ^ {\star} (s _ {t}) \\ \geq (1 - \alpha) \mathbb {E} _ {P _ {\mathcal {D} ^ {*}}} \left[ \gamma^ {h} (\hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h})) \right] - \alpha V ^ {\star} (s _ {t}) \tag {220} \\ \geq (1 - \alpha) \gamma^ {h} \mathbb {E} _ {P _ {\mathcal {D} ^ {\star}}} \left[ \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h}) \right] - \frac {\alpha}{1 - \gamma} \\ \end{array}
$$

where we drop the term $\tilde { \alpha } \mathbb { E } _ { \mathcal { D } ^ { \circ } } \left[ \hat { Q } _ { \mathrm { a c } } ^ { + } ( s _ { t } , a _ { t : t + h } ^ { \circ } ) \right] \geq 0$ for the second inequality.

By combining Equation (220) (when $\pi _ { \mathrm { a c } } ^ { + } ( \cdot  { | } s _ { t } ) \cap \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \circ } } ( \cdot  { | } s _ { t } ) ) \neq \emptyset )$ ) with Equation (216))(when $\pi _ { \mathrm { a c } } ^ { + } ( \cdot \mid s _ { t } ) \cap \operatorname { s u p p } ( P _ { \mathcal { D } ^ { \circ } } ( \cdot \mid s _ { t } ) ) = \emptyset )$ ), we can now recursively bound $\hat { V } _ { \mathrm { a c } } ^ { + } ( s _ { t } ) - V ^ { \star } ( s _ { t } )$ as follows:

$$
\hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) - V ^ {\star} (s _ {t}) \geq - \frac {\alpha}{(1 - \gamma) (1 - \gamma^ {h} (1 - \alpha))}. \tag {221}
$$

Combining this with the result from Lemma 7, we have both a lower-bound and an upper-bound on $\hat { V } _ { \mathrm { a c } } ^ { + }$ in terms of $V ^ { \star }$ :

$$
V ^ {\star} (s _ {t}) - C _ {\alpha} \leq \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) \leq V ^ {\star} (s _ {t}) + C _ {\vartheta}. \tag {222}
$$

where $\begin{array} { r } { C _ { \alpha } : = \frac { \alpha } { ( 1 - \gamma ) ( 1 - \gamma ^ { h } ( 1 - \alpha ) ) } , C _ { \vartheta } : = \frac { \vartheta _ { h } } { 1 - \gamma ^ { h } } } \end{array}$

Now, we are ready to bound $V ^ { \star } - Q ^ { \bullet } ( s _ { t } , a _ { t } ^ { \circ } )$ for the second case (when $\pi _ { \mathrm { a c } } ^ { + } ( \cdot \mid s _ { t } ) \cap \mathrm { s u p p } ( P _ { \mathcal D ^ { \circ } } ( \cdot \mid$ $s _ { t } ) ) \neq \emptyset )$ . For notation convenience, for the following equation, we use $P _ { \mathcal { D } ^ { \circ } }$ as an abbreviation for $P _ { \mathcal { D } ^ { \circ } } ( \cdot \mid s _ { t } , a _ { t : t + h } ^ { \circ } )$ .

$$
\begin{array}{l} V ^ {\star} (s _ {t}) - Q ^ {\bullet} (s _ {t}, a _ {t} ^ {\circ}) \\ = V ^ {\star} - \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}}} \left[ r _ {t} + \gamma V ^ {\bullet} (s _ {t + 1}) \right] \\ \leq V ^ {\star} - \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}}} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} (s _ {t + h}) - \gamma (V ^ {\star} (s _ {t + 1}) - V ^ {\bullet} (s _ {t + 1})) \right] \\ \leq \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) + C _ {\alpha} - \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}}} \left[ R _ {t: t + h} + \gamma^ {h} (\hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - C _ {\vartheta}) - \gamma (V ^ {\star} (s _ {t + 1}) - V ^ {\bullet} (s _ {t + 1})) \right] \\ = \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) - \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}}} \left[ R _ {t: t + h} + \gamma^ {h} (\hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h})) \right] + C _ {\alpha} + \gamma^ {h} C _ {\vartheta} + \mathbb {E} _ {T (\cdot | s _ {t}, a _ {t} ^ {\circ})} \left[ \gamma (V ^ {\star} (s _ {t + 1}) - V ^ {\bullet} (s _ {t + 1})) \right] \\ = C _ {\alpha} + \gamma^ {h} C _ {\vartheta} + \gamma \mathbb {E} _ {T (\cdot | s _ {t}, a _ {t} ^ {\circ})} [ V ^ {\star} (s _ {t + 1}) - V ^ {\bullet} (s _ {t + 1}) ], \tag {223} \\ \end{array}
$$

where the first inequality uses Lemma $^ { 8 , }$

$$
\mathbb {E} _ {T \left(\cdot \mid s _ {t}, a _ {t} ^ {\circ}\right)} \left[ r _ {t} + \gamma V ^ {\star} \left(s _ {t + 1}\right) \right] \geq \mathbb {E} _ {P _ {\mathcal {D} ^ {\circ}} \left(\cdot \mid s _ {t}, a _ {t: t + h} ^ {\circ}\right)} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} \left(s _ {t + h}\right) \right], \tag {224}
$$

and the second inequality uses the lowerbound for $\hat { V } _ { \mathrm { a c } } ^ { + } ( s _ { t } )$ and the upperbound for $\hat { V } _ { \mathrm { a c } } ^ { + } ( s _ { t + h } )$ .

Combining the bounds of $V ^ { \star } - Q ^ { \bullet } ( s _ { t } , a _ { t } ^ { \circ } )$ for both cases (Equation (215) and Equation (223)), we have

$$
\begin{array}{l} V ^ {\star} - V ^ {\bullet} (s _ {t}) \leq \max \left(\gamma \mathbb {E} _ {T (\cdot | s _ {t}, a _ {t} ^ {\diamond})} \left[ \left(V ^ {\star} (s _ {t + 1}) - V ^ {\bullet} (s _ {t + 1})\right) \right], \right. \\ C _ {\alpha} + \gamma^ {h} C _ {\vartheta} + \gamma \mathbb {E} _ {T (\cdot | s _ {t}, a _ {t} ^ {\circ})} [ V ^ {\star} (s _ {t + 1}) - V ^ {\bullet} (s _ {t + 1}) ] \Big) \\ = C _ {\alpha} + \gamma^ {h} C _ {\vartheta} + \gamma \mathbb {E} _ {T (\cdot | s _ {t}, a _ {t} ^ {\circ})} \left[ V ^ {\star} \left(s _ {t + 1}\right) - V ^ {\bullet} \left(s _ {t + 1}\right) \right] \tag {225} \\ \leq \frac {C _ {\alpha} + \gamma^ {h} C _ {\vartheta}}{1 - \gamma} \\ = \frac {\alpha}{(1 - \gamma) ^ {2} (1 - \gamma^ {h} (1 - \alpha))} + \frac {\vartheta \gamma^ {h}}{(1 - \gamma) (1 - \gamma^ {h})}. \\ \end{array}
$$

□

# F.12 PROOF THEOREM 5

Theorem 5 (Closed-loop AC Policy under Bounded OV) Let ${ \mathcal { D } } ^ { \star }$ be the data distribution collected by an optimal policy. Assume D can be decomposed into a mixture of data distributions $\{ \mathcal { D } ^ { \star } , \mathcal { D } _ { 1 } , \dot { \mathcal { D } } _ { 2 } , \cdot \cdot \cdot \mathcal { D } _ { M } \}$ such that each data distribution component satisfies Assumption 1 and for some $\vartheta _ { h } ^ { L } , \vartheta _ { h } ^ { G } \geq 0 .$ they satisfy the following two conditions:

1. Locally bounded optimality variability condition: every $\mathcal { D } _ { i }$ (including $\mathcal { D } ^ { \star } )$ exhibits $\vartheta _ { h } ^ { L } .$ - bounded variability in optimality conditioned on $s _ { t } , a _ { t }$ for all $( s _ { t } , a _ { t } ) \in \mathrm { s u p p } ( P _ { \mathcal { D } _ { i } } ( s _ { t } , a _ { t } ) )$ , and   
2. Globally bounded optimality variability condition: D as a whole exhibits ϑGh -variability in optimality conditioned on $s _ { t } , a _ { t : t + h }$ for all $\left( s _ { t } , a _ { t : t + h } \right) \in$ supp $\left( P _ { \mathcal { D } } ( s _ { t } , a _ { t : t + h } ) \right) \dot { }$ .

Then for all $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s _ {t}) - V ^ {\bullet} (s _ {t}) \leq \frac {\vartheta_ {h} ^ {L}}{1 - \gamma} + \frac {\vartheta_ {h} ^ {G} + \gamma^ {h} \min (\vartheta_ {h} ^ {L} , \vartheta_ {h} ^ {G})}{(1 - \gamma) (1 - \gamma^ {h})} \leq \vartheta_ {h} ^ {L} H + 2 \vartheta_ {h} ^ {G} H \bar {H}. \tag {30}
$$

Proof of Theorem 5. Consider any $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ . Let $a _ { t : t + h } ^ { + } = \pi _ { \mathrm { a c } } ^ { + } ( s _ { t } )$

$$
a _ {t: t + h} ^ {\circ} := \arg \max _ {a _ {t: t + h} \in \operatorname{supp} (P _ {\mathcal {D} ^ {\star}} (a _ {t: t + h} | s _ {t}))} \left[ \mathbb {E} _ {P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t}, a _ {t: t + h})} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} (s _ {t + h}) \right] \right]. \tag {226}
$$

We first observe that

$$
\mathbb {E} _ {P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t}, a _ {t: t + h} ^ {\circ})} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} (s _ {t + h}) \right] \geq V ^ {\star} (s _ {t}), \tag {227}
$$

because

$$
V ^ {\star} (s _ {t}) = \mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t})} \left[ \mathbb {E} _ {P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t}, a _ {t: t + h})} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} (s _ {t + h}) \right] \right], \tag {228}
$$

and the maximum value of a random variable is no less than its expectation.

Let

$$
\tilde {Q} _ {\min} (s _ {t}, a _ {t: t + h} ^ {\circ}) := \min _ {\operatorname{supp} (P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {\circ}))} [ R _ {t: t + h} + V ^ {\star} (s _ {t + h}) ], \tag {229}
$$

$$
\tilde {Q} _ {\max} (s _ {t}, a _ {t: t + h} ^ {\circ}) := \max _ {\operatorname{supp} (P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {\circ}))} \left[ R _ {t: t + h} + V ^ {\star} (s _ {t + h}) \right]. \tag {230}
$$

Since D exhibits $\vartheta _ { h } ^ { G }$ -variability in optimality, we have

$$
\tilde {Q} _ {\min} (s _ {t}, a _ {t: t + h} ^ {\circ}) \geq \tilde {Q} _ {\max} (s _ {t}, a _ {t: t + h} ^ {\circ}) - \vartheta_ {h} ^ {G}. \tag {231}
$$

$$
\begin{array}{l} V ^ {\star} (s _ {t}) - Q ^ {\star} (s _ {t}, a _ {t} ^ {+}) \\ = V ^ {\star} (s _ {t}) - \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) + \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) - Q ^ {\star} (s _ {t}, a _ {t} ^ {+}) \\ = V ^ {\star} (s _ {t}) - \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h} ^ {+}) + \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h} ^ {+}) - Q ^ {\star} (s _ {t}, a _ {t} ^ {+}) \\ \leq V ^ {\star} (s _ {t}) - \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h} ^ {\circ}) + \vartheta_ {h} ^ {L} + \gamma^ {h} \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {+})} \left[ \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h}) \right] \tag {232} \\ = V ^ {\star} (s _ {t}) - \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h} ^ {\circ}) + \vartheta_ {h} ^ {L} + \gamma^ {h} \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {+})} \left[ \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - Q ^ {\star} (s _ {t + h}, a _ {t + h} ^ {+}) \right] - \\ \gamma^ {h} \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {+})} \left[ V ^ {\star} (s _ {t + h}) - Q ^ {\star} (s _ {t + h}, a _ {t + h} ^ {+}) \right]. \\ \end{array}
$$

We can use it to lower-bound $\hat { V } _ { \mathrm { a c } } ^ { + } ( s _ { t } )$ as follows:

$$
\begin{array}{l} \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) = \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h} ^ {+}) \\ \geq \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h} ^ {\circ}) \\ = \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {\circ})} \left[ R _ {t: t + h} + \gamma^ {h} \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) \right] \\ = \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {\circ})} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} (s _ {t + h}) \right] + \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {\circ})} \left[ \gamma^ {h} (\hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h})) \right] \\ \geq \tilde {Q} _ {\min} (s _ {t}, a _ {t: t + h} ^ {\circ}) + \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {\circ})} \left[ \gamma^ {h} (\hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h})) \right] \\ \geq \tilde {Q} _ {\max} (s _ {t}, a _ {t: t + h} ^ {\circ}) - \vartheta_ {h} ^ {G} + \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {\circ})} \left[ \gamma^ {h} (\hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h})) \right] \\ \geq \mathbb {E} _ {P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t}, a _ {t: t + h} ^ {\circ})} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} (s _ {t + h}) \right] - \vartheta_ {h} ^ {G} + \gamma^ {h} \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {\circ})} \left[ \big (\hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h}) \big) \right] \\ \geq V ^ {\star} (s _ {t}) - \vartheta_ {h} ^ {G} + \gamma^ {h} \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {\circ})} \left[ (\hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h})) \right] \\ \geq V ^ {\star} (s _ {t}) - \frac {\vartheta_ {h} ^ {G}}{1 - \gamma^ {h}}. \tag {233} \\ \end{array}
$$

Let $\mathbb { M } ^ { + } = \{ \tilde { D } _ { 1 } , \cdot \cdot \cdot \tilde { D } _ { M ^ { + } } \}$ be all data distributions from $\{ \mathcal { D } ^ { \star } , \mathcal { D } _ { 1 } , \mathcal { D } _ { 2 } , \cdot \cdot \cdot , \mathcal { D } _ { M } \}$ where $( s _ { t } , a _ { t : t + h } ^ { + } )$ is in the support. Let ${ \tilde { \mathcal { D } } } ^ { + }$ be any mixture of M where each mixture component has non-zero weight:

$$
P _ {\tilde {\mathcal {D}} ^ {+}} = \sum_ {i = 1} ^ {M} w _ {i} P _ {\tilde {\mathcal {D}} _ {i}}, \tag {234}
$$

where $w _ { i } > 0 , \sum _ { i } w _ { i } = 1$ .

Let

$$
\tilde {Q} _ {\min} ^ {\star} \left(s _ {t}, a _ {t}\right) := \min _ {\operatorname{supp} \left(P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t}, a _ {t})\right)} \left[ R _ {t: t + h} + V ^ {\star} \left(s _ {t + h}\right) \right], \tag {235}
$$

$$
\tilde {Q} _ {\max} ^ {\star} \left(s _ {t}, a _ {t}\right) := \max _ {\operatorname{supp} \left(P _ {\mathcal {D} ^ {\star}} (\cdot | s _ {t}, a _ {t})\right)} \left[ R _ {t: t + h} + V ^ {\star} \left(s _ {t + h}\right) \right], \tag {236}
$$

$$
\tilde {Q} _ {\min} ^ {i} (s _ {t}, a _ {t}) := \min _ {\operatorname{supp} (P _ {\mathcal {D} ^ {i}} (\cdot | s _ {t}, a _ {t}))} \left[ R _ {t: t + h} + V ^ {\star} (s _ {t + h}) \right], \tag {237}
$$

$$
\tilde {Q} _ {\max} ^ {i} (s _ {t}, a _ {t}) := \max _ {\operatorname{supp} (P _ {\mathcal {D} ^ {i}} (\cdot | s _ {t}, a _ {t}))} \left[ R _ {t: t + h} + V ^ {\star} (s _ {t + h}) \right], \tag {238}
$$

$$
\tilde {Q} _ {\max} ^ {+} (s _ {t}, a _ {t} ^ {+}) := \max _ {\operatorname{supp} (P _ {\bar {\mathcal {D}} ^ {+}} (\cdot | s _ {t}, a _ {t} ^ {+}))} \left[ R _ {t: t + h} + V ^ {\star} (s _ {t + h}) \right], \tag {239}
$$

$$
\tilde {Q} _ {\max} ^ {+} (s _ {t}, a _ {t: t + h} ^ {+}) := \max _ {\operatorname{supp} (P _ {\mathcal {D} ^ {+}} (\cdot | s _ {t}, a _ {t: t + h} ^ {+}))} \left[ R _ {t: t + h} + V ^ {\star} (s _ {t + h}) \right]. \tag {240}
$$

The minimum and the maximum is over the remaining trajectory conditioned on $s _ { t } , a _ { t }$ or $s _ { t } , a _ { t : t + h }$ that is still in the support of the corresponding data distribution.

From the $\vartheta _ { h } ^ { L }$ -bounded variability in optimality and the Assumption 1 of each data mixture, we observe that

$$
Q ^ {\star} (s _ {t}, a _ {t}) \geq \tilde {Q} _ {\min} ^ {i} (s _ {t}, a _ {t}) \geq \tilde {Q} _ {\max} ^ {i} (s _ {t}, a _ {t}) - \vartheta_ {h} ^ {L}, \quad \forall i \in \{1, 2, \dots , M \} \tag {241}
$$

$$
Q ^ {\star} (s _ {t}, a _ {t}) \geq \tilde {Q} _ {\min} ^ {\star} (s _ {t}, a _ {t}) \geq \tilde {Q} _ {\max} ^ {\star} (s _ {t}, a _ {t}) - \vartheta_ {h} ^ {L}. \tag {242}
$$

We can then derive that

$$
\tilde {Q} _ {\max} ^ {+} \left(s _ {t}, a _ {t} ^ {+}\right) = \max \left(\tilde {Q} _ {\max} ^ {\star} \left(s _ {t}, a _ {t} ^ {+}\right), \tilde {Q} _ {\max} ^ {1} \left(s _ {t}, a _ {t} ^ {+}\right), \dots , \tilde {Q} _ {\max} ^ {M} \left(s _ {t}, a _ {t} ^ {+}\right)\right) \tag {243}
$$

$$
\leq Q ^ {\star} (s _ {t}, a _ {t}) + \vartheta_ {h} ^ {L}.
$$

With this, we can now upper-bound $\hat { V } _ { \mathrm { a c } } ^ { + } ( s _ { t } )$ as follows:

$$
\begin{array}{l} \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) = \hat {Q} _ {\mathrm{ac}} ^ {+} (s _ {t}, a _ {t: t + h} ^ {+}) \\ = \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {+})} \left[ R _ {t: t + h} + \gamma^ {h} \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) \right] \\ = \mathbb {E} _ {P _ {\tilde {\mathcal {D}} ^ {+}} (\cdot | s _ {t}, a _ {t: t + h} ^ {+})} \left[ R _ {t: t + h} + \gamma^ {h} \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) \right] \\ = \mathbb {E} _ {P _ {\tilde {\mathcal {D}} ^ {+}} (\cdot | s _ {t}, a _ {t: t + h} ^ {+})} \left[ R _ {t: t + h} + \gamma^ {h} V ^ {\star} (s _ {t + h}) \right] + \gamma^ {h} \mathbb {E} _ {P _ {\tilde {\mathcal {D}} ^ {+}} (\cdot | s _ {t}, a _ {t: t + h} ^ {+})} \left[ \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h}) \right] \\ \leq \tilde {Q} _ {\max} ^ {+} (s _ {t}, a _ {t: t + h} ^ {+}) + \gamma^ {h} \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {+})} \left[ \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h}) \right] \\ \leq \tilde {Q} _ {\max} ^ {+} (s _ {t}, a _ {t} ^ {+}) + \gamma^ {h} \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {+})} \left[ \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h}) \right] \\ \leq Q ^ {\star} (s _ {t}, a _ {t} ^ {+}) + \vartheta_ {h} ^ {L} + \gamma^ {h} \mathbb {E} _ {P _ {\mathcal {D}} (\cdot | s _ {t}, a _ {t: t + h} ^ {+})} \left[ \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t + h}) - V ^ {\star} (s _ {t + h}) \right]. \tag {244} \\ \end{array}
$$

Let

$$
\Delta (s _ {t}) := V ^ {\star} (s _ {t}) - Q ^ {\star} (s _ {t}, a _ {t} ^ {+}). \tag {245}
$$

$$
\hat {\Delta} (s _ {t}) := \hat {V} _ {\mathrm{ac}} ^ {+} (s _ {t}) - Q ^ {\star} (s _ {t}, a _ {t} ^ {+}). \tag {246}
$$

From the inequalities above, we have

$$
\hat {\Delta} (s _ {t}) \leq \vartheta_ {h} ^ {L} + \gamma^ {h} \sup _ {s _ {t + h}} \left[ \hat {\Delta} (s _ {t + h}) - \Delta (s _ {t + h}) \right], \tag {247}
$$

$$
0 \leq \Delta (s _ {t}) \leq \frac {\vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} + \hat {\Delta} (s _ {t}), \tag {248}
$$

$$
\hat {\Delta} (s _ {t}) - \Delta (s _ {t}) \leq \min \left\{\frac {\vartheta_ {h} ^ {G}}{1 - \gamma^ {h}}, \hat {\Delta} (s _ {t}) \right\}. \tag {249}
$$

The minimum operator allows us to obtain two upper-bounds on $\Delta \mathrm { i }$ :

$$
\Delta (s _ {t}) \leq \vartheta_ {h} ^ {L} + \frac {(1 + \gamma^ {h}) \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}}, \tag {250}
$$

$$
\Delta (s _ {t}) \leq \frac {\vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} + \hat {\Delta} (s _ {t}) \leq \frac {\vartheta_ {h} ^ {L} + \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}}. \tag {251}
$$

Finally, combining these two upper-bounds together and recursively applying the inequality yields our desired results:

$$
V ^ {\star} (s _ {t}) - Q ^ {\bullet} (s _ {t}, a _ {t} ^ {+}) \leq \frac {\vartheta_ {h} ^ {L}}{1 - \gamma} + \frac {\vartheta_ {h} ^ {L}}{(1 - \gamma) (1 - \gamma^ {h})} + \frac {\gamma^ {h} \min (\vartheta_ {h} ^ {G} , \vartheta_ {h} ^ {L})}{(1 - \gamma) (1 - \gamma^ {h})}. \tag {252}
$$

![](images/5f88dad7b21af04093100daefd3c1c3c930cb500e2e60d6db5cedd01bffad68b.jpg)

# F.13 PROOF OF THEOREM 6

Theorem 6 (Worst-case Closed-loop AC Policy under BOV) For any $h \ > \ 1 , \gamma \in$ $\begin{array} { r } { ( 0 , 1 ) , \vartheta _ { h } ^ { G } , \vartheta _ { h } ^ { L } \in \left( 0 , \frac { \gamma - \gamma ^ { h } } { 4 ( 1 - \gamma ) } \right] , c \in \left[ 0 , \frac { \gamma - \gamma ^ { h } } { 4 ( 1 - \gamma ^ { h } ) } \right) , \sigma ^ { \cdot } \in \left( 0 , \frac { \operatorname* { m i n } ( \vartheta _ { h } ^ { G } , \vartheta _ { h } ^ { L } ) } { 1 - \gamma } \right) } \end{array}$ 0, γ−γ4(1−γ) h0, γ−γ4(1−γh)  , there exists M and D satisfying the assumptions in Theorem 5 such that there exists $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ , where

$$
V ^ {\star} (s _ {t}) - V ^ {\bullet} (s _ {t}) = \frac {\vartheta_ {h} ^ {L}}{1 - \gamma} + \frac {\vartheta_ {h} ^ {G} + \gamma^ {h} \min (\vartheta_ {h} ^ {L} , \vartheta_ {h} ^ {G})}{(1 - \gamma) (1 - \gamma^ {h})} - \sigma , V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) \geq \frac {c}{1 - \gamma}. \tag {31}
$$

To show that our upper bound is achievable, we need to carefully design both the MDP and the data distribution. For clarity of the proof, we divide up the construction into two parts. The first part (Lemma 9) focuses on designing part of the MDP and two data distributions $\bar { \mathcal { D } } ^ { \star }$ and D⋄ such that any action chunk that has a value bigger than $\begin{array} { r } { V ^ { \star } - \frac { \vartheta _ { h } ^ { G } } { 1 - \gamma ^ { h } } } \end{array}$ ϑ h1−γh is preferred over the action chunks in D⋆ ${ \mathcal { D } } ^ { \star }$ and $\mathcal { D } ^ { \diamond }$ . The second part (Lemma 10) focuses on constructing the remaining MDP and the $\mathcal { D } ^ { \triangle }$ that contains the action chunk that $\pi _ { \mathrm { a c } } ^ { + }$ picks where $\hat { V } _ { \mathrm { a c } } ^ { + }$ overestimates the value of this action chunk by $\begin{array} { r } { \vartheta _ { h } ^ { L } + \frac { \gamma ^ { h } \operatorname* { m i n } ( \vartheta ^ { L } , \vartheta _ { h } ^ { G } ) } { 1 - \gamma ^ { h } } } \end{array}$ 1−γh . Finally, we assemble these two results (combining $\mathcal { D } ^ { \star } , \mathcal { D } ^ { \circ } , \mathcal { D } ^ { \triangle } )$ to show that the MDP and the mixture data achieve our upper-bound exactly.

Lemma 9 (“The Castle”) For $\begin{array} { r } { \delta \in ( 0 , 1 ) , \vartheta _ { h } ^ { G } < \frac { \gamma - \gamma ^ { h } } { 2 ( 1 - \gamma ) } . } \end{array}$ , consider a 2-state, 2-action MDP in Figure 12. Let there be two data distributions, ${ \mathcal { D } } ^ { \star }$ and $\mathcal { D } ^ { \circ } , \mathcal { D } ^ { \star }$ is collected by the following optimal closed-loop policy from X and $Y { : }$

$$
\pi^ {\star} (X) = 0, \pi^ {\star} (Y) = 1. \tag {253}
$$

$\mathcal { D } ^ { \diamond }$ is collected by the following optimal closed-loop policy from X and $Y \colon$

$$
\pi^ {\diamond} (X) = 1, \pi^ {\diamond} (Y) = 0. \tag {254}
$$

Let D be a mixture of ${ \mathcal { D } } ^ { \star }$ and $\mathcal { D } ^ { \diamond }$ with

$$
P _ {\mathcal {D}} = (1 - \varsigma) P _ {\mathcal {D} ^ {*}} + \varsigma P _ {\mathcal {D} ^ {\diamond}}. \tag {255}
$$

There exists $c _ { 1 } \in ( 0 , 1 / 2 )$ such that

1. $\mathcal { D } ^ { \star }$ and $\mathcal { D } ^ { \diamond }$ both individually exhibits 0-variability in optimality conditioned on $s _ { t } , a _ { t }$ for all $s _ { t } , a _ { t } \in \mathrm { s u p p } \bigl ( P _ { \mathcal { D } } \bigl ( s _ { t } , a _ { t } \bigr ) \bigr )$ ,   
2. D exhibits $\vartheta _ { h } ^ { G }$ -variability in optimality conditioned on $s _ { t } , a _ { t : t + h }$ for all $s _ { t } , a _ { t : t + h } \in$ supp $( P _ { \mathcal { D } } ( s _ { t } , \ddot { a } _ { t : t + h } ) )$ ,

and

$$
\hat {V} _ {\mathrm{ac}} ^ {+} (X) = \hat {V} _ {\mathrm{ac}} ^ {+} (Y) = \frac {1 - \gamma + \varsigma (\gamma - \gamma^ {h})}{2 (1 - \gamma^ {h}) (1 - \gamma)} - \frac {\varsigma \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}}. \tag {256}
$$

Proof. Set

$$
c _ {1} = \frac {(1 - \gamma) \vartheta_ {h} ^ {G}}{\gamma - \gamma^ {h}}. \tag {257}
$$

We first check whether $c _ { 1 } ~ \in ~ ( 0 , 1 / 2 )$ . For the upper-bound, it is clear that $c _ { 1 } < 1 / 2$ because $\begin{array} { r } { \vartheta _ { h } ^ { G } < \frac { \gamma - \gamma ^ { h } } { 2 ( 1 - \gamma ) } } \end{array}$ < γ−γh . For the lower-bound, $c > 0$ because all terms in the fraction are positive.

We now check the two optimality variability conditions. The first (local) one is trivial because $\pi ^ { \diamond }$ always receives $r = 1 / 2 - c _ { 1 }$ and $\pi ^ { \star }$ always receives $r = 1 / 2$ , and the optimal value for X and $Y$ are both $V ^ { \star } ( X ) = V ^ { \star } ( Y ) = \frac { 1 } { 2 ( 1 - \gamma ) }$ .

![](images/efb92c234282191575288363b7e7d0b092e6bdd606ef07029c0f018663547b4c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    Y1["Y"] --> a0["a=0"]
    Y1 --> a1["a=1"]
    X1["X"] --> a1a["a=1"]
    X1 --> a0b["a=0"]
    Y1 --> r1r1[" r = 1/2 - c1 "]
    X1 --> r1r2[" r = 1/2 - c1 "]
    Y1 --> δ1[" δ "]
    X1 --> δ2[" δ "]
    Y1 --> δ3[" δ "]
    X1 --> δ4[" δ "]
    Y1 --> δ5[" δ "]
    X1 --> δ6[" δ "]
    Y1 --> δ7[" δ "]
    X1 --> δ8[" δ "]
    Y1 --> δ9[" δ "]
    X1 --> δ10[" δ "]
    Y1 --> δ11[" δ "]
    X1 --> δ12[" δ "]
    Y1 --> δ13[" δ "]
    X1 --> δ14[" δ "]
    Y1 --> δ15[" δ "]
    X1 --> δ16[" δ "]
    Y1 --> δ17[" δ "]
    X1 --> δ18[" δ "]
    Y1 --> δ19[" δ "]
    X1 --> δ20[" δ "]
    Y1 --> δ21[" δ "]
    X1 --> δ22[" δ "]
    Y1 --> δ23[" δ "]
    X1 --> δ24[" δ "]
    Y1 --> δ25[" δ "]
    X1 --> δ26[" δ "]
    Y1 --> δ27[" δ "]
    X1 --> δ28[" δ "]
    Y1 --> δ29[" δ "]
    X1 --> δ30[" δ "]
    Y1 --> δ31[" δ "]
    X1 --> δ32[" δ "]
    Y1 --> δ33[" δ "]
    X1 --> δ34[" δ "]
    Y1 --> δ35[" δ "]
    X1 --> δ36[" δ "]
    Y1 --> δ37[" δ "]
    X1 --> δ38[" δ "]
    Y1 --> δ39[" δ "]
    X1 --> δ40[" δ "]
    Y1 --> δ41[" δ "]
    X1 --> δ42[" δ "]
    Y1 --> δ43[" δ "]
    X1 --> δ44[" δ "]
    Y1 --> δ45[" δ "]
    X1 --> δ46[" δ "]
    Y1 --> δ47[" δ "]
    X1 --> δ48[" δ "]
    Y1 --> δ49[" δ "]
    X1 --> δ50[" δ "]
    Y1 --> δ51[" δ "]
    X1 --> δ52[" δ "]
    Y1 --> δ53[" δ "]
    X1 --> δ54[" δ "]
    Y1 --> δ55[" δ "]
    X1 --> δ56[" δ "]
    Y1 --> δ57[" δ "]
    X1 --> δ58[" δ "]
    Y1 --> δ59[" δ "]
    X1 --> δ60[" δ "]
    Y1 -->δ61["δ "]
    X1 -->δ62["δ "]
    Y1 -->δ63["δ "]
    X1 -->δ64["δ "]
    Y1 -->δ65["δ "]
    X1 -->δ66["δ "]
    Y1 -->δ67["δ "]
    X1 -->δ68["δ "]
    Y1 -->δ69["δ "]
    X1 -->δ70["δ "]
    Y1 -->δ71["δ "]
    X1 -->δ72["δ "]
    Y1 -->δ73["δ "]
    X1 -->δ74["δ "]
    Y1 -->δ75["δ "]
    X1 -->δ76["δ "]
    Y1 -->δ77["δ "]
    X1 -->δ78["δ "]
    Y1 -->δ79["δ "]
    X1 -->δ80["δ "]
    Y1 -->δ81["δ "]
    X1 -->δ82["δ "]
    Y1 -->δ83["δ "]
    X1 -->δ84["δ "]
    Y1 -->δ85["δ "]
    X1 -->δ86["δ "]
    Y1 -->δ87["δ "]
    X1 -->δ88["δ "]
    Y1 -->δ89["δ "]
```
</details>

Figure 12: MDP construction Part 1 for Theorem 6 (“the castle”). This diagram describes state X and $Y$ and how actions $a = 0$ and $a = 1$ transition between them. The main purpose of this construction is to make $\hat { V } _ { \mathrm { a c } } ^ { + } ( X )$ underestimate $V ^ { \star }$ by exactly $\vartheta _ { h } ^ { G } / ( 1 - \gamma ^ { h } )$ . This allows the action chunk that appears in the second part of the construction to be preferred (by $\pi _ { \mathrm { a c } } ^ { + } )$ over the action chunks that start with $a = 0 \mathrm { o r } a = 1$ .

Next, we check the second (global) condition by analyzing all possible states and action chunks in D. We observe that for any $a _ { t : t + h }$ that starts with $a _ { t } = 0 .$ , we have

$$
\tilde {Q} _ {\min} (X, a _ {t: t + h}) = \frac {1 - 2 c _ {1} (\gamma - \gamma^ {h})}{2 (1 - \gamma)}, \tag {258}
$$

$$
\tilde {Q} _ {\max} (X, a _ {t: t + h}) = \frac {1}{2 (1 - \gamma)}, \tag {259}
$$

which gives

$$
\tilde {Q} _ {\max} (X, a _ {t: t + h}) - \tilde {Q} _ {\min} (X, a _ {t: t + h}) = \vartheta_ {h} ^ {G}. \tag {260}
$$

By symmetry, we also have

$$
\tilde {Q} _ {\max} (Y, a _ {t: t + h}) - \tilde {Q} _ {\min} (Y, a _ {t: t + h}) = \vartheta_ {h} ^ {G}. \tag {261}
$$

for all $\scriptstyle a _ { t : t + h }$ that starts with $a _ { t } = 1$ .

Now, for any $\scriptstyle a _ { t : t + h }$ that starts with $a _ { t } = 1$ , we have

$$
\tilde {Q} _ {\min} (X, a _ {t: t + h}) = \frac {\gamma - 2 c _ {1} (\gamma - \gamma^ {h})}{2 (1 - \gamma)}, \tag {262}
$$

$$
\tilde {Q} _ {\max} (X, a _ {t: t + h}) = \frac {\gamma}{2 (1 - \gamma)}, \tag {263}
$$

which admits the same gap as the case when $a _ { t } ~ = ~ 0$ . The same also holds for $Y$ with $a _ { t } =$ 1. Thus, D exhibits $\vartheta _ { h } ^ { \bar { G } }$ -variability in optimality conditioned on $s _ { t } , a _ { t : t + h }$ for all $s _ { t } , a _ { t : t + h } \in$ supp $P _ { \mathcal { D } } ( s _ { t } , a _ { t : t + h } ) )$ .

Finally, we check for the value,

$$
\begin{array}{l} \hat {V} _ {\mathrm{ac}} ^ {+} (X) = \hat {V} _ {\mathrm{ac}} ^ {+} (Y) = (1 - \varsigma) / 2 + \varsigma (1 / 2 + (1 - 2 c _ {1}) \frac {\gamma - \gamma^ {h}}{2 (1 - \gamma)}) \\ = \frac {1}{1 - \gamma^ {h}} \left[ 1 / 2 + \varsigma \frac {\left(1 - 2 c _ {1}\right) \left(\gamma - \gamma^ {h}\right)}{2 (1 - \gamma)} \right] \tag {264} \\ = \frac {1}{2 (1 - \gamma^ {h})} \left[ 1 + \varsigma \frac {\gamma - \gamma^ {h} - 2 (1 - \gamma) \vartheta_ {h} ^ {G}}{1 - \gamma} \right] \\ = \frac {1 - \gamma + \varsigma (\gamma - \gamma^ {h})}{2 (1 - \gamma^ {h}) (1 - \gamma)} - \frac {\varsigma \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}}, \\ \end{array}
$$

as desired.

![](images/d0473d27b20456d00af5c61fa5d8b3e8a83b121bbc01cc056b632cb670aeb49a.jpg)

Lemma 10 (“The Flower”) Assume $\begin{array} { r } { \vartheta _ { h } ^ { G } \in \left( 0 , \frac { 1 - \gamma ^ { h } } { 8 } \right] , \vartheta _ { h } ^ { L } \in \left( 0 , \frac { \gamma - \gamma ^ { h } } { 4 ( 1 - \gamma ) } \right] , \gamma \in ( 0 , 1 ) } \end{array}$ 0, γ−γ4(1−γ) , and Consider a 5-state, 3-action MDP in Figure 13 building on top of the transitions that already in Figure 12. Let $\mathcal { D } ^ { \triangle }$ be a data distribution induced by a cycling, time-dependent (with a time cycle length of h) policy $\pi ^ { \triangle }$ (we use the subscript to indicate the time step from 0 to h − 1):

$$
\pi_ {0} ^ {\triangle} (s _ {t} = X) = \pi_ {0} ^ {\triangle} (s _ {t} = \tilde {X}) = 2, \tag {265}
$$

$$
\pi_ {0} ^ {\triangle} \left(s _ {t} = Y\right) = 3 \tag {266}
$$

$$
\pi_ {k} ^ {\triangle} (s _ {t + k} = \tilde {C}) = \pi_ {k} ^ {\triangle} (s _ {t + k} = \tilde {D}) = 2, \quad \forall k \in \{1, 2, \dots , h - 2 \}, \tag {267}
$$

$$
\pi_ {k} ^ {\triangle} (s _ {t + h - 1} = \tilde {C}) = \pi_ {k} ^ {\triangle} (s _ {t + h - 1} = \tilde {D}) = 0, \tag {268}
$$

$$
\pi_ {k} ^ {\triangle} (s _ {t + k} = X) = 0, \quad \forall k \in \{1, 2, \dots , h - 1 \}, \tag {269}
$$

$$
\pi_ {k} ^ {\triangle} (s _ {t + k} = Y) = 1, \quad \forall k \in \{1, 2, \dots , h - 1 \}. \tag {270}
$$

Let $\hat { V } _ { \mathrm { a c } } ^ { + }$ be the nominal value of the action chunking policy $\pi _ { \mathrm { a c } } ^ { + }$ learned from $\mathcal { D } ^ { \triangle }$ and let

$$
\Delta = \vartheta_ {h} ^ {L} + \frac {\vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} + \frac {\gamma^ {h} \min (\vartheta_ {h} ^ {G} , \vartheta_ {h} ^ {L})}{1 - \gamma^ {h}}. \tag {271}
$$

For any $c \in \left[ 0 , { \frac { \gamma - \gamma ^ { h } } { 4 ( 1 - \gamma ^ { h } ) } } \right)$ , there exists some $0 < c _ { 2 } \leq 1 / 2 , 0 < c _ { 3 } \leq 1 / 2 , \delta , \delta _ { 2 } \in ( 0 , 1 )$ , such that for every $\begin{array} { r } { 0 < \tilde { \Delta } < \dot { \operatorname* { m i n } } \left( \Delta , \frac { 2 \vartheta _ { h } ^ { G } } { 1 - \gamma ^ { h } } \right) } \end{array}$ ,

1. $\mathcal { D } ^ { \triangle }$ exhibits 0-variability in optimality conditioned on $s _ { t } , a _ { t : t + h }$ for all $s _ { t } , a _ { t : t + h } \in$ supp( $P _ { \mathcal { D } ^ { \triangle } } \left( s _ { t } , a _ { t : t + h } \right) )$ ,   
2. $\mathcal { D } ^ { \triangle }$ exhibits $\vartheta _ { h } ^ { L } .$ -variability in optimality conditioned on $s _ { t } , a _ { t }$ for all $s _ { t } , a _ { t } \in$ supp $( P _ { D } \triangle ( s _ { t } , \ddot { a _ { t } } ) )$ ,

and

$$
\hat {V} _ {\mathrm{ac}} ^ {+} (X) = \frac {1}{2 (1 - \gamma)} - \frac {\vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} + \tilde {\Delta}, \tag {272}
$$

$$
V ^ {\star} (X) - V ^ {\bullet} (X) = \frac {\Delta - \tilde {\Delta}}{1 - \gamma}, \tag {273}
$$

$$
V ^ {\star} (X) - V _ {\mathrm{ac}} ^ {+} (X) \geq \frac {c}{1 - \gamma}, \tag {274}
$$

$$
V ^ {\star} (X) - V _ {\mathrm{ac}} ^ {\star} (X) \geq \frac {c}{1 - \gamma}. \tag {275}
$$

Proof. Without the loss of generality, we assume we always start from state X. Due to symmetry, the same analysis applies to state $Y$ (with the first action being $a _ { t } = 3$ rather than $a _ { t } = 2 )$ .

![](images/f860189c7894de78ff696d56e48f9737c829c25ed433ec4053fb03d80ea38b43.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X -->|a=0, r=1/2| A
    X -->|1-δ₂| B
    B -->|a=2, r=(1+c₄)/2| C
    B -->|a=2, r=1/2| D
    D -->|a=0, r=(1+c₄)/2| E
    E --> F
    F -->|a=0, r=(1+c₃)/2| G
    G --> C
    C -->|a=2, r=(1+c₃)/2| H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> R
    R --> S
    S --> T
    T --> U
    U --> V
    V --> W
    W --> X
    X <--> (1-δ₂)(1-δ) --> R = 1-c₂/2 --> Y
    R = 1-c₂/2 --> (1-δ₂)δ --> Y
    R = 0 <--> a3 --> Q
    R = 0 <--> a2 --> Q
    R = 0 <--> δ2 --> Q
    R = 0 <--> δ2 --> Q
    R = 0 <--> δ2 --> Q
    R = 0 <--> δ2 --> Q
    R = 0 <--> δ2 --> Q
```
</details>

Figure 13: MDP construction Part 2 for Theorem 6 (“the flower”). This diagram describes the remaining states C˜, D˜ and $\tilde { X } .$ , and what actions $a = 2$ and $a = 3$ do in state X and Y . The main purpose of this construction is to make $\hat { V } _ { \mathrm { a c } } ^ { + } ( X )$ overestimate the optimal value of the action chunks that $\pi _ { \mathrm { a c } } ^ { + } , Q ^ { \star } ( X , a _ { t } ^ { + } )$ , by exactly $\vartheta _ { h } ^ { L } + \gamma ^ { h } \operatorname* { m i n } ( \vartheta _ { h } ^ { L } , \vartheta _ { h } ^ { G } ) / ( 1 - \gamma ^ { h } )$ .

Due to cycling nature of the data collection policy, we observe that all action chunks starting from X are in the form of $a _ { t : t + h } = ( 2 , \underset { } { \overset { . . . } { \underbrace { \dots } } } )$ or $a _ { t : t + h } = ( 2 , 2 , \cdot \cdot \cdot , 2 , 0 )$ . These two possibilities |{z}0’s and 1’s correspond to two different paths that the data collection policy takes:

• $a _ { t : t + h } ^ { \circ } = ( 2 , \underset { \cdots } { \substack { } } ) :$ : Stay in either X or Y . The agent going on this path receives a |{z}0’s and 1’s constant reward of $1 / 2$ except the first step where it receives a reward of $( 1 - c _ { 2 } ) / 2$ .   
• $a _ { t : t + h } ^ { \triangle } = ( 2 , 2 , \cdot \cdot \cdot , 2 , 0 )$ : Visit $\tilde { C }$ and then stays there for $h - 1$ until it goes out with $a = 0$ to visit $\tilde { X }$ . The agent going on this path receives a constant reward of $( 1 + c _ { 3 } ) / 2$ except the first step where it receives a reward of $( 1 - c _ { 2 } ) / 2$ .

Similarly, all action chunks starting from $\tilde { X }$ are in the form of $a _ { t : t + h } = ( 2 , \mathbf { \alpha } _ { \underbrace { \left. \cdots \right. } } ) \operatorname { o r } a _ { t : t + h } =$ |{z}0’s and 1’s $( 2 , 2 , \cdots , 2 , 0 )$ . These two possibilities correspond to two different paths that the data collection policy takes:

• $a _ { t : t + h } ^ { \circ } = ( 2 , \underbrace { \cdot \cdot \cdot } _ { 0 ^ { : } \mathrm { s } \mathrm { a n d } 1 ^ { \circ } \mathrm { s } } )$ |{z}0’s and 1’s : Stay in either X or Y . The agent going on this path receives a constant reward of $1 / 2$ .

• $a _ { t : t + h } ^ { \triangle } = ( 2 , 2 , \cdot \cdot \cdot , 2 , 0 )$ : Visit $\tilde { C }$ and then stays there for $h - 1$ until it goes out with $a = 0$ to visit $\tilde { X }$ . The agent going on this path receives a constant reward of $( 1 + c _ { 4 } ) / 2$ except the first step where it receives a reward of $1 / 2$ .

Now, we divide up the problem into two cases depending on the relative values of $\vartheta _ { h } ^ { L }$ and $\vartheta _ { h } ^ { G }$ .

1. Case $\vartheta _ { h } ^ { L } \geq \vartheta _ { h } ^ { G }$ :

Set

$$
c _ {2} = 2 \left[ \vartheta_ {h} ^ {L} + \frac {(1 + \gamma^ {h}) \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} \right] - 2 \tilde {\Delta} > 0, \tag {276}
$$

$$
c _ {3} = \frac {2 (1 - \gamma) \vartheta_ {h} ^ {L}}{\gamma - \gamma^ {h}} > 0, \tag {277}
$$

$$
c _ {4} = \frac {2 (1 - \gamma) \vartheta_ {h} ^ {G}}{\gamma - \gamma^ {h}} > 0. \tag {278}
$$

Next, we check that $c _ { 2 } , c _ { 3 } , c _ { 4 } \leq 1 / 2$ .

We first observe that

$$
(1 - \gamma) (1 - \gamma^ {h}) - 2 (\gamma - \gamma^ {h}) = 1 - 3 \gamma + \gamma^ {h} (\gamma + 1) \leq 1 - 3 \gamma + \gamma (\gamma + 1) = (1 - \gamma) ^ {2} \geq 0. \tag {279}
$$

Dividing both sides by $8 ( 1 - \gamma )$ yields

$$
\frac {1 - \gamma^ {h}}{8} \geq \frac {\gamma - \gamma^ {h}}{4 (1 - \gamma)} \geq \vartheta_ {h} ^ {L} \geq \vartheta_ {h} ^ {G}. \tag {280}
$$

Now, using the inequality above, we have

$$
\begin{array}{l} c _ {2} = 2 \left[ \vartheta_ {h} ^ {L} + \frac {(1 + \gamma^ {h}) \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} \right] - \tilde {2} \Delta \\ \leq 2 \left[ \vartheta_ {h} ^ {L} + \frac {(1 + \gamma^ {h}) \vartheta_ {h} ^ {L}}{1 - \gamma^ {h}} \right] \tag {281} \\ \leq \frac {4 \vartheta_ {h} ^ {L}}{1 - \gamma^ {h}} \\ \leq 1 / 2. \\ \end{array}
$$

Furthermore,

$$
c _ {4} \leq c _ {3} = \frac {2 (1 - \gamma) \vartheta_ {h} ^ {L}}{\gamma - \gamma^ {h}} \leq 1 / 2. \tag {282}
$$

Next, we check the data distribution $\mathcal { D } ^ { \triangle }$ satisfies both optimality variability conditions. We first note that we only need to check for $s _ { t } \in \{ X , \tilde { X } \}$ because all other states are out of the support due to the cycling nature of the data collection policies. The first (global) optimality condition is trivial because the h-step reward received is deterministic conditioned on $a _ { t : t + h } \in \{ a _ { t : t + h } ^ { \circ } , a _ { t : t + h } ^ { \triangle } \}$ , and the optimal value of $V ^ { \star } ( s _ { t + h } )$ is always $\frac { 1 } { 2 ( 1 - \gamma ) }$ . This leads to 0-variability in optimality conditioned on $s _ { t } , a _ { t : t + h }$ . For the second (local) optimality condition, we check the difference in optimality for two paths from $s _ { t } , a _ { t } = 2$ for both $s _ { t } = X$ and $s _ { t } = \tilde { X }$ .

For $s _ { t } = X$ , the optimality gap is

$$
c _ {3} \frac {\gamma - \gamma^ {h}}{2 (1 - \gamma^ {h})} = \vartheta_ {h} ^ {L}. \tag {283}
$$

For $s _ { t } = \tilde { X }$ , the optimality gap is

$$
c _ {4} \frac {\gamma - \gamma^ {h}}{2 (1 - \gamma^ {h})} = \vartheta_ {h} ^ {G} \leq \vartheta_ {h} ^ {L}. \tag {284}
$$

This concludes that the second (local) optimality condition is also satisfied.

Next, we first analyze which action chunk $\pi _ { \mathrm { a c } } ^ { + }$ prefers by computing $\hat { Q } _ { \mathrm { a c } } ^ { + } \mathrm { \tilde { s } } \mathrm { : }$

$$
\hat {Q} _ {\mathrm{ac}} ^ {+} (X, a _ {t: t + h} ^ {\circ}) = \frac {1}{2} \left[ (1 - c _ {2}) + \frac {\gamma - \gamma^ {h}}{1 - \gamma} \right] + \gamma^ {h} \hat {V} _ {\mathrm{ac}} ^ {+} (X), \tag {285}
$$

$$
\hat {Q} _ {\mathrm{ac}} ^ {+} (X, a _ {t: t + h} ^ {\triangle}) = \frac {1}{2} \left[ (1 - c _ {2}) + (1 + c _ {3}) \frac {\gamma - \gamma^ {h}}{1 - \gamma} \right] + \gamma^ {h} \hat {V} _ {\mathrm{ac}} ^ {+} (\tilde {X}), \tag {286}
$$

$$
\hat {Q} _ {\mathrm{ac}} ^ {+} (\tilde {X}, a _ {t: t + h} ^ {\circ}) = \frac {1}{2} \left[ \frac {1 - \gamma^ {h}}{1 - \gamma} \right] + \gamma^ {h} \hat {V} _ {\mathrm{ac}} ^ {+} (X), \tag {287}
$$

$$
\hat {Q} _ {\mathrm{ac}} ^ {+} (\tilde {X}, a _ {t: t + h} ^ {\triangle}) = \frac {1}{2} \left[ 1 + (1 + c _ {4}) \frac {\gamma - \gamma^ {h}}{1 - \gamma} \right] + \gamma^ {h} \hat {V} _ {\mathrm{ac}} ^ {+} (\tilde {X}). \tag {288}
$$

We first observe that

$$
\begin{array}{l} \hat {Q} _ {\mathrm{ac}} ^ {+} (\tilde {X}, a _ {t: t + h} ^ {\triangle}) - \hat {Q} _ {\mathrm{ac}} ^ {+} (X, a _ {t: t + h} ^ {\triangle}) = \frac {1}{2} \left[ c _ {2} - (c _ {3} - c _ {4}) \frac {\gamma - \gamma^ {h}}{1 - \gamma} \right] \\ = \vartheta_ {h} ^ {L} + \frac {\left(1 + \gamma^ {h}\right) \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} - \vartheta_ {h} ^ {L} + \vartheta_ {h} ^ {G} - \tilde {\Delta} \tag {289} \\ = \frac {2 \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} - \tilde {\Delta} \\ > 0. \\ \end{array}
$$

Also,

$$
\hat {Q} _ {\mathrm{ac}} ^ {+} \left(\tilde {X}, a _ {t: t + h} ^ {\circ}\right) - \hat {Q} _ {\mathrm{ac}} ^ {+} \left(X, a _ {t: t + h} ^ {\circ}\right) = c _ {2} > 0 \tag {290}
$$

Therefore,

$$
\begin{array}{l} \hat {V} _ {\mathrm{ac}} ^ {+} (X) = \max (\hat {Q} _ {\mathrm{ac}} ^ {+} (X, a _ {t: t + h} ^ {\circ}), \hat {Q} _ {\mathrm{ac}} ^ {+} (X, a _ {t: t + h} ^ {\triangle})) \\ <   \max (\hat {Q} _ {\mathrm{ac}} ^ {+} (\tilde {X}, a _ {t: t + h} ^ {\circ}), \hat {Q} _ {\mathrm{ac}} ^ {+} (\tilde {X}, a _ {t: t + h} ^ {\triangle})) \tag {291} \\ = \hat {V} _ {\mathrm{ac}} ^ {+} (\tilde {X}). \\ \end{array}
$$

Now, we can compare the values for the action chunks for X and $\tilde { X }$ :

$$
\hat {Q} _ {\mathrm{ac}} ^ {+} (X, a _ {t: t + h} ^ {\triangle}) - \hat {Q} _ {\mathrm{ac}} ^ {+} (X, a _ {t: t + h} ^ {\circ}) = c _ {3} \frac {\gamma - \gamma^ {h}}{2 (1 - \gamma)} + \gamma^ {h} (\hat {V} _ {\mathrm{ac}} ^ {+} (\tilde {X}) - \hat {V} _ {\mathrm{ac}} ^ {+} (X)) > 0, \tag {292}
$$

$$
\hat {Q} _ {\mathrm{ac}} ^ {+} (\tilde {X}, a _ {t: t + h} ^ {\triangle}) - \hat {Q} _ {\mathrm{ac}} ^ {+} (\tilde {X}, a _ {t: t + h} ^ {\circ}) = c _ {4} \frac {\gamma - \gamma^ {h}}{2 (1 - \gamma)} + \gamma^ {h} (\hat {V} _ {\mathrm{ac}} ^ {+} (\tilde {X}) - \hat {V} _ {\mathrm{ac}} ^ {+} (X)) > 0, \tag {293}
$$

since c3, c4 > 0 and h > 1, 0 < γ < 1 (and thus γ−γh1−γ > $c _ { 3 } , c _ { 4 } > 0$ $h > 1 , 0 < \gamma < 1$ $\frac { \gamma - \gamma ^ { h } } { 1 - \gamma } > 0 )$ 1- .

This concludes that $\pi _ { \mathrm { a c } } ^ { + } ( X ) = \pi _ { \mathrm { a c } } ^ { + } ( \tilde { X } ) = a _ { t : t + h } ^ { \bigtriangleup } = ( 2 , 2 , \cdot \cdot \cdot , 2 , 0 )$ and thus

$$
\hat {V} _ {\mathrm{ac}} ^ {+} (\tilde {X}) = \frac {1 - \gamma + (\gamma - \gamma^ {h}) (1 + c _ {4})}{2 (1 - \gamma^ {h}) (1 - \gamma)}, \tag {294}
$$

and

$$
\hat {V} _ {\mathrm{ac}} ^ {+} (X) = \frac {1}{2} \left[ \left(1 - c _ {2}\right) + \left(1 + c _ {3}\right) \frac {\gamma - \gamma^ {h}}{1 - \gamma} \right] + \frac {\gamma^ {h}}{1 - \gamma^ {h}} \hat {V} _ {\mathrm{ac}} ^ {+} (\tilde {X}) \tag {295}
$$

$$
= \frac {1}{2 (1 - \gamma)} - \frac {\vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} + \frac {\tilde {\Delta}}{2}.
$$

We can now compute the remaining values as follows:

$$
V ^ {\star} (X) = \frac {1}{2 (1 - \gamma)}, \tag {296}
$$

$$
Q ^ {\star} (X, a = 2) = \frac {(1 - c _ {2}) (1 - \gamma) + \gamma}{2 (1 - \gamma)}, \tag {297}
$$

$$
Q ^ {\bullet} (X, a = 2) = \frac {1 - c _ {2}}{2 (1 - \gamma)}. \tag {298}
$$

Substituting the value of $c _ { 2 }$ yields

$$
V ^ {\star} (X) - V ^ {\bullet} (X) = \frac {\vartheta_ {h} ^ {L}}{1 - \gamma} + \frac {(1 + \gamma^ {h}) \vartheta_ {h} ^ {G}}{(1 - \gamma) (1 - \gamma^ {h})} - \frac {\tilde {\Delta}}{2 (1 - \gamma)}. \tag {299}
$$

2. Case $\vartheta _ { h } ^ { L } < \vartheta _ { h } ^ { G }$

Set

$$
\Delta = 2 \left[ \frac {\vartheta_ {h} ^ {L} + \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} \right] \tag {300}
$$

$$
c _ {2} = 2 \left[ \frac {\vartheta_ {h} ^ {L} + \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} \right] - \tilde {\Delta} > 0, \tag {301}
$$

$$
c _ {3} = c _ {4} = \frac {2 (1 - \gamma) \vartheta_ {h} ^ {L}}{\gamma - \gamma^ {h}} > 0 \tag {302}
$$

where again $\tilde { \Delta }$ is any value that satisfies $0 < \tilde { \Delta } \le \Delta$ .

From the definitions above and the value range of $\begin{array} { r } { \vartheta _ { h } ^ { G } ( \vartheta _ { h } ^ { G } \leq \frac { 1 - \gamma ^ { h } } { 4 } ) } \end{array}$ , it is clear that

$$
c _ {3} = c _ {4} <   c _ {2} \leq \frac {4 \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} \leq \frac {2 (1 - \gamma)}{\gamma - \gamma^ {h}} \leq 1 / 2. \tag {303}
$$

Next, we check the data distribution $\mathcal { D } ^ { \triangle }$ satisfies both optimality variability conditions. With the same argument as the previous case, we can quickly conclude that the global optimality condition is satisfied. We just need to show the remaining local optimality condition. We repeat the procedure from the previous case.

For $s _ { t } = X$ , the local optimality gap is

$$
c _ {3} \frac {\gamma - \gamma^ {h}}{2 (1 - \gamma^ {h})} = \vartheta_ {h} ^ {L}. \tag {304}
$$

For $s _ { t } = \tilde { X }$ the local optimality gap is the same because $c _ { 4 } = c _ { 3 } \mathrm { { : } }$

$$
c _ {4} \frac {\gamma - \gamma^ {h}}{2 (1 - \gamma^ {h})} = \vartheta_ {h} ^ {L}. \tag {305}
$$

This concludes that the second (local) optimality condition is also satisfied for the second case.

Now, we can follow the same procedure as the previous case to show that $\hat { Q } _ { \mathrm { a c } } ^ { + } ( X , a _ { t : t + h } ^ { \triangle } ) ~ -$ $\hat { Q } _ { \mathrm { a c } } ^ { + } ( X , a _ { t : t + h } ^ { \circ } ) > 0$ and $\hat { Q } _ { \mathrm { a c } } ^ { + } ( \tilde { X } , a _ { t : t + h } ^ { \bigtriangleup } ) - \hat { Q } _ { \mathrm { a c } } ^ { + } ( \tilde { X } , a _ { t : t + h } ^ { \circ } ) > 0 .$ .

This concludes that $\pi _ { \mathrm { a c } } ^ { + } ( X ) = \pi _ { \mathrm { a c } } ^ { + } ( \tilde { X } ) = a _ { t : t + h } ^ { \bigtriangleup } = ( 2 , 2 , \cdot \cdot \cdot , 2 , 0 )$ , and thus

$$
\hat {V} _ {\mathrm{ac}} ^ {+} (\tilde {X}) = \frac {1}{2} \left[ \frac {1 - \gamma + (1 + c _ {3}) (\gamma - \gamma^ {h})}{(1 - \gamma) (1 - \gamma^ {h})} \right], \tag {306}
$$

and

$$
\hat {V} _ {\mathrm{ac}} ^ {+} (X) = \frac {1}{2} \left[ \left(1 - c _ {2}\right) + \left(1 + c _ {3}\right) \frac {\gamma - \gamma^ {h}}{1 - \gamma} \right] + \frac {\gamma^ {h}}{1 - \gamma^ {h}} \hat {V} _ {\mathrm{ac}} ^ {+} (\tilde {X}) \tag {307}
$$

$$
= \frac {1}{2 (1 - \gamma)} - \frac {\vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} + \frac {\tilde {\Delta}}{2}.
$$

Repeating the same procedure as the previous case, we obtain

$$
V ^ {\star} (X) - Q ^ {\star} (X, a = 2) = \frac {\vartheta_ {h} ^ {L} + \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} - \tilde {\Delta}, \tag {308}
$$

resulting in an optimality of

$$
V ^ {\star} (X) - V ^ {\bullet} (X) = \frac {\vartheta_ {h} ^ {L} + \vartheta_ {h} ^ {G}}{(1 - \gamma) (1 - \gamma^ {h})} - \frac {\tilde {\Delta}}{1 - \gamma}. \tag {309}
$$

# 3. Sub-optimality $o f V _ { \mathrm { a c } } ^ { + }$

Finally, we can use a pretty crude upper-bound on the actual value of the action chunking policy $\pi _ { \mathrm { a c } } ^ { + }$ (reparameterizing $\tilde { \delta } _ { 2 } = 1 - ( 1 - \delta _ { 2 } ) ^ { h } )$ :

$$
V _ {\mathrm{ac}} ^ {+} (X) \leq (1 - \tilde {\delta} _ {2}) \left[ (1 - c _ {2}) / 2 + \frac {\delta (\gamma - \gamma^ {h})}{2 (1 - \gamma)} + \gamma^ {h} V _ {\mathrm{ac}} ^ {+} (X) \right] + \frac {\tilde {\delta} _ {2}}{1 - \gamma} \tag {310}
$$

$$
\leq \frac {1 - \tilde {\delta} _ {2}}{2 (1 - \gamma^ {h}) (1 - \gamma)} \left[ 1 - \gamma + \delta (\gamma - \gamma^ {h}) \right] + \frac {\tilde {\delta} _ {2}}{1 - \gamma}. \tag {311}
$$

Set $\delta = 1 / 2 ,$ , we have

$$
V _ {\mathrm{ac}} ^ {+} (X) \leq \frac {1 - \tilde {\delta} _ {2}}{2 (1 - \gamma^ {h}) (1 - \gamma)} \left[ 1 - \gamma / 2 - \gamma^ {h} / 2 \right] + \frac {\tilde {\delta} _ {2}}{1 - \gamma}. \tag {312}
$$

We set

$$
\delta_ {2} = 1 - \left[ 1 - \frac {\gamma - \gamma^ {h} - 4 c (1 - \gamma^ {h})}{2 - 3 \gamma^ {h} + \gamma} \right] ^ {1 / h}, \tag {313}
$$

which results in

$$
\tilde {\delta} _ {2} = \frac {\gamma - \gamma^ {h} - 4 c (1 - \gamma^ {h})}{2 - 3 \gamma^ {h} + \gamma}. \tag {314}
$$

It is clear that 0 < δ2 < 1 because c < 4(1−γh) $0 < \delta _ { 2 } < 1$ $\begin{array} { r } { c < \frac { \gamma - \gamma ^ { h } } { 4 ( 1 - \gamma ^ { h } ) } } \end{array}$ γ−γh and 2−3γh+γ < 1. $\textstyle { \frac { \gamma - \gamma ^ { h } } { 2 - 3 \gamma ^ { h } + \gamma } } < 1$ γ−γh

Substituting $\tilde { \delta } _ { 2 }$ in the bound of $V _ { \mathrm { a c } } ^ { + } ( X )$ above, we obtain

$$
V ^ {\star} (X) - V _ {\mathrm{ac}} ^ {+} (X) \geq \frac {c}{1 - \gamma}. \tag {315}
$$

□

Proof of Theorem 6. Let

$$
\Delta = \vartheta_ {h} ^ {L} + \frac {\vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} + \frac {\gamma^ {h} \min (\vartheta_ {h} ^ {G} , \vartheta_ {h} ^ {L})}{1 - \gamma^ {h}}. \tag {316}
$$

Consider the 5-state, 3-action MDP constructed in Lemma 9 and Lemma 10 and a data distribution consisting of a mixture of three data distributions $\mathcal { D } ^ { \star } , \mathcal { D } ^ { \diamond }$ (from Lemma 9) and $\mathcal { D } ^ { \triangle }$ (from Lemma 10):

$$
P _ {\mathcal {D}} = \alpha (1 - \varsigma) P _ {\mathcal {D} ^ {\star}} + \varsigma P _ {\mathcal {D} ^ {\diamond}} + (1 - \alpha) P _ {\mathcal {D} ^ {\triangle}}. \tag {317}
$$

We set α to be any value between 0 and 1 (non-inclusive) and set ς as any positive value such that

$$
\varsigma <   \frac {(\gamma - \gamma^ {h}) - 2 \vartheta_ {h} ^ {G} (1 - \gamma) + 2 \tilde {\Delta} (1 - \gamma) (1 - \gamma^ {h})}{(\gamma - \gamma^ {h}) - 2 \vartheta_ {h} ^ {G} (1 - \gamma)}, \tag {318}
$$

where $\begin{array} { r } { \tilde { \Delta } = \sigma ( 1 - \gamma ) < \operatorname* { m i n } ( \vartheta _ { h } ^ { L } , \vartheta _ { h } ^ { G } ) < \operatorname* { m i n } ( \Delta , \frac { 2 \vartheta _ { h } ^ { G } } { 1 - \gamma ^ { h } } ) } \end{array}$ (satisfying the condition for $\tilde { \Delta }$ in Lemma 10).

The numerator and the denominator are both positive:

$$
(\gamma - \gamma^ {h}) - 2 \vartheta_ {h} ^ {G} (1 - \gamma) + 2 \tilde {\Delta} (1 - \gamma) (1 - \gamma^ {h}) > (\gamma - \gamma^ {h}) - 2 \vartheta_ {h} ^ {G} (1 - \gamma) > 0, \tag {319}
$$

meaning such ς always exists.

Substituting the inequality to the result of Lemma 9 results in

$$
\frac {1 - \gamma + \varsigma (\gamma - \gamma^ {h})}{2 (1 - \gamma^ {h}) (1 - \gamma)} - \frac {\varsigma \vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} <   \frac {1}{2 (1 - \gamma)} - \frac {\vartheta_ {h} ^ {G}}{1 - \gamma^ {h}} + \tilde {\Delta}, \tag {320}
$$

which shows that $\pi _ { \mathrm { a c } } ^ { + }$ will always prefer $a _ { t : t + h } ^ { \triangle }$ over action chunks in ${ \mathcal { D } } ^ { \star }$ and $\mathcal { D } ^ { \diamond }$

This means that the value $\hat { V } _ { \mathrm { a c } } ^ { + }$ and the action chunking policy $\pi _ { \mathrm { a c } } ^ { + }$ we learn from D coincides with these of $\mathcal { D } ^ { \triangle }$ , allowing us to directly use the results of Lemma 10.

Thus, we can conclude that

$$
V ^ {\star} (s _ {t}) - V _ {\mathrm{ac}} ^ {+} (s _ {t}) \geq \frac {c}{1 - \gamma}, \tag {321}
$$

and

$$
V ^ {\star} (X) - V ^ {\bullet} (X) = \frac {\Delta - \tilde {\Delta}}{1 - \gamma} = \frac {\vartheta_ {h} ^ {L}}{1 - \gamma} + \frac {\vartheta_ {h} ^ {G}}{(1 - \gamma) (1 - \gamma^ {h})} + \frac {\gamma^ {h} \min (\vartheta_ {h} ^ {L} , \vartheta_ {h} ^ {G})}{(1 - \gamma) (1 - \gamma^ {h})} - \sigma , \tag {322}
$$

as desired.

![](images/db72dea1060cc95cd0a4d43aba7b9924b8b8822d2f8553dbdb0972f46f1e0398.jpg)

# F.14 PROOF OF PROPOSITION 5

Proposition 5 (Worst-case analysis of n-step return backup) For any $n ~ \in ~ \mathbb { N } ^ { + } , ~ \tilde { \delta } _ { n } ~ \in$ $( 0 , \gamma - \gamma ^ { n } )$ and $\sigma \in \left( 0 , \tilde { \delta } _ { n } / ( 1 - \gamma ) \right)$ , there exists an MDP M, and a $\tilde { \delta } _ { n }$ -optimal data distribution D with supp $( P _ { \mathcal { D } } ( s _ { t } , a _ { t } ) ) \supseteq \operatorname { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } , a _ { t } ) )$ such that for some $s \in \operatorname { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V _ {\mathrm{ac}} ^ {+} (s) - V _ {n} ^ {+} (s) = \frac {\tilde {\delta} _ {n}}{1 - \gamma} - \sigma , \tag {43}
$$

and for all $s \in \operatorname { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ ,

$$
V ^ {\star} (s) = V _ {\mathrm{ac}} ^ {+} (s). \tag {44}
$$

Proof. Consider an MDP in Figure 14. Let D be the data collected by the following policy:

$$
\pi (a = 0 \mid X) = \pi (a = 1 \mid X) = 1 / 2, \tag {323}
$$

$$
\pi (a = 0 \mid Y) = \alpha , \tag {324}
$$

$$
\pi (a = 1 \mid Y) = 1 - \alpha . \tag {325}
$$

![](images/d2d37fb24b8ed13311ac3ab4bf83bd6712cf78af6b39543e98c39ccadebaf854.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["X"] -->|a=0\nr=1-c₂| B["Z"]
    A -->|a=1\nr=1| C["Y"]
    C -->|a=0\nr=1| D["a=0\nr=1"]
    A -->|a=1\nr=1| C
    C -->|a=1\nr=1| A
    D -->|a=0\nr=1-c₂| B
```
</details>

Figure 14: An MDP where the learned action chunking policies are optimal and the learned n-step return policies can be arbitrarily sub-optimal.

It is clear that the optimal policy is $\pi ^ { \star } ( X ) = 1 , \pi ^ { \star } ( Y ) = 0$ .

Since the dynamics are deterministic, the data distribution D is strongly open-loop consistent and thus by Theorem $3 , V _ { \mathrm { a c } } ( s _ { t } ) = V ^ { \star } ( s _ { t } )$ for all $s _ { t } \in \mathrm { s u p p } ( P _ { \mathcal { D } ^ { \star } } ( s _ { t } ) )$ .

To make sure the data distribution satisfies the ˜δn-optimal condition, we set

$$
\alpha = 1 - \frac {\tilde {\delta} _ {n}}{\gamma - \gamma^ {n}}, \quad c _ {2} = \tilde {\delta} _ {n} - (1 - \gamma) \sigma . \tag {326}
$$

It is clear that at state $Y , \pi _ { n } ^ { + } ( Y ) = 0$ . We can then calculate the optimality gap for $\hat { Q } _ { n } ^ { + }$ exactly as follows:

$$
V ^ {\star} (X) - \hat {Q} _ {n} ^ {+} (X, a = 0) = \frac {c _ {2}}{1 - \gamma} = \frac {\tilde {\delta} _ {n}}{1 - \gamma} - \sigma , \tag {327}
$$

$$
V ^ {\star} (X) - \hat {Q} _ {n} ^ {+} (X, a = 1) = \frac {1}{1 - \gamma} - \frac {1 - \gamma + \alpha (\gamma - \gamma^ {n})}{(1 - \gamma) (1 - \gamma^ {n})} = \frac {\tilde {\delta} _ {n}}{1 - \gamma}. \tag {328}
$$

Since $\sigma > 0 , \pi _ { n } ^ { + } ( X ) = 0$ . Now, we can compute $V _ { n } ( X )$ as follows:

$$
V ^ {\star} (X) - V _ {n} (X) = \frac {c _ {2}}{1 - \gamma} = \frac {\tilde {\delta} _ {n}}{1 - \gamma} - \sigma , \tag {329}
$$

as desired.

![](images/f45c21c34719f6393c22a0f0ccf6f90fbdfbd6b74f81c67f400de35873be694a.jpg)

# F.15 PROOF OF PROPOSITION 4

Proposition 4 (Deterministic Dynamics are Weakly Open-loop Consistent) If a transition dynamics M is ε-deterministic, then any data D collected from M is weakly εh-open-loop consistent with respect to M for any $h \in \mathbb { N } ^ { + }$ as long as $\varepsilon _ { h } \geq 3 ( 1 - ( 1 - \varepsilon ) ^ { h - 1 } )$ .

Proof. Since T is ε-deterministic, it can be represented as $T ( \cdot \mid s , a ) = ( 1 - \varepsilon ) \delta _ { f ( s , a ) } + \varepsilon \tilde { T } ( \cdot \mid s , a )$ for some $f : { \mathcal { S } } \times { \mathcal { A } } \to { \mathcal { S } }$ and $\tilde { T } : \mathcal { S } \times \mathcal { A }  \Delta _ { \mathcal { S } }$ . Let $f ( s , a _ { 1 } , \cdots , a _ { h } ) = f ( \cdot \cdot \cdot f ( f ( s , a _ { 1 } ) , a _ { 2 } ) \cdot \cdot \cdot a _ { h } )$

Let $I \in \{ 0 , 1 \}$ a binary indicator variable that is 1 if and only if

$$
s _ {t + k + 1} = f \left(s _ {t + k}, a _ {t + k}\right), \forall k \in \{0, 1, 2, \dots , h - 1 \} \tag {330}
$$

Intuitively $I = 1$ when the trajectory is generated deterministically until but not including the last state $s _ { h }$ in the trajectory chunk.

From the fact that T is ε-deterministic, we know that

$$
P _ {\mathcal {D}} (I _ {h} = 1) \geq (1 - \varepsilon) ^ {h - 1} \tag {331}
$$

We also have

$$
P _ {\mathcal {D}} \left(a _ {t: t + h} \mid s _ {t}\right) = P _ {\mathcal {D}} \left(I _ {h} = 1\right) P _ {\mathcal {D}} \left(a _ {t: t + h} \mid s _ {t}, I _ {h} = 1\right) + P _ {\mathcal {D}} \left(I _ {h} = 0\right) P _ {\mathcal {D}} \left(a _ {t: t + h} \mid s _ {t}, I = 0\right) \tag {332}
$$

Then we have

$$
D _ {\mathrm{TV}} (P _ {\mathcal {D}} (a _ {t: t + h} \mid s _ {t}) \parallel P _ {\mathcal {D}} (a _ {t: t + h} \mid s _ {t}, I _ {h} = 1)) \leq (1 - (1 - \varepsilon) ^ {h - 1}) \tag {333}
$$

If we transform each distribution of $\scriptstyle a _ { t : t + h }$ deterministically by $f ( s _ { t } , \cdot )$ , by data processing inequality (DPI; Lemma 4), we have

$$
D _ {\mathrm{TV}} \left(\mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t})} \left[ \delta_ {f (s _ {t}, a _ {t: t + h})} \right] \right\| \mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t}, I _ {h} = 1)} \left[ \delta_ {f (s _ {t}, a _ {t: t + h})} \right]) \leq (1 - (1 - \varepsilon) ^ {h - 1}) \tag {334}
$$

Similarly, we have

$$
D _ {\mathrm{TV}} (P _ {\mathcal {D}} (a _ {t: t + h + 1} \mid s _ {t}) \parallel P _ {\mathcal {D}} (a _ {t: t + h + 1} \mid s _ {t}, I _ {h + 1} = 1)) \leq (1 - (1 - \varepsilon) ^ {h}) \tag {335}
$$

which can be also deterministically transformed by taking $a _ { t : t + h + 1 } \mapsto ( f ( s _ { t } , \cdot ) , a _ { t + h } )$ (again with DPI, Lemma 4) to obtain

$$
D _ {\mathrm{TV}} \left(\mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t})} \left[ \pi_ {\mathcal {D}} ^ {\circ} (a _ {t + h} \mid s _ {t}, a _ {t: t + h}) \mathbb {I} _ {f (s _ {t}, a _ {t: t + h})} \right] \right\| \tag {336}
$$

$$
\mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t}, I _ {h + 1} = 1)} \left[ \pi_ {\mathcal {D}} ^ {\circ} (a _ {t + h} \mid s _ {t}, a _ {t: t + h}, I _ {h + 1} = 1) \mathbb {I} _ {f (s _ {t}, a _ {t: t + h})} \right] \Bigg) \leq (1 - (1 - \varepsilon) ^ {h})
$$

Now, if we analyze the distribution of $s _ { t + h }$ subject to the open-loop execution of the action sequence from $P _ { \mathcal { D } } ( \cdot \mid s _ { t } )$ and break it up into the deterministic and the non-deterministic case, we get

$$
\mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t})} \left[ T _ {a _ {t: t + h}} (\cdot | s _ {t}) \right] = P _ {T} (I = 1) \mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t})} \left[ \delta_ {f (s _ {t}, a _ {t: t + h})} \right] + \tag {337}
$$

$$
P _ {T} (I = 0) \mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t})} \left[ T _ {a _ {t: t + h}} (\cdot | s _ {t}, I _ {h} = 0) \right]
$$

Note that $P _ { T } ( I = 1 )$ denotes the probability that an open-loop executed trajectory using $\begin{array} { r }  a _ { t : t + h } \sim \begin{array} { r } { \begin{array} { r l r } \end{array} } \end{array} \end{array}$ $P _ { \mathcal { D } } ( \cdot \ | \ s _ { t } )$ is deterministic. This is different from $\bar { P } _ { \mathcal { D } } ( I _ { h } \bar { = } \ 1 )$ because the latter is based on $P _ { \mathcal { D } } ( s _ { t : t + h + 1 } , a _ { t : t + h } )$ whereas $P _ { T } ( I _ { h } = 1 )$ is based on the open-loop trajectory distribution: $P _ { \mathcal { D } } ( \cdot \mid$ $\begin{array} { r } { s _ { t } \big ) \prod _ { k = 0 } ^ { h - 1 } T \big ( s _ { t + k } \ | \ s _ { t } , a _ { t : t + k } \big ) } \end{array}$ . They both admit the same lower bound of $2 ( 1 - ( 1 - \varepsilon ) ^ { h - 1 } )$ .

Therefore,

$$
D _ {\mathrm{TV}} \left(\mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t})} \left[ T _ {a _ {t: t + h}} (\cdot \mid s _ {t}) \right] \| \mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t})} \left[ \delta_ {f (s _ {t}, a _ {t: t + h})} \right]\right) \leq (1 - (1 - \varepsilon) ^ {h - 1}) \tag {338}
$$

Similarly for the state-action case, we can multiply both side by the same conditional distribution $\pi _ { \mathcal { D } } ^ { \circ } \left( a _ { t + h } ~ | ~ s _ { t } , a _ { t : t + h } \right)$ which preserves the TV bound. For the left-hand side, we have

$$
P _ {\mathcal {D}} ^ {\circ} \left(s _ {t + h}, a _ {t + h} \mid s _ {t}\right) = \mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t})} \left[ \pi_ {\mathcal {D}} ^ {\circ} \left(a _ {t + h} \mid s _ {t}, a _ {t: t + h}\right) T _ {a _ {t: t + h}} \left(s _ {t + h} \mid s _ {t}\right) \right] \tag {339}
$$

Therefore, we get

$$
\begin{array}{l} D _ {\mathrm{TV}} \left(P _ {\mathcal {D}} ^ {\circ} \left(s _ {t + h}, a _ {t + h} \mid s _ {t}\right) \| \mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t})} \left[ \pi_ {\mathcal {D}} ^ {\circ} \left(a _ {t + h} \mid s _ {t}, a _ {t: t + h}\right) \mathbb {I} _ {f \left(s _ {t}, a _ {t: t + h}\right)} \right]\right) \tag {340} \\ \leq \left(1 - (1 - \varepsilon) ^ {h - 1}\right) \\ \end{array}
$$

We also have

$$
P _ {\mathcal {D}} \left(s _ {t + h} \mid s _ {t}\right) = (1 - \varepsilon) ^ {h - 1} P _ {\mathcal {D}} \left(s _ {t + h} \mid s _ {t}, I = 1\right) + (1 - (1 - \varepsilon) ^ {h - 1}) P _ {\mathcal {D}} \left(s _ {t + h} \mid s _ {t}, I _ {h} = 0\right) \tag {341}
$$

Similarly, we have

$$
\begin{array}{l} D _ {\mathrm{TV}} \left(P _ {\mathcal {D}} \left(s _ {t + h} \mid s _ {t}\right) \| P _ {\mathcal {D}} \left(s _ {t + h} \mid s _ {t}, I _ {h} = 1\right)\right) \\ P _ {\text {TV}} \left(P _ {\mathcal {D}} \left(s _ {t + h} \mid s _ {t}\right) \| P _ {\mathcal {D}} \left[ f _ {t - h} - [ f _ {t - h} ]\right) <   (1 - (1 - h) ^ {h - 1}) \right. \end{array} \tag {342}
$$

$$
= D _ {\mathrm{TV}} \left(P _ {\mathcal {D}} \left(s _ {t + h} \mid s _ {t}\right) \| \mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t}, I _ {h} = 1)} \left[ \delta_ {f \left(s _ {t}, a _ {t: t + h}\right)} \right]\right) \leq (1 - (1 - \varepsilon) ^ {h - 1})
$$

For state-action, we can also get

$$
P _ {\mathcal {D}} \left(s _ {t + h}, a _ {t + h} \mid s _ {t}\right) = (1 - \varepsilon) ^ {h} P _ {\mathcal {D}} \left(s _ {t + h}, a _ {t + h} \mid s _ {t}, I _ {h + 1} = 1\right) \tag {343}
$$

$$
+ (1 - (1 - \varepsilon) ^ {h}) P _ {\mathcal {D}} \left(s _ {t + h}, a _ {t + h} \mid s _ {t}, I _ {h + 1} = 0\right)
$$

which can be turned into the TV distance bound:

$$
\begin{array}{l} D _ {\mathrm{TV}} (P _ {\mathcal {D}} (s _ {t + h}, a _ {t + h} \mid s _ {t}) \| P _ {\mathcal {D}} (s _ {t + h}, a _ {t + h} \mid s _ {t}, I _ {h + 1} = 1)) \\ = D _ {\mathrm{TV}} \left(P _ {\mathcal {D}} \left(s _ {t + h}, a _ {t + h} \mid s _ {t}\right) \right\| \tag {344} \\ \end{array}
$$

$$
\mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t}, I _ {h + 1} = 1)} \left[ \pi_ {\mathcal {D}} ^ {\circ} (a _ {t + h} \mid s _ {t}, a _ {t: t + h}, I _ {h + 1} = 1) \mathbb {I} _ {f (s _ {t}, a _ {t: t + h})} \right]
$$

$$
\leq (1 - (1 - \varepsilon) ^ {h})
$$

Connecting all three total variation inequality (Equations (334), (338) and (342)) together, we get

$$
D _ {\mathrm{TV}} \left(P _ {\mathcal {D}} \left(s _ {t + h} \mid s _ {t}\right) \| \mathbb {E} _ {a _ {t: t + h} \sim P _ {\mathcal {D}} (\cdot | s _ {t})} \left[ T _ {a _ {t: t + h}} (\cdot \mid s _ {t}) \right]\right) \leq 3 (1 - (1 - \varepsilon) ^ {h - 1}) \leq \varepsilon_ {h} \tag {345}
$$

Connecting all three total variable inequality for state-action (Equations (336), (339) and (344)) together, we get

$$
\begin{array}{l} D _ {\mathrm{TV}} (P _ {\mathcal {D}} ^ {\circ} (s _ {t + h - 1}, a _ {t + h - 1} \mid s _ {t}) \parallel P _ {\mathcal {D}} (s _ {t + h}, a _ {t + h} \mid s _ {t})) \leq 3 - 2 (1 - \varepsilon) ^ {h - 1} - (1 - \varepsilon) ^ {h - 2} \\ \leq 3 (1 - (1 - \varepsilon) ^ {h - 1}) \tag {346} \\ \leq \varepsilon_ {h} \\ \end{array}
$$

Therefore, D is $\varepsilon _ { h }$ -open-loop consistent as desired.

![](images/99ff57e378c7104a760fb6d6108b2cc80a58bba05eb85365fb43e795a4574b86.jpg)