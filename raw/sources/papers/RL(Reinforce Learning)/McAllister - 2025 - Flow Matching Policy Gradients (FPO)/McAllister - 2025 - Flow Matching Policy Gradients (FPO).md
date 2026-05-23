# Flow Matching Policy Gradients

David McAllister1∗ Songwei Ge1∗ Brent Yi1∗ Chung Min Kim1 Ethan Weber1 Hongsuk Choi1 Haiwen Feng1,2 Angjoo Kanazawa1 1 UC Berkeley 2 Max Planck Institute for Intelligent Systems

# Abstract

Flow-based generative models, including diffusion models, excel at modeling continuous distributions in high-dimensional spaces. In this work, we introduce Flow Policy Optimization (FPO), a simple on-policy reinforcement learning algorithm that brings flow matching into the policy gradient framework. FPO casts policy optimization as maximizing an advantage-weighted ratio computed from the conditional flow matching loss, in a manner compatible with the popular PPO-clip framework. It sidesteps the need for exact likelihood computation while preserving the generative capabilities of flow-based models. Unlike prior approaches for diffusion-based reinforcement learning that bind training to a specific sampling method, FPO is agnostic to the choice of diffusion or flow integration at both training and inference time. We show that FPO can train diffusion-style policies from scratch in a variety of continuous control tasks. We find that flow-based models can capture multimodal action distributions and achieve higher performance than Gaussian policies, particularly in under-conditioned settings. For an overview of FPO’s key ideas, see our accompanying blog post: flowreinforce.github.io

# 1 Introduction

Flow-based generative models—particularly diffusion models—have emerged as powerful tools for generative modeling across the domains of images [1–3], videos [4–6], speech [7], audio [8], robotics [9], and molecular dynamics [10]. In parallel, reinforcement learning (RL) has proven to be effective for optimizing neural networks with non-differentiable objectives, and is widely used as a post-training strategy for aligning foundation models with task-specific goals [11, 12].

In this work, we introduce Flow Policy Optimization (FPO), a policy gradient algorithm for optimizing flow-based generative models. FPO reframes policy optimization as maximizing an advantageweighted ratio computed from the conditional flow matching (CFM) objective [13]. Intuitively, FPO shapes probability flow to transform Gaussian noise into high-reward actions by reinforcing its experience using flow matching. The method is simple to implement and can be readily integrated into standard techniques for stochastic policy optimization. We use a PPO-inspired surrogate objective for our experiments, which trains stably and serves as a drop-in replacement for Gaussian policies.

FPO offers several key advantages. It sidesteps the complex likelihood calculations typically associated with flow-based models, instead using the flow matching loss as a surrogate for log-likelihood in the policy gradient. This aligns the objective directly with increasing the evidence lower bound of high-reward actions. Unlike previous methods that reframe the denoising process as an MDP, binding the training to specific sampling methods and extending the credit-assignment horizon, FPO treats the sampling procedure as a black box during rollouts. This distinction allows for flexible integration with any sampling approach—whether deterministic or stochastic, first- or higher-order, and with any number of integration steps during training or inference.

We theoretically analyze FPO’s correctness and empirically validate its performance across a diverse set of tasks. These include a GridWorld environment, 10 continuous control tasks from MuJoCo Playground [14], and high-dimensional humanoid control—all trained from scratch. FPO demonstrates robustness across tasks, enabling effective training of flow-based policies in high-dimensional domains. We probe flow policies learned in the toy GridWorld environment and find that on states with multiple possible optimal actions, it learns multimodal action distributions, unlike Gaussian policies. On humanoid control tasks, we show that the expressivity of flow matching enables singlestage training of under-conditioned control policies, where only root-level commands are provided. In contrast, standard Gaussian policies struggle to learn viable walking behaviors in such cases. This highlights the practical benefits of the more powerful distribution modeling enabled by FPO. Finally, we discuss limitations and future work.

# 2 Related Work

Policy Gradients. We study on-policy reinforcement learning, where a parameterized policy is optimized to maximize cumulative reward in a provided environment. This is commonly solved with policy gradient techniques, which bypass the need for differentiable environment rewards by weighting action log-probabilities with observed rewards or advantages [15–23]. Policy gradient methods are central in learning policies for general continuous control tasks [24, 25], robot locomotion [26–29] and manipulation [30–33]. They have also been adopted increasingly for searching through and refining prior distributions in pretrained generative models. This has proven effective for alignment with human preferences [34, 35] and improving reasoning using verifiable rewards [36, 37].

In this work, we propose a simple algorithm for training flow-based generative policies, such as diffusion models, under the policy gradient framework. By leveraging recent insights from flow matching [13], we train policies that can represent richer distributions than the diagonal Gaussians that are most frequently used for reinforcement learning for continuous control [26–29, 32, 33], while remaining compatible with standard actor-critic training techniques.

Diffusion Models. Diffusion models are powerful tools for modeling complex continuous distributions and have achieved remarkable success across a wide range of domains. These models have become the predominant approach for generating images [38–41], videos [42–44, 4], audio [7, 45, 46, 8], and more recently, robot actions [9, 47, 48]. In these applications, diffusion models aim to sample from a data distribution of interest, whether scraped from the internet or collected through human teleoperation.

Flow matching [13] simplifies and generalizes the diffusion model framework. It learns a vector field that transports samples from a tractable prior distribution to the target data distribution. The conditional flow matching (CFM) objective trains the model to denoise data that has been perturbed with Gaussian noise. Given data x and noise $\epsilon \in \mathcal { N } ( 0 , I )$ , the CFM objective can be expressed as:

$$
\mathcal {L} _ {\mathrm{CFM}, \theta} = \mathbb {E} _ {\tau , q (x), p _ {\tau} (x _ {\tau} | x)} \| \hat {v} _ {\theta} (x _ {\tau}, \tau) - u (x _ {\tau}, \tau \mid x) \| _ {2} ^ {2}, \tag {1}
$$

where $x _ { \tau } = \alpha _ { \tau } x + \sigma _ { \tau } \epsilon$ represents the partially noised sample at flow step τ , an interpolation of noise and data with a schedule defined by hyperparameters $\alpha _ { \tau }$ and $\sigma _ { \tau } . \hat { v } _ { \boldsymbol { \theta } } ( x _ { \tau } , \tau )$ is the model’s estimate of the velocity to the original data, and $u ( x _ { \tau } , \tau \mid x )$ is the conditional flow $x - \epsilon .$ . The model can also estimate the denoised sample x or noise component ϵ as the optimization target instead of velocity. The learned velocity field is a continuous mapping that transports samples from a simple, tractable distribution (e.g. Gaussian noise) to the training data distribution through ODE integration.

Optimizing likelihoods directly through flow models is possible, but requires divergence estimation [49] and is computationally prohibitive. Instead, flow matching optimizes variational lower bounds of the likelihood with the simple denoising loss above. In this work, we leverage flow matching directly within the policy gradient formulation. This approach trains diffusion models from rewards without prohibitively expensive likelihood computations.

Diffusion Policies. Diffusion-based policies have shown promising results in robotics and decisionmaking applications [50, 51, 47]. Most existing approaches train these models via behavior cloning [52, 9], where the policy is supervised to imitate expert trajectories without using reward feedback. Motivated by the strong generative capabilities of diffusion and flow-based models, several works have explored using reinforcement learning to fine-tune diffusion models, particularly in domains like text-to-image generation [53–55].

Recent work by Psenka et al. [56] explores off-policy training of diffusion policies via Q-score matching. While off-policy reinforcement learning continues to make progress [57, 58], on-policy methods dominate practical applications today. Methods like DDPO [54], DPPO [59], and Flow-GRPO [55] adopt on-policy policy gradient methods by treating initial noise values as observations from the environment, framing the denoising process as a Markov decision process, and training each step as a Gaussian policy using PPO. Our approach differs by directly integrating the conditional flow matching (CFM) objective into a PPO-like framework, maintaining the structure of the standard diffusion forward and reverse processes. Since FPO integrates flow matching as its fundamental primitive, it is agnostic to the choice of sampling method during both training and inference, just like flow matching for behavior cloning.

# 3 Flow Matching Policy Gradients

# 3.1 Policy Gradients and PPO

The goal of reinforcement learning is to learn a policy $\pi _ { \theta }$ that maximizes expected return in a provided environment. At each iteration of online reinforcement learning, the policy is rolled out to collect batches of observation, action, and reward tuples $\left( o _ { t } , a _ { t } , r _ { t } \right)$ for each environment timestep t. These rollouts can used in the policy gradient objective [15] to increase likelihood of actions that result in higher rewards:

$$
\max _ {\theta} \mathbb {E} _ {a _ {t} \sim \pi_ {\theta} (a _ {t} | o _ {t})} \left[ \log \pi_ {\theta} (a _ {t} \mid o _ {t}) \hat {A} _ {t} \right], \tag {2}
$$

where $\hat { A } _ { t }$ is an advantage estimated from the rollout’s rewards $r _ { t }$ and a learned value function [60].

The vanilla policy gradient is valid only locally around the current policy parameters. Large updates can lead to policy collapse or unstable learning. To address this, PPO [20] incorporates a trust region by clipping the likelihood ratio:

$$
\max _ {\theta} \mathbb {E} _ {a _ {t} \sim \pi_ {\theta_ {\text {old}}} (a _ {t} | o _ {t})} \left[ \min \left(r (\theta) \hat {A} _ {t}, \operatorname{clip} (r (\theta), 1 - \varepsilon^ {\text {clip}}, 1 + \varepsilon^ {\text {clip}}) \hat {A} _ {t}\right) \right], \tag {3}
$$

where $\varepsilon ^ { \mathrm { c l i p } }$ is a tunable threshold and $r ( \theta )$ is the ratio between current and old action likelihoods:

$$
r (\theta) = \frac {\pi_ {\theta} (a _ {t} \mid o _ {t})}{\pi_ {\text { old }} (a _ {t} \mid o _ {t})}. \tag {4}
$$

PPO is popular choice for on-policy reinforcement learning because of its stability, simplicity, and performance. Like the standard policy gradient, however, it requires exact likelihoods for sampled actions. These quantities are tractable for simple Gaussian or categorical action spaces, but computationally prohibitive to estimate for flow matching and diffusion models.

# 3.2 Flow Policy Optimization

We introduce Flow Policy Optimization (FPO), an online reinforcement learning algorithm for policies represented as flow models $\hat { v } _ { \theta }$ . There are two key differences in practice from Gaussian PPO. During rollouts, a flow model transforms random noise into actions via a sequence of learned transformations [13], enabling much more expressive policies than those used in standard PPO. Also, to update the policy, the Gaussian likelihoods are replaced with a transformed flow matching loss.

Instead of updating exact likelihoods, we propose a proxy $\hat { r } ^ { \mathrm { F P O } }$ for the log-likelihood ratio. FPO’s overall objective is the same as Equation 3, but with the ratio substituted:

$$
\max _ {\theta} \mathbb {E} _ {a _ {t} \sim \pi_ {\theta} (a _ {t} | o _ {t})} \left[ \min \left(\hat {r} ^ {\mathrm{FPO}} (\theta) \hat {A} _ {t}, \operatorname{clip} \left(\hat {r} ^ {\mathrm{FPO}} (\theta), 1 - \varepsilon^ {\text { clip }}, 1 + \varepsilon^ {\text { clip }}\right) \hat {A} _ {t}\right) \right]. \tag {5}
$$

Intuitively, FPO’s goal is to steer the policy’s probability flow toward high-return behavior. Instead of computing likelihoods, we construct a simple ratio estimate using standard flow matching losses:

$$
\hat {r} ^ {\mathrm{FPO}} (\theta) = \exp \left(\hat {\mathcal {L}} _ {\mathrm{CFM}, \theta_ {\text {old}}} \left(a _ {t}; o _ {t}\right) - \hat {\mathcal {L}} _ {\mathrm{CFM}, \theta} \left(a _ {t}; o _ {t}\right)\right), \tag {6}
$$

which, as we will discuss, can be derived from optimizing the evidence lower bound.

For a given action and observation pair, $\hat { \mathcal { L } } _ { \mathrm { C F M } , \theta } ( a _ { t } ; o _ { t } )$ is an estimate of the per-sample conditional flow matching loss $\mathcal { L } _ { \mathrm { C F M } , \theta } ( a _ { t } ; o _ { t } )$ :

$$
\hat {\mathcal {L}} _ {\mathrm{CFM}, \theta} (a _ {t}; o _ {t}) = \frac {1}{N _ {\mathrm{mc}}} \sum_ {i} ^ {N _ {\mathrm{mc}}} \ell_ {\theta} (\tau_ {i}, \epsilon_ {i}) \tag {7}
$$

$$
\ell_ {\theta} \left(\tau_ {i}, \epsilon_ {i}\right) = \left| \left| \hat {v} _ {\theta} \left(a _ {t} ^ {\tau_ {i}}, \tau_ {i}; o _ {t}\right) - \left(a _ {t} - \epsilon_ {i}\right) \right| \right| _ {2} ^ {2} \tag {8}
$$

$$
a _ {t} ^ {\tau_ {i}} = \alpha_ {\tau_ {i}} a _ {t} + \sigma_ {\tau_ {i}} \epsilon_ {i}, \tag {9}
$$

where we denote flow timesteps with τ and environment timesteps with t. We include both timesteps in $a _ { t } ^ { \tau }$ , which represents an action at rollout time t with noise level τ following Equation 1. We use the same $\epsilon _ { i } \sim N ( 0 , I ) $ and $\tau _ { i } \in [ 0 , 1 ]$ samples between $\hat { \mathcal { L } } _ { \mathrm { C F M } , \theta _ { \mathrm { o l d } } }$ and $\hat { \mathcal { L } } _ { \mathrm { C F M } , \theta }$ .

Properties. FPO’s ratio estimate in Equation 6 serves as a drop-in replacement for the PPO likelihood ratio. FPO therefore inherits compatibility with advantage estimation methods like GAE [60] and GRPO [23]. Without loss of generality, it is also compatible with flow and diffusion implementations based on estimating noise ϵ [38] or clean action $a _ { t }$ [1], which can be reweighted for mathematical equivalence to $\mathcal { L } _ { \boldsymbol { \theta } , \mathrm { C F M } } \left[ 6 1 , 1 3 \right]$ . We leverage this property in our FPO ratio derivation below.

# 3.3 FPO Surrogate Objective

Exact likelihood is computationally expensive even to estimate in flow-based models. Instead, it is common to optimize the evidence lower bound (ELBO) as a proxy for log-likelihood:

$$
\mathrm{ELBO} _ {\theta} (a _ {t} \mid o _ {t}) = \log \pi_ {\theta} (a _ {t} \mid o _ {t}) - \mathcal {D} _ {\theta} ^ {\mathrm{KL}}, \tag {10}
$$

where $\mathcal { D } _ { \theta } ^ { \mathrm { K L } }$ is the KL gap between the ELBO and true log-likelihood and $\pi _ { \theta }$ is the distribution captured by sampling from the flow model. Both flow matching and diffusion models optimize the ELBO using a conditional flow matching loss, a simple MSE denoising objective [62, 13]. The FPO ratio (Equation 11) leverages the fact that flow models can be trained via ELBO objectives. Specifically, we compute the ratio of ELBOs under the current and old policies:

$$
r ^ {\mathrm{FPO}} (\theta) = \frac {\exp (\mathrm{ELBO} _ {\theta} (a _ {t} \mid o _ {t}))}{\exp (\mathrm{ELBO} _ {\theta_ {\mathrm{old}}} (a _ {t} \mid o _ {t}))}. \tag {11}
$$

Decomposing this ratio reveals a scaled variant of the true likelihood ratio (Equation 4):

$$
r ^ {\mathrm{FPO}} (\theta) = \underbrace {\frac {\pi_ {\theta} (a _ {t} \mid o _ {t})}{\pi_ {\theta_ {\text { old}}} (a _ {t} \mid o _ {t})}} _ {\text { Likelihood }} \underbrace {\frac {\exp (\mathcal {D} _ {\theta_ {\text { old }}} ^ {\mathrm{KL}})}{\exp (\mathcal {D} _ {\theta} ^ {\mathrm{KL}})}} _ {\text { Inv.   KL   Gap }}. \tag {12}
$$

Here, the ratio decomposes into the standard likelihood ratio and an inverse correction term involving the KL gap. Maximizing this ratio therefore increases the modeled likelihood while reducing the KL gap—both of which are beneficial for policy optimization. The former encourages the policy to favor actions with positive advantage, while the latter tightens the approximation to the true log-likelihood.

# 3.4 Estimating the FPO Ratio with Flow Matching

We estimate the FPO ratio using the flow matching objective directly, which follows from the relationship between the weighted denoising loss $\mathcal { L } _ { \theta } ^ { w }$ and the ELBO established by Kingma and Gao [63]. ${ \mathcal { L } } _ { \theta } ^ { w }$ is a more general form of the flow matching and denoising diffusion loss that parameterizes the model as predicting $\hat { \epsilon } _ { \theta }$ , an estimate of the true noise ϵ present in the model input.

The weighted denoising loss ${ \mathcal { L } } _ { \theta } ^ { w }$ for a clean action $a _ { t }$ takes the form:

$$
\mathcal {L} _ {\theta} ^ {w} (a _ {t}) = \frac {1}{2} \mathbb {E} _ {\tau \sim \mathcal {U} (0, 1), \epsilon \sim \mathcal {N} (0, I)} \left[ w (\lambda_ {\tau}) \cdot \left(- \frac {d \lambda}{d \tau}\right) \cdot \| \hat {\epsilon} _ {\theta} (a _ {t} ^ {\tau}; \lambda_ {\tau}) - \epsilon \| _ {2} ^ {2} \right], \tag {13}
$$

where w is a choice of weighting and $\lambda _ { \tau }$ represents the log-SNR at noise level τ . We estimate this value with Monte Carlo draws of timestep τ and noise ϵ:

$$
\ell_ {\theta} ^ {w} (\tau , \epsilon) = \frac {1}{2} w \left(\lambda_ {\tau}\right) \cdot \left(- \frac {d \lambda}{d \tau}\right) \cdot \| \hat {\epsilon} _ {\theta} \left(a _ {t} ^ {\tau}; \lambda_ {\tau}\right) - \epsilon \| _ {2} ^ {2}. \tag {14}
$$

The choice of weighting w incorporates the conditional flow matching loss and standard diffusion loss as specific cases of a more general family $\mathcal { L } _ { \theta } ^ { w } ( a _ { t } )$ .

We focus here on the constant weight case $w ( \lambda _ { \tau } ) = 1$ (diffusion schedule), which yields the simplest theoretical connection. Similar results hold for many popular schedules, including optimal transport and variance preserving schedules [13]. Please see the supplementary material for details.

For the diffusion schedule, [63] proves that:

$$
\mathcal {L} _ {\theta} ^ {w} (a _ {t}) = - \mathrm{ELBO} _ {\theta} (a _ {t}) + c, \tag {15}
$$

where c is a constant w.r.t θ. Geometrically, minimizing ${ \mathcal { L } } _ { \theta } ^ { w } ( a _ { t } )$ points the flow more toward $a _ { t }$ . Minimizing ${ \mathcal { L } } _ { \theta } ^ { w }$ also maximizes the ELBO (Eq. 10) and thus the likelihood of $a _ { t }$ , so flowing toward a specific action makes it more likely. This intuition aligns naturally with the policy gradient objective: we want to increase the probability of high-advantage actions. By redirecting flow toward such actions (i.e., minimizing their diffusion loss), we make them more likely under the learned policy.

Using this relationship, we express the FPO ratio (Eq. 11) in terms of the flow matching objective:

$$
r _ {\theta} ^ {\mathrm{FPO}} = \frac {\exp (\mathrm{ELBO} _ {\theta} (a _ {t} | o _ {t}))}{\exp (\mathrm{ELBO} _ {\theta_ {\mathrm{old}}} (a _ {t} | o _ {t}))} = \exp (\mathcal {L} _ {\theta_ {\mathrm{old}}} ^ {w} (a _ {t}) - \mathcal {L} _ {\theta} ^ {w} (a _ {t})), \tag {16}
$$

where $\mathcal { L } _ { \theta } ^ { w }$ , as per Equation 7, can be estimated by averaging over $N _ { \mathrm { m c } }$ draws of $( \tau , \epsilon )$ . We find the sample count $N _ { \mathrm { m c } }$ to be a useful hyperparameter for controlling learning efficiency. This estimator recovers the exact FPO ratio in the limit, although we use only a few draws in practice.

One possible concern with smaller $N _ { \mathrm { m c } }$ values is bias. A ratio estimated from only one (τ , ϵ) pair,

$$
\hat {r} _ {\theta} ^ {\mathrm{FPO}} (\tau , \epsilon) = \exp (\ell_ {\theta_ {\mathrm{old}}} ^ {w} (\tau , \epsilon) - \ell_ {\theta} ^ {w} (\tau , \epsilon)), \tag {17}
$$

is in expectation only an upper-bound of the true ratio. This can be shown by Jensen’s inequality:

$$
\mathbb {E} _ {\tau , \epsilon} [ \hat {r} _ {\theta} ^ {\mathrm{FPO}} (\tau , \epsilon) ] \geq r _ {\theta} ^ {\mathrm{FPO}}. \tag {18}
$$

To understand the upward bias, we can use the log-derivative trick to decompose the FPO gradient:

$$
\nabla_ {\theta} \hat {r} _ {\theta} ^ {\mathrm{FPO}} (\tau , \epsilon) = - \hat {r} _ {\theta} ^ {\mathrm{FPO}} (\tau , \epsilon) \nabla_ {\theta} \ell_ {\theta} ^ {w} (\tau , \epsilon). \tag {19}
$$

Since the gradient operator commutes with expectation, the gradient term on the right side is unbiased:

$$
\mathbb {E} _ {\tau , \epsilon} [ - \nabla_ {\theta} \ell_ {\theta} ^ {w} (\tau , \epsilon) ] = - \nabla_ {\theta} \mathcal {L} _ {\theta} ^ {w} (a _ {t}) = \nabla_ {\theta} \mathrm{ELBO} _ {\theta} (a _ {t}). \tag {20}
$$

In other words, gradient estimates are directionally unbiased even with worst-case overestimation of ratios. Our experiments are consistent with this result: while additional samples help, we observe empirically in Section 4.2 that FPO can be trained to outperform Gaussian PPO even with $N _ { \mathrm { m c } } = 1$ .

Algorithm 1 details FPO’s practical implementation using this mathematical framework.

# 3.5 Denoising MDP Comparison

Existing algorithms [54, 59, 55] for on-policy reinforcement learning with diffusion models reformulate the denoising process itself as a Markov Decision Process (MDP). These approaches bypass flow model likelihoods by instead treating every step in the sampling chain as its own action, each parameterized as a Gaussian policy step. This has a few limitations that FPO addresses.

First, denoising MDPs multiply the horizon length by the number of denoising steps (typically 10-50), which increases the difficulty of credit assignment. Second, these MDPs do not consider the initial noise sample during likelihood computation. Instead, these noise values are treated as observations from the environment [59]—this significantly increases the dimensionality of the learning problem. Finally, denoising MDP methods are limited to stochastic sampling procedures by construction. Instead, since FPO employs flow matching, it inherits the flexibility of sampler choices from standard flow/diffusion models. These include fast deterministic samplers, higher-order integration, and choosing any number of sampling steps. Perhaps most importantly, FPO is simpler to implement because it does not require a custom sampler or the notion of extra environment steps.

Algorithm 1 Flow Policy Optimization (FPO)   
Require: Policy parameters $\theta$ , value function parameters $\phi$ , clip parameter $\epsilon$ , MC samples $N_{\mathrm{mc}}$ 1: while not converged do  
2: Collect trajectories using any flow model sampler and compute advantages $\hat{A}_t$ 3: For each action, store $N_{\mathrm{mc}}$ timestep-noise pairs $\{(\tau_i, \epsilon_i)\}$ and compute $\ell_\theta(\tau_i, \epsilon_i)$ 4: $\theta_{\mathrm{old}} \leftarrow \theta$ 5: for each optimization epoch do  
6: Sample mini-batch from collected trajectories  
7: for each state-action pair $(o_t, a_t)$ and corresponding MC samples $\{(\tau_i, \epsilon_i)\}$ do  
8: Compute $\ell_\theta(\tau_i, \epsilon_i)$ using stored $(\tau_i, \epsilon_i)$ 9: $\hat{r}_\theta \leftarrow \exp\left(-\frac{1}{N_{\mathrm{mc}}}\sum_{i=1}^{N_{\mathrm{mc}}}\left(\ell_\theta(\tau_i, \epsilon_i) - \ell_{\theta_{\mathrm{old}}}( \tau_i, \epsilon_i)\right)\right)$ 10: $L^{\mathrm{FPO}}(\theta) \leftarrow \min(\hat{r}_\theta \hat{A}_t,\operatorname{clip}(\hat{r}_\theta, 1 \pm \epsilon) \hat{A}_t)$ 11: end for  
12: $\theta \leftarrow \text{Optimizer}(\theta, \nabla_\theta \sum L^{\mathrm{FPO}}(\theta))$ 13: end for  
14: Update value function parameters $\phi$ like standard PPO  
15: end while

![](49301b89da7d33e9dcff95adbca78dd36bfdade94fa92378138c700f0b6911c0.jpg)

<details>
<summary>text_image</summary>

Gridworld with 2 Goals
Goal ★ Agent
</details>

![](26d146de8cc9bda6698932b6f5d16588b7794d832daa2f0fe1b3e8893903b401.jpg)

<details>
<summary>text_image</summary>

Learned Flow and Target Action Distribution at ★
Denoising steps →
</details>

![](9df621821201482c13920d663f1e0b61e9889c1d28a38cc8c65b6fcbc1d969ac.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["start"] --> B["end"]
    C["start"] --> D["end"]
    E["start"] --> F["end"]
    G["Start"] --> H["End"]
    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
```
</details>

Figure 1: Grid World. (Left) 25×25 GridWorld with green goal cells. Each arrow shows a denoised action sampled from the FPO-trained policy, conditioned on a different latent noise vector. (Center) At the saddle-point state (⋆) shown on the left, we visualize three denoising steps τ as the initial Gaussian gradually transforms into the target distribution through the learned flow, illustrated by the deformation of the coordinate grid. (Right) Sampled trajectories from the same starting states reach different goals, illustrating the multimodal behavior captured by FPO.

# 4 Experiments

We assess FPO’s effectiveness by evaluating it in multiple domains. Our experiments include: (1) an illustrative GridWorld environment using Gymnasium [64, 65], (2) continuous control tasks with MuJoCo Playground [14, 66], and (3) physics-based humanoid control in Isaac Gym [67]. These tasks vary in dimensionality, reward sparsity, horizon length, and simulation environments.

# 4.1 GridWorld

We first test FPO on a 25×25 GridWorld environment designed to probe the policy’s ability to capture multimodal action distributions. As shown in Figure 1 left, the environment consists of two high reward regions located as the top and bottom of the map (green cells). The reward is sparse: agents receive a single reward upon reaching a goal or a penalty, with no intermediate rewards. This setup creates saddle points where multiple distinct actions can lead to equally successful outcomes, offering a natural opportunity to model diverse behaviors.

We train a diffusion policy from scratch using FPO by modifying a standard implementation [68] of PPO. The policy is parameterized as a two-layer MLP modeling $p ( a _ { t } \mid s , a _ { t } ^ { \tau } )$ , where $a _ { t } \in \mathbb { R } ^ { 2 }$ is the action, $s \in \mathbb { R } ^ { 2 }$ is the grid state, and $a _ { t } ^ { \tau } \in \mathbb { R } ^ { 2 }$ is the latent noise vector at noise level τ , initialized from $\mathcal { N } ( 0 , I )$ at $\tau = 0$ . FPO consistently maximizes the return in this environment. The arrows in Figure 1 left shows denoised actions at each grid location, computed by conditioning on a random $\boldsymbol { a } _ { t } ^ { \top } \sim \mathcal { N } ( \boldsymbol { 0 } , I )$ and running 10 steps of Euler integration. In Figure 1 center, we probe the learned policy by visualizing the flow over its denoising steps at the saddle point. The initial Gaussian evolves into a bimodal distribution, demonstrating that the policy captures the multi-modality of the solution at this location. Figure 1 right shows multiple trajectories sampled from the policy, initialized from various fixed starting positions. The agent exhibits multimodal behavior, with trajectories from the same starting state reaching different goals. Even when heading toward the same goal, the paths vary significantly, reflecting the policy’s ability to model diverse action sequences.

![](d6169416aabd573796e7d227dbbd3bab79306db2569a6b858fd51d6ad14728e4.jpg)

Figure 2: Comparison between FPO and Gaussian PPO [20] on DM Control Suite tasks. Results show evaluation reward mean and standard error (y-axis) over 60M environment steps (x-axis). We run 5 seeds for each task; the curve with the highest terminal evaluation reward is bolded.   
![](a24c42458b8de054de2af7343d14bf7c1a2e58ed8d01af08a3def0758c13405b.jpg)  
Figure 3: Comparison between FPO and DPPO [59] on DM Control Suite tasks. Results show evaluation reward mean and standard error (y-axis) over 60M environment steps (x-axis). We run 5 seeds for each task; the curve with the highest terminal evaluation reward is bolded.

We also train a Gaussian policy using PPO, which successfully reaches the goal regions. Compared to FPO, it exhibits more deterministic behavior, consistently favoring the nearest goal with less variation in trajectory patterns. Results are included in the supplemental material (Appendix A.2).

# 4.2 MuJoCo Playground

Next, we evaluate FPO for continuous control using MuJoCo Playground [14]. We compare three policy learning algorithms: (i) a Gaussian policy trained using PPO, (ii) a diffusion policy trained using FPO, and (iii) a diffusion policy trained using DPPO [59]. We evaluate these algorithms on 5 seeds for each of 10 environments adapted from the DeepMind Control Suite [69, 70]. Results are reported in Figures 2 and 3.

Policy implementations. For the Gaussian policy baseline, we run the Brax [71]-based implementation used by MuJoCo Playground [14]’s PPO training scripts. We also use Brax PPO as a starting point for implementing both FPO and DPPO. Following Section 3.2, only small changes are required for FPO: noisy action and timestep inputs are included as input to the policy network, Gaussian sampling is replaced with flow sampling, and the PPO loss’s likelihood ratio is replaced with the FPO ratio approximation. For DPPO, we make the same policy network modification, but apply stochastic sampling [55] during rollouts. We also augment each action in the experience buffer with the exact sampling path that was taken to reach it. Following the two-layer MDP formulation in DPPO [59], we then replace intractable action likelihoods with noise-conditioned sampling path likelihoods.

<table><tr><td>Methods</td><td>Goal conditioning</td><td>Success rate (↑)</td><td>Alive duration (↑)</td><td>MPJPE (↓)</td></tr><tr><td>Gaussian PPO</td><td>All joints</td><td>98.7%</td><td>200.46</td><td>31.62</td></tr><tr><td>FPO</td><td>All joints</td><td>96.4%</td><td>198.00</td><td>41.98</td></tr><tr><td>Gaussian PPO</td><td>Root + Hands</td><td>46.5%</td><td>142.50</td><td>97.65</td></tr><tr><td>FPO</td><td>Root + Hands</td><td>70.6%</td><td>171.32</td><td>62.91</td></tr><tr><td>Gaussian PPO</td><td>Root</td><td>29.8%</td><td>114.06</td><td>123.70</td></tr><tr><td>FPO</td><td>Root</td><td>54.3%</td><td>152.90</td><td>73.55</td></tr></table>

Table 2: Humanoid Control Quantitative Metrics. We compare FPO with Gaussian PPO with different conditioning goals, and report the success rate, alive duration, and MPJPE averaged over all motion sequences.

Hyperparameters. We match hyperparameters in Gaussian PPO, FPO, and DPPO training whenever possible: following the provided configurations in Playground [14], all experiments use ADAM [72], 60M total environment steps, batch size 1024, and 16 updates per batch. For FPO and DPPO, we use 10 sampling steps, set learning rates to 3e-4, and swept clipping epsilon $\varepsilon ^ { \mathrm { c l i p } } \in \{ 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 3 \}$ . For DPPO, we perturb each denoising step with Gaussian noise with standard deviation $\sigma _ { t } ,$ , which we swee $) \in \{ 0 . 0 1 , 0 . 0 5 , 0 . 1 \}$ . We found that $\varepsilon ^ { \mathrm { c l i p } } = 0 . 0 5$ produces the best FPO results and $\varepsilon ^ { \mathrm { c l i p } } = 0 . 2 , \sigma _ { t } = \mathrm { \bar { 0 } } . 0 5$ produced the best DPPO results; we use these values for all experiments. For fairness, we also tuned learning rates and clipping epsilons for Gaussian PPO. We provide more details about hyperparameters and baseline tuning in Appendix A.3.

Results. We observe in Figures 2 and 3 that FPO-optimized policies outperform both Gaussian PPO and DPPO on the Playground tasks. It outperforms both baselines in 8 of 10 tasks.

Analysis. In Table 1, we present average evaluation rewards for baselines, FPO, and several variations of FPO. We observe: (1) (τ, ϵ) sampling is important. Decreasing the number of sampled pairs generally decreases evaluation rewards. More samples can improve learning without requiring more expensive environment steps. (2) ϵ-MSE is preferable over u-MSE in Playground. ϵ-MSE refers to computing flow matching losses by first converting velocity estimates to ϵ noise values; u-MSE refers to MSE directly on velocity estimates. In Playground, we found that the former produces higher average rewards. We hypothesize that this is because ϵ scale is invariant to action scale, which results in better generalization for $\varepsilon ^ { \mathrm { c l i p } }$ choices. For fairness, we also performed learning rate and clipping ratio sweeps for the u-MSE ablation. (3) Clipping. Like Gaussian PPO, the choice of $\varepsilon ^ { \mathrm { c l i p } }$ in FPO significantly impacts performance.

<table><tr><td>Method</td><td>Reward</td></tr><tr><td>Gaussian PPO</td><td>667.8±66.0</td></tr><tr><td> $Gaussian\ PPO^{\dagger}$ </td><td>577.2±74.4</td></tr><tr><td>DPPO</td><td>652.5±83.7</td></tr><tr><td> $FPO^{\ddagger}$ </td><td>759.3±45.3</td></tr><tr><td>FPO, 1 (τ, ε)</td><td>691.6±50.3</td></tr><tr><td>FPO, 4 (τ, ε)</td><td>731.2±58.2</td></tr><tr><td>FPO, u-MSE</td><td>664.6±48.5</td></tr><tr><td>FPO,  $\varepsilon^{clip}=0.1$ </td><td>623.3±76.3</td></tr><tr><td>FPO,  $\varepsilon^{clip}=0.2$ </td><td>526.4±76.8</td></tr></table>

Table 1: FPO variant comparison. We report averages and standard errors across MuJoCo tasks. †Using default hyperparameters from MuJoCo Playground. ‡FPO results use $8 \left( \tau , \epsilon \right)$ pairs, $\boldsymbol { \epsilon } { \mathrm { - } } \mathbf { M } \mathbf { S } \mathbf { E } , \boldsymbol { \varepsilon } ^ { \mathrm { c l i p } } = 0 . 0 5$ .

# 4.3 Humanoid Control

Physics-aware humanoid control is higher-dimensional than standard MuJoCo benchmarks, making it a stringent test of FPO’s generality. We therefore train a humanoid policy to track motion-capture (MoCap) trajectories in the PHC setting [73], using the open-source Puffer-PHC implementation as our baseline2. This experiment follows the goal-conditioned imitation-learning paradigm pioneered by DeepMimic [74], in which simulated characters learn to reproduce reference motions. Depending on the deployment needs, these reference signals (goals) can be as rich as full-body joint information or as sparse as root joint (pelvis) commands, providing the flexibility required for reliable sim-to-real transfer [29]. The problem with sparse goals is under-conditioned and significantly more challenging, requiring the policy to fill in the missing joint specification in a manner that is physically plausible.

![](dc91f18e992aca90bcd463604f9decfb3f2ec9fcd3ca0de7199f69aeb70eda7e.jpg)

<details>
<summary>line</summary>

| Step     | FPO   | Gaussian PPO | All Joints | Root  | Root + Hand |
| -------- | ----- | ------------ | ---------- | ----- | ----------- |
| 0        | 8.0   | 6.0          | 25.0       | 7.0   | 12.0        |
| 200.0M   | 9.0   | 6.5          | 35.0       | 8.0   | 14.0        |
| 400.0M   | 10.0  | 7.0          | 38.0       | 9.0   | 15.0        |
| 600.0M   | 11.0  | 7.5          | 39.0       | 10.0  | 16.0        |
| 800.0M   | 12.0  | 8.0          | 40.0       | 11.0  | 17.0        |
| 1000.0M  | 13.0  | 8.5          | 41.0       | 12.0  | 18.0        |
| 1200.0M  | 14.0  | 9.0          | 42.0       | 13.0  | 19.0        |
</details>

(a) Episode return along training.

![](574e835ba008dcd9d6f94fabdab7ec6b238646477133da885727d695557b9266.jpg)

<details>
<summary>text_image</summary>

Reference
FPO
Gaussian
</details>

(b) Root+hand conditioning.

![](6222d493261207baf2887852ef278291ddf9ffbb829391e9613c487e626493e3.jpg)

<details>
<summary>natural_image</summary>

3D illustration of five stylized human figures walking on a textured pink surface (no text or symbols)
</details>

(c) Rough terrain locomotion.   
Figure 4: Physics-based Humanoid Control. (a) The curves show that FPO performance is close to that of Gaussian-PPO when conditioning on all joints and surpasses it when goals are reduced to the root or root+hands, indicating stronger robustness to sparse conditioning. (b) In the root+hands goal setting, FPO (blue) tracks the reference motion (grey) while Gaussian-PPO (orange) falls. (c) Trained with terrain randomization, FPO walks stably across procedurally generated rough ground.

Implementation details. Our simulated agent is an SMPL-based humanoid with 24 actuated joints, each offering six degrees of freedom and organized in a kinematic tree rooted at the pelvis, simulated in Isaac Gym [67]. The policy receives both proprioceptive observations and goal information computed from the motion-capture reference. A single policy is trained to track AMASS [75] motions following PHC [73]. We use the root height, joint positions, rotations, velocity, and angular velocity in a local coordinate frame as the robot state. For goal conditioning, we compute the difference between the tracking joint information (positions, rotations, velocity, and angular velocity) and the current robot’s joint information, as well as the tracking joint locations and rotations. We explore both full conditioning, i.e., conditioning on all joint targets, and under conditioning, i.e., conditioning only on the root or the root and hands targets. The latter matches the target signals typically provided by a joystick or VR controller. Please note that the same imitation reward based on all joints is used for both conditioning experiments. The per-joint tracking reward is computed as in DeepMimic [74].

Evaluation. For evaluation, we compute the success rate, considering an imitation unsuccessful if the average distance between the body joints and the reference motion exceeds 0.5 meters at any point during the sequence. We also report the average duration the agent stays alive till it completes the tracking or falls. Finally, we compute the global mean per-joint position error (MPJPE) on the conditioned goals.

Results. Figure 4a shows that we successfully train FPO from scratch on this high-dimensional control task. With full joint conditioning, FPO performance is close to Gaussian PPO. However, when the model is under-conditioned—e.g., conditioned only on the root or the root and hands—FPO outperforms Gaussian PPO, highlighting the advantage of flow-based policies. While prior methods can also achieve sparse-goal control, they often rely on training a teacher policy that conditions on full joint reference first and then distilling the knowledge to sparse conditioned policies [76, 29, 77] or training a separate encoder observing sparse references [78, 79].

Figure 4b visualizes the behaviors in the root+hands setting (left-to-right: reference motion, FPO, Gaussian-PPO); FPO tracks the target closely, whereas the Gaussian policy drifts. Table 2 quantifies these trends, with FPO achieving much higher success rates in the under-conditioned scenarios. Finally, as illustrated in Fig. 4c, FPO trained with terrain randomization enables the humanoid to traverse rough terrain, showing potential for sim-to-real transfer. Please see the supplemental video for more qualitative results.

# 5 Discussion and Limitations

We introduce Flow Policy Optimization (FPO), an algorithm for training flow-based generative models using policy gradients. FPO reformulates policy optimization as minimizing an advantage-weighted conditional flow matching (CFM) objective, enabling stable training without requiring explicit likelihood computation. It integrates easily with PPO-style algorithms, and crucially, preserves the flow-based structure of the policy—allowing the resulting model to be used with standard flow-based mechanisms such as sampling, distillation, and fine-tuning. We demonstrate FPO across a range of control tasks, including a challenging humanoid setting where it enables training from scratch under sparse goal conditioning, where Gaussian policies fail to learn.

The training and deployment of flow-based policies is generally more computationally intensive than for corresponding Gaussian policies. FPO also lacks established machinery such as KL divergence estimation for adaptive learning rates and entropy regularization.

We also explored applying FPO to fine-tune a pre-trained image diffusion model using reinforcement learning. While promising in principle, we found this setting to be unstable in practice—likely due to the issue of fine-tuning diffusion models on its own output multiple times as noted in recent works [80– 82]. In particular, we observed sensitivity to classifier-free guidance (CFG) that compounds with self-generated data, even outside of the RL framework. This suggests that the instability is not a limitation of FPO itself, but a broader challenge in applying reinforcement learning to image generation. Please see the supplementary material for more detail.

Despite these limitations, FPO offers a simple and flexible bridge between flow-based models and online reinforcement learning. We are particularly excited to see future work apply FPO in settings where flow-based policies are already pretrained—such as behavior-cloned diffusion policies in robotics—where FPO’s compatibility and simplicity may offer practical benefits for fine-tuning with task reward.

# Acknowledgments

We thank Qiyang (Colin) Li, Oleg Rybkin, Lily Goli and Michael Psenka for helpful discussions and feedback on the manuscript. We thank Arthur Allshire, Tero Karras, Miika Aittala, Kevin Zakka and Seohong Park for insightful input and feedback on implementation details and the broader context of this work. This project was funded in part by NSF:CNS-2235013, IARPA DOI/IBC No. 140D0423C0035, and Bakar fellows. CK and BY are supported by NSF fellowship. SG is supported by the NVIDIA Graduate Fellowship

# References

[1] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.   
[2] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. 2022.   
[3] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.   
[4] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.   
[5] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.   
[6] Veo-Team, :, Agrim Gupta, Ali Razavi, Andeep Toor, Ankush Gupta, Dumitru Erhan, Eleni Shaw, Eric Lau, Frank Belletti, Gabe Barth-Maron, Gregory Shaw, Hakan Erdogan, Hakim Sidahmed, Henna Nandwani, Hernan Moraldo, Hyunjik Kim, Irina Blok, Jeff Donahue, José Lezama, Kory Mathewson, Kurtis David, Matthieu Kim Lorrain, Marc van Zee, Medhini Narasimhan, Miaosen Wang, Mohammad Babaeizadeh, Nelly Papalampidi, Nick Pezzotti, Nilpa Jha, Parker Barnes, Pieter-Jan Kindermans, Rachel Hornung, Ruben Villegas, Ryan Poplin, Salah Zaiem, Sander Dieleman, Sayna Ebrahimi, Scott Wisdom, Serena Zhang, Shlomi Fruchter, Signe Nørly, Weizhe Hua, Xinchen Yan, Yuqing Du, and Yutian Chen. Veo 2. 2024. URL https://deepmind.google/technologies/veo/veo-2/.   
[7] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D. Plumbley. Audioldm: Text-to-audio generation with latent diffusion models, 2023. URL https://arxiv.org/abs/2301.12503.   
[8] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis, 2021. URL https://arxiv.org/abs/2009.09761.   
[9] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 2024.   
[10] Sanjeev Raja, Martin Šípka, Michael Psenka, Tobias Kreiman, Michal Pavelka, and Aditi S Krishnapriyan. Action-minimization meets generative modeling: Efficient transition path sampling with the onsager-machlup functional. arXiv preprint arXiv:2504.18506, 2025.   
[11] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025.   
[12] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.   
[13] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. URL https://arxiv.org/abs/2210.02747.   
[14] Kevin Zakka, Baruch Tabanpour, Qiayuan Liao, Mustafa Haiderbhai, Samuel Holt, Jing Yuan Luo, Arthur Allshire, Erik Frey, Koushil Sreenath, Lueder A Kahrs, et al. Mujoco playground. arXiv preprint arXiv:2502.08844, 2025.

[15] Richard S. Sutton, David McAllester, Satinder P. Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Proceedings of the 12th International Conference on Neural Information Processing Systems (NeurIPS), pages 1057– 1063, 1999.   
[16] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 1992.   
[17] Sham M. Kakade. A natural policy gradient. In Proceedings of the 14th International Conference on Neural Information Processing Systems (NeurIPS), pages 1531–1538, 2002.   
[18] Jan Peters and Stefan Schaal. Natural actor–critic. Neurocomputing, 71(7–9):1180–1190, 2008.   
[19] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.   
[20] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.   
[21] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Tim Harley, Timothy Lillicrap, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Proceedings of the 33rd International Conference on Machine Learning (ICML), pages 1928–1937, 2016.   
[22] Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, and Nando de Freitas. Sample efficient actor–critic with experience replay. In Proceedings of the 30th International Conference on Neural Information Processing Systems (NeurIPS), pages 1061–1071, 2016.   
[23] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
[24] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking deep reinforcement learning for continuous control. In International conference on machine learning, pages 1329–1338. PMLR, 2016.   
[25] Shengyi Huang, Quentin Gallouédec, Florian Felten, Antonin Raffin, Rousslan Fernand Julien Dossa, Yanxiao Zhao, Ryan Sullivan, Viktor Makoviychuk, Denys Makoviichuk, Mohamad H Danesh, et al. Open rl benchmark: Comprehensive tracked experiments for reinforcement learning. arXiv preprint arXiv:2402.03046, 2024.   
[26] Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter. Learning to walk in minutes using massively parallel deep reinforcement learning. In Proceedings of the 5th Conference on Robot Learning, volume 164 of Proceedings of Machine Learning Research, pages 91–100. PMLR, 2022. URL https://proceedings.mlr.press/v164/rudin22a.html.   
[27] Clemens Schwarke, Victor Klemm, Matthijs van der Boon, Marko Bjelonic, and Marco Hutter. Curiosity-driven learning of joint locomotion and manipulation tasks. In Proceedings of The 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pages 2594–2610. PMLR, 2023. URL https://proceedings.mlr.press/v229/ schwarke23a.html.   
[28] Mayank Mittal, Nikita Rudin, Victor Klemm, Arthur Allshire, and Marco Hutter. Symmetry considerations for learning task symmetric robot policies. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 7433–7439, 2024. doi: 10.1109/ICRA57147.2024.10611493.   
[29] Arthur Allshire, Hongsuk Choi, Junyi Zhang, David McAllister, Anthony Zhang, Chung Min Kim, Trevor Darrell, Pieter Abbeel, Jitendra Malik, and Angjoo Kanazawa. Visual imitation enables contextual humanoid control. arXiv preprint arXiv:2505.03729, 2025.

[30] Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, et al. Solving rubik’s cube with a robot hand. arXiv preprint arXiv:1910.07113, 2019.   
[31] Tao Chen, Jie Xu, and Pulkit Agrawal. A system for general in-hand object re-orientation. Conference on Robot Learning, 2021.   
[32] Haozhi Qi, Brent Yi, Sudharshan Suresh, Mike Lambeta, Yi Ma, Roberto Calandra, and Jitendra Malik. General in-hand object rotation with vision and touch. In Conference on Robot Learning, pages 2549–2564. PMLR, 2023.   
[33] Haozhi Qi, Brent Yi, Mike Lambeta, Yi Ma, Roberto Calandra, and Jitendra Malik. From simple to complex skills: The case of in-hand object reorientation. arXiv preprint arXiv:2501.05439, 2025.   
[34] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 2022.   
[35] Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences, 2023. URL https://arxiv.org/abs/ 1706.03741.   
[36] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.   
[37] Mistral-AI, :, Abhinav Rastogi, Albert Q. Jiang, Andy Lo, Gabrielle Berrada, Guillaume Lample, Jason Rute, Joep Barmentlo, Karmesh Yadav, Kartik Khandelwal, Khyathi Raghavi Chandu, Léonard Blier, Lucile Saulnier, Matthieu Dinot, Maxime Darrin, Neha Gupta, Roman Soletskyi, Sagar Vaze, Teven Le Scao, Yihan Wang, Adam Yang, Alexander H. Liu, Alexandre Sablayrolles, Amélie Héliou, Amélie Martin, Andy Ehrenberg, Anmol Agarwal, Antoine Roux, Arthur Darcet, Arthur Mensch, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault,

Chris Bamford, Christian Wallenwein, Christophe Renaudin, Clémence Lanfranchi, Darius Dabert, Devon Mizelle, Diego de las Casas, Elliot Chane-Sane, Emilien Fugier, Emma Bou Hanna, Gauthier Delerce, Gauthier Guinet, Georgii Novikov, Guillaume Martin, Himanshu Jaju, Jan Ludziejewski, Jean-Hadrien Chabran, Jean-Malo Delignon, Joachim Studnia, Jonas Amar, Josselin Somerville Roberts, Julien Denize, Karan Saxena, Kush Jain, Lingxiao Zhao, Louis Martin, Luyu Gao, Lélio Renard Lavaud, Marie Pellat, Mathilde Guillaumin, Mathis Felardos, Maximilian Augustin, Mickaël Seznec, Nikhil Raghuraman, Olivier Duchenne, Patricia Wang, Patrick von Platen, Patryk Saffer, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Pavankumar Reddy Muddireddy, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Romain Sauvestre, Rémi Delacourt, Sanchit Gandhi, Sandeep Subramanian, Shashwat Dalal, Siddharth Gandhi, Soham Ghosh, Srijan Mishra, Sumukh Aithal, Szymon Antoniak, Thibault Schueller, Thibaut Lavril, Thomas Robert, Thomas Wang, Timothée Lacroix, Valeriia Nemychnikova, Victor Paltz, Virgile Richard, Wen-Ding Li, William Marshall, Xuanyu Zhang, and Yunhao Tang. Magistral, 2025. URL https://arxiv.org/abs/2506.10910.   
[38] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 2020.   
[39] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022. URL https://arxiv.org/abs/2010.02502.   
[40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models, 2022. URL https://arxiv.org/ abs/2112.10752.   
[41] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution, 2020. URL https://arxiv.org/abs/1907.05600.   
[42] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models, 2022. URL https://arxiv.org/abs/2204.03458.   
[43] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data, 2022. URL https://arxiv.org/abs/ 2209.14792.   
[44] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models, 2022. URL https: //arxiv.org/abs/2210.02303.   
[45] Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, and Mikhail Kudinov. Gradtts: A diffusion probabilistic model for text-to-speech, 2021. URL https://arxiv.org/abs/ 2105.06337.   
[46] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, Najim Dehak, and William Chan. Wavegrad 2: Iterative refinement for text-to-speech synthesis, 2021. URL https://arxiv.org/abs/2106.09660.   
[47] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π0: A vision-language-action flow model for general robot control, 2024. URL https://arxiv. org/abs/2410.24164.   
[48] NVIDIA, :, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots, 2025. URL https://arxiv.org/abs/2503.14734.

[49] Marta Skreta, Lazar Atanackovic, Avishek Joey Bose, Alexander Tong, and Kirill Neklyudov. The superposition of diffusion models using the itô density estimator, 2025. URL https: //arxiv.org/abs/2412.17762.   
[50] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 2024.   
[51] Anurag Ajay, Yilun Du, Abhi Gupta, Joshua B. Tenenbaum, Tommi S. Jaakkola, and Pulkit Agrawal. Is conditional generative modeling all you need for decision making? In The Eleventh International Conference on Learning Representations, 2023.   
[52] Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. arXiv preprint arXiv:2205.09991, 2022.   
[53] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023.   
[54] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.   
[55] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.   
[56] Michael Psenka, Alejandro Escontrela, Pieter Abbeel, and Yi Ma. Learning a diffusion model policy from rewards via q-score matching. arXiv preprint arXiv:2312.11752, 2023.   
[57] Younggyo Seo, Carmelo Sferrazza, Haoran Geng, Michal Nauman, Zhao-Heng Yin, and Pieter Abbeel. Fasttd3: Simple, fast, and capable reinforcement learning for humanoid control, 2025. URL https://arxiv.org/abs/2505.22642.   
[58] Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods, 2018. URL https://arxiv.org/abs/1802.09477.   
[59] Allen Z Ren, Justin Lidard, Lars L Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion policy policy optimization. arXiv preprint arXiv:2409.00588, 2024.   
[60] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.   
[61] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35: 26565–26577, 2022.   
[62] Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models, 2023. URL https://arxiv.org/abs/2107.00630.   
[63] Diederik P. Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation, 2023. URL https://arxiv.org/abs/2303.00848.   
[64] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym, 2016.   
[65] Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U Balis, Gianluca De Cola, Tristan Deleu, Manuel Goulão, Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032, 2024.   
[66] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, 2012.

[67] Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, et al. Isaac gym: High performance gpu-based physics simulation for robot learning. arXiv preprint arXiv:2108.10470, 2021.   
[68] Eric Yang Yu. Ppo-for-beginners: A simple, well-styled ppo implementation in pytorch. https://github.com/ericyangyu/PPO-for-Beginners, 2020. GitHub repository.   
[69] Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite. arXiv preprint arXiv:1801.00690, 2018.   
[70] Saran Tunyasuvunakool, Alistair Muldal, Yotam Doron, Siqi Liu, Steven Bohez, Josh Merel, Tom Erez, Timothy Lillicrap, Nicolas Heess, and Yuval Tassa. dm\_control: Software and tasks for continuous control. Software Impacts, 6:100022, 2020.   
[71] C Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier Bachem. Brax–a differentiable physics engine for large scale rigid body simulation. arXiv preprint arXiv:2106.13281, 2021.   
[72] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.   
[73] Zhengyi Luo, Jinkun Cao, Kris Kitani, Weipeng Xu, et al. Perpetual humanoid control for real-time simulated avatars. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10895–10904, 2023.   
[74] Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel Van de Panne. Deepmimic: Exampleguided deep reinforcement learning of physics-based character skills. ACM Transactions On Graphics (TOG), 37(4):1–14, 2018.   
[75] Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. Amass: Archive of motion capture as surface shapes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5442–5451, 2019.   
[76] Chen Tessler, Yunrong Guo, Ofir Nabati, Gal Chechik, and Xue Bin Peng. Maskedmimic: Unified physics-based character control through masked motion inpainting. ACM Transactions on Graphics (TOG), 43(6):1–21, 2024.   
[77] Yixuan Li, Yutang Lin, Jieming Cui, Tengyu Liu, Wei Liang, Yixin Zhu, and Siyuan Huang. Clone: Closed-loop whole-body humanoid teleoperation for long-horizon tasks. arXiv preprint arXiv:2506.08931, 2025.   
[78] Zhengyi Luo, Jinkun Cao, Josh Merel, Alexander Winkler, Jing Huang, Kris Kitani, and Weipeng Xu. Universal humanoid motion representations for physics-based control. arXiv preprint arXiv:2310.04582, 2023.   
[79] Zhengyi Luo, Jinkun Cao, Sammy Christen, Alexander Winkler, Kris Kitani, and Weipeng Xu. Omnigrasp: Grasping diverse objects with simulated humanoids. In Advances in Neural Information Processing Systems, volume 37, pages 2161–2184, 2024.   
[80] Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. Ai models collapse when trained on recursively generated data. Nature, 631(8022): 755–759, 2024.   
[81] Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, and Ross Anderson. The curse of recursion: Training on generated data makes models forget. arXiv preprint arXiv:2305.17493, 2023.   
[82] Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, and Richard G Baraniuk. Self-consuming generative models go mad. International Conference on Learning Representations (ICLR), 2024.

[83] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.   
[84] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. URL https:// arxiv.org/abs/2207.12598.

# Flow Matching Policy Gradients Supplementary Material

In this supplementary material, we discuss the deferred proofs of technical results, elaborate on the details of our experiments, and present additional visual results for the grid world, humanoid control, and image finetuning experiments.

# A.1 FPO Derivation

The mathematical details presented in this section provide expanded derivations and additional context for the theoretical results outlined in Section 3 of the main text. Specifically, we elaborate on the connection between the conditional flow matching objective and the evidence lower bound (ELBO) first mentioned in Section 3.4, and provide complete derivations for the FPO ratio introduced in Section 3.3. These details are included for completeness and to situate our work within the theoretical framework established by Kingma et al. [63], but are not necessary for understanding the core FPO algorithm or implementing it in practice.

First, we detail the different popular loss weightings used when training flow matching models laid out by Kingma et al. [63]. These weightings, denoted as $w ( \lambda _ { t } )$ , determine how losses at different noise levels contribute to the overall objective and lead to different theoretical interpretations of Flow Policy Optimization.

Then, we show the more general result, which is that FPO optimizes the advantage-weighted expected ELBO of the noise-perturbed data. Specifically, for any monotonic weighting function (including Optimal Transport CFM schedules [13]), we can express the weighted loss as:

$$
\mathcal {L} _ {\theta} ^ {w} (a _ {t}) = - \mathbb {E} _ {p _ {w} (\tau), q (a _ {t} ^ {\tau} | a _ {t})} [ \mathrm{ELBO} _ {\tau} (a _ {t} ^ {\tau}) ] + c _ {1}, \tag {21}
$$

where $p _ { w } ( \tau )$ is the distribution over timesteps induced by the weighting function, and $\mathrm { E L B O } _ { \tau } ( a _ { t } ^ { \tau } )$ is the evidence lower bound at noise level τ for the perturbed action $a _ { t } ^ { \tau }$ .

This means that FPO increases the likelihood of high-reward samples and the intermediate noisy samples $a _ { t } ^ { \tau }$ from the sample path. By weighting this objective with advantages $\hat { A } _ { \tau }$ , we guide the policy to direct probability flow toward action neighborhoods that produce higher reward.

For diffusion schedules with uniform weighting $w ( \lambda _ { \tau } ) = 1$ , we show a somewhat stronger theoretical result. In this special case, the weighted loss directly corresponds to maximizing the ELBO of clean actions:

$$
- \mathrm{ELBO} (a _ {t}) = \frac {1}{2} \mathbb {E} _ {\tau \sim U (0, 1), \epsilon \sim \mathcal {N} (0, I)} \left[ - \frac {d \lambda}{d \tau} \cdot \| \hat {\epsilon} _ {\theta} (a _ {t} ^ {\tau}; \lambda_ {\tau}) - \epsilon \| _ {2} ^ {2} \right] + c _ {2}, \tag {22}
$$

which is a more direct connection to maximum likelihood estimation.

# A.1.1 Loss Weighting Choices

Most popular instantiations of flow-based and diffusion models can be reparameterized in the weighted loss scheme proposed by Kingma et al. [63]. This unified framework expresses each version as an instance of a weighted denoising loss:

$$
\mathcal {L} _ {\theta} ^ {w} (x) = \frac {1}{2} \mathbb {E} _ {\tau \sim U (0, 1), \epsilon \sim \mathcal {N} (0, I)} [ w (\lambda_ {\tau}) \cdot - \frac {d \lambda}{d \tau} \cdot \| \hat {\epsilon} _ {\theta} (a _ {t} ^ {\tau}; \lambda_ {\tau}) - \epsilon \| _ {2} ^ {2} ], \tag {23}
$$

where $w ( \lambda _ { \tau } )$ is a time-dependent function that determines the relative importance of different noise levels.

For those with a loss weight that varies monotonically with noise timestep τ , the aforementioned relationship between the weighted loss and expected ELBO holds. Specifically, when $w ( \lambda _ { \tau } )$ is monotonically increasing with τ , Kingma et al. prove:

$$
\mathcal {L} _ {\theta} ^ {w} \left(a _ {t}\right) = - \mathbb {E} _ {p _ {w} (\tau), q \left(a _ {t} ^ {\tau} \mid a _ {t}\right)} \left[ \mathrm{ELBO} _ {\tau} \left(a _ {t} ^ {\tau}\right) \right] + c _ {1}, \tag {24}
$$

where $c _ { 1 }$ is a constant, and does not vary with model parameters.

These monotonic weightings include several popular schedules: (1) standard diffusion with uniform weighting $w ( \lambda _ { \tau } ) = \bar { 1 } \ [ 3 8 ]$ , (2) optimal transport linear interpolation schedule [13], which yields $w ( \lambda _ { \tau } ) = e ^ { - \lambda / 2 } ,$ , and (3) velocity prediction (v-prediction) with cosine schedule [83], which also yields $v ( \lambda _ { \tau } ) = e ^ { - \lambda / 2 }$ .

# A.1.2 Flow Matching as Expected ELBO Optimization

To derive FPO in the more general flow matching case, we begin with the standard policy gradient objective, but replace direct likelihood maximization with maximization of the ELBO for noiseperturbed data:

$$
\max _ {\theta} \mathbb {E} _ {a _ {t} \sim \pi_ {\theta} (a _ {t} | o _ {t})} \left[ \mathbb {E} _ {p _ {w} (\tau), q (a _ {t} ^ {\tau} | a _ {t})} \left[ \mathrm{ELBO} _ {\tau} \left(a _ {t} ^ {\tau}\right) \right] \cdot \hat {A} _ {t} \right], \tag {25}
$$

where t is temporal rollout time and τ is diffusion/flow noise timestep.

This formulation directly leverages the result from Kingma et al. [63] that for monotonic weightings, the weighted denoising loss equals the negative expected ELBO of noise-perturbed data plus a constant:

$$
\mathcal {L} _ {\theta} ^ {w} \left(a _ {t}\right) = - \mathbb {E} _ {p _ {w} (\tau), q \left(a _ {t} ^ {\tau} \mid a _ {t}\right)} \left[ \mathrm{ELBO} _ {\tau} \left(a _ {t} ^ {\tau}\right) \right] + c _ {1}. \tag {26}
$$

To apply this within a trust region approach similar to PPO, we need to define a ratio between the current and old policies. Since we are working with expected ELBOs, the appropriate ratio becomes:

$$
r ^ {\mathrm{FPO}} (\theta) = \frac {\exp \left(\mathbb {E} _ {p _ {w} (\tau) , q \left(a _ {t} ^ {\tau} \mid a _ {t}\right)} \left[ \mathrm{ELBO} _ {\tau} \left(a _ {t} ^ {\tau}\right) \right] _ {\theta}\right)}{\exp \left(\mathbb {E} _ {p _ {w} (\tau) , q \left(a _ {t} ^ {\tau} \mid a _ {t}\right)} \left[ \mathrm{ELBO} _ {\tau} \left(a _ {t} ^ {\tau}\right) \right] _ {\theta , \text {old}}\right)} \tag {27}
$$

This ratio represents the relative likelihood of actions and their noisy versions under the current policy compared to the old policy.

It is important to note that the constant $c _ { 1 }$ in the ELBO equivalence depends only on the noise schedule endpoints $\lambda _ { m i n }$ and $\lambda _ { m a x }$ , the data distribution, and the forward process, but not on the model parameter θ. This is critical for our derivation. It ensures that within a single trust region data collection and training episode, this constant remains identical between the old policy $\theta _ { o l d }$ and the updated policy θ. Consequently, when forming the ratio $r ^ { \mathrm { F P O } } ( \theta )$ , these constants cancel out:

$$
r ^ {\mathrm{FPO}} (\theta) = \frac {\exp \left(\mathbb {E} _ {p _ {w} (\tau) , q \left(a _ {t} ^ {\tau} \mid a _ {t}\right)} \left[ \mathrm{ELBO} _ {\tau} \left(a _ {t} ^ {\tau}\right) \right] _ {\theta} + c _ {1}\right)}{\exp \left(\mathbb {E} _ {p _ {w} (\tau) , q \left(a _ {t} ^ {\tau} \mid a _ {t}\right)} \left[ \mathrm{ELBO} _ {\tau} \left(a _ {t} ^ {\tau}\right) \right] _ {\theta , \text {old}} + c _ {1}\right)} = \frac {\exp \left(\mathbb {E} _ {p _ {w} (\tau) , q \left(a _ {t} ^ {\tau} \mid a _ {t}\right)} \left[ \mathrm{ELBO} _ {\tau} \left(a _ {t} ^ {\tau}\right) \right] _ {\theta}\right)}{\exp \left(\mathbb {E} _ {p _ {w} (\tau) , q \left(a _ {t} ^ {\tau} \mid a _ {t}\right)} \left[ \mathrm{ELBO} _ {\tau} \left(a _ {t} ^ {\tau}\right) \right] _ {\theta , \text {old}}\right)} \tag {28}
$$

We estimate this ratio through Monte Carlo sampling of timesteps τ and noise ϵ:

$$
\hat {r} ^ {\mathrm{FPO}} (\tau , \epsilon) = \exp (- \ell_ {\theta} (\tau , \epsilon) + \ell_ {\theta , \text { old }} (\tau , \epsilon)), \tag {29}
$$

where $\begin{array} { r } { \ell _ { \theta } ( \tau , \epsilon ) = \frac { 1 } { 2 } [ - \dot { \lambda } ( \tau ) ] \| \hat { \epsilon } _ { \theta } ( a _ { t } ^ { \tau } ; \lambda _ { \tau } ) - \epsilon \| ^ { 2 } } \end{array}$ is the reparameterized conditional flow matching loss for a single draw of random variables ϵ and τ .

As discussed in the main text, $\hat { r } ^ { \mathrm { F P O } }$ overestimates the scale but unbiasedly estimates the direction of the gradient. We can reduce or eliminate the scale bias by drawing more samples of τ and ϵ.

# A.1.3 FPO with Diffusion Schedules

For the special case of standard diffusion schedules with uniform weighting $w ( \lambda _ { t } ) = 1$ , we can derive a stronger theoretical result connecting our optimization objective directly to the ELBO of clean (non-noised) data.

![](26ff14ba2718567906c73262a2547cc44295623497dd03dd1634891505a45287.jpg)

<details>
<summary>text_image</summary>

Gridworld with 2 Goals
Gaussian Policy
Goal
</details>

![](59bb67279146a0ad6ff2831da7233beb291c042a762ba7983d693eca7ec59377.jpg)

<details>
<summary>text_image</summary>

noise = 0.0
</details>

![](287f8ec5251bbc42bb35ff6d4fe4c1de36c100ad8e63447af1d13e24a49b2233.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["start"] --> B["end"]
    B --> C["End"]
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#dfd,stroke:#333
    note right of C: noise = 0.1
```
</details>

![](7931e637287c94777c2edb3f3b2d911d65fc18f58bce06216f86e018aed76a7e.jpg)

<details>
<summary>text_image</summary>

noise = 0.5
</details>

Figure A.1: GridWorld with Gaussian Policy. Left) 25 × 25 GridWorld with green goal cells. Each arrow shows an action predicted by the Gaussian policy. Right) Four rollouts under test-time noise perturbations $( \sigma = 0 . 0 , 0 . 1 , 0 . \bar { 5 } )$ . While the Gaussian policy achieves the goal, its trajectories lack diversity and hit the same goal consistently when given the same initialization point.

As shown by Kingma et al. [63], when using uniform weighting, the weighted loss directly corresponds to the negative ELBO of the clean data plus a constant:

$$
- \mathrm{ELBO} (a _ {t}) = \frac {1}{2} \mathbb {E} _ {\tau \sim U (0, 1), \epsilon \sim \mathcal {N} (0, I)} \left[ - \frac {d \lambda}{d \tau} \cdot \| \hat {\epsilon} _ {\theta} (a _ {t} ^ {\tau}; \lambda_ {\tau}) - \epsilon \| _ {2} ^ {2} \right] + c _ {2}, \tag {30}
$$

where $c _ { 2 }$ is a different constant than $c _ { 1 }$ that also does not depend on model parameter θ.

This means that minimizing the unweighted loss $( w ( \lambda _ { \tau } ) = 1 )$ is equivalent to maximizing the ELBO of the clean action $a _ { t } ,$ providing a more direct connection to traditional maximum likelihood estimation.

In the context of FPO, we can therefore express our advantage-weighted objective as:

$$
\max _ {\theta} \mathbb {E} _ {a _ {t} \sim \pi_ {\theta} (a _ {t} | o _ {t})} \left[ \mathrm{ELBO} _ {\theta} (a _ {t}) \cdot \hat {A} _ {t} \right] \tag {31}
$$

In this case, the objective direct increases a lower bound of the log-likelihood of clean actions $a _ { t }$ weighted by their advantages, rather than over noise-perturbed actions.

The FPO ratio in this case becomes:

$$
r ^ {\mathrm{FPO}} (\theta) = \frac {\exp \left(\mathrm{ELBO} _ {\theta} \left(a _ {t}\right)\right)}{\exp \left(\mathrm{ELBO} _ {\theta , \text { old }} \left(a _ {t}\right)\right)} \tag {32}
$$

This specific case highlights the close relationship between FPO and traditional maximum likelihood methods common for PPO [20]. FPO still retains the computational advantages of avoiding explicit likelihood computations.

As in the general case, our Monte Carlo estimator exhibits upward bias of gradient scale. We can use the same PPO clipping mechanism to control the magnitude of parameter changes.

# A.1.4 Advantage-Weighed Flow Matching Discussion

Advantage estimates are typically zero-centered to reduce variance in estimating the policy gradient. Flow matching, however, learns probability flows which must be nonnegative by construction. Since advantages function as loss weights in this context, they should remain positive for mathematical consistency. A constant shift does not affect policy gradient optimization, which follows from the same baseline-invariance property that justifies using advantages in the first place. We find that both processed and unprocessed advantages work empirically.

# A.2 GridWorld

Figure A.1 shows results from the Gaussian policy on the same Grid World trained using PPO. While the Gaussian policy can learn optimal behaviors, the trajectories resulting from it are not as diverse as those of the diffusion policy. We visualize 4 samples from the Gaussian policy with 0.0, 0.1, and 0.5 random noise perturbations at test time (Fig. A.1, right). Note that despite being initialized at the midpoint of the environment, all shown positions lead to a single goal mode, never both.

<table><tr><td rowspan="2">Learning Rate</td><td colspan="6">Clipping Epsilon ( $\varepsilon^{\text{clip}}$ )</td></tr><tr><td>0.3</td><td>0.2</td><td>0.1</td><td>0.05</td><td>0.03</td><td>0.01</td></tr><tr><td>0.0001</td><td>589.5</td><td>648.5</td><td>646.6</td><td>608.6</td><td>500.5</td><td>458.5</td></tr><tr><td>0.001</td><td>556.0</td><td>646.1</td><td>654.6</td><td>636.2</td><td>562.6</td><td>471.8</td></tr><tr><td>0.003</td><td>548.9</td><td>603.1</td><td>586.4</td><td>535.7</td><td>480.8</td><td>400.8</td></tr><tr><td>0.0003</td><td>567.0</td><td>631.8</td><td>667.8</td><td>650.9</td><td>570.4</td><td>492.0</td></tr><tr><td>0.0005</td><td>544.8</td><td>586.8</td><td>629.5</td><td>559.7</td><td>505.6</td><td>406.5</td></tr></table>

Table A.1: Hyperparameter sweep for Gaussian PPO on the subset of Playground tasks that we evaluate on. All quantities are average rewards across 10 tasks, with 5 seeds per task. The default configuration in Playground [14] (before tuning) uses learning rate 1e-3 and clipping epsilon 0.3; the tuned variant we use for results in the main paper body sets learning rate to 3e-4 and clipping epsilon to 0.1.

![](8a0b517e79ec50f3cbc957f6cee7144f4d10035597d46f23ed0bd37f598bc7fa.jpg)  
Figure A.2: Gaussian PPO baseline results before and after tuning. We tune clipping epsilon and learning rate to maximize average performance across tasks. Results show evaluation reward mean and standard error (y-axis) over 60M environment steps (x-axis). We run 5 seeds for each task; the curve with the highest terminal evaluation reward is bolded.

# A.3 MuJoCo Playground

Table A.2 shows hyperparameters used for PPO training in the MuJoCo Playground environment. These are imported directly from the configurations provided by MuJoCo Playground [14], but after sweeping hyperparameters to tune learning rate and clipping coefficients (Table A.1). We visualize improvements from this sweep in Figure A.2. Our flow matching and diffusion-based policies use the same hyperparameters, but adjust the clipping coefficient, turn off the entropy coefficient, and for DPPO [59], introduce a stochastic sampling variance to account for the change in policy representation.

# A.4 Humanoid Control

In Table A.3, we report the detailed hyperparameters that we used for training both the Gaussian policy with PPO and the Diffusion policy with FPO in the humanoid control experiment. Note that we use the same set of hyperparameters for both policies. In our project webpage, we also provide videos showing qualitative comparisons between the Gaussian policy and ours on tracking an under-conditioned reference, and visual results of FPO on different terrains.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="2">Discount factor (γ)</td><td>0.995 (most environments)</td></tr><tr><td>0.95 (BallInCup, FingerSpin)</td></tr><tr><td>GAE λ</td><td>0.95</td></tr><tr><td>Value loss coefficient</td><td>0.25</td></tr><tr><td>Entropy coefficient</td><td>0.01</td></tr><tr><td>Reward scaling</td><td>10.0</td></tr><tr><td>Normalize advantage</td><td>True</td></tr><tr><td>Normalize observations</td><td>True</td></tr><tr><td>Action repeat</td><td>1</td></tr><tr><td>Unroll length</td><td>30</td></tr><tr><td>Batch size</td><td>1024</td></tr><tr><td>Number of minibatches</td><td>32</td></tr><tr><td>Number of updates per batch</td><td>16</td></tr><tr><td>Number of environments</td><td>2048</td></tr><tr><td>Number of evaluations</td><td>10</td></tr><tr><td>Number of timesteps</td><td>60M</td></tr><tr><td>Policy network</td><td>MLP (4 hidden layers, 32 units)</td></tr><tr><td>Value network</td><td>MLP (5 hidden layers, 256 units)</td></tr><tr><td>Optimizer</td><td>Adam</td></tr></table>

Table A.2: PPO hyperparameters imported from MuJoCo playground [14].

<table><tr><td>Hyperparameter</td><td>Value</td><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="4">Policy Settings</td></tr><tr><td>Hidden size</td><td>512</td><td>Solver step size</td><td>0.1</td></tr><tr><td>Action perturbation std</td><td>0.05</td><td>Target KL divergence</td><td>None</td></tr><tr><td>Number of environments</td><td>4096</td><td>Normalize advantage</td><td>True</td></tr><tr><td colspan="4">Training Settings</td></tr><tr><td>Batch size</td><td>131072</td><td>Minibatch size</td><td>32768</td></tr><tr><td>Learning rate</td><td>0.0001</td><td>LR annealing</td><td>False</td></tr><tr><td>LR decay rate</td><td>1.5e-4</td><td>LR decay floor</td><td>0.2</td></tr><tr><td>Update epochs</td><td>4</td><td>L2 regularization coef.</td><td>0.0</td></tr><tr><td>GAE lambda</td><td>0.2</td><td>Discount factor (γ)</td><td>0.98</td></tr><tr><td>Clipping coefficient</td><td>0.01</td><td>Value function coefficient</td><td>1.2</td></tr><tr><td>Clip value loss</td><td>True</td><td>Value loss clip coefficient</td><td>0.2</td></tr><tr><td>Max gradient norm</td><td>10.0</td><td>Entropy coefficient</td><td>0.0</td></tr><tr><td>Discriminator coefficient</td><td>5.0</td><td>Bound coefficient</td><td>10.0</td></tr></table>

Table A.3: Policy training hyperparameters for humanoid control.

# A.5 Image Reward Fine-tuning

We explore fine-tuning a pre-trained image diffusion model on a non-differentiable task using the JPEG image compression gym proposed in DDPO [54]. We report this experiment as a negative result for FPO, due to the difficulty of fine-tuning diffusion models on their own output. Specifically, we find that repeatedly generating samples from a text-to-image diffusion model and training on them is highly unstable, even with manually-specified uniform advantages. We believe that this is related to classifier-free guidance (CFG) [84]. CFG is necessary to generate realistic images, however it is sensitive to hyperparameters, where too much or too little guidance introduces artifacts such as blur or oversaturation that do not reflect the original training data. Sometimes these artifacts are not visible to human eyes. These artifacts are further amplified over successive iterations of RL epochs, ultimately dominating the training signal.

![](4ec451875d97538532afed28f76adc4bcabb9cf8e38afe73688a137d1e2aed64.jpg)

<details>
<summary>natural_image</summary>

Brown butterfly resting on dry, cracked ground (no text or symbols visible)
</details>

![](f7bee7000f91b31dc9b8eaf06abc2b8d321698782a144c298f07fbeb80b5b278.jpg)

<details>
<summary>natural_image</summary>

Close-up of a brown butterfly perched on a green plant stem (no text or symbols visible)
</details>

![](efa4a290c88d686ea111a53b1648775ed66b2feaa413fb6b2b4da0ada807f9af.jpg)

<details>
<summary>natural_image</summary>

Close-up of a butterfly with transparent wings and visible wing patterns, set against a blurred green background (no text or symbols)
</details>

![](fed9436fc64d0ec94f27293eef1f3cc8f93af52943d36d2fcebc866d515d4d2d.jpg)

<details>
<summary>natural_image</summary>

Close-up of a textured green surface with wavy patterns and no visible text or symbols
</details>

![](293fbeb482501ac10ae014db380e7213823561fc27008414170a774b4d979291.jpg)

<details>
<summary>natural_image</summary>

Black-and-white photo of a tabby cat sitting outdoors, no visible text or symbols
</details>

![](4f1cc45a4736cbd3fda45a3c54800b9f5f0bbc3592a28f831b2d9b483e7aceee.jpg)

<details>
<summary>natural_image</summary>

Close-up of a gray tabby cat with striped collar and black spots, standing against a plain background (no text or symbols visible)
</details>

![](725a61e81a018f1b7accb7cda0ca6169687a7c8393d9aa72109e4a4301775197.jpg)

<details>
<summary>natural_image</summary>

Close-up of a black-and-white cat with striped whiskers, sitting on a plain surface (no text or symbols visible)
</details>

![](b298f8a8246ca7c8062adb693f04b54344333846fb4f7a3d899643058e5f7251.jpg)

<details>
<summary>natural_image</summary>

Close-up of a cat's face with detailed blue and orange patterned fur (no text or symbols)
</details>

Figure A.3: Image Generation at Different Training Steps. We generate images using Stable Diffusion 1.5 finetuned with FPO as training progresses. We manually set all advantages to 1 to eliminate the reward signal and investigate the dynamics of sampling from a text-to-image diffusion model then training on the results in a loop. In the top row, we display images from a training run using a classifier-free guidance (CFG) scale of 4. In the bottom row, we display images from a training run using a CFG scale of 2. Low CFG scales tend to encourage bluriness while high CFG scales encourage saturation and sharp geometric artifacts. Both diverge after a few hundred epochs even with tuned hyperparameters.

This phenomenon aligns with challenges previously identified in the literature on fine-tuning generative models on their own outputs [80–82]. To illustrate this, we fine-tune Stable Diffusion with all advantages set to 1 to eliminate the reward signal. This is equivalent to fine-tuning on self-generation data in an online manner. We explore CFG scales of 2 and 4 in Figure A.3. We find that both CFG scales induce quality regression. Specifically, the CFG scale of 2 makes the generation more blurry, while the scale of 2 causes the generated images to feature high saturation and geometry patterns. Both eventually diverge to abstract geometric patterns.