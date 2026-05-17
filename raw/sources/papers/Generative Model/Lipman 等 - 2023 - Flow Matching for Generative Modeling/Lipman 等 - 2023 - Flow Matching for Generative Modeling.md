# FLOW MATCHING FOR GENERATIVE MODELING

Yaron Lipman1,2 Ricky T. Q. Chen1 Heli Ben-Hamu2 Maximilian Nickel1 Matt Le1 1Meta AI (FAIR) 2Weizmann Institute of Science

# ABSTRACT

We introduce a new paradigm for generative modeling built on Continuous Normalizing Flows (CNFs), allowing us to train CNFs at unprecedented scale. Specifically, we present the notion of Flow Matching (FM), a simulation-free approach for training CNFs based on regressing vector fields of fixed conditional probability paths. Flow Matching is compatible with a general family of Gaussian probability paths for transforming between noise and data samples—which subsumes existing diffusion paths as specific instances. Interestingly, we find that employing FM with diffusion paths results in a more robust and stable alternative for training diffusion models. Furthermore, Flow Matching opens the door to training CNFs with other, non-diffusion probability paths. An instance of particular interest is using Optimal Transport (OT) displacement interpolation to define the conditional probability paths. These paths are more efficient than diffusion paths, provide faster training and sampling, and result in better generalization. Training CNFs using Flow Matching on ImageNet leads to consistently better performance than alternative diffusion-based methods in terms of both likelihood and sample quality, and allows fast and reliable sample generation using off-the-shelf numerical ODE solvers.

# 1 INTRODUCTION

Deep generative models are a class of deep learning algorithms aimed at estimating and sampling from an unknown data distribution. The recent influx of amazing advances in generative modeling, e.g., for image generation Ramesh et al. (2022); Rombach et al. (2022), is mostly facilitated by the scalable and relatively stable training of diffusion-based models Ho et al. (2020); Song et al. (2020b). However, the restriction to simple diffusion processes leads to a rather confined space of sampling probability paths, resulting in very long training times and the need to adopt specialized methods (e.g., Song et al. (2020a); Zhang & Chen (2022)) for efficient sampling.

In this work we consider the general and deterministic framework of Continuous Normalizing Flows (CNFs; Chen et al. (2018)). CNFs are capable of modeling arbitrary probability path

and are in particular known to encompass the probability paths modeled by diffusion processes (Song et al., 2021). However, aside from diffusion that can be trained efficiently via, e.g., denoising score matching (Vincent, 2011), no scalable CNF training algorithms are known. Indeed, maximum likelihood training (e.g., Grathwohl et al. (2018)) require expensive numerical ODE simulations, while existing simulation-free methods either involve intractable integrals (Rozen et al., 2021) or biased gradients (Ben-Hamu et al., 2022).

The goal of this work is to propose Flow Matching (FM), an efficient simulation-free approach to training CNF models, allowing the adoption of general probability paths to supervise CNF training. Importantly, FM breaks the barriers for scalable CNF training beyond diffusion, and sidesteps the need to reason about diffusion processes to directly work with probability paths.

![](images/fc02f342f561902536cbf5b90176f0ccd872b394d373be2d841652e06f17afad.jpg)  
Figure 1: Unconditional ImageNet-128 samples of a CNF trained using Flow Matching with Optimal Transport probability paths.

In particular, we propose the Flow Matching objective (Section 3), a simple and intuitive training objective to regress onto a target vector field that generates a desired probability path. We first show that we can construct such target vector fields through per-example (i.e., conditional) formulations. Then, inspired by denoising score matching, we show that a per-example training objective, termed Conditional Flow Matching (CFM), provides equivalent gradients and does not require explicit knowledge of the intractable target vector field. Furthermore, we discuss a general family of per-example probability paths (Section 4) that can be used for Flow Matching, which subsumes existing diffusion paths as special instances. Even on diffusion paths, we find that using FM provides more robust and stable training, and achieves superior performance compared to score matching. Furthermore, this family of probability paths also includes a particularly interesting case: the vector field that corresponds to an Optimal Transport (OT) displacement interpolant (McCann, 1997). We find that conditional OT paths are simpler than diffusion paths, forming straight line trajectories whereas diffusion paths result in curved paths. These properties seem to empirically translate to faster training, faster generation, and better performance.

We empirically validate Flow Matching and the construction via Optimal Transport paths on ImageNet, a large and highly diverse image dataset. We find that we can easily train models to achieve favorable performance in both likelihood estimation and sample quality amongst competing diffusion-based methods. Furthermore, we find that our models produce better trade-offs between computational cost and sample quality compared to prior methods. Figure 1 depicts selected unconditional ImageNet $1 2 8 \times 1 2 8$ samples from our model.

# 2 PRELIMINARIES: CONTINUOUS NORMALIZING FLOWS

Let $\mathbb { R } ^ { d }$ denote the data space with data points $x = ( x ^ { 1 } , \ldots , x ^ { d } ) \in \mathbb { R } ^ { d }$ . Two important objects we use in this paper are: the probability density path $p : [ 0 , 1 ] \times \mathbb { R } ^ { d }  \mathbb { R } _ { > 0 }$ , which is a time dependent1 probability density function, i.e., $\textstyle \int p _ { t } ( x ) d x \ = { \dot { 1 } }$ , and a time-dependent vector field, $v : [ 0 , 1 ] \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ . A vector field $v _ { t }$ can be used to construct a time-dependent diffeomorphic map, called a flow, $\phi : [ 0 , 1 ] \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , defined via the ordinary differential equation (ODE):

$$
\frac {d}{d t} \phi_ {t} (x) = v _ {t} \left(\phi_ {t} (x)\right) \tag {1}
$$

$$
\phi_ {0} (x) = x \tag {2}
$$

Previously, Chen et al. (2018) suggested modeling the vector field $v _ { t }$ with a neural network, $v _ { t } ( x ; \theta )$ , where $\theta \in \mathbb { R } ^ { p }$ are its learnable parameters, which in turn leads to a deep parametric model of the flow $\phi _ { t }$ , called a Continuous Normalizing Flow (CNF). A CNF is used to reshape a simple prior density $p _ { 0 }$ (e.g., pure noise) to a more complicated one, $p _ { 1 }$ , via the push-forward equation

$$
p _ {t} = \left[ \phi_ {t} \right] _ {*} p _ {0} \tag {3}
$$

where the push-forward (or change of variables) operator $^ *$ is defined by

$$
\left[ \phi_ {t} \right] _ {*} p _ {0} (x) = p _ {0} \left(\phi_ {t} ^ {- 1} (x)\right) \det  \left[ \frac {\partial \phi_ {t} ^ {- 1}}{\partial x} (x) \right]. \tag {4}
$$

A vector field $v _ { t }$ is said to generate a probability density path $p _ { t }$ if its flow $\phi _ { t }$ satisfies equation 3. One practical way to test if a vector field generates a probability path is using the continuity equation, which is a key component in our proofs, see Appendix B. We recap more information on CNFs, in particular how to compute the probability $p _ { 1 } ( x )$ at an arbitrary point $x \in \mathbb { R } ^ { d }$ in Appendix C.

# 3 FLOW MATCHING

Let $x _ { 1 }$ denote a random variable distributed according to some unknown data distribution $q ( x _ { 1 } )$ . We assume we only have access to data samples from $q ( x _ { 1 } )$ but have no access to the density function itself. Furthermore, we let $p _ { t }$ be a probability path such that $p _ { 0 } = p$ is a simple distribution, e.g., the standard normal distribution $p ( x ) = \mathcal { N } ( x | 0 , I )$ , and let $p _ { 1 }$ be approximately equal in distribution to $q$ . We will later discuss how to construct such a path. The Flow Matching objective is then designed to match this target probability path, which will allow us to flow from $p _ { 0 }$ to $p _ { 1 }$ .

Given a target probability density path $p _ { t } ( x )$ and a corresponding vector field $u _ { t } ( x )$ , which generates $p _ { t } ( x )$ , we define the Flow Matching (FM) objective as

$$
\mathcal {L} _ {\mathrm {F M}} (\theta) = \mathbb {E} _ {t, p _ {t} (x)} \| v _ {t} (x) - u _ {t} (x) \| ^ {2}, \tag {5}
$$

where $\theta$ denotes the learnable parameters of the CNF vector field $v _ { t }$ (as defined in Section 2), $t \sim$ $\boldsymbol { \mathcal { U } } [ 0 , 1 ]$ (uniform distribution), and $x \sim p _ { t } ( x )$ . Simply put, the FM loss regresses the vector field $u _ { t }$ with a neural network $v _ { t }$ . Upon reaching zero loss, the learned CNF model will generate $p _ { t } ( x )$ .

Flow Matching is a simple and attractive objective, but na¨ıvely on its own, it is intractable to use in practice since we have no prior knowledge for what an appropriate $p _ { t }$ and $u _ { t }$ are. There are many choices of probability paths that can satisfy $p _ { 1 } ( x ) \approx q \bar { ( x ) }$ , and more importantly, we generally don’t have access to a closed form $u _ { t }$ that generates the desired $p _ { t }$ . In this section, we show that we can construct both $p _ { t }$ and $u _ { t }$ using probability paths and vector fields that are only defined per sample, and an appropriate method of aggregation provides the desired $p _ { t }$ and $u _ { t }$ . Furthermore, this construction allows us to create a much more tractable objective for Flow Matching.

# 3.1 CONSTRUCTING $p _ { t } , u _ { t }$ FROM CONDITIONAL PROBABILITY PATHS AND VECTOR FIELDS

A simple way to construct a target probability path is via a mixture of simpler probability paths: Given a particular data sample $x _ { 1 }$ we denote by $p _ { t } ( x | x _ { 1 } )$ a conditional probability path such that it satisfies $p _ { 0 } ( x | x _ { 1 } ) = p ( x )$ at time $t = 0$ , and we design $p _ { 1 } ( x | x _ { 1 } )$ at $t = 1$ to be a distribution concentrated around $x = x _ { 1 }$ , e.g., $p _ { 1 } ( x | x _ { 1 } ) = \mathcal { N } ( x | x _ { 1 } , \sigma ^ { 2 } I )$ , a normal distribution with $x _ { 1 }$ mean and a sufficiently small standard deviation $\sigma > 0$ . Marginalizing the conditional probability paths over $q ( x _ { 1 } )$ give rise to the marginal probability path

$$
p _ {t} (x) = \int p _ {t} (x | x _ {1}) q \left(x _ {1}\right) d x _ {1}, \tag {6}
$$

where in particular at time $t = 1$ , the marginal probability $p _ { 1 }$ is a mixture distribution that closely approximates the data distribution $q$ ,

$$
p _ {1} (x) = \int p _ {1} (x | x _ {1}) q \left(x _ {1}\right) d x _ {1} \approx q (x). \tag {7}
$$

Interestingly, we can also define a marginal vector field, by “marginalizing” over the conditional vector fields in the following sense (we assume $p _ { t } ( x ) > 0$ for all $t$ and $x$ ):

$$
u _ {t} (x) = \int u _ {t} \left(x \mid x _ {1}\right) \frac {p _ {t} \left(x \mid x _ {1}\right) q \left(x _ {1}\right)}{p _ {t} (x)} d x _ {1}, \tag {8}
$$

where $u _ { t } ( \cdot | x _ { 1 } ) : \mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ is a conditional vector field that generates $p _ { t } ( \cdot | x _ { 1 } )$ . It may not seem apparent, but this way of aggregating the conditional vector fields actually results in the correct vector field for modeling the marginal probability path.

Our first key observation is this:

The marginal vector field (equation 8) generates the marginal probability path (equation 6).

This provides a surprising connection between the conditional VFs (those that generate conditional probability paths) and the marginal VF (those that generate the marginal probability path). This connection allows us to break down the unknown and intractable marginal VF into simpler conditional VFs, which are much simpler to define as these only depend on a single data sample. We formalize this in the following theorem.

Theorem 1. Given vector fields $u _ { t } ( x | x _ { 1 } )$ that generate conditional probability paths $p _ { t } ( x | x _ { 1 } )$ , for any distribution $q ( x _ { 1 } )$ , the marginal vector field $u _ { t }$ in equation 8 generates the marginal probability path $p _ { t }$ in equation 6, i.e., $u _ { t }$ and $p _ { t }$ satisfy the continuity equation (equation 26).

The full proofs for our theorems are all provided in Appendix A. Theorem 1 can also be derived from the Diffusion Mixture Representation Theorem in Peluchetti (2021) that provides a formula for the marginal drift and diffusion coefficients in diffusion SDEs.

# 3.2 CONDITIONAL FLOW MATCHING

Unfortunately, due to the intractable integrals in the definitions of the marginal probability path and VF (equations 6 and 8), it is still intractable to compute $u _ { t }$ , and consequently, intractable to na¨ıvely compute an unbiased estimator of the original Flow Matching objective. Instead, we propose a simpler objective, which surprisingly will result in the same optima as the original objective. Specifically, we consider the Conditional Flow Matching (CFM) objective,

$$
\mathcal {L} _ {\mathrm {C F M}} (\theta) = \mathbb {E} _ {t, q (x _ {1}), p _ {t} (x | x _ {1})} \| v _ {t} (x) - u _ {t} (x | x _ {1}) \| ^ {2}, \tag {9}
$$

where $t \sim \mathcal { U } [ 0 , 1 ]$ , $x _ { 1 } \sim q ( x _ { 1 } )$ , and now $x \sim p _ { t } ( x | x _ { 1 } )$ . Unlike the FM objective, the CFM objective allows us to easily sample unbiased estimates as long as we can efficiently sample from $p _ { t } ( x | x _ { 1 } )$ and compute $u _ { t } ( x | x _ { 1 } )$ , both of which can be easily done as they are defined on a per-sample basis. Our second key observation is therefore:

The FM (equation 5) and CFM (equation 9) objectives have identical gradients w.r.t. $\theta$ .

That is, optimizing the CFM objective is equivalent (in expectation) to optimizing the FM objective. Consequently, this allows us to train a CNF to generate the marginal probability path $p _ { t }$ —which in particular, approximates the unknown data distribution $q$ at $t { = } 1 { - }$ without ever needing access to either the marginal probability path or the marginal vector field. We simply need to design suitable conditional probability paths and vector fields. We formalize this property in the following theorem.

Theorem 2. Assuming that $p _ { t } ( x ) > 0$ for all $x ~ \in ~ \mathbb { R } ^ { d }$ and $t \in [ 0 , 1 ]$ , then, up to a constant independent of $\theta$ , $\mathcal { L } _ { c F M }$ and $\mathcal { L } _ { \boldsymbol { F M } }$ are equal. Hence, $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \boldsymbol { F M } } ( \boldsymbol { \theta } ) = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \boldsymbol { c F M } } ( \boldsymbol { \theta } )$ .

# 4 CONDITIONAL PROBABILITY PATHS AND VECTOR FIELDS

The Conditional Flow Matching objective works with any choice of conditional probability path and conditional vector fields. In this section, we discuss the construction of $p _ { t } ( x | x _ { 1 } )$ and $u _ { t } ( x | x _ { 1 } )$ for a general family of Gaussian conditional probability paths. Namely, we consider conditional probability paths of the form

$$
p _ {t} (x \mid x _ {1}) = \mathcal {N} (x \mid \mu_ {t} (x _ {1}), \sigma_ {t} (x _ {1}) ^ {2} I), \tag {10}
$$

where $\mu : [ 0 , 1 ] \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ is the time-dependent mean of the Gaussian distribution, while $\sigma :$ $[ 0 , 1 ] \times \mathbb { R } \to \mathbb { R } _ { > 0 }$ describes a time-dependent scalar standard deviation (std). We set $\mu _ { 0 } ( x _ { 1 } ) = 0$ and $\sigma _ { 0 } ( x _ { 1 } ) = 1$ , so that all conditional probability paths converge to the same standard Gaussian noise distribution at $t = 0$ , $p ( x ) = \mathcal { N } ( x | 0 , I )$ . We then set $\mu _ { 1 } ( x _ { 1 } ) = x _ { 1 }$ and $\sigma _ { 1 } ( x _ { 1 } ) = \sigma _ { \mathrm { { m i n } } }$ , which is set sufficiently small so that $p _ { 1 } ( x | x _ { 1 } )$ is a concentrated Gaussian distribution centered at $x _ { 1 }$ .

There is an infinite number of vector fields that generate any particular probability path (e.g., by adding a divergence free component to the continuity equation, see equation 26), but the vast majority of these is due to the presence of components that leave the underlying distribution invariant—for instance, rotational components when the distribution is rotation-invariant—leading to unnecessary extra compute. We decide to use the simplest vector field corresponding to a canonical transformation for Gaussian distributions. Specifically, consider the flow (conditioned on $x _ { 1 }$ )

$$
\psi_ {t} (x) = \sigma_ {t} \left(x _ {1}\right) x + \mu_ {t} \left(x _ {1}\right). \tag {11}
$$

When $x$ is distributed as a standard Gaussian, $\psi _ { t } ( x )$ is the affine transformation that maps to a normally-distributed random variable with mean $\mu _ { t } ( x _ { 1 } )$ and std $\sigma _ { t } ( x _ { 1 } )$ . That is to say, according to equation 4, $\psi _ { t }$ pushes the noise distribution $p _ { 0 } ( x | x _ { 1 } ) = p ( x )$ to $p _ { t } ( x | x _ { 1 } )$ , i.e.,

$$
\left[ \psi_ {t} \right] _ {*} p (x) = p _ {t} \left(x \mid x _ {1}\right). \tag {12}
$$

This flow then provides a vector field that generates the conditional probability path:

$$
\frac {d}{d t} \psi_ {t} (x) = u _ {t} \left(\psi_ {t} (x) \mid x _ {1}\right). \tag {13}
$$

Reparameterizing $p _ { t } ( x | x _ { 1 } )$ in terms of just $x _ { 0 }$ and plugging equation 13 in the CFM loss we get

$$
\mathcal {L} _ {\mathrm {C F M}} (\theta) = \mathbb {E} _ {t, q (x _ {1}), p (x _ {0})} \left\| v _ {t} \left(\psi_ {t} \left(x _ {0}\right)\right) - \frac {d}{d t} \psi_ {t} \left(x _ {0}\right) \right\| ^ {2}. \tag {14}
$$

Since $\psi _ { t }$ is a simple (invertible) affine map we can use equation 13 to solve for $u _ { t }$ in a closed form. Let $f ^ { \prime }$ denote the derivative with respect to time, i.e., $\textstyle f ^ { \prime } { \overset { \cdot } { = } } { \frac { d } { d t } } f$ , for a time-dependent function $f$ .

Theorem 3. Let $p _ { t } ( x | x _ { 1 } )$ be a Gaussian probability path as in equation $I O$ , and $\psi _ { t }$ its corresponding flow map as in equation 11. Then, the unique vector field that defines $\psi _ { t }$ has the form:

$$
u _ {t} (x \mid x _ {1}) = \frac {\sigma_ {t} ^ {\prime} \left(x _ {1}\right)}{\sigma_ {t} \left(x _ {1}\right)} \left(x - \mu_ {t} \left(x _ {1}\right)\right) + \mu_ {t} ^ {\prime} \left(x _ {1}\right). \tag {15}
$$

Consequently, $u _ { t } ( x | x _ { 1 } )$ generates the Gaussian path $p _ { t } ( x | x _ { 1 } )$ .

# 4.1 SPECIAL INSTANCES OF GAUSSIAN CONDITIONAL PROBABILITY PATHS

Our formulation is fully general for arbitrary functions $\mu _ { t } ( x _ { 1 } )$ and $\sigma _ { t } ( x _ { 1 } )$ , and we can set them to any differentiable function satisfying the desired boundary conditions. We first discuss the special cases that recover probability paths corresponding to previously-used diffusion processes. Since we directly work with probability paths, we can simply depart from reasoning about diffusion processes altogether. Therefore, in the second example below, we directly formulate a probability path based on the Wasserstein-2 optimal transport solution as an interesting instance.

Example I: Diffusion conditional VFs. Diffusion models start with data points and gradually add noise until it approximates pure noise. These can be formulated as stochastic processes, which have strict requirements in order to obtain closed form representation at arbitrary times $t$ , resulting in Gaussian conditional probability paths $p _ { t } ( x | x _ { 1 } )$ with specific choices of mean $\mu _ { t } ( x _ { 1 } )$ and std $\sigma _ { t } ( x _ { 1 } )$ (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020b). For example, the reversed (noise data) Variance Exploding (VE) path has the form

$$
p _ {t} (x) = \mathcal {N} \left(x \mid x _ {1}, \sigma_ {1 - t} ^ {2} I\right), \tag {16}
$$

where $\sigma _ { t }$ is an increasing function, $\sigma _ { 0 } = 0$ , and $\sigma _ { 1 } \gg 1$ . Next, equation 16 provides the choices of $\mu _ { t } ( x _ { 1 } ) = x _ { 1 }$ and $\sigma _ { t } ( x _ { 1 } ) = \sigma _ { 1 - t }$ . Plugging these into equation 15 of Theorem 3 we get

$$
u _ {t} (x \mid x _ {1}) = - \frac {\sigma_ {1 - t} ^ {\prime}}{\sigma_ {1 - t}} (x - x _ {1}). \tag {17}
$$

The reversed (noise data) Variance Preserving (VP) diffusion path has the form

$$
p _ {t} (x \mid x _ {1}) = \mathcal {N} (x \mid \alpha_ {1 - t} x _ {1}, \left(1 - \alpha_ {1 - t} ^ {2}\right) I), \text {w h e r e} \alpha_ {t} = e ^ {- \frac {1}{2} T (t)}, T (t) = \int_ {0} ^ {t} \beta (s) d s, \tag {18}
$$

and $\beta$ is the noise scale function. Equation 18 provides the choices of $\mu _ { t } ( x _ { 1 } ) \ = \ \alpha _ { 1 - t } x _ { 1 }$ and $\sigma _ { t } ( x _ { 1 } ) = \sqrt { 1 - \alpha _ { 1 - t } ^ { 2 } }$ . Plugging these into equation 15 of Theorem 3 we get

$$
u _ {t} \left(x \mid x _ {1}\right) = \frac {\alpha_ {1 - t} ^ {\prime}}{1 - \alpha_ {1 - t} ^ {2}} \left(\alpha_ {1 - t} x - x _ {1}\right) = - \frac {T ^ {\prime} (1 - t)}{2} \left[ \frac {e ^ {- T (1 - t)} x - e ^ {- \frac {1}{2} T (1 - t)} x _ {1}}{1 - e ^ {- T (1 - t)}} \right]. \tag {19}
$$

Our construction of the conditional VF $\dot { \boldsymbol u } _ { t } ( \boldsymbol x | \boldsymbol x _ { 1 } )$ does in fact coincide with the vector field previously used in the deterministic probability flow (Song et al. (2020b), equation 13) when restricted to these conditional diffusion processes; see details in Appendix D. Nevertheless, combining the diffusion conditional VF with the Flow Matching objective offers an attractive training alternative—which we find to be more stable and robust in our experiments—to existing score matching approaches.

Another important observation is that, as these probability paths were previously derived as solutions of diffusion processes, they do not actually reach a true noise distribution in finite time. In practice, $p _ { 0 } ( x )$ is simply approximated by a suitable Gaussian distribution for sampling and likelihood evaluation. Instead, our construction provides full control over the probability path, and we can just directly set $\mu _ { t }$ and $\sigma _ { t }$ , as we will do next.

Example II: Optimal Transport conditional VFs. An arguably more natural choice for conditional probability paths is to define the mean and the std to simply change linearly in time, i.e.,

$$
\mu_ {t} (x) = t x _ {1}, \text {a n d} \sigma_ {t} (x) = 1 - \left(1 - \sigma_ {\min }\right) t. \tag {20}
$$

According to Theorem 3 this path is generated by the VF

$$
u _ {t} \left(x \mid x _ {1}\right) = \frac {x _ {1} - \left(1 - \sigma_ {\min }\right) x}{1 - \left(1 - \sigma_ {\min }\right) t}, \tag {21}
$$

![](images/9ae6f9ad2c2b4a5253536a67ec1a07c3679d476691051364a4d28f44a5bdd108.jpg)

![](images/5e8db6f64cc221f016cb99df5c04f1868ea3bb4efd408e2c066486fb96c3bea0.jpg)

![](images/13c19448a8af8660bfc78216899b0f4cde45a071484f19836692ee8983169476.jpg)

![](images/fa01303a618772c357aef12ad5c04cd2a61baad9af717dab7d2021ad5f279e51.jpg)

t = 0.0   
t = 1/3   
t = 2/3   
t = 1.0   
![](images/691857bddc89d11aea33ffe06d1300889537d3c68e5a98c448214bbb46c86712.jpg)  
Diffusion path – conditional score function

![](images/5ef7684a5eab8e692bb5531d03c08c00b934fe2c94605f59eafd104a107ae03d.jpg)

![](images/e210abb94179f0c811f31d5047e6441cfe9c66001ee6cb1842631c9c0818e85f.jpg)

![](images/7413f01575fef32ab5b3c83500425b54922a75de672999b5a2df57f619a568d3.jpg)

Figure 2: Compared to the diffusion path’s conditional score function, the OT path’s conditional vector field has constant direction in time and is arguably simpler to fit with a parametric model. Note the blue color denotes larger magnitude while red color denotes smaller magnitude.   
![](images/774a173d7b3cafe12fa042f60c39133c23027deb584baf354be7ddfdd746f685.jpg)  
OT path – conditional vector field

t = 0.0

$t = 1 / 3$

t = 2/3

$t = 1 . 0$

which, in contrast to the diffusion conditional VF (equation 19), is defined for all $t \in [ 0 , 1 ]$ . The conditional flow that corresponds to $u _ { t } ( x | x _ { 1 } )$ is

$$
\psi_ {t} (x) = \left(1 - \left(1 - \sigma_ {\min }\right) t\right) x + t x _ {1}, \tag {22}
$$

and in this case, the CFM loss (see equations 9, 14) takes the form:

$$
\mathcal {L} _ {\mathrm {C F M}} (\theta) = \mathbb {E} _ {t, q (x _ {1}), p (x _ {0})} \left\| v _ {t} \left(\psi_ {t} (x _ {0})\right) - \left(x _ {1} - \left(1 - \sigma_ {\min }\right) x _ {0}\right) \right\| ^ {2}. \tag {23}
$$

Allowing the mean and std to change linearly not only leads to simple and intuitive paths, but it is actually also optimal in the following sense. The conditional flow $\psi _ { t } ( x )$ is in fact the Optimal Transport (OT) displacement map between the two Gaussians $p _ { 0 } ( x | x _ { 1 } )$ and $p _ { 1 } ( x | x _ { 1 } )$ . The OT interpolant, which is a probability path, is defined to be (see Definition 1.1 in McCann (1997)):

$$
p _ {t} = [ (1 - t) \mathrm {i d} + t \psi ] _ {\star} p _ {0} \tag {24}
$$

where $\psi : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ is the OT map pushing $p _ { 0 }$ to $p _ { 1 }$ , id denotes the identity map, i.e., $\operatorname { i d } ( x ) = x$ and $( 1 - t ) \mathrm { i d } + t \psi$ is called the OT displacement map. Example 1.7 in McCann (1997) shows, that in our case of two Gaussians where the first is a standard one, the OT displacement map takes the form of equation 22.

Intuitively, particles under the OT displacement map always move in straight line trajectories and with constant speed. Figure 3 depicts sampling paths for the diffusion and OT conditional VFs. Interestingly, we find that sampling trajectory from diffusion paths can “overshoot” the final sample, resulting in unnecessary backtracking, whilst the OT paths are guaranteed to stay straight.

![](images/d45ae2bf5adb27806aa4375a7550f3116d03b8cd31613de7917da7db73f20f8e.jpg)  
Diffusion

![](images/e32678f5266f802ab3c13b3b7ec1b625f3cf7ec873d49db4965ed0faff4a42c3.jpg)  
OT   
Figure 3: Diffusion and OT trajectories.

Figure 2 compares the diffusion conditional score function (the re-

gression target in a typical diffusion methods), i.e., $\nabla \log p _ { t } ( x | x _ { 1 } )$ with $p _ { t }$ defined as in equation 18, with the OT conditional VF (equation 21). The start $\left( p _ { 0 } \right)$ and end $( p _ { 1 } )$ Gaussians are identical in both examples. An interesting observation is that the OT VF has a constant direction in time, which arguably leads to a simpler regression task. This property can also be verified directly from equation 21 as the VF can be written in the form $u _ { t } ( x | x _ { 1 } ) = g ( t ) h ( x | x _ { 1 } )$ . Figure 8 in the Appendix shows a visualization of the Diffusion VF. Lastly, we note that although the conditional flow is optimal, this by no means imply that the marginal VF is an optimal transport solution. Nevertheless, we expect the marginal vector field to remain relatively simple.

# 5 RELATED WORK

Continuous Normalizing Flows were introduced in (Chen et al., 2018) as a continuous-time version of Normalizing Flows (see e.g., Kobyzev et al. (2020); Papamakarios et al. (2021) for an overview). Originally, CNFs are trained with the maximum likelihood objective, but this involves expensive ODE simulations for the forward and backward propagation, resulting in high time complexity due to the sequential nature of ODE simulations. Although some works demonstrated the capability of CNF generative models for image synthesis (Grathwohl et al., 2018), scaling up to very high dimensional images is inherently difficult. A number of works attempted to regularize the ODE to be easier to solve, e.g., using augmentation (Dupont et al., 2019), adding regularization terms (Yang & Karniadakis, 2019; Finlay et al., 2020; Onken et al., 2021; Tong et al., 2020; Kelly et al., 2020), or stochastically sampling the integration interval (Du et al., 2022). These works merely aim to regularize the ODE but do not change the fundamental training algorithm.

![](images/8d8f7aa0121817b8e3d65c6fe26ceccbb809ede652c0a2b1d3eb4b5550eae8cf.jpg)  
Figure 4: (left) Trajectories of CNFs trained with different objectives on 2D checkerboard data. The OT path introduces the checkerboard pattern much earlier, while FM results in more stable training. (right) FM with OT results in more efficient sampling, solved using the midpoint scheme.

In order to speed up CNF training, some works have developed simulation-free CNF training frameworks by explicitly designing the target probability path and the dynamics. For instance, Rozen et al. (2021) consider a linear interpolation between the prior and the target density but involves integrals that were difficult to estimate in high dimensions, while Ben-Hamu et al. (2022) consider general probability paths similar to this work but suffers from biased gradients in the stochastic minibatch regime. In contrast, the Flow Matching framework allows simulation-free training with unbiased gradients and readily scales to very high dimensions.

Another approach to simulation-free training relies on the construction of a diffusion process to indirectly define the target probability path (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song & Ermon, 2019). Song et al. (2020b) shows that diffusion models are trained using denoising score matching (Vincent, 2011), a conditional objective that provides unbiased gradients with respect to the score matching objective. Conditional Flow Matching draws inspiration from this result, but generalizes to matching vector fields directly. Due to the ease of scalability, diffusion models have received increased attention, producing a variety of improvements such as loss-rescaling (Song et al., 2021), adding classifier guidance along with architectural improvements (Dhariwal & Nichol, 2021), and learning the noise schedule (Nichol & Dhariwal, 2021; Kingma et al., 2021). However, (Nichol & Dhariwal, 2021) and (Kingma et al., 2021) only consider a restricted setting of Gaussian conditional paths defined by simple diffusion processes with a single parameter—in particular, it does not include our conditional OT path. In an another line of works, (De Bortoli et al., 2021; Wang et al., 2021; Peluchetti, 2021) proposed finite time diffusion constructions via diffusion bridges theory resolving the approximation error incurred by infinite time denoising constructions. While existing works make use of a connection between diffusion processes and continuous normalizing flows with the same probability path (Maoutsa et al., 2020b; Song et al., 2020b; 2021), our work allows us to generalize beyond the class of probability paths modeled by simple diffusion. With our work, it is possible to completely sidestep the diffusion process construction and reason directly with probability paths, while still retaining efficient training and log-likelihood evaluations. Lastly, concurrently to our work (Liu et al., 2022; Albergo & Vanden-Eijnden, 2022) arrived at similar conditional objectives for simulation-free training of CNFs, while Neklyudov et al. (2023) derived an implicit objective when $u _ { t }$ is assumed to be a gradient field.

# 6 EXPERIMENTS

We explore the empirical benefits of using Flow Matching on the image datasets of CIFAR-10 (Krizhevsky et al., 2009) and ImageNet at resolutions 32, 64, and 128 (Chrabaszcz et al., 2017; Deng et al., 2009). We also ablate the choice of diffusion path in Flow Matching, particularly between the standard variance preserving diffusion path and the optimal transport path. We discuss how sample generation is improved by directly parameterizing the generating vector field and using the Flow Matching objective. Lastly we show Flow Matching can also be used in the conditional generation setting. Unless otherwise specified, we evaluate likelihood and samples from the model using dopri5 (Dormand & Prince, 1980) at absolute and relative tolerances of 1e-5. Generated samples can be found in the Appendix, and all implementation details are in Appendix E.

Table 1: Likelihood (BPD), quality of generated samples (FID), and evaluation time (NFE) for the same model trained with different methods.   

<table><tr><td rowspan="2">Model</td><td colspan="3">CIFAR-10</td><td colspan="3">ImageNet 32×32</td><td colspan="3">ImageNet 64×64</td><td rowspan="2">Model</td><td colspan="2">ImageNet 128×128</td></tr><tr><td>NLL↓</td><td>FID↓</td><td>NFE↓</td><td>NLL↓</td><td>FID↓</td><td>NFE↓</td><td>NLL↓</td><td>FID↓</td><td>NFE↓</td><td>NLL↓</td><td>FID↓</td></tr><tr><td>Ablations</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>MGAN (Hoang et al., 2018)</td><td>-</td><td>58.9</td></tr><tr><td>DDPM</td><td>3.12</td><td>7.48</td><td>274</td><td>3.54</td><td>6.99</td><td>262</td><td>3.32</td><td>17.36</td><td>264</td><td>PacGAN2 (Lin et al., 2018)</td><td>-</td><td>57.5</td></tr><tr><td>Score Matching</td><td>3.16</td><td>19.94</td><td>242</td><td>3.56</td><td>5.68</td><td>178</td><td>3.40</td><td>19.74</td><td>441</td><td>Logo-GAN-AE (Sage et al., 2018)</td><td>-</td><td>50.9</td></tr><tr><td>ScoreFlow</td><td>3.09</td><td>20.78</td><td>428</td><td>3.55</td><td>14.14</td><td>195</td><td>3.36</td><td>24.95</td><td>601</td><td>Self-cond. GAN (Lučić et al., 2019)</td><td>-</td><td>41.7</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Uncond. BigGAN (Lučić et al., 2019)</td><td>-</td><td>25.3</td></tr><tr><td>FM w/ Diffusion</td><td>3.10</td><td>8.06</td><td>183</td><td>3.54</td><td>6.37</td><td>193</td><td>3.33</td><td>16.88</td><td>187</td><td>PGMGAN (Armandpour et al., 2021)</td><td>-</td><td>21.7</td></tr><tr><td>FM w/ OT</td><td>2.99</td><td>6.35</td><td>142</td><td>3.53</td><td>5.02</td><td>122</td><td>3.31</td><td>14.45</td><td>138</td><td>FM w/ OT</td><td>2.90</td><td>20.9</td></tr></table>

![](images/f8181327fe0f8afecdcd060b04f4e81bfa2bf6d69c50b8b513e25501da92df97.jpg)  
Score Matching w/ Diffusion

![](images/0ece500ff3051652795a28d1eadbac304822e5caab1cb55a2f7ab60511d58fad.jpg)  
Flow Matching w/ Diffusion

![](images/e6cacdb1618bda5d3f493a9ef1650b14b07f1d2c547d821696cc1d99442918f8.jpg)  
Flow Matching w/ OT   
Figure 6: Sample paths from the same initial noise with models trained on ImageNet $6 4 \times 6 4$ . The OT path reduces noise roughly linearly, while diffusion paths visibly remove noise only towards the end of the path. Note also the differences between the generated images.

# 6.1 DENSITY MODELING AND SAMPLE QUALITY ON IMAGENET

We start by comparing the same model architecture, i.e., the U-Net architecture from Dhariwal & Nichol (2021) with minimal changes, trained on CIFAR-10, and ImageNet 32/64 with different popular diffusion-based losses: DDPM from (Ho et al., 2020), Score Matching (SM) (Song et al., 2020b), and Score Flow (SF) (Song et al., 2021); see Appendix E.1 for exact details. Table 1 (left) summarizes our results alongside these baselines reporting negative log-likelihood (NLL) in units of bits per dimension (BPD), sample quality as measured by the Frechet Inception Distance (FID; Heusel et al. (2017)), and averaged number of function evaluations (NFE) required for the adaptive solver to reach its a prespecified numerical tolerance, averaged over 50k samples. All models are trained using the same architecture, hyperparameter values and number of training iterations, where baselines are allowed more iterations for better convergence. Note that these are unconditional models. On both CIFAR-10 and ImageNet, FM-OT consistently obtains best results across all our quantitative measures compared to competing methods. We are noticing a higher that usual FID performance in CIFAR-10 compared to previous works (Ho et al., 2020; Song et al., 2020b; 2021) that can possibly be explained by the fact that our used architecture was not optimized for CIFAR-10.

Secondly, Table 1 (right) compares a model trained using Flow Matching with the OT path on ImageNet at resolution $1 2 8 \times 1 2 8$ . Our FID is state-of-the-art with the exception of IC-GAN (Casanova et al., 2021) which uses conditioning with a self-supervised ResNet50 model, and therefore is left out of this table. Figures 11, 12, 13 in the Appendix show non-curated samples from these models.

Faster training. While existing works train diffusion models with a very high number of iterations (e.g., $1 . 3 \mathrm { m }$ and 10m iterations are reported by Score Flow and VDM, respectively), we find that Flow Matching generally converges much faster. Figure 5 shows FID curves during training of Flow Matching and all baselines for ImageNet $6 4 \times 6 4$ ; FM-OT is able to lower the FID faster and to a greater extent than the alternatives. For ImageNet-128 Dhariwal & Nichol (2021) train for $4 . 3 6 \mathrm { m }$ iterations with batch size 256, while FM (with $2 5 \%$ larger model) used $5 0 0 \mathrm { k }$ iterations with batch size $1 . 5 \mathrm { k } .$ , i.e., $33 \%$ less image

![](images/38f1815a3214da1def7b5874550814b7c7cb198a8d98145c71d70850bfebcd3d.jpg)  
Figure 5: Image quality during training, ImageNet $6 4 \times 6 4$ .

throughput; see Table 3 for exact details. Furthermore, the cost of sampling from a model can drastically change during training for score matching, whereas the sampling cost stays constant when training with Flow Matching (Figure 10 in Appendix).

# 6.2 SAMPLING EFFICIENCY

For sampling, we first draw a random noise sample $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ then compute $\phi _ { 1 } ( x _ { 0 } )$ by solving equation 1 with the trained VF, $v _ { t }$ , on the interval $t \in [ 0 , 1 ]$ using an ODE solver. While diffusion

![](images/cee77bb0f17ba1d00dd9be5a0817fc5897caff60583993a1eed92690bf26ce85.jpg)  
Error of ODE solution

![](images/1ab106745ad98dfb1db8c371f399e3a14c30c32cae8927485bddfc2d399b030e.jpg)  
Flow matching w/ OT

![](images/a3ebcbbc711d2acdab418b89b77298b909328a457ff8924bb0172d32780f72ff.jpg)  
Flow matching w/ Diffusion

![](images/d48546cb84737c5820809a78c6c37a65147ce9770fd12b783a67bd57fb48529e.jpg)  
Score matching w/ Diffusion   
Figure 7: Flow Matching, especially when using OT paths, allows us to use fewer evaluations for sampling while retaining similar numerical error (left) and sample quality (right). Results are shown for models trained on ImageNet $3 2 \times 3 2$ , and numerical errors are for the midpoint scheme.

models can also be sampled through an SDE formulation, this can be highly inefficient and many methods that propose fast samplers (e.g., Song et al. (2020a); Zhang & Chen (2022)) directly make use of the ODE perspective (see Appendix D). In part, this is due to ODE solvers being much more efficient—yielding lower error at similar computational costs (Kloeden et al., 2012)—and the multitude of available ODE solver schemes. When compared to our ablation models, we find that models trained using Flow Matching with the OT path always result in the most efficient sampler, regardless of ODE solver, as demonstrated next.

Sample paths. We first qualitatively visualize the difference in sampling paths between diffusion and OT. Figure 6 shows samples from ImageNet-64 models using identical random seeds, where we find that the OT path model starts generating images sooner than the diffusion path models, where noise dominates the image until the very last time point. We additionally depict the probability density paths in 2D generation of a checkerboard pattern, Figure 4 (left), noticing a similar trend.

Low-cost samples. We next switch to fixed-step solvers and compare low $( \leq 1 0 0 )$ NFE samples computed with the ImageNet-32 models from Table 1. In Figure 7 (left), we compare the per-pixel MSE of low NFE solutions compared with 1000 NFE solutions (we use 256 random noise seeds), and notice that the FM with OT model produces the best numerical error, in terms of computational cost, requiring roughly only $60 \%$ of the NFEs to reach the same error threshold as diffusion models. Secondly, Figure 7 (right) shows how FID changes as a result of the computational cost, where we find FM with OT is able to achieve decent FID even at very low NFE values, producing better tradeoff between sample quality and cost compared to ablated models. Figure 4 (right) shows low-cost sampling effects for the 2D checkerboard experiment.

# 6.3 CONDITIONAL SAMPLING FROM LOW-RESOLUTION IMAGES

Lastly, we experimented with Flow Matching for conditional image generation. In particular, upsampling images from $6 4 \times 6 4$ to $2 5 6 \times 2 5 6$ . We follow the evaluation procedure in (Saharia et al., 2022) and compute the FID of the upsampled validation images; baselines include reference (FID of original validation set), and regression. Results are in Table 2. Upsampled image samples are shown in Figures 14, 15 in the Appendix. FM-OT achieves similar PSNR and SSIM values to (Saharia et al., 2022) while

Table 2: Image super-resolution on the ImageNet validation set.   

<table><tr><td>Model</td><td>FID↓</td><td>IS↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>Reference</td><td>1.9</td><td>240.8</td><td>-</td><td>-</td></tr><tr><td>Regression</td><td>15.2</td><td>121.1</td><td>27.9</td><td>0.801</td></tr><tr><td>SR3 (Saharia et al., 2022)</td><td>5.2</td><td>180.1</td><td>26.4</td><td>0.762</td></tr><tr><td>FM w/ OT</td><td>3.4</td><td>200.8</td><td>24.7</td><td>0.747</td></tr></table>

considerably improving on FID and IS, which as argued by (Saharia et al., 2022) is a better indication of generation quality.

# 7 CONCLUSION

We introduced Flow Matching, a new simulation-free framework for training Continuous Normalizing Flow models, relying on conditional constructions to effortlessly scale to very high dimensions. Furthermore, the FM framework provides an alternative view on diffusion models, and suggests forsaking the stochastic/diffusion construction in favor of more directly specifying the probability path, allowing us to, e.g., construct paths that allow faster sampling and/or improve generation. We experimentally showed the ease of training and sampling when using the Flow Matching framework, and in the future, we expect FM to open the door to allowing a multitude of probability paths (e.g., non-isotropic Gaussians or more general kernels altogether).

# 8 SOCIAL RESPONSIBILITY

Along side its many positive applications, image generation can also be used for harmful proposes. Using content-controlled training sets and image validation/classification can help reduce these uses. Furthermore, the energy demand for training large deep learning models is increasing at a rapid pace (Amodei et al., 2018; Thompson et al., 2020), focusing on methods that are able to train using less gradient updates / image throughput can lead to significant time and energy savings.

# REFERENCES

Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022.   
Dario Amodei, Danny Hernandez, Girish SastryJack, Jack Clark, Greg Brockman, and Ilya Sutskever. Ai and compute. https://openai.com/blog/ai-and-compute/, 2018.   
Mohammadreza Armandpour, Ali Sadeghian, Chunyuan Li, and Mingyuan Zhou. Partition-guided gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5099–5109, 2021.   
Heli Ben-Hamu, Samuel Cohen, Joey Bose, Brandon Amos, Aditya Grover, Maximilian Nickel, Ricky T. Q. Chen, and Yaron Lipman. Matching normalizing flows and probability paths on manifolds. arXiv preprint arXiv:2207.04711, 2022.   
Arantxa Casanova, Marlene Careil, Jakob Verbeek, Michal Drozdzal, and Adriana Romero Soriano. Instance-conditioned gan. Advances in Neural Information Processing Systems, 34:27517–27529, 2021.   
Ricky T. Q. Chen. torchdiffeq, 2018. URL https://github.com/rtqichen/ torchdiffeq.   
Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018.   
Patryk Chrabaszcz, Ilya Loshchilov, and Frank Hutter. A downsampled variant of imagenet as an alternative to the cifar datasets. arXiv preprint arXiv:1707.08819, 2017.   
Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schrodinger ¨ bridge with applications to score-based generative modeling. (arXiv:2106.01357), Dec 2021. doi: 10.48550/arXiv.2106.01357. URL http://arxiv.org/abs/2106.01357. arXiv:2106.01357 [cs, math, stat].   
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.   
Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id= AAWuCvzaVt.   
John R Dormand and Peter J Prince. A family of embedded runge-kutta formulae. Journal of computational and applied mathematics, 6(1):19–26, 1980.   
Shian Du, Yihong Luo, Wei Chen, Jian Xu, and Delu Zeng. To-flow: Efficient continuous normalizing flows with temporal optimization adjoint with moving speed, 2022. URL https: //arxiv.org/abs/2203.10335.   
Emilien Dupont, Arnaud Doucet, and Yee Whye Teh. Augmented neural odes. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d´ Alche-Buc, E. Fox, and R. Gar- ´ nett (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper/2019/file/ 21be9a4bd4f81549a9d1d241981cec3c-Paper.pdf.

Chris Finlay, Jorn-Henrik Jacobsen, Levon Nurbekyan, and Adam M. Oberman. How to train your ¨ neural ode: the world of jacobian and kinetic regularization. In ICML, pp. 3154–3164, 2020. URL http://proceedings.mlr.press/v119/finlay20a.html.   
Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. Ffjord: Free-form continuous dynamics for scalable reversible generative models, 2018. URL https: //arxiv.org/abs/1810.01367.   
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.   
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.   
Quan Hoang, Tu Dinh Nguyen, Trung Le, and Dinh Phung. Mgan: Training generative adversarial nets with multiple generators. In International conference on learning representations, 2018.   
Jacob Kelly, Jesse Bettencourt, Matthew J Johnson, and David K Duvenaud. Learning differential equations that are easy to solve. Advances in Neural Information Processing Systems, 33:4370– 4380, 2020.   
Diederik P Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id= 2LdBqxc1Yv.   
Peter Eris Kloeden, Eckhard Platen, and Henri Schurz. Numerical solution of SDE through computer experiments. Springer Science & Business Media, 2012.   
Ivan Kobyzev, Simon JD Prince, and Marcus A Brubaker. Normalizing flows: An introduction and review of current methods. IEEE transactions on pattern analysis and machine intelligence, 43 (11):3964–3979, 2020.   
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.   
Zinan Lin, Ashish Khetan, Giulia Fanti, and Sewoong Oh. Pacgan: The power of two samples in generative adversarial networks. Advances in neural information processing systems, 31, 2018.   
Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.   
Mario Luciˇ c, Michael Tschannen, Marvin Ritter, Xiaohua Zhai, Olivier Bachem, and Sylvain Gelly. ´ High-fidelity image generation with fewer labels. In International conference on machine learning, pp. 4183–4192. PMLR, 2019.   
Dimitra Maoutsa, Sebastian Reich, and Manfred Opper. Interacting particle solutions of fokker–planck equations through gradient–log–density estimation. Entropy, 22(8):802, jul 2020a. doi: 10.3390/e22080802. URL https://doi.org/10.3390%2Fe22080802.   
Dimitra Maoutsa, Sebastian Reich, and Manfred Opper. Interacting particle solutions of fokker– planck equations through gradient–log–density estimation. Entropy, 22(8):802, 2020b.   
Robert J McCann. A convexity principle for interacting gases. Advances in mathematics, 128(1): 153–179, 1997.   
Kirill Neklyudov, Daniel Severo, and Alireza Makhzani. Action matching: A variational method for learning stochastic dynamics from samples, 2023. URL https://openreview.net/ forum?id=T6HPzkhaKeS.   
Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pp. 8162–8171. PMLR, 2021.

Derek Onken, Samy Wu Fung, Xingjian Li, and Lars Ruthotto. Ot-flow: Fast and accurate continuous normalizing flows via optimal transport. Proceedings of the AAAI Conference on Artificial Intelligence, 35(10):9223–9232, May 2021. URL https://ojs.aaai.org/index.php/ AAAI/article/view/17113.   
George Papamakarios, Eric T Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. J. Mach. Learn. Res., 22(57):1–64, 2021.   
Stefano Peluchetti. Non-denoising forward-time diffusions. 2021.   
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨ resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.   
Noam Rozen, Aditya Grover, Maximilian Nickel, and Yaron Lipman. Moser flow: Divergencebased generative modeling on manifolds. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=qGvMv3undNJ.   
Alexander Sage, Eirikur Agustsson, Radu Timofte, and Luc Van Gool. Logo synthesis and manipulation with clustered generative adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5879–5888, 2018.   
Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.   
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pp. 2256–2265. PMLR, 2015.   
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020a.   
Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d´ Alche-Buc, E. Fox, and R. Gar- ´ nett (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper/2019/file/ 3001ef257407d5a371a96dcd947c7d93-Paper.pdf.   
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020b.   
Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of scorebased diffusion models. In Thirty-Fifth Conference on Neural Information Processing Systems, 2021.   
Neil C Thompson, Kristjan Greenewald, Keeheon Lee, and Gabriel F Manso. The computational limits of deep learning. arXiv preprint arXiv:2007.05558, 2020.   
Alexander Tong, Jessie Huang, Guy Wolf, David Van Dijk, and Smita Krishnaswamy. Trajectorynet: A dynamic optimal transport network for modeling cellular dynamics. In International conference on machine learning, pp. 9526–9536. PMLR, 2020.   
Cedric Villani. ´ Optimal transport: old and new, volume 338. Springer, 2009.   
Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011.

Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, and Can Yang. Deep generative learning via schrodinger bridge. (arXiv:2106.10410), Jul 2021. doi: 10.48550/arXiv.2106.10410. URL ¨ http://arxiv.org/abs/2106.10410. arXiv:2106.10410 [cs].   
Liu Yang and George E. Karniadakis. Potential flow generator with $\$ 123$ optimal transport regularity for generative models. CoRR, abs/1908.11462, 2019. URL http://arxiv.org/abs/ 1908.11462.   
Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022.

# A THEOREM PROOFS

Theorem 1. Given vector fields $u _ { t } ( x | x _ { 1 } )$ that generate conditional probability paths $p _ { t } ( x | x _ { 1 } )$ , for any distribution $q ( x _ { 1 } )$ , the marginal vector field $u _ { t }$ in equation 8 generates the marginal probability path $p _ { t }$ in equation $6$ , i.e., $u _ { t }$ and $p _ { t }$ satisfy the continuity equation (equation 26).

Proof. To verify this, we check that $p _ { t }$ and $u _ { t }$ satisfy the continuity equation (equation 26):

$$
\begin{array}{l} \frac {d}{d t} p _ {t} (x) = \int \left(\frac {d}{d t} p _ {t} (x | x _ {1})\right) q (x _ {1}) d x _ {1} = - \int \operatorname {d i v} \left(u _ {t} (x | x _ {1}) p _ {t} (x | x _ {1})\right) q (x _ {1}) d x _ {1} \\ = - \mathrm {d i v} \Big (\int u _ {t} (x | x _ {1}) p _ {t} (x | x _ {1}) q (x _ {1}) d x _ {1} \Big) = - \mathrm {d i v} \Big (u _ {t} (x) p _ {t} (x) \Big), \\ \end{array}
$$

where in the second equality we used the fact that $u _ { t } ( \cdot | x _ { 1 } )$ generates $p _ { t } ( \cdot | x _ { 1 } )$ , in the last equality we used equation 8. Furthermore, the first and third equalities are justified by assuming the integrands satisfy the regularity conditions of the Leibniz Rule (for exchanging integration and differentiation).

![](images/6f1d8e5c29d9908d12a8a47b67e76822bed7357017e7ca6dc9ce47aa8fa4cb57.jpg)

Theorem 2. Assuming that $p _ { t } ( x ) > 0$ for all $x ~ \in ~ \mathbb { R } ^ { d }$ and $t \in [ 0 , 1 ]$ , then, up to a constant independent of $\theta$ , $\mathcal { L } _ { c F M }$ and $\mathcal { L } _ { \boldsymbol { F M } }$ are equal. Hence, $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \boldsymbol { F M } } ( \boldsymbol { \theta } ) = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \boldsymbol { c F M } } ( \boldsymbol { \theta } )$ .

Proof. To ensure existence of all integrals and to allow the changing of integration order (by Fubini’s Theorem) in the following we assume that $q ( x )$ and $p _ { t } ( x | x _ { 1 } )$ are decreasing to zero at a sufficient speed as $\| x \| \to \infty$ , and that $u _ { t } , v _ { t } , \nabla _ { \theta } v _ { t }$ are bounded.

First, using the standard bilinearity of the 2-norm we have that

$$
\begin{array}{l} \left\| v _ {t} (x) - u _ {t} (x) \right\| ^ {2} = \left\| v _ {t} (x) \right\| ^ {2} - 2 \left\langle v _ {t} (x), u _ {t} (x) \right\rangle + \left\| u _ {t} (x) \right\| ^ {2} \\ \left\| v _ {t} (x) - u _ {t} (x | x _ {1}) \right\| ^ {2} = \left\| v _ {t} (x) \right\| ^ {2} - 2 \left\langle v _ {t} (x), u _ {t} (x | x _ {1}) \right\rangle + \left\| u _ {t} (x | x _ {1}) \right\| ^ {2} \\ \end{array}
$$

Next, remember that $u _ { t }$ is independent of $\theta$ and note that

$$
\begin{array}{l} \mathbb {E} _ {p _ {t} (x)} \| v _ {t} (x) \| ^ {2} = \int \| v _ {t} (x) \| ^ {2} p _ {t} (x) d x = \int \| v _ {t} (x) \| ^ {2} p _ {t} (x | x _ {1}) q (x _ {1}) d x _ {1} d x \\ = \mathbb {E} _ {q (x _ {1}), p _ {t} (x | x _ {1})} \| v _ {t} (x) \| ^ {2}, \\ \end{array}
$$

where in the second equality we use equation 6, and in the third equality we change the order of integration. Next,

$$
\begin{array}{l} \mathbb {E} _ {p _ {t} (x)} \left\langle v _ {t} (x), u _ {t} (x) \right\rangle = \int \left\langle v _ {t} (x), \frac {\int u _ {t} (x | x _ {1}) p _ {t} (x | x _ {1}) q (x _ {1}) d x _ {1}}{p _ {t} (x)} \right\rangle p _ {t} (x) d x \\ = \int \left\langle v _ {t} (x), \int u _ {t} (x | x _ {1}) p _ {t} (x | x _ {1}) q (x _ {1}) d x _ {1} \right\rangle d x \\ = \int \left\langle v _ {t} (x), u _ {t} (x | x _ {1}) \right\rangle p _ {t} (x | x _ {1}) q (x _ {1}) d x _ {1} d x \\ = \mathbb {E} _ {q (x _ {1}), p _ {t} (x | x _ {1})} \left\langle v _ {t} (x), u _ {t} (x | x _ {1}) \right\rangle , \\ \end{array}
$$

where in the last equality we change again the order of integration.

![](images/0265f83a467d1c8628e01ac5076806ea64eb90d3396af769d61dee61a816acce.jpg)

Theorem 3. Let $p _ { t } ( x | x _ { 1 } )$ be a Gaussian probability path as in equation 10, and $\psi _ { t }$ its corresponding flow map as in equation 11. Then, the unique vector field that defines $\psi _ { t }$ has the form:

$$
u _ {t} (x \mid x _ {1}) = \frac {\sigma_ {t} ^ {\prime} \left(x _ {1}\right)}{\sigma_ {t} \left(x _ {1}\right)} \left(x - \mu_ {t} \left(x _ {1}\right)\right) + \mu_ {t} ^ {\prime} \left(x _ {1}\right). \tag {15}
$$

Consequently, $u _ { t } ( x | x _ { 1 } )$ generates the Gaussian path $p _ { t } ( x | x _ { 1 } )$ .

Proof. For notational simplicity let $w _ { t } ( x ) = u _ { t } ( x | x _ { 1 } )$ . Now consider equation 1:

$$
\frac {d}{d t} \psi_ {t} (x) = w _ {t} (\psi_ {t} (x)).
$$

Since $\psi _ { t }$ is invertible (as $\sigma _ { t } ( x _ { 1 } ) > 0 \mathrm { ~ , ~ }$ ) we let $x = \psi ^ { - 1 } ( y )$ and get

$$
\psi_ {t} ^ {\prime} (\psi^ {- 1} (y)) = w _ {t} (y), \tag {25}
$$

where we used the apostrophe notation for the derivative to emphasis that $\psi _ { t } ^ { \prime }$ is evaluated at $\psi ^ { - 1 } ( y )$ . Now, inverting $\psi _ { t } ( x )$ provides

$$
\psi_ {t} ^ {- 1} (y) = \frac {y - \mu_ {t} (x _ {1})}{\sigma_ {t} (x _ {1})}.
$$

Differentiating $\psi _ { t }$ with respect to $t$ gives

$$
\psi_ {t} ^ {\prime} (x) = \sigma_ {t} ^ {\prime} \left(x _ {1}\right) x + \mu_ {t} ^ {\prime} \left(x _ {1}\right).
$$

Plugging these last two equations in equation 25 we get

$$
w _ {t} (y) = \frac {\sigma_ {t} ^ {\prime} (x _ {1})}{\sigma_ {t} (x _ {1})} (y - \mu_ {t} (x _ {1})) + \mu_ {t} ^ {\prime} (x _ {1})
$$

as required.

![](images/bf38dda69920736edd4b93ec4b86d2b39262a05f66cf94b1b9357ae19b8db120.jpg)

# B THE CONTINUITY EQUATION

One method of testing if a vector field $v _ { t }$ generates a probability path $p _ { t }$ is the continuity equation (Villani, 2009). It is a Partial Differential Equation (PDE) providing a necessary and sufficient condition to ensuring that a vector field $v _ { t }$ generates $p _ { t }$ ,

$$
\frac {d}{d t} p _ {t} (x) + \operatorname {d i v} \left(p _ {t} (x) v _ {t} (x)\right) = 0, \tag {26}
$$

where the divergence operator, div, is defined with respect to the spatial variable $\boldsymbol { x } = ( x ^ { 1 } , \dots , x ^ { d } )$ , i.e., div = Pdi=1 ∂∂xi . $\begin{array} { r } { \mathrm { d i v } = \sum _ { i = 1 } ^ { d } \frac { \partial } { \partial x ^ { i } } } \end{array}$

# C COMPUTING PROBABILITIES OF THE CNF MODEL

We are given an arbitrary data point $x _ { 1 } \in \mathbb { R } ^ { d }$ and need to compute the model probability at that point, i.e., $p _ { 1 } ( x _ { 1 } )$ . Below we recap how this can be done covering the basic relevant ODEs, the scaling of the divergence computation, taking into account data transformations (e.g., centering of data), and Bits-Per-Dimension computation.

ODE for computing $p _ { 1 } ( x _ { 1 } )$ The continuity equation with equation 1 lead to the instantaneous change of variable (Chen et al., 2018; Ben-Hamu et al., 2022):

$$
\frac {d}{d t} \log p _ {t} \left(\phi_ {t} (x)\right) + \operatorname {d i v} \left(v _ {t} \left(\phi_ {t} (x)\right) = 0. \right.
$$

Integrating $t \in [ 0 , 1 ]$ gives:

$$
\log p _ {1} \left(\phi_ {1} (x)\right) - \log p _ {0} \left(\phi_ {0} (x)\right) = - \int_ {0} ^ {1} \operatorname {d i v} \left(v _ {t} \left(\phi_ {t} (x)\right)\right) d t \tag {27}
$$

Therefore, the log probability can be computed together with the flow trajectory by solving the ODE:

$$
\frac {d}{d t} \left[ \begin{array}{c} \phi_ {t} (x) \\ f (t) \end{array} \right] = \left[ \begin{array}{c} v _ {t} \left(\phi_ {t} (x)\right) \\ - \operatorname {d i v} \left(v _ {t} \left(\phi_ {t} (x)\right)\right) \end{array} \right] \tag {28}
$$

Given initial conditions

$$
\left[ \begin{array}{l} \phi_ {0} (x) \\ f (0) \end{array} \right] = \left[ \begin{array}{l} x _ {0} \\ c \end{array} \right]. \tag {29}
$$

the solution $\left[ \phi _ { t } ( x ) , f ( t ) \right] ^ { T }$ is uniquely defined (up to some mild conditions on the $\mathrm { V F } v _ { t } .$ ). Denote $x _ { 1 } = \phi _ { 1 } ( x )$ , and according to equation 27,

$$
f (1) = c + \log p _ {1} \left(x _ {1}\right) - \log p _ {0} \left(x _ {0}\right). \tag {30}
$$

Now, we are given an arbitrary $x _ { 1 }$ and want to compute $p _ { 1 } ( x _ { 1 } )$ . For this end, we will need to solve equation 28 in reverse. That is,

$$
\frac {d}{d s} \left[ \begin{array}{l} \phi_ {1 - s} (x) \\ f (1 - s) \end{array} \right] = \left[ \begin{array}{c} - v _ {1 - s} \left(\phi_ {1 - s} (x)\right) \\ \operatorname {d i v} \left(v _ {1 - s} \left(\phi_ {1 - s} (x)\right)\right) \end{array} \right] \tag {31}
$$

and we solve this equation for $s \in [ 0 , 1 ]$ with the initial conditions at $s = 0$ :

$$
\left[ \begin{array}{l} \phi_ {1} (x) \\ f (1) \end{array} \right] = \left[ \begin{array}{l} x _ {1} \\ 0 \end{array} \right]. \tag {32}
$$

From uniqueness of ODEs, the solution will be identical to the solution of equation 28 with initial conditions in equation 29 where $c = \log p _ { 0 } ( x _ { 0 } ) - \log p _ { 1 } ( x _ { 1 } )$ . This can be seen from equation 30 and setting $f ( 1 ) = 0$ . Therefore we get that

$$
f (0) = \log p _ {0} \left(x _ {0}\right) - \log p _ {1} \left(x _ {1}\right)
$$

and consequently

$$
\log p _ {1} \left(x _ {1}\right) = \log p _ {0} \left(x _ {0}\right) - f (0). \tag {33}
$$

To summarize, to compute $p _ { 1 } ( x _ { 1 } )$ we first solve the ODE in equation 31 with initial conditions in equation 32, and the compute equation 33.

Unbiased estimator to $p _ { 1 } ( x _ { 1 } )$ Solving equation 31 requires computation of div of VFs in $\mathbb { R } ^ { d }$ which is costly. Grathwohl et al. (2018) suggest to replace the divergence by the (unbiased) Hutchinson trace estimator,

$$
\frac {d}{d s} \left[ \begin{array}{l} \phi_ {1 - s} (x) \\ \tilde {f} (1 - s) \end{array} \right] = \left[ \begin{array}{c} - v _ {1 - s} (\phi_ {1 - s} (x)) \\ z ^ {T} D v _ {1 - s} (\phi_ {1 - s} (x)) z \end{array} \right], \tag {34}
$$

where $z \in \mathbb { R } ^ { d }$ is a sample from a random variable such that $\mathbb { E } z z ^ { T } = I$ . Solving the ODE in equation 34 exactly (in practice, with a small controlled error) with initial conditions in equation 32 leads to

$$
\begin{array}{l} \mathbb {E} _ {z} \left[ \log p _ {0} (x _ {0}) - \tilde {f} (0) \right] = \log p _ {0} (x _ {0}) - \mathbb {E} _ {z} \left[ \tilde {f} (0) - \tilde {f} (1) \right] \\ = \log p _ {0} (x _ {0}) - \mathbb {E} _ {z} \left[ \int_ {0} ^ {1} z ^ {T} D v _ {1 - s} (\phi_ {1 - s} (x)) z d s \right] \\ = \log p _ {0} (x _ {0}) - \int_ {0} ^ {1} \mathbb {E} _ {z} \left[ z ^ {T} D v _ {1 - s} (\phi_ {1 - s} (x)) z \right] d s \\ = \log p _ {0} (x _ {0}) - \int_ {0} ^ {1} \operatorname {d i v} \left(v _ {1 - s} \left(\phi_ {1 - s} (x)\right)\right) d s \\ = \log p _ {0} (x _ {0}) - (f (0) - f (1)) \\ = \log p _ {0} (x _ {0}) - (\log p _ {0} (x _ {0}) - \log p _ {1} (x _ {1})) \\ = \log p _ {1} (x _ {1}), \\ \end{array}
$$

where in the third equality we switched order of integration assuming the sufficient condition of Fubini’s theorem hold, and in the previous to last equality we used equation 30. Therefore the random variable

$$
\log p _ {0} \left(x _ {0}\right) - \tilde {f} (0) \tag {35}
$$

is an unbiased estimator for $\log p _ { 1 } ( x _ { 1 } )$ . To summarize, for a scalable unbiased estimation of $p _ { 1 } ( x _ { 1 } )$ we first solve the ODE in equation 34 with initial conditions in equation 32, and then output equation 35.

Transformed data Often, before training our generative model we transform the data, e.g., we scale and/or translate the data. Such a transformation is denoted by $\varphi ^ { - 1 } : \mathbb { R } ^ { d } \to \mathbb { R } ^ { \dot { d } }$ and our generative model becomes a composition

$$
\psi (x) = \varphi \circ \phi (x)
$$

where $\phi : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ is the model we train. Given a prior probability $p _ { 0 }$ we have that the push forward of this probability under $\psi$ (equation 3 and equation 4) takes the form

$$
\begin{array}{l} p _ {1} (x) = \psi_ {*} p _ {0} (x) = p _ {0} \left(\phi^ {- 1} \left(\varphi^ {- 1} (x)\right)\right) \det  \left[ D \phi^ {- 1} \left(\varphi^ {- 1} (x)\right) \right] \det  \left[ D \varphi^ {- 1} (x) \right] \\ = \left(\phi_ {*} p _ {0} \left(\varphi^ {- 1} (x)\right)\right) \det  \left[ D \varphi^ {- 1} (x) \right] \\ \end{array}
$$

and therefore

$$
\log p _ {1} (x) = \log \phi_ {*} p _ {0} \left(\varphi^ {- 1} (x)\right) + \log \det  \left[ D \varphi^ {- 1} (x) \right].
$$

For images $d = H \times W \times 3$ and we consider a transform $\phi$ that maps each pixel value from $[ - 1 , 1 ]$ to [0, 256]. Therefore,

$$
\varphi (y) = 2 ^ {7} (y + 1)
$$

and

$$
\varphi^ {- 1} (x) = 2 ^ {- 7} x - 1
$$

For this case we have

$$
\log p _ {1} (x) = \log \phi_ {*} p _ {0} \left(\varphi^ {- 1} (x)\right) - 7 d \log 2. \tag {36}
$$

Bits-Per-Dimension (BPD) computation BPD is defined by

$$
\operatorname {B P D} = \mathbb {E} _ {x _ {1}} \left[ - \frac {\log_ {2} p _ {1} \left(x _ {1}\right)}{d} \right] = \mathbb {E} _ {x _ {1}} \left[ - \frac {\log p _ {1} \left(x _ {1}\right)}{d \log 2} \right] \tag {37}
$$

Following equation 36 we get

$$
\mathrm {B P D} = - \frac {\log \phi_ {*} p _ {0} (\varphi^ {- 1} (x))}{d \log 2} + 7
$$

and $\log { \phi _ { * } p _ { 0 } ( \varphi ^ { - 1 } ( x ) ) }$ is approximated using the unbiased estimator in equation 35 over the transformed data $\varphi ^ { - 1 } ( x _ { 1 } )$ . Averaging the unbiased estimator on a large test test $x _ { 1 }$ provides a good approximation to the test set BPD.

# D DIFFUSION CONDITIONAL VECTOR FIELDS

We derive the vector field governing the Probability Flow ODE (equation 13 in Song et al. (2020b)) for the VE and VP diffusion paths (equation 18) and note that it coincides with the conditional vector fields we derive using Theorem 3, namely the vector fields defined in equations 16 and 19.

We start with a short primer on how to find a conditional vector field for the probability path described by the Fokker-Planck equation, then instantiate it for the VE and VP probability paths.

Since in the diffusion literature the diffusion process runs from data at time $t = 0$ to noise at time $t = 1$ , we will need the following lemma to translate the diffusion VFs to our convention of $t = 0$ corresponds to noise and $t = 1$ corresponds to data:

Lemma 1. Consider a flow defined by a vector field $u _ { t } ( x )$ generating probability density path $p _ { t } ( x )$ . Then, the vector field $\tilde { u } _ { t } ( x ) = - u _ { 1 - t } ( x )$ generates the path $\tilde { p } _ { t } ( x ) = p _ { 1 - t } ( x )$ when initiated from $\tilde { p } _ { 0 } ( x ) = p _ { 1 } ( x )$ .

Proof. We use the continuity equation (equation 26):

$$
\begin{array}{l} \frac {d}{d t} \tilde {p} _ {t} (x) = \frac {d}{d t} p _ {1 - t} (x) = - p _ {1 - t} ^ {\prime} (x) \\ = \operatorname {d i v} \left(p _ {1 - t} (x) u _ {1 - t} (x)\right) \\ = - \operatorname {d i v} \left(\tilde {p} _ {t} (x) (- u _ {1 - t} (x))\right) \\ \end{array}
$$

and therefore $\tilde { u } _ { t } ( x ) = - u _ { 1 - t } ( x )$ generates $\tilde { p } _ { t } ( x )$ .

![](images/cd0b11812703f5d27641a06f2320767cad4308a87bacdfae00d14d3dac806da8.jpg)

Conditional VFs for Fokker-Planck probability paths Consider a Stochastic Differential Equation (SDE) of the standard form

$$
d y = f _ {t} d t + g _ {t} d w \tag {38}
$$

with time parameter $t$ , drift $f _ { t }$ , diffusion coefficient $g _ { t }$ , and $d w$ is the Wiener process. The solution $y _ { t }$ to the SDE is a stochastic process, i.e., a continuous time-dependent random variable, the probability density of which, $p _ { t } ( y _ { t } )$ , is characterized by the Fokker-Planck equation:

$$
\frac {d p _ {t}}{d t} = - \operatorname {d i v} \left(f _ {t} p _ {t}\right) + \frac {g _ {t} ^ {2}}{2} \Delta p _ {t} \tag {39}
$$

where $\Delta$ represents the Laplace operator (in $y ,$ ), namely $\operatorname { d i v } \nabla$ , where $\nabla$ is the gradient operator (also in $y$ ). Rewriting this equation in the form of the continuity equation can be done as follows (Maoutsa et al., 2020a):

$$
\frac {d p _ {t}}{d t} = - \mathrm {d i v} \Big (f _ {t} p _ {t} - \frac {g ^ {2}}{2} \frac {\nabla p _ {t}}{p _ {t}} p _ {t} \Big) = - \mathrm {d i v} \Big (\big (f _ {t} - \frac {g _ {t} ^ {2}}{2} \nabla \log p _ {t} \big) p _ {t} \Big) = - \mathrm {d i v} \Big (w _ {t} p _ {t} \Big)
$$

where the vector field

$$
w _ {t} = f _ {t} - \frac {g _ {t} ^ {2}}{2} \nabla \log p _ {t} \tag {40}
$$

satisfies the continuity equation with the probability path $p _ { t }$ , and therefore generates $p _ { t }$

Variance Exploding (VE) path The SDE for the VE path is

$$
d y = \sqrt {\frac {d}{d t} \sigma_ {t} ^ {2}} d w,
$$

where $\sigma _ { 0 } = 0$ and increasing to infinity as $t  1$ . The SDE is moving from data, $y _ { 0 }$ , at $t = 0$ to noise, $y _ { 1 }$ , at $t = 1$ with the probability path

$$
p _ {t} (y | y _ {0}) = \mathcal {N} (y | y _ {0}, \sigma_ {t} ^ {2} I).
$$

The conditional VF according to equation 40 is:

$$
w _ {t} (y | y _ {0}) = \frac {\sigma_ {t} ^ {\prime}}{\sigma_ {t}} (y - y _ {0})
$$

Using Lemma 1 we get that the probability path

$$
\tilde {p} _ {t} (y | y _ {0}) = \mathcal {N} (y | y _ {0}, \sigma_ {1 - t} ^ {2} I)
$$

is generated by

$$
\tilde {w} _ {t} (y | y _ {0}) = - \frac {\sigma_ {1 - t} ^ {\prime}}{\sigma_ {1 - t}} (y - y _ {0}),
$$

which coincides with equation 17.

Variance Preserving (VP) path The SDE for the VP path is

$$
d y = - \frac {T ^ {\prime} (t)}{2} y + \sqrt {T ^ {\prime} (t)} d w,
$$

where $\begin{array} { r } { T ( t ) = \int _ { 0 } ^ { t } \beta ( s ) d s } \end{array}$ , $t \in [ 0 , 1 ]$ . The SDE coefficients are therefore

$$
f _ {s} (y) = - \frac {T ^ {\prime} (s)}{2} y, \quad g _ {s} = \sqrt {T ^ {\prime} (s)}
$$

and

$$
p _ {t} (y | y _ {0}) = \mathcal {N} (y | e ^ {- \frac {1}{2} T (t)} y _ {0}, (1 - e ^ {- T (t)}) I).
$$

Plugging these choices in equation 40 we get the conditional VF

$$
w _ {t} \left(y \mid y _ {0}\right) = \frac {T ^ {\prime} (t)}{2} \left(\frac {y - e ^ {- \frac {1}{2} T (t)} y _ {0}}{1 - e ^ {- T (t)}} - y\right) \tag {41}
$$

Using Lemma 1 to reverse the time we get the conditional VF for the reverse probability path:

$$
\begin{array}{l} \tilde {w} _ {t} (y | y _ {0}) = - \frac {T ^ {\prime} (1 - t)}{2} \left(\frac {y - e ^ {- \frac {1}{2} T (1 - t)} y _ {0}}{1 - e ^ {- T (1 - t)}} - y\right) \\ = - \frac {T ^ {\prime} (1 - t)}{2} \left[ \frac {e ^ {- T (1 - t)} y - e ^ {- \frac {1}{2} T (1 - t)} y _ {0}}{1 - e ^ {- T (1 - t)}} \right], \\ \end{array}
$$

which coincides with equation 19.

![](images/dc6eb3c92e8f7679707368d70bb967d967c51c429094ea11e1a92d501da689fd.jpg)  
Diffusion path – conditional vector field

![](images/62df1d6dd50cdc3126866bc79360f2469e36ef48c0453d53469ffdd79a850627.jpg)  
Figure 8: VP Diffusion path’s conditional vector field. Compare to Figure 2.   
Figure 9: Trajectories of CNFs trained with ScoreFlow (Song et al., 2021) and DDPM (Ho et al., 2020) losses on 2D checkerboard data, using the same learning rate and other hyperparameters as Figure 4.

# E IMPLEMENTATION DETAILS

For the 2D example we used an MLP with 5-layers of 512 neurons each, while for images we used the UNet architecture from Dhariwal & Nichol (2021). For images, we center crop images and resize to the appropriate dimension, whereas for the $3 2 \times 3 2$ and $6 4 \times 6 4$ resolutions we use the same pre-processing as (Chrabaszcz et al., 2017). The three methods (FM-OT, FM-Diffusion, and SM-Diffusion) are always trained on the same architecture, same hyper-parameters, and for the same number of epochs.

# E.1 DIFFUSION BASELINES

Losses. We consider three options as diffusion baselines that correspond to the most popular diffusion loss parametrizations (Song & Ermon, 2019; Song et al., 2021; Ho et al., 2020; Kingma et al., 2021). We will assume general Gaussian path form of equation 10, i.e.,

$$
p _ {t} (x \mid x _ {1}) = \mathcal {N} (x \mid \mu_ {t} (x _ {1}), \sigma_ {t} ^ {2} (x _ {1}) I).
$$

Score Matching loss is

$$
\begin{array}{l} \mathcal {L} _ {\mathrm {S M}} (\theta) = \mathbb {E} _ {t, q \left(x _ {1}\right), p _ {t} \left(x \mid x _ {1}\right)} \lambda (t) \| s _ {t} (x) - \nabla \log p _ {t} (x \mid x _ {1}) \| ^ {2} (42) \\ = \mathbb {E} _ {t, q \left(x _ {1}\right), p _ {t} \left(x \mid x _ {1}\right)} \lambda (t) \left\| s _ {t} (x) - \frac {x - \mu_ {t} \left(x _ {1}\right)}{\sigma_ {t} ^ {2} \left(x _ {1}\right)} \right\| ^ {2}. (43) \\ \end{array}
$$

Taking $\lambda ( t ) = \sigma _ { t } ^ { 2 } ( x _ { 1 } )$ corresponds to the original Score Matching (SM) loss from Song & Ermon (2019), while considering $\lambda ( \bar { t ) } = \beta ( 1 - t )$ ( $\beta$ is defined below) corresponds to the Score Flow (SF) loss motivated by an NLL upper bound (Song et al., 2021); $s _ { t }$ is the learnable score function. DDPM (Noise Matching) loss from Ho et al. (2020) (equation 14) is

$$
\begin{array}{l} \mathcal {L} _ {\mathrm {N M}} (\theta) = \mathbb {E} _ {t, q \left(x _ {1}\right), p _ {t} \left(x \mid x _ {1}\right)} \left\| \epsilon_ {t} (x) - \frac {x - \mu_ {t} \left(x _ {1}\right)}{\sigma_ {t} \left(x _ {1}\right)} \right\| ^ {2} (44) \\ = \mathbb {E} _ {t, q \left(x _ {1}\right), p _ {0} \left(x _ {0}\right)} \left\| \epsilon_ {t} \left(\sigma_ {t} \left(x _ {1}\right) x _ {0} + \mu_ {t} \left(x _ {1}\right)\right) - x _ {0} \right\| ^ {2} (45) \\ \end{array}
$$

where $p _ { 0 } ( x ) = \mathcal { N } ( x | 0 , I )$ is the standard Gaussian, and $\epsilon _ { t }$ is the learnable noise function.

Diffusion path. For the diffusion path we use the standard VP diffusion (equation 19), namely,

$$
\mu_ {t} (x _ {1}) = \alpha_ {1 - t} x _ {1}, \quad \sigma_ {t} (x _ {1}) = \sqrt {1 - \alpha_ {1 - t} ^ {2}}, \quad \text {w h e r e} \alpha_ {t} = e ^ {- \frac {1}{2} T (t)}, \quad T (t) = \int_ {0} ^ {t} \beta (s) d s,
$$

Table 3: Hyper-parameters used for training each model   

<table><tr><td></td><td>CIFAR10</td><td>ImageNet-32</td><td>ImageNet-64</td><td>ImageNet-128</td></tr><tr><td>Channels</td><td>256</td><td>256</td><td>192</td><td>256</td></tr><tr><td>Depth</td><td>2</td><td>3</td><td>3</td><td>3</td></tr><tr><td>Channels multiple</td><td>1,2,2,2</td><td>1,2,2,2</td><td>1,2,3,4</td><td>1,1,2,3,4</td></tr><tr><td>Heads</td><td>4</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Heads Channels</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Attention resolution</td><td>16</td><td>16,8</td><td>32,16,8</td><td>32,16,8</td></tr><tr><td>Dropout</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Effective Batch size</td><td>256</td><td>1024</td><td>2048</td><td>1536</td></tr><tr><td>GPUs</td><td>2</td><td>4</td><td>16</td><td>32</td></tr><tr><td>Epochs</td><td>1000</td><td>200</td><td>250</td><td>571</td></tr><tr><td>Iterations</td><td>391k</td><td>250k</td><td>157k</td><td>500k</td></tr><tr><td>Learning Rate</td><td>5e-4</td><td>1e-4</td><td>1e-4</td><td>1e-4</td></tr><tr><td>Learning Rate Scheduler</td><td>Polynomial Decay</td><td>Polynomial Decay</td><td>Constant</td><td>Polynomial Decay</td></tr><tr><td>Warmup Steps</td><td>45k</td><td>20k</td><td>-</td><td>20k</td></tr></table>

with, as suggested in Song et al. (2020b), $\beta ( s ) = \beta _ { \mathrm { m i n } } + s ( \beta _ { \mathrm { m a x } } - \beta _ { \mathrm { m i n } } )$ and consequently

$$
T (s) = \int_ {0} ^ {s} \beta (r) d r = s \beta_ {\mathrm {m i n}} + \frac {1}{2} s ^ {2} (\beta_ {\mathrm {m a x}} - \beta_ {\mathrm {m i n}}),
$$

where $\beta _ { \mathrm { m i n } } = 0 . 1$ , $\beta _ { \mathrm { m a x } } = 2 0$ and time is sampled in $[ 0 , 1 - \epsilon ]$ , $\epsilon = 1 0 ^ { - 5 }$ for training and likelihood and $\epsilon = 1 0 ^ { - 5 }$ for sampling.

Sampling. Score matching samples are produced by solving the ODE (equation 1) with the vector field

$$
u _ {t} (x) = - \frac {T ^ {\prime} (1 - t)}{2} [ s _ {t} (x) - x ]. \tag {46}
$$

DDPM samples are computed with equation 46 after setting $s _ { t } ( x ) \ = \ \epsilon _ { t } ( x ) / \sigma _ { t }$ , where $\sigma _ { t } =$ $\sqrt { 1 - \alpha _ { 1 - t } ^ { 2 } }$ .

# E.2 TRAINING & EVALUATION DETAILS

We report the hyper-parameters used in Table 3. We use full 32 bit-precision for training CIFAR10 and ImageNet-32 and 16-bit mixed precision for training ImageNet-64/128/256. All models are trained using the Adam optimizer with the following parameters: $\beta _ { 1 } = 0 . 9$ , $\beta _ { 2 } = 0 . 9 9 9$ , weight decay $= 0 . 0$ , and $\epsilon = 1 e { - 8 }$ . All methods we trained (i.e., FM-OT, FM-Diffusion, SM-Diffusion) using identical architectures, with the same parameters for the the same number of Epochs (see Table 3 for details). We use either a constant learning rate schedule or a polynomial decay schedule (see Table 3). The polynomial decay learning rate schedule includes a warm-up phase for a specified number of training steps. In the warm-up phase, the learning rate is linearly increased from $1 e { - } 8$ to the peak learning rate (specified in Table 3). Once the peak learning rate is achieved, it linearly decays the learning rate down to $1 e { - } 8$ until the final training step.

When reporting negative log-likelihood, we dequantize using the standard uniform dequantization. We report an importance-weighted estimate using

$$
\log \frac {1}{K} \sum_ {k = 1} ^ {K} p _ {t} \left(x + u _ {k}\right), \text {w h e r e} u _ {k} \sim \mathcal {U} (0, 1), \tag {47}
$$

with $x$ is in $\{ 0 , . . . , 2 5 5 \}$ and solved at $t \ = \ 1$ with an adaptive step size solver dopri5 with $\mathtt { i t o l } { = } \mathtt { r t o l } { = } 1 { \mathtt { e } } { - } 5$ $=$ using the torchdiffeq (Chen, 2018) library. Estimated values for different values of $K$ are in Table 4.

Table 4: Negative log-likelihood (in bits per dimension) on the test set with different values of $K$ using uniform dequantization.   

<table><tr><td rowspan="2">Model</td><td colspan="3">CIFAR-10</td><td colspan="3">ImageNet 32×32</td><td colspan="3">ImageNet 64×64</td></tr><tr><td>K=1</td><td>K=20</td><td>K=50</td><td>K=1</td><td>K=5</td><td>K=15</td><td>K=1</td><td>K=5</td><td>K=10</td></tr><tr><td colspan="10">Ablation</td></tr><tr><td>DDPM</td><td>3.24</td><td>3.14</td><td>3.12</td><td>3.62</td><td>3.57</td><td>3.54</td><td>3.36</td><td>3.33</td><td>3.32</td></tr><tr><td>Score Matching</td><td>3.28</td><td>3.18</td><td>3.16</td><td>3.65</td><td>3.59</td><td>3.57</td><td>3.43</td><td>3.41</td><td>3.40</td></tr><tr><td>ScoreFlow</td><td>3.21</td><td>3.11</td><td>3.09</td><td>3.63</td><td>3.57</td><td>3.55</td><td>3.39</td><td>3.37</td><td>3.36</td></tr><tr><td colspan="10">Ours</td></tr><tr><td>FMw/ Diffusion</td><td>3.23</td><td>3.13</td><td>3.10</td><td>3.64</td><td>3.58</td><td>3.56</td><td>3.37</td><td>3.34</td><td>3.33</td></tr><tr><td>FMw/ OT</td><td>3.11</td><td>3.01</td><td>2.99</td><td>3.62</td><td>3.56</td><td>3.53</td><td>3.35</td><td>3.33</td><td>3.31</td></tr></table>

![](images/dca1b8a2ef4b8c2faf59571ba212cda057adfe105b18147ae46cf35c700498bf.jpg)  
Figure 10: Function evaluations for sampling during training, for models trained on CIFAR-10 using dopri5 solver with tolerance 1e−5. $1 e ^ { - 5 }$

When computing FID/Inception scores for CIFAR10, ImageNet-32/64 we use the TensorFlow GAN library 2. To remain comparable to Dhariwal & Nichol (2021) for ImageNet-128 we use the evaluation script they include in their publicly available code repository 3.

# F ADDITIONAL TABLES AND FIGURES

![](images/1581de7c58573042b0fb39bc5298c8bc44f50718b28aea4fa56b3517ceb9b706.jpg)  
Figure 11: Non-curated unconditional ImageNet-32 generated images of a CNF trained with FM-OT.

![](images/a2047d26ac7d998517204258c2001f59081eef119c33fa37b459a449a7465720.jpg)  
Figure 12: Non-curated unconditional ImageNet-64 generated images of a CNF trained with FM-OT.

![](images/6c87250478f7923829b5038495b4c5b60ef94cc4f1fde7b7d84d05058208f835.jpg)  
Figure 13: Non-curated unconditional ImageNet-128 generated images of a CNF trained with FM-OT.

![](images/f76a11618c91caab7c56a6e2c229d2cf6d1138ec88921a576c912f9341cceefc.jpg)

![](images/d6f079a9a610954184f4f6794b017d9c395911ee9e650b195ef99f5506f4f0f1.jpg)

![](images/4a1ce65e14c781b85034f9940aa54b83ecc4e3c64b533a0d76a9f31aa8b4b849.jpg)

![](images/a9c8995d1f34eb82311612b26317aff495846b0478a7d6c284fa15a874884867.jpg)

![](images/d388d0da8d184c8902fa2db67ba52eec28e908a1eb17d4bc3209a7455b87b810.jpg)

![](images/6fd67d5212f681309d5b2f8ee62fa652ea6035cd232d48d6ed2e7c73718bc3bd.jpg)

![](images/b0b65babe7adf0dc4d78c83c0cc8cc37a7d996ae306ef3fd9c6834b71d48aba6.jpg)

![](images/5d86b50aed1262473f53c8d4f74d14ec552c419e966ae1f53e64630073c44d39.jpg)

![](images/cdf822a1467401d886998c442a1f4f6c14fc6880f52d11e12ca160f3d4bab5fa.jpg)

![](images/da0da4a570bb1b4bb4f68e8848b3b444c386bed32170d4300fe22896f7b94626.jpg)

![](images/48a1c711ccc311ff821d72fb0c0316f2f96facf382ba7fab5ec9dcdf61267c5f.jpg)

![](images/1bf1ea44de1f15715f4e66d8124025e62371aa9fe9cc205993b03831586bbc42.jpg)  
Figure 14: Conditional generation $6 4 \times 6 4 \to 2 5 6 \times 2 5 6$ . Flow Matching OT upsampled images from validation set.

![](images/823b26ddeebe718372316fcb48a0c99a0ecd4f38eb839c85a6a89be57ddcaa3c.jpg)

![](images/66b0ddd45c7e7154e70a77e72157cb508a0b6c2c1579cffc8f58cae7bbb9e4b1.jpg)

![](images/5d992736b9c775509d629e3916942d992a1e3677524febf335552b9b8eae58a3.jpg)

![](images/69305e9467b6840e11a1e6499795d9a14f0e7afc5a54bfb9ce07ecba70ddcdc5.jpg)

![](images/554b2f813a04e7d7ef60d4f42251692ead1d0a53ad3d48a912752624283b493b.jpg)

![](images/be31df334c10df087013df05a089e982e6e3218e2054658267069a6490ccff7e.jpg)

![](images/39e3e0b17a2e3ee32ef585ba219998749384fe41d670a74d4c5968d08ab67127.jpg)

![](images/ffc86e23358106b59fe11f0ac3f352753fc63b2e2201c716c80de0d254cca741.jpg)

![](images/81ad5fe91eb7052bede876fc81fa36e924659d6b09fb4ed8126e6508ac14321f.jpg)

![](images/8f3d056cad175f940ae3ed79966c21b702935cbb8655afc85ad459fcc2cf8bce.jpg)

![](images/9c34c3c7789717e448a0ac3e02d14b0bf7d3460928c8ff0f32d96609c3efe910.jpg)

![](images/a6d81c23836eadc761424f59aaf2d2c0d148ae9f4b8212d638c56ab33711f0b1.jpg)  
Figure 15: Conditional generation $6 4 \times 6 4 \to 2 5 6 \times 2 5 6$ . Flow Matching OT upsampled images from validation set.

![](images/21c11414ef4b5c374f182b844399bf1c869468a495c72d5a7b2ab20012236a7f.jpg)

![](images/6789c6562cb9066c5de5d9d530fca180005f3851d20f94dfd881eddaba9a0bbc.jpg)  
Figure 16: Generated samples from the same initial noise, but with varying number of function evaluations (NFE). Flow matching with OT path trained on ImageNet-128.

![](images/c208d5c6148d692b5f1b205e75b03b6d0c3dff9c08986c353539d058f5e006c8.jpg)  
Figure 17: Generated samples from the same initial noise, but with varying number of function evaluations (NFE). Flow matching with OT path trained on ImageNet $2 5 6 \times 2 5 6$ .