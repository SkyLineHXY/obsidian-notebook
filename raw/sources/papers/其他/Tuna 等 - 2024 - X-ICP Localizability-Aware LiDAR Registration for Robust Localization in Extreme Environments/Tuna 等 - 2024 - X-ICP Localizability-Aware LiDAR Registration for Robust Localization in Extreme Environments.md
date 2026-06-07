# X-ICP: Localizability-Aware LiDAR Registration for Robust Localization in Extreme Environments

Turcan Tuna†,‡, Julian Nubert†, Yoshua Nava‡, Shehryar Khattak†, Marco Hutter†

Abstract—Modern robotic systems are required to operate in challenging environments, which demand reliable localization under challenging conditions. LiDAR-based localization methods, such as the Iterative Closest Point (ICP) algorithm, can suffer in geometrically uninformative environments that are known to deteriorate point cloud registration performance and push optimization toward divergence along weakly constrained directions. To overcome this issue, this work proposes i) a robust fine-grained localizability detection module, and ii) a localizability-aware constrained ICP optimization module, which couples with the localizability detection module in a unified manner. The proposed localizability detection is achieved by utilizing the correspondences between the scan and the map to analyze the alignment strength against the principal directions of the optimization as part of its fine-grained LiDAR localizability analysis. In the second part, this localizability analysis is then integrated into the scan-to-map point cloud registration to generate drift-free pose updates by enforcing controlled updates or leaving the degenerate directions of the optimization unchanged. The proposed method is thoroughly evaluated and compared to state-of-the-art methods in simulated and real-world experiments1, demonstrating the performance and reliability improvement in LiDAR-challenging environments. In all experiments, the proposed framework demonstrates accurate and generalizable localizability detection and robust pose estimation without environment-specific parameter tuning.

Index Terms—robust localization, LiDAR localizability, constrained ICP, optimization degeneracy, environment degeneracy

# I. INTRODUCTION

R ELIABLE robot pose estimation and map creation arecore capabilities that enable mobile robots to operate core capabilities that enable mobile robots to operate autonomously. While it is possible to achieve robot localization with global positioning or fixed physical references (e.g., markers), these methods are expensive or difficult to scale to new environments. With the advent of increasingly capable exteroceptive sensors, the research community has focused on solving the simultaneous localization and mapping (SLAM) problem with a variety of algorithms [1], [2]. SLAM can be achieved with different types of sensors [3]; however, this work focuses on LiDAR-based sensing due to its ability to provide reliable and accurate range measurements in the form of point clouds. The success of LiDAR-based approaches in

This work is supported in part by the EU Horizon 2020 programme grant agreement No. 852044, 101016970, and 101070405, EU Horizon 2021 programme grant agreement No. 101070596, the NCCR digital fabrication and robotics, the ETH Zurich Research Grant No. 21-1 ETH-27, the SNF project No. 188596, and the Max Planck ETH Center for Learning Systems.

†The authors are with the Robotics Systems Lab, ETH Zürich.   
‡The authors are with the ANYbotics A.G.   
1https://youtu.be/SviLl7q69aA

![](images/c539481d1c41fc34fdc4e8ddea7b2768f73601083f8a3663924886558d719ad8.jpg)

<details>
<summary>text_image</summary>

A
B
C
Start / Finish
15m
A
B
C
X-ICP
Hinduja et al.
Zhang et al.
5m
0m
</details>

Fig. 1. Top Row: Ground truth map and path of the robot during the Seemühle experiment. Certain sections of the environment are illustrated through real images. Bottom Row: Point cloud maps created using the proposed approach and compared against two state-of-the-art methods. The color bar indicates the point-to-point distance error with respect to the ground truth map.

competitions such as the KITTI [4] and the recent HILTI [5] benchmarks support the strength of LiDAR-based approaches.

a) Point Cloud Registration: Most popular LiDAR-based SLAM algorithms perform point cloud registration using iterative error minimization techniques to estimate the correct pose difference between two point clouds, often referred to as scan-to-scan registration. Similarly, point cloud registration can be done against a map, referred to as scan-to-map registration, significantly reducing drift compared to scan-to-scan. In a typical LiDAR-based SLAM framework, scan-to-map registration is the core step where the pose of the incoming point cloud data in the map is calculated. To date, the most well-known point cloud registration method is the iterative closest point (ICP) algorithm [6], [7], still utilized by recent research [8], [9].

b) Current Limitations: Although the ICP algorithm and its variants are commonly used, for practical applications, their performance is limited by four sources of error [10], [11]. These errors include a) the risk of converging to a local minima, b) the sensitivity to sensor measurement noise and bias, c) utilization of an inaccurate transformation prior, and d) the lack of geometric constraints provided by the environment for the underlying optimization problem. While the robotic community has developed solutions and frameworks that attenuate different error sources, the inaccurate transformation prior and the lack of geometric constraints can still cause modern LiDAR-based SLAM systems to fail when deployed in challenging environments. Although both problems are crucial, this work focuses on minimizing the adverse effect of the lack of geometrical constraints provided by the operational environment. In self-symmetrical environments, environments with perceptual aliasing, the geometric constraints along the axis of symmetry can be indistinguishable from noise, and the total number of unique constraints might be insufficient for optimization to converge. As a result, the optimization might converge to a noise-induced optimum [12], referred to as a ill-conditioned or degenerate solution. Tunnels, wide open spaces, tight corridors, and narrow doorways are instances of such degraded environments [13]. An example of a real-world degenerate underground tunnel is shown in the top row of Fig. 1. As discussed in recent research [14]–[16], to enable robots to operate in these challenging scenarios, and to handle the solution ill-conditioning caused by degenerate planar or tunnel-like environments, localizability-awareness is required.

c) Proposed Approach: As a response to the LiDAR degeneracy problem, this work proposes a robust localizabilityaware point cloud registration (ICP) framework that enables LiDAR-based SLAM systems to operate in featureless eXtreme environments, called X-ICP. The proposed framework, shown in Fig. 2, solves both the detection of LiDAR degeneracy and the mitigation of the adverse effects of this degeneracy on the optimization. The two sub-modules of the proposed approach are localizability detection module, abbreviated as Loc.-Module, and optimization module, abbreviated as Opt.-Module. The Loc.-Module utilizes the point and surface-normal correspondences between the scan and the map to analyze the contribution strength along the optimization’s principal directions. In contrast to prior work [14], the localizability detection is applied to the scan-to-map registration, and the localizability detection is done in the optimization eigenspace, which allows the detection to be independent of the sensor orientation in the map. The resulting proposed solution enables reliable localizability detection in various environmental configurations, such as underground environments and large open spaces, without parameter tuning. Furthermore, the localizability detection is fine-grained and can classify the current optimization directions into three {localizable, partially-localizable, and non-localizable} categories.

Given the localizability information, the Opt.-Module calculates optimization constraints by salvaging the registration correspondences and employs these constraints in the low-level optimization for the point cloud registration. More specifically, this module leaves the point cloud registration initial guess unchanged in the non-localizable directions and enforces controlled pose updates for partially-localizable directions while having no effect on localizable directions. Notably, the Opt.- Module is independent of the ICP cost function and can be used independently in combination with other optimizationbased systems since the adverse effects of the degenerate directions are mitigated by employing the localizability information within the iterative optimization problem. The result of this optimization utilizes the information contained in the correspondences while using the robot odometry information along the ill-conditioned directions.

The proposed framework is tested extensively with multiple simulated environments and real-world missions in various environments and with different sensor setups. The field experiments and analyses suggest that the proposed framework can reliably detect localizability in diverse, challenging environments without requiring heuristic parameter tuning and achieves increased robustness and accuracy in degraded environments. The proposed framework consistently outperforms the state-of-the-art robotic approaches [12], [17] throughout all experiments performed in challenging and partly degenerate scenarios, as shown in Fig. 1.

d) Contributions: The main contributions of this work are as follows:

• A heuristic-free and multi-category localizability detection algorithm is developed to reliably identify the localizability state of the principal directions of the underlying point cloud registration optimization problem.   
• A method to salvage scarce information from point correspondences is developed to utilize the available information in partially-localizable optimization directions.   
• A novel localizability-aware constrained ICP optimization module is developed for robust point cloud registration in degenerate scenarios.   
• A variety of experiments are conducted to evaluate the efficacy of the proposed framework and to compare the results against the state-of-the-art methods.

Additional content, data, and supplementary material are provided on the project page2.

# II. RELATED WORK

LiDAR-based SLAM systems rely on point cloud registration methods to estimate the robot pose. A brief review of these methods is presented in Section II-A. Furthermore, related work on degeneracy detection is discussed in Section II-B, followed by a discussion of the approaches that constrain the underlying ill-conditioned optimization problem for point cloud registration in Section II-C.

# A. Point Cloud Registration Methods

Point cloud registration is considered a mature research field, and multiple unique approaches [18]–[21] are proposed to achieve robust, fast, and accurate pose estimation. Among these, the most widely used registration algorithm for LiDARbased registration is the ICP algorithm [22]. The ICP algorithm iteratively finds the transformation between two point clouds given an initial transformation. This is achieved by minimizing a pre-defined cost function that measures the error between point pairs in source and target point clouds. Various cost functions have been proposed, such as the point-to-point [6], point-to-plane [7], point-to-line [23], point-to-Gaussian [24], and symmetric point-to-plane [25]. Beyond that, works also combine different cost functions, statistical measures [19], [26], and even employ data-driven methods [13] to achieve more robust registration. Despite these promising alternatives, the point-to-plane cost function [7] is still among the preferred solutions for state-of-the-art robust LiDAR-based SLAM systems [27]–[29] due to its simplicity and effectiveness, as demonstrated in real-world deployments [3], [5].

In recent years, certifiable algorithms for robust point cloud registration have gained interest [30]–[32]. These algorithms target to identify whether the optimization solution is globally optimal. Similarly, to perform reliable point cloud registration in the presence of outliers and strong noise, approaches such as [33], [34] proposed using robust norms. Nevertheless, conceptually, the problem definition of these works differs from the one of this work. In X-ICP, the goal is to identify possible degenerate cases before the registration, while certifiable algorithms analyze the solution optimality after the optimization and provide a decisive metric on optimality. Considering this fact alone, certifiable algorithms do not inherently provide a solution to perceptual aliasing. On the other hand, robust norms filter noisy measurements to reveal the underlying optimization minima. However, in the absence of information or the presence of perceptual aliasing, robust norms do not help to overcome the underlying problem.

# B. Degeneracy Detection

Point cloud registration techniques have demonstrated reliable performance in various practical applications; however, the underlying iterative minimization problem can become degenerate in challenging geometrically self-symmetric or featureless environments. This problem is often noticed in the form of LiDAR-slip when traversing along the self-similar directions of the environment, e.g., narrow and long building corridors, severely degrading the robot pose estimation performance. Many techniques have been proposed to detect degenerate conditions by modeling it as part of the uncertainty or covariance of the pose estimation process [10], [11], [35], [36]. However, a unified uncertainty representation is often not tractable regarding sources of individual errors and tends to be over-optimistic [37]. Motivated by this reasoning, direct degeneracy detection methods have been proposed.

1) Geometric Methods: Geometric approaches utilize the relationship between the registration cost function and the environment to analyze the quality of the pose estimation process. Among the first to investigate the geometric stability of the point-to-plane ICP, [38] proposed a sampling-based method to select the most valuable points in a scan to improve the conditioning of the optimization process. Building on this idea [39] proposed improvements such as the iterative center of mass calculation, rotation normalization, and cyclic point addition to improve the efficiency of the method. In a similar direction, IMLS-SLAM [40] includes the contribution of a point to the matching procedure to ensure the observability of the optimization; however, the additional overhead makes the method unsuitable for real-time applications. Similarly, to estimate the localizability of an environment using the eigenspace of the Hessian, authors in [16] propose to measure the constraint strength of a point and surface-normal pair as the sensitivity of measurements w.r.t to the optimization states. Although this formulation is theoretically grounded and wellstructured, the given parameterization does not use the full parametric range of the localizability analysis to distinguish the localizable and non-localizable directions due to alignment based normalization. Furthermore, it does not account for the scale difference between rotation and translation sub-spaces, which can be significant in practical applications. Crucially, this method requires a prior-built point cloud map of the environment. The proposed localizability-awareness framework, X-ICP, also uses a geometry-based approach, similar to [16], but does not require a prior map of the environment, the scale difference between rotation and translation is taken into account, and the localizability detection is more fine-grained (three levels vs. binary).

2) Optimization-based Methods: For degeneracy detection, several approaches have proposed different metrics to quantify the state of the optimization. The work in [41] proposes to use the observability Gramian as a measure of insufficient sensor measurements required to constrain the optimization. Similarly, [29], [42], [43] proposed to use the condition number of the optimization Hessian as a single combined degeneracy metric for all 6-DoF of the pose estimation optimization problem. With similar reasoning, [44], [45] proposed to utilize the determinant of the fisher information matrix as the degeneracy detection metric instead. Reasoning that a combined degeneracy metric is not representative for both translation and rotational sub-spaces, [46] proposes to utilize relative condition number only for the translational sub-space to detect optimization degeneracy along each translational direction. In contrast, CompSLAM [28] uses the D-optimality criterion [47] as a degeneracy detection metric to detect underconstrained environments for different modalities of a robust sensor fusion. Although practical in nature, these approaches subject different degeneracy metrics to a threshold to identify degeneracy, which is not only heuristic in nature but difficult to generalize as these metrics depend on the environment’s structure and the amount of information observed during an operation instance. Furthermore, the smooth transition of the optimization problem from degenerate to non-degenerate is difficult to capture with a binary degeneracy detection method.

For degeneracy-aware LiDAR-based SLAM systems, the seminal work of [12] proposes both a degeneracy detection metric called degeneracy factor and a degeneracy mitigation method named solution remapping. The degeneracy factor utilizes the minimum eigenvalue of the Hessian matrix of the optimization to detect the degeneracy, and the solution remapping method utilizes the detected degeneracy to project the optimization solution only along the well-constrained directions. This work has been adopted by multiple LiDAR-based SLAM frameworks [28], [48]–[51] and is considered state-ofthe-art; however, certain aspects can limit its efficacy. i) Being binary in nature, the method depends on the heuristic tuning of thresholds for operation in different environments [14]. ii) As eigenvalues represent the scale of their respective eigenvectors, thresholds for translation and rotation cannot be represented by a singular value. iii) As solution remapping projects the solution along the well-constraint directions, it assumes that wellconditioned directions remain completely unaffected by the ill-conditioning. However, this assumption may be incorrect in severely degenerate environments since the optimization might diverge before the solution remapping method can project it. Echoing similar reasoning, [17] proposes to improve solution remapping by using the relative condition number of the optimization directions to set the eigenvalue threshold automatically. These state-of-the-art methods [12], [17] are widely adopted in the field of robotics and are used as baseline methods for comparison in this work.

3) Data-driven Methods: With the advent of learning-based methods, data-driven methods provide promising alternatives to perform degeneracy detection. The work in [52] formulates localizability as a function of overlap between scans and uses a support vector classifier to learn a risk metric for point cloud registration. Similarly, OverlapNet [53] showed the importance of using overlap between point cloud scans to identify the similarity of environments. In contrast, [35] proposes direct localizability quantification using a learned pose estimation uncertainty metric. Furthermore, combining covariance estimation and localizability detection, [54] proposes a deep-learned entropy-based metric. Although successful, these methods rely on extensive ground truth data for learning and are unsuitable for real-time operation. To alleviate reliance on data, [14] proposes to leverage simulation for training and only consider the current LiDAR scan to predict a 6-DoF localizability metric. The authors use sparse 3D convolutions to show generalization across different sensors and environments through real-world experiments. However, this approach is limited to scan-to-scan point cloud registration.

# C. Constrained Optimization in Point Cloud Mapping

Constrained optimization techniques are well-known in literature; however, their application to point cloud registration has only recently attracted more attention. Among the first, [55] presented a constrained optimization method for penetration-free point cloud registration, improving the quality of pose estimation. Similarly, [56] uses non-linear equality constraints to reduce the linearization error of rotation estimation for point cloud registration. To improve robustness against sensor noise and correspondence outliers, [57] proposes to use an augmented Lagrangian to solve a constrained optimization problem by adding each measurement as a separate constraint. A recent work [58] introduces the addition of soft constraints as costs to the ICP optimization to ensure trajectory continuity between different scans. In contrast, [30] formulates the point cloud registration problem as a constrained quadratic program to provide globally optimal point cloud registration results. These methods demonstrate improved robustness and accuracy for the global point cloud registration problem; however, they do not address the utility of constraints towards limiting the effect of degeneracy in the optimization.

Addressing this limitation, using the degeneracy detection formulation from [12], Hinduja et al. [17] proposed a partial factor formulation to incorporate the point cloud registration result into a pose graph formulation in a degeneracy-aware manner. The authors used a re-projection matrix as covariance to reflect the well-constraint directions of the registration result. While this method shows a straightforward way of integrating degeneracy awareness into sensor fusion frameworks, the adverse effects of degeneracy on low-level ICP optimization are not addressed.

On the other hand, the authors in [46] propose to use the relative condition number to detect degeneracy and to penalize the motion change along the degenerate directions by introducing constraints in a factor graph formulation in the form of a smooth cost function. However, this work is only tested in 2D scenarios and does not consider introducing constraints directly in the optimization process. Given the discussion, the use of degeneracy analysis to constrain the ICP registration problem for robot operation in challenging and degenerate environments remains an open problem that this work aims to address.

# III. PROBLEM FORMULATION & PRELIMINARIES

This section provides an overview of the point cloud registration process and formulates the problem for operation in LiDAR degenerate environments. All vectors and matrices are denoted in bold, with matrices expressed with Capital letters.

# A. Point Cloud Registration

The problem of point cloud registration is defined as finding the rigid body transformation $\mathbf { \cal T } _ { \mathtt { M L } } \ \in \ S E ( 3 )$ , that best aligns a reading point cloud of $N _ { p }$ points $\mathring { \mathbf { \mathbf { \phi } } } _ { \mathbf { \lambda } } \doteq \mathbf { \phi } \mathbb { R } ^ { 3 \times N _ { p } }$ in LiDAR frame (denoted as L) to a reference point cloud of $N _ { q }$ points, $\mathbf { \Psi } _ { \mathrm { M } } \pmb { Q } \in \mathbb { R } ^ { 3 \times N _ { q } }$ in map frame (denoted as M). The rigid transformation $T _ { \mathrm { M L } } ~ = ~ \left\lceil R _ { \mathrm { M L } } \ | \ _ { \mathrm { M } } t _ { \mathrm { M L } } \right\rceil$ , consists of a rotation matrix $R \in \ S O ( 3 )$ , and a translation vector $\textbf { \textit { t } } \in { \mathbb { R } } ^ { 3 }$ , with $\mathbf { \Psi } _ { t } ~ = ~ [ t _ { x } , ~ t _ { y } , ~ t _ { z } ] ^ { \intercal }$ . For each reading point $\mathbf { \Delta } _ { \mathrm { L } } \pmb { p } _ { \mathbf { \Delta } } \in \mathrm { ~ \mathbb { R } ^ { 3 } ~ }$ in $_ \mathrm { L } P$ , the closest reference point $\mathsf { \pmb { M } } \pmb { q } \in \mathbb { R } ^ { 3 }$ in $\mathsf { \Pi } _ { \mathtt { M } } \pmb { Q }$ , is found through a correspondence search, often in the form of a k-d tree search. This data association process is defined as $\mathcal { M } \in \mathbb { R } ^ { 6 \times N } =$ matching  ${ \bf \mathrm { \Omega } } _ { \mathrm { L } } P , ~ { \bf { \mathrm { n } } } Q , ~ T _ { \mathrm { L M , i n i t } } ) ~ =$ $\left\{ { \bf \Phi } ( \mathtt  \} _ { \mathtt { M } } p , \ \mathtt { \backslash } _ { \mathtt { M } } q , \ \mathtt { \backslash } _ { \mathtt { M } } n \} ) \ : ( \mathtt { \mathtt { m } } p \in \mathtt { \rVert } _ { \mathtt { M } } P ) , \ \mathtt { \backslash } _ { ( \mathtt { M } } q \in \mathtt { \rVert } _ { \mathtt { M } } Q ) \right\}$ where Mp and $\mathtt { M } \mathbf { q }$ are the matched point pairs and $\mathsf { \Delta } _ { \mathtt { M } } \pmb { n } \in \mathbb { R } ^ { 3 } , \ \| \mathsf { \mathbf { M } } \pmb { n } \| = 1$ is the surface-normal vector of point $\mathbb { M } ^ { q . }$ Furthermore, $N \leq N _ { p }$ is the number of matched points and indicates the size of the problem for the rest of the work. The initial transformation $\pmb { T } _ { \mathrm { L M , i n i t } }$ is provided as an initial guess to transform the scan data to the reference frame, to improve the matching process and optimization convergence characteristics. While the accuracy of this initial transformation is critical for the convergence of the minimization [11], an analysis of the effect and quality of this initial transformation is outside the scope of this work.

![](images/7acad84d07ea54028bb86799acf3f6c461b219a01ab5a300d28da3b65a241eab.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Sensors"] --> B["LiDAR"]
    C["Prior"] --> D["Process Block"]
    B --> E["Scan"]
    D --> F["Transformed Scan"]
    E --> G["Map"]
    F --> H["ICP Loop"]
    G --> I["Correspondences n_i p_i × n_i"]
    H --> J["Matching"]
    I --> K["Section VI Localizability Aware Opt."]
    J --> L["Pos Update"]
    K --> M["Optimized Pose M x*"]
    L --> M
    M --> N["Localizability-Aware Point Cloud Registration"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style E fill:#cfc,stroke:#333
    style G fill:#fcc,stroke:#333
    style H fill:#fcc,stroke:#333
    style I fill:#fcc,stroke:#333
    style J fill:#fcc,stroke:#333
    style K fill:#fcc,stroke:#333
    style L fill:#fcc,stroke:#333
    style M fill:#fcc,stroke:#333
```
</details>

Fig. 2. Overview of the proposed localizability-aware point cloud registration framework. The pose prior is used to transform and undistort the input point cloud, which is, together with the existing point cloud map, fed to the iterative ICP optimization loop. The optimized drift-free pose within the ICP loop is calculated using the proposed localizability detection (Section V) and aware optimization (Section VI) modules.

Multiple error functions have been proposed for point cloud alignment; in this work, the point-to-plane [7] cost function is used. The ICP minimization problem with the point-to-plane cost function can be defined as follows:

$$
\min _ {\boldsymbol {R}, \boldsymbol {t}} \sum_ {i = 1} ^ {N} \left| \left| \left(\left(\boldsymbol {R} \boldsymbol {p} _ {i} + \boldsymbol {t}\right) - \boldsymbol {q} _ {i}\right) \cdot \boldsymbol {n} _ {i} \right| \right| _ {2}. \tag {1}
$$

Different solvers, such as singular value decomposition (SVD) [59], LU decomposition, Gauss-Newton, and Levenberg-Marquardt, can be used to solve this minimization problem. In this work, the focus lies on direct linear algebra solvers like SVD, which exists for all matrices.

Following the derivation of Pomerleau et al. [60], the minimization (1) can be reformulated as a quadratic cost optimization problem as:

$$
\begin{array}{l} \min _ {\boldsymbol {x} \in \mathbb {R} ^ {6}} \boldsymbol {x} ^ {T} \underbrace {\left(\sum_ {i = 1} ^ {N} \underbrace {\left[ \begin{array}{c} \left(\boldsymbol {p} _ {i} \times \boldsymbol {n} _ {i}\right) \\ \boldsymbol {n} _ {i} \end{array} \right]} _ {\boldsymbol {A}} \underbrace {\left[ \begin{array}{c} \left(\boldsymbol {p} _ {i} \times \boldsymbol {n} _ {i}\right) ^ {T} \boldsymbol {n} _ {i} ^ {T} \end{array} \right]} _ {\boldsymbol {A} ^ {\top}}\right)} _ {\boldsymbol {A} ^ {\prime}} \boldsymbol {x} \tag {2} \\ - 2 \boldsymbol {x} ^ {T} \underbrace {\left(\sum_ {i = 1} ^ {N} \underbrace {\left[ \begin{array}{c} (\boldsymbol {p} _ {i} \times \boldsymbol {n} _ {i}) \\ \boldsymbol {n} _ {i} \end{array} \right]} _ {A} \boldsymbol {n} _ {i} ^ {T} (\boldsymbol {q} _ {i} - \boldsymbol {p} _ {i})\right)} _ {\boldsymbol {b} ^ {\prime}} + \text {Const.} \\ \end{array}
$$

Here $\begin{array} { r l r } { { \textbf {  { x } } } } & { = } & { [ { \pmb { r } } ^ { \top } \quad { \pmb { t } } ^ { \top } ] ^ { \top } \qquad \in \quad \mathbb { R } ^ { 6 } } \end{array}$ are the optimization variables, with $r ~ \in ~ { \mathfrak { s o } } ( 3 )$ being the rotation vector (lie algebra of $S O ( 3 ) )$ and $\pm \in \mathbb { R } ^ { 3 }$ . Moreover, $\pmb { A } ^ { \prime } \in \mathbb { R } ^ { 6 \times 6 }$ denotes the Hessian of the optimization problem, and $\pmb { b } ^ { \prime } \in \pmb { \zeta }$ $\mathbb { R } ^ { 6 }$ incorporates the constraints between the point clouds. The Hessian constitutes the second-moment matrix of the optimization and defines the local behavior of the Jacobian. Moreover, the Equation (2) with optimization variable x can be reformulated as a least squares optimization problem as follows:

$$
\min _ {\boldsymbol {x} \in \mathbb {R} ^ {6}} \left| \left| \boldsymbol {A} ^ {\prime} \boldsymbol {x} - \boldsymbol {b} ^ {\prime} \right| \right| _ {2}. \tag {3}
$$

Solving this minimization problem is simple for a (semi-) positive definite matrix $A ^ { \prime } .$ . The solution of this $6 \times 6$ linear equation system will result in the optimal alignment translation vector t and rotation r under the performed linearizations. During ICP, due to non-linearities and the resulting iterative nature of the algorithm, the described operations are repeated until convergence.

# B. Operation in Degenerate Environments

In practical applications, the described point cloud registration process can fail due to LiDAR degeneracy induced by the absence of geometrically informative structures. The solution $\mathbf { T } _ { \mathrm { M L } }$ of the registration step becomes under-constrained, which means that one or multiple dimensions of the 6-DoF transformation are (almost) not observable from the point correspondences. Consequently, the primary focus of this work is to address the problem of finding the optimal transformation $\mathbf { T } _ { \mathrm { M L } }$ and determining the difficult-to-estimate directions in the presence of environmental degeneracy. While previous research on point cloud registration tends to disregard this particular scenario, this work introduces a dedicated solution to tackle these challenging situations, allowing for effective operation even in extreme scenarios.

# IV. SYSTEM OVERVIEW

To reliably perform point cloud registration and pose estimation in LiDAR degenerate environments, this work proposes X-ICP, a localizability-guided constrained point cloud registration method. An overview of the proposed framework is shown in Fig. 2. The proposed components for detecting and mitigating LiDAR degeneracy are denoted as Loc.- Module and Opt.-Module. For demonstration purposes, both components are embedded into a scan-to-map ICP registration system [61] developed by ANYbotics, which is based on a modified version of libpointmatcher, an open-sourced point cloud registration library [62]. The mapping pipeline based on this point cloud registration framework runs at 5 Hz and uses the common point-to-plane ICP cost function. Note that the proposed contributions of this work are applicable to any point-to-plane iterative optimization-based registration framework and are independent of the scan-to-map implementation. As discussed in Section III-A, the robustness and accuracy of the scan-to-map registration depend on the quality of the initial guess. Hence, X-ICP performs better with a robust state estimator compared to a dead-reckoning system.

![](images/c91a674a934a40534512e527649d79302f349c9d2924552cc6e108d79721ab41.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Information Analysis"] --> B["Filtering"]
    B --> C["Categorization"]
    C --> D["Degenerate"]
    D --> E{Degenerate}
    E -->|No| F["Lc(j) ≥ κ2 or Ls(j) ≥ κ3"]
    E -->|Yes| G["Ωvj = none"]
    E --> H["Ωvj = partial"]
    A --> I["A (Sec. V-A) Information Analysis"]
    I --> J["Count vs Contribution Value"]
    J --> K["45° vj s1 w1"]
    K --> L["κf"]
    L --> M["0 to 1"]
```
</details>

Fig. 3. Overview of the localizability detection module. A - Information Analysis: An exemplary histogram shown for one eigenvector ${ \pmb v } _ { j }$ direction. The histogram is color-coded to illustrate the strong- ( ) and the weak-contribution $( \mathbf { \epsilon } | \mathbf { \epsilon } )$ regions. The red $\textup { ( ) }$ region contains the to-be-filtered information pairs. B - Filtering: This step measures the alignment of all vectors in the histogram to the eigenvector $v _ { j } ,$ with $( \mathbf { \index { s 1 } } , \mathbf { \mathop { w 1 } } )$ illustrating strong and weak alignment vectors, respectively. C - Categorization: Localizability categories are assigned to each optimization eigenvector based on a decision tree.

# A. Localizability Detection Module Overview

The goal of the Loc.-Module is to approximate the null space of the ICP optimization Hessian $A ^ { \prime } .$ . The localizability vector η (spanning all 6-DoFs of the optimization problem) is introduced to achieve this goal. This localizability vector indicates which eigenvectors should be considered to lie within Hessian’s null space. Since the localizability analysis is performed directly in eigenspace (using the eigenvectors), the degenerate direction of the environment does not need to align with either the robot or the map frame. The localizability vector is defined as

$$
\begin{array}{l} \boldsymbol {\eta} = \left\{\boldsymbol {\eta} _ {t}, \boldsymbol {\eta} _ {r} \right\} \in \mathbb {R} ^ {6} \\ = \left(\left\{\eta_ {\mathrm{L}} \boldsymbol {v} _ {t _ {1}}, \eta_ {\mathrm{L}} \boldsymbol {v} _ {t _ {2}}, \eta_ {\mathrm{L}} \boldsymbol {v} _ {t _ {3}} \right\}, \left\{\eta_ {\mathrm{L}} \boldsymbol {v} _ {r _ {1}}, \eta_ {\mathrm{L}} \boldsymbol {v} _ {r _ {2}}, \eta_ {\mathrm{L}} \boldsymbol {v} _ {t _ {3}} \right\}\right) ^ {\top}, \tag {4} \\ \end{array}
$$

where $\mathbf { \Phi } _ { \mathrm { L } } { \mathbf { v } } _ { t _ { i } } = V _ { t } ( \dots , \ j ) , \ \forall \ j \ \in \ \{ 1 , \ 2 , \ 3 \}$ are the translational eigenvectors of the Hessian $A ^ { \prime }$ corresponding to t variables and expressed in the LiDAR frame, whereas ${ \tt L } { \bf { \it v } } _ { r _ { i } } =$ $V _ { r } ( \ldots , \ j ) , \ \forall \ j \in \{ 1 , \ 2 , \ 3 \}$ correspond only to the rotation r. More details on how to obtain the eigenvector matrices $\mathbf { } V _ { t }$ and $V _ { r }$ are explained in Section V-A. The localizability vector η represents the localizability state of each eigenvector in the form of a categorical variable; these discrete set of categorical variables are defined as $\eta _ { j } ~ \in ~ \{ n o n e ,$ partial, $f u l l \}$ with $j \in \{ 1 , \cdots 6 \}$ , where the categories correspond to being nonlocalizable, partially-localizable, and localizable, respectively. The sequence of actions for each localizability category will be explained in Section V-C.

# B. Localizability Aware Optimization Module Overview

Utilizing the output of the Loc.-Module, i.e., the discrete set of localizability categories η, the goal of the Opt.- Module is constructing and solving the constrained optimization problem to estimate the optimal state $\pmb { x } ^ { * }$ for the optimization problem (3), and is explained in Section VI. In this part, constrained optimization techniques based on Lagrangian-multipliers are used to obtain the best possible solution given the observed localizability. The possible outcomes include the registration initial guess to be unchanged in the directions where $\eta _ { j } = n o n e ,$ changed in a controlled manner for $\pmb { \eta } _ { j } = p a r t i a l$ or updated without any constraints for $\eta _ { j } = f u l l$ .

The complete X-ICP framework (cf. Fig. 2) offers reliable robot pose estimation in the presence of LiDAR degeneracy. The main contributions of this work are explained and highlighted in Sections V and VI.

# V. LOCALIZABILITY DETECTION MODULE

In this section, the details of the Loc.-Module are described according to the information flow shown in Fig. 3. Localizability detection aims to measure the information within the correspondences to identify the under-constrained directions correctly. As shown in Fig. 3, given the correspondences, the first task is the Information Analysis indicated by Fig. 3-A, analyzing the relations between the Hessian and the geometric information from the environment. In the second step, the Filtering (Fig. 3-B), the redundant contribution values from the Information analysis (Fig. 3-A) step are filtered out. Finally, the filtered contribution information is inferred, leading to a fine-grained Categorization step, as shown in Fig. 3-C. After the correspondence search, the matched correspondences are transformed back to the LiDAR frame (L) to eliminate the scale effect of the map’s physical size on the localizability of rotational eigenvectors, and the localizability analysis is performed in this frame.

# A. Information Analysis

1) Principal Component Analysis: The information analysis starts with an eigenvalue analysis of the Hessian matrix of the optimization problem. For the derivation of the matrix for a point-to-plane ICP cost function, the reader is referred to Section III, where the Hessian is provided as A′ in Equation (3). The Hessian can be divided into sub-matrices based on the relation to the minimization variables x:

$$
\boldsymbol {A} ^ {\prime} = \left[ \begin{array}{c c} \boldsymbol {A} _ {r r} ^ {\prime} & \boldsymbol {A} _ {r t} ^ {\prime} \\ \boldsymbol {A} _ {t r} ^ {\prime} & \boldsymbol {A} _ {t t} ^ {\prime} \end{array} \right] _ {6 \times 6}.
$$

Here, $\pmb { A } _ { r r } ^ { \prime } \in \mathbb { R } ^ { 3 \times 3 }$ exclusively contains information related to the rotation variables. Similarly, $\pmb { A } _ { t t } ^ { \prime } \in \mathbb { R } ^ { 3 \times 3 }$ exclusively contains information related to the translation variables.

Moreover, as discussed in Section III-A, the ICP Jacobian consists of two independent elements, the n and $\mathbf { \nabla } _ { \mathbf { \pmb { p } } } \times \mathbf { \nabla } _ { \mathbf { \pmb { n } } }$ for translation and rotation, respectively.

As it is not trivial to treat $t \in \mathbb { R } ^ { 3 }$ and $r \in { \mathfrak { s o } } ( 3 )$ together due to differences in scale and type, only $\pmb { A } _ { t t } ^ { \prime }$ and $\pmb { A } _ { r r } ^ { \prime }$ are included in the eigen-analysis of the SVD. If all the elements of the Hessian are used for the eigen-analysis, the scale difference between rotation and translation elements will complicate setting heuristic free localizability parameters. For the rotation and translation components, the resulting eigen-decomposition is given as

$$
\pmb {A} _ {t t} ^ {\prime} = \pmb {V} _ {t} \Sigma_ {t} \pmb {V} _ {t} ^ {\top}, \quad \pmb {A} _ {r r} ^ {\prime} = \pmb {V} _ {r} \Sigma_ {r} \pmb {V} _ {r} ^ {\top},
$$

where $V _ { \mathrm { ~ \scriptsize ~ t ~ } } \in \ \mathbb { R } ^ { 3 \times 3 }$ and $V _ { \boldsymbol { r } } ~ \in ~ \mathbb { R } ^ { 3 \times 3 }$ are the eigenvectors t  in matrix form, and $\Sigma _ { t } ~ \in ~ \{ \mathrm { d i a g } ( \pmb { v } ) ~ : ~ \pmb { v } ~ \in ~ \mathbb { R } _ { \geq 0 } ^ { n } \}$ and $\Sigma _ { r } ~ \in ~ \{ \mathrm { d i a g } ( \pmb { v } ) ~ : ~ \pmb { v } ~ \in ~ \mathbb { R } _ { > 0 } ^ { n } \}$ are diagonal matrices with the eigenvalues of $\pmb { A } _ { t t } ^ { \prime }$ and $\pmb { A } _ { r r } ^ { \prime }$ as the diagonal entries, respectively. Intuitively, the eigenvalues in $\Sigma _ { t }$ and $\Sigma _ { r }$ provide a direct measure of the information along each eigenvector it is paired with. However, as discussed in Section II, eigenvalues can behave inconsistently in different environments and for varying sensors, and hence are not used directly as part of this work’s localizability estimation.

2) Information Pair Contribution: The second part of the proposed information analysis is to formulate the contribution of each information pair, defined as $\left( \mathbf { \Phi } _ { \mathrm { L } } \mathbf { \Phi } _ { p } , \mathbf { \Phi } _ { \mathrm { L } } \mathbf { n } \right)$ . A formal relationship is required between the information pairs and the optimization cost to assess how much a pair contributes to the cost. Gelfand et al. [38] showed that the squared summation of these contributions provides reasonable estimates of the eigenvalues of the optimization Hessian, suggesting that the Jacobian-based localizability formulation and eigenvalues are correlated. Similar to other works [38], [39], [63], X-ICP utilizes the elements of the Jacobian (instead of the Hessian) directly, as inferred from Equation (2), simplifying the formulation and allowing for more practical deployment in various environments while keeping the formulation correlated to the optimization Jacobian.

a) Analogy to Classical Mechanics: In an analogy to classical mechanics, the optimization Jacobian measures the magnitude of the wrench induced locally by each information pair. A wrench system consist of elements such as force(n) and torque $( { \pmb { \tau } } = { \pmb p } \times { \pmb n } )$ . An illustration of the underlying wrench system and localizability concept in 2D is provided in Fig. 4, illustrating an environment consisting of two perpendicular and a semi-circular wall. Moreover, the shown map frame M does not need to align with the principal directions of the environment $( \mathrm { e } . \mathrm { g } . , v _ { t _ { 1 } } )$ , as the localizability detection of X-ICP is done in the optimization’s eigenspace, rendering it invariant to the orientation of the robot with respect to the environment.

Information analysis of the point and surface normal pairs $( p _ { i } , n _ { i } ) , \forall \ i \ \in \ \{ 1 , 2 , \ 3 \}$ show different contributions towards the localizability of different (optimization) principal directions. As an example, the surface normals $^ { n _ { 1 , 2 , 3 } }$ provide translational localizability contributions towards the direction of the eigenvector $\pmb { v } _ { t _ { 1 } }$ (due to $n _ { 1 , 2 , 3 } \cdot v _ { t _ { 1 } } > 0 )$ , whereas the surface normal $\mathbf { \Delta } _ { \mathbf { n } _ { 4 } }$ does not provide any contribution for this direction. As another example, for the rotational contribution around the z-axis (in M), it holds: $\tau _ { 1 } > \tau _ { 3 } > \tau _ { 2 } \approx 0$ .

![](images/676aeefa2ecb736148c7e8a57508ae6ba665adcb1d13d92fefb73d5a5854b82f.jpg)

<details>
<summary>text_image</summary>

p1
n1
p2
τ2 ≈ 0
n2
τ1
LiDAR vt1
τ3
Center
n4
n3
p3
y
x
M
</details>

Fig. 4. A 2D-example illustrating the contribution to localizability. Points p (green arrows), surface normals n (blue arrows), and the LiDAR center (red dot ) are shown. The span of one of the eigenvectors is depicted in orange. Finally, the induced torques (τ ) are shown for three point- and normal-pairs.

Although intuitive, the torque formulation does not provide a generalizable localizability parameterization without normalization. The reason is that the point p can be at arbitrary long distances, leading to higher torque values, which limits the generalizability of localizability formulation in different environments for practical applications. Kwok and Tang [39] studied the influence of this scale difference for different normalization techniques in ICP. They suggest that maximum norm-normalization performs better than momentnormalization, or average norm-normalization, as done in [38]. This holds for obtaining a reliable solution to the ICP problem, where spatial relations between correspondence are crucial for the estimation. However, this is not necessarily the case for point-wise contribution calculation for localizability estimation. Considering this and the requirement that the point norm should not affect the contribution value directly, this work proposes to use moment normalization, which maps the torque values to a unit sphere (cf. Equation (5)).

After the moment normalization of the individual torque vectors, the wrench system is stacked for all available information pairs to form the information matrices as follows:

$$
\boldsymbol {\mathcal {F}} _ {r} = \left[ \frac {\boldsymbol {p} _ {1} \times \boldsymbol {n} _ {1}}{\| \boldsymbol {p} _ {1} \times \boldsymbol {n} _ {1} \| _ {2}} \quad \dots \quad \frac {\boldsymbol {p} _ {N} \times \boldsymbol {n} _ {N}}{\| \boldsymbol {p} _ {N} \times \boldsymbol {n} _ {N} \| _ {2}} \right] ^ {\top}, \tag {5}
$$

$$
\boldsymbol {\mathcal {F}} _ {t} = \left[ \begin{array}{c c c} \boldsymbol {n} _ {1} & \ldots & \boldsymbol {n} _ {N} \end{array} \right] ^ {\top}.
$$

Here, $\pmb { \mathcal { F } } _ { r } \in \mathbb { R } ^ { N \times 3 }$ and $\mathcal { F } _ { t } \in \mathbb { R } ^ { N \times 3 }$ are the rotational and translational information matrices, respectively. The final task in the information analysis step is to compute the localizability contribution from the information matrices $\mathcal { F } _ { r }$ and $\mathcal { F } _ { t }$ . Instead of defining the localizability information in Equation (5) as a function of quadratic terms $\mathrm { ( i . e ~ \ ( } p \times n \mathrm { ) } ^ { 2 }$ and $\left( n \right) ^ { 2 } )$ , it is defined linearly. The used terms $( p \times n )$ and (n) prevent scale compression of the contributions.

For numeric stability, $\mathrm { i f } \left| \tau \right|$ is close to zero (i.e., the surfacenormal and point vectors are near-parallel), the information pair is dropped. An example of this phenomenon is shown for the $\pmb { p } _ { 2 } .$ , n2 information pair depicted in Fig. 4. Furthermore, the moment normalization is only applied for information pairs with $| \tau | \geq 1$ , preventing the inflation of torque values from within onto the unit sphere, which might push the Loc.-Module towards optimistic localizability estimation.

b) Localizability Contributions: The localizability concept is defined in the eigenspace of the optimization, and hence, can be obtained for every eigenvector direction. This ensures that the detection is not affected by the orientation of the LiDAR or the robot w.r.t the environment, which is useful for practical applications. Thus, the information matrices defined in Equation (5) are projected onto the eigenspace of the translation and rotation Hessians. To achieve this, the eigenvector matrices $\mathbf { V } _ { r }$ and $\mathbf { V } _ { t }$ will be used as follows:

$$
\boldsymbol {\mathcal {I}} _ {r} = \left(\boldsymbol {\mathcal {F}} _ {r} \cdot \mathbf {V} _ {r}\right) ^ {| \cdot |}, \quad \boldsymbol {\mathcal {I}} _ {t} = \left(\boldsymbol {\mathcal {F}} _ {t} \cdot \mathbf {V} _ {t}\right) ^ {| \cdot |}. \tag {6}
$$

Here, $\pmb { \mathcal { I } } _ { r } , \pmb { \mathcal { I } } _ { t } \in \mathbb { R } ^ { N \times 3 }$ are the localizability contributions for all information pairs $\{ p , n \}$ , projected by the eigenvectors in $\{ \mathbf { V } _ { r } , \ \mathbf { V } _ { t } \}$ . The $( \ldots ) ^ { | \cdot | }$ operator indicates the element-wise absolute value of the vector. Concurrently, the scalar values in $\scriptstyle { \mathcal { Z } } _ { r }$ and $\scriptstyle \pmb { \mathscr { Z } } _ { t }$ are direct indicators of localizability contribution of a certain direction. If the scalar value is $\begin{array} { r } { \mathcal { I } ( i , \ . . . ) = } \end{array}$ $1 . 0 , i \in \ 1 , \ldots , N$ , the direction’s localizability contribution is maximum, and if $\mathcal { T } ( i , \ . . . ) \ = \ 0 . 0 , \ i \ \in \ 1 , . . . , N$ , the information pair has no contribution to the localizability of the eigenvector direction.

An example of the localizability contributions is provided with a histogram as shown in Fig. 3-A for a single translation eigenvector. The contribution values are dominantly at the lower-end, indicating low contribution. Nevertheless, there is a high peak at around the contribution value of 1.0, suggesting the presence of a small but highly contributing structure. While pair-wise quantification of localizability contributions is essential, this pair-wise information needs to be filtered and consolidated for categorizing the localizability state of the eigenvectors of the optimization Hessian.

# B. Filtering

Given the localizability contributions defined in Equation $( 6 ) , \mathcal { T } = [ \mathcal { T } _ { r } , \mathcal { T } _ { t } ] \in \dot { \mathbb { R } } ^ { N \times 6 }$ , the goal of the filtering step is to remove redundant information and to render the present information interpretable.

1) Filtering Low Contribution: An example of redundant information is shown in Fig. 3-A, where the low contribution region highlighted in red ${ \bf \Xi } ( \mathrm { ~ \bf ~ \Lambda ~ } )$ dominates the analysis. If the localizability contribution is small, it might become indistinguishable from measurement- or feature extraction noise. This step addresses this issue by employing binary element-wise filtering as an outlier rejection step. The filtering operation is implemented as re-assignment:

$$
\boldsymbol {\mathcal {I}} _ {c} ^ {\prime} (i, j) = \left\{ \begin{array}{l l} \boldsymbol {\mathcal {I}} (i, j), & \text { if   } \boldsymbol {\mathcal {I}} (i, j) \geq \kappa_ {f} \\ 0, & \text { otherwise } \end{array} \right. \tag {7}
$$

Here the indices are defined as $i \in \{ 1 , \ldots , N \}$ and $j \in$ $\{ 1 , \ldots , 6 \} . \ \mathcal { T } _ { c } ^ { \prime }$ is the filtered localizability contribution matrix, which contains all reliable localizability contribution values. Moreover, $\kappa _ { f }$ is the filtering parameter, the first user-defined parameter. Since this parameter captures the sensor and feature extraction noise for different LiDAR sensors, it should be re-adjusted. This parameter is set to $\kappa _ { f } = \cos ( 8 0 ^ { \circ } ) \approx 0 . 1 7 3 6$ throughout all Velodyne experiments. For the Seemühle experiment with Ouster OS0-128, $\kappa _ { f }$ is set to $c o s ( 6 0 ^ { \circ } ) = 0 . 5$ due to a higher point variance noisecharacteristic compared to the Velodyne ${ \mathrm { V L P } } { \cdot } 1 6 ,$ resulting in more aggressive filtering; localizability contribution values with an alignment angle higher than $6 0 ^ { \circ }$ are filtered out.

2) Filtering High Contribution: Using this filtered localizability contribution $\mathcal { Z } _ { c } ^ { \prime }$ , the contributions larger than $\kappa _ { f } .$ , can be combined to summarize the available geometrical information:

$$
\mathcal {L} _ {c} (j) = \sum_ {i = 1} ^ {N} \mathcal {I} _ {c} ^ {\prime} (i, j). \tag {8}
$$

Here, $\pmb { \mathcal { L } } _ { c } \in \mathbb { R } ^ { 1 \times 6 }$ is the combined localizability contribution vector over all reliable information pairs. Here, a higher number of filtered contribution values indicate higher available contribution information. Matrix $\mathcal { Z } _ { c } ^ { \prime }$ still contains weak but reliable contributions, which might be required during partial localizability. However, a measure of the strongest contributions is still required to identify the more fine-grained status of the localizability. The filtering step is based on geometrical vector alignment; only the alignment values greater than $\cos ( 4 5 ^ { \circ } ) \approx 0 . 7 0 7$ will be considered a strong contribution, justified through geometric relations. In Fig. 3- B, the separation of strong and weak alignment regions against an eigenvector is visualized. The inner green cone indicates the region with strong vector alignment, whereas the yellow region indicates the weaker one. This separation is formulated as follows:

$$
\mathcal {I} _ {s} ^ {\prime} (i, j) = \left\{ \begin{array}{l l} \mathcal {I} _ {c} ^ {\prime} (i, j), & \text { if } \mathcal {I} _ {c} ^ {\prime} (i, j) \geq c o s (4 5 ^ {\circ}) \\ 0, & \text { otherwise }, \end{array} \right. \tag {9}
$$

$$
\boldsymbol {\mathcal {L}} _ {s} (j) = \sum_ {i = 1} ^ {N} \boldsymbol {\mathcal {I}} _ {s} ^ {\prime} (i, j).
$$

Here, $\pmb { { \mathcal { L } } } _ { s } \in \mathbb { R } ^ { 1 \times 6 }$ is the strong localizability contribution vector. Similar to the combined localizability contribution, this expression is also affected by an increased number of filtered localizability contribution values; however, it is less sensitive to the sensor noise and only affected by incorrect point correspondence errors due to the second filtering step. These strong and combined localizability contribution vectors are crucial for the next step, where these vectors will be used to categorize the localizability.

# C. Categorization

In X-ICP, the localizability categorization is defined by a set of discrete categorical variables, as described in Section IV-A. In principle, the chosen localizability contribution parameterization allows for a continuous localizability formulation. This continuous formulation could be included in the optimization objective as soft constraints in the form of additional cost elements in the optimization objective. However, removing the discretization step, introduces further challenges, such as balancing the cost elements and possible constraint violations.

Considering this point, X-ICP introduces discrete localizability categorization while keeping the underlying localizability contribution formulation continuous. This discretization allows for the introduction of hard constraints into the least squares optimization problem in X-ICP, which is advantageous in degeneracy mitigation; the satisfaction of these constraints avoids divergence along the constrained directions.

a) Localizability Parameters: To achieve this goal, three localizability parameters are introduced: i) Parameter $\kappa _ { 1 }$ is the safety localizability threshold defining the lower bound of being fully localizable. ii) Parameter $\kappa _ { 2 }$ regulates the transition from being localizable to partially-localizable and represents the upper bound of being partially localizable. iii) Parameter $\kappa _ { 3 }$ is the minimum information threshold that covers the case when the environment has sparse yet well-contributing information and regulates the transition from being partiallylocalizable to non-localizable.   
b) Parameter Choice: The three thresholds provide a natural relationship among each other: $\kappa _ { 1 } \geq \kappa _ { 2 } > \kappa _ { 3 }$ . These parameters are set based on the basin of convergence of the employed ICP algorithm. The rules for setting the parameters are as follows:

1) As $\kappa _ { 1 }$ sets the boundary between being localizable and {partial-localizability, non-localizable}, it can be set (almost) arbitrarily high. Note that a too high value of $\kappa _ { 1 }$ introduces the additional computational cost of the localizability detection, while the partial-localizability utilizes the available information still adequately. An example value of $\kappa _ { 1 } = 2 5 0$ allows the ICP optimization to run with a minimum of 250 perfectly aligned pairs.   
2) $\kappa _ { 2 }$ is a parameter selected based on the system characteristics as robustness, and optimizer. Jointly with $\kappa _ { 3 }$ , it defines how the partial-localizability should be approached. It should be set between $\kappa _ { 1 }$ and $\kappa _ { 3 }$ .   
3) Finally, $\kappa _ { 3 }$ sets the boundary between the partiallocalizable and non-localizable status. A value of 35 allows the constrained ICP optimization to run in a controlled manner with a minimum of 35 sampled pairs.

Increasing $\kappa _ { 1 }$ would make the degeneracy detection more aggressive, with only minimal added computational cost. On the other hand, $\kappa _ { 2 }$ sets the uncertain region, for which partiallocalizability takes place. This allows the user to adjust the behavior of the degeneracy awareness framework in exchange for the computational cost of correspondence re-sampling. Lastly, lowering $\kappa _ { 3 }$ increases the localizability risk of the system, which is user-defined. Using these intuitions and the noise properties of point-to-plane ICP, in this work, the localizability parameters are set to $\kappa _ { 1 } ~ = ~ 2 5 0 , ~ \kappa _ { 2 } ~ = ~ 1 8 0 .$ , and $\kappa _ { 3 } ~ = ~ 3 5$ for all environments and sensors shown in Section VII.

c) Decision Tree: These parameters are used in a decision tree to get the localizability categories as shown in Fig. 3-C. The decision tree takes the filtered localizability contribution matrices and an eigenvector as input. The decision tree operates eigenvector-wise; thus, the required binary comparisons are repeated for all 6 principal directions. First, for each eigenvector direction, $\kappa _ { 1 }$ is compared against $\mathcal { L } _ { c } ,$ and $\kappa _ { 2 }$ against $\mathcal { L } _ { s } .$ If either of these comparisons suggests that the problem is well-constrained, then the direction is localizable, $\eta _ { \pmb { v } _ { j } } ~ = ~ f u l l$ . Secondly, suppose the first comparisons suggest that the problem is not well-constrained; in this case, $\kappa _ { 2 }$ is compared against $\scriptstyle { \mathcal { L } } _ { c }$ and $\kappa _ { 3 }$ against $\pmb { { \mathcal { L } } } _ { s }$ to understand the amount of present information. If either of these comparisons holds, the localizability category is assigned as partial, $\eta _ { \pmb { v } _ { j } } = p a r t i a l$ , otherwise it is assigned non-localizable, $\eta _ { \pmb { v } _ { j } } = n o n e$ .

![](images/cf4720e900ba39d5dce9136c9d2c8bd41518690169231fca38fb8ac670d8dd6a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph A (Sec. VI-A)
        A1["3D Coordinate System with t, r"] --> A2["Constrained Optimization"]
        A2 --> A3["3D Coordinate System with v_j ∈ V_t, V_r"]
    end

    subgraph B (Sec. VI-B)
        B1["3D Coordinate System with x_{t2}^*, x_{r2}^*, v_j"] --> B2["Constrained Optimization"]
        B2 --> B3["3D Coordinate System with x_{t1}^*, x_{r1}^*"]
    end

    A1 -->|η| A2
    A2 -->|x*| B2
```
</details>

Fig. 5. Overview of the constrained optimization module. The Opt.-Module steps are: i) linear Constraint Calculation, which is in the form of a 3D plane, and ii) Constrained Optimization which employs these constraints.

# VI. LOCALIZABILITY AWARE OPTIMIZATION MODULE

The Opt.-Module is responsible for the calculation and usage of additional constraints to perform reliable optimization in cases of degeneracy. An overview of this module is presented in Fig. 5. Given the localizability categories η and eigenvectors $\mathtt { L } ^ { v _ { j } }$ , the objective of this module is to find the optimal solution $\mathbf { \nabla } _ { \mathbf { \mathcal { X } } } \ast \mathbf { \ v { x } }$ , even for an ill-conditioned optimization. In Section VI-A, the calculation of optimization constraints for each localizability category is detailed and referred to as Constraint Calculation. In Section VI-B, these constraints are then integrated into the optimization problem (Constrained Optimization). Since the full point cloud registration is performed in the map frame M, all data in this section is also expressed in the map frame. At each iteration of ICP, also the eigenvectors are rotated to the map frame by applying $\pmb { R } _ { \mathrm { M } , \mathrm { L } _ { k } } = \pmb { R } _ { \mathrm { M } , \mathrm { L } _ { 1 } } \cdot \pmb { R } _ { \mathrm { L } _ { 1 } , \mathrm { L } _ { 2 } } \cdot \cdot \cdot \cdot \cdot \pmb { R } _ { \mathrm { L } _ { k - 1 } , \mathrm { L } _ { k } }$ , where k refers to the current index of the ICP iteration.

# A. Constraint Calculation

There are multiple ways to enforce linear equality constraints during optimization. In this module, the constraint should limit the solution space of the optimization in 3D as depicted in Fig. 5-A, along the ill-conditioned directions. Corresponding optimization eigenvector directions that are going to be constrained are expressed in the same geometrical frames as the optimization variables t and r. Each constraint direction is given by ${ \pmb v } _ { \{ t , r \} _ { j } } , ~ j \in \{ 1 , 2 , 3 \}$ , and the admissible magnitude of the pose update is defined by values $\scriptstyle t _ { 0 }$ and $\mathbf { \boldsymbol { r } } _ { 0 }$ which are calculated per constraint direction. The constraints are defined as

$$
\boldsymbol {v} _ {t _ {j}} ^ {\top} \cdot (\boldsymbol {t} - \boldsymbol {t} _ {0}) = 0,
$$

$$
\boldsymbol {v} _ {r _ {j}} ^ {\top} \cdot (\boldsymbol {r} - \boldsymbol {r} _ {0}) = 0,
$$

where v⊤{t,r}j $\boldsymbol { v } _ { \{ t , r \} _ { j } } ^ { \top }$ are eigenvectors in translational $( \in \textbf { } V _ { t } )$ and rotational subspace $( \in V _ { r } )$ , respectively. Each line of Equation (10) defines a 3D plane with normal vectors ${ \pmb v } _ { \{ t , r \} _ { . } }$ j and plane points $\scriptstyle t _ { 0 }$ and $\mathbf { \boldsymbol { r } } _ { 0 }$ .

Conceptually, $\scriptstyle t _ { 0 }$ and $\mathbf { \boldsymbol { r } } _ { 0 }$ represent the pose estimates in the direction of the eigenvector they correspond to. An example of such a single constraint is illustrated in Fig. 5-A. Given this constraint, a certain amount of the ICP pose update is imposed by $\scriptstyle t _ { 0 }$ or $\mathbf { \nabla } _ { \mathbf { r } _ { 0 } } .$ , while the final optimized solution $( x _ { t } ^ { * } , r _ { r } ^ { * } )$ has to lie on the constraint plane as shown in Fig. 5-B.

1) Localizability Categories: Optimization constraints are added based on the localizability category of the corresponding eigenvector ${ \pmb v } _ { j }$ . Here, the implications of different types of localizability categories are explained:

• $\eta _ { v _ { i } } = f u l l ;$ The optimization is carried out nominally without added constraints.   
• $\eta _ { v _ { j } } = n o n e \colon$ The optimization direction along ${ \pmb v } _ { j }$ is found to be non-localizable, and a constraint in this direction is employed by setting $\scriptstyle t _ { 0 }$ or $\mathbf { \boldsymbol { r } } _ { 0 }$ to ${ \bf 0 } _ { 3 \times 1 }$ . The pose update calculated by the ICP optimization will vanish along directions $\eta _ { \pmb { v } _ { j } } = n o n e .$ .   
• $\eta _ { v _ { j } } = p a r t i a l { : }$ The optimization direction along ${ \pmb v } _ { j }$ is found to be partially-localizable. The partial constraint value is calculated by re-sampling the available information from the correspondences, limiting the pose update in the direction of $\boldsymbol { v } _ { j }$ . The exact procedure is explained in the following section.

2) Partial Localizability: Calculating the partial localizability constraint values requires re-sampling the ICP correspondences M. The number of pairs that need to be resampled depends on the categorization process (cf. Fig. 3-C). If the statement $\mathcal { L } _ { c } \geq \kappa _ { 2 }$ is true, then the information pairs used for calculating $\scriptstyle { \mathcal { L } } _ { c }$ will be re-used for the calculation of the constraints. Otherwise, if the statement $\mathcal { L } _ { s } ~ \geq ~ \kappa _ { 3 }$ is true, then the information pairs used to calculate the $\mathcal { L } _ { s }$ will be re-used. If both statements are true, then the information pairs used for calculating $\scriptstyle { \mathcal { L } } _ { c }$ will be re-used. Once partial localizability is detected for any direction, the correspondence re-sampling process is employed, which aims to find the information pairs with the highest localizability contribution in the direction of the eigenvector. These pairs are then used to calculate a reliable pose estimate. This is achieved by solving a simplified least-squares minimization problem of the resampled pairs (cf. Equation 11 and 12), yielding the constraint values $\mathbf { \delta } _ { t _ { 0 } , r _ { 0 } }$ . The overall re-sampling process is as follows:

1) Acquire the degenerate eigenvector, ${ \pmb v } _ { j }$ and the previously calculated localizability contribution vectors, $\mathcal { L } _ { s }$ and $\mathcal { L } _ { c } .$   
2) Decide on the number of pairs to sample based on $\pmb { \mathcal { L } } _ { c } \geq$ $\kappa _ { 2 }$ and $\mathcal { L } _ { s } \geq \kappa _ { 3 }$ conditions.   
3) Select the pairs based on their localizability contribution values. The localizability contribution value calculation is explained in depth in Section V-A.   
4) Based on the sub-space of ${ \pmb v } _ { j } .$ , the simplified minimization problem in either Equation (11) or (12) is solved.

For the case of $\pmb { v } _ { j } ~ \in ~ \pmb { V } _ { t } ,$ , the constraint value $\scriptstyle t _ { 0 }$ can be computed by using the re-sampled pairs and solving the following minimization problem:

$$
\begin{array}{l} \begin{array}{l} ^ {r e} \boldsymbol {A} _ {t} = [ ^ {r e} \boldsymbol {n} ] [ ^ {r e} \boldsymbol {n} ] ^ {\top}, \quad^ {r e} \boldsymbol {b} _ {t} = [ ^ {r e} \boldsymbol {n} ] [ ^ {r e} \boldsymbol {n} ] ^ {\top} (^ {r e} \boldsymbol {q} - ^ {r e} \boldsymbol {p}) \\ \cdot | | _ {r e, 1, r e, 1} | | \end{array} \tag {11} \\ \min _ {\boldsymbol {t} _ {0} \in \mathbb {R} ^ {3}} \left| \left| ^ {r e} \boldsymbol {A} _ {t} \boldsymbol {t} _ {0} - ^ {r e} \boldsymbol {b} _ {t} \right| \right| _ {2}. \\ \end{array}
$$

Here, the re-sampled information pairs are denoted as $\left\{ { { ^ { r e } } } p , \{ { ^ { r e } } n , { ^ { r e } } q \} \right\} \in \mathcal { M }$ . Similarly, for the case of $\pmb { v } _ { j } \in V _ { r }$ the constraint value $\mathbf { \boldsymbol { r } } _ { 0 }$ is calculated as follows:

$$
\begin{array}{l} { } ^ { r e } \boldsymbol { A } _ { r } = [ { } ^ { r e } \boldsymbol { p } \times { } ^ { r e } \boldsymbol { n } ] [ { } ^ { r e } \boldsymbol { p } \times { } ^ { r e } \boldsymbol { n } ] ^ { \top } \\ { } ^ { r e } \boldsymbol { b } _ { r } = [ { } ^ { r e } \boldsymbol { p } \times { } ^ { r e } \boldsymbol { n } ] [ { } ^ { r e } \boldsymbol { n } ] ^ { \top } ( { } ^ { r e } \boldsymbol { q } - { } ^ { r e } \boldsymbol { p } ) \tag {12} \\ \min _ {\boldsymbol {r} _ {0} \in \mathbb {R} ^ {3}} \left| \left| ^ {r e} \boldsymbol {A} _ {r} \boldsymbol {r} _ {0} - ^ {r e} \boldsymbol {b} _ {r} \right| \right| _ {2}. \\ \end{array}
$$

The outputs $\scriptstyle t _ { 0 }$ or $\mathbf { \boldsymbol { r } } _ { 0 }$ are the motion estimates based on the re-sampled correspondences in the direction of the degenerate eigenvector ${ \pmb v } _ { j } .$ As described in Section V, partial localizability indicates available but sparse information along the degenerate directions, leveraged by the re-sampling process in the form of less-noisy and more reliable correspondences.

However, as the data used in these minimization problems are deliberately selected to provide information in a specific direction, the Hessian matrices ${ \bf \Delta } ^ { r e } { \bf A } _ { t }$ and ${ } ^ { r e } A _ { \eta }$ might not be well-conditioned in some cases. To mitigate this risk, first, the Hessian matrices are factorized via LU decomposition with pivoting [64]. Here the pivoting operation increases accuracy by interchanging rows to make the pivot element larger than any element below. Next, to reduce the possible adverse effects of ill-conditioning of the factorized problem, RIF preconditioning [65] is applied. An alternative to this would be the sampling of points in the well-constrained directions [38] in addition to the degenerate directions, improving the conditioning of the Hessian. Yet, this re-sampling is not performed here due to its additional computational burden. Without these precautions, the calculated constraints from the re-sampling might not be accurate, and the residuals generated through projection onto the eigenvectors might negatively affect the convergence of the well-constrained directions.

# B. Constrained Optimization

In the final step, the calculated constraint values $\scriptstyle t _ { 0 }$ and $\mathbf { \boldsymbol { r } } _ { 0 }$ are incorporated into the least squares minimization problem. In order to be applicable, the individual constraints of Equation (10) must be extended and re-arranged to 6D:

$$
\begin{array}{l} \left[ \mathbf {0} _ {1 \times 3}, \boldsymbol {v} _ {j} \right] \cdot \boldsymbol {x} = \boldsymbol {v} _ {j} \cdot \boldsymbol {t} _ {0}, \quad \text { if } \boldsymbol {v} _ {j} \in \boldsymbol {V} _ {t}, \tag {13} \\ \left[ \boldsymbol {v} _ {j}, \mathbf {0} _ {1 \times 3} \right] \cdot \boldsymbol {x} = \boldsymbol {v} _ {j} \cdot \boldsymbol {r} _ {0}, \quad \text { if } \boldsymbol {v} _ {j} \in V _ {r}. \\ \end{array}
$$

Next, the constraints are brought to a matrix form of $C x = d .$ The number of constraints corresponds to the amount of none and partial categories along all eigenvectors, and is denoted as $m _ { t }$ and $m _ { r } .$ , with the total number of constraints $c = m _ { t } +$ $m _ { r } \le 6$ . The final constraint matrix is

$$
\underbrace {\left[ \begin{array}{c c} \mathbf {0} _ {m _ {r} \times 3} & \boldsymbol {v} _ {j} \\ \vdots & \vdots \\ \boldsymbol {v} _ {j} & \mathbf {0} _ {m _ {t} \times 3} \end{array} \right]} _ {\boldsymbol {C} _ {(m _ {r} + m _ {t}) \times 6}} \boldsymbol {x} = \underbrace {\left[ \begin{array}{c} \boldsymbol {v} _ {j} \cdot \boldsymbol {r} _ {0} \\ \vdots \\ \boldsymbol {v} _ {j} \cdot \boldsymbol {t} _ {0} \end{array} \right]} _ {\boldsymbol {d} _ {(m _ {r} + m _ {t}) \times 1}}, \tag {14}
$$

where each row indicates an equality constraint. The whole optimization can then be re-expressed as

$$
\min _ {\boldsymbol {x} \in \mathbb {R} ^ {6}} \left. \left| \left| \boldsymbol {A} ^ {\prime} \boldsymbol {x} - \boldsymbol {b} ^ {\prime} \right| \right| _ {2}, \right. \tag {15}
$$

$\mathrm { s . t . } \quad C x - d = 0 ,$

with $\boldsymbol { C } ~ \in ~ \mathbb { R } ^ { c \times 6 }$ and $\textbf { \textit { d } } \in \ \mathbb { R } ^ { c \times 1 }$ . Problem (15) can be transformed into an unconstrained optimization problem by introducing Lagrangian multipliers [66], resulting in the final augmented linear least squares minimization problem

$$
\min _ {\boldsymbol {x} ^ {\prime} \in \mathbb {R} ^ {6}} \left| \left| \boldsymbol {A} ^ {\prime \prime} \boldsymbol {x} ^ {\prime} - \boldsymbol {b} ^ {\prime \prime} \right| \right| _ {2}, \tag {16}
$$

with augmented optimization vector $\mathbf { { x } ^ { \prime } } ~ = ~ [ \mathbf { { x } ^ { * } } ^ { \top } , ~ { \lambda ^ { * } } ^ { \top } ] ^ { \top }$ with Lagrangian multipliers $\pmb { \lambda } \in \mathbb { R } ^ { c \times 1 }$ , an example of such an optimization problem is visualized in Fig. 5-B for an optimization problem with a single equality constraint. The augmented matrices in Equation (16) are defined as:

$$
\underbrace {\left[ \begin{array}{c c} 2 \boldsymbol {A} ^ {\top} \boldsymbol {A} & \boldsymbol {C} ^ {\top} \\ \boldsymbol {C} & \boldsymbol {0} \end{array} \right]} _ {\boldsymbol {A} ^ {\prime \prime}} \underbrace {\left[ \begin{array}{c} \boldsymbol {x} ^ {*} \\ \boldsymbol {\lambda} ^ {*} \end{array} \right]} _ {\boldsymbol {x} ^ {\prime}} = \underbrace {\left[ \begin{array}{c} 2 \boldsymbol {A} ^ {\top} \boldsymbol {b} \\ \boldsymbol {d} \end{array} \right]} _ {\boldsymbol {b} ^ {\prime \prime}}.
$$

This optimization problem can be solved via SVD, providing the optimal pose estimation $\boldsymbol { x ^ { \prime } } ^ { * }$ for the current ICP iteration.

It should be noted that in the case of a truly bad initial guess, the Opt.-Module will not be able to solve the point cloud registration problem reliably. Two factors contribute to this: i) inaccuracies in the correspondence search, affected by the bad initial guess. ii) In cases of complete degeneracy, the non-updated optimization directions will use the initial guess, which may result in an incorrect point cloud registration. While this is the case, this initial guess is utilized for all methods throughout this work and hence, affects all compared methods alike.

# VII. RESULTS

In this section, the experimental setup is discussed in Section VII-A, followed by the performance evaluation of the proposed framework through Sections VII-C to VII-F. The attached video1 demonstrates the robot field deployment and summarizes the proposed framework. Finally, to validate the efficacy of individual sub-modules of the proposed solution, ablation studies are presented in Section VII-G.

# A. Hardware & Implementation Details

The proposed localizability-aware ICP framework is integrated into a modified C++ point cloud registration framework [61] developed by ANYbotics, which is based on the open-source registration library libpointmatcher [62], to demonstrate its suitability for challenging real-world applications. An ANYmal-C [68] legged robot, shown in Fig. 1 and Fig. 6-(a), equipped with a Velodyne VLP-16 LiDAR, an inertial measurement unit (IMU), and joint encoders, was used in all field experiments. The initial transformation $\pmb { T } _ { \mathrm { M L , i n i t } }$ for point cloud registration and correspondence search is provided by ANYmal’s leg odometry module [69], which utilizes the IMU and joint encoder measurements. As discussed in [69], the performance of this legged odometry module depends on the contact estimation performance, which is known to suffer on rough terrain. Finally, the point cloud motion compensation is performed at the driver level using the same odometry pose estimates in the transformation tree. All evaluations are performed on a laptop equipped with an Intel i7-9750H CPU, equivalent to that available on the robot.

![](images/e52df53888c11d928dd0788d73baacd3bf7ec10e47443ee3e51cbc8ed6d8152d.jpg)

<details>
<summary>natural_image</summary>

Interior view of a dimly lit tunnel with illuminated archway and train carabiner (no visible text or symbols)
</details>

![](images/531d89c6b2b840ddadb614986e1808a586b7e6ccade176ec340f86ac1605703c.jpg)

<details>
<summary>natural_image</summary>

Two-panel image: top shows a dark, cloudy landscape with visible cracks; bottom shows a grassy field with a tripod-mounted camera on the path (no text or symbols)
</details>

Fig. 6. Overview of the three real-world experiment sites: a) ANYmal in the Seemühle Underground Mine (VII-D), b) an on-board image of the Rümlang Construction Site (VII-E) showing the featureless planarity, and c) the ground truth collection at Opfikon City Park (VII-F) with the RTC360 [67] sensor.

# B. Algorithmic Comparisons

To facilitate a fair comparison with X-ICP, the current compared state-of-the-art methods (Zhang et al. [12] and Hinduja et al. [17]) are re-implemented within the same ICP registration pipeline. The method of Zhang et al. [12] requires an eigenvalue threshold for degeneracy detection, which is empirically set to 120 to ensure good degeneracy detection for all experiments except for the Opfikon Park dataset in Section VII-F, where multiple eigenvalue threshold values are compared. The proposed framework utilizes the same localizability parameters $\kappa _ { \{ 1 , 2 , 3 \} }$ for all experiments.

# C. Simulation Study

To evaluate the localizability detection performance of the proposed method, a set of tests are performed in simulation for better control over the quantities affecting the ICP registration, such as the quality of the point cloud registration prior. In addition, the designed simulation environments, shown in Fig. 7, are designed to feature smooth planar surfaces, selfsimilar corridors, a cylindrical room, and an open area - all are conditions known to induce ICP degeneracy.

1) Translational Degeneracy: In this study, pure translational degeneracy is investigated with a semi-circular tunnel environment with degeneracy along the longitudinal direction of the tunnel. This environment is shown in Fig. 7- C, with the robot path depicted in red. To penalize overreliance on the legged odometry pose prior, noise is added to the initial guess before being given to the ICP algorithm for all methods. The noise is sampled from a velocitydependent normal distributions $\mathcal { N } ( \mu _ { t } , { \sigma _ { t } } ^ { 2 } )$ and $\mathcal { N } ( \mu _ { r } , { \sigma _ { r } } ^ { 2 } )$ . As an example, for velocities $0 . 5 \mathrm { m } / \mathrm { s }$ and $0 . 2 \mathrm { r a d / s }$ the distribution variables would be $\mu _ { t } = 0 \mathrm { c m } , \ \sigma _ { t } = 0 . 0 1 2 5 \mathrm { m }$ , $\mu _ { r } ~ = ~ 0 \mathrm { r a d }$ , $\sigma _ { r } = 0 . 0 0 5 \mathrm { r a d } .$ The mapping results of this study, including an error map and the predicted localizability categories, are shown in Fig. 8. In this example, X-ICP is able to identify the degeneracy correctly along one direction, despite given the noisy registration prior. The localizability detection of Zhang et al. [12] detects degeneracy along three directions, which is incorrect. Furthermore, as seen from the error map, the generated map using Zhang et al.’s [12] method shows higher error and drift along the principal direction of the tunnel, demonstrating the advantages of X-ICP, which can successfully identify and mitigate translational degeneracy.

![](images/f44f98770dc68456c3d9af8a35b677e3fca918c3ccd7e9c7685b879101e54a1d.jpg)

<details>
<summary>text_image</summary>

A
A-1
A-2
Start / End
Map Frame
A-3
z
y
x
B
Start / End
C
End
Start
10m
</details>

Fig. 7. An illustration of the three simulated environments of Section VII-C. The degenerate directions are shown in yellow (translational) and blue (rotational) for all sub-figures, whereas the robot path is shown in red. The ground truth point clouds of the combined degeneracy (A), the rotational degeneracy (B), and the translational degeneracy (C) simulation environments are depicted above. Furthermore, A-1 is a snapshot of X-ICP’s detection of 3-axes degeneracy, A-2 is a snapshot of X-ICP being exposed to a single-axis translational degeneracy, while A-3 shows the misalignment between the map frame and the degeneracy direction, highlighting the importance of directioninvariant localizability detection in eigenspace.

2) Rotational Degeneracy: The rotational degeneracy is simulated in a cylindrical environment (cf. Fig. 7-B) with known degeneracy for rotation axes perpendicular to the cylinder’s ground surface. Similar to the translational degeneracy test, a motion-based odometry noise is added to the initial prior provided to the ICP algorithm. The results of this study are shown in Fig. 9. It can be seen that X-ICP identifies the rotational degeneracy correctly, and the resulting point cloud map contains small errors. X-ICP’s degeneracy detection nicely overlaps with the time intervals of the robot walking near the center of the environment, indicated as green overlays in Fig. 9. In contrast, the localizability detection of Zhang et al. [12] indicates degeneracy when the robot is near the circular wall, which is incorrect. This underlines the sensitivity of the detection algorithm w.r.t. the observed number of points. Furthermore, as seen from the map error comparison, the map produced using Zhang et al.’s [12] method shows higher error when compared to the map produced by X-ICP, demonstrating its efficacy for operation in rotation degenerate environments.

![](images/e0aad23a79f21e537370cb81e1710a7113de376ef28cc07c80c4da463b1c7f19.jpg)

<details>
<summary>text_image</summary>

X-ICP
Zhang et al.
0.35m
0m
</details>

![](images/a97b37db83ba2b32cc9dcfbb1a80a4eef528850572916f5bf58fe1cf1d037454.jpg)

<details>
<summary>bar</summary>

| Time[s] | ην = Full | ην = Partial | ην = None |
| ------- | --------- | ------------ | --------- |
| 0       | 31        | 0            | 0         |
| 5       | 31        | 0            | 0         |
| 10      | 31        | 0            | 0         |
| 15      | 31        | 0            | 0         |
| 20      | 31        | 0            | 0         |
| 25      | 31        | 0            | 0         |
| 30      | 31        | 0            | 0         |
</details>

![](images/14c70934bb7c2dd2b4465dda02e2cec78d12453c333a438a5ae10d9a8994ede8.jpg)

<details>
<summary>line</summary>

| Time[s] | η_v = Well-Constrained | η_v = Degenerate |
| ------- | ---------------------- | ---------------- |
| 0       | 12                     | 10               |
| 5       | 12                     | 10               |
| 10      | 12                     | 10               |
| 15      | 12                     | 18               |
| 20      | 12                     | 20               |
| 25      | 12                     | 25               |
| 30      | 12                     | 30               |
</details>

Fig. 8. Top: The point cloud maps produced using X-ICP and Zhang et al. methods for the translational degeneracy dataset. The color bar indicates the point-to-point distance error with respect to the ground truth map. Bottom: Plots showing the estimated localizability categories for both methods.   
![](images/b67d2770a1658e2c52d530675bfa82fce5f08abe12d8f4f8a8ff3a5073ff43d9.jpg)

<details>
<summary>natural_image</summary>

Two 3D thermal or density visualization panels labeled X-ICP and Zhang et al., with a color scale bar indicating intensity (0m to 0.5m)
</details>

![](images/db68c4898f2e83fbb0f14903270f84bd1548334d6309de07dcb862960fb7d4a3.jpg)

<details>
<summary>bar</summary>

| Time [s] | η_v = Full | η_v = Partial | η_v = None |
| -------- | ---------- | ------------- | ---------- |
| 0        | 0          | 0             | 0          |
| 20       | 0          | 1             | 1          |
| 40       | 0          | 1             | 1          |
| 60       | 0          | 1             | 1          |
| 80       | 0          | 1             | 1          |
| 100      | 0          | 1             | 1          |
| 120      | 0          | 1             | 1          |
</details>

![](images/900b8d6fa30fd025533e0025d96ae555985a3899c84cb8ed1ce055d950fc0b5c.jpg)

<details>
<summary>line</summary>

| Time[s] | η_v = Well-Constrained | η_v = Degenerate |
| ------- | ---------------------- | ----------------- |
| 0       | 80                     | 75                |
| 20      | 80                     | 75                |
| 40      | 80                     | 75                |
| 60      | 80                     | 75                |
| 80      | 80                     | 75                |
| 100     | 80                     | 75                |
| 120     | 80                     | 75                |
</details>

Fig. 9. Top: The point cloud maps produced using X-ICP and method from Zhang et al. [12] for the rotational degeneracy dataset. The color bar indicates the point-to-point distance error. Bottom: Plots showing the estimated localizability categories for both methods. The time interval of the robot being close to the center of the cylinder is highlighted as a shaded green area.

3) Combined Degeneracy: In this experiment shown in Fig. 7-A, the robot starts in a corridor-like area, navigates to an open space (Fig. 7-A-1), and returns to the starting position, traversing a total distance of 246 m at a nominal velocity of 0.5 m/s. In the corridor sections (shown in Fig. 7- A-2 and A-3), the ICP registration is expected to degenerate in one axis along the corridor. In contrast, in the open section (Fig. 7-A-1), the degeneracy is expected to occur along two translational directions parallel to the ground plane and in one rotational direction perpendicular to the ground plane. The results shown in Fig. 10 validate the correct degeneracy detection performance of the proposed work; non-localizability is detected along one axis throughout, corresponding to the corridor sections. Furthermore, non-localizability along two additional axes is only detected for the 200-300s interval, corresponding to the open section of the environment. The localizability detection of X-ICP is compared against the state-of-the-art methods [12], [17] in the bottom rows of Fig. 10, and a comparison of the three maps is presented in the top of Fig. 10. As discussed in the introduction of the simulation study, to eliminate the effect of an imperfect initial guess, in this simulation experiment, a perfect prior is fed to all three methods. The method of Hinduja et al. [17] remains overly pessimistic and fully relies on the ICP pose prior for the registration, resulting in less drift than Zhang et al. [12], which in contrast remains overly optimistic in the detection of rotation degeneracy, leading to a broken map. The section where Zhang et al. [12] extensively drifts is indicated with point A in Fig. 10 and the corresponding time frame is highlighted with the shaded area on the localizability estimation plots. The proposed method, which also utilizes the partial localizability information along degenerate directions, shows minimal map error w.r.t. the ground-truth map.

# D. Seemühle Underground Mine

Next, the proposed method is evaluated and studied in three real-world environments. In the first experiment, the ANYmal robot traversed an abandoned underground mine in Switzerland called Seemühle. This environment constitutes several challenges: First, the ground of the mine is uneven due to sharp rocks and rails, leading to constant foot slippage, deteriorating the ground contact estimation and, thus, the odometry prior [69] for point cloud registration. Second, the environment contains a long tunnel segment with smooth arched walls (cf. Fig. 6-a and Fig. 1-A, B, C) that do not provide reliable constraints along the principal direction of the tunnel. The environment was scanned using a Leica RTC 360 scanner (cf. Fig. 1), which is used to generate ground truth map and robot trajectory, the latter using the method in [70]. The same experiment was performed twice using the same ANYmal robot; once with the Velodyne VLP-16 LiDAR and once with the Ouster OS0-128 LiDAR, allowing for a comparative analysis of the effect of different point cloud density configurations on the proposed method. The robot traversed a total distance of 521.8 m during this experiment.

1) Velodyne VLP-16 LiDAR: The error heat maps of the three methods for the experiment conducted using the Velodyne VLP-16 LiDAR sensor are shown in the bottom row of Fig. 1. Both of the compared state-of-the-art approaches suffer from the impact of environmental degeneracy, prohibiting a drift-free traversal. The method of Zhang et al. [12] performs comparably to the proposed solution until entering the tunnel on the way back to the starting point. In this second part, the degeneracy detection works incorrectly, resulting in LiDAR slip. The localizability categories estimated by Zhang et al. [12] are shown at the bottom of Fig. 11, indicating the incorrect and pessimistic localizability estimation. The solution of Hinduja et al. [17] shows a larger registration error throughout the experiment, as depicted in the error heat map, due to a pessimistic localizability estimation as shown in the middle figure of Fig. 11. This causes over-relying on the noisy odometry prior instead of registration, thus creating an incorrect final point cloud map.

![](images/12807c43a227a39220ff37382babcebf761ce9c485403677dbab9c957b006ffd.jpg)

<details>
<summary>text_image</summary>

X-ICP
Hinduja et al.
Zhang et al.
5m
0m
</details>

![](images/eddf9dff8f28af10c2b6b55cee6b24dbc2b4d135c8f94021c2bb9a5d1a50ac88.jpg)

<details>
<summary>line</summary>

| Time [s] | ην = Full | ην = Partial | ην = None |
| -------- | --------- | ------------ | --------- |
| 0        | 0         | 0            | 0         |
| 25       | 0         | 1            | 0         |
| 50       | 0         | 0            | 0         |
| 75       | 0         | 0            | 0         |
| 100      | 0         | 0            | 0         |
| 125      | 0         | 0            | 0         |
| 150      | 0         | 0            | 0         |
| 175      | 0         | 0            | 0         |
| 200      | 0         | 0            | 0         |
| 225      | 0         | 0            | 0         |
| 250      | 0         | 1            | 0         |
| 275      | 0         | 1            | 1         |
| 300      | 0         | 1            | 1         |
| 325      | 0         | 1            | 1         |
| 350      | 0         | 1            | 1         |
| 375      | 0         | 1            | 1         |
| 400      | 0         | 1            | 1         |
| 425      | 0         | 1            | 1         |
</details>

![](images/2a142b8d7d903f8079097212ef507a5bd3330e73670e0f2e298f0973f63d7150.jpg)

<details>
<summary>line</summary>

| Time[s] | ην = Well-Constrained | ην = Degenerate |
| ------- | --------------------- | ---------------- |
| 0       | ~0                    | ~0               |
| 25      | ~0                    | Yes              |
| 125     | ~0                    | Yes              |
| 175     | ~0                    | Yes              |
| 275     | ~0                    | Yes              |
| 325     | ~0                    | Yes              |
| 375     | ~0                    | Yes              |
| 425     | ~0                    | Yes              |
| 425     | Yes                   | Yes              |
</details>

![](images/0c67092a621b5f7e71dddccdffaa8b0570d18f99c0eb11053489cba4137ba030.jpg)

<details>
<summary>line</summary>

| Time[s] | η_v = Well-Constrained | η_v = Degenerate |
| ------- | --------------------- | ---------------- |
| 0       | 0                     | 0                |
| 25      | 0                     | 0                |
| 50      | 0                     | 0                |
| 75      | 0                     | 0                |
| 100     | 0                     | 0                |
| 125     | 0                     | 0                |
| 150     | 0                     | 0                |
| 175     | 0                     | 0                |
| 200     | 0                     | 0                |
| 225     | 0                     | 0                |
| 250     | 0                     | 0                |
| 275     | 0                     | 0                |
| 300     | 0                     | 0                |
| 325     | 0                     | 0                |
| 350     | 0                     | 0                |
| 375     | 0                     | 0                |
| 400     | 0                     | 0                |
| 425     | 0                     | 0                |
</details>

Fig. 10. Top: The generated point cloud maps of the three methods for the combined simulation environment. The color bar indicates the point-to-point distance error of the produced maps. Bottom: The estimated localizability categories of X-ICP and the state-of-the-art methods are shown. The region corresponding to point A in the top figure is highlighted in the plot.

In contrast, X-ICP performs reliable and consistent registration throughout the tunnel segment, as demonstrated by the low error of the produced point cloud map in Fig. 1. The corresponding localizability estimation is shown in the top plot of Fig. 11, underlining the ability of the X-ICP Loc.-Module to capture the tunnel section at the beginning and end of the experiment. Here, the advantage of detecting localizability in eigenspace becomes clear; although the robot moves and rotates within the environment, the degeneracy only affects a single direction of the optimization, in this case, $\pmb { v } _ { t _ { 1 } }$ . Since this vector is defined in the eigenspace, it does not need to align with any of the Cartesian axes in the ICP or optimization frame. A close-up of the localizability detection in the tunnel is provided again in the top row of Fig. 11. The localizability categorization in certain parts of the tunnels shows the ability to distinguish the subtle differences at the bending of the tunnel point (B in Fig. 1), allowing the optimization to utilize the given information for better registration.

![](images/01428a9bc9eb53a3b5ec330e7ba80f5ea46ad6b712980604046178b69864b10d.jpg)

<details>
<summary>line</summary>

| Time[s] | η_v = Full | η_v = Partial | η_v = None |
| ------- | ---------- | ------------- | ---------- |
| 0       | v_t3       | v_t3          | v_t3       |
| 150     | v_t3       | v_t3          | v_t3       |
| 750     | v_t3       | v_t3          | v_t3       |
| 900     | v_t3       | v_t3          | v_t3       |
</details>

![](images/f5cceb332dd02883c91e807cb98e2d1c085c1dcfc71652ab814633155b3f6071.jpg)

<details>
<summary>line</summary>

| Time[s] | ην = Full | ην = Partial | ην = None |
| ------- | --------- | ------------ | --------- |
| 30      | 30        | 30           | 30        |
| 50      | 30        | 30           | 30        |
| 70      | 30        | 30           | 30        |
| 100     | 30        | 30           | 30        |
| 120     | 30        | 30           | 30        |
| 150     | 30        | 30           | 30        |
</details>

Hinduja et al. Localizability Categories

![](images/990cf3ae2320cfe64003c9012b4239b4c611dd6ca0de236bb070cf480ed40fc9.jpg)

<details>
<summary>line</summary>

| Time[s] | ην = Well-Constrained | ην = Degenerate |
| ------- | --------------------- | --------------- |
| 0       | Yes                   | No              |
| 150     | Yes                   | Yes             |
| 300     | Yes                   | Yes             |
| 600     | Yes                   | Yes             |
| 750     | Yes                   | Yes             |
| 900     | Yes                   | Yes             |
</details>

Zhang et al. Localizability Categories   
![](images/356b0b2d4541661a16c98e1737a04c8a93ed55a4fd29341276cae097b5e7a345.jpg)

<details>
<summary>scatter</summary>

| Time[s] | η_v = Well-Constrained | η_v = Degenerate |
| ------- | ---------------------- | ----------------- |
| 0       | 0                      | 0                 |
| 150     | 0                      | 150               |
| 750     | 0                      | 750               |
| 900     | 0                      | 900               |
</details>

Fig. 11. Localizability predictions of different methods for the Seemühle VLP-16 experiment. The localizability categories by the proposed X-ICP (top), the method from Hinduja et al. [17] (middle), and of Zhang et al. [12] (bottom) are shown.

The absolute pose error (APE), relative pose error (RPE) [71], and end position errors are calculated using the EVO evaluation package3. APE is a measure of the global accuracy of the estimated robot trajectory. Results for two different trajectory alignment methods are reported in Table I for the cases of either i) aligning the first 15 m (≈200 poses) of the trajectory, or ii) aligning the first pose of the ground truth and the estimated poses. In addition to APE, the end drift, whereas the proposed solution provides an accurate localization solution for real-world applications in challenging environments. The RPE error is measured in Table II to analyze local pose estimation accuracy and relative pose drift. The results demonstrate that the proposed method performs better than the state-of-the-art methods in local consistency.

![](images/e98dd04a1e54e736b581d44386a1d7db78d5fba834a66f4f58536625976cd730.jpg)

<details>
<summary>text_image</summary>

X-ICP
Hinduja et al.
Zhang et al.
</details>

X-ICP Localizability Categories   
![](images/b9920b510b28f564200087430672cacbaa595dd45d4eb233a05c2403114eaf0c.jpg)

<details>
<summary>line</summary>

| Time[s] | η_v = Full | η_v = Partial | η_v = None |
| ------- | ---------- | ------------- | ---------- |
| 0       | 100        | 100           | 100        |
| 100     | 100        | 100           | 100        |
| 200     | 100        | 100           | 100        |
| 300     | 100        | 100           | 100        |
| 400     | 100        | 100           | 100        |
| 500     | 100        | 100           | 100        |
| 600     | 100        | 100           | 100        |
| 700     | 100        | 100           | 100        |
| 800     | 100        | 100           | 100        |
| 900     | 100        | 100           | 100        |
| 1000    | 100        | 100           | 100        |
| 1100    | 100        | 100           | 100        |
| 1200    | 100        | 100           | 100        |
</details>

Hinduja et al. Localizability Categories   
![](images/13f404260ff9a8ddafbd6ff74aa42d1407a55f9d4afc48927c92636b13833f6f.jpg)

<details>
<summary>scatter</summary>

| Time[s] | η_v = Well-Constrained | η_v = Degenerate |
| ------- | ---------------------- | ----------------- |
| 0       | Yes                    | No                |
| 100     | Yes                    | Yes               |
| 200     | Yes                    | Yes               |
| 300     | Yes                    | Yes               |
| 400     | Yes                    | Yes               |
| 500     | Yes                    | Yes               |
| 600     | Yes                    | Yes               |
| 700     | Yes                    | Yes               |
| 800     | Yes                    | Yes               |
| 900     | Yes                    | Yes               |
| 1000    | Yes                    | Yes               |
| 1100    | Yes                    | Yes               |
| 1200    | Yes                    | Yes               |
</details>

Zhang et al. Localizability Categories   
![](images/66bd3b592f742556e7b0b97d7b6526e99bb131ea4c70fc9008cab665283dc4d4.jpg)

<details>
<summary>line</summary>

| Time[s] | η_v (Well-Constrained) | η_v (Degenerate) |
| ------- | ---------------------- | ---------------- |
| 0       | 100                    | 100              |
| 1000    | 100                    | 100              |
</details>

Fig. 12. Top Row: Point cloud maps of the three approaches for the Ouster OS0-128 Seemühle experiment. The color bar indicates the point-to-point distance error. Bottom Row: The estimated localizability categories of X-ICP and the state-of-the-art methods are shown.

TABLE II RPE PER 10 m TRAVERSED DISTANCE FOR SEEMÜHLE MINE EXPERIMENT WITH VLP-16 (BEST IN BOLD). 

<table><tr><td></td><td>Translation $\mu(\sigma)[m]$ </td><td>Rotation $\mu(\sigma)[\text{deg}]$ </td></tr><tr><td>X-ICP (Proposed)</td><td>0.17(0.12)</td><td>0.86(0.42)</td></tr><tr><td>Zhang et al. [12]</td><td>0.20(0.14)</td><td>0.93(0.51)</td></tr><tr><td>Hinduja et al. [17]</td><td>0.26(0.14)</td><td>1.28(0.74)</td></tr></table>

TABLE I APE ERROR FOR VLP-16 SEEMÜHLE EXPERIMENT (BEST IN BOLD). 

<table><tr><td rowspan="2"></td><td colspan="2">First 15 m Alignment</td><td colspan="3">Origin Alignment</td></tr><tr><td>Translation μ(σ)[m]</td><td>Rotation μ(σ)[deg]</td><td>Translation μ(σ)[m]</td><td>Rotation μ(σ)[deg]</td><td>Last Position Error[m]</td></tr><tr><td>X-ICP (Proposed)</td><td>2.05(1.23)</td><td>2.55(0.76)</td><td>2.45(1.35)</td><td>2.50(1.03)</td><td>0.27</td></tr><tr><td>Zhang et al. [12]</td><td>3.36(1.74)</td><td>4.06(1.37)</td><td>3.73(1.80)</td><td>4.11(1.52)</td><td>6.37</td></tr><tr><td>Hinduja et al. [17]</td><td>5.79(5.26)</td><td>7.67(4.72)</td><td>8.16(4.83)</td><td>8.03(4.73)</td><td>24.17</td></tr></table>

position error is also calculated as the difference between the last estimated robot position and the ground truth position. Both the APE metric and the end translation errors in Table I indicate that the compared state-of-the-art methods globally 2) Ouster OS0-128: To investigate the robustness and applicability of the proposed approach to different sensor setups, the experiment is repeated using an Ouster OS0-128 LiDAR with a much higher point density and larger FoV. To accommodate the increase in sensor noise, the filtering threshold $\kappa _ { f }$ is reduced to 60◦. As seen in the top row of Fig. 12, all three approaches benefit from this higher density in data and perform better than their performance with the VLP-16 LiDAR data. Despite the improvement, the solution from Hinduja et al. [17] still performs sub-optimally and generates a map with visible drift. On the other hand, Zhang et al. [12] can complete the trajectory without substantial drift. Moreover, the X-ICP solution shows less map point-to-point error in the inner parts of the cave. In addition, the localizability categories are shown at the bottom row of Fig. 12. Similar to the experiment with VLP-16, the degeneracy in the tunnel section of the environment is well-captured by X-ICP while the solution from Hinduja et al. [17] estimates localizability pessimistically. Interestingly, the method of Zhang et al. [12] fails to detect degeneracy in the tunnel section compared to the experiment with VLP-16. Previously, Nubert et al. [14] investigated and showed that the higher point density from Ouster OS0-128 LiDAR results in bigger eigenvalues; hence, the localizability threshold of Zhang et al. [12] requires heuristic tuning.

![](images/77f5d5ed126ce349272902f94cc8e51044ca9f76f7a4547c0a584923aedf7149.jpg)

<details>
<summary>text_image</summary>

X-ICP
Hinduja et al.
Zhang et al.
</details>

![](images/dfd02cefed10022812d280fcacce32df086c2db6d29d5777d2201c03633e26c5.jpg)

<details>
<summary>bar</summary>

| Time [s] | ην = Full | ην = Partial | ην = None |
| -------- | --------- | ------------ | --------- |
| 0        | 0         | 0            | 0         |
| 100      | 0         | 100          | 0         |
| 200      | 0         | 150          | 0         |
| 300      | 0         | 200          | 0         |
| 400      | 0         | 250          | 0         |
</details>

![](images/f7826ed1de442ed3f988b347e3340afffaf6c00b43639ea118c86a44ae5ce902.jpg)

<details>
<summary>bar</summary>

| Time [s] | Well-Constrained | Degenerate |
| -------- | ---------------- | ---------- |
| 0        | Yes              | No         |
| 100      | Yes              | No         |
| 200      | Yes              | Yes        |
| 300      | Yes              | Yes        |
| 400      | Yes              | Yes        |
| 500      | Yes              | Yes        |
</details>

![](images/3a11bd3a780ff52469216a1f3ebcf7b48db6ba560880fec58cd9e2d94b38cbb1.jpg)

<details>
<summary>bar</summary>

| Time [s] | Well-Constrained | Degenerate |
| -------- | ---------------- | ---------- |
| 0        | 0                | 0          |
| 50       | 0                | 0          |
| 100      | 0                | 0          |
| 150      | 0                | 0          |
| 200      | 0                | 0          |
| 250      | 0                | 0          |
| 300      | 0                | 30         |
| 350      | 0                | 0          |
| 400      | 0                | 0          |
| 450      | 0                | 0          |
| 500      | 0                | 0          |
</details>

Fig. 13. Top Row: Point cloud maps of X-ICP and the two state-of-the-art approaches for the Rümlang experiment. Bottom: The estimated localizability categories by X-ICP and the state-of-the-art methods are compared.

# E. Rümlang Construction Site

In the next experiment, the robot navigated in a largescale construction site in Rümlang, Switzerland, posing an under-constrained scenario for translation along the ground plane and rotation perpendicular to it. A picture taken in this open environment is shown in Fig. 6-b), demonstrating its structureless planarity. During the 153 m long test, the robot started next to a garage-like structure, traversing to the open area where it performed a couple of in-spot rotations before returning back. The in-spot rotations challenge the yaw estimation performance of the registration methods.

![](images/8afe5a82a55d8aca3c837cc40e753f8b2930f32bef98c8cb4511d974653582c2.jpg)

<details>
<summary>line</summary>

| Time[s] | Vt3    | Vt2    | Vt1    | Vr3    | Vr2    | Vr1    |
| ------- | ------ | ------ | ------ | ------ | ------ | ------ |
| 0       | 1.0e4  | 0.6e4  | 0.2e4  | 0.8e4  | 0.6e4  | 0.2e4  |
| 100     | 0.8e4  | 0.5e4  | 0.1e4  | 0.7e4  | 0.5e4  | 0.1e4  |
| 200     | 0.9e4  | 0.6e4  | 0.1e4  | 0.7e4  | 0.5e4  | 0.1e4  |
| 300     | 1.0e4  | 0.7e4  | 0.1e4  | 0.7e4  | 0.6e4  | 0.1e4  |
| 400     | 1.2e4  | 0.8e4  | 0.2e4  | 0.7e4  | 0.7e4  | 0.2e4  |
| 500     | 1.1e4  | 0.7e4  | 0.2e4  | 0.7e4  | 0.7e4  | 0.2e4  |
</details>

![](images/3b7879a95d716b3ed507af8f527fd922a761867d9fcb6906c110fb2c326c1228.jpg)

<details>
<summary>line</summary>

| Time[s] | v_t2  | v_t1  | v_r1  |
| ------- | ----- | ----- | ----- |
| 150     | 200   | 150   | 250   |
| 160     | 100   | 80    | 120   |
| 170     | 150   | 100   | 130   |
| 180     | 180   | 120   | 140   |
| 190     | 160   | 110   | 130   |
| 200     | 140   | 90    | 120   |
</details>

![](images/39353e3fcef1f609628e8db73cca99fd8a63567e6dd00d56b242327f8c0cad1f.jpg)

<details>
<summary>line</summary>

| Time[s] | v_t3    | v_t2    | v_t1    | v_r3    | v_r2    | v_r1    |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| 0       | ~0.9e4  | ~0.6e4  | ~0.2e4  | ~0.8e4  | ~0.4e4  | ~0.2e4  |
| 100     | ~0.8e4  | ~0.4e4  | ~0.1e4  | ~0.7e4  | ~0.3e4  | ~0.1e4  |
| 200     | ~0.8e4  | ~0.4e4  | ~0.1e4  | ~0.7e4  | ~0.3e4  | ~0.1e4  |
| 300     | ~0.8e4  | ~0.4e4  | ~0.1e4  | ~0.7e4  | ~0.3e4  | ~0.1e4  |
| 400     | ~0.8e4  | ~0.4e4  | ~0.1e4  | ~0.7e4  | ~0.3e4  | ~0.1e4  |
| 500     | ~1.2e4  | ~0.6e4  | ~0.2e4  | ~0.8e4  | ~0.5e4  | ~0.2e4  |
</details>

![](images/124ec31e67e8ce5e5efd9d75a381cb2b9ef9f7f65ba05432b9e12916cd4713b6.jpg)

<details>
<summary>line</summary>

| Time[s] | v_t2 | v_t1 | v_t3 |
| ------- | ---- | ---- | ---- |
| 150     | 100  | 100  | 100  |
| 160     | 50   | 50   | 50   |
| 170     | 50   | 50   | 50   |
| 180     | 50   | 50   | 50   |
| 190     | 50   | 50   | 50   |
| 200     | 50   | 50   | 50   |
</details>

Fig. 14. Top row: Evolution of $\mathcal { L } _ { c }$ throughout the Rümlang experiment. The shaded orange region (zoomed-in middle row) represents the active region of the localizability awareness. Bottom row: The difference between $\mathcal { L } _ { c }$ and $\mathcal { L } _ { s }$ is provided to illustrate the filtered information pairs.

The experimental results are presented in Fig. 13, with the top row showing point cloud maps generated using the three different methods and the bottom row showing the associated localizability category estimation of different methods. It can be seen that relying on the method by Zhang et al. [12] leads to poor performance due to its reliance on the eigenvalues of the optimization Hessian matrix. In particular, the detection of rotation and translation degeneracy with a single threshold is difficult, as the scale of eigenvalues between rotation and translation sub-spaces differ significantly. The method by Hinduja et al. [17] performs better in this highly degenerate environment due to its pessimistic localizability detection for pose estimation, it relies more on the legged odometry prior, which performs well on the flat concrete ground. Nevertheless, upon closer observation, map distortion in the form of blurry rigid structures can be noted due to errors accumulated from the drift in the pose prior. In contrast, the proposed method produces a consistent map of the environment preserving fine details, with its localizability detection performance shown in the second row of Fig. 13.

Finally, X-ICP not only correctly detects non-localizability along the three degenerate directions in the open area, but it also captures the smooth transition between partial and non-localizability. The smooth evolvement of the localizability categories can be better understood by observing the contribution changes provided in Fig. 14. The three degenerate directions are easily identifiable with their comparably low combined contribution value. In addition, inspecting the strong contribution plot reveals direction ${ \pmb v } _ { r _ { 1 } }$ to be mostly partially localizable. This is inferred through the strong contribution peaks that exceed $\kappa _ { 3 } = 3 5$ .

# F. Opfikon City Park

To evaluate the efficacy of the proposed method in natural environments, an experiment was conducted in an outdoor park at Opfikon, Switzerland. In this experiment, the robot in total traversed 235 m over soft terrain and a suspension bridge, both adversely affecting the quality of the legged odometry prior. In addition, the vegetation renders reliable surface-normal extraction difficult. Beyond that, towards the end of the experiment, the robot enters an open unstructured area near the center of the park, posing a LiDAR degenerate situation. The ground truth map of the environment, shown in Fig. 15, is collected using a Leica RTC 360 scanner (Fig. 6-c).

![](images/ae2a863f0ac85a1d293ca905edd2528e6a71d3e5a0049844f33888c4209d35b2.jpg)

<details>
<summary>text_image</summary>

A
Finish
Start
35m
B
A.2
A.1
</details>

Fig. 15. Ground truth point cloud map of the Opfikon City Park. A: The highlighted trees are later investigated in Figure 16 as a demonstration of the effectiveness of X-ICP. B: Satellite view of the Opfikon City Park.

Unlike the previous tests, the degeneracy detection threshold of Zhang et al. [12] is adapted to highlight the importance of heuristic tuning, as demonstrated in Fig. 17. A sub-optimal threshold degrades the quality of the map, and hence, in favor of [12], the threshold is tuned to be 200 (instead of 120).

A map error evaluation of the three methods is presented in Fig. 16. The magnified parts show the most accurate and fine-detailed map to be produced by the proposed method. Hinduja et al. [17] remains over-pessimistic and introduces significant drift in the presence of noisy odometry prior. Moreover, despite the tuned degeneracy detection threshold for Zhang et al. [12], the effect of LiDAR degeneracy is clearly visible in the tree section compared to the proposed method. This result concludes the evaluation of the proposed method across natural and man-made environments and emphasizes the advantage of the proposed (heuristic-free) localizabilityaware registration.

![](images/a834eaecf81c4c229e5f3305de7c55e68cc505b276b1db538abca8073816da42.jpg)

<details>
<summary>text_image</summary>

A.2
A.1
X-ICP
Zhang et al. Thr=200
Hinduja et al.
</details>

Fig. 16. Zoomed-in point cloud maps of the three investigated approaches for the Opfikon experiment (cf. Fig. 15). The error color scale is identical to Fig 17. A.1 and A.2 depict ground truth and real images for the tree region.

# G. Ablation Studies

The results discussed in the previous section validate the advantage of employing X-ICP in different LiDAR-challenging environments. To better understand the improvements, the effect of different components of the proposed method is investigated in an ablation study. In particular, a simplified version of X-ICP, referred to as Xs-ICP, is compared in view of fine-grained localizability categorization and the impact of performing the localizability analysis in every iteration of the ICP algorithm. In Xs-ICP, the partial-localizability category is discarded to reduce complexity, and the localizability parameters are set as $ \kappa _ { 1 } ~ \geq ~ \kappa _ { 2 } ~ = ~ \kappa _ { 3 } .$ , with values $\kappa _ { 1 } = 2 5 0$ and $\kappa _ { 2 } ~ = ~ 1 8 0$ . Without employing partial localizability, Xs-ICP performs localizability detection similar to the literature [12], [17] in a binary fashion. The decision tree of Xs-ICP is also simplified, with comparisons of only $\pmb { \mathcal { L } } _ { c } \geq \kappa _ { 1 }$ and $\pmb { { \mathcal { L } } } _ { s } \geq \kappa _ { 3 }$ . If either of these statements holds, the ICP optimization problem is well-constrained in the direction of the eigenvector. Finally, Xs-ICP performs the categorization step only in the first iteration of the ICP algorithm while in X-ICP, the categorization is performed at every iteration of the ICP algorithm.

![](images/b80ebd5f71806b61a37ba33ecb249234c498be5195b383281d713868d370d0c8.jpg)

<details>
<summary>text_image</summary>

Zhang et al.
Thr=80
Zhang et al.
Thr=120
Zhang et al.
Thr=200
>1m
0m
</details>

Fig. 17. Point cloud maps produced using the approach by Zhang et al. [12] for different eigenvalue thresholds for the Opfikon experiment. The color bar indicates the point-to-point distance error w.r.t. the ground truth map.

![](images/5ef3b450c75ee0d20e603ddd66f9d26d0504fdceb9b7ea6c503bc18278412943.jpg)

<details>
<summary>text_image</summary>

X-ICP
Xs-ICP
5m
0m
</details>

![](images/9d36813dfa8495b28cb86376b5d1801dba94c66b670d97df7632c4c8e6a79f4e.jpg)

<details>
<summary>line</summary>

| Time [s] | V_t1 (Ω_v = Full) | V_t1 (Ω_v = Partial) | V_t1 (Ω_v = None) |
| -------- | ----------------- | -------------------- | ----------------- |
| 0        | ~150              | ~150                 | ~150              |
| 150      | ~150              | ~150                 | ~150              |
| 300      | ~150              | ~150                 | ~150              |
| 450      | ~150              | ~150                 | ~150              |
| 600      | ~150              | ~150                 | ~150              |
| 750      | ~150              | ~150                 | ~150              |
| 900      | ~150              | ~150                 | ~150              |
</details>

Fig. 18. Top Row: Point cloud maps of the proposed X-ICP and Xs-ICP for the Seemühle experiment, using the VLP-16 LiDAR. The color bar indicates the point-to-point distance error. Bottom Row: Predicted localizability categories of X-ICP and Xs-ICP for the tunnel section .

1) Seemühle: The two versions of the proposed method are compared on the Seemühle experiment with VLP-16 LiDAR data. The results are shown in Fig. 18. An increase in the robot pose drift is apparent when partial-localizability is disabled in Xs-ICP, implying that using the three-level localizability detection can improve point cloud registration compared to two-level localizability detection. The localizability detection of the two variants is studied in the bottom plot of Fig. 18. In particular, when traversing the tunnel for the second time, the partial-localizability dominates the localizability category alongside the degenerate tunnel direction, while Xs-ICP estimates the localizability as non-localizable and fully relies on the prior for this eigenvector direction. Quantitative results of the RPE metric, shown in Table IV, indicate that Xs-ICP performs comparably for rotation estimation while X-ICP performs significantly better for translation estimation. On the other hand, the APE metric and the end translation errors in Table III indicate consistently better performance of X-ICP.   
2) Rümlang: A similar study is conducted for the Rümlang experiment, refer to Fig. 19. While the mapping performance of X-ICP and Xs-ICP is satisfactory, upon closer inspection, the point clouds registered for walls are duplicated

![](images/d68357bd4dc80f5aa10da183298451dedfecc285a2b3105874612b210e91aa07.jpg)

<details>
<summary>text_image</summary>

Xs-ICP
X-ICP
A
C
A
35m
</details>

![](images/1e7ba6af6529781222a553fd8942b5a0b56eba46204603cdb28e6c40093a6537.jpg)

<details>
<summary>bar</summary>

| Contribution Values | Count |
| ------------------- | ----- |
| 0.0                 | 150   |
| 0.2                 | 50    |
| 0.4                 | 20    |
| 0.6                 | 10    |
| 0.8                 | 5     |
| 1.0                 | 1000  |
| 0.2                 | 800   |
| 0.4                 | 300   |
| 0.6                 | 100   |
| 0.8                 | 50    |
| 1.0                 | 20    |
| 0.2                 | 900   |
| 0.4                 | 600   |
| 0.6                 | 300   |
| 0.8                 | 150   |
| 1.0                 | 50    |
</details>

![](images/07c718afc0ce0043c12910207eff7e3c98c2a55f2c1fb9316244809b3eca0152.jpg)

<details>
<summary>bar</summary>

| Contribution Values | Count |
| ------------------- | ----- |
| 0.0                 | 200   |
| 0.2                 | 150   |
| 0.4                 | 250   |
| 0.6                 | 300   |
| 0.8                 | 400   |
| 1.0                 | 900   |
</details>

Fig. 19. Top: Point cloud maps of X-ICP and Xs-ICP approaches for the Rümlang test site. A close-up of point A illustrates the detected degeneracy. Bottom: Contribution values are shown for point A as processed by X-ICP.

TABLE III APE ERROR FOR VLP-16 SEEMÜHLE EXPERIMENT (BEST IN BOLD). 

<table><tr><td rowspan="2"></td><td colspan="2">First 15 m Alignment</td><td colspan="3">Origin Alignment</td></tr><tr><td>Translation $\mu(\sigma)[m]$ </td><td>Rotation $\mu(\sigma)[deg]$ </td><td>Translation $\mu(\sigma)[m]$ </td><td>Rotation $\mu(\sigma)[deg]$ </td><td>Last PositionError[m]</td></tr><tr><td>X-ICP (Proposed)</td><td>2.05(1.23)</td><td>2.55(0.76)</td><td>2.45(1.35)</td><td>2.50(1.03)</td><td>0.27</td></tr><tr><td>Xs-ICP (Proposed)</td><td>2.29(1.22)</td><td>3.06(1.19)</td><td>2.68(1.26)</td><td>3.08(1.3)</td><td>5.34</td></tr></table>

TABLE IV RPE PER 10 m TRAVERSED DISTANCE FOR SEEMÜHLE MINE EXPERIMENT WITH VLP-16 (BEST IN BOLD). 

<table><tr><td></td><td>Translation $\mu(\sigma)[m]$ </td><td>Rotation $\mu(\sigma)[\text{deg}]$ </td></tr><tr><td>X-ICP (Proposed)</td><td>0.17(0.12)</td><td>0.86(0.42)</td></tr><tr><td>Xs-ICP (Proposed)</td><td>0.19(0.13)</td><td>0.85(0.47)</td></tr></table>

for the Xs-ICP method, indicating a slight misalignment in rotation. Consequently, the partial category improves localizability estimation and mapping performance also for this case. A snapshot of the localizability analysis at point A is shown in the lower part of Fig. 19, illustrating the detected partial-localizability for one of the directions. The histograms show three directions; two of these directions are shown as yellow arrows in the bottom right image, which are the non-localizable translation directions, while the blue arrow indicates the partially-localizable rotation direction, justified by the few but highly informative information pairs seen at the bottom right histogram.

3) Computational Time and Scalability: Computational time and scalability analyses are performed to evaluate realtime applicability and the computational overhead of the proposed localizability aware registration methods. The scalability analysis in Fig. 20 shows the overhead introduced by the localizability detection per ICP iteration, w.r.t. the number of correspondences of reading and reference point clouds. For X-ICP, the one- and three-axes degeneracy constitute the most common cases in real-world deployments (depicted in Fig 20). Since Xs-ICP does not require any re-sampling of points, the shown result represents all degeneracy conditions. As seen, both X-ICP and Xs-ICP methods are real-time capable for cases ranging from sparse 16-beam LiDARs to dense 128- beam LiDARs. By design, Xs-ICP requires less computation than X-ICP. Additionally, the real-time applicability of X-ICP and Xs-ICP is demonstrated by measuring the total scanto-map registration time in the Rümlang experiment on two computing platforms; one mobile and one desktop class system (Intel i7-9750H & i9-13900K) are used. All evaluations shown in Table V are performed in a single-threaded fashion for simplicity. The resulting computational analysis demonstrates the capability of X-ICP to run in real-time (µ = 32.19 ms) on mobile robotic systems such as ANYmal. Moreover, in the absence of LiDAR degeneracy, the computational burden of X-ICP and Xs-ICP is negligible compared to the baseline.

Computation Overhead Per Iteration vs Number of Matches   
![](images/e61624a8d298ca662bc7927a78541eac656bdc2ada86c4553a0e6f76e2f5901c.jpg)

<details>
<summary>line</summary>

| Number of Matches[pairs] | Xs-ICP | X-ICP 1-DoF Non-Loc | X-ICP 3-DoF Non-Loc |
| ------------------------ | ------ | ------------------- | ------------------- |
| 5000                     | 0.5    | 1.5                 | 2.5                 |
| 7500                     | 0.6    | 2.0                 | 3.0                 |
| 10000                    | 0.8    | 2.5                 | 3.5                 |
| 12500                    | 1.0    | 3.0                 | 4.0                 |
| 15000                    | 1.2    | 3.5                 | 4.5                 |
| 17500                    | 1.4    | 4.0                 | 5.0                 |
| 20000                    | 1.6    | 4.5                 | 5.5                 |
| 22500                    | 1.8    | 5.0                 | 6.0                 |
</details>

Fig. 20. Scalability analysis of X-ICP and Xs-ICP. The purple region ( and the light-orange region ( ) span the number of possible matches for the VLP-16 LiDAR and the Ouster OS0-128 LiDAR, respectively.

TABLE V THE SCAN-TO-MAP REGISTRATION TIME OF DIFFERENT REGISTRATION STRATEGIES FOR THE RÜMLANG DATASET. 

<table><tr><td></td><td>X-ICP $\mu (\sigma)$  [ms]</td><td>Xs-ICP $\mu (\sigma)$  [ms]</td><td>*Baseline $\mu (\sigma)$  [ms]</td></tr><tr><td>Intel i9-13900K</td><td>12.65 (3.51)</td><td>11.14 (3.1)</td><td>11.1 (1.45)</td></tr><tr><td>Intel i7-9750H</td><td>32.19 (10.7)</td><td>29.42 (9.35)</td><td>20.05 (2.21)</td></tr></table>

\* The baseline statistics are calculated until 150 s into the experiment to prevent the effect of degeneracy.

# VIII. CONCLUSIONS AND FUTURE WORK

This work presented X-ICP, a localizability-aware LiDAR point cloud registration method, to enable robust and reliable pose estimation in challenging LiDAR degenerate environments. The proposed approach detects environmental degeneracy and calculates additional constraints for the ICP optimization problem. Adding these constraints to the optimization prohibits pose updates along degenerate directions. Moreover, by introducing partial localizability, the proposed method benefits from sparse but valuable information often present in real-world scenarios. The efficacy of X-ICP is demonstrated through three real-world experiments and simulation-based analysis, all containing challenging environments causing ICP optimization to be ill-conditioned for accurate pose estimation. An ablation study further underlines the presented design choices. Currently, the proposed method is (similar to related methods) sensitive to the quality of the initial guess. In the future, this sensitivity of the point cloud registration method to initial guess quality should be reduced. Moreover, the sensordependent selection process of $\kappa _ { 1 }$ is planned to be improved by introducing point-wise confidence filter values to the pairs as weights to reduce the effect of noise on the contribution calculation. Finally, the proposed fine-grained localizability will be integrated into a graph-based degeneracy-aware sensor fusion framework [72] in the form of partial factors [17].

# REFERENCES

[1] R. Latif, K. Dahmane, and A. Saddik, “Slam algorithm: Overview and evaluation in a heterogeneous system,” Enabling Machine Learning Applications in Data Science, pp. 165–177, 2021.   
[2] C. Cadena et al., “Past, present, and future of simultaneous localization and mapping: Toward the robust-perception age,” IEEE Transactions on Robotics, vol. 32, 2016.   
[3] K. Ebadi et al., “Present and future of slam in extreme underground environments,” arXiv preprint arXiv:2208.01787, 2022.   
[4] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? the kitti vision benchmark suite,” in 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012, pp. 3354–3361.   
[5] M. Helmberger, K. Morin, B. Berner, N. Kumar, G. Cioffi, and D. Scaramuzza, “The hilti slam challenge dataset,” IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 7518–7525, 2022.   
[6] P. J. Besl and N. D. McKay, “Method for registration of 3-d shapes,” in Sensor fusion IV: control paradigms and data structures, vol. 1611. Spie, 1992, pp. 586–606.   
[7] K.-L. Low, “Linear least-squares optimization for point-to-plane icp surface registration,” Chapel Hill, University of North Carolina, vol. 4, no. 10, pp. 1–3, 2004.   
[8] I. Vizzo, T. Guadagnino, B. Mersch, L. Wiesmann, J. Behley, and C. Stachniss, “Kiss-icp: In defense of point-to-point icp–simple, accurate, and robust registration if done the right way,” arXiv preprint arXiv:2209.15397, 2022.   
[9] E. Jelavic, J. Nubert, and M. Hutter, “Open3d slam: Point cloud based mapping and localization for education,” in Robotic Perception and Mapping: Emerging Techniques, ICRA 2022 Workshop. ETH Zurich, Robotic Systems Lab, 2022, p. 24.   
[10] A. Censi, “An accurate closed-form estimate of icp’s covariance,” in Proceedings 2007 IEEE international conference on robotics and automation. IEEE, 2007, pp. 3167–3172.   
[11] M. Brossard, S. Bonnabel, and A. Barrau, “A new approach to 3d icp covariance estimation,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 744–751, 2020.   
[12] J. Zhang, M. Kaess, and S. Singh, “On degeneracy of optimization-based state estimation problems,” in 2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2016, pp. 809–816.   
[13] J. Nubert, S. Khattak, and M. Hutter, “Self-supervised learning of lidar odometry for robotic applications,” in 2021 IEEE International Conference on Robotics and Automation (ICRA), 2021, pp. 9601–9607.   
[14] J. Nubert, E. Walther, S. Khattak, and M. Hutter, “Learning-based localizability estimation for robust lidar localization,” in IEEE International Conference on Intelligent Robots and Systems (IROS), 2022.   
[15] W. Zhen, S. Zeng, and S. Soberer, “Robust localization and localizability estimation with a rotating laser scanner,” in IEEE International Conference on Robotics and Automation (ICRA), 2017.   
[16] W. Zhen and S. Scherer, “Estimating the localizability in tunnel-like environments using lidar and uwb,” in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 4903–4908.   
[17] A. Hinduja, B.-J. Ho, and M. Kaess, “Degeneracy-aware factors with applications to underwater slam,” in IEEE International Conference on Intelligent Robots and Systems (IROS), 2019.   
[18] P. Biber and W. Straßer, “The normal distributions transform: A new approach to laser scan matching,” in IEEE International Conference on Intelligent Robots and Systems (IROS), 2003.   
[19] A. Segal, D. Haehnel, and S. Thrun, “Generalized-icp.” in Robotics: science and systems, vol. 2, no. 4. Seattle, WA, 2009, p. 435.

[20] J. Zhang and S. Singh, “Loam: Lidar odometry and mapping in realtime.” in Robotics: Science and Systems, vol. 2, 2014.   
[21] J. Behley and C. Stachniss, “Efficient surfel-based slam using 3d laser range data in urban environments.” in Robotics: Science and Systems, vol. 2018, 2018, p. 59.   
[22] F. Pomerleau, “Applied registration for robotics: Methodology and tools for icp-like algorithms,” Ph.D. dissertation, ETH Zurich, 2013.   
[23] A. Censi, “An icp variant using a point-to-line metric,” in IEEE International Conference on Robotics and Automation, 2008.   
[24] P. Babin, P. Dandurand, V. Kubelka, P. Giguère, and F. Pomerleau, “Large-scale 3d mapping of subarctic forests,” in Field and Service Robotics. Springer, 2021, pp. 261–275.   
[25] S. Rusinkiewicz, “A symmetric objective function for icp,” ACM Transactions on Graphics (TOG), vol. 38, no. 4, pp. 1–7, 2019.   
[26] M. Yokozuka, K. Koide, S. Oishi, and A. Banno, “Litamin2: Ultra light lidar-based slam using geometric approximation applied with kldivergence,” in IEEE International Conference on Robotics and Automation (ICRA), 2021.   
[27] M. Ramezani, K. Khosoussi, G. Catt, P. Moghadam, J. Williams, P. Borges, F. Pauling, and N. Kottege, “Wildcat: Online continuous-time 3d lidar-inertial slam,” arXiv preprint arXiv:2205.12595, 2022.   
[28] S. Khattak et al., “Complementary multi–modal sensor fusion for resilient robot pose estimation in subterranean environments,” in IEEE International Conference on Unmanned Aircraft Systems (ICUAS), 2020.   
[29] K. Ebadi, M. Palieri, S. Wood, C. Padgett, and A.-a. Aghamohammadi, “Dare-slam: Degeneracy-aware and resilient loop closing in perceptually-degraded environments,” Journal of Intelligent & Robotic Systems, vol. 102, no. 1, pp. 1–25, 2021.   
[30] H. Yang, J. Shi, and L. Carlone, “Teaser: Fast and certifiable point cloud registration,” IEEE Transactions on Robotics, vol. 37, 2020.   
[31] H. Yang and L. Carlone, “A quaternion-based certifiably optimal solution to the wahba problem with outliers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 1665–1674.   
[32] L. Carlone, D. M. Rosen, G. Calafiore, J. J. Leonard, and F. Dellaert, “Lagrangian duality in 3d slam: Verification techniques and optimal solutions,” in IEEE International Conference on Intelligent Robots and Systems (IROS), 2015.   
[33] N. Chebrolu, T. Läbe, O. Vysotska, J. Behley, and C. Stachniss, “Adaptive robust kernels for non-linear least squares problems,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 2240–2247, 2021.   
[34] H. Yang, P. Antonante, V. Tzoumas, and L. Carlone, “Graduated nonconvexity for robust spatial perception: From non-minimal solvers to global outlier rejection,” IEEE Robotics and Automation Letters, 2020.   
[35] D. Landry, F. Pomerleau, and P. Giguere, “Cello-3d: Estimating the covariance of icp in the real world,” in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 8190–8196.   
[36] A. De Maio and S. Lacroix, “Deep bayesian icp covariance estimation,” arXiv preprint arXiv:2202.11607, 2022.   
[37] S. Bonnabel, M. Barczyk, and F. Goulette, “On the covariance of icpbased scan-matching techniques,” in 2016 American Control Conference (ACC). IEEE, 2016, pp. 5498–5503.   
[38] N. Gelfand, L. Ikemoto, S. Rusinkiewicz, and M. Levoy, “Geometrically stable sampling for the icp algorithm,” in IEEE International Conference on 3-D Digital Imaging and Modeling, 2003.   
[39] T.-H. Kwok and K. Tang, “Improvements to the iterative closest point algorithm for shape registration in manufacturing,” Journal of Manufacturing Science and Engineering, vol. 138, no. 1, 2016.   
[40] J.-E. Deschaud, “Imls-slam: Scan-to-model matching based on 3d data,” in IEEE International Conference on Robotics and Automation, 2018.   
[41] Z. Rong and N. Michael, “Detection and prediction of near-term state estimation degradation via online nonlinear observability analysis,” in IEEE International Symposium on Safety, Security, and Rescue Robotics (SSRR), 2016.   
[42] H. Cho, S. Yeon, H. Choi, and N. Doh, “Detection and compensation of degeneracy cases for imu-kinect integrated continuous slam with plane features,” Sensors, vol. 18, no. 4, p. 935, 2018.   
[43] A. Tagliabue et al., “Lion: Lidar-inertial observability-aware navigator for vision-denied environments,” in International Symposium on Experimental Robotics, 2020.   
[44] Y. Liu, J. Wang, and Y. Huang, “A localizability estimation method for mobile robots based on 3d point cloud feature,” in IEEE International Conference on Real-time Computing and Robotics (RCAR), 2021.   
[45] L. Dong, W. Chen, and J. Wang, “Efficient feature extraction and localizability based matching for lidar slam,” in IEEE International Conference on Robotics and Biomimetics (ROBIO), 2021.

[46] S. B. Nashed, J. J. Park, R. Webster, and J. W. Durham, “Robust rank deficient slam,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 6603–6608.   
[47] M. L. R. Arévalo, “On the uncertainty in active slam: representation, propagation and monotonicity,” Ph.D. dissertation, Universidad de Zaragoza, 2018.   
[48] J. Jiao, H. Ye, Y. Zhu, and M. Liu, “Robust odometry and mapping for multi-lidar systems with online extrinsic calibration,” IEEE Transactions on Robotics, 2021.   
[49] E. Westman and M. Kaess, “Degeneracy-aware imaging sonar simultaneous localization and mapping,” IEEE Journal of Oceanic Engineering, vol. 45, no. 4, pp. 1280–1294, 2019.   
[50] R. Ren, H. Fu, H. Xue, Z. Sun, K. Ding, and P. Wang, “Towards a fully automated 3d reconstruction system based on lidar and gnss in challenging scenarios,” Remote Sensing, vol. 13, no. 10, p. 1981, 2021.   
[51] H. Zhou, Z. Yao, and M. Lu, “Lidar/uwb fusion based slam with antidegeneration capability,” IEEE Transactions on Vehicular Technology, vol. 70, no. 1, pp. 820–830, 2020.   
[52] S. Nobili, G. Tinchev, and M. Fallon, “Predicting alignment risk to prevent localization failure,” in 2018 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018, pp. 1003–1010.   
[53] X. Chen et al., “Overlapnet: A siamese network for computing lidar scan similarity with applications to loop closing and localization,” Autonomous Robots, 2022.   
[54] Y. Gao, S. Q. Wang, J. H. Li, M. Q. Hu, H. Y. Xia, H. Hu, and L. J. Wang, “A prediction method of localizability based on deep learning,” IEEE Access, vol. 8, pp. 110 103–110 115, 2020.   
[55] S. Floery, “Constrained matching of point clouds and surfaces (ph. d. thesis),” Technische Universitt Wien, 2010.   
[56] C. Olsson and A. Eriksson, “Solving quadratically constrained geometrical problems using lagrangian duality,” in 2008 19th International Conference on Pattern Recognition. IEEE, 2008, pp. 1–5.   
[57] S. Bouaziz, A. Tagliasacchi, and M. Pauly, “Sparse iterative closest point,” in Computer graphics forum, vol. 32, no. 5. Wiley Online Library, 2013, pp. 113–123.   
[58] P. Dellenbach, J.-E. Deschaud, B. Jacquet, and F. Goulette, “Ct-icp: Real-time elastic lidar odometry with loop closure,” in IEEE International Conference on Robotics and Automation (ICRA), 2022.   
[59] M. E. Wall, A. Rechtsteiner, and L. M. Rocha, “Singular value decomposition and principal component analysis,” in A practical approach to microarray data analysis. Springer, 2003, pp. 91–109.   
[60] F. Pomerleau, F. Colas, R. Siegwart, et al., “A review of point cloud registration algorithms for mobile robotics,” Foundations and Trends® in Robotics, vol. 4, no. 1, pp. 1–104, 2015.   
[61] Y. Nava et al., “Lidar-inertial odometry,” ANYbotics,” Report, 2022. [Online]. Available: http://hdl.handle.net/20.500.11850/580200   
[62] F. Pomerleau, F. Colas, R. Siegwart, and S. Magnenat, “Comparing icp variants on real-world data sets,” Autonomous Robots, vol. 34, 2013.   
[63] T.-H. Kwok, “Dnss: Dual-normal-space sampling for 3-d icp registration,” IEEE Transactions on Automation Science and Engineering, vol. 16, no. 1, pp. 241–252, 2018.   
[64] C. Fu, X. Jiao, and T. Yang, “Efficient sparse lu factorization with partial pivoting on distributed memory architectures,” IEEE Transactions on Parallel and Distributed Systems, vol. 9, no. 2, pp. 109–125, 1998.   
[65] M. Benzi and M. Tuma, “A robust incomplete factorization precondi-˘ tioner for positive definite matrices,” Numerical Linear Algebra with Applications, vol. 10, no. 5-6, pp. 385–400, 2003.   
[66] D. P. Bertsekas, Constrained optimization and Lagrange multiplier methods. Academic press, 2014.   
[67] A. Biasion, T. Moerwald, B. Walser, and G. Walsh, “A new approach to the terrestrial laser scanner workflow: the rtc360 solution,” Geospatial Information for a Smarter Life and Environmental Resilience, 2019.   
[68] M. Hutter et al., “Anymal-toward legged robots for harsh environments,” Advanced Robotics, vol. 31, 2017.   
[69] M. Bloesch, M. Burri, H. Sommer, R. Siegwart, and M. Hutter, “The two-state implicit filter recursive estimation for mobile robots,” IEEE Robotics and Automation Letters, vol. 3, no. 1, pp. 573–580, 2017.   
[70] M. Ramezani et al., “The newer college dataset: Handheld lidar, inertial and vision with ground truth,” in IEEE International Conference on Intelligent Robots and Systems (IROS), 2020.   
[71] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of rgb-d slam systems,” in IEEE international conference on intelligent robots and systems, 2012.   
[72] J. Nubert, S. Khattak, and M. Hutter, “Graph-based multi-sensor fusion for consistent localization of autonomous construction robots,” in International Conference on Robotics and Automation (ICRA), 2022.

![](images/3fad21a5248024cc11652ce04c9c21665dffb33c55363e1409b9dfe8b672fc1c.jpg)

<details>
<summary>natural_image</summary>

Portrait of a smiling man wearing glasses and a collared shirt (no text or symbols visible)
</details>

Turcan Tuna is a PhD student in the Robotic Systems Lab at ETH Zurich. He received his M.Sc. in Robotics, Systems & Control in 2022 from ETH Zurich. Previously, he completed a double major, B.Sc in Mechanical Engineering and Control & Automation Engineering, at Istanbul Technical University. He graduated from both of his B.Sc majors with distinction. His research interests include developing and deploying robust localization, perception, and mapping frameworks on robotic systems.

![](images/c255a6fb169b82c96692e0250b78a750d8e086f4b1560045ae1a6ca2708de32d.jpg)

<details>
<summary>natural_image</summary>

Portrait of a smiling man in a light blue shirt against a plain background (no text or symbols visible)
</details>

Julian Nubert is a PhD student in the Robotic Systems Lab at ETH Zurich. He received his M.Sc. in Robotics, Systems & Control in 2020 from ETH Zurich. He is also affiliated with the Max Planck Institute through the MPI ETH Center for Learning Systems. His research interests lie in the field of robust robot perception, and how it can be used for the deployment of mobile robotic systems. Julian received the ETH silver medal and was awarded the Willi-Studer-Price for his accomplishments during his master studies.

![](images/51ba4117b33d1da3cbc499da546ca9ec22204039c2471c18b91dc8dee4a975ff.jpg)

<details>
<summary>natural_image</summary>

Portrait of a bald man with glasses and beard wearing a black shirt (no text or symbols visible)
</details>

Yoshua Nava is a Perception Software Engineer at ANYbotics AG (Zurich, Switzerland), where he specializes in point cloud-based localization and mapping. He received his masters degree in Systems, Control and Robotics from KTH Royal Institute of Technology (Stockholm, Sweden), and his bachelor’s degree from Universidad Católica Andrés Bello (Caracas, Venezuela). His research interests broadly cover localization and mapping as a core skill for robot mobility, and as a way to increase situational awareness for robots and human operators.

![](images/3d10d22017ef8ae6f267566d868213307c7502ae4f232f1bd0babae57b9cc8fc.jpg)

<details>
<summary>natural_image</summary>

Portrait of a smiling man wearing a checkered shirt (no text or symbols visible)
</details>

Shehryar Khattak is a currently a Robotics Technologist at the NASA - Jet Propulsion Lab. Previously, he was a post-doctoral researcher at the Robotics Systems Lab at ETH Zurich and the perception lead of team CERBERUS, which won the DARPA SubT Challenge (2021). He received his Ph.D. (2019) and MS (2017) in Computer Science from the University of Nevada, Reno. He also holds an MS in Aerospace Engineering from KAIST (2012) and a BS in Mechanical Engineering from GIKI (2009). His research focuses on developing

perception algorithms to support real-time localization and mapping for field robotics applications.

![](images/2016410ac025a3ad3817933e50d04e2b6ec97afd03e1f31fe5aab55ab873ce9e.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man in a suit and white shirt (no text or symbols visible)
</details>

Marco Hutter is Associate Professor for Robotic Systems at ETH Zurich. He received his M.Sc. and PhD from ETH Zurich in 2009 and 2013 in the field of design, actuation, and control of legged robots. His research interests are in the development of novel machines and actuation concepts together with the underlying control, planning, and machine learning algorithms for locomotion and manipulation. Marco is the recipient of an ERC Starting Grant, PI of the NCCRs robotics and digital fabrication, PI in various EU projects and international challenges,

a co-founder of several ETH Startups such as ANYbotics AG.