# FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter

Wei Xu1, Fu Zhang1

Abstract— This paper presents a computationally efficient and robust LiDAR-inertial odometry framework. We fuse LiDAR feature points with IMU data using a tightly-coupled iterated extended Kalman filter to allow robust navigation in fastmotion, noisy or cluttered environments where degeneration occurs. To lower the computation load in the presence of a large number of measurements, we present a new formula to compute the Kalman gain. The new formula has computation load depending on the state dimension instead of the measurement dimension. The proposed method and its implementation are tested in various indoor and outdoor environments. In all tests, our method produces reliable navigation results in realtime: running on a quadrotor onboard computer, it fuses more than 1,200 effective feature points in a scan and completes all iterations of an iEKF step within 25 ms. Our codes are opensourced on Github2.

## I. INTRODUCTION

Simultaneous localization and mapping (SLAM) is a fundamental prerequisite of mobile robots, such as unmanned aerial vehicles (UAVs). Visual (-inertial) odometry (VO), such as Stereo VO [1] and Monocular VO [2, 3] are commonly used on mobile robots due to their lightweight and low-cost. Although providing rich RGB information, visual solutions lack direct depth measurements and require much computation resources to reconstruct the 3D environment for trajectory planning. Moreover, they are very sensitive to lighting conditions. Light detection and ranging (LiDAR) sensors could overcome all these difficulties but have been too costly (and bulky) for small-scale mobile robots.

Solid-state LiDARs recently emerge as main trends in LiDAR developments, such as those based on microelectro-mechanical-system (MEMS) scanning [4] and rotating prisms [5]. These LiDARs are very cost-effective (in a cost range similar to global shutter cameras), lightweight (can be carried by a small-scale UAV), and of high performance (producing active and direct 3D measurements of long-range and high-accuracy). These features make such LiDARs viable for UAVs, especially industrial UAVs, which need to acquire accurate 3D maps of the environments (e.g., aerial mapping) or may operate in cluttered environments with severe illumination variations (e.g., post-disaster search and inspection).

Despite the great potentiality, solid-state LiDARs bring new challenges to SLAM: 1) the feature points in LiDAR measurements are usually the geometrical structures (e.g., edges and planes) in the environments. When the UAV is operating in cluttered environments where no strong features are present, the LiDAR-based solution easily degenerates. This problem is more obvious when the LiDAR has a small FoV. 2) Due to the high-resolution along the scanning direction, a LiDAR scan usually contains many feature points (e.g., a few thousand). While these feature points are not adequate to reliably determine the pose in case of degeneration, tightly fusing such a large number of feature points to IMU measurements requires tremendous computational resources that are not affordable by the UAV onboard computer. 3) Since the LiDAR samples point sequentially with a few laser/receiver pairs, laser points in a scan are always sampled at different times, resulting in motion distortion that will significantly degrade a scan registration [6]. The constant rotations of UAV propellers and motors also introduce significant noises to the IMU measurements.

![](images/ac57322163adf8acfa16cf8e7b9a99a36ca38e9c9396af6ce067ad92c71be666.jpg)

<details>
<summary>text_image</summary>

Camera
LiDAR (Livox AVIA)
Onboard Computer
(DJI Manifold 2-C)
Airframe
(280mm WheelBase)
</details>

Fig. 1. Our LiDAR-inertial navigation system runs on a Livox AVIA LiDAR3 and a DJI Manifold 2-C onboard computer4, all on a customized small-scale quadrotor UAV (280 mm wheelbase). The RGB camera is not used in our algorithm, but only for visualization. Video is available at https://youtu.be/iYCY6T79oNU

To make the LiDAR navigation viable for small-scale mobile robots such as UAVs, we propose the FAST-LIO, a computationally efficient and robust LiDAR-inertial odometry package. More specifically, our contributions are as follows: 1) To cope with fast-motion, noisy or cluttered environments where degeneration occurs, we adopt a tightly-coupled iterated Kalman filter to fuse LiDAR feature points with IMU measurements. We propose a formal back-propagation process to compensate for the motion distortion; 2) To lower the computation load caused by a large number of LiDAR feature points, we propose a new formula for computing the Kalman gain and prove its equivalence to the conventional Kalman gain formula. The new formula has a computation complexity depending on the state dimension instead of the measurement dimension. 3) We implement our formulations into a fast and robust LiDAR-inertial odometry software package. The system is able to run on a small-scale quadrotor onboard computer. 4) We conduct experiments in various indoor and outdoor environments and with actual UAV flight tests (Fig. 1) to validate the system’s robustness when fast motion or intense vibration noise exists.

The remaining paper is organized as follows: In Section. II, we discuss relevant research works. We give an overview of the complete system pipeline and the details of each key component in Section. III. The experiments are presented in Section. IV, followed by conclusions in Section. V.

## II. RELATED WORKS

Existing works on LiDAR SLAM are extensive. Here we limit our review to the most relevant works: LiDAR-only odometry and mapping, loosely-coupled and tightly-coupled LiDAR-Inertial fusion methods.

## A. LiDAR Odometry and Mapping

Besl et al. [6] propose an iterated closest points (ICP) method for scan registration, which builds the basis for LiDAR odometry. ICP performs well for dense 3D scans. However, for sparse point clouds of LiDAR measurements, the exact point matching required by ICP rarely exists. To cope with this problem, Segal et al. [7] propose a generalized-ICP based on the point-to-plane distance. Then Zhang et al. [8] combine this ICP method with a pointto-edge distance and developed a LiDAR odometry and mapping (LOAM) framework. Thereafter, many variants of LOAM have been developed, such as LeGO-LOAM [9] and LOAM-Livox [10]. While these methods work well for structured environments and LiDARs of large FoV, they are very vulnerable to featureless environments or small FoV LiDARs [10].

## B. Loosely-coupled LiDAR-Inertial Odometry

IMU measurements are commonly used to mitigate the problem of LiDAR degeneration in featureless environments. Loosely-coupled LiDAR-inertial odometry (LIO) methods typically process the LiDAR and IMU measurements separately and fuse their results later. For example, IMU-aided LOAM [8] takes the pose integrated from IMU measurements as the initial estimate for LiDAR scan registration. Zhen et al. [11] fuse the IMU measurements and the Gaussian Particle Filter output of LiDAR measurements using the error-state EKF. Balazadegan et al [12] add the IMU-gravity model to estimate the 6-DOF ego-motion to aid the LiDAR scan registration. Zuo et al. [13] use a Multi-State Constraint Kalman Filter (MSCKF) to fuse the scan registration results with IMU and visual measurements. A common procedure of the loosely-coupled approach is obtaining a pose measurement by registering a new scan and then fusing the pose measurement with IMU measurements. The separation between scan registration and data fusion reduces the computation load. However, it ignores the correlation between the system’s other states (e.g., velocity) and the pose of the new scan. Moreover, in the case of featureless environments, the scan registration could degenerate in certain directions and causes unreliable fusion in later stages.

## C. Tightly-coupled LiDAR-Inertial Odometry

Unlike the loosely-coupled methods, tightly-coupled LiDAR-inertial odometry methods typically fuse the raw feature points (instead of scan registration results) of LiDAR with IMU data. There are two main approaches to tightlycoupled LIO: optimization-based and filter-based. Geneva et al. [14] use a graph optimization with IMU pre-integration constrains [15] and plane constraints [16] from LiDAR feature points. Recently, Ye et al. [17] propose the LIOM package which uses a similar graph optimization but is based on edge and plane features. For filter-based methods, Bry [18] uses a Gaussian Particle Filter (GPF) to fuse the data of IMU and a planar 2D LiDAR. This method has also been used in the Boston Dynamics Atlas humanoid robot. Since the computation complexity of particle filter grows quickly with the number of feature points and the system dimension, Kalman filter and its variants are usually more preferred, such as extended Kalman filter [19], unscented Kalman filter [20], and iterated Kalman filter [21].

Our method falls into the tightly-coupled approach. We adopt an iterated extended Kalman filter similar to [21] to mitigate linearization errors. Kalman filter (and its variants) has a time complexity O  m2 where m is the measurement dimension [22], this may lead to remarkably high computation load when dealing with a large number of LiDAR measurements. Naive down-sampling would reduce the number of measurements but at the cost of information loss. [21] reduces the number of measurements by extracting and fitting ground planes similar to [9]. This, however, does not apply to aerial applications where the ground plane may not always present.

## III. METHODOLOGY

## A. Framework Overview

This paper will use the notations shown in Table I. The overview of our workflow is shown in Fig. 2. The LiDAR inputs are fed into the feature extraction module to obtain planar and edge features. Then the extracted features and IMU measurements are fed into our state estimation module for state estimation at 10Hz−50Hz. The estimated pose then registers the feature points to the global frame and merges them with the feature points map built so far. The updated map is finally used to register further new points in the next step.

## B. System Description

1)  /  operator:

![](images/bb0017b35f49cf5c386799fa7b9d9a46977f40acf48df41aa5e91bc0ced3225b.jpg)  
Fig. 2. System overview of FAST-LIO. (a): the overall pipeline; (b): the forward and backward propagation.

TABLE I SOME IMPORTANT NOTATIONS

<table><tr><td>Symbols</td><td>Meaning</td></tr><tr><td> $t_{k}$ </td><td>The scan-end time of the  $k$ -th LiDAR scan.</td></tr><tr><td> $\tau_{i}$ </td><td>The  $i$ -th IMU sample time in a LiDAR scan.</td></tr><tr><td> $\rho_{j}$ </td><td>The  $j$ -th feature point&#x27;s sample time in a LiDAR scan.</td></tr><tr><td> $I_{i}, I_{j}, I_{k}$ </td><td>The IMU body frame at the time  $\tau_{i}, \rho_{j}$  and  $t_{k}$ .</td></tr><tr><td> $L_{j}, L_{k}$ </td><td>The LiDAR body frame at the time  $\rho_{j}$  and  $t_{k}$ .</td></tr><tr><td> $\mathbf{x}, \widehat{\mathbf{x}}, \bar{\mathbf{x}}$ </td><td>The ground-true, propagated, and updated value of  $\mathbf{x}$ .</td></tr><tr><td> $\widetilde{\mathbf{x}}$ </td><td>The error between ground-true  $\mathbf{x}$  and its estimation  $\bar{\mathbf{x}}$ .</td></tr><tr><td> $\widehat{\mathbf{x}}^{\kappa}$ </td><td>The  $\kappa$ -th update of  $\mathbf{x}$  in the iterated Kalman filter.</td></tr><tr><td> $\mathbf{x}_{i}, \mathbf{x}_{j}, \mathbf{x}_{k}$ </td><td>The vector (e.g.,state)  $\mathbf{x}$  at time  $\tau_{i}, \rho_{j}$  and  $t_{k}$ .</td></tr><tr><td> $\check{\mathbf{x}}_{j}$ </td><td>Estimate of  $\mathbf{x}_{j}$  relative to  $\mathbf{x}_{k}$  in the back propagation.</td></tr></table>

Let M be the manifold of dimension n in consideration $( \mathbf { e . g . } , \mathcal { M } = S O ( 3 ) )$ . Since manifolds are locally homeomorphic to $\mathbb { R } ^ { n }$ , we can establish a bijective mapping from a local neighborhood on $\mathcal { M }$ to its tangent space $\mathbb { R } ^ { n }$ via two encapsulation operators  and  [23]:

$$
\boxplus : \mathcal {M} \times \mathbb {R} ^ {n} \rightarrow \mathcal {M}; \quad \boxminus : \mathcal {M} \times \mathcal {M} \rightarrow \mathbb {R} ^ {n}
$$

$$
\mathcal {M} = S O (3): \mathbf {R} \boxplus \mathbf {r} = \mathbf {R} \mathrm{Exp} (\mathbf {r}); \quad \mathbf {R} _ {1} \boxminus \mathbf {R} _ {2} = \mathrm{Log} (\mathbf {R} _ {2} ^ {\mathrm{T}} \mathbf {R} _ {1})
$$

$$
\mathcal {M} = \mathbb {R} ^ {n}: \quad \mathbf {a} \boxplus \mathbf {b} = \mathbf {a} + \mathbf {b}; \quad \mathbf {a} \boxminus \mathbf {b} = \mathbf {a} - \mathbf {b}
$$

where Exp $\scriptstyle ( \mathbf { r } ) = \mathbf { I } + { \frac { \mathbf { r } } { \| \mathbf { r } \| } }$ sin $\begin{array} { r } { ( \| \mathbf { r } \| ) + \frac { \mathbf { r } ^ { 2 } } { \| \mathbf { r } \| ^ { 2 } } \left( 1 - \cos \left( \| \mathbf { r } \| \right) \right) } \end{array}$ is the exponential map [23] and Log(·) is its inverse map. For a compound manifold $\mathcal { M } = S O ( 3 ) \times \mathbb { R } ^ { n }$ we have:

$$
\left[ \begin{array}{c} \mathbf {R} \\ \mathbf {a} \end{array} \right] \boxplus \left[ \begin{array}{c} \mathbf {r} \\ \mathbf {b} \end{array} \right] = \left[ \begin{array}{c} \mathbf {R} \boxplus \mathbf {r} \\ \mathbf {a} + \mathbf {b} \end{array} \right]; \quad \left[ \begin{array}{c} \mathbf {R} _ {1} \\ \mathbf {a} \end{array} \right] \boxminus \left[ \begin{array}{c} \mathbf {R} _ {2} \\ \mathbf {b} \end{array} \right] = \left[ \begin{array}{c} \mathbf {R} _ {1} \boxminus \mathbf {R} _ {2} \\ \mathbf {a} - \mathbf {b} \end{array} \right]
$$

From the above definition, it is easy to verify that

$$
(\mathbf {x} \boxplus \mathbf {u}) \boxminus \mathbf {x} = \mathbf {u}; \mathbf {x} \boxplus (\mathbf {y} \boxminus \mathbf {x}) = \mathbf {y}; \forall \mathbf {x}, \mathbf {y} \in \mathcal {M}, \forall \mathbf {u} \in \mathbb {R} ^ {n}.
$$

## 2) Continuous model:

Assuming an IMU is rigidly attached to the LiDAR with a known extrinsic ${ ^ I } { \bf T } _ { L } \ = \ ( { ^ { \bar { I } } { \bf R } _ { L } } , \ { ^ I } { \bf p } _ { L } )$ . Taking the IMU frame (denoted as I) as the body frame of reference leads to a kinematic model:

$$
\begin{array}{l} { } ^ { G } \dot { \mathbf { p } } _ { I } = { } ^ { G } \mathbf { v } _ { I } , \quad { } ^ { G } \dot { \mathbf { v } } _ { I } = { } ^ { G } \mathbf { R } _ { I } \left( \mathbf { a } _ { m } - \mathbf { b } _ { \mathbf { a } } - \mathbf { n } _ { \mathbf { a } } \right) + { } ^ { G } \mathbf { g } , \quad { } ^ { G } \dot { \mathbf { g } } = \mathbf { 0 } \\ { } ^ { G } \dot { \mathbf { R } } _ { I } = { } ^ { G } \mathbf { R } _ { I } \left\lfloor \boldsymbol { \omega } _ { m } - \mathbf { b } _ { \omega } - \mathbf { n } _ { \omega } \right\rfloor _ { \wedge } , \dot { \mathbf { b } } _ { \omega } = \mathbf { n } _ { \mathrm{b} \omega } , \dot { \mathbf { b } } _ { \mathrm{a} } = \mathbf { n } _ { \mathrm{ba} } \tag {1} \\ \end{array}
$$

where ${ { \bf \Lambda } ^ { G } } { \bf p } _ { I } , \ { { \bf \Lambda } ^ { G } } { \bf R } _ { I }$ are the position and attitude of IMU in the global frame (i.e., the first IMU frame, denoted as G),

$G _ { \mathbf { g } }$ is the unknown gravity vector in the global frame, $\mathbf { a } _ { m }$ and $\omega _ { m }$ are IMU measurements, $\mathbf { n _ { a } }$ and $\mathbf { n } _ { \omega }$ are the white noise of IMU measurements, $\mathbf { b _ { a } }$ and $\mathbf { b } _ { \omega }$ are the IMU bias modelled as the random walk process with Gaussian noises $\mathbf { n _ { b a } }$ and $\mathbf { n } _ { \mathbf { b } \omega } ,$ , and the notation $\lfloor \mathbf { a } \rfloor _ { \wedge }$ denotes the skewsymmetric matrix of vector $\textbf { a } \in \ \mathbb { R } ^ { 3 }$ that maps the cross product operation.

## 3) Discrete model:

Based on the  operation defined above, we can discretize the continuous model in (1) at the IMU sampling period ∆t using a zero-order holder. The resultant discrete model is

$$
\mathbf {x} _ {i + 1} = \mathbf {x} _ {i} \boxplus (\Delta t \mathbf {f} (\mathbf {x} _ {i}, \mathbf {u} _ {i}, \mathbf {w} _ {i})) \tag {2}
$$

where i is the index of IMU measurements, the function f, state x, input u and noise w are defined as below:

$$
\mathcal {M} = S O (3) \times \mathbb {R} ^ {1 5}, \dim (\mathcal {M}) = 1 8
$$

$$
\mathbf {x} \doteq \left[ ^ {G} \mathbf {R} _ {I} ^ {T} \quad^ {G} \mathbf {p} _ {I} ^ {T} \quad^ {G} \mathbf {v} _ {I} ^ {T} \quad \mathbf {b} _ {\omega} ^ {T} \quad \mathbf {b} _ {\mathbf {a}} ^ {T} \quad^ {G} \mathbf {g} ^ {T} \right] ^ {T} \in \mathcal {M}
$$

$$
\mathbf {u} \doteq \left[ \begin{array}{c c} \boldsymbol {\omega} _ {m} ^ {T} & \mathbf {a} _ {m} ^ {T} \end{array} \right] ^ {T}, \mathbf {w} \doteq \left[ \begin{array}{c c c c} \mathbf {n} _ {\omega} ^ {T} & \mathbf {n} _ {\mathbf {a}} ^ {T} & \mathbf {n} _ {\mathbf {b} \omega} ^ {T} & \mathbf {n} _ {\mathbf {b a}} ^ {T} \end{array} \right] ^ {T}
$$

$$
\mathbf {f} \left(\mathbf {x} _ {i}, \mathbf {u} _ {i}, \mathbf {w} _ {i}\right) = \left[ \begin{array}{c} \boldsymbol {\omega} _ {m _ {i}} - \mathbf {b} _ {\boldsymbol {\omega} _ {i}} - \mathbf {n} _ {\boldsymbol {\omega} _ {i}} \\ ^ G \mathbf {v} _ {I _ {i}} \\ ^ G \mathbf {R} _ {I _ {i}} \left(\mathbf {a} _ {m _ {i}} - \mathbf {b} _ {\mathbf {a} _ {i}} - \mathbf {n} _ {\mathbf {a} _ {i}}\right) + ^ {G} \mathbf {g} _ {i} \\ \mathbf {n} _ {\mathbf {b} \boldsymbol {\omega} _ {i}} \\ \mathbf {n} _ {\mathbf {b a} _ {i}} \\ \mathbf {0} _ {3 \times 1} \end{array} \right] \tag {3}
$$

## 4) Preprocessing of LiDAR measurements:

LiDAR measurements are point coordinates in its local body frame. Since the raw LiDAR points are sampled at a very high rate (e.g., 200kHz), it is usually not possible to process each new point once being received. A more practical approach is to accumulate these points for a certain time and process them all at once. In FAST-LIO, the minimum accumulation interval is set to 20 ms, leading to up to 50 $H z$ full state estimation (i.e., odometry output) and map update as shown in Fig. 2 (a). Such an accumulated set of points is called a scan, and the time for processing it is denoted as $t _ { k }$ (see Fig. 2 (b)). From the raw points, we extract planar points with high local smoothness [8] and edge points with low local smoothness as in [10]. Assume the number of feature points is $m ,$ each is sampled at time $\rho _ { j } \in ( t _ { k - 1 } , t _ { k } ]$ and is denoted as ${ L _ { j } } _ { \mathbf { p } _ { f _ { j } } }$ , where $L _ { j }$ is the LiDAR local frame at the time $\rho _ { j }$ . During a LiDAR scan, there are also multiple IMU

$$
\mathbf {F} _ {\widetilde {\mathbf {x}}} = \left[ \begin{array}{c c c c c c} \operatorname{Exp} (- \widehat {\boldsymbol {\omega}} _ {i} \Delta t) & \mathbf {0} & \mathbf {0} & - \mathbf {A} (\widehat {\boldsymbol {\omega}} _ {i} \Delta t) ^ {T} \Delta t & \mathbf {0} & \mathbf {0} \\ \mathbf {0} & \mathbf {I} & \mathbf {I} \Delta t & \mathbf {0} & \mathbf {0} & \mathbf {0} \\ - ^ {G} \widehat {\mathbf {R}} _ {I _ {i}} \lfloor \widehat {\mathbf {a}} _ {i} \rfloor_ {\wedge} \Delta t & \mathbf {0} & \mathbf {I} & \mathbf {0} & - ^ {G} \widehat {\mathbf {R}} _ {I _ {i}} \Delta t & \mathbf {I} \Delta t \\ \mathbf {0} & \mathbf {0} & \mathbf {0} & \mathbf {I} & \mathbf {0} & \mathbf {0} \\ \mathbf {0} & \mathbf {0} & \mathbf {0} & \mathbf {0} & \mathbf {I} & \mathbf {0} \\ \mathbf {0} & \mathbf {0} & \mathbf {0} & \mathbf {0} & \mathbf {0} & \mathbf {I} \end{array} \right],   \mathbf {F} _ {\mathbf {w}} = \left[ \begin{array}{c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c} - \mathbf {A} (\widehat {\boldsymbol {\omega}} _ {i} \Delta t) ^ {T} \Delta t & \mathbf {0} & \mathbf {0} & \mathbf {0} & \mathbf {0} \\ \mathbf {0} & \mathbf {0} & \mathbf {0} & \mathbf {0} & \mathbf {0} \\ \mathbf {0} & - ^ {G} \widehat {\mathbf {R}} _ {I _ {i}} \Delta t & \mathbf {0} & \mathbf {0} \\ \mathbf {0} & \mathbf {0} & \mathbf {I} \Delta t & \mathbf {0} \\ \mathbf {0} & \mathbf {0} & \mathbf {0} & \mathbf {I} \Delta t \\ \mathbf {0} & \mathbf {0} & \mathbf {0} & \mathbf {0} \end{array} \right] (7)
$$

measurements, each sampled at time $\tau _ { i } ~ \in ~ [ t _ { k - 1 } , t _ { k } ]$ with the respective state $\mathbf { x } _ { i }$ as in (2). Notice that the last LiDAR feature point is the end of a scan, $\mathrm { i } . \mathrm { e } . , \rho _ { m } = t _ { k }$ , while the IMU measurements may not necessarily be aligned with the start or end of the scan.

## C. State Estimation

To estimate the states in the state formulation (2), we use an iterated extended Kalman filter. Moreover, we characterize the estimation covariance in the tangent space of the state estimate as in [23, 24]. Assume the optimal state estimate of the last LiDAR scan at $t _ { k - 1 }$ is $\bar { \bf x } _ { k - 1 }$ with covariance matrix $\bar { \mathbf { P } } _ { k - 1 }$ . Then $\bar { \mathbf { P } } _ { k - 1 }$ represents the covariance of the random error state vector defined below:

$$
\widetilde {\mathbf {x}} _ {k - 1} \doteq \mathbf {x} _ {k - 1} \boxminus \bar {\mathbf {x}} _ {k - 1} = \left[ \delta \boldsymbol {\theta} ^ {T} G \widetilde {\mathbf {p}} _ {I} ^ {T} G \widetilde {\mathbf {v}} _ {I} ^ {T} \widetilde {\mathbf {b}} _ {\omega} ^ {T} \widetilde {\mathbf {b}} _ {\mathbf {a}} ^ {T} G \widetilde {\mathbf {g}} ^ {T} \right] ^ {T}
$$

where $\delta { \pmb \theta } = \mathrm { L o g } ( { \bf \mathrm { \bf { \bar { G } } } } \bar { \bf R } _ { I } ^ { T G } { \bf R } _ { I } )$ is the attitude error and the rests are standard additive errors (i.e., the error in the estimate x¯ of a quantity x is $\widetilde { \mathbf { x } } = \mathbf { x } - \bar { \mathbf { x } } )$ . Intuitively, the attitude error $\delta \pmb { \theta }$ edescribes the (small) deviation between the true and the estimated attitude. The main advantage of this error definition is that it allows us to represent the attitude uncertainty by the $3 \times 3$ covariance matrix E $\left\{ \delta \pmb { \theta } \delta \pmb { \theta } ^ { T } \right\}$ . Since the attitude has 3 degree of freedom (DOF), this is a minimal representation.

## 1) Forward Propagation:

The forward propagation is performed once receiving an IMU input (see Fig. 2). More specifically, the state is propagated following (2) by setting the process noise $\mathbf { w } _ { i }$ to zero:

$$
\widehat {\mathbf {x}} _ {i + 1} = \widehat {\mathbf {x}} _ {i} \boxplus (\Delta t \mathbf {f} (\widehat {\mathbf {x}} _ {i}, \mathbf {u} _ {i}, \mathbf {0})); \widehat {\mathbf {x}} _ {0} = \bar {\mathbf {x}} _ {k - 1}. \tag {4}
$$

where $\Delta t = \tau _ { i + 1 } - \tau _ { i }$ . To propagate the covariance, we use the error state dynamic model obtained below:

$$
\begin{array}{l} \widetilde {\mathbf {x}} _ {i + 1} = \mathbf {x} _ {i + 1} \boxminus \widehat {\mathbf {x}} _ {i + 1} \\ = \left(\mathbf {x} _ {i} \boxplus \Delta t \mathbf {f} \left(\mathbf {x} _ {i}, \mathbf {u} _ {i}, \mathbf {w} _ {i}\right)\right) \boxminus \left(\widehat {\mathbf {x}} _ {i} \boxplus \Delta t \mathbf {f} \left(\widehat {\mathbf {x}} _ {i}, \mathbf {u} _ {i}, \mathbf {0}\right)\right) \tag {5} \\ \stackrel {(2 3)} {\simeq} \mathbf {F} _ {\widetilde {\mathbf {x}}} \widetilde {\mathbf {x}} _ {i} + \mathbf {F} _ {\mathbf {w}} \mathbf {w} _ {i}. \\ \end{array}
$$

The matrix $\mathbf { F } _ { \widetilde { \mathbf { x } } }$ and $\mathbf { F _ { w } }$ in (5) is computed following ethe Appendix. A. The result is shown in $( 7 )$ , where $\widehat { \omega } _ { i } =$ $\omega _ { m _ { i } } - \widehat { \mathbf { b } } _ { \omega _ { i } } , \widehat { \mathbf { a } } _ { i } = \mathbf { a } _ { m _ { i } } - \widehat { \mathbf { b } } _ { \mathbf { a } }$ and ${ \mathbf { A } \left( { \mathbf { u } } \right) } ^ { - 1 }$ b follows the same bdefinition in [25] as below:

$$
\mathbf {A} (\mathbf {u}) ^ {- 1} = \mathbf {I} - \frac {1}{2} \left\lfloor \mathbf {u} \right\rfloor_ {\wedge} + (1 - \alpha (\| \mathbf {u} \|)) \frac {\left\lfloor \mathbf {u} \right\rfloor_ {\wedge} ^ {2}}{\| \mathbf {u} \| ^ {2}} \tag {6}
$$

$$
\alpha (\mathrm{m}) = \frac {\mathrm{m}}{2} \cot \left(\frac {\mathrm{m}}{2}\right) = \frac {\mathrm{m}}{2} \frac {\cos (\mathrm{m} / 2)}{\sin (\mathrm{m} / 2)}
$$

Denoting the covariance of white noises w as $\mathbf { Q } ,$ then the propagated covariance $\widehat { \mathbf { P } } _ { i }$ can be computed iteratively following the below equation.

$$
\widehat {\mathbf {P}} _ {i + 1} = \mathbf {F} _ {\widetilde {\mathbf {x}}} \widehat {\mathbf {P}} _ {i} \mathbf {F} _ {\widetilde {\mathbf {x}}} ^ {T} + \mathbf {F} _ {\mathbf {w}} \mathbf {Q} \mathbf {F} _ {\mathbf {w}} ^ {T}; \quad \widehat {\mathbf {P}} _ {0} = \bar {\mathbf {P}} _ {k - 1}. \tag {8}
$$

The propagation continues until reaching the end time of a new scan at $t _ { k }$ where the propagated state and covariance are denoted as $\widehat { \mathbf { x } } _ { k } , \widehat { \mathbf { P } } _ { k }$ . Then $\widehat { \mathbf { P } } _ { k }$ represents the covariance of the error between the ground-truth state $\mathbf { x } _ { k }$ and the state propagation $\widehat { \mathbf { x } } _ { k } \ ( \mathrm { i . e . , } \ \mathbf { x } _ { k } \boxed { } \widehat { \mathbf { x } } _ { k } )$ .

## 2) Backward Propagation and Motion Compensation:

When the points accumulation time interval is reached at time $t _ { k }$ , the new scan of feature points should be fused with the propagated state $\widehat { \mathbf { x } } _ { k }$ and covariance $\widehat { \mathbf { P } } _ { k }$ to produce an optimal state update. However, although the new scan is at time $t _ { k } ,$ the feature points are measured at their respective sampling time $\rho _ { j } \leq t _ { k }$ (see Section. III-B.4 and Fig. 2 (b)), causing a mismatch in the body frame of reference.

To compensate the relative motion $( \mathrm { i . e . }$ , motion distortion) between time $\rho _ { j }$ and time tk, we propagate (2) backward as $\check { \mathbf { x } } _ { j - 1 } = \check { \mathbf { x } } _ { j } \boxplus \left( - \Delta t \mathbf { f } ( \check { \mathbf { x } } _ { j } , \mathbf { u } _ { j } , \mathbf { 0 } ) \right)$ , starting from zero pose and rests states (e.g., velocity and bias) from $\widehat { \mathbf { x } } _ { k }$ . The backward propagation is performed at the frequency of feature point, which is usually much higher than the IMU rate. For all the feature points sampled between two IMU measurements, we use the left IMU measurement as the input in the back propagation. Furthermore, noticing that the last three block elements (corresponding to the gyro bias, accelerometer bias, and extrinsic) of $\mathbf { f } ( \mathbf { x } _ { j } , \mathbf { u } _ { j } , \mathbf { 0 } )$ (see (3)) are zeros, the back propagation can be reduced to:

$$
\begin{array}{l} { } ^ { I _ { k } } \check { \mathbf { p } } _ { I _ { j - 1 } } = { } ^ { I _ { k } } \check { \mathbf { p } } _ { I _ { j } } - { } ^ { I _ { k } } \check { \mathbf { v } } _ { I _ { j } } \Delta t , \quad s . f . { } ^ { I _ { k } } \check { \mathbf { p } } _ { I _ { m } } = \mathbf { 0 } ; \\ { } ^ { I _ { k } } \check { \mathbf { v } } _ { I _ { j - 1 } } = { } ^ { I _ { k } } \check { \mathbf { v } } _ { I _ { j } } - { } ^ { I _ { k } } \check { \mathbf { R } } _ { I _ { j } } ( \mathbf { a } _ { m _ { i - 1 } } - \widehat { \mathbf { b } } _ { \mathbf { a } _ { k } } ) \Delta t - { } ^ { I _ { k } } \widehat { \mathbf { g } } _ { k } \Delta t , \\ s. f. ^ {I _ {k}} \check {\mathbf {v}} _ {I _ {m}} = ^ {G} \widehat {\mathbf {R}} _ {I _ {k}} ^ {T G} \widehat {\mathbf {v}} _ {I _ {k}}, ^ {I _ {k}} \widehat {\mathbf {g}} _ {k} = ^ {G} \widehat {\mathbf {R}} _ {I _ {k}} ^ {T G} \widehat {\mathbf {g}} _ {k}; \\ { } ^ { I _ { k } } \check { \mathbf { R } } _ { I _ { j - 1 } } = { } ^ { I _ { k } } \check { \mathbf { R } } _ { I _ { j } } \operatorname{Exp} ( ( \widehat { \mathbf { b } } _ { \boldsymbol { \omega } _ { k } } - \boldsymbol { \omega } _ { m _ { i - 1 } } ) \Delta t ) , s . f . { } ^ { I _ { k } } \mathbf { R } _ { I _ { m } } = \mathbf { I } . \tag {9} \\ \end{array}
$$

where $\rho _ { j - 1 } \in [ \tau _ { i - 1 } , \tau _ { i } ) , \Delta t = \rho _ { j } - \rho _ { j - 1 }$ , and $s . f .$ means “starting from”.

The backward propagation will produce a relative pose between time $\rho _ { j }$ and the scan-end time $\begin{array} { r l } { t _ { k } \colon } & { { } ^ { I _ { k } } \check { \mathbf { T } } _ { I _ { j } } } & { = } \end{array}$ $\left( { { { I } _ { k } } } { { \check { \mathbf { R } } } _ { I _ { j } } } , \mathbf { { \check { \phi } } } ^ { I _ { k } } { { \check { \mathbf { p } } } _ { I _ { j } } } \right)$ local measurement ${ \cal L } _ { j } { \bf \sigma _ { p _ { f _ { j } } } }$ to scan-end measurement $\boldsymbol { L } _ { k }  _ { \mathbf { p } _ { f _ { j } } }$ as follows (see Fig. 2):

$$
{ } ^ { L _ { k } } \mathbf { p } _ { f _ { j } } = { } ^ { I } \mathbf { T } _ { L } ^ { - 1 I _ { k } } \check { \mathbf { T } } _ { I _ { j } } { } ^ { I } \mathbf { T } _ { L } { } ^ { L _ { j } } \mathbf { p } _ { f _ { j } } . \tag {10}
$$

where ${ { I } _ { { \mathbf { T } } _ { L } } }$ is the known extrinsic (see Section. III-B.2). Then the projected point ${ L } _ { k }  _ { \mathbf { p } _ { f _ { j } } }$ is used to construct a residual in the following section.

## 3) Residual computation:

With the motion compensation in (10), we can view the scan of feature points $\left\{ { \begin{array} { r l } \end{array} } { L _ { k } } _ { \mathbf { p } _ { f _ { j } } } \right\}$ time $t _ { k }$ and use it to construct the residual. Assume the current iteration of the iterated Kalman filter is $\kappa ,$ and the corresponding state estimate is $\widehat { \mathbf { x } } _ { k } ^ { \kappa }$ . When $\boldsymbol { \kappa } = 0 , \widehat { \mathbf { x } } _ { k } ^ { \kappa } = \widehat { \mathbf { x } } _ { k } .$ , b b bthe predicted state from the propagation in (4). Then, the feature points $\{ { ^ { L _ { k } } } _ { \mathbf { p } _ { f _ { j } } } \}$ can be transformed to the global frame as below:

$$
{ } ^ { G } \widehat { \mathbf { p } } _ { f _ { j } } ^ { \kappa } = { } ^ { G } \widehat { \mathbf { T } } _ { I _ { k } } ^ { \kappa   I } \mathbf { T } _ { L } { } ^ { L _ { k } } \mathbf { p } _ { f _ { j } } ; j = 1 , \cdots , m . \tag {11}
$$

For each LiDAR feature point, the closest plane or edge defined by its nearby feature points in the map is assumed to be where the point truly belongs to. That is, the residual is defined as the distance between the feature point’s estimated global frame coordinate $\boldsymbol { G } _ { \widehat { \mathbf { p } } _ { f _ { j } } ^ { \kappa } }$ and the nearest plane (or edge) in the map. Denoting $\mathbf { u } _ { j }$ the normal vector (or edge orientation) of the corresponding plane (or edge), on which lying a point ${ } ^ { G } { \bf q } _ { j }$ , then the residual $\mathbf { z } _ { j } ^ { \kappa }$ is computed as:

$$
\mathbf {z} _ {j} ^ {\kappa} = \mathbf {G} _ {j} \left(^ {G} \widehat {\mathbf {p}} _ {f _ {j}} ^ {\kappa} - ^ {G} \mathbf {q} _ {j}\right) \tag {12}
$$

where ${ \bf G } _ { j } = { \bf u } _ { i } ^ { T }$ for planar features and $\mathbf { G } _ { j } = \lfloor \mathbf { u } _ { j } \rfloor _ { \wedge }$ for edge features. The computation of the $\mathbf { u } _ { j }$ and the search of nearby points in the map, which define the corresponding plane or edge, is achieved by building a KD-tree of the points in the most recent map [10]. Moreover, we only consider residuals whose norm is below certain threshold (e.g., 0.5m). Residuals exceeding this threshold are either outliers or newly observed points.

## 4) Iterated state update:

To fuse the residual $\mathbf { z } _ { j } ^ { \kappa }$ computed in (12) with the state prediction $\widehat { \mathbf { x } } _ { k }$ and covariance $\widehat { \mathbf { P } } _ { k }$ propagated from the IMU data, we need to linearize the measurement model that relates the residual $\mathbf { z } _ { j } ^ { \kappa }$ to the ground-truth state $\mathbf { x } _ { k }$ and measurement noise. The measurement noise originates from the LiDAR ranging and beam-directing noise $\mathbf { \breve { L } } _ { j } \mathbf { \Phi _ { n _ { f _ { j } } } }$ when measuring the point ${ \cal L } _ { j } { \bf \sigma _ { p _ { f } } }$ . Removing this noise from the point measurement ${ L _ { j } } _ { \mathbf { p } _ { f _ { j } } }$ leads to the true point location

$$
{ } ^ { L _ { j } } \mathbf { p } _ { f _ { j } } ^ { \mathrm{gt} } = { } ^ { L _ { j } } \mathbf { p } _ { f _ { j } } - { } ^ { L _ { j } } \mathbf { n } _ { f _ { j } } . \tag {13}
$$

This true point, after projecting to the frame $L _ { k }$ via (10) and then to the global frame with the ground-truth state ${ \bf x } _ { k } ~ ( \mathrm { i . e . }$ , pose), should lie exactly on the plane (or edge) in the map. That is, plugging (13) into (10), then into (11), and further into (12) should result in zero. i.e.,

$$
\mathbf {0} = \mathbf {h} _ {j} \left(\mathbf {x} _ {k}, ^ {L _ {j}} \mathbf {n} _ {f _ {j}}\right) = \mathbf {G} _ {j} \left(^ {G} \mathbf {T} _ {I _ {k}} ^ {I _ {k}} \check {\mathbf {T}} _ {I _ {j}} ^ {I} \mathbf {T} _ {L} \left(^ {L _ {j}} \mathbf {p} _ {f _ {j}} - ^ {L _ {j}} \mathbf {n} _ {f _ {j}}\right) - ^ {G} \mathbf {q} _ {j}\right)
$$

Approximating the above equation by its first order approximation made at $\widehat { \mathbf { x } } _ { k } ^ { \kappa }$ leads to

$$
\begin{array}{l} \mathbf {0} = \mathbf {h} _ {j} \left(\mathbf {x} _ {k}, ^ {L _ {j}} \mathbf {n} _ {f _ {j}}\right) \simeq \mathbf {h} _ {j} \left(\widehat {\mathbf {x}} _ {k} ^ {\kappa}, \mathbf {0}\right) + \mathbf {H} _ {j} ^ {\kappa} \widetilde {\mathbf {x}} _ {k} ^ {\kappa} + \mathbf {v} _ {j} \tag {14} \\ = \mathbf {z} _ {j} ^ {\kappa} + \mathbf {H} _ {j} ^ {\kappa} \widetilde {\mathbf {x}} _ {k} ^ {\kappa} + \mathbf {v} _ {j} \\ \end{array}
$$

where $\widetilde { \mathbf { x } } _ { k } ^ { \kappa } = \mathbf { x } _ { k } \boxminus \widehat { \mathbf { x } } _ { k } ^ { \kappa }$ (or equivalently $\mathbf { x } _ { k } = \widehat { \mathbf { x } } _ { k } ^ { \kappa } \mathbb { H } \widetilde { \mathbf { x } } _ { k } ^ { \kappa } )$ , Hκj e b is the Jacobin matrix of $\mathbf h _ { j } \left( \widehat { \mathbf x } _ { k } ^ { \kappa } \boxed { \mathbb { Q } } \widetilde { \mathbf x } _ { k } ^ { \kappa } , \mathbf \Lambda ^ { L _ { j } } \mathbf { n } _ { f _ { j } } \right)$ b e   with respect to $\widetilde { \mathbf { x } } _ { k } ^ { \kappa }$ , evaluated at zero, and $\mathbf { v } _ { j } \in \mathcal { N } ( \mathbf { 0 } , \mathbf { R } _ { j } )$ comes from the ekraw measurement noise ${ L } _ { j } { \bf \Phi _ { n _ { f } } } ^ { \bf \bar { \Phi } }$ .

Notice that the prior distribution of $\mathbf { x } _ { k }$ obtained from the forward propagation in Section. III-C.1 is for

$$
\mathbf {x} _ {k} \boxminus \widehat {\mathbf {x}} _ {k} = \left(\widehat {\mathbf {x}} _ {k} ^ {\kappa} \boxplus \widetilde {\mathbf {x}} _ {k} ^ {\kappa}\right) \boxminus \widehat {\mathbf {x}} _ {k} = \widehat {\mathbf {x}} _ {k} ^ {\kappa} \boxminus \widehat {\mathbf {x}} _ {k} + \mathbf {J} ^ {\kappa} \widetilde {\mathbf {x}} _ {k} ^ {\kappa} \tag {15}
$$

where $\mathbf { J } ^ { \kappa }$ is the partial differentiation of $( \widehat { \mathbf { x } } _ { k } ^ { \kappa } \boxplus \widetilde { \mathbf { x } } _ { k } ^ { \kappa } ) \boxed { \varTheta } \widehat { \mathbf { x } } _ { k }$ with respect to $\widetilde { \mathbf { x } } _ { k } ^ { \kappa }$ evaluated at zero:

$$
\mathbf {J} ^ {\kappa} = \left[ \begin{array}{c c} \mathbf {A} \left(^ {G} \widehat {\mathbf {R}} _ {I _ {k}} ^ {\kappa} \boxminus {} ^ {G} \widehat {\mathbf {R}} _ {I _ {k}}\right) ^ {- T} & \mathbf {0} _ {3 \times 1 5} \\ \mathbf {0} _ {1 5 \times 3} & \mathbf {I} _ {1 5 \times 1 5} \end{array} \right] \tag {16}
$$

where ${ \bf A } ( \cdot ) ^ { - 1 }$ is defined in (6). For the first iteration $( \mathrm { i . e . }$ , the case of extended Kalman filter), $\widehat { \mathbf { x } } _ { k } ^ { \kappa } { = } \widehat { \mathbf { x } } _ { k } ,$ then $\mathbf { J } ^ { \kappa } { = } \mathbf { I } .$ .

Combining the prior in (15) with the posteriori distribution from (14) yields the maximum a-posteriori estimate (MAP):

$$
\min _ {\widetilde {\mathbf {x}} _ {k} ^ {\kappa}} \left(\| \mathbf {x} _ {k} \boxminus \widehat {\mathbf {x}} _ {k} \| _ {\widehat {\mathbf {P}} _ {k} ^ {- 1}} ^ {2} + \sum_ {j = 1} ^ {m} \| \mathbf {z} _ {j} ^ {\kappa} + \mathbf {H} _ {j} ^ {\kappa} \widetilde {\mathbf {x}} _ {k} ^ {\kappa} \| _ {\mathbf {R} _ {j} ^ {- 1}} ^ {2}\right) \tag {17}
$$

where $\| \mathbf { x } \| _ { \mathbf { M } } ^ { 2 } = \mathbf { x } ^ { T }$ Mx. Substituting the linearization of the prior in (15) into (17) and optimizing the resultant quadratic cost leads to the standard iterated Kalman filter [21], which can be computed below (to simplify the notation, let ${ \bf H } = [ { \bf H } _ { 1 } ^ { \kappa ^ { T } } , \cdot \cdot \cdot , \hat { { \bf H } _ { m } ^ { \star } } ] ^ { T } , { \bf R } = \mathrm { d i a g } ( \hat { { \bf R } _ { 1 , \cdot \cdot } } \cdot \cdot { \bf R } _ { m } ) , { \bf P } =$ $\left( \mathbf { J } ^ { \kappa } \right) ^ { - 1 } \widehat { \mathbf { P } } _ { k } ( \mathbf { J } ^ { \kappa } ) ^ { - T }$ , and $\mathbf { z } _ { k } ^ { \kappa } = \left[ { \mathbf { z } _ { 1 } ^ { \kappa } } ^ { T } , \cdot \cdot \cdot , \mathbf { z } _ { m } ^ { \kappa } \right] ^ { T } ) \colon$ z m

$$
\begin{array}{l} \mathbf {K} = \mathbf {P H} ^ {T} (\mathbf {H P H} ^ {T} + \mathbf {R}) ^ {- 1}, \\ \widehat {\mathbf {x}} _ {k} ^ {\kappa + 1} = \widehat {\mathbf {x}} _ {k} ^ {\kappa} \boxplus \left(- \mathbf {K} \mathbf {z} _ {k} ^ {\kappa} - (\mathbf {I} - \mathbf {K H}) \left(\mathbf {J} ^ {\kappa}\right) ^ {- 1} \left(\widehat {\mathbf {x}} _ {k} ^ {\kappa} \boxminus \widehat {\mathbf {x}} _ {k}\right)\right). \tag {18} \\ \end{array}
$$

The updated estimate $\widehat { \mathbf { x } } _ { k } ^ { \kappa + 1 }$ is then used to compute the residual in Section. III-C.3 and repeat the process until $( \mathrm { i } . \mathrm { e } . , \| \widehat { \mathbf { x } } _ { k } ^ { \kappa + 1 } \bigtriangledown \widehat { \mathbf { x } } _ { k } ^ { \kappa } \| < \epsilon )$ . After convergence, the optimal state estimation and covariance is:

$$
\bar {\mathbf {x}} _ {k} = \hat {\mathbf {x}} _ {k} ^ {\kappa + 1}, \bar {\mathbf {P}} _ {k} = (\mathbf {I} - \mathbf {K H}) \mathbf {P} \tag {19}
$$

A problem with the commonly used Kalman gain form in (18) is that it requires to invert the matrix ${ \bf { H P H } } ^ { \mathbf { { \bar { T } } } } { + } { \bf { R } }$ which is in the dimension of the measurements. In practice, the number of LiDAR feature points are very large in number, inverting a matrix of this size is prohibitive. As such, existing works [21, 26] only use a small number of measurements. In this paper, we show that this limitation can be avoided. The intuition originates from (17) where the cost function is over the state, hence the solution should be calculated with complexity depending on the state dimension. In fact, if directly solving (17), we can obtain the same solution in (18) but with a new form of Kalman gain shown below:

$$
\mathbf {K} = \left(\mathbf {H} ^ {T} \mathbf {R} ^ {- 1} \mathbf {H} + \mathbf {P} ^ {- 1}\right) ^ {- 1} \mathbf {H} ^ {T} \mathbf {R} ^ {- 1}. \tag {20}
$$

We prove in Appendix B that the two forms of Kalman gains are indeed equivalent based on the matrix inverse lemma [27]. Since the LiDAR measurements are independent, the covariance matrix R is (block) diagonal and hence the new formula only requires to invert two matrices both in the dimension of state instead of measurements. The new formula greatly saves the computation as the state dimension is usually much lower than measurements in LIO (e.g., more than 1,000 effective feature points in a scan for 10 Hz scan rate while the state dimension is only 18).

## 5) The algorithm:

Our state estimation is summarized in Algorithm 1.

Algorithm 1: State Estimation  
Input : Last optimal estimation $\bar{x}_{k-1}$ and $\bar{P}_{k-1}$ , IMU inputs $(\mathbf{a}_{m}, \omega_{m})$ in current scan; LiDAR feature points $L_{j}p_{f_{j}}$ in current scan.

1 Forward propagation to obtain state prediction $\widehat{x}_{k}$ via (4) and covariance prediction $\widehat{P}_{k}$ via (8);

2 Backward propagation to obtain $L_{k}p_{f_{j}}$ via (9), (10);

3 $\kappa = -1$ , $\widehat{x}_{k}^{\kappa=0} = \widehat{x}_{k}$ ;

4 repeat

5 $\kappa = \kappa + 1$ ;

6 Compute $J^{\kappa}$ via (16) and $\mathbf{P} = (\mathbf{J}^{\kappa})^{-1}\widehat{\mathbf{P}}_{k}(\mathbf{J}^{\kappa})^{-T}$ ;

7 Compute residual $z_{j}^{\kappa}$ (12) and Jocobin $H_{j}^{\kappa}$ (14);

8 Compute the state update $\widehat{x}_{k}^{\kappa+1}$ via (18) with the Kalman gain K from (20);

9 until $\|\widehat{x}_{k}^{\kappa+1}\boxminus\widehat{x}_{k}^{\kappa}\| < \epsilon$ ;

10 $\bar{x}_{k} = \widehat{x}_{k}^{\kappa+1}$ ; $\bar{P}_{k} = (\mathbf{I} - \mathbf{KH})\mathbf{P}$ .

Output: Current optimal estimation $\bar{x}_{k}$ and $\bar{P}_{k}$ .

## D. Map Update

With the state update $\bar { \bf x } _ { k }$ (hence ${ } ^ { G } \bar { \mathbf { T } } _ { I _ { k } } = ( ^ { G } \bar { \mathbf { R } } _ { I _ { k } } , ^ { G } \bar { \mathbf { p } } _ { I _ { k } } ) ) .$ each feature point $\bigl ( { \mathbf { } } ^ { L _ { k } } \mathbf { p } _ { f _ { j } } \bigr )$ projected to the body frame $L _ { k }$ (see (10)) is then transformed to the global frame via:

$$
{ } ^ { G } \bar { \mathbf { p } } _ { f _ { j } } = { } ^ { G } \bar { \mathbf { T } } _ { I _ { k } } { } ^ { I } \mathbf { T } _ { L } { } ^ { L _ { k } } \mathbf { p } _ { f _ { j } } ; j = 1 , \cdots , m . \tag {21}
$$

These features points are finally appended to the existing map containing feature points from all previous steps.

## E. Initialization

To obtain a good initial estimate of the system state (e.g., gravity vector $\breve { G } _ { \mathbf { g } } ,$ bias, and noise covariance) so to speedup the state estimator, initialization is required. In FAST-LIO, the initialization is simple: keeping the LiDAR static for several seconds (2 seconds for all the experiments in this paper), the collected data is then used to initialize the IMU bias and the gravity vector. If non-repetitive scanning is supported by the LiDAR (e.g., Livox AVIA), keeping static also allows the LiDAR to capture an initial high-resolution map that is beneficial for the subsequent navigation.

## IV. EXPERIMENT RESULTS

## A. Computational Complexity Experiments

In order to validate the computational efficiency of the proposed new formula for computing Kalman gains. We intentionally replace the computation of Kalman gains with the old formula in our system and compare their computation time under the same system pipeline and number of feature points. The results are shown in Table. II. It is obvious that the complexity of the new formula is much lower than the old one.

TABLE II THE RUNNING TIMES OF TWO KALMAN GAIN FORMULAS

<table><tr><td>Feature Num.</td><td>307</td><td>717</td><td>998</td><td>1243</td><td>1453</td><td>1802</td></tr><tr><td>Old Formula (ms)</td><td>7.1</td><td>23.4</td><td>109.3</td><td>251</td><td>1219</td><td>1621</td></tr><tr><td>New Formula (ms)</td><td>0.07</td><td>0.11</td><td>0.25</td><td>0.37</td><td>0.59</td><td>1.16</td></tr></table>

![](images/9fffb7633d0ebaa10726b107c4221098dd00e1d191ffb47965206001c06a23d4.jpg)

<details>
<summary>natural_image</summary>

Two-panel scientific image showing a device with glowing green light patterns and red baseline, alongside a zoomed-in view of the device's internal structure (no text or symbols)
</details>

Fig. 3. During the flight experiment, the UAV is automatically flying in a circle path with 1.8 m radius and 1.4 m height. The circle path is conducted repeatedly for 4 times with different periods (6-10 s). The yaw command of the UAV maintains constant during the flight. In the end, the UAV is manually controlled to land at the take-off point, which enables us to measure the drift.

## B. UAV Flight Experiments

In order to validate the robustness and computational efficiency of FAST-LIO in actual mobile robots, we build a small-scale quadrotor that can carry a Livox Avia LiDAR with $7 0 ^ { \circ }$ FoV and a DJI Manifold 2-C onboard computer with a 1.8 $G H z$ quad-core Intel i7-8550U CPU and 8 GB RAM, as shown in Fig. 1. The UAV has only 280 mm wheelbase, and the LiDAR is directly installed on the airframe. The LiDAR-inertial odometry is sent to the flight controller tracking a circle trajectory (Fig. 3). The actual flight experiments show that FAST-LIO can achieve real-time and stable odometry output and mapping in a maximum of 50 $H z$ for the indoor environment. The flight trajectory and mapping results of 50 $H z$ frame rate indoor experiment are shown in Fig. 3. The average number of effective feature points and running time is 270 and 6.7 ms, respectively , the drift is smaller than 0.3% (0.08 m drift over 32 m trajectory). The flight video can be found at https://youtu.be/iYCY6T79oNU.

## C. Indoor Experiments

Then we test FAST-LIO in a challenging indoor environment with large rotation speeds. In order to generate large rotation, the sensor suite is held on hands and undergoes quickly shaking. Fig. 4 shows the angular velocity and acceleration during the experiment. It is seen that the angular velocity often exceeds 100 deg/s. A state of the art implementation of LOAM on Livox $\mathrm { L i D A R s } ^ { 5 }$ [10] and LOAM with IMU6 [8] are also tested as comparisons when the feature extraction are replaced with the one of FAST-LIO. The results show that FAST-LIO can output odometry faster and more stable than others, as shown in Fig. 5 and Table. III. It should be noted that the LOAM+IMU is a loosely-coupled

5https://github.com/Livox-SDK/livox\_mapping  
6https://github.com/Livox-SDK/livox\_horizon\_loam

TABLE III COMPARISON OF PROCESSING TIME FOR A LIDAR SCAN AT 10Hz

<table><tr><td>Packages</td><td>Num. of effective features</td><td>Running time</td></tr><tr><td>LOAM</td><td>1107</td><td>59 ms</td></tr><tr><td>LOAM+IMU</td><td>1107</td><td>44 ms</td></tr><tr><td>FAST-LIO</td><td>1430</td><td>23 ms</td></tr></table>

![](images/8578e6f5761bdfbd9b2d2de5212d3b42cf8c3802f3ffed61c4c23c5d2dbddc1d.jpg)

<details>
<summary>line chart</summary>

| time (s) | x     | y     | z     |
| -------- | ----- | ----- | ----- |
| 0        | 0     | 0     | 0     |
| 5        | 0     | 0     | 0     |
| 10       | 0     | 0     | 0     |
| 15       | 0     | 0     | 0     |
| 20       | 0     | 0     | 0     |
| 25       | 0     | 0     | 0     |
| 30       | 0     | 0     | 0     |
| 35       | 0     | 0     | 0     |
| 40       | 0     | 0     | 0     |
| 45       | 0     | 0     | 0     |
| 50       | 0     | 0     | 0     |
</details>

![](images/e5dc295579883bc4dd78e865576fb3f85281e4af6e2f847a6a1667179c0174a8.jpg)

<details>
<summary>line chart</summary>

| time (s) | x    | y    | z    |
| -------- | ---- | ---- | ---- |
| 0        | 0    | 0    | 10   |
| 10       | 0    | 0    | 10   |
| 20       | 0    | -10  | 10   |
| 30       | 0    | 0    | 10   |
| 40       | 0    | 0    | 10   |
| 50       | 0    | 0    | 10   |
</details>

Fig. 4. The angular velocity and acceleration in the indoor experiments.

method, hence results in inconsistent mapping. To further verify the mapping result, we perform a second experiment in the same environment but with a much slower motion. The map built by FAST-LIO is shown in the lower-right figure of Fig. 4. Since the two experiments have non-identical movements, it leads to slight visual differences at places occlusions occur. The rest mapping results are very close.

## D. Outdoor Experiments

Here we show the performance of FAST-LIO in outdoor environments. Fig. 6 shows the mapping results (displaying all raw points) of the Main Building in the University of Hong Kong. The sensor suite is handheld during the data collection and returned to the starting position after traveling around 140m. The drift in this experiment is smaller than 0.05% (0.07 m drift over 140 m trajectory). The scan rate is set to 10 $H z$ in this experiment, and the average processing time of a scan is 25 ms with average 1497 effective feature points.

Further, we compare FAST-LIO with $\mathrm { L I N S } ^ { 7 }$ [21]. To make a fair comparison, we use the dataset from LINS [21], which is a seaport area data collected by a Velodyne VLP-16 and an Xsens MTiG-710 $\mathrm { I M U } ^ { 7 }$ . The results show that the FAST-LIO can achieve better mapping accuracy (see Fig. 7) and only consumes 7.3 ms processing time in average while LINS takes 34.5 ms in average, both running at 10Hz. It should be noted that since the EKF formula in LINS package has high computational complexity (see Section. III-C.4)), it down samples the feature points to average 147 points in a scan (while 784 in a scan for FAST-LIO). This leads to degraded mapping accuracy for LINS. The result in Fig. 7 shows all the feature points (before down-sample) of FAST-LIO and LINS. All the experiments are conducted on the DJI Manifold2 onboard computer.

![](images/966294610bf26572b44a5958fc702f0d39a57735517df1a3fef209d6cac60e0b.jpg)

<details>
<summary>natural_image</summary>

Fluorescent microscopy image showing green and red emission patterns in a biological structure (no text or symbols)
</details>

![](images/d91625dbe0cffed37bd5642f465c66f78ca82539df318a61f937662c212bc689.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a mechanical assembly with green and red lighting (no text or symbols visible)
</details>

![](images/75dcea56f14ec1c68c07631735659fdb10f5bc3df37bba1018f9ebd4a92dfbb0.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a building interior with color-coded temperature gradient (no text or symbols)
</details>

![](images/177f2959df9171b87832899271c0d04186ee427b8a66807509bfb53bd3b00585.jpg)

<details>
<summary>text_image</summary>

FAST-LIO + SLOW MOTION
</details>

Fig. 5. The Mapping results of different LIO packages in an indoor environment with large rotation speed.

![](images/c380acecb001823351ffc42ce51525cbd16193642cbb9eee6e8ec6acb818761c.jpg)

<details>
<summary>natural_image</summary>

Nighttime aerial view of a historic building complex with illuminated red and green light trails, showing people walking on a bridge (no visible text or symbols)
</details>

Fig. 6. Mapping results of the Main Building, University of Hong Kong. The LiDAR platform, the same one in Fig. 1, is handheld randomly walking to scan the building. In order to show the drift, the experiment are started and ended at the same place.

## V. CONCLUSION

This paper proposed FAST-LIO, a computationally efficient and robust LiDAR-inertial odometry framework by a tightly-coupled iterated Kalman filter. We used the forward and backward propagation to predict the states and compensate for the motion in a LiDAR scan. Besides, we proved and implemented an equivalent formula that can achieve much lower complexity for the Kalman gain computation. FAST-LIO was tested in the UAV flight experiment, challenging indoor environment with large rotation speed and outdoor environment. In all tests, our method produced precise, realtime, and reliable navigation results.

## APPENDIX

## A. Computation of $\mathbf { F } _ { \widetilde { \mathbf { x } } }$ and $\mathbf { F _ { w } }$

Recall $\begin{array} { r l r } { { \bf x } _ { i } } & { { } = } & { \widehat { \bf x } _ { i } \boxplus \ \widetilde { \bf x } _ { i } , } \end{array}$ , denote $\begin{array} { r l } { \mathbf { g } \left( \widetilde { \mathbf { x } } _ { i } , \mathbf { w } _ { i } \right) } & { { } = } \end{array}$ $\mathbf { f } ( \mathbf { x } _ { i } , \mathbf { u } _ { i } , \mathbf { w } _ { i } ) \Delta t = \mathbf { f } ( \widehat { \mathbf { x } } _ { i } \boxplus \widetilde { \mathbf { x } } _ { i } , \mathbf { u } _ { i } , \mathbf { w } _ { i } ) \Delta t$ . Then the error state

![](images/c1da697af0696442565713a3d65cf945d4b2c449fdc9768898472ca8877b69ae.jpg)  
Fig. 7. Comparison between LINS [21] and FAST-LIO. The data is from [21] and is collected by a Velodyne VLP-16 LiDAR and an Xsens MTiG-710 IMU, the green straight line in the center is the odometry output.

model (5) is rewriten as:

$$
\widetilde {\mathbf {x}} _ {i + 1} = \underbrace {\left(\left(\widehat {\mathbf {x}} _ {i} \boxplus \widetilde {\mathbf {x}} _ {i}\right) \boxplus \mathbf {g} \left(\widetilde {\mathbf {x}} _ {i} , \mathbf {w} _ {i}\right)\right) \boxminus \left(\widehat {\mathbf {x}} _ {i} \boxplus \mathbf {g} (\mathbf {0} , \mathbf {0})\right)} _ {\mathbf {G} \left(\widetilde {\mathbf {x}} _ {i}, \mathbf {g} \left(\widetilde {\mathbf {x}} _ {i}, \mathbf {w} _ {i}\right)\right)} \tag {22}
$$

Following the chain rule of partial differention, the matrix $\mathbf { F } _ { \widetilde { \mathbf { x } } }$ and $\mathbf { F _ { w } }$ in (5) are computed as below.

$$
\mathbf {F} _ {\widetilde {\mathbf {x}}} = \left. \left(\frac {\partial \mathbf {G} (\widetilde {\mathbf {x}} _ {i} , \mathbf {g} (\mathbf {0} , \mathbf {0}))}{\partial \widetilde {\mathbf {x}} _ {i}} + \frac {\partial \mathbf {G} (\mathbf {0} , \mathbf {g} (\widetilde {\mathbf {x}} _ {i} , \mathbf {0}))}{\partial \mathbf {g} (\widetilde {\mathbf {x}} _ {i} , \mathbf {0})} \frac {\partial \mathbf {g} (\widetilde {\mathbf {x}} _ {i} , \mathbf {0})}{\partial \widetilde {\mathbf {x}} _ {i}}\right) \right| _ {\widetilde {\mathbf {x}} _ {i} = \mathbf {0}} \tag {23}
$$

$$
\mathbf {F} _ {\mathbf {w}} = \left. \left(\frac {\partial \mathbf {G} (\mathbf {0} , \mathbf {g} (\mathbf {0} , \mathbf {w} _ {i}))}{\partial \mathbf {g} (\mathbf {0} , \mathbf {w} _ {i})} \frac {\partial \mathbf {g} (\mathbf {0} , \mathbf {w} _ {i})}{\partial \mathbf {w} _ {i}}\right) \right| _ {\mathbf {w} _ {i} = \mathbf {0}}
$$

## B. Equivalent Kalman Gain formula

Based on the matrix inverse lemma [27], we can get:

$$
\left(\mathbf {P} ^ {- 1} + \mathbf {H} ^ {T} \mathbf {R} ^ {- 1} \mathbf {H}\right) ^ {- 1} = \mathbf {P} - \mathbf {P H} ^ {T} \left(\mathbf {H P H} ^ {T} + \mathbf {R}\right) ^ {- 1} \mathbf {H P}
$$

Substituting above into (20), we can get:

$$
\begin{array}{l} \mathbf {K} = \left(\mathbf {H} ^ {T} \mathbf {R} ^ {- 1} \mathbf {H} + \mathbf {P} ^ {- 1}\right) ^ {- 1} \mathbf {H} ^ {T} \mathbf {R} ^ {- 1} \\ = \mathbf {P H} ^ {T} \mathbf {R} ^ {- 1} - \mathbf {P H} ^ {T} \left(\mathbf {H P H} ^ {T} + \mathbf {R}\right) ^ {- 1} \mathbf {H P H} ^ {T} \mathbf {R} ^ {- 1} \\ \end{array}
$$

Now note that $\mathbf { H } \mathbf { P } \mathbf { H } ^ { T } \mathbf { R } ^ { - 1 } = \left( \mathbf { H } \mathbf { P } \mathbf { H } ^ { T } + \mathbf { R } \right) \mathbf { R } ^ { - 1 } - \mathbf { I }$ . Substituting it into above, we can get the standard Kalman gain formula in (18), as shown below.

$$
\mathbf {K} = \mathbf {P H} ^ {T} \mathbf {R} ^ {- 1} - \mathbf {P H} ^ {T} \mathbf {R} ^ {- 1} + \mathbf {P H} ^ {T} (\mathbf {H P H} ^ {T} + \mathbf {R}) ^ {- 1}
$$

$$
= \mathbf {P H} ^ {T} \left(\mathbf {H P H} ^ {T} + \mathbf {R}\right) ^ {- 1}. \quad \blacksquare
$$

## REFERENCES

[1] K. Sun, K. Mohta, B. Pfrommer, M. Watterson, S. Liu, Y. Mulgaonkar, C. J. Taylor, and V. Kumar, “Robust stereo visual inertial odometry for fast autonomous flight,” IEEE Robotics and Automation Letters, vol. 3, no. 2, pp. 965–972, 2018.  
[2] T. Qin, P. Li, and S. Shen, “Vins-mono: A robust and versatile monocular visual-inertial state estimator,” IEEE Transactions on Robotics, vol. 34, no. 4, pp. 1004–1020, 2018.  
[3] C. Forster, M. Pizzoli, and D. Scaramuzza, “Svo: Fast semi-direct monocular visual odometry,” in 2014 IEEE international conference on robotics and automation (ICRA). IEEE, 2014, pp. 15–22.  
[4] D. Wang, C. Watkins, and H. Xie, “Mems mirrors for lidar: A review,” Micromachines, vol. 11, no. 5, p. 456, 2020.  
[5] Z. Liu, F. Zhang, and X. Hong, “Low-cost retina-like robotic lidars based on incommensurable scanning,” IEEE/ASME Transactions on Mechatronics, pp. 1–1, 2021.  
[6] P. J. Besl and N. D. McKay, “Method for registration of 3-d shapes,” in Sensor fusion IV: control paradigms and data structures, vol. 1611. International Society for Optics and Photonics, 1992, pp. 586–606.  
[7] A. Segal, D. Haehnel, and S. Thrun, “Generalized-icp.” in Robotics: science and systems, vol. 2, no. 4. Seattle, WA, 2009, p. 435.  
[8] J. Zhang and S. Singh, “Loam: Lidar odometry and mapping in realtime.” in Robotics: Science and Systems, vol. 2, no. 9, 2014.  
[9] T. Shan and B. Englot, “Lego-loam: Lightweight and groundoptimized lidar odometry and mapping on variable terrain,” in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018, pp. 4758–4765.  
[10] J. Lin and F. Zhang, “Loam livox: A fast, robust, high-precision lidar odometry and mapping package for lidars of small fov,” in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 3126–3131.  
[11] W. Zhen, S. Zeng, and S. Soberer, “Robust localization and localizability estimation with a rotating laser scanner,” in 2017 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2017, pp. 6240–6245.  
[12] Y. Balazadegan Sarvrood, S. Hosseinyalamdary, and Y. Gao, “Visuallidar odometry aided by reduced imu,” ISPRS international journal of geo-information, vol. 5, no. 1, p. 3, 2016.  
[13] X. Zuo, P. Geneva, W. Lee, Y. Liu, and G. Huang, “Lic-fusion: Lidarinertial-camera odometry,” in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019, pp. 5848–5854.  
[14] P. Geneva, K. Eckenhoff, Y. Yang, and G. Huang, “Lips: Lidarinertial 3d plane slam,” in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018, pp. 123–130.  
[15] C. Forster, L. Carlone, F. Dellaert, and D. Scaramuzza, “On-manifold preintegration for real-time visual–inertial odometry,” IEEE Transactions on Robotics, vol. 33, no. 1, pp. 1–21, 2016.  
[16] M. Hsiao, E. Westman, and M. Kaess, “Dense planar-inertial slam with structural constraints,” in 2018 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018, pp. 6521–6528.  
[17] H. Ye, Y. Chen, and M. Liu, “Tightly coupled 3d lidar inertial odometry and mapping,” in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 3144–3150.  
[18] A. Bry, A. Bachrach, and N. Roy, “State estimation for aggressive flight in gps-denied environments using onboard sensing,” in 2012 IEEE International Conference on Robotics and Automation. IEEE, 2012, pp. 1–8.  
[19] J. A. Hesch, F. M. Mirzaei, G. L. Mariottini, and S. I. Roumeliotis, “A laser-aided inertial navigation system (l-ins) for human localization in unknown indoor environments,” in 2010 IEEE International Conference on Robotics and Automation. IEEE, 2010, pp. 5376–5382.  
[20] Z. Cheng, D. Liu, Y. Yang, T. Ling, X. Chen, L. Zhang, J. Bai, Y. Shen, L. Miao, and W. Huang, “Practical phase unwrapping of interferometric fringes based on unscented kalman filter technique,” Optics express, vol. 23, no. 25, pp. 32 337–32 349, 2015.  
[21] C. Qin, H. Ye, C. E. Pranata, J. Han, S. Zhang, and M. Liu, “Lins: A lidar-inertial state estimator for robust and efficient navigation,” in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 8899–8906.  
[22] M. Raitoharju and R. Piche, “On computational complexity reduction´ methods for kalman filter extensions,” IEEE Aerospace and Electronic Systems Magazine, vol. 34, no. 10, pp. 2–19, 2019.  
[23] C. Hertzberg, R. Wagner, U. Frese, and L. Schroder, “Integrating ¨ generic sensor fusion algorithms with sound state representations through encapsulation of manifolds,” Information Fusion, vol. 14, no. 1, pp. 57–77, 2013.  
[24] W. Xu, D. He, Y. Cai, and F. Zhang, “Robots state estimation and observability analysis based on statistical motion models,” 2020.  
[25] F. Bullo and R. M. Murray, “Proportional derivative (pd) control on the euclidean group,” in European control conference, vol. 2, 1995, pp. 1091–1097.  
[26] M. Bloesch, M. Burri, S. Omari, M. Hutter, and R. Siegwart, “Iterated extended kalman filter based visual-inertial odometry using direct photometric feedback,” The International Journal of Robotics Research, vol. 36, no. 10, pp. 1053–1072, 2017.  
[27] N. J. Higham, Accuracy and stability of numerical algorithms. SIAM, 2002.