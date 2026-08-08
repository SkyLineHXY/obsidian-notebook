---
title: "GR00T N1.6"
source: "https://research.nvidia.com/labs/gear/gr00t-n1_6/"
author:
published: 2025-12-15
created: 2026-08-08
description:
tags:
  - "clippings"
---
## GR00T N1.6An Improved Open Foundation Model for Generalist Humanoid Robots

[GitHub](https://github.com/NVIDIA/Isaac-GR00T/tree/main) [Model Card](https://huggingface.co/nvidia/GR00T-N1.6-3B/)

## Introduction

We introduce GR00T N1.6, an improved version of the GR00T N1.5 foundation model for humanoid robots. With several architecture, data and modeling improvements, we find that N1.6 outperforms N1.5 on both simulated manipulation benchmarks and on real bimanual YAM, Agibot Genie-1 and Unitree G1 robots, detailed below. We expect users of N1.6 should observe better post-training performance compared to N1.5.

## Model and Data Improvements

Architectural changes:

- Base VLM: We use an internal NVIDIA Cosmos-2B VLM variant. The VLM supports flexible resolution and can encode images in their native aspect ratio without padding. The VLM is trained on both general vision-language tasks and embodied reasoning tasks like next action prediction.
- Uses 2x larger DiT (32 layers vs 16 layers in N1.5).
- Removes N1.5's post-VLM 4-layer transformer adapter. Instead, we unfreeze the top 4 layers of the VLM during pretraining.
- Predicts state-relative action chunks for most embodiments, rather than absolute joint angles or EEF positions.

Beyond the N1.5 data mixture, the N1.6 pretraining data additionally includes several thousand hours of teleoperated data from:

- Bimanual YAM arms
- AGIBot Genie1
- Simulated Galaxea R1 Pro on the BEHAVIOR suite
- Whole-Body Locomanipulation with Unitree G1
![Pretraining data distribution](https://research.nvidia.com/labs/gear/n1_6/training_data_distribution_v3.svg)

Weighting of training data in GR00T N1.6 pretraining.

### Bimanual YAM Demo Videos

### AgiBot Demo Videos

### Unitree G1 Locomanipulation Demo Videos

## Experiments

GR00T N1.6 was pretrained for 300K steps with global batch size 16384.

In the following robot experiments, we further post-train on small task-specific datasets; typically 10K-30K steps with global batch size 1K or less.

## Discussion

For GR00T N1.6, we conduct more complex real-world robot experiments than GR00T N1.5, requiring long-horizon reasoning, dexterity, and multi-tasking abilities. When scaling up real-world experiments, we incorporate various lessons learned from the robot learning community to improve model success rates during rollouts.

- Relative actions are used as the default action space for most embodiments. Our experiments show that relative actions produce smoother and more accurate motions than absolute actions. However, with small datasets, relative actions are prone to error accumulation, which impacts correction ability.
- Pretrained statistics can improve performance when the task distribution is similar to the pretraining data; otherwise, the model may underfit, so we use post-training statistics when distributions differ.
- GR00T N1.6 converges faster than GR00T N1.5, leading to smoother actions, but requires more careful tuning to prevent overfitting. We apply stronger state regularization, additional data augmentations, and co-training with pretraining data to regularize the model during post-training.
- DAgger effectively improves model performance; this is recommended to be used when the model is underperforming in real-world experiments.
- Test-time and train-time RTC provide performance boosts to motion smoothness and robustness during asynchronous rollouts. We employ this technique in Unitree G1 and Bimanual YAM experiments.
- Multi-task language following and out-of-distribution task generalization continue to be challenging for current VLA models. More fine-grained subtask annotation can improve language following, but not yet reaching robust generalization. This will be a continuous effort in future research.

Overall, GR00T N1.6 represents an improvement over GR00T N1.5 across diverse embodiments. We expect users to benefit from improved performance in bimanual manipulation and locomanipulation tasks.

Foundation ModelHumanoid RobotVLMLanguage Following

Authors (alphabetical):

\*GEAR Team,Allison Azzolini,Johan Bjorck,Valts Blukis,Fernando Castañeda,Rahul Chand,Yan Chang,Danyi Chen,Nikita Cherniadev,Xingye Da,Runyu Ding,Shunjia Ding,Hassan Eslami,Linxi "Jim" Fan,Yu Fang,Max Fu,Shenyuan Gao,Yunhao Ge,Fengyuan Hu,Spencer Huang,Joel Jang,Xiaowei Jiang,Yunfan Jiang,Ryan Julian,Kaushil Kundalia,Jan Kautz,Zhiqi Li,Kevin Lin,Wei Liu,Runyu Lu,Zhengyi Luo,Loic Magne,Yunze Man,Ajay Mandlekar,Abhishek Mishra,Avnish Narayan,Connor Pederson,Nadun Ranawaka,Scott Reed,Sunil Srinivasa,You Liang Tan,Guanzhi Wang,Jing Wang,Qi Wang,Shihao Wang,Jimmy Wu,Yubo Wu,Yuqi Xie,Tianyi Xiong,Mengda Xu,Yinzhen Xu,Fu-En Yang,Seonghyeon Ye,Zhiding Yu,K.R. Zentner,Zhe Zhang,Kaiyuan Zheng,Ruijie Zheng,Yuke Zhu

Acknowledgements:

We thank various members of NVIDIA who have contributed to data curation, robot system development, testing GR00T N1.6, and advising. Members includes: Alec Nagal, Amanpreet Singh, Amy Nguyen, Ashley Kim, Chi-Pin Huang, Curie Park, Greg Lo, Isabel Zuluaga, Ivy Tam, Jeremy Chimienti, Jiasheng Gu, Jinwei Gu, Jonathan Tremblay, Juan Zuluaga, Leilee Naderi, Lion Park, Mimoza Huynh, Min-Hung Chen, Nic Ma, Rowland O Flaherty, Sean Gillen, Tri Cao, Tsung-Yi Lin, Yashraj Narang, and, many more