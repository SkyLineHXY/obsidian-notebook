---
title: "GR00T N1.5"
source: "https://research.nvidia.com/labs/gear/gr00t-n1_5/"
author:
published: 2025-06-11
created: 2026-08-08
description:
tags:
  - "clippings"
---
## GR00T N1.5An Improved Open Foundation Model for Generalist Humanoid Robots

[GitHub](https://github.com/NVIDIA/Isaac-GR00T/tree/main) [Model Card](https://huggingface.co/nvidia/GR00T-N1.5-3B/)

## Introduction

We introduce GR00T N1.5, an upgraded version of the GR00T N1 foundation model for humanoid robots. With several architecture, data and modeling improvements, we find that N1.5 outperforms N1 on both simulated manipulation benchmarks and on the real GR-1 robot, detailed below. We expect users of N1.5 should observe better performance compared to N1, in particular improved generalization and better language following ability.

## Model and Data Updates

### Architecture

As with N1, GR00T N1.5 uses an [NVIDIA Eagle](https://github.com/NVlabs/EAGLE) VLM to encode text and visual observations. The vision-language embeddings from the VLM are then cross-attended to by the DiT which processes the state and noised actions.

![GR00T N1.5 Architecture Diagram](https://research.nvidia.com/labs/gear/n1_5/architecture.svg)

The main differences from N1 are the following:

- The VLM model is frozen during both pretraining and finetuning.
- The adapter MLP connecting the vision encoder to the LLM is simplified and adds layer normalization to both visual and text token embeddings input to the LLM.

We found that these modifications greatly improved language following and generalization.

### Improved VLM Grounding Capabilities

We have updated the VLM of GR00T N1.5, starting from Eagle 2.5 and tuned for better grounding capabilities and physical understanding. On RefCOCOg and our internal GEAR GR-1 grounding dataset with referring expressions, we observe that the N1.5 VLM model performs favorably compared to Qwen2.5-VL-3B - a comparable open-source model.

| Model | Size | GR-1 grounding IoU (↑) | RefCOCOg-val IoU (↑) |
| --- | --- | --- | --- |
| Qwen2.5VL | 3B | 35.5 | 85.2 |
| GR00T N1.5 VLM | 2.1B | **40.4** | **89.6** |

![Grounding example 1](https://research.nvidia.com/labs/gear/n1_5/grounding_example2.png) ![Grounding example 2](https://research.nvidia.com/labs/gear/n1_5/grounding_example1.png)

Left: Example annotations from our test dataset. Right: example model output.

### Joint Policy Learning and World Modeling Objective

In addition to the flow matching loss used by N1, for N1.5 we add Future LAtent Representation Alignment (see [FLARE](https://research.nvidia.com/labs/gear/flare) project). Rather than generatively modeling future frames, FLARE aligns the model with target future embeddings. We find that adding FLARE both improves policy performance and unlocks the ability to learn from human videos.

### Training

We trained GR00T N1.5 for 250K steps on 1K H100 GPUs with global batch size 16384. As in N1, we used AdamW with cosine learning rate schedule with warmup ratio 0.05. We used FLARE loss coefficient 0.2 for both pretraining and posttraining.

Our pretraining mixture included internal GR-1 data, OpenXE, simulated GR-1 (a.k.a. DexMG), neural trajectories from [DreamGen](https://research.nvidia.com/labs/gear/dreamgen), and AgiBot-Beta:

![Pretraining data distribution](https://research.nvidia.com/labs/gear/n1_5/train_data_mix.svg)

Distribution of training data in GR00T N1.5 pretraining.

## Experimental Results

### Architecture validation

In order to tune the model architecture for N1.5, we trained policies from scratch on two sim robot benchmarks requiring language following: [Language Table](https://interactive-language.github.io/) and a set of five simulated GR-1 tasks requring language ("Sim GR-1 Language"). We find that the N1.5 architecture achieves significantly higher success rates on both benchmarks, indicating stronger language-conditioned control ability.

| Benchmark | GR00T N1 (scratch) | GR00T N1.5 (scratch) |
| --- | --- | --- |
| Language table | 52.8% | **93.2%** |
| Sim GR-1 Language | 36.4% | **54.4%** |

### Data-limited post-training in simulated environments

Following the GR00T N1 evaluation protocol, we evaluate N1.5's performance in data-limited post-training. In the case of Sim GR-1, we can evaluate both fewshot and 0-shot, since the the pretraining mixture includes other Sim GR-1 tasks with the same embodiment. We find that N1.5 is significantly better in the very low data regime (0-shot and 30 demos).

| Simulation Benchmark | GR00T N1 | GR00T N1.5 |
| --- | --- | --- |
| RoboCasa, 30 Demos per Task | 17.4 | **47.5** |
| Sim GR-1, 0-shot | 39.6 | **43.9** |
| SimGR-1, 30 Demos per Task | 43.2 | **47.4** |

### Real GR-1 language following

We add a simple language following task to the real GR-1 evaluation: two fruits are on a table and the robot is asked to place one of them onto a plate. The initial position of the target fruit is sampled to be closer to either the left or the right hand with 50% probability.

| Setting | GR00T N1 | GR00T N1.5 |
| --- | --- | --- |
| Language following rate | 46.6% | **93.3%** |
| Overall success rate | 43.3% | **83.0%** |

We find that N1.5 significantly improves over N1 in terms of its ability to follow language commands on the real GR-1 robot. Although both policies consistently pick and place *some* fruit onto the plate, N1.5 has a much higher language following rate, leading to a higher overall success rate.

### Learning to manipulate novel objects from human ego videos

To evaluate the model's generalization ability, we evaluate pick and place performance using a set of 10 novel objects not seen during pretraining.

![Novel objects](https://research.nvidia.com/labs/gear/n1_5/novel_objects.png)

Novel Objects

As shown in the [FLARE project](https://research.nvidia.com/labs/gear/flare), future latent representation alignment enables learning directly from human ego videos. This allows learning to manipulate novel objects from human videos and minimal robot demonstrations. Using N1.5, we found that this also works zero-shot.

Demonstrations using a novel object, captured from a GoPro (left) and the GR-1 robot (right).

| Setting | GR00T N1 | GR00T N1.5 |
| --- | --- | --- |
| 0-shot | 0% | **15.0%** |
| FLARE post-trained on human videos including novel objects | \- | **55.0%** |

Novel object generalization performance. We observe that N1.5 both performs better 0-shot, and also benefits from co-training with human videos.

### Generalization to novel behaviors using Neural Trajectories

To generalize beyond the teleoperation data and enable humanoid robots to learn new tasks in new environments, we use [DreamGen](https://research.nvidia.com/labs/gear/dreamgen) to generate synthetic robot data for training.

DreamGen 4-step pipeline.

Through the DreamGen pipeline, we show that GR00T N1.5 can achieve non-trivial results on 12 new verbs (see the [DreamGen](https://research.nvidia.com/labs/gear/dreamgen) blog post for task details), which were added to the pretraining data through our pipeline. GR00T N1 showed only weak generalization to new verbs, only repeating the tasks contained in pretraining (e.g., pick and place). We find that GR00T N1.5 achieved a 38.3% success rate across 12 DreamGen tasks, versus 13.1% for GR00T N1. Although these new verbs can be considered "zero-shot" in the sense that we never collected teleoperation data for these tasks, we still train explicitly on them via DreamGen trajectories; leaving full zero-shot verb and environment generalization to future work.

### Post-training on Unitree G1

We post-train GR00T N1 and N1.5 on 1K teleoperation episodes collected on the Unitree G1 robot. As in the GR-1 language following experiment, we initialize the scene with one target object and one distractor object, with the target object having equal probability of being closer to either the left or the right hand. We observe that the post-trained GR00T N1.5 achieves much higher success rate than N1 for previously seen objects (toy fruits seen in the GR-1 pretraining corpus), and also shows generalization to various previously unseen objects.

| Model | GR00T N1, 1K Demos | GR00T N1.5, 1K Demos | GR00T N1.5, 1K Demos |
| --- | --- | --- | --- |
| Task | Place 1 of 2 fruits onto plate; 4 total fruits | Place 1 of 2 fruits onto plate; 4 total fruits | Place 1 of 2 objects onto plate; 5 novel objects |
| Scene | ![](https://research.nvidia.com/labs/gear/n1_5/g1_2of4.jpg) | ![](https://research.nvidia.com/labs/gear/n1_5/g1_2of4.jpg) | ![](https://research.nvidia.com/labs/gear/n1_5/g1_2of5.jpg) |
| Success rate | 44.0% | **98.8%** | 84.2% |

(Left Image) Prompt: Place the bag of chips to the pink plate; (Right Image) Prompt: Place the soap to the blue plate.

## Discussion

Overall, we see that GR00T-N1.5 is a significant improvement over GR00T-N1. It achieves higher success rate, can use more diverse data sources, and has significantly improved language following capabilities. We attribute these improvements to the improved grounding capabilities, usage of the FLARE loss and the diverse data from DreamGen. The model will be open-sourced shortly, and we hope practitioners will observe improved results when finetuning GR00T-N1.5 on their own robots.

Foundation ModelHumanoid RobotVLMLanguage Following

Authors (alphabetical):

Johan Bjorck,Valts Blukis,Fernando Castañeda,Nikita Cherniadev,Xingye Da,Runyu Ding,Linxi "Jim" Fan,Yu Fang,Dieter Fox,Fengyuan Hu,Spencer Huang,Joel Jang,Xiaowei Jiang,Kaushil Kundalia,Jan Kautz,Zhiqi Li,Kevin Lin,Zongyu Lin,Loic Magne,Yunze Man,Ajay Mandlekar,Avnish Narayan,Soroush Nasiriany,Scott Reed,You Liang Tan,Guanzhi Wang,Jing Wang,Qi Wang,Shihao Wang,Jiannan Xiang,Yuqi Xie,Yinzhen Xu,Seonghyeon Ye,Zhiding Yu,Yizhou Zhao,Zhe Zhang,Ruijie Zheng,Yuke Zhu

Acknowledgements:

We thank various members of NVIDIA who have contributed to data curation, robot system development, testing GR00T N1.6, and advising. Members includes: