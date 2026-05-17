---
title: "starVLA/starVLA: StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing"
source: "https://github.com/starVLA/starVLA"
author:
published:
created: 2026-04-19
description: "StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing - starVLA/starVLA"
tags:
  - "clippings"
---
## StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing

An open-source research platform for integrating and exploring cutting-edge technologies for generalist robots.

[![Technical Report](https://camo.githubusercontent.com/550fe963dc0bbb9e64e835487a0d7710eba61697facdee2e34fad638ed2c3af3/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f61725869762d323630342e30353031342d7265643f7374796c653d666f722d7468652d6261646765266c6f676f3d6172786976)](https://arxiv.org/abs/2604.05014) [![WeChat](https://camo.githubusercontent.com/4d2fda4df29bb13298d9f77199f20825a3c5d8a358871e26a1b86de0ab1dc967/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5765436861742de58aa0e585a5e8aea8e8aebae7bea42d627269676874677265656e3f7374796c653d666f722d7468652d6261646765266c6f676f3d776563686174)](https://github.com/starVLA/starVLA/issues/64#issuecomment-3715403845)

> **📢 Citation Update:** Our technical report is now on arXiv ([2604.05014](https://arxiv.org/abs/2604.05014)). We kindly invite you to use the [updated BibTeX](#citation) for any ongoing or future citations. If you have already cited StarVLA in a previous version of your work, we would greatly appreciate it if you could update the citation entry in your camera-ready or future revisions. Thank you for your understanding and support! 🙏

---

In StarVLA (also a pun on "start VLA" ), each functional component (model, data, trainer, config, evaluation, etc.) follows a top-down, intuitive separation and high-cohesion, low-coupling principle, enabling plug-and-play design, rapid prototyping, and independent debugging.

## News

> **⚠️ Branch notice:** The `starVLA_dev` branch is where we actively merge new features and may be temporarily unstable. For verified results, use the stable `starVLA` branch. Thanks to StarVLA's low-coupling design, switching between branches is painless. We encourage trying `starVLA_dev` and welcome PRs if you spot any issues!

> **💡 Tip:** Files under any `**/bar/` directory are git-ignored, so you can place your custom scripts there (e.g., `examples/LIBERO/train_files/bar/my_train.sh`) without polluting the repo.

**\[2026/04/09\]** 🔜 🚀 unified **multi-benchmark co-training** example (combining LIBERO, SimplerEnv, RoboTwin, VLA-Arena, etc.) is coming soon. Stay tuned!

**\[2026/04/09\]** 🎯 Thanks to the [RLinf](https://rlinf.readthedocs.io/) team, StarVLA now supports **RL post-training**! Check out the [StarVLA × RLinf tutorial](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/starvla.html) to get started.

**\[2026/04/09\]** 🔥 **WM4A (World Model for Action)** is now integrated! Use pretrained video-generation DiT models (Cosmos-Predict2, Wan2.2) as backbones for action prediction. See [docs/WM4A.md](https://github.com/starVLA/starVLA/blob/starVLA_dev/docs/WM4A.md) for architecture details and training instructions.

**\[2026/03/29\]** 🔥 Thanks to the [ABot-M0](https://github.com/amap-cvlab/ABot-Manipulation) team for providing the [pre-trained weights](https://www.modelscope.cn/models/amap_cvlab/ABot-M0-Pretrain). For `Qwen3-VL 4B`, you can reload the `qwen_vl_interface` module in various frameworks!

**\[2026/03/19\]** 🔥 StarVLA now provides a complete real-robot development case with [Franka robot examples](https://github.com/starVLA/starVLA/pull/198)!

**\[2026/03/03\]** 🔥 We now support [**Qwen3.5** as a backbone for VLA](https://github.com/starVLA/starVLA/pull/172) — the fastest integration in the community ⚡ With more model size options: **0.8B, 2B, 4B, and 9B**! Build your VLA flexibly on top of native multimodal models!

**\[2026/01/29\]** 🔥 StarVLA [Training Efficiency Report](https://github.com/starVLA/starVLA/issues/158) & [Training Curves](https://github.com/starVLA/starVLA/issues/68) released! Training configs and efficiency benchmarks for community reference.

**\[2026/01/29\]** Calvin benchmark experiments were conducted by the UNT team. For inquiries, please contact Zhijie Song ([1600013008@pku.edu.cn](mailto:1600013008@pku.edu.cn)) or Feng Yan ([bphengyan@163.com](mailto:bphengyan@163.com)).

**\[2025/12/25\]** We've simultaneously established pipelines for [Behavior-1K](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/Behavior), [RoboTwin 2.0](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/Robotwin), and CALVIN. We'd love to collaborate and share baseline results for more benchmarks with the community!

**Prior Timeline**

**\[2025/12/25\]** We've released RoboCasa evaluation support, which was trained **without pretraining and reached SOTA performance**. Check out more details in [examples/Robocasa\_tabletop](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/Robocasa_tabletop).

**\[2025/12/15\]** Completed a release regression check to ensure the public code runs smoothly. Routine updates—including recent support for the LeRobot dataset v3.0 and DeepSpeed ZeRO-3—will continue to appear in the [🚧 Daily Development Log](https://github.com/starVLA/starVLA/issues/64#issue-3727060165).

**\[2025/12/09\]** Became the first open-source repository to support training with [train your vlm](https://github.com/starVLA/starVLA/blob/starVLA_dev/starVLA/training/train_starvlm.py), [train your vla](https://github.com/starVLA/starVLA/blob/starVLA_dev/starVLA/training/train_starvla.py), and [train your vla with vlm](https://github.com/starVLA/starVLA/blob/starVLA_dev/starVLA/training/train_starvla_cotrain.py). Check out how to co-train your VLA with multimodal data in [examples/CoTrainVLM](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/CoTrainVLM/README.md).

**\[2025/11/12\]** We now support [Florence-2](https://github.com/anyantudre/Florence-2-Vision-Language-Model) as a smaller VLM for resource-constrained development. StarVLA can now run on a single A100 GPU. See the [🚀Train with a smaller VLM](https://github.com/starVLA/starVLA/blob/starVLA_dev/docs/faq.md#how-to-train-with-a-smaller-vlm) section for more details.

**\[2025/10/30\]:** We released the LIBERO Training & Evaluation README. Results are very promising. More details are in [examples/LIBERO](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/LIBERO).

**\[2025/10/25\]:** We fixed several script links and so everything is smoother now. Thanks to the community for the feedback.

## Overview and Key Features

[![Overview of the StarVLA framework](https://github.com/starVLA/starVLA/raw/starVLA_dev/assets/starVLA_overview.png)](https://github.com/starVLA/starVLA/blob/starVLA_dev/assets/starVLA_overview.png) *Overview of the StarVLA framework. We present a unified and modular pipeline that connects heterogeneous data sources, pluggable dataloaders, and flexible data representations with a standardized model forwarding interface. The framework supports diverse vision-language foundation models and VLA architectures, enabling end-to-end training and deployment.*

**Various VLA Frameworks**

All variants share the same data interface and infrastructure; only the action head differs.

- **StarVLA-FAST**: Autoregressive discrete action tokens via a fast tokenizer (à la π₀-fast).
- **StarVLA-OFT**: Parallel continuous action decoding with an MLP head (à la OpenVLA-OFT/EO).
- **StarVLA-PI**: Flow-Matching action expert for diffusion-based continuous actions (à la π₀).
- **StarVLA-GR00T**: Dual-system architecture — VLM as System 2, Flow-Matching as System 1 (à la GR00T).

[![StarVLA variant architectures](https://github.com/starVLA/starVLA/raw/starVLA_dev/assets/starvla_variants.png)](https://github.com/starVLA/starVLA/blob/starVLA_dev/assets/starvla_variants.png)

**Various Training Recipes**

Every recipe is paradigm-agnostic and applies uniformly to all supported frameworks.

- Supervised fine-tuning (SFT)
- Multimodal Multi-objectives Co-Training
- Cross-embodiment Co-Training
- Reinforcement Learning Adaptation
**Broad Benchmark Integration**

Achieve **state-of-the-art (SOTA) performance** on a variety of benchmarks, as follows:

- **SimplerEnV**
- **LIBERO**
- **LIBERO-plus**
- **Robocasa**
- **RoboTwin**
- **BEHAVIOR**
- **SO101**
- **Calvin** \*See details in [`examples/calvin`](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/calvin)
- **RLBench**

---

## 🎒 Quick Start

> **📖 New to StarVLA?** Check out our step-by-step [**Quick Start Guide**](https://github.com/starVLA/starVLA/blob/starVLA_dev/docs/starVLA_guideline.md) — a complete walkthrough from installation to training to evaluation using the LIBERO benchmark.

---

## Benchmark Results

**Results on LIBERO**

[![LIBERO modules](https://github.com/starVLA/starVLA/raw/starVLA_dev/assets/starvla_LIBERO.png)](https://github.com/starVLA/starVLA/blob/starVLA_dev/assets/starvla_LIBERO.png)

**Results on SimplerEnv**

[![SimplerEnv modules](https://github.com/starVLA/starVLA/raw/starVLA_dev/assets/starvla_simpleEnv.png)](https://github.com/starVLA/starVLA/blob/starVLA_dev/assets/starvla_simpleEnv.png)

**Results on RoboCasa GR1**

[![RoboCasa modules](https://github.com/starVLA/starVLA/raw/starVLA_dev/assets/stavla_RoboCasa.png)](https://github.com/starVLA/starVLA/blob/starVLA_dev/assets/stavla_RoboCasa.png)

**Results on Calvin\_D\_D**

[![Calvin_D_D modules](https://github.com/starVLA/starVLA/raw/starVLA_dev/assets/calvin.png)](https://github.com/starVLA/starVLA/blob/starVLA_dev/assets/calvin.png)

We have more results for RoboCasa, RoboTwin 2.0, Behavior-1k, Calvin. See our [🍀 Overleaf](https://www.overleaf.com/read/qqtwrnprctkf#d5bdce), which continuously presents our real-time experimental results.

---

## Model Zoo

See the full list of released models and checkpoints in [docs/model\_zoo.md](https://github.com/starVLA/starVLA/blob/starVLA_dev/docs/model_zoo.md).

---

## Start Building Your VLA Like Lego!

👇 StarVLA achieves "Lego-like" development via the following designs:

**1\. Smoke test any submodule**

StarVLA emphasizes a modular model design. Each major framework file can be run standalone for rapid debugging and smoke-testing your code. For example:

```
# model
python starVLA/model/framework/QwenOFT.py --config_yaml starvla_cotrain_oxe.yaml
# dataloader
python starVLA/dataloader/lerobot_datasets.py --config_yaml starvla_cotrain_oxe.yaml
```

Note: `starVLA/model/framework/yourframework.py` is the single external API surface of the model; it should mirror (be structurally isomorphic to) the framework diagram in your paper.

**2\. Explicit model boundaries**

StarVLA follows top‑down decomposition and the principle of high cohesion & low coupling.

For example:

- Dataloader
	- Returns a raw, model‑agnostic dict only; no model‑specific preprocessing (e.g., tokenizer, image encoding).
		- A single sample should include (add/remove as needed):
		- image: list\[PIL.Image\] | np.ndarray
				- lang: str
				- action: np.ndarray\[T, action\_dim\]
				- state: Optional\[np.ndarray\[..., state\_dim\]\]

Both `framework.forward()` and `framework.predict_action()` operate directly on raw inputs, keeping train/test boundaries explicit and easy to hack.

**3\. Flexible configuration system**

StarVLA uses a single global configuration object Parameters are passed primarily via extensible dicts, allowing overrides and controlled redundancy.

---

## FAQ

See [docs/faq.md](https://github.com/starVLA/starVLA/blob/starVLA_dev/docs/faq.md) for common questions on configuration, freezing, learning rates, checkpointing, smaller VLMs, and more.

## Contributing

StarVLA was co-founded by [Jinhui Ye](https://jhuiye.com/) and [Weiyu Guo](https://weiyuguo.com/), and has been gradually expanded by community contributors. Community contributors are the driving force behind StarVLA's growing ecosystem. We deeply appreciate every PR, bug fix, and piece of feedback from the open-source community — your efforts keep StarVLA evolving rapidly. A full, continuously updated contributor list is maintained at [starvla.github.io/contributors](https://starvla.github.io/contributors).

Thanks to all the people who have contributed to StarVLA:

[![](https://camo.githubusercontent.com/ee7ac2ba6e639f8924daea5df37460c93b292f0f73432a4101ae33162411d008/68747470733a2f2f636f6e747269622e726f636b732f696d6167653f7265706f3d73746172564c412f73746172564c41266d61783d31303026636f6c756d6e733d3135)](https://github.com/starVLA/starVLA/graphs/contributors)

See [docs/CONTRIBUTING.md](https://github.com/starVLA/starVLA/blob/starVLA_dev/docs/CONTRIBUTING.md) for guidelines on reporting bugs, proposing features, and submitting PRs.

### Projects Based on StarVLA

**NeuroVLA**: [*A Brain-like Embodied Intelligence for Fluid and Fast Reflexive Robotics Control*](https://github.com/guoweiyu/NeuroVLA)

**PhysBrain**: [*Human Egocentric Data as a Bridge from Vision Language Models to Physical Intelligence*](https://zgc-embodyai.github.io/PhysBrain)

**TwinBrainVLA**: [*TwinBrainVLA: Unleashing the Potential of Generalist VLMs for Embodied Tasks via Asymmetric Mixture-of-Transformers*](https://github.com/ZGC-EmbodyAI/TwinBrainVLA)

**LangForce**: [*LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries*](https://github.com/ZGC-EmbodyAI/LangForce)

Examples:

```
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml  \
  --num_processes 8 \
  starVLA/training/train_internvla.py \
  --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
  --framework.qwenvl.base_vlm Qwen/Qwen2.5-VL-7B-Instruct \ # override framework choice
  --framework.qwenvl.base_vlm Qwen/Qwen2.5-VL-7B-Instruct \ # override framework choice
  --framework.action_model.new_module ${module_name} \ # plug-in a new module to action model
```

⚠️: `framework.action_model.new_module` only adds to the global config; its behavior is on your framework.

**Q: Can I freeze the VLM via parameters?**

A: Yes. StarVLA uses a regex / name list to control freezing. Example:

```
--trainer.freeze_modules "qwen_vl_interface.model.model.visual,dino_encoder" \
```

Tips: You can `print(your_model)` first to check the relative paths of your modules and list them as comma-separated values. (implementation in `TrainerUtils.freeze_backbones`.)

**Q: Can I set different learning rates for different modules?**

A: Yes, starVLA also uses name: value dict to control learning group. Config example:

```
trainer:
  learning_rate:
    base: 1e-05      # other modules
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
```

(Also referenced in `trainer_tools.build_param_lr_groups`.)

**Q: Can I resume training from a checkpoint?**

A: Yes, somehow can. Specify the latest checkpoint path in `config.yaml`, e.g.:

```
trainer:
  pretrained_checkpoint: path_to_steps_10000.pt
  reload_modules: "action_model"
```

Empty `reload_modules` means full load all model. However, starVLA does not save `optimizer state`. It requires a lot of memory/disk and bring limited benefit.

**🚀 Train with a smaller VLM**
```
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $SLURM_PROCID \
  --num_machines $SLURM_NNODES \
  --num_processes=${TOTAL_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm microsoft/Florence-2-large \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project your_project \
  --wandb_entity your_name
```

Note: To ensure better compatibility with already released checkpoints, we are continuing to use `--framework.qwenvl`. This parameter will be unified in the next release.

StarVLA is released under the MIT License, which permits commercial use, modification, distribution, and private use. Rebases are allowed for forks and feature branches; when rebasing from upstream StarVLA, use descriptive commit messages (e.g., "chore: rebase from StarVLA") and keep at least the two latest upstream commits as separate. See [License](https://github.com/starVLA/starVLA/blob/starVLA_dev/LICENSE) for details.

```
@article{community2026starvla,
  title={StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing},
  ={Community, StarVLA},
  journal={arXiv preprint arXiv:2604.05014},
  year={2026}
}
```

## Acknowledgements

This project draws inspiration and references from several notable open-source initiatives, including:

- [LeRobot](https://github.com/huggingface/lerobot)
- [GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/main)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Qwen-VL](https://github.com/QwenLM/Qwen3-VL/tree/main)
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [ABot-Manipulation](https://github.com/amap-cvlab/ABot-Manipulation)

The codebase was originally forked from [InternVLA-M1](https://github.com/InternRobotics/InternVLA-M1).

## Star History

Here's how our community has grown over time:

[![Star History Chart](https://camo.githubusercontent.com/621936e52cab463050ad4ae175ebd4fb963d0d5fd0275b4f1b7bb635f2151be1/68747470733a2f2f6170692e737461722d686973746f72792e636f6d2f7376673f7265706f733d73746172564c412f73746172564c4126747970653d64617465266c6567656e643d626f74746f6d2d7269676874)](https://www.star-history.com/#starVLA/starVLA&type=date&legend=bottom-right)