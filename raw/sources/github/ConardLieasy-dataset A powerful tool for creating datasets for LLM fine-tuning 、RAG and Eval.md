---
title: "ConardLi/easy-dataset: A powerful tool for creating datasets for LLM fine-tuning 、RAG and Eval"
source: https://github.com/ConardLi/easy-dataset/blob/main/README.zh-CN.md
author:
published:
created: 2026-04-28
description: A powerful tool for creating datasets for LLM fine-tuning 、RAG and Eval - ConardLi/easy-dataset
tags:
  - clippings
---
[![](https://github.com/ConardLi/easy-dataset/raw/main/public/imgs/bg2.png)](https://github.com/ConardLi/easy-dataset/blob/main/public/imgs/bg2.png)

[![ConardLi%2Feasy-dataset | Trendshift](https://camo.githubusercontent.com/86b0e84acc4351f8857c009a77c700960d9b2c994fe10c95b801ae746e9dcb0f/68747470733a2f2f7472656e6473686966742e696f2f6170692f62616467652f7265706f7369746f726965732f3133393434)](https://trendshift.io/repositories/13944)

**一个强大的大型语言模型微调数据集创建工具**

[简体中文](https://github.com/ConardLi/easy-dataset/blob/main/README.zh-CN.md) | [English](https://github.com/ConardLi/easy-dataset/blob/main/README.md) | [Türkçe](https://github.com/ConardLi/easy-dataset/blob/main/README.tr.md)

[功能特点](#%E5%8A%9F%E8%83%BD%E7%89%B9%E7%82%B9) • [快速开始](#%E6%9C%AC%E5%9C%B0%E8%BF%90%E8%A1%8C) • [使用文档](https://docs.easy-dataset.com/) • [贡献](#%E8%B4%A1%E7%8C%AE) • [许可证](#%E8%AE%B8%E5%8F%AF%E8%AF%81)

如果喜欢本项目，请给本项目留下 Star⭐️，或者请作者喝杯咖啡呀 => [打赏作者](https://github.com/ConardLi/easy-dataset/blob/main/public/imgs/aw.jpg) ❤️！

## 概述

Easy Dataset 是一个专为创建大型语言模型数据集而设计的应用程序。它提供了直观的界面，内置了强大的文档解析工具、智能分割算法、数据清洗和数据增强能力，可以将各种格式的领域文献转化为高质量结构化数据集，可用于模型微调、RAG、模型效果评估等场景。

[![Easy Dataset 产品架构图](https://github.com/ConardLi/easy-dataset/raw/main/public/imgs/arc3.png)](https://github.com/ConardLi/easy-dataset/blob/main/public/imgs/arc3.png)

## 新闻

🎉🎉 Easy Dataset 1.7.0 版本上线全新的评估能力，你可以轻松将领域文献转换为评估数据集（测试集），并且可以自动执行多维度评估任务，另外还配备人工盲测系统，可以轻松助你完成垂直领域模型评估、模型微调后效果评估、RAG 召回率评估等需求，使用教程： [https://www.bilibili.com/video/BV1CRrVB7Eb4/](https://www.bilibili.com/video/BV1CRrVB7Eb4/)

## 功能特点

### 📄 文档处理与数据生成

- **智能文档处理** ：支持 PDF、Markdown、DOCX、TXT、EPUB 等多种格式智能识别和处理
- **智能文本分割** ：支持多种智能文本分割算法（Markdown 结构、递归分隔符、固定长度、代码智能分块等），支持自定义可视化分段
- **智能问题生成** ：从每个文本片段中自动提取相关问题，支持问题模板和批量生成
- **领域标签树** ：基于文档目录智能构建全局领域标签树，具备全局理解和自动打标能力
- **答案生成** ：使用 LLM API 为每个问题生成全面的答案和思维链（COT），支持 AI 智能优化
- **数据清洗** ：智能清洗文本块内容，去除噪音数据，提升数据质量

### 🔄 多种数据集类型

- **单轮问答数据集** ：标准的问答对格式，适合基础微调
- **多轮对话数据集** ：支持自定义角色和场景的多轮对话格式
- **图片问答数据集** ：基于图片生成视觉问答数据，支持多种导入方式（目录、PDF、压缩包）
- **数据蒸馏** ：无需上传文档，直接从领域主题自动生成标签树和问题

### 📊 模型评估体系

- **评估数据集** ：支持生成判断题、单选题、多选题、简答题、开放题等多种题型的评估测试集
- **模型自动评估** ：使用教师模型（Judge Model）自动评估模型回答质量，支持自定义评分规则
- **人工盲测 (Arena)** ：双盲对比两个模型的回答质量，消除偏见进行公正评判
- **AI 质量评估** ：对生成的数据集进行自动质量评分和筛选

### 🛠️ 高级功能

- **自定义提示词** ：项目级自定义各类提示词模板（问题生成、答案生成、数据清洗等）
- **GA 组合生成** ：文体-受众对生成，丰富数据多样性
- **任务管理中心** ：后台批量任务处理，支持任务监控和中断
- **资源监控看板** ：Token 消耗统计、调用次数追踪、模型性能分析
- **模型测试 Playground** ：支持最多 3 个模型同时对比测试

### 📤 导出与集成

- **多种导出格式** ：支持 Alpaca、ShareGPT、Multilingual-Thinking 等格式，JSON/JSONL 文件类型
- **平衡导出** ：按标签配置导出数量，实现数据集均衡
- **LLaMA Factory 集成** ：一键生成 LLaMA Factory 配置文件
- **Hugging Face 上传** ：直接将数据集上传至 Hugging Face Hub

### 🤖 模型支持

- **广泛的模型兼容** ：兼容所有遵循 OpenAI 格式的 LLM API
- **多提供商支持** ：OpenAI、MiniMax、Ollama（本地模型）、智谱 AI、阿里百炼、OpenRouter 等
- **视觉模型** ：支持 Gemini、Claude 等视觉模型用于 PDF 解析和图片问答

### 🌐 用户体验

- **用户友好界面** ：为技术和非技术用户设计的现代化直观 UI
- **多语言支持** ：完整的中英文界面支持
- **数据集广场** ：发现和探索各种公开数据集资源
- **桌面客户端** ：提供 Windows、macOS、Linux 桌面应用

## 快速演示

ed3.mp4<video src="https://private-user-images.githubusercontent.com/30708545/444058272-6ddb1225-3d1b-4695-90cd-aa4cb01376a8.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzczNDIyMDMsIm5iZiI6MTc3NzM0MTkwMywicGF0aCI6Ii8zMDcwODU0NS80NDQwNTgyNzItNmRkYjEyMjUtM2QxYi00Njk1LTkwY2QtYWE0Y2IwMTM3NmE4Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA0MjglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNDI4VDAyMDUwM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJkMWI1NzEyOTVkYzg4ZDAxOTAzZDUwNTAzYzc0OWI4NTFiMDlkMWEyM2U5OTRlYzYzMTM5NzczNDI1NzFlYzMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT12aWRlbyUyRm1wNCJ9.HOUAZG6NzXa7i5FLP3Y710XVh1T1KT87OnAQOhPSeOc" controls="controls"></video>

## 本地运行

### 下载客户端

<table><tbody><tr><td width="20%" align="center"><b>Windows</b></td><td width="30%" align="center" colspan="2"><b>MacOS</b></td><td width="20%" align="center"><b>Linux</b></td></tr><tr><td align="center"><a href="https://github.com/ConardLi/easy-dataset/releases/latest"><br><b>Setup.exe</b></a></td><td align="center"><a href="https://github.com/ConardLi/easy-dataset/releases/latest"><br><b>Intel</b></a></td><td align="center"><a href="https://github.com/ConardLi/easy-dataset/releases/latest"><br><b>M</b></a></td><td align="center"><a href="https://github.com/ConardLi/easy-dataset/releases/latest"><br><b>AppImage</b></a></td></tr></tbody></table>

### 使用 NPM 安装

1. 克隆仓库：
```
git clone https://github.com/ConardLi/easy-dataset.git
cd easy-dataset
```
1. 安装依赖：
```
npm install
```
1. 启动开发服务器：
```
npm run build

npm run start
```
1. 打开浏览器并访问 `http://localhost:1717`

### 使用官方 Docker 镜像

1. 克隆仓库：
```
git clone https://github.com/ConardLi/easy-dataset.git
cd easy-dataset
```
1. 更改 `docker-compose.yml` 文件：
```
services:
  easy-dataset:
    image: ghcr.io/conardli/easy-dataset
    container_name: easy-dataset
    ports:
      - '1717:1717'
    volumes:
      - ./local-db:/app/local-db
      - ./prisma:/app/prisma
    restart: unless-stopped
```

> **注意：** 建议直接使用当前代码仓库目录下的 `local-db` 和 `prisma` 文件夹作为挂载路径，这样可以和 NPM 启动时的数据库路径保持一致。

> **注意：** 数据库文件会在首次启动时自动初始化，无需手动执行 `npm run db:push` 。

1. 使用 docker-compose 启动
```
docker-compose up -d
```
1. 打开浏览器并访问 `http://localhost:1717`

### 使用本地 Dockerfile 构建

如果你想自行构建镜像，可以使用项目根目录中的 Dockerfile：

1. 克隆仓库：
```
git clone https://github.com/ConardLi/easy-dataset.git
cd easy-dataset
```
1. 构建 Docker 镜像：
```
docker build -t easy-dataset .
```
1. 运行容器：
```
docker run -d \
  -p 1717:1717 \
  -v ./local-db:/app/local-db \
  -v ./prisma:/app/prisma \
  --name easy-dataset \
  easy-dataset
```

> **注意：** 建议直接使用当前代码仓库目录下的 `local-db` 和 `prisma` 文件夹作为挂载路径，这样可以和 NPM 启动时的数据库路径保持一致。

> **注意：** 数据库文件会在首次启动时自动初始化，无需手动执行 `npm run db:push` 。

1. 打开浏览器，访问 `http://localhost:1717`

## 文档

- 有关所有功能和 API 的详细文档，请访问我们的 [文档站点](https://docs.easy-dataset.com/)
- 查看本项目的演示视频： [Easy Dataset 演示视频](https://www.bilibili.com/video/BV1y8QpYGE57/)
- 查看本项目的论文： [Easy Dataset: A Unified and Extensible Framework for Synthesizing LLM Fine-Tuning Data from Unstructured Documents](https://arxiv.org/abs/2507.04009v1)

## 社区教程

- [使用 Easy Dataset 完成测试集生成和模型评估](https://www.bilibili.com/video/BV1CRrVB7Eb4/)
- [Easy Dataset × LLaMA Factory: 让大模型高效学习领域知识](https://buaa-act.feishu.cn/wiki/KY9xwTGs1iqHrRkjXBwcZP9WnL9)
- [Easy Dataset 使用实战: 如何构建高质量数据集？](https://www.bilibili.com/video/BV1MRMnz1EGW)
- [Easy Dataset 1.4 重点功能更新解读](https://www.bilibili.com/video/BV1fyJhzHEb7/)
- [Easy Dataset 1.6 重点功能更新解读](https://www.bilibili.com/video/BV1Rq1hBtEJa/)
- [大模型微调数据集: 基础知识科普](https://docs.easy-dataset.com/zhi-shi-ke-pu)
- [实战案例1：生成汽车图片识别数据集](https://docs.easy-dataset.com/bo-ke/shi-zhan-an-li/an-li-1-sheng-cheng-qi-che-tu-pian-shi-bie-shu-ju-ji)
- [实战案例2：评论情感分类数据集](https://docs.easy-dataset.com/bo-ke/shi-zhan-an-li/an-li-2-ping-lun-qing-gan-fen-lei-shu-ju-ji)
- [实战案例3：物理学多轮对话数据集](https://docs.easy-dataset.com/bo-ke/shi-zhan-an-li/an-li-3-wu-li-xue-duo-lun-dui-hua-shu-ju-ji)
- [实战案例4：AI 智能体安全数据集](https://docs.easy-dataset.com/bo-ke/shi-zhan-an-li/an-li-4ai-zhi-neng-ti-an-quan-shu-ju-ji)
- [实战案例5：从图文 PPT 中提取数据集](https://docs.easy-dataset.com/bo-ke/shi-zhan-an-li/an-li-5-cong-tu-wen-ppt-zhong-ti-qu-shu-ju-ji)

## 贡献

我们欢迎社区的贡献！如果您想为 Easy Dataset 做出贡献，请按照以下步骤操作：

1. Fork 仓库
2. 创建新分支（ `git checkout -b feature/amazing-feature` ）
3. 进行更改
4. 提交更改（ `git commit -m '添加一些惊人的功能'` ）
5. 推送到分支（ `git push origin feature/amazing-feature` ）
6. 打开 Pull Request（提交至 DEV 分支）

请确保适当更新测试并遵守现有的编码风格。

## 加交流群 & 联系作者

[https://docs.easy-dataset.com/geng-duo/lian-xi-wo-men](https://docs.easy-dataset.com/geng-duo/lian-xi-wo-men)

## 许可证

本项目采用 AGPL 3.0 许可证 - 有关详细信息，请参阅 [LICENSE](https://github.com/ConardLi/easy-dataset/blob/main/LICENSE) 文件。

## 引用

如果您觉得此项目有帮助，请考虑以下列格式引用

```
@misc{miao2025easydataset,
  title={Easy Dataset: A Unified and Extensible Framework for Synthesizing LLM Fine-Tuning Data from Unstructured Documents},
  ={Ziyang Miao and Qiyu Sun and Jingyuan Wang and Yuchen Gong and Yaowei Zheng and Shiqi Li and Richong Zhang},
  year={2025},
  eprint={2507.04009},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2507.04009}
}
```

## Star History

[![Star History Chart](https://camo.githubusercontent.com/815547b5635414191d3d12224c1949c59864585339fbea7678f23e2aae857843/68747470733a2f2f6170692e737461722d686973746f72792e636f6d2f7376673f7265706f733d436f6e6172644c692f656173792d6461746173657426747970653d44617465)](https://www.star-history.com/#ConardLi/easy-dataset&Date)

<sub>由 <a href="https://github.com/ConardLi">ConardLi</a> 用 ❤️ 构建 • 关注我： <a href="https://github.com/ConardLi/easy-dataset/blob/main/public/imgs/weichat.jpg">公众号</a> ｜ <a href="https://space.bilibili.com/474921808">B站</a> ｜ <a href="https://juejin.cn/user/3949101466785709">掘金</a> ｜ <a href="https://www.zhihu.com/people/wen-ti-chao-ji-duo-de-xiao-qi">知乎</a> ｜ <a href="https://www.youtube.com/@garden-conard">Youtube</a></sub>