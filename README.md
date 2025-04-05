# 图像分析工作流 - LangChainVision

基于 LangGraph、LangChain和PyTorch的图像分析工作流系统，提供通用图像处理和遥感图像处理功能。用户可以通过Web界面上传图像，并根据自然语言需求自动选择合适的分析任务。

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [系统要求](#系统要求)
- [安装依赖](#安装依赖)
- [Ollama模型配置](#Ollama模型配置)
- [使用方法](#使用方法)
- [工作流程概述](#工作流程概述)
- [常见问题](#常见问题)
- [代码贡献](#代码贡献)

## 项目简介

LangChainVision是一个智能图像分析系统，通过LangGraph编排不同的处理步骤，包括需求分析、图像输入、预处理、模型选择、推理和结果输出。系统能够根据用户的自然语言需求自动选择适当的分析任务，支持多种视觉AI能力。

主要功能：

1. **需求分析**：使用LLM自动理解用户的需求并选择合适的分析任务
2. **图像分类**：识别图像中的主要对象或场景类别
3. **物体检测与标注**：检测、定位并标注图像中的多个对象
4. **图像描述**：提供图像的详细文本描述
5. **图像增强**：提升图像质量和分辨率
6. **遥感图像分析**：专业的遥感图像处理能力

## 系统架构

该项目采用MCP（模块-组件-包）架构，以增强代码的可维护性和可扩展性：

```
src/
├── agents/           # 智能体模块 
│   ├── routing.py    # 路由函数
│   └── workflow.py   # 工作流构建
├── config/           # 配置模块
│   └── settings.py   # 全局设置
├── models/           # 模型定义
├── processors/       # 处理器模块
│   ├── analysis_processor.py    # 需求分析
│   ├── image_processor.py       # 图像处理
│   ├── model_processor.py       # 模型推理
│   └── output_processor.py      # 结果处理
├── providers/        # 服务提供者
│   └── model_provider.py        # 模型提供
├── schemas/          # 数据模型
│   └── state.py      # 状态定义
└── utils/            # 工具函数
    ├── image_utils.py           # 图像工具
    └── model_utils.py           # 模型工具
```

系统使用LangGraph状态图来编排工作流程，实现了一个灵活、可扩展的流程控制机制：

![系统架构图](./static/images/architecture.png)

## 系统要求

- Python 3.9+
- CUDA支持的GPU (推荐用于YOLOv5和SwinIR模型)
- Ollama服务 (用于需求分析和图像描述)
- 至少8GB RAM
- 约10GB磁盘空间 (用于存储模型)

## 安装依赖

首先，确保您的 Python 环境中已安装以下依赖：

```bash
pip install -r requirements.txt
```

## Ollama模型配置

该项目使用Ollama服务来加载和运行LLM模型。请按照以下步骤配置：

1. 从 [Ollama官网](https://ollama.ai/) 下载并安装Ollama
2. 启动Ollama服务
3. 下载必要的模型：

```bash
ollama pull phi4-mini:latest
ollama pull llava:13b
```

## 使用方法

### 使用Flask Web界面

1. 启动应用程序：

```bash
python app_new.py
```

2. 在浏览器中访问 `http://localhost:5000`
3. 选择"通用图像分析"或"遥感图像分析"
4. 上传图像并输入分析需求或选择任务类型
5. 查看分析结果

### 命令行使用

您也可以通过Python代码直接使用工作流：

```python
from src.agents.workflow import execute_workflow

# 执行工作流
result = execute_workflow(
    image_path="path/to/your/image.jpg",
    user_requirement="识别图像中的所有物体并标注它们的位置"
)

# 查看结果
print(f"任务类型: {result['task']}")
print(f"结果: {result['result_message']}")
```

## 工作流程概述

系统的基本工作流程如下：

1. **需求分析**：分析用户的自然语言需求，确定任务类型
2. **图像输入**：加载并验证图像
3. **图像预处理**：根据任务类型进行图像预处理
4. **模型选择**：根据任务类型选择合适的模型
5. **模型推理**：执行模型推理
6. **结果处理**：处理、格式化并返回结果

在每个步骤之间，系统都包含错误处理机制，确保即使在某个环节出现问题时，也能以优雅的方式处理并提供有用的反馈。

## 常见问题

**Q: 为什么会出现"Ollama服务未启动或不可用"的错误？**
A: 请确保已安装并启动Ollama服务，并下载了必要的模型（phi4-mini和llava）。

**Q: 系统支持哪些图像格式？**
A: 系统支持JPEG、PNG、TIFF等常见图像格式。

**Q: 如何添加新的分析任务？**
A: 在`src/config/settings.py`中添加新的任务类型常量，然后在相应的处理器中实现任务逻辑，最后在工作流中添加相应的节点和路由。

## 代码贡献

欢迎为本项目做出贡献，请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个Pull Request

---

> 注意：本项目依赖于多个开源项目和预训练模型，使用前请确保遵守相应的许可证要求。
