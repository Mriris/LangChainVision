# OmniParser: 基于纯视觉的GUI代理屏幕解析工具

<p align="center">
  <img src="imgs/logo.png" alt="Logo">
</p>
<!-- <a href="https://trendshift.io/repositories/12975" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12975" alt="microsoft%2FOmniParser | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a> -->

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[项目页面](https://microsoft.github.io/OmniParser/)] [[V2 博客文章](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)] [[模型 V2](https://huggingface.co/microsoft/OmniParser-v2.0)] [[模型 V1.5](https://huggingface.co/microsoft/OmniParser)] [[HuggingFace 空间演示](https://huggingface.co/spaces/microsoft/OmniParser-v2)]

**OmniParser** 是一种全面的方法，用于将用户界面截图解析为结构化且易于理解的元素，这显著增强了 GPT-4V 生成准确对应于界面相应区域的操作的能力。

## 新闻
- [2025/3] 我们支持轨迹的本地日志记录，这样您可以使用 OmniParser+OmniTool 为您所在领域喜欢的代理构建训练数据管道。[文档编写中]
- [2025/3] 我们正在逐步添加多代理协调功能并改进 OmniTool 的用户界面，以提供更好的体验。
- [2025/2] 我们发布了 OmniParser V2 [检查点](https://huggingface.co/microsoft/OmniParser-v2.0)。[观看视频](https://1drv.ms/v/c/650b027c18d5a573/EWXbVESKWo9Buu6OYCwg06wBeoM97C6EOTG6RjvWLEN1Qg?e=alnHGC)
- [2025/2] 我们推出了 OmniTool：通过 OmniParser 和您选择的视觉模型控制 Windows 11 虚拟机。OmniTool 开箱即支持以下大型语言模型 - OpenAI (4o/o1/o3-mini)、DeepSeek (R1)、Qwen (2.5VL) 或 Anthropic Computer Use。[观看视频](https://1drv.ms/v/c/650b027c18d5a573/EehZ7RzY69ZHn-MeQHrnnR4BCj3by-cLLpUVlxMjF4O65Q?e=8LxMgX)
- [2025/1] V2 即将推出。我们在新的定位基准 [Screen Spot Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/main) 上使用 OmniParser v2（即将发布）取得了 39.5% 的最新技术水平！在[这里](https://github.com/microsoft/OmniParser/tree/master/docs/Evaluation.md)阅读更多详情。
- [2024/11] 我们发布了更新版本 OmniParser V1.5，其特点是 1) 更精细/小图标检测，2) 预测每个屏幕元素是否可交互。示例见 demo.ipynb。
- [2024/10] OmniParser 是 huggingface 模型中心的第一大热门模型（从 2024/10/29 开始）。
- [2024/10] 欢迎查看我们在 [huggingface space](https://huggingface.co/spaces/microsoft/OmniParser) 上的演示！（敬请期待 OmniParser + Claude Computer Use）
- [2024/10] 交互区域检测模型和图标功能描述模型已发布！[Hugginface 模型](https://huggingface.co/microsoft/OmniParser)
- [2024/09] OmniParser 在 [Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/) 上取得了最佳性能！

## 安装
首先克隆仓库，然后安装环境：
```python
cd OmniParser
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

确保您在 weights 文件夹中下载了 V2 权重（确保 caption weights 文件夹名为 icon_caption_florence）。如果没有，使用以下命令下载：
```
   # 将模型检查点下载到本地目录 OmniParser/weights/
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
   mv weights/icon_caption weights/icon_caption_florence
```

<!-- ## [deprecated]
Then download the model ckpts files in: https://huggingface.co/microsoft/OmniParser, and put them under weights/, default folder structure is: weights/icon_detect, weights/icon_caption_florence, weights/icon_caption_blip2. 

For v1: 
convert the safetensor to .pt file. 
```python
python weights/convert_safetensor_to_pt.py

For v1.5: 
download 'model_v1_5.pt' from https://huggingface.co/microsoft/OmniParser/tree/main/icon_detect_v1_5, make a new dir: weights/icon_detect_v1_5, and put it inside the folder. No weight conversion is needed. 
``` -->

## 示例：
我们在 demo.ipynb 中整理了一些简单示例。

## Gradio 演示
要运行 gradio 演示，只需运行：
```python
python gradio_demo.py
```

## 模型权重许可
对于 huggingface 模型中心上的模型检查点，请注意 icon_detect 模型采用 AGPL 许可证，因为这是继承自原始 yolo 模型的许可证。而 icon_caption_blip2 和 icon_caption_florence 采用 MIT 许可证。请参阅每个模型文件夹中的 LICENSE 文件：https://huggingface.co/microsoft/OmniParser。

## 📚 引用
我们的技术报告可以在[这里](https://arxiv.org/abs/2408.00203)找到。
如果您发现我们的工作有用，请考虑引用我们的工作：
```
@misc{lu2024omniparserpurevisionbased,
      title={OmniParser for Pure Vision Based GUI Agent}, 
      author={Yadong Lu and Jianwei Yang and Yelong Shen and Ahmed Awadallah},
      year={2024},
      eprint={2408.00203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00203}, 
}
```
