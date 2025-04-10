<img src="../imgs/header_bar.png" alt="OmniTool Header" width="100%">

# OmniTool

通过 OmniParser 和您选择的视觉模型来控制 Windows 11 虚拟机。

## 亮点：

1. **OmniParser V2** 比 V1 快 60%，现在能够理解各种操作系统、应用程序和应用内图标！
2. **OmniBox** 比其他用于代理测试的 Windows 虚拟机节省 50% 的磁盘空间，同时提供相同的计算机使用 API
3. **OmniTool** 开箱即支持以下视觉模型 - OpenAI (4o/o1/o3-mini)、DeepSeek (R1)、Qwen (2.5VL) 或 Anthropic Computer Use

## 概述

共有三个组件：

<table style="border-collapse: collapse; border: none;">
  <tr>
    <td style="border: none;"><img src="../imgs/omniparsericon.png" width="50"></td>
    <td style="border: none;"><strong>omniparserserver</strong></td>
    <td style="border: none;">运行 OmniParser V2 的 FastAPI 服务器。</td>
  </tr>
  <tr>
    <td style="border: none;"><img src="../imgs/omniboxicon.png" width="50"></td>
    <td style="border: none;"><strong>omnibox</strong></td>
    <td style="border: none;">在 Docker 容器中运行的 Windows 11 虚拟机。</td>
  </tr>
  <tr>
    <td style="border: none;"><img src="../imgs/gradioicon.png" width="50"></td>
    <td style="border: none;"><strong>gradio</strong></td>
    <td style="border: none;">用于提供命令并观察推理和 OmniBox 执行过程的用户界面</td>
  </tr>
</table>

## 展示视频
| OmniParser V2 | [观看视频](https://1drv.ms/v/c/650b027c18d5a573/EWXbVESKWo9Buu6OYCwg06wBeoM97C6EOTG6RjvWLEN1Qg?e=alnHGC) |
|--------------|------------------------------------------------------------------|
| OmniTool    | [观看视频](https://1drv.ms/v/c/650b027c18d5a573/EehZ7RzY69ZHn-MeQHrnnR4BCj3by-cLLpUVlxMjF4O65Q?e=8LxMgX) |


## 注意事项：

1. 虽然 **OmniParser V2** 可以在 CPU 上运行，但如果你想在 GPU 机器上快速运行，我们已将其分离出来
2. **OmniBox** Windows 11 虚拟机 Docker 依赖于 KVM，因此只能在 Windows 和 Linux 上快速运行。这可以在 CPU 机器上运行（不需要 GPU）。
3. Gradio UI 也可以在 CPU 机器上运行。我们建议在同一台 CPU 机器上运行 **omnibox** 和 **gradio**，在 GPU 服务器上运行 **omniparserserver**。

## 安装步骤

1. **omniparserserver**：

   a. 如果你已经有 OmniParser 的 conda 环境，可以直接使用。否则请按照以下步骤创建一个

   b. 通过 `conda --version` 确保已安装 conda，或从 [Anaconda 网站](https://www.anaconda.com/download/success) 安装

   c. 使用 `cd OmniParser` 导航到仓库根目录

   d. 使用 `conda create -n "omni" python==3.12` 创建 conda python 环境

   e. 使用 `conda activate omni` 设置要使用的 python 环境

   f. 使用 `pip install -r requirements.txt` 安装依赖项

   g. 如果你已经有 conda 环境，可以从这里继续。

   h. 确保你已经在 weights 文件夹中下载了 V2 权重（**确保 caption weights 文件夹名为 icon_caption_florence**）。如果没有，使用以下命令下载：
   ```
   rm -rf weights/icon_detect weights/icon_caption weights/icon_caption_florence 
   for folder in icon_caption icon_detect; do huggingface-cli download microsoft/OmniParser-v2.0 --local-dir weights --repo-type model --include "$folder/*"; done
   mv weights/icon_caption weights/icon_caption_florence
   ```

   h. 使用 `cd OmniParser/omnitool/omniparserserver` 导航到服务器目录

   i. 使用 `python -m omniparserserver` 启动服务器

2. **omnibox**：

   a. 确保你有 30GB 的剩余空间（ISO 文件 5GB，Docker 容器 400MB，存储文件夹 20GB）

   b. 安装 Docker Desktop

   c. 访问 [Microsoft Evaluation Center](https://info.microsoft.com/ww-landing-windows-11-enterprise.html)，接受服务条款，下载 **Windows 11 Enterprise Evaluation（90 天试用版，英语，美国）** ISO 文件 [~6GB]。将文件重命名为 `custom.iso` 并复制到目录 `OmniParser/omnitool/omnibox/vm/win11iso`

   d. 使用 `cd OmniParser/omnitool/omnibox/scripts` 导航到虚拟机管理脚本目录

   e. 使用 `./manage_vm.sh create` 构建 Docker 容器 [400MB] 并将 ISO 安装到存储文件夹 [20GB]。安装过程如下图所示，根据下载速度将耗时 20-90 分钟（通常约 60 分钟）。完成后，终端将显示 `VM + server is up and running!`。你可以通过 NoVNC 查看器（http://localhost:8006/vnc.html?view_only=1&autoconnect=1&resize=scale）查看桌面，观察正在安装的应用程序。设置完成后，NoVNC 查看器中显示的终端窗口将不会出现在桌面上。如果你能看到它，请等待，不要点击任何地方！
![image](https://github.com/user-attachments/assets/6bd18f81-18e2-4bc5-9170-293a6699481d)

   f. 首次创建后，VM 状态将保存在 `vm/win11storage` 中。之后你可以使用 `./manage_vm.sh start` 和 `./manage_vm.sh stop` 管理虚拟机。要删除虚拟机，使用 `./manage_vm.sh delete` 并删除 `OmniParser/omnitool/omnibox/vm/win11storage` 目录。

3. **gradio**：

   a. 使用 `cd OmniParser/omnitool/gradio` 导航到 gradio 目录

   b. 使用 `conda activate omni` 确保已激活 conda python 环境

   c. 使用 `python app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000` 启动服务器

   d. 打开终端输出中的 URL，设置你的 API 密钥，开始使用 AI 代理！

## 常见安装错误
### OmniBox 安装时间过长
如果你的网速较慢，想要一个预装应用较少的最小 VM，可以在首次创建容器和 VM 时注释掉 [此文件](https://github.com/microsoft/OmniParser/blob/master/omnitool/omnibox/vm/win11setup/setupscripts/setup.ps1) 中第 57 至 350 行，这些行定义了要安装的所有应用程序。确保在创建 VM 时按照下一节的出厂重置说明清除任何之前的 omnibox 设置。

### 验证错误：Windows Host 未响应
如果在点击提交按钮后在 Gradio 中收到此错误，表明 VM 中运行的接收 Gradio 命令并移动鼠标/键盘的服务器不可用。你可以通过运行 `curl http://localhost:5000/probe` 验证这一点。确保你的 `omnibox` 已完全设置完成（不应再有终端窗口）。有关时间，请参阅 omnibox 部分。如果你已设置好 omnibox，可能需要稍等片刻。

如果等待 10 分钟仍不起作用，尝试使用脚本命令停止（`./manage_vm.sh stop`）并启动（`./manage_vm.sh start`）你的 omnibox VM。

如果仍不起作用，删除你的 VM（`./manage_vm.sh delete`），保留存储文件夹，然后再次运行创建。由于使用现有存储文件夹，过程会很快。

最后，如果仍然不起作用，你想将 VM 重置为出厂设置（创建新 VM）：
1. 运行 `./manage_vm.sh delete`
2. 删除 `vm/win11storage` 文件夹
3. 运行 `./manage_vm.sh create`

### libpaddle：找不到指定的模块
OmniParser 使用的 OCR 库是 Paddle，它在 Windows 上依赖于 C++ Redistributable。如果你使用 Windows，确保已安装它，然后重新运行安装 requirements.txt。更多详情请参见 [这里](https://github.com/microsoft/OmniParser/issues/140#issuecomment-2670619168)。

## 风险和缓解措施
为了与微软 AI 原则和负责任的 AI 实践保持一致，我们通过使用负责任的 AI 数据训练图标描述模型来进行风险缓解，这有助于模型尽可能避免推断图标图像中可能出现的个人敏感属性（如种族、宗教等）。同时，我们鼓励用户仅将 OmniParser 应用于不包含有害/暴力内容的屏幕截图。对于 OmniTool，我们使用微软威胁建模工具进行威胁模型分析。我们建议人类保持在循环中，以最大限度地降低风险。


## 致谢 
感谢在我们代码开发过程中提供的宝贵资源：[Claude Computer Use](https://github.com/anthropics/anthropic-quickstarts/blob/main/computer-use-demo/README.md)、[OS World](https://github.com/xlang-ai/OSWorld)、[Windows Agent Arena](https://github.com/microsoft/WindowsAgentArena) 和 [computer_use_ootb](https://github.com/showlab/computer_use_ootb)。
我们感谢 Francesco Bonacci、Jianwei Yang、Dillon DuPont、Yue Wu、Anh Nguyen 提供的有用建议和反馈。
特别感谢 @keyserjaya 提供的 omnibox 安装截图。
