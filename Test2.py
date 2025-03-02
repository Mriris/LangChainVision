import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.models as models
from typing import Dict, Any
from langgraph.graph import StateGraph, END
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from langchain_ollama import OllamaLLM
import io


# 定义状态对象，用于在节点间传递数据
class ImageState(Dict[str, Any]):
    image_path: str
    image: np.ndarray
    task: str
    model: Any
    output: Any
    user_requirement: str
    analysis_result: str


# 需求分析节点
def analysis_node(state: ImageState) -> ImageState:
    analysis_model = OllamaLLM(model="phi4-mini:latest", base_url="http://localhost:11434")
    user_requirement = state["user_requirement"]

    # 优化后的提示
    prompt = (
        "You are an assistant that determines the appropriate image analysis task based on user requirements. "
        "The tasks are: "
        "- Classification: Identify the main object or category in the image (e.g., 'What is this animal?'). "
        "- Annotation: Detect and locate multiple objects in the image (e.g., 'Find all objects and their positions'). "
        "- Interpretation: Provide a detailed description of the image (e.g., 'Describe what is happening in the image'). "
        "Examples: "
        "- 'Identify the main object' -> Classification "
        "- 'Describe the image' -> Interpretation "
        "- 'Detect all objects' -> Annotation "
        "Analyze the following requirement and respond with only the task name: {user_requirement}"
    ).format(user_requirement=user_requirement)

    # 调用模型并打印分析结果以调试
    analysis_result = analysis_model.invoke(prompt)
    state["analysis_result"] = analysis_result.strip().lower()
    print(f"分析结果: {state['analysis_result']}")  # 调试用

    # 严格匹配任务名称
    if state["analysis_result"] == "classification":
        state["task"] = "classification"
    elif state["analysis_result"] == "annotation":
        state["task"] = "annotation"
    elif state["analysis_result"] == "interpretation":
        state["task"] = "interpretation"
    else:
        state["task"] = "interpretation"  # 默认任务
        print("警告：分析结果未明确指定任务，默认设置为 interpretation")

    return state


# 图像输入节点
def image_input_node(state: ImageState) -> ImageState:
    image_path = r"C:\0Program\Python\LangChainVision\example\2011_000006.jpg"
    if not image_path.endswith(('.jpg', '.png', '.jpeg')):
        raise ValueError("请提供有效的图像文件（.jpg, .png, .jpeg）")
    state["image_path"] = image_path
    return state


# 图像预处理节点
def preprocess_node(state: ImageState) -> ImageState:
    image = cv2.imread(state["image_path"])
    task = state["task"]
    if task == "classification":
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
    elif task == "annotation":
        pass
    elif task == "interpretation":
        image = cv2.resize(image, (384, 384))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
    state["image"] = image
    return state


# 模型选择节点
def model_selection_node(state: ImageState) -> ImageState:
    task = state["task"]
    if task == "classification":
        from torchvision.models.resnet import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.eval()
    elif task == "annotation":
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    elif task == "interpretation":
        model = OllamaLLM(model="llava:13b", base_url="http://localhost:11434")
    else:
        raise ValueError("不支持的任务类型")
    state["model"] = model
    return state


# 模型推理节点
def inference_node(state: ImageState) -> ImageState:
    task = state["task"]
    model = state["model"]
    image = state["image"]
    if task == "classification":
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
        state["output"] = output.argmax(dim=1).item()
    elif task == "annotation":
        results = model(image)
        state["output"] = results.xyxy[0].cpu().numpy()
    elif task == "interpretation":
        # 将图像从 (C, H, W) 转换为 (H, W, C)
        image_hwc = image.transpose(1, 2, 0)
        # 缩放回 [0, 255] 并转换为 uint8
        image_scaled = (image_hwc * 255).astype(np.uint8)
        # 执行颜色转换
        image_rgb = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # 将 PIL Image 转换为 bytes
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()

        # 调用模型，使用 invoke 方法和 bytes 格式的图像
        prompt = "Describe this image in Chinese."
        response = model.invoke(prompt, images=[image_bytes])
        state["output"] = response.strip()
    return state


# 结果输出节点
def output_node(state: ImageState) -> ImageState:
    task = state["task"]
    output = state["output"]
    if task == "classification":
        with open("imagenet_classes.txt", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"分类结果：{classes[output]}")
    elif task == "annotation":
        image = state["image"].copy()
        boxes = state["output"]
        model = state["model"]
        class_names = model.names
        for box in boxes:
            x_min, y_min, x_max, y_max, conf, class_id = box
            class_id = int(class_id)
            label = f"{class_names[class_id]}: {conf:.2f}"
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.savefig("annotated_image.jpg")
        print(f"标注结果：{output}")
    elif task == "interpretation":
        print(f"图像描述：{output}")
    return state


# 创建工作流程
workflow = StateGraph(ImageState)
workflow.add_node("analysis", analysis_node)
workflow.add_node("image_input", image_input_node)
workflow.add_node("preprocess", preprocess_node)
workflow.add_node("model_selection", model_selection_node)
workflow.add_node("inference", inference_node)
workflow.add_node("display_output", output_node)

# 定义边
workflow.add_edge("analysis", "image_input")
workflow.add_edge("image_input", "preprocess")
workflow.add_edge("preprocess", "model_selection")
workflow.add_edge("model_selection", "inference")
workflow.add_edge("inference", "display_output")
workflow.add_edge("display_output", END)

# 设置入口点
workflow.set_entry_point("analysis")

# 编译工作流程
app = workflow.compile()

# 在运行工作流前捕获用户需求
user_requirement = input("请输入您的需求：")
initial_state = {"task": None, "user_requirement": user_requirement}
app.invoke(initial_state)
