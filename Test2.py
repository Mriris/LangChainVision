import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.models as models
from typing import Dict, Any, TypedDict, List, Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from langchain_ollama import OllamaLLM
import io
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()


# 使用TypedDict定义状态类型
class ImageState(TypedDict):
    image_path: str
    image: Any
    task: Literal["classification", "annotation", "interpretation", ""]
    model: Any
    output: Any
    user_requirement: str
    analysis_result: str
    error: str
    status: str
    history: List[Dict[str, Any]]


# 结果模型结构
class AnalysisResult:
    def __init__(self, task, confidence, reasoning):
        self.task = task
        self.confidence = confidence
        self.reasoning = reasoning

    def model_dump(self):
        return {
            "task": self.task,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


# 定义条件函数，用于路由决策
def should_retry(state: ImageState) -> Literal["retry", "continue"]:
    """决定是否需要重试分析"""
    if state.get("error") and "analysis" in state.get("error", ""):
        return "retry"
    return "continue"


def route_by_task(state: ImageState) -> str:
    """根据任务类型路由到不同处理流程"""
    if state.get("error"):
        return "error_handler"
    return state.get("task", "interpretation")  # 默认为interpretation


# 需求分析节点
def analysis_node(state: ImageState) -> ImageState:
    # 为状态添加历史记录
    if "history" not in state:
        state["history"] = []

    try:
        analysis_model = OllamaLLM(
            model="phi4-mini:latest",
            base_url="http://localhost:11434",
            temperature=0.1  # 降低温度以获得更确定性的结果
        )
        user_requirement = state["user_requirement"]

        # 更结构化的提示词，引导模型返回JSON格式
        prompt = """
        你是一个专业的计算机视觉任务分析专家。请分析用户的需求并确定最合适的图像分析任务。

        可选任务:
        1. classification: 识别图像中的主要对象或类别（例如："这是什么动物？"）
        2. annotation: 检测并定位图像中的多个对象（例如："找出所有物体及其位置"）
        3. interpretation: 提供图像的详细描述（例如："描述图像中发生的事情"）

        请分析以下用户需求: "{user_requirement}"

        以JSON格式返回你的分析结果：
        ```json
        {{
            "task": "任务名称(classification/annotation/interpretation)",
            "confidence": 0.xx,
            "reasoning": "你的分析理由"
        }}
        ```
        只返回有效的JSON，不要包含其他文本。
        """.format(user_requirement=user_requirement)

        # 调用模型并解析结果
        analysis_response = analysis_model.invoke(prompt)

        # 提取JSON部分
        json_match = re.search(r'```json\s*(.*?)\s*```', analysis_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = analysis_response.strip()

        # 清理可能的额外字符
        json_str = re.sub(r'[^\x00-\x7F]+', '', json_str)

        try:
            result = json.loads(json_str)
            # 创建结果对象
            parsed_result = AnalysisResult(
                task=result.get("task", "interpretation"),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "")
            )

            state["analysis_result"] = parsed_result.model_dump()
            state["task"] = parsed_result.task.lower()
            print(f"分析结果: {state['task']} (置信度: {parsed_result.confidence})")

            # 记录到历史
            state["history"].append({
                "step": "analysis",
                "result": parsed_result.model_dump()
            })

        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始响应: {analysis_response}")
            state["task"] = "interpretation"  # 默认任务
            state["error"] = f"analysis_json_error: {str(e)}"

    except Exception as e:
        state["error"] = f"analysis_error: {str(e)}"
        state["task"] = "interpretation"  # 默认任务

    return state


# 错误处理节点
def error_handler_node(state: ImageState) -> ImageState:
    error = state.get("error", "未知错误")
    print(f"处理错误: {error}")

    # 清除错误并设置默认值
    state["error"] = ""
    if "task" not in state or not state["task"]:
        state["task"] = "interpretation"

    state["status"] = "已从错误恢复，使用默认任务"

    # 记录到历史
    if "history" in state:
        state["history"].append({
            "step": "error_handler",
            "error": error,
            "recovery": "使用默认任务"
        })

    return state


# 图像输入节点
def image_input_node(state: ImageState) -> ImageState:
    try:
        # 支持自定义图像路径或使用默认路径
        image_path = state.get("image_path", "")
        if not image_path:
            image_path = r"C:\0Program\Python\LangChainVision\example\2011_000006.jpg"

        # 验证图像文件
        if not os.path.exists(image_path):
            raise ValueError(f"图像文件不存在: {image_path}")

        if not image_path.endswith(('.jpg', '.png', '.jpeg')):
            raise ValueError("请提供有效的图像文件（.jpg, .png, .jpeg）")

        state["image_path"] = image_path

        # 记录到历史
        if "history" in state:
            state["history"].append({
                "step": "image_input",
                "path": image_path
            })

    except Exception as e:
        state["error"] = f"image_input_error: {str(e)}"

    return state


# 图像预处理节点
def preprocess_node(state: ImageState) -> ImageState:
    try:
        image_path = state["image_path"]
        task = state["task"]

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 根据任务进行预处理
        if task == "classification":
            image = cv2.resize(image, (224, 224))
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))  # CHW格式
        elif task == "annotation":
            # 保持原始图像用于YOLOv5
            pass
        elif task == "interpretation":
            image = cv2.resize(image, (384, 384))
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))  # CHW格式

        state["image"] = image

        # 记录到历史
        if "history" in state:
            state["history"].append({
                "step": "preprocess",
                "task": task,
                "image_shape": str(image.shape)
            })

    except Exception as e:
        state["error"] = f"preprocess_error: {str(e)}"

    return state


# 模型选择节点
def model_selection_node(state: ImageState) -> ImageState:
    try:
        task = state["task"]
        print(f"开始加载模型，任务类型: {task}")

        if task == "classification":
            from torchvision.models.resnet import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            model.eval()
            model_name = "ResNet50"
        elif task == "annotation":
            try:
                try:
                    # 首先尝试使用GPU
                    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                    model_name = "YOLOv5s (GPU)"
                except Exception as gpu_err:
                    print(f"YOLOv5 GPU加载错误: {str(gpu_err)}")
                    print("尝试使用CPU加载YOLOv5...")
                    # 如果失败，使用CPU
                    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True, device='cpu')
                    model_name = "YOLOv5s (CPU)"
                print(f"成功加载模型: {model_name}")
            except Exception as yolo_err:
                print(f"YOLOv5加载错误: {str(yolo_err)}")
                raise RuntimeError(f"无法加载YOLOv5模型: {str(yolo_err)}")
        elif task == "interpretation":
            try:
                model = OllamaLLM(
                    model="llava:13b",
                    base_url="http://localhost:11434",
                    temperature=0.7
                )
                # 测试模型连接
                test_result = model.invoke("测试连接")
                if not test_result:
                    raise RuntimeError("LLaVA模型返回空结果")
                model_name = "LLaVA-13B"
            except Exception as llm_err:
                print(f"LLM模型加载错误: {str(llm_err)}")
                print("请确保Ollama服务正在运行，并已下载llava:13b模型")
                raise RuntimeError(f"无法加载或连接LLM模型: {str(llm_err)}")
        else:
            raise ValueError(f"不支持的任务类型: {task}")

        state["model"] = model
        print(f"模型 '{model_name}' 加载成功")

        # 记录到历史
        if "history" in state:
            state["history"].append({
                "step": "model_selection",
                "task": task,
                "model": model_name
            })

    except Exception as e:
        error_msg = f"model_selection_error: {str(e)}"
        print(error_msg)
        state["error"] = error_msg
        # 为了防止后续步骤失败，给model设置一个占位符
        if task == "classification":
            # 创建一个简单的dummy模型
            class DummyModel:
                def __call__(self, x):
                    return torch.tensor([[1.0]])
            state["model"] = DummyModel()
        elif task == "annotation":
            # 创建一个返回空检测结果的假YOLOv5模型
            class DummyYOLO:
                def __init__(self):
                    self.names = {0: "person", 1: "object"}
                
                def __call__(self, img):
                    class DummyResults:
                        def __init__(self):
                            self.xyxy = [torch.zeros((0, 6))]
                    return DummyResults()
            state["model"] = DummyYOLO()
        elif task == "interpretation":
            # 创建一个返回固定文本的假LLM
            class DummyLLM:
                def invoke(self, prompt, images=None):
                    return "无法加载LLM模型，请检查Ollama服务是否正在运行"
            state["model"] = DummyLLM()

    return state


# 模型推理节点
def inference_node(state: ImageState) -> ImageState:
    try:
        task = state["task"]
        model = state["model"]
        image = state["image"]

        if task == "classification":
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = model(image_tensor)
            state["output"] = output.argmax(dim=1).item()

        elif task == "annotation":
            original_image = cv2.imread(state["image_path"])
            if original_image is None:
                raise ValueError(f"无法读取图像文件: {state['image_path']}")
                
            results = model(original_image)  # YOLOv5需要原始图像
            state["output"] = results.xyxy[0].cpu().numpy()
            state["original_image"] = original_image

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

            # 调用模型
            prompt = """
            请详细描述这张图片，包括:
            1. 图片中的主要对象和场景
            2. 物体的位置关系
            3. 颜色、形状和纹理特征
            4. 可能的场景背景和故事

            请用中文回答，尽量详细但不要臆测不存在的内容。
            """
            response = model.invoke(prompt, images=[image_bytes])
            state["output"] = response.strip()
        else:
            raise ValueError(f"不支持的任务类型: {task}")

        # 验证输出是否存在且有效
        if "output" not in state or state["output"] is None:
            raise ValueError(f"模型未能生成有效输出")

        # 记录到历史
        if "history" in state:
            state["history"].append({
                "step": "inference",
                "task": task,
                "output_type": str(type(state["output"]))
            })

    except Exception as e:
        error_msg = f"inference_error: {str(e)}"
        print(error_msg)
        state["error"] = error_msg
        # 确保有默认输出，防止后续步骤出错
        if "output" not in state:
            if task == "classification":
                state["output"] = 0  # 默认类别ID
            elif task == "annotation":
                state["output"] = np.array([])  # 空检测结果
            elif task == "interpretation":
                state["output"] = "无法生成图像描述"  # 默认描述

    return state


# 结果输出节点
def output_node(state: ImageState) -> ImageState:
    try:
        task = state["task"]
        
        # 检查输出是否存在
        if "output" not in state:
            raise KeyError("state中缺少'output'键，可能是推理阶段未成功完成")
        
        output = state["output"]

        if task == "classification":
            try:
                with open("imagenet_classes.txt", "r") as f:
                    classes = [line.strip() for line in f.readlines()]
                class_name = classes[output]
            except Exception as e:
                class_name = f"类别ID: {output}"
                print(f"警告: 无法加载类别名称 - {str(e)}")

            result = f"分类结果：{class_name}"
            print(result)
            state["result_message"] = result

        elif task == "annotation":
            original_image = state.get("original_image")
            if original_image is None:
                original_image = cv2.imread(state["image_path"])
                if original_image is None:
                    raise ValueError(f"无法读取图像: {state['image_path']}")

            boxes = state["output"]
            model = state["model"]
            class_names = model.names

            # 创建注释副本
            annotated_image = original_image.copy()

            detection_results = []
            for box in boxes:
                x_min, y_min, x_max, y_max, conf, class_id = box
                class_id = int(class_id)
                label = f"{class_names[class_id]}: {conf:.2f}"
                detection_results.append({
                    "class": class_names[class_id],
                    "confidence": float(conf),
                    "box": [float(x_min), float(y_min), float(x_max), float(y_max)]
                })

                # 在图像上绘制边界框
                cv2.rectangle(
                    annotated_image,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    annotated_image,
                    label,
                    (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # 保存注释图像
            output_path = "annotated_image.jpg"
            image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 8))
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.savefig(output_path)
            plt.close()

            state["detection_results"] = detection_results
            state["annotated_image_path"] = output_path
            
            # 直接设置result_message
            state["result_message"] = f"标注结果已保存至 {output_path}, 检测到 {len(detection_results)} 个对象"
            print(f"设置结果消息: {state['result_message']}")
            
            # 作为备份，在状态集合中再添加一个副本
            state["annotations"] = {
                "message": state["result_message"],
                "path": output_path,
                "detections": detection_results
            }

        elif task == "interpretation":
            result = f"图像描述：\n{output}"
            print(result)
            state["result_message"] = result

        # 设置完成状态
        state["status"] = "completed"

        # 记录到历史
        if "history" in state:
            state["history"].append({
                "step": "output",
                "task": task,
                "result": state.get("result_message", "")
            })

        # 检查并打印最终结果，方便调试
        final_message = state.get("result_message", "")
        print(f"输出节点结束，result_message: {final_message}")
        
        # 最后再检查一次result_message是否存在
        if not state.get("result_message") and "output" in state:
            print("警告: 最终结果消息为空，使用默认消息")
            if task == "classification":
                state["result_message"] = f"分类结果 (ID): {output}"
            elif task == "annotation":
                if "annotations" in state and state["annotations"].get("message"):
                    state["result_message"] = state["annotations"]["message"]
                else:
                    state["result_message"] = f"标注完成，原始结果: {len(boxes)} 个物体"
            elif task == "interpretation":
                state["result_message"] = f"图像描述 (截断): {str(output)[:100]}..."

    except KeyError as e:
        state["error"] = f"output_error: {str(e)}"
        print(f"输出节点错误(键): {str(e)}")
        # 提供一个默认的结果消息
        state["result_message"] = "处理过程中出现错误，未能生成有效结果"
        state["status"] = "error"
    except Exception as e:
        state["error"] = f"output_error: {str(e)}"
        print(f"输出节点错误: {str(e)}")
        state["result_message"] = f"处理错误: {str(e)}"
        state["status"] = "error"

    return state


# 创建工作流程
workflow = StateGraph(ImageState)

# 添加节点
workflow.add_node("analysis", analysis_node)
workflow.add_node("image_input", image_input_node)
workflow.add_node("preprocess", preprocess_node)
workflow.add_node("model_selection", model_selection_node)
workflow.add_node("inference", inference_node)
workflow.add_node("display_output", output_node)
workflow.add_node("error_handler", error_handler_node)

# 添加条件路由
# 从分析到后续步骤的条件路由
workflow.add_conditional_edges(
    "analysis",
    should_retry,
    {
        "retry": "analysis",  # 如果分析失败，重试
        "continue": "image_input"  # 否则继续
    }
)

# 定义标准边
workflow.add_edge("image_input", "preprocess")
workflow.add_edge("preprocess", "model_selection")
workflow.add_edge("model_selection", "inference")
workflow.add_edge("inference", "display_output")
workflow.add_edge("display_output", END)
workflow.add_edge("error_handler", "image_input")  # 错误处理后重新开始

# 设置入口点
workflow.set_entry_point("analysis")

# 编译工作流程
app = workflow.compile()


# 主函数
def main():
    # 在运行工作流前捕获用户需求
    user_requirement = input("请输入您的图像分析需求：")

    # 初始化状态
    initial_state = {
        "user_requirement": user_requirement,
        "task": "",
        "error": "",
        "status": "started",
        "history": []
    }

    # 运行工作流
    final_state = app.invoke(initial_state)

    # 输出结果摘要
    print("\n" + "=" * 50)
    print("工作流程执行完成")
    print(f"执行任务: {final_state.get('task', '未知')}")
    print(f"状态: {final_state.get('status', '未知')}")

    if final_state.get("result_message"):
        print("\n结果:")
        print(final_state.get("result_message"))

    if final_state.get("error"):
        print(f"\n遇到错误: {final_state.get('error')}")

    print("=" * 50)

    return final_state


if __name__ == "__main__":
    main()
