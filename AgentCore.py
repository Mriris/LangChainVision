import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.models as models
from typing import Dict, Any, TypedDict, List, Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import matplotlib
import time

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
    task: Literal["classification", "annotation", "interpretation", "enhancement", ""]
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
        4. enhancement: 提高图像质量、超分辨率或清晰度（例如："提高图像质量"、"增强图像清晰度"）

        请分析以下用户需求: "{user_requirement}"

        返回严格格式的JSON，必须使用英文引号和标点，不要使用中文引号或标点:
        {{
            "task": "任务名称(classification/annotation/interpretation/enhancement)",
            "confidence": 0.xx,
            "reasoning": "你的分析理由"
        }}
        
        不要返回markdown格式和其他任何文本，只返回纯JSON。避免使用中文标点如"，"、"："等，只使用英文标点。
        """.format(user_requirement=user_requirement)

        # 调用模型并解析结果
        analysis_response = analysis_model.invoke(prompt)
        print(f"原始分析响应: {analysis_response[:100]}...")  # 打印响应开头部分用于调试

        # 提取JSON部分
        json_match = re.search(r'```json\s*(.*?)\s*```', analysis_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            print("从markdown代码块提取JSON")
        else:
            # 尝试直接从响应中提取JSON对象
            json_start = analysis_response.find('{')
            json_end = analysis_response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = analysis_response[json_start:json_end]
                print("直接从文本提取JSON对象")
            else:
                json_str = analysis_response.strip()
                print("使用整个响应作为JSON")

        # 清理和修复JSON字符串
        # 1. 替换中文标点符号为英文标点符号
        punctuation_map = {
            '，': ',',
            '。': '.',
            '：': ':',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '！': '!',
            '？': '?',
            '（': '(',
            '）': ')',
            '【': '[',
            '】': ']',
            '、': ',',
            '；': ';'
        }
        for cn_punct, en_punct in punctuation_map.items():
            json_str = json_str.replace(cn_punct, en_punct)
        
        # 2. 清理非ASCII字符，但保留基本的中文文本
        json_str = re.sub(r'[^\x00-\x7F\u4e00-\u9fff]+', '', json_str)
        
        # 3. 确保JSON键使用双引号
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        print(f"清理后的JSON字符串: {json_str[:100]}...")  # 打印处理后的JSON字符串开头

        try:
            # 尝试解析JSON
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
            print(f"解析失败的JSON字符串: {json_str}")
            
            # 尝试通过正则表达式直接提取任务类型
            task_match = re.search(r'"task"\s*:\s*"([^"]+)"', json_str)
            if task_match:
                task = task_match.group(1).lower()
                print(f"通过正则表达式提取的任务: {task}")
                
                # 验证任务类型是否有效
                if task in ["classification", "annotation", "interpretation", "enhancement"]:
                    state["task"] = task
                    print(f"使用正则提取的任务类型: {task}")
                    
                    # 尝试提取置信度
                    confidence_match = re.search(r'"confidence"\s*:\s*(0\.\d+)', json_str)
                    confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                    
                    # 创建一个基本的结果对象
                    parsed_result = AnalysisResult(
                        task=task,
                        confidence=confidence,
                        reasoning="通过正则表达式从错误的JSON中提取"
                    )
                    
                    state["analysis_result"] = parsed_result.model_dump()
                    
                    # 记录到历史
                    state["history"].append({
                        "step": "analysis",
                        "result": parsed_result.model_dump()
                    })
                else:
                    # 无法从正则提取有效任务，使用默认值
                    state["task"] = "interpretation"  # 默认任务
                    print(f"无法提取有效任务类型，使用默认值: interpretation")
            else:
                # 从用户需求智能判断任务类型
                user_req = user_requirement.lower()
                if "什么" in user_req or "识别" in user_req or "类别" in user_req or "分类" in user_req:
                    task = "classification"
                elif "找出" in user_req or "检测" in user_req or "标记" in user_req or "标注" in user_req or "位置" in user_req:
                    task = "annotation"
                elif "描述" in user_req or "说明" in user_req or "讲解" in user_req or "解释" in user_req:
                    task = "interpretation"
                elif "增强" in user_req or "提高" in user_req or "改善" in user_req or "优化" in user_req or "清晰" in user_req:
                    task = "enhancement"
                else:
                    task = "interpretation"  # 默认任务
                
                state["task"] = task
                print(f"从用户需求智能判断任务类型: {task}")
                state["error"] = f"analysis_json_error: {str(e)}, 使用智能判断的任务类型"

        # 释放分析模型资源
        try:
            del analysis_model
            import gc
            gc.collect()
            # 尝试通知Ollama释放资源
            try:
                import requests
                requests.post('http://localhost:11434/api/cancel', json={})
                print("已请求Ollama释放phi4-mini分析模型资源")
            except:
                pass
        except Exception as cleanup_err:
            print(f"释放分析模型资源时出错: {str(cleanup_err)}")

    except Exception as e:
        state["error"] = f"analysis_error: {str(e)}"
        state["task"] = "interpretation"  # 默认任务
        print(f"分析过程出现错误: {str(e)}")

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
            # 注意：预处理后图像仍在CPU上作为NumPy数组
            # 会在inference_node中将其转换为适当设备上的张量
        elif task == "annotation":
            # 保持原始图像用于YOLOv5
            pass
        elif task == "enhancement":
            # 保持原始图像用于SwinIR
            state["original_image"] = image.copy()
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

        # 在加载新模型前先释放可能的旧模型资源
        try:
            if "model" in state and state["model"] is not None:
                del state["model"]
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("已释放先前加载的模型资源")
        except Exception as cleanup_err:
            print(f"释放旧模型资源时出错: {str(cleanup_err)}")

        if task == "classification":
            from torchvision.models.resnet import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            model.eval()
            model_name = "ResNet50"
            
            # 将模型移动到GPU如果可用
            if torch.cuda.is_available():
                model = model.cuda()
                print("已将ResNet模型移动到GPU")

        elif task == "annotation":
            try:
                try:
                    # 首先尝试使用GPU
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    print(f"YOLOv5将使用设备: {device}")
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
        elif task == "enhancement":
            try:
                # 导入必要的库
                import sys
                import os
                
                # SwinIR模型定义和加载
                try:
                    # 动态导入SwinIR模型
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if not os.path.exists(os.path.join(current_dir, "models")):
                        os.makedirs(os.path.join(current_dir, "models"))
                        
                    # 检查SwinIR模型文件
                    model_path = os.path.join(current_dir, "models", "swinir_model.py")
                    if not os.path.exists(model_path):
                        import urllib.request
                        # 使用代理下载模型定义文件
                        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
                        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
                        url = "https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/models/network_swinir.py"
                        urllib.request.urlretrieve(url, model_path)
                        print(f"已下载SwinIR模型定义文件到: {model_path}")
                    
                    # 将模型目录添加到系统路径
                    sys.path.append(os.path.join(current_dir, "models"))
                    from swinir_model import SwinIR
                    
                    # 创建SwinIR模型实例
                    model = SwinIR(
                        upscale=4,
                        img_size=64,
                        window_size=8,
                        img_range=1.,
                        depths=[6, 6, 6, 6, 6, 6],
                        embed_dim=180,
                        num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2,
                        upsampler='pixelshuffle',
                        resi_connection='1conv'
                    )
                    
                    # 检查并加载预训练的权重
                    weights_path = os.path.join(current_dir, "weights", "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
                    if os.path.exists(weights_path):
                        print(f"找到SwinIR预训练权重文件: {weights_path}")
                        # 确定使用的设备
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        print(f"将使用设备: {device}")
                        # 加载权重，允许自动选择设备
                        pretrained_model = torch.load(weights_path, map_location=device)
                        model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model else pretrained_model, strict=True)
                        model_name = "SwinIR (用于图像增强)"
                        print(f"成功加载SwinIR预训练权重")
                    else:
                        # 尝试加载其他可能存在的权重文件
                        alt_weights_path = os.path.join(current_dir, "weights", "SwinIR_model.pth")
                        if os.path.exists(alt_weights_path):
                            print(f"找到替代SwinIR权重文件: {alt_weights_path}")
                            # 确定使用的设备
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            print(f"将使用设备: {device}")
                            pretrained_model = torch.load(alt_weights_path, map_location=device)
                            model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model else pretrained_model, strict=False)
                            model_name = "SwinIR (使用替代权重)"
                            print(f"成功加载替代SwinIR权重")
                        else:
                            print(f"未找到SwinIR权重文件，使用未初始化模型")
                            model_name = "SwinIR (未初始化)"
                    
                    # 确保使用可用设备
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device)
                    model.eval()
                    print(f"SwinIR模型已加载到设备: {device}, 类型: {device.type}")
                    # 确认CUDA是否可用
                    if torch.cuda.is_available():
                        print(f"CUDA可用，使用GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        print("CUDA不可用，使用CPU模式")
                    
                except Exception as swinir_err:
                    print(f"SwinIR模型加载错误: {str(swinir_err)}")
                    # 备用方案：使用OpenCV的超分辨率模型
                    print("尝试使用OpenCV的EDSR模型...")
                    sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
                    edsr_path = os.path.join(current_dir, "weights", "EDSR_x4.pb")
                    if os.path.exists(edsr_path):
                        sr_model.readModel(edsr_path)
                        sr_model.setModel("edsr", 4)  # 4x超分辨率
                        model = sr_model
                        model_name = "EDSR (OpenCV超分辨率)"
                        print(f"成功加载EDSR模型")
                    else:
                        print(f"未找到EDSR模型文件: {edsr_path}")
                        # 最后的备用计划：检查是否有RealESRGAN模型
                        esrgan_path = os.path.join(current_dir, "weights", "RealESRGAN_x4plus.pth")
                        if os.path.exists(esrgan_path):
                            print(f"找到RealESRGAN模型，但尚未实现加载逻辑")
                            raise RuntimeError(f"暂不支持RealESRGAN")
                        else:
                            raise RuntimeError(f"未找到任何可用的图像增强模型")
                
            except Exception as e:
                print(f"图像增强模型加载错误: {str(e)}")
                raise RuntimeError(f"无法加载图像增强模型: {str(e)}")
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
        elif task == "enhancement":
            # 创建一个返回固定文本的假SwinIR
            class DummySwinIR:
                def invoke(self, prompt, images=None):
                    return "无法加载SwinIR模型，请检查Ollama服务是否正在运行"
            state["model"] = DummySwinIR()
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
            # 确保模型和输入张量在同一设备上
            device = next(model.parameters()).device
            print(f"模型设备: {device}")
            image_tensor = image_tensor.to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
            state["output"] = output.argmax(dim=1).item()
            # 释放不再需要的模型
            del model
            torch.cuda.empty_cache()

        elif task == "annotation":
            original_image = cv2.imread(state["image_path"])
            if original_image is None:
                raise ValueError(f"无法读取图像文件: {state['image_path']}")
                
            results = model(original_image)  # YOLOv5需要原始图像
            state["output"] = results.xyxy[0].cpu().numpy()
            state["original_image"] = original_image
            # 释放不再需要的YOLO模型
            del model
            torch.cuda.empty_cache()
            
        elif task == "enhancement":
            try:
                # 获取原始图像
                original_image = state.get("original_image")
                if original_image is None:
                    original_image = cv2.imread(state["image_path"])
                    if original_image is None:
                        raise ValueError(f"无法读取图像文件: {state['image_path']}")
                    state["original_image"] = original_image
                
                # 检查模型类型并进行相应处理
                if isinstance(model, cv2.dnn_superres.DnnSuperResImpl):
                    # 使用OpenCV的DNN超分辨率
                    enhanced_image = model.upsample(original_image)
                    state["output"] = enhanced_image
                else:
                    # 使用SwinIR或其他PyTorch模型
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    print(f"增强推理使用设备: {device}")
                    
                    # 预处理图像
                    img_bgr = original_image
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    
                    # 将图像转换为张量
                    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                    img_tensor = img_tensor.to(device)
                    
                    start_time = time.time()
                    # 执行推理
                    with torch.no_grad():
                        output = model(img_tensor)
                    
                    inference_time = time.time() - start_time
                    print(f"增强推理完成，耗时: {inference_time:.2f}秒")
                    
                    # 后处理
                    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
                    output = (output * 255.0).round().astype(np.uint8)
                    
                    # 转回BGR格式用于OpenCV
                    enhanced_image = cv2.cvtColor(output.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    state["output"] = enhanced_image
                
                # 释放增强模型资源
                del model
                torch.cuda.empty_cache()
                
                # 保存对比图像
                comparison_image = np.hstack((original_image, cv2.resize(state["output"], (original_image.shape[1], original_image.shape[0]))))
                comparison_path = "image_comparison.jpg"
                cv2.imwrite(comparison_path, comparison_image)
                state["comparison_image_path"] = comparison_path
                
                # 保存增强后的图像
                enhanced_path = "enhanced_image.jpg"
                cv2.imwrite(enhanced_path, state["output"])
                state["enhanced_image_path"] = enhanced_path
                
            except Exception as e:
                print(f"图像增强推理错误: {str(e)}")
                state["error"] = f"enhancement_inference_error: {str(e)}"
                # 提供默认输出以防止后续步骤出错
                state["output"] = state.get("original_image", np.zeros((100, 100, 3), dtype=np.uint8))

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
            
            # 尝试告诉Ollama释放资源
            try:
                import requests
                requests.post('http://localhost:11434/api/cancel', json={})
                print("已请求Ollama释放LLaVA模型资源")
            except Exception as release_err:
                print(f"请求释放Ollama资源失败: {str(release_err)}")
            
        else:
            raise ValueError(f"不支持的任务类型: {task}")

        # 验证输出是否存在且有效
        if "output" not in state or state["output"] is None:
            raise ValueError(f"模型未能生成有效输出")

        # 清理模型引用和显存
        if task in ["classification", "annotation", "enhancement"]:
            if "model" in state:
                del state["model"]
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                print(f"{task}任务模型资源已释放")

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
            elif task == "enhancement":
                state["output"] = state.get("original_image", np.zeros((100, 100, 3), dtype=np.uint8))
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
            
        elif task == "enhancement":
            # 获取增强后的图像路径
            enhanced_path = state.get("enhanced_image_path", "enhanced_image.jpg")
            comparison_path = state.get("comparison_image_path", "image_comparison.jpg")
            
            # 获取原始图像和增强图像的尺寸
            original_image = state.get("original_image")
            enhanced_image = state.get("output")
            
            if original_image is not None and enhanced_image is not None:
                orig_h, orig_w = original_image.shape[:2]
                enh_h, enh_w = enhanced_image.shape[:2]
                
                # 提供有关增强的详细信息
                enhancement_info = f"原始图像分辨率: {orig_w}x{orig_h} → 增强后分辨率: {enh_w}x{enh_h}"
                scale_factor = round(enh_w / orig_w, 1)
                
                result = f"图像增强完成！\n{enhancement_info}\n放大倍数: {scale_factor}x\n增强图像已保存至 {enhanced_path}\n对比图像已保存至 {comparison_path}"
            else:
                result = f"图像增强完成！增强图像已保存至 {enhanced_path}\n对比图像已保存至 {comparison_path}"
            
            print(result)
            state["result_message"] = result
            
            # 作为备份，在状态集合中再添加一个副本
            state["enhancement"] = {
                "message": state["result_message"],
                "enhanced_path": enhanced_path,
                "comparison_path": comparison_path
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
            elif task == "enhancement":
                state["result_message"] = f"图像增强完成！增强图像已保存至 {enhanced_path}\n对比图像已保存至 {comparison_path}"
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

    # 对于增强任务，显示图像
    if final_state.get("task") == "enhancement" and not final_state.get("error"):
        try:
            enhanced_path = final_state.get("enhanced_image_path", "enhanced_image.jpg")
            comparison_path = final_state.get("comparison_image_path", "image_comparison.jpg")
            
            if os.path.exists(comparison_path):
                comparison_img = cv2.imread(comparison_path)
                plt.figure(figsize=(12, 6))
                plt.imshow(cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB))
                plt.title("对比图（左：原图，右：增强后）")
                plt.axis('off')
                plt.show()
        except Exception as e:
            print(f"显示图像时出错: {str(e)}")

    # 释放模型和GPU显存
    cleanup_resources()
    
    print("=" * 50)

    return final_state

# 添加清理资源的函数
def cleanup_resources():
    try:
        print("开始释放模型资源...")
        
        # 清理Torch占用的GPU内存
        if torch.cuda.is_available():
            # 确保所有GPU操作都已完成
            torch.cuda.synchronize()
            
            # 清理缓存
            torch.cuda.empty_cache()
            
            # 关闭和清除模型
            print("释放GPU显存...")
            import gc
            gc.collect()
            
            # 检查释放后的显存状态
            if hasattr(torch.cuda, 'memory_allocated'):
                allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)
                print(f"GPU显存状态 - 已分配: {allocated_memory:.2f}MB, 已保留: {reserved_memory:.2f}MB")
        
        # 尝试释放Ollama相关资源
        try:
            import requests
            # 告诉Ollama服务停止当前模型运行（如果支持）
            try:
                # 请求Ollama API释放模型
                requests.post('http://localhost:11434/api/cancel', json={})
                print("已请求Ollama取消当前运行的模型")
            except Exception as ollama_err:
                print(f"Ollama API请求失败: {str(ollama_err)}")
                
            # 手动进行垃圾收集
            gc.collect()
            print("已执行垃圾收集")
        except Exception as api_err:
            print(f"尝试释放Ollama资源时出错: {str(api_err)}")
            
        print("资源清理完成")
    except Exception as e:
        print(f"清理资源时出错: {str(e)}")


if __name__ == "__main__":
    main()
