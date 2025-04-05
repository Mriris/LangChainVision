import torch
import numpy as np
import cv2
from PIL import Image
import io
from typing import Dict, Any, List, Optional

from src.schemas.state import ImageState
from src.providers.model_provider import ModelProvider
from src.utils.model_utils import release_model, get_device
from src.processors.image_processor import ImageProcessor
from src.config.settings import (
    TASK_CLASSIFICATION,
    TASK_ANNOTATION,
    TASK_INTERPRETATION,
    TASK_ENHANCEMENT
)

class ModelProcessor:
    """模型处理器，负责模型选择和推理"""
    
    @staticmethod
    def select_model(state: ImageState) -> ImageState:
        """
        根据任务类型选择合适的模型
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态，包含选择的模型
        """
        try:
            task = state["task"]
            
            # 首先尝试释放先前可能加载的模型
            if "model" in state and state["model"] is not None:
                release_model(state["model"])
                
            # 根据任务类型选择模型
            if task == TASK_CLASSIFICATION:
                model = ModelProvider.get_classification_model()
                
            elif task == TASK_ANNOTATION:
                model = ModelProvider.get_object_detection_model()
                
            elif task == TASK_ENHANCEMENT:
                model = ModelProvider.get_enhancement_model()
                
            elif task == TASK_INTERPRETATION:
                model = ModelProvider.get_vision_model()
                
            else:
                raise ValueError(f"不支持的任务类型: {task}")
                
            # 更新状态
            state["model"] = model
            
            # 记录到历史
            if "history" in state:
                state["history"].append({
                    "step": "model_selection",
                    "task": task,
                    "model_type": str(type(model).__name__)
                })
                
        except Exception as e:
            error_msg = f"model_selection_error: {str(e)}"
            print(error_msg)
            state["error"] = error_msg
            
            # 使用备用模型
            state["model"] = ModelProvider.create_dummy_models(task)
            
        return state
    
    @staticmethod
    def run_inference(state: ImageState) -> ImageState:
        """
        运行模型推理
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态，包含模型输出
        """
        try:
            task = state["task"]
            model = state["model"]
            image = state["image"]
            
            # 根据任务类型运行不同的推理逻辑
            if task == TASK_CLASSIFICATION:
                ModelProcessor._run_classification(state)
                
            elif task == TASK_ANNOTATION:
                ModelProcessor._run_object_detection(state)
                
            elif task == TASK_ENHANCEMENT:
                ModelProcessor._run_enhancement(state)
                
            elif task == TASK_INTERPRETATION:
                ModelProcessor._run_interpretation(state)
                
            else:
                raise ValueError(f"不支持的任务类型: {task}")
                
            # 验证输出是否存在且有效
            if "output" not in state or state["output"] is None:
                raise ValueError(f"模型未能生成有效输出")
                
            # 清理非PyTorch模型的资源
            if task in [TASK_CLASSIFICATION, TASK_ANNOTATION, TASK_ENHANCEMENT]:
                if "model" in state:
                    # 仅保留引用，不真正删除
                    # 实际删除将在后续步骤中完成
                    pass
                    
            # 记录到历史
            if "history" in state:
                state["history"].append({
                    "step": "inference",
                    "task": task,
                    "output_type": str(type(state["output"]).__name__)
                })
                
        except Exception as e:
            error_msg = f"inference_error: {str(e)}"
            print(error_msg)
            state["error"] = error_msg
            
            # 为输出设置默认值
            if "output" not in state:
                state["output"] = ModelProcessor._get_default_output(task)
                
        return state
    
    @staticmethod
    def _run_classification(state: ImageState) -> None:
        """运行图像分类推理"""
        model = state["model"]
        image = state["image"]
        
        # 转换为张量并移到适当设备
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # 运行推理
        with torch.no_grad():
            output = model(image_tensor)
            
        # 获取分类结果
        class_idx = output.argmax(dim=1).item()
        state["output"] = class_idx
        
        # 释放资源
        del model
        state["model"] = None
        torch.cuda.empty_cache()
    
    @staticmethod
    def _run_object_detection(state: ImageState) -> None:
        """运行物体检测推理"""
        model = state["model"]
        
        # YOLOv5需要读取原始图像
        original_image = cv2.imread(state["image_path"])
        if original_image is None:
            raise ValueError(f"无法读取图像文件: {state['image_path']}")
        
        # 运行检测
        results = model(original_image)
        
        # 存储结果
        state["output"] = results.xyxy[0].cpu().numpy()
        state["original_image"] = original_image
        
        # 释放资源
        del model
        state["model"] = None
        torch.cuda.empty_cache()
    
    @staticmethod
    def _run_enhancement(state: ImageState) -> None:
        """运行图像增强推理"""
        model = state["model"]
        
        # 获取原始图像
        original_image = state.get("original_image")
        if original_image is None:
            original_image = cv2.imread(state["image_path"])
            if original_image is None:
                raise ValueError(f"无法读取图像文件: {state['image_path']}")
            state["original_image"] = original_image
        
        # 根据模型类型进行不同处理
        if isinstance(model, cv2.dnn_superres.DnnSuperResImpl):
            # OpenCV超分辨率模型
            enhanced_image = model.upsample(original_image)
            state["output"] = enhanced_image
        else:
            # PyTorch模型（如SwinIR）
            device = get_device()
            
            # 预处理
            img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            
            # 推理
            with torch.no_grad():
                output = model(img_tensor)
            
            # 后处理
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = (output * 255.0).round().astype(np.uint8)
            
            # 转回BGR
            enhanced_image = cv2.cvtColor(output.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            state["output"] = enhanced_image
        
        # 释放资源
        del model
        state["model"] = None
        torch.cuda.empty_cache()
    
    @staticmethod
    def _run_interpretation(state: ImageState) -> None:
        """运行图像描述推理"""
        model = state["model"]
        image = state["image"]
        
        # 将图像转换为字节格式（适用于多模态模型）
        image_bytes = ImageProcessor.prepare_image_for_vision_model(image)
        
        # 构建提示词
        prompt = """
        请详细描述这张图片，包括:
        1. 图片中的主要对象和场景
        2. 物体的位置关系
        3. 颜色、形状和纹理特征
        4. 可能的场景背景和故事

        请用中文回答，尽量详细但不要臆测不存在的内容。
        """
        
        # 调用模型
        response = model.invoke(prompt, images=[image_bytes])
        
        # 存储结果
        state["output"] = response.strip()
        
        # 尝试释放Ollama资源
        try:
            import requests
            requests.post('http://localhost:11434/api/cancel', json={})
        except:
            pass
    
    @staticmethod
    def _get_default_output(task: str) -> Any:
        """获取任务的默认输出，用于错误恢复"""
        if task == TASK_CLASSIFICATION:
            return 0  # 默认类别ID
        elif task == TASK_ANNOTATION:
            return np.array([])  # 空检测结果
        elif task == TASK_ENHANCEMENT:
            # 创建一个小型空图像
            return np.zeros((100, 100, 3), dtype=np.uint8)
        elif task == TASK_INTERPRETATION:
            return "无法生成图像描述，请检查模型连接。"
        else:
            return None 