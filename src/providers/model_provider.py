import os
import torch
import cv2
import sys
from typing import Any, Dict, Optional, List, Tuple, Union
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from langchain_ollama import OllamaLLM

from src.config.settings import (
    MODEL_WEIGHTS_DIR, 
    MODEL_CACHE_DIR,
    OLLAMA_BASE_URL,
    ANALYSIS_MODEL,
    VISION_MODEL,
    YOLO_MODEL
)

from src.utils.model_utils import (
    get_device, 
    move_model_to_device,
    download_model_file
)

class ModelProvider:
    """模型提供者，负责管理和提供各种模型"""
    
    @staticmethod
    def get_classification_model() -> torch.nn.Module:
        """获取图像分类模型（ResNet50）"""
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.eval()
        
        # 移动到GPU（如果可用）
        device = get_device()
        model = model.to(device)
        
        print(f"ResNet50分类模型已加载到{device}")
        return model
        
    @staticmethod
    def get_object_detection_model() -> Any:
        """获取物体检测模型（YOLOv5）"""
        device = get_device()
        print(f"YOLOv5将使用设备: {device}")
        
        try:
            # 尝试从torch hub加载模型
            model = torch.hub.load('ultralytics/yolov5', YOLO_MODEL, pretrained=True)
            print(f"成功加载YOLOv5模型: {YOLO_MODEL}")
            return model
        except Exception as e:
            print(f"从torch hub加载YOLO失败: {str(e)}")
            
            # 尝试使用CPU
            try:
                model = torch.hub.load(
                    'ultralytics/yolov5', 
                    YOLO_MODEL, 
                    pretrained=True, 
                    force_reload=True, 
                    device='cpu'
                )
                print(f"使用CPU成功加载YOLOv5模型: {YOLO_MODEL}")
                return model
            except Exception as cpu_e:
                print(f"使用CPU加载YOLO也失败: {str(cpu_e)}")
                raise RuntimeError(f"无法加载YOLOv5模型: {str(e)}")
    
    @staticmethod
    def get_enhancement_model() -> torch.nn.Module:
        """获取图像增强模型（SwinIR）"""
        try:
            # 检查并创建模型目录
            if not os.path.exists(MODEL_CACHE_DIR):
                os.makedirs(MODEL_CACHE_DIR)
                
            # 检查SwinIR模型文件
            model_path = os.path.join(MODEL_CACHE_DIR, "swinir_model.py")
            if not os.path.exists(model_path):
                # 下载模型定义文件
                url = "https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/models/network_swinir.py"
                download_success = download_model_file(url, model_path, use_proxy=True)
                if not download_success:
                    raise RuntimeError("下载SwinIR模型定义文件失败")
                print(f"已下载SwinIR模型定义文件到: {model_path}")
            
            # 将模型目录添加到系统路径
            if MODEL_CACHE_DIR not in sys.path:
                sys.path.append(MODEL_CACHE_DIR)
                
            # 导入SwinIR
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
            
            # 加载预训练权重
            weights_path = os.path.join(MODEL_WEIGHTS_DIR, "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
            if os.path.exists(weights_path):
                device = get_device()
                pretrained_model = torch.load(weights_path, map_location=device)
                model.load_state_dict(
                    pretrained_model['params'] if 'params' in pretrained_model else pretrained_model, 
                    strict=True
                )
                print(f"成功加载SwinIR预训练权重")
            else:
                # 尝试加载替代权重
                alt_weights_path = os.path.join(MODEL_WEIGHTS_DIR, "SwinIR_model.pth")
                if os.path.exists(alt_weights_path):
                    device = get_device()
                    pretrained_model = torch.load(alt_weights_path, map_location=device)
                    model.load_state_dict(
                        pretrained_model['params'] if 'params' in pretrained_model else pretrained_model, 
                        strict=False
                    )
                    print(f"成功加载替代SwinIR权重")
                else:
                    print(f"未找到SwinIR权重文件，使用未初始化模型")
            
            # 移动到GPU（如果可用）
            device = get_device()
            model = model.to(device)
            model.eval()
            
            print(f"SwinIR增强模型已加载到{device}")
            return model
            
        except Exception as e:
            print(f"加载SwinIR失败: {str(e)}")
            
            # 尝试使用OpenCV的超分辨率模型作为备选
            try:
                print("尝试使用OpenCV的EDSR模型...")
                sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
                edsr_path = os.path.join(MODEL_WEIGHTS_DIR, "EDSR_x4.pb")
                
                if os.path.exists(edsr_path):
                    sr_model.readModel(edsr_path)
                    sr_model.setModel("edsr", 4)  # 4x超分辨率
                    print(f"成功加载EDSR模型")
                    return sr_model
                else:
                    raise RuntimeError(f"未找到EDSR模型文件")
            except Exception as edsr_e:
                print(f"加载EDSR也失败: {str(edsr_e)}")
                raise RuntimeError(f"无法加载任何图像增强模型")
    
    @staticmethod
    def get_llm(model_name: Optional[str] = None, temperature: float = 0.1) -> OllamaLLM:
        """获取LLM模型"""
        # 使用指定模型名或默认的分析模型
        model_name = model_name or ANALYSIS_MODEL
        
        try:
            llm = OllamaLLM(
                model=model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature
            )
            
            # 测试连接
            test_result = llm.invoke("测试连接")
            if not test_result:
                raise RuntimeError(f"{model_name}模型返回空结果")
                
            print(f"成功加载LLM模型: {model_name}")
            return llm
        except Exception as e:
            print(f"加载LLM模型失败: {str(e)}")
            raise RuntimeError(f"无法加载LLM模型 {model_name}: {str(e)}")
    
    @staticmethod
    def get_vision_model() -> OllamaLLM:
        """获取视觉语言模型（LLaVA）"""
        try:
            model = OllamaLLM(
                model=VISION_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.7
            )
            
            # 测试连接
            test_result = model.invoke("测试连接")
            if not test_result:
                raise RuntimeError(f"{VISION_MODEL}模型返回空结果")
                
            print(f"成功加载视觉模型: {VISION_MODEL}")
            return model
        except Exception as e:
            print(f"加载视觉模型失败: {str(e)}")
            raise RuntimeError(f"无法加载视觉模型 {VISION_MODEL}: {str(e)}")
            
    @staticmethod
    def create_dummy_models(task: str) -> Any:
        """创建备用模型，当真实模型加载失败时使用"""
        if task == "classification":
            class DummyModel:
                def __call__(self, x):
                    return torch.tensor([[1.0]])
            return DummyModel()
            
        elif task == "annotation":
            class DummyYOLO:
                def __init__(self):
                    self.names = {0: "person", 1: "object"}
                
                def __call__(self, img):
                    class DummyResults:
                        def __init__(self):
                            self.xyxy = [torch.zeros((0, 6))]
                    return DummyResults()
            return DummyYOLO()
            
        elif task == "enhancement":
            class DummySwinIR:
                def invoke(self, prompt, images=None):
                    return "无法加载增强模型，请检查模型文件是否存在"
            return DummySwinIR()
            
        elif task == "interpretation":
            class DummyLLM:
                def invoke(self, prompt, images=None):
                    return "无法加载LLM模型，请检查Ollama服务是否正在运行"
            return DummyLLM()
            
        else:
            class DummyModel:
                def __call__(self, *args, **kwargs):
                    return None
            return DummyModel() 