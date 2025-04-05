import os
import cv2
import numpy as np
from PIL import Image
import io
import torch
from typing import Dict, Any, List, Optional, Tuple

from src.schemas.state import ImageState
from src.config.settings import (
    TASK_CLASSIFICATION,
    TASK_ANNOTATION,
    TASK_INTERPRETATION,
    TASK_ENHANCEMENT
)
from src.utils.image_utils import (
    read_image,
    preprocess_for_classification,
    preprocess_for_interpretation,
    create_comparison_image,
    save_annotated_image
)
from src.config.settings import (
    ANNOTATED_IMAGE_PATH,
    ENHANCED_IMAGE_PATH,
    COMPARISON_IMAGE_PATH
)

class ImageProcessor:
    """图像处理器，提供图像输入和预处理功能"""
    
    @staticmethod
    def load_image(state: ImageState) -> ImageState:
        """
        加载图像
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        try:
            # 获取图像路径
            image_path = state.get("image_path", "")
            if not image_path:
                raise ValueError("缺少图像路径")
                
            # 验证图像文件是否存在
            if not os.path.exists(image_path):
                raise ValueError(f"图像文件不存在: {image_path}")
                
            # 验证文件类型
            valid_extensions = ('.jpg', '.png', '.jpeg', '.tif', '.tiff')
            if not image_path.lower().endswith(valid_extensions):
                raise ValueError(f"不支持的图像格式，请提供以下格式之一: {valid_extensions}")
                
            # 记录到历史
            if "history" in state:
                state["history"].append({
                    "step": "image_input",
                    "path": image_path
                })
                
        except Exception as e:
            error_msg = f"image_input_error: {str(e)}"
            print(error_msg)
            state["error"] = error_msg
            
        return state
    
    @staticmethod
    def preprocess(state: ImageState) -> ImageState:
        """
        根据任务类型对图像进行预处理
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态，包含预处理后的图像
        """
        try:
            image_path = state["image_path"]
            task = state["task"]
            
            # 读取图像
            image = read_image(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
                
            # 根据任务类型进行预处理
            if task == TASK_CLASSIFICATION:
                # 分类任务: 调整尺寸和格式 (C, H, W)，归一化
                processed_image = preprocess_for_classification(image)
                
            elif task == TASK_ANNOTATION:
                # 物体检测任务: 保持原始图像
                processed_image = image.copy()
                
            elif task == TASK_ENHANCEMENT:
                # 图像增强任务: 保持原始图像，并存储副本
                processed_image = image.copy()
                state["original_image"] = image.copy()
                
            elif task == TASK_INTERPRETATION:
                # 图像描述任务: 调整尺寸和格式 (C, H, W)，归一化
                processed_image = preprocess_for_interpretation(image)
                
            else:
                # 未知任务类型
                raise ValueError(f"未知的任务类型: {task}")
                
            # 存储处理后的图像
            state["image"] = processed_image
            
            # 记录到历史
            if "history" in state:
                state["history"].append({
                    "step": "preprocess",
                    "task": task,
                    "image_shape": str(processed_image.shape)
                })
                
        except Exception as e:
            error_msg = f"preprocess_error: {str(e)}"
            print(error_msg)
            state["error"] = error_msg
            
        return state
    
    @staticmethod
    def save_results(state: ImageState) -> ImageState:
        """
        保存处理结果（如标注图像、增强图像等）
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态，包含结果文件路径
        """
        try:
            task = state["task"]
            
            if task == TASK_ANNOTATION:
                # 保存标注图像
                ImageProcessor._save_annotation_results(state)
                
            elif task == TASK_ENHANCEMENT:
                # 保存增强图像和对比图
                ImageProcessor._save_enhancement_results(state)
                
        except Exception as e:
            error_msg = f"save_results_error: {str(e)}"
            print(error_msg)
            state["error"] = error_msg
            
        return state
    
    @staticmethod
    def _save_annotation_results(state: ImageState) -> None:
        """保存物体检测/标注结果"""
        # 获取原始图像
        original_image = state.get("original_image")
        if original_image is None:
            original_image = read_image(state["image_path"])
            if original_image is None:
                raise ValueError(f"无法读取图像: {state['image_path']}")
        
        # 获取检测结果和模型
        boxes = state["output"]
        model = state["model"]
        class_names = model.names
        
        # 创建检测结果列表
        detection_results = []
        for box in boxes:
            x_min, y_min, x_max, y_max, conf, class_id = box
            class_id = int(class_id)
            detection_results.append({
                "class": class_names[class_id],
                "confidence": float(conf),
                "box": [float(x_min), float(y_min), float(x_max), float(y_max)]
            })
        
        # 保存标注图像
        save_annotated_image(original_image, boxes, class_names, ANNOTATED_IMAGE_PATH)
        
        # 更新状态
        state["detection_results"] = detection_results
        state["annotated_image_path"] = ANNOTATED_IMAGE_PATH
        state["result_message"] = f"标注结果已保存至 {ANNOTATED_IMAGE_PATH}, 检测到 {len(detection_results)} 个对象"
    
    @staticmethod
    def _save_enhancement_results(state: ImageState) -> None:
        """保存图像增强结果"""
        # 获取原始图像和增强图像
        original_image = state.get("original_image")
        if original_image is None:
            original_image = read_image(state["image_path"])
            if original_image is None:
                raise ValueError(f"无法读取图像: {state['image_path']}")
        
        enhanced_image = state["output"]
        
        # 保存增强图像
        cv2.imwrite(ENHANCED_IMAGE_PATH, enhanced_image)
        state["enhanced_image_path"] = ENHANCED_IMAGE_PATH
        
        # 创建对比图像
        create_comparison_image(original_image, enhanced_image, COMPARISON_IMAGE_PATH)
        state["comparison_image_path"] = COMPARISON_IMAGE_PATH
        
        # 获取原始和增强图像的尺寸
        orig_h, orig_w = original_image.shape[:2]
        enh_h, enh_w = enhanced_image.shape[:2]
        
        # 计算放大倍数
        scale_factor = round(enh_w / orig_w, 1)
        
        # 更新状态
        enhancement_info = f"原始图像分辨率: {orig_w}x{orig_h} → 增强后分辨率: {enh_w}x{enh_h}"
        state["result_message"] = f"图像增强完成！\n{enhancement_info}\n放大倍数: {scale_factor}x"
    
    @staticmethod
    def prepare_image_for_vision_model(image: np.ndarray) -> bytes:
        """
        将图像准备为视觉模型可接受的格式
        
        Args:
            image: NumPy格式的图像
            
        Returns:
            图像的字节数据
        """
        # 如果图像是CHW格式，转换为HWC
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
            
        # 如果图像已归一化，恢复到 [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
            
        # 确保图像是RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 从BGR转为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # 转换为PIL图像
        pil_image = Image.fromarray(image_rgb)
        
        # 将图像转换为字节
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format='PNG')
        
        return image_bytes.getvalue() 