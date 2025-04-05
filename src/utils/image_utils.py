import os
import cv2
import numpy as np
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional, Union
import torch

def read_image(image_path: str) -> Optional[np.ndarray]:
    """
    读取图像文件
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        numpy数组格式的图像，读取失败时返回None
    """
    if not os.path.exists(image_path):
        return None
    
    try:
        # 使用OpenCV读取图像
        image = cv2.imread(image_path)
        if image is None:
            # 尝试使用PIL读取
            pil_image = Image.open(image_path)
            image = np.array(pil_image)
            # 如果是RGB格式，转换为BGR
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        print(f"读取图像失败: {str(e)}")
        return None

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    调整图像大小
    
    Args:
        image: 原始图像
        target_size: 目标尺寸 (width, height)
        
    Returns:
        调整大小后的图像
    """
    return cv2.resize(image, target_size)

def preprocess_for_classification(image: np.ndarray) -> np.ndarray:
    """
    为分类任务预处理图像
    
    Args:
        image: 原始图像
        
    Returns:
        预处理后的图像, shape为(C, H, W), 值范围为[0, 1]
    """
    # 调整大小为224x224
    image = cv2.resize(image, (224, 224))
    # 归一化
    image = image / 255.0
    # 转换为CHW格式
    image = np.transpose(image, (2, 0, 1))
    return image

def preprocess_for_interpretation(image: np.ndarray) -> np.ndarray:
    """
    为图像描述任务预处理图像
    
    Args:
        image: 原始图像
        
    Returns:
        预处理后的图像, shape为(C, H, W), 值范围为[0, 1]
    """
    # 调整大小为384x384
    image = cv2.resize(image, (384, 384))
    # 归一化
    image = image / 255.0
    # 转换为CHW格式
    image = np.transpose(image, (2, 0, 1))
    return image

def image_to_base64(image: Union[str, np.ndarray]) -> str:
    """
    将图像转换为Base64编码的字符串
    
    Args:
        image: 图像路径或numpy数组
        
    Returns:
        Base64编码的字符串
    """
    if isinstance(image, str):
        # 如果是路径，读取图像
        if not os.path.exists(image):
            return ""
        with open(image, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    else:
        # 如果是numpy数组，转换为JPEG
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            return ""
        return base64.b64encode(buffer).decode('utf-8')

def create_comparison_image(original: np.ndarray, enhanced: np.ndarray, 
                           output_path: str) -> str:
    """
    创建原始图像和增强图像的对比图
    
    Args:
        original: 原始图像
        enhanced: 增强后的图像
        output_path: 输出路径
        
    Returns:
        保存的图像路径
    """
    # 调整增强图像的大小与原始图像匹配
    enhanced_resized = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    
    # 创建并排显示的对比图
    comparison = np.hstack((original, enhanced_resized))
    
    # 保存对比图
    cv2.imwrite(output_path, comparison)
    
    return output_path

def save_annotated_image(image: np.ndarray, detections: List[Dict], 
                        class_names: Dict[int, str], output_path: str) -> str:
    """
    保存标注后的图像
    
    Args:
        image: 原始图像
        detections: 检测结果列表
        class_names: 类别名称字典
        output_path: 输出路径
        
    Returns:
        保存的图像路径
    """
    # 创建副本
    annotated_image = image.copy()
    
    # 在图像上绘制检测结果
    for det in detections:
        x_min, y_min, x_max, y_max, conf, class_id = det
        class_id = int(class_id)
        label = f"{class_names[class_id]}: {conf:.2f}"
        
        # 绘制边界框
        cv2.rectangle(
            annotated_image,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            (0, 255, 0),
            2
        )
        # 添加标签
        cv2.putText(
            annotated_image,
            label,
            (int(x_min), int(y_min) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    # 转换为RGB用于Matplotlib
    image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # 保存图像
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def pil_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """
    将PIL图像转换为字节
    
    Args:
        image: PIL图像
        format: 图像格式
        
    Returns:
        图像字节
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()

def cleanup_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        # 确保所有GPU操作都已完成
        torch.cuda.synchronize()
        
        # 清理缓存
        torch.cuda.empty_cache()
        
        # 手动进行垃圾收集
        import gc
        gc.collect()
        
        # 打印当前GPU内存使用情况
        if hasattr(torch.cuda, 'memory_allocated'):
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f"GPU内存状态 - 已分配: {allocated_memory:.2f}MB, 已保留: {reserved_memory:.2f}MB") 