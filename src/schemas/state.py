from typing import Dict, Any, List, Literal, Optional, Union
from typing_extensions import TypedDict
import numpy as np
from pydantic import BaseModel

# 使用TypedDict定义LangGraph状态类型
class ImageState(TypedDict):
    """LangGraph工作流状态类型"""
    image_path: str
    image: Any  # numpy array或其他图像数据
    task: Literal["classification", "annotation", "interpretation", "enhancement", ""]
    model: Any
    output: Any
    user_requirement: str
    analysis_result: Optional[Dict[str, Any]]
    error: str
    status: str
    history: List[Dict[str, Any]]
    # 可选字段
    original_image: Optional[Any]
    enhanced_image_path: Optional[str]
    comparison_image_path: Optional[str]
    annotated_image_path: Optional[str]
    detection_results: Optional[List[Dict[str, Any]]]
    result_message: Optional[str]
    
# 分析结果模型
class AnalysisResult(BaseModel):
    """任务分析结果模型"""
    task: str
    confidence: float
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task": self.task,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }

# 检测结果模型
class DetectionResult(BaseModel):
    """物体检测结果模型"""
    class_name: str
    confidence: float
    box: List[float]  # [x_min, y_min, x_max, y_max]
    
# API响应模型
class TaskResponse(BaseModel):
    """API响应模型"""
    task: str
    status: str
    message: str
    error: Optional[str] = None
    is_remote_sensing: Optional[bool] = False 