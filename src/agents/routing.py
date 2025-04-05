from typing import Literal
from src.schemas.state import ImageState
from src.config.settings import (
    TASK_CLASSIFICATION,
    TASK_ANNOTATION,
    TASK_INTERPRETATION,
    TASK_ENHANCEMENT
)

def should_retry(state: ImageState) -> Literal["retry", "continue"]:
    """
    决定是否需要重试分析
    
    Args:
        state: 当前状态
        
    Returns:
        'retry' 如果需要重试分析，否则 'continue'
    """
    # 如果有错误并且是分析阶段的错误，则重试
    if state.get("error") and "analysis" in state.get("error", ""):
        print("检测到分析阶段错误，尝试重试...")
        return "retry"
    return "continue"

def has_error(state: ImageState) -> Literal["error", "continue"]:
    """
    检查状态中是否有错误
    
    Args:
        state: 当前状态
        
    Returns:
        'error' 如果存在错误，否则 'continue'
    """
    if state.get("error"):
        print(f"检测到错误: {state.get('error')}")
        return "error"
    return "continue"

def route_by_task(state: ImageState) -> Literal["classification", "annotation", "interpretation", "enhancement", "error"]:
    """
    根据任务类型路由到不同处理流程
    
    Args:
        state: 当前状态
        
    Returns:
        路由目标（任务类型或错误处理）
    """
    # 先检查是否有错误
    if state.get("error"):
        return "error"
    
    # 根据任务类型路由
    task = state.get("task", TASK_INTERPRETATION)  # 默认使用interpretation
    
    # 验证任务类型是否有效
    valid_tasks = [
        TASK_CLASSIFICATION, 
        TASK_ANNOTATION, 
        TASK_INTERPRETATION, 
        TASK_ENHANCEMENT
    ]
    
    if task not in valid_tasks:
        print(f"未知任务类型: {task}，使用默认值: {TASK_INTERPRETATION}")
        state["task"] = TASK_INTERPRETATION
        return TASK_INTERPRETATION
    
    print(f"路由到任务: {task}")
    return task 