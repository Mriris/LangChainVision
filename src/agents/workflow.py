from langgraph.graph import StateGraph, END
from typing import Dict, Any, List

from src.schemas.state import ImageState
from src.processors.analysis_processor import AnalysisProcessor
from src.processors.image_processor import ImageProcessor
from src.processors.model_processor import ModelProcessor 
from src.processors.output_processor import OutputProcessor
from src.agents.routing import should_retry, has_error, route_by_task
from src.utils.model_utils import cleanup_resources
from src.config.settings import (
    TASK_CLASSIFICATION,
    TASK_ANNOTATION,
    TASK_INTERPRETATION,
    TASK_ENHANCEMENT
)

class WorkflowBuilder:
    """工作流构建器，使用LangGraph创建图像处理工作流"""
    
    @staticmethod
    def build() -> StateGraph:
        """
        构建并返回工作流图
        
        Returns:
            编译好的状态图
        """
        # 创建状态图
        workflow = StateGraph(ImageState)
        
        # 添加核心处理节点
        workflow.add_node("analysis", AnalysisProcessor.process)
        workflow.add_node("image_input", ImageProcessor.load_image)
        workflow.add_node("preprocess", ImageProcessor.preprocess)
        workflow.add_node("model_selection", ModelProcessor.select_model)
        workflow.add_node("inference", ModelProcessor.run_inference)
        workflow.add_node("save_results", ImageProcessor.save_results)
        workflow.add_node("output_processor", OutputProcessor.process_output)
        workflow.add_node("error_handler", OutputProcessor.handle_error)
        
        # 添加任务特定的节点
        # (这些可以根据实际需要添加)
        
        # 条件路由：分析 -> 重试/继续
        workflow.add_conditional_edges(
            "analysis",
            should_retry,
            {
                "retry": "analysis",  # 如果分析错误，重试
                "continue": "image_input"  # 否则继续
            }
        )
        
        # 错误检查：图像输入 -> 错误/继续
        workflow.add_conditional_edges(
            "image_input",
            has_error,
            {
                "error": "error_handler",
                "continue": "preprocess"
            }
        )
        
        # 错误检查：预处理 -> 错误/继续
        workflow.add_conditional_edges(
            "preprocess",
            has_error,
            {
                "error": "error_handler",
                "continue": "model_selection"
            }
        )
        
        # 错误检查：模型选择 -> 错误/继续
        workflow.add_conditional_edges(
            "model_selection",
            has_error,
            {
                "error": "error_handler",
                "continue": "inference"
            }
        )
        
        # 错误检查：推理 -> 错误/继续
        workflow.add_conditional_edges(
            "inference",
            has_error,
            {
                "error": "error_handler",
                "continue": "save_results"
            }
        )
        
        # 添加最后的处理步骤
        workflow.add_edge("save_results", "output_processor")
        workflow.add_edge("output_processor", END)
        
        # 错误处理后重新开始图像输入
        workflow.add_edge("error_handler", "image_input")
        
        # 设置入口点
        workflow.set_entry_point("analysis")
        
        # 编译工作流程
        return workflow.compile()

# 创建工作流实例
workflow_app = WorkflowBuilder.build()

def execute_workflow(image_path: str, user_requirement: str) -> Dict[str, Any]:
    """
    执行图像处理工作流
    
    Args:
        image_path: 图像路径
        user_requirement: 用户需求描述
        
    Returns:
        工作流执行结果
    """
    try:
        # 初始化状态
        initial_state = {
            "image_path": image_path,
            "user_requirement": user_requirement,
            "task": "",
            "error": "",
            "status": "started",
            "history": []
        }
        
        # 执行工作流
        print(f"开始处理图像: {image_path}, 需求: {user_requirement}")
        final_state = workflow_app.invoke(initial_state)
        
        # 清理资源
        cleanup_resources()
        
        return final_state
    except Exception as e:
        print(f"工作流执行错误: {str(e)}")
        
        # 确保即使出错也释放资源
        cleanup_resources()
        
        # 返回错误状态
        return {
            "error": f"workflow_execution_error: {str(e)}",
            "status": "error",
            "task": "",
            "result_message": f"执行过程中出错: {str(e)}"
        } 