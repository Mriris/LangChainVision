import os
from typing import Dict, Any, List, Optional, Tuple

from src.schemas.state import ImageState
from src.config.settings import (
    TASK_CLASSIFICATION,
    TASK_ANNOTATION,
    TASK_INTERPRETATION,
    TASK_ENHANCEMENT
)

class OutputProcessor:
    """输出处理器，负责格式化和返回结果"""
    
    @staticmethod
    def process_output(state: ImageState) -> ImageState:
        """
        处理模型输出并生成最终结果
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态，包含结果信息
        """
        try:
            task = state["task"]
            
            # 检查输出是否存在
            if "output" not in state:
                raise KeyError("state中缺少'output'键，可能是推理阶段未成功完成")
            
            # 根据任务类型处理输出
            if task == TASK_CLASSIFICATION:
                OutputProcessor._process_classification_output(state)
            
            elif task == TASK_ANNOTATION:
                # 假设标注图像已在推理阶段保存
                # 状态中已包含detection_results和annotated_image_path
                if "result_message" not in state:
                    state["result_message"] = f"标注完成，检测到 {len(state.get('detection_results', []))} 个物体"
            
            elif task == TASK_ENHANCEMENT:
                # 假设增强图像已在推理阶段保存
                # 状态中已包含enhanced_image_path和comparison_image_path
                if "result_message" not in state:
                    state["result_message"] = "图像增强完成！已生成高清晰度图像"
            
            elif task == TASK_INTERPRETATION:
                # 直接使用模型输出作为描述结果
                if "result_message" not in state:
                    state["result_message"] = f"图像描述：\n{state['output']}"
            
            else:
                state["result_message"] = "未知任务类型，无法处理结果"
            
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
            print(f"输出处理完成，result_message: {final_message}")
            
        except KeyError as e:
            error_msg = f"output_error: 缺少必要的数据: {str(e)}"
            print(error_msg)
            state["error"] = error_msg
            state["result_message"] = "处理过程中出现错误，未能生成有效结果"
            state["status"] = "error"
            
        except Exception as e:
            error_msg = f"output_error: {str(e)}"
            print(error_msg)
            state["error"] = error_msg
            state["result_message"] = f"处理错误: {str(e)}"
            state["status"] = "error"
            
        return state
    
    @staticmethod
    def _process_classification_output(state: ImageState) -> None:
        """处理分类任务输出"""
        output = state["output"]
        
        # 尝试加载类别名称
        try:
            with open("imagenet_classes.txt", "r") as f:
                classes = [line.strip() for line in f.readlines()]
            class_name = classes[output]
        except Exception as e:
            class_name = f"类别ID: {output}"
            print(f"警告: 无法加载类别名称 - {str(e)}")
        
        # 更新状态
        state["result_message"] = f"分类结果：{class_name}"
        
    @staticmethod
    def handle_error(state: ImageState) -> ImageState:
        """
        处理错误并尝试恢复
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        error = state.get("error", "未知错误")
        print(f"处理错误: {error}")
        
        # 清除错误并设置默认值
        state["error"] = ""
        if "task" not in state or not state["task"]:
            state["task"] = TASK_INTERPRETATION
            
        state["status"] = "已从错误恢复，使用默认任务"
        
        # 记录到历史
        if "history" in state:
            state["history"].append({
                "step": "error_handler",
                "error": error,
                "recovery": "使用默认任务"
            })
            
        return state 