import re
import json
from typing import Dict, Any

from src.schemas.state import ImageState, AnalysisResult
from src.providers.model_provider import ModelProvider
from src.utils.model_utils import release_model
from src.config.settings import (
    TASK_CLASSIFICATION,
    TASK_ANNOTATION,
    TASK_INTERPRETATION,
    TASK_ENHANCEMENT
)

class AnalysisProcessor:
    """需求分析处理器，负责分析用户需求并确定任务类型"""
    
    @staticmethod
    def process(state: ImageState) -> ImageState:
        """
        处理用户需求，确定任务类型
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        # 为状态添加历史记录
        if "history" not in state:
            state["history"] = []
            
        try:
            # 加载分析模型
            analysis_model = ModelProvider.get_llm()
            user_requirement = state["user_requirement"]
            
            # 结构化的提示词
            prompt = f"""
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
            """
            
            # 调用模型
            response = analysis_model.invoke(prompt)
            print(f"原始分析响应: {response[:100]}...")  # 调试信息
            
            # 解析结果
            result = AnalysisProcessor._parse_analysis_response(response, user_requirement)
            
            # 更新状态
            state["analysis_result"] = result.to_dict()
            state["task"] = result.task.lower()
            print(f"分析结果: {state['task']} (置信度: {result.confidence})")
            
            # 记录历史
            state["history"].append({
                "step": "analysis",
                "result": result.to_dict()
            })
            
            # 释放模型资源
            release_model(analysis_model)
            
        except Exception as e:
            error_msg = f"analysis_error: {str(e)}"
            print(f"分析过程出现错误: {error_msg}")
            state["error"] = error_msg
            state["task"] = TASK_INTERPRETATION  # 默认任务
            
        return state
    
    @staticmethod
    def _parse_analysis_response(response: str, user_requirement: str) -> AnalysisResult:
        """
        解析模型返回的分析结果
        
        Args:
            response: 模型响应
            user_requirement: 用户需求（用于备用判断）
            
        Returns:
            解析后的分析结果
        """
        # 提取JSON部分
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            print("从markdown代码块提取JSON")
        else:
            # 尝试直接从响应中提取JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                print("直接从文本提取JSON对象")
            else:
                json_str = response.strip()
                print("使用整个响应作为JSON")
        
        # 清理JSON字符串
        json_str = AnalysisProcessor._clean_json_string(json_str)
        
        try:
            # 解析JSON
            result = json.loads(json_str)
            return AnalysisResult(
                task=result.get("task", TASK_INTERPRETATION),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "")
            )
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {str(e)}")
            
            # 尝试通过正则表达式直接提取
            task = AnalysisProcessor._extract_task_with_regex(json_str)
            if task:
                # 提取置信度
                confidence_match = re.search(r'"confidence"\s*:\s*(0\.\d+)', json_str)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                
                return AnalysisResult(
                    task=task,
                    confidence=confidence,
                    reasoning="通过正则表达式从错误的JSON中提取"
                )
            
            # 通过关键词智能判断
            return AnalysisProcessor._determine_task_from_requirement(user_requirement)
    
    @staticmethod
    def _clean_json_string(json_str: str) -> str:
        """清理JSON字符串，修复常见问题"""
        # 替换中文标点符号为英文标点符号
        punctuation_map = {
            '，': ',', '。': '.', '：': ':', '"': '"', '"': '"',
            ''': "'", ''': "'", '！': '!', '？': '?', '（': '(',
            '）': ')', '【': '[', '】': ']', '、': ',', '；': ';'
        }
        for cn_punct, en_punct in punctuation_map.items():
            json_str = json_str.replace(cn_punct, en_punct)
        
        # 清理非ASCII字符，但保留基本的中文文本
        json_str = re.sub(r'[^\x00-\x7F\u4e00-\u9fff]+', '', json_str)
        
        # 确保JSON键使用双引号
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        return json_str
    
    @staticmethod
    def _extract_task_with_regex(json_str: str) -> str:
        """使用正则表达式从错误的JSON中提取任务类型"""
        task_match = re.search(r'"task"\s*:\s*"([^"]+)"', json_str)
        if not task_match:
            return ""
            
        task = task_match.group(1).lower()
        
        # 验证任务类型是否有效
        valid_tasks = [
            TASK_CLASSIFICATION, 
            TASK_ANNOTATION, 
            TASK_INTERPRETATION, 
            TASK_ENHANCEMENT
        ]
        
        return task if task in valid_tasks else ""
    
    @staticmethod
    def _determine_task_from_requirement(user_requirement: str) -> AnalysisResult:
        """根据用户需求智能判断任务类型"""
        user_req = user_requirement.lower()
        
        if "什么" in user_req or "识别" in user_req or "类别" in user_req or "分类" in user_req:
            task = TASK_CLASSIFICATION
        elif "找出" in user_req or "检测" in user_req or "标记" in user_req or "标注" in user_req or "位置" in user_req:
            task = TASK_ANNOTATION
        elif "描述" in user_req or "说明" in user_req or "讲解" in user_req or "解释" in user_req:
            task = TASK_INTERPRETATION
        elif "增强" in user_req or "提高" in user_req or "改善" in user_req or "优化" in user_req or "清晰" in user_req:
            task = TASK_ENHANCEMENT
        else:
            task = TASK_INTERPRETATION  # 默认任务
        
        print(f"通过关键词智能判断任务类型: {task}")
        
        return AnalysisResult(
            task=task,
            confidence=0.6,
            reasoning="通过关键词从用户需求中智能判断"
        ) 