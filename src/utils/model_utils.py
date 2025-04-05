import os
import sys
import torch
import gc
import requests
from typing import Any, Dict, Optional, Tuple

def get_device() -> torch.device:
    """
    获取当前可用的设备（GPU或CPU）
    
    Returns:
        torch.device: CUDA设备或CPU
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def move_model_to_device(model: torch.nn.Module) -> Tuple[torch.nn.Module, torch.device]:
    """
    将模型移动到合适的设备（GPU或CPU）
    
    Args:
        model: PyTorch模型
        
    Returns:
        (model, device): 移动后的模型和使用的设备
    """
    device = get_device()
    model = model.to(device)
    return model, device

def release_model(model: Any) -> None:
    """
    释放模型资源
    
    Args:
        model: 要释放的模型
    """
    try:
        # 删除模型引用
        del model
        
        # 手动触发垃圾回收
        gc.collect()
        
        # 如果有GPU，清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("模型资源已释放")
    except Exception as e:
        print(f"释放模型资源时出错: {str(e)}")

def release_ollama_model() -> None:
    """释放Ollama模型资源"""
    try:
        # 请求Ollama API取消当前运行的模型
        requests.post('http://localhost:11434/api/cancel', json={})
        print("已请求Ollama释放模型资源")
    except Exception as e:
        print(f"释放Ollama资源时出错: {str(e)}")

def cleanup_resources() -> None:
    """清理所有资源（内存、GPU、API等）"""
    # 清理Torch内存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"GPU资源清理 - 已分配: {allocated_memory:.2f}MB, 已保留: {reserved_memory:.2f}MB")
    
    # 手动垃圾回收
    gc.collect()
    
    # 清理Ollama资源
    release_ollama_model()
    
    print("所有资源已清理")

def download_model_file(url: str, dest_path: str, use_proxy: bool = False) -> bool:
    """
    从URL下载模型文件
    
    Args:
        url: 下载URL
        dest_path: 目标路径
        use_proxy: 是否使用代理
        
    Returns:
        bool: 下载是否成功
    """
    try:
        if use_proxy:
            # 设置代理
            os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
            os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        
        import urllib.request
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"下载模型文件失败: {str(e)}")
        return False 