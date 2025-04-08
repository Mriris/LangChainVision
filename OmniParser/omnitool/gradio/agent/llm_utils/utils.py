import base64

def is_image_path(text):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif")
    if text.endswith(image_extensions):
        return True
    else:
        return False

def encode_image(image_path):
    """Encode image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_token_usage_from_ollama_response(result):
    """
    从Ollama响应中安全地提取token使用量
    
    Args:
        result: Ollama响应的JSON对象
        
    Returns:
        int: 提取的token使用量，如果无法提取则返回0
    """
    try:
        # 尝试不同的字段名称获取token用量
        if "eval_count" in result:
            return int(result["eval_count"])
        elif "usage" in result and isinstance(result["usage"], dict):
            return int(result["usage"].get("total_tokens", 0))
        elif "request_id" in result:
            # 处理只返回request_id的情况
            print(f"Ollama只返回了request_id，使用0作为默认值")
            return 0
        else:
            # 其他情况使用默认值
            return 0
    except (ValueError, TypeError) as e:
        print(f"Ollama token用量格式错误: {e}，使用0作为默认值")
        return 0

def safe_int_conversion(value, default=0):
    """
    安全地将值转换为整数，如果转换失败则返回默认值
    
    Args:
        value: 要转换的值
        default: 转换失败时返回的默认值
        
    Returns:
        int: 转换后的整数值或默认值
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default