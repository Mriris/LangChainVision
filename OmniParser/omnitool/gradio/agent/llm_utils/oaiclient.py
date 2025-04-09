import os
import logging
import base64
import requests
from .utils import is_image_path, encode_image

def run_oai_interleaved(messages: list, system: str, model_name: str, api_key: str, max_tokens=256, temperature=0, provider_base_url: str = "https://api.openai.com/v1"):    
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}
    final_messages = [{"role": "system", "content": system}]

    # 检查是否为Dashscope API调用
    is_dashscope = "dashscope" in provider_base_url.lower()
    
    # Dashscope API的token限制
    dashscope_max_input_tokens = 129000 if is_dashscope else float('inf')
    
    # 简单的token估算函数（每个词大约1.3个token）
    def estimate_tokens(text):
        return int(len(text.split()) * 1.3)
    
    # 用于跟踪估计的token数量
    estimated_total_tokens = estimate_tokens(system)
    
    # 对于Dashscope API，确保限制文本长度
    if is_dashscope:
        print(f"检测到Dashscope API调用，将限制输入tokens在{dashscope_max_input_tokens}以内")

    if type(messages) == list:
        for item in messages:
            contents = []
            if isinstance(item, dict):
                for cnt in item["content"]:
                    if isinstance(cnt, str):
                        if is_image_path(cnt) and 'o3-mini' not in model_name:
                            # 03 mini does not support images
                            base64_image = encode_image(cnt)
                            content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            # 图像的token估算（根据经验值）
                            estimated_image_tokens = 1024  # 粗略估计
                            estimated_total_tokens += estimated_image_tokens
                        else:
                            # 对于Dashscope，如果预计超出限制，截断文本
                            if is_dashscope and estimate_tokens(cnt) + estimated_total_tokens > dashscope_max_input_tokens:
                                # 保留约90%的token限制给当前文本
                                available_tokens = max(1000, dashscope_max_input_tokens - estimated_total_tokens)
                                # 粗略截断（按比例）
                                max_chars = int(available_tokens / 1.3 * 5)  # 假设平均每个token约5个字符
                                truncated_text = cnt[:max_chars]
                                print(f"已截断文本，从{len(cnt)}字符到{len(truncated_text)}字符")
                                cnt = truncated_text
                            
                            content = {"type": "text", "text": cnt}
                            estimated_total_tokens += estimate_tokens(cnt)
                    else:
                        # in this case it is a text block from anthropic
                        text_content = str(cnt)
                        
                        # 对于Dashscope，如果预计超出限制，截断文本
                        if is_dashscope and estimate_tokens(text_content) + estimated_total_tokens > dashscope_max_input_tokens:
                            available_tokens = max(1000, dashscope_max_input_tokens - estimated_total_tokens)
                            max_chars = int(available_tokens / 1.3 * 5)
                            truncated_text = text_content[:max_chars]
                            print(f"已截断文本块，从{len(text_content)}字符到{len(truncated_text)}字符")
                            text_content = truncated_text
                        
                        content = {"type": "text", "text": text_content}
                        estimated_total_tokens += estimate_tokens(text_content)
                        
                    contents.append(content)
                    
                message = {"role": 'user', "content": contents}
            else:  # str
                text_content = item
                
                # 对于Dashscope，如果预计超出限制，截断文本
                if is_dashscope and estimate_tokens(text_content) + estimated_total_tokens > dashscope_max_input_tokens:
                    available_tokens = max(1000, dashscope_max_input_tokens - estimated_total_tokens)
                    max_chars = int(available_tokens / 1.3 * 5)
                    truncated_text = text_content[:max_chars]
                    print(f"已截断字符串消息，从{len(text_content)}字符到{len(truncated_text)}字符")
                    text_content = truncated_text
                
                contents.append({"type": "text", "text": text_content})
                estimated_total_tokens += estimate_tokens(text_content)
                message = {"role": "user", "content": contents}
            
            final_messages.append(message)

    
    elif isinstance(messages, str):
        text_content = messages
        
        # 对于Dashscope，如果预计超出限制，截断文本
        if is_dashscope and estimate_tokens(text_content) + estimated_total_tokens > dashscope_max_input_tokens:
            available_tokens = max(1000, dashscope_max_input_tokens - estimated_total_tokens)
            max_chars = int(available_tokens / 1.3 * 5)
            truncated_text = text_content[:max_chars]
            print(f"已截断单条消息，从{len(text_content)}字符到{len(truncated_text)}字符")
            text_content = truncated_text
        
        final_messages = [{"role": "user", "content": text_content}]
        estimated_total_tokens += estimate_tokens(text_content)

    # 检查预估的token总数是否接近限制
    if is_dashscope:
        print(f"估计输入token数: {estimated_total_tokens}/{dashscope_max_input_tokens}")
        if estimated_total_tokens > dashscope_max_input_tokens * 0.9:
            print("警告: 输入token数接近Dashscope限制")

    payload = {
        "model": model_name,
        "messages": final_messages,
    }
    if 'o1' in model_name or 'o3-mini' in model_name:
        payload['reasoning_effort'] = 'low'
        payload['max_completion_tokens'] = max_tokens
    else:
        payload['max_tokens'] = max_tokens

    # 针对Dashscope的特殊处理
    if is_dashscope:
        # 确保max_tokens不超过Dashscope允许的限制
        payload['max_tokens'] = min(payload.get('max_tokens', max_tokens), 2048)
        print(f"Dashscope API调用，设置max_tokens={payload['max_tokens']}")

    response = requests.post(
        f"{provider_base_url}/chat/completions", headers=headers, json=payload
    )


    try:
        text = response.json()['choices'][0]['message']['content']
        token_usage = int(response.json()['usage']['total_tokens'])
        return text, token_usage
    except Exception as e:
        print(f"Error in interleaved openAI: {e}. This may due to your invalid API key. Please check the response: {response.json()} ")
        return f"Error: {e}", 0