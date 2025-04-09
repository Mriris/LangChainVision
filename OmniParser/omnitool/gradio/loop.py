"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
代理采样循环，调用Anthropic API和本地实现的anthropic定义的计算机使用工具。
"""
from collections.abc import Callable
from enum import StrEnum

from anthropic import APIResponse
from anthropic.types import (
    TextBlock,
)
from anthropic.types.beta import (
    BetaContentBlock,
    BetaMessage,
    BetaMessageParam
)
from tools import ToolResult

from agent.llm_utils.omniparserclient import OmniParserClient
from agent.anthropic_agent import AnthropicActor
from agent.vlm_agent import VLMAgent
from agent.vlm_agent_with_orchestrator import VLMOrchestratedAgent
from executor.anthropic_executor import AnthropicExecutor

BETA_FLAG = "computer-use-2024-10-22"

class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OPENAI = "openai"
    GROQ = "groq"
    DASHSCOPE = "dashscope"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
    APIProvider.OPENAI: "gpt-4o",
    APIProvider.GROQ: "mixtral-8x7b-32768",
    APIProvider.DASHSCOPE: "qwen-vl-plus",
}

def sampling_loop_sync(
    *,
    model: str,
    provider: APIProvider | None,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = 2,
    max_tokens: int = 4096,
    omniparser_url: str,
    ollama_server_url: str = None,
    ollama_model: str = None,
    ollama_params: dict = None,
    save_folder: str = "./uploads"
):
    """
    助手/工具交互的计算机使用的同步代理采样循环。
    """
    print('在sampling_loop_sync中，模型:', model)
    print('ollama参数:', ollama_params)  # 打印参数以便调试
    omniparser_client = OmniParserClient(url=f"http://{omniparser_url}/parse/")
    if model == "claude-3-5-sonnet-20241022":
        # 注册Actor和Executor
        actor = AnthropicActor(
            model=model, 
            provider=provider,
            api_key=api_key, 
            api_response_callback=api_response_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images
        )
    elif model in set(["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", "omniparser + qwen2.5vl"]):
        actor = VLMAgent(
            model=model,
            provider=provider,
            api_key=api_key,
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images
        )
    elif model in set(["omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated", "omniparser + R1-orchestrated", "omniparser + qwen2.5vl-orchestrated"]):
        actor = VLMOrchestratedAgent(
            model=model,
            provider=provider,
            api_key=api_key,
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
            save_folder=save_folder
        )
    elif model == "omniparser + ollama":
        # 使用VLMAgent处理Ollama模型
        actor = VLMAgent(
            model=model,
            provider="ollama",  # 强制设为ollama
            api_key=None,  # Ollama不需要API密钥
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
            ollama_server_url=ollama_server_url,
            ollama_model=ollama_model
        )
        # 增加调试输出，确认Ollama模型配置
        print(f"Ollama配置 - 模型: {ollama_model}, 服务器: {ollama_server_url}")
        if ollama_params:
            print(f"Ollama自定义参数: {ollama_params}")
    else:
        raise ValueError(f"不支持模型 {model}")
    executor = AnthropicExecutor(
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
    )
    print(f"模型已初始化: {model}, 提供商: {provider}")
    
    tool_result_content = None
    
    print(f"开始消息循环。用户消息: {messages}")
    
    if model == "claude-3-5-sonnet-20241022": # Anthropic循环
        while True:
            parsed_screen = omniparser_client() # parsed_screen: {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, "screen_info"}
            screen_info_block = TextBlock(text='以下是当前UI屏幕的结构化可访问信息，包括您可以操作的文本和图标，在预测下一步操作时请考虑这些信息。请注意，您仍需要截图获取图像: \n' + parsed_screen['screen_info'], type='text')
            screen_info_dict = {"role": "user", "content": [screen_info_block]}
            messages.append(screen_info_dict)
            tools_use_needed = actor(messages=messages)

            for message, tool_result_content in executor(tools_use_needed, messages):
                yield message
        
            if not tool_result_content:
                return messages

            messages.append({"content": tool_result_content, "role": "user"})
    
    elif model in set(["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", "omniparser + qwen2.5vl", "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated", "omniparser + R1-orchestrated", "omniparser + qwen2.5vl-orchestrated", "omniparser + ollama"]):
        while True:
            parsed_screen = omniparser_client()
            # 为Ollama模型传递参数
            if model == "omniparser + ollama":
                tools_use_needed, vlm_response_json = actor(messages=messages, parsed_screen=parsed_screen, ollama_params=ollama_params)
            else:
                tools_use_needed, vlm_response_json = actor(messages=messages, parsed_screen=parsed_screen)

            for message, tool_result_content in executor(tools_use_needed, messages):
                yield message
        
            if not tool_result_content:
                return messages

            messages.append({"content": tool_result_content, "role": "user"})