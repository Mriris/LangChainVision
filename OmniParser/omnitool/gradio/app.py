"""
python app.py --windows_host_url localhost:8006 --omniparser_server_url localhost:8000
"""

import os
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import cast
import argparse
import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock
from loop import (
    APIProvider,
    sampling_loop_sync,
)
from tools import ToolResult
import requests
from requests.exceptions import RequestException
import base64
from json_repair import repair_json

CONFIG_DIR = Path("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

INTRO_TEXT = '''
OmniParser让您可以将任何视觉-语言模型转变为AI代理。我们目前支持**OpenAI (4o/o1/o3-mini)、DeepSeek (R1)、Qwen (2.5VL)或Anthropic Computer Use (Sonnet)**。

输入消息并点击提交以启动OmniTool。点击停止暂停，点击聊天框中的垃圾桶图标清除消息历史。
'''

def parse_arguments():

    parser = argparse.ArgumentParser(description="Gradio App")
    parser.add_argument("--windows_host_url", type=str, default='localhost:8006')
    parser.add_argument("--omniparser_server_url", type=str, default="localhost:8000")
    parser.add_argument("--ollama_server_url", type=str, default="http://localhost:11434")
    return parser.parse_args()
args = parse_arguments()


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "model" not in state:
        state["model"] = "omniparser + gpt-4o"
    if "provider" not in state:
        state["provider"] = "openai"
    if "openai_api_key" not in state:  # Fetch API keys from environment variables
        state["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    if "anthropic_api_key" not in state:
        state["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
    if "ollama_server_url" not in state:
        state["ollama_server_url"] = args.ollama_server_url
    if "ollama_model" not in state:
        state["ollama_model"] = "llama3"
    if "ollama_params" not in state:
        # 设置默认的Ollama参数
        state["ollama_params"] = {
            "temperature": 0.7,
            "num_ctx": 4096,
            "top_p": 0.9,
            "top_k": 40
        }
    if "api_key" not in state:
        state["api_key"] = ""
    if "auth_validated" not in state:
        state["auth_validated"] = False
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 2
    if 'chatbot_messages' not in state:
        state['chatbot_messages'] = []
    if 'stop' not in state:
        state['stop'] = False

async def main(state):
    """Render loop for Gradio"""
    setup_state(state)
    return "设置完成"

def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "请输入您的Anthropic API密钥以继续。"
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "您必须设置AWS凭证才能使用Bedrock API。"
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "请设置CLOUD_ML_REGION环境变量以使用Vertex API。"
        try:
            google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except DefaultCredentialsError:
            return "您的Google Cloud凭证设置不正确。"

def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        print(f"调试: 加载{filename}时出错: {e}")
    return None

def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        print(f"调试: 保存{filename}时出错: {e}")

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output

def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
    
        print(f"_render_message: {str(message)[:100]}")
        
        if isinstance(message, str):
            return message
        
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult)
            or message.__class__.__name__ == "ToolResult"
        )
        if not message or (
            is_tool_result
            and hide_images
            and not hasattr(message, "error")
            and not hasattr(message, "output")
        ):  # return None if hide_images is True
            return
        # render tool result
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"错误: {message.error}"
            if message.base64_image and not hide_images:
                # somehow can't display via gr.Image
                # image_data = base64.b64decode(message.base64_image)
                # return gr.Image(value=Image.open(io.BytesIO(image_data)))
                return f'<img src="data:image/png;base64,{message.base64_image}">'

        elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
            return f"分析: {message.text}"
        elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
            # return f"Tool Use: {message.name}\nInput: {message.input}"
            return f"接下来我将执行以下操作: {message.input}"
        else:  
            return message

    def _truncate_string(s, max_length=500):
        """Truncate long strings for concise printing."""
        if isinstance(s, str) and len(s) > max_length:
            return s[:max_length] + "..."
        return s
    # processing Anthropic messages
    message = _render_message(message, hide_images)
    
    if sender == "bot":
        chatbot_state.append((None, message))
    else:
        chatbot_state.append((message, None))
    
    # Create a concise version of the chatbot state for printing
    concise_state = [(_truncate_string(user_msg), _truncate_string(bot_msg))
                        for user_msg, bot_msg in chatbot_state]
    # print(f"chatbot_output_callback chatbot_state: {concise_state} (truncated)")

def valid_params(user_input, state):
    """Validate all requirements and return a list of error messages."""
    errors = []
    
    for server_name, url in [('Windows Host', 'localhost:5000'), ('OmniParser Server', args.omniparser_server_url)]:
        try:
            url = f'http://{url}/probe'
            response = requests.get(url, timeout=3)
            if response.status_code != 200:
                errors.append(f"{server_name}没有响应")
        except RequestException as e:
            errors.append(f"{server_name}没有响应")
    
    # 检查API密钥，如果不是ollama提供商才需要
    if state["provider"] != "ollama" and not state["api_key"].strip():
        errors.append("未设置LLM API密钥")
    
    # 检查Ollama服务器状态
    if state["provider"] == "ollama":
        try:
            ollama_url = state["ollama_server_url"].rstrip("/")
            response = requests.get(f"{ollama_url}/api/tags", timeout=3)
            if response.status_code != 200:
                errors.append("Ollama服务器未响应")
        except RequestException as e:
            errors.append(f"Ollama服务器连接错误: {str(e)}")

    if not user_input:
        errors.append("未提供计算机使用请求")
    
    return errors

def get_ollama_models(server_url):
    """从Ollama服务器获取可用模型列表"""
    try:
        ollama_url = server_url.rstrip("/")
        response = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            # 提取模型名称列表
            models = [model["name"] for model in data.get("models", [])]
            return models
        return []
    except Exception as e:
        print(f"获取Ollama模型列表失败: {str(e)}")
        return []

def process_input(user_input, state):
    # Reset the stop flag
    if state["stop"]:
        state["stop"] = False

    errors = valid_params(user_input, state)
    if errors:
        raise gr.Error("验证错误: " + ", ".join(errors))
    
    # Append the user message to state["messages"]
    state["messages"].append(
        {
            "role": Sender.USER,
            "content": [TextBlock(type="text", text=user_input)],
        }
    )

    # Append the user's message to chatbot_messages with None for the assistant's reply
    state['chatbot_messages'].append((user_input, None))
    yield state['chatbot_messages']  # Yield to update the chatbot UI with the user's message

    print("state")
    print(state)

    # 获取Ollama参数（如果有）
    ollama_params = state.get("ollama_params", {})

    # Run sampling_loop_sync with the chatbot_output_callback
    for loop_msg in sampling_loop_sync(
        model=state["model"],
        provider=state["provider"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state['chatbot_messages'], hide_images=False),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        max_tokens=16384,
        omniparser_url=args.omniparser_server_url,
        ollama_server_url=state.get("ollama_server_url"),
        ollama_model=state.get("ollama_model"),
        ollama_params=ollama_params
    ):  
        if loop_msg is None or state.get("stop"):
            yield state['chatbot_messages']
            print("任务结束。关闭循环。")
            break
            
        yield state['chatbot_messages']  # Yield the updated chatbot_messages to update the chatbot UI

def stop_app(state):
    state["stop"] = True
    return "应用已停止"

def get_header_image_base64():
    try:
        # Get the absolute path to the image relative to this script
        script_dir = Path(__file__).parent
        image_path = script_dir.parent.parent / "imgs" / "header_bar_thin.png"
        
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f'data:image/png;base64,{encoded_string}'
    except Exception as e:
        print(f"加载头图失败: {e}")
        return None

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.HTML("""
        <style>
        .no-padding {
            padding: 0 !important;
        }
        .no-padding > div {
            padding: 0 !important;
        }
        .markdown-text p {
            font-size: 18px;  /* Adjust the font size as needed */
        }
        </style>
    """)
    state = gr.State({})
    
    setup_state(state.value)
    
    header_image = get_header_image_base64()
    if header_image:
        gr.HTML(f'<img src="{header_image}" alt="OmniTool标题" width="100%">', elem_classes="no-padding")
        gr.HTML('<h1 style="text-align: center; font-weight: normal;">Omni<span style="font-weight: bold;">Tool</span></h1>')
    else:
        gr.Markdown("# OmniTool")

    if not os.getenv("HIDE_WARNING", False):
        gr.Markdown(INTRO_TEXT, elem_classes="markdown-text")


    with gr.Accordion("设置", open=True): 
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    label="模型",
                    choices=["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", 
                            "omniparser + qwen2.5vl", "claude-3-5-sonnet-20241022", 
                            "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", 
                            "omniparser + o3-mini-orchestrated", "omniparser + R1-orchestrated", 
                            "omniparser + qwen2.5vl-orchestrated", "omniparser + ollama"],
                    value="omniparser + gpt-4o",
                    interactive=True,
                )
            with gr.Column():
                only_n_images = gr.Slider(
                    label="保留最近N张截图",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    interactive=True
                )
        with gr.Row():
            with gr.Column(1):
                provider = gr.Dropdown(
                    label="API提供商",
                    choices=[option.value for option in APIProvider] + ["ollama"],
                    value="openai",
                    interactive=False,
                )
            with gr.Column(2):
                api_key = gr.Textbox(
                    label="API密钥",
                    type="password",
                    value=state.value.get("api_key", ""),
                    placeholder="在此粘贴您的API密钥",
                    interactive=True,
                )
        
        # 添加Ollama特定设置（条件显示）
        with gr.Row(visible=False) as ollama_settings:
            with gr.Column(1):
                ollama_server = gr.Textbox(
                    label="Ollama服务器URL",
                    value=args.ollama_server_url,
                    placeholder="http://localhost:11434",
                    interactive=True,
                )
            with gr.Column(2):
                # 将文本框改为下拉框，移除不支持的placeholder参数
                ollama_model_selector = gr.Dropdown(
                    label="Ollama模型",
                    choices=["加载中..."],
                    value="",
                    interactive=True,
                )
                refresh_models_btn = gr.Button("刷新模型列表")
        
        # 添加Ollama参数设置面板
        with gr.Row(visible=False) as ollama_params_panel:
            with gr.Column(1):
                temperature = gr.Slider(
                    label="Temperature (温度)",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=0.7,
                    info="值越高，回答越有创意；值越低，回答越连贯保守"
                )
                num_ctx = gr.Slider(
                    label="Context Window (上下文窗口)",
                    minimum=512,
                    maximum=16384,
                    step=512,
                    value=4096,
                    info="控制模型能够使用的token数量"
                )
            with gr.Column(1):
                top_p = gr.Slider(
                    label="Top P",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=0.9,
                    info="与top_k配合使用，值越高，文本越多样化"
                )
                top_k = gr.Slider(
                    label="Top K",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=40,
                    info="减少生成无意义内容的概率，值越高答案越多样化"
                )

    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="输入消息发送给Omniparser + X ...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="发送", variant="primary")
        with gr.Column(scale=1, min_width=50):
            stop_button = gr.Button(value="停止", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="聊天历史", autoscroll=True, height=580)
        with gr.Column(scale=3):
            iframe = gr.HTML(
                f'<iframe src="http://{args.windows_host_url}/vnc.html?view_only=1&autoconnect=1&resize=scale" width="100%" height="580" allow="fullscreen"></iframe>',
                container=False,
                elem_classes="no-padding"
            )

    def update_model(model_selection, state):
        state["model"] = model_selection
        print(f"模型已更新为: {state['model']}")
        
        if model_selection == "claude-3-5-sonnet-20241022":
            provider_choices = [option.value for option in APIProvider if option.value != "openai"]
        elif model_selection in set(["omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated"]):
            provider_choices = ["openai"]
        elif model_selection == "omniparser + R1":
            provider_choices = ["groq"]
        elif model_selection == "omniparser + qwen2.5vl":
            provider_choices = ["dashscope"]
        elif model_selection == "omniparser + ollama":
            provider_choices = ["ollama"]
            state["provider"] = "ollama"
        else:
            provider_choices = [option.value for option in APIProvider]
        default_provider_value = provider_choices[0]

        provider_interactive = len(provider_choices) > 1
        api_key_placeholder = f"{default_provider_value.title()} API密钥" if default_provider_value != "ollama" else "不需要API密钥"

        # Update state
        state["provider"] = default_provider_value
        if default_provider_value != "ollama":
            state["api_key"] = state.get(f"{default_provider_value}_api_key", "")
        else:
            state["api_key"] = ""  # ollama不需要API密钥

        # 显示或隐藏Ollama设置
        ollama_settings_visible = default_provider_value == "ollama"
        
        # 如果选择Ollama，自动获取模型列表
        ollama_models = []
        if ollama_settings_visible:
            ollama_models = get_ollama_models(state["ollama_server_url"])
            if ollama_models:
                state["ollama_model"] = ollama_models[0]
            else:
                state["ollama_model"] = ""

        # Calls to update other components UI
        provider_update = gr.update(
            choices=provider_choices,
            value=default_provider_value,
            interactive=provider_interactive
        )
        api_key_update = gr.update(
            placeholder=api_key_placeholder,
            value=state["api_key"],
            interactive=default_provider_value != "ollama"  # ollama不需要API密钥，因此禁用输入框
        )
        ollama_settings_update = gr.update(visible=ollama_settings_visible)
        ollama_params_update = gr.update(visible=ollama_settings_visible)
        
        # 更新Ollama模型选择器
        ollama_model_selector_update = gr.update(
            choices=ollama_models if ollama_models else ["无可用模型或无法连接到Ollama服务器"],
            value=state["ollama_model"] if ollama_models else "",
            interactive=bool(ollama_models)
        )

        return provider_update, api_key_update, ollama_settings_update, ollama_model_selector_update, ollama_params_update

    def update_only_n_images(only_n_images_value, state):
        state["only_n_most_recent_images"] = only_n_images_value
   
    def update_provider(provider_value, state):
        # Update state
        state["provider"] = provider_value
        if provider_value != "ollama":
            state["api_key"] = state.get(f"{provider_value}_api_key", "")
            api_key_placeholder = f"{provider_value.title()} API密钥"
            api_key_interactive = True
        else:
            state["api_key"] = ""  # ollama不需要API密钥
            api_key_placeholder = "不需要API密钥"
            api_key_interactive = False
        
        # 显示或隐藏Ollama设置
        ollama_settings_visible = provider_value == "ollama"
        
        # 如果选择Ollama，自动获取模型列表
        ollama_models = []
        if ollama_settings_visible:
            ollama_models = get_ollama_models(state["ollama_server_url"])
            if ollama_models:
                state["ollama_model"] = ollama_models[0]
            else:
                state["ollama_model"] = ""
        
        # Calls to update other components UI
        api_key_update = gr.update(
            placeholder=api_key_placeholder,
            value=state["api_key"],
            interactive=api_key_interactive
        )
        ollama_settings_update = gr.update(visible=ollama_settings_visible)
        ollama_params_update = gr.update(visible=ollama_settings_visible)
        
        # 更新Ollama模型选择器
        ollama_model_selector_update = gr.update(
            choices=ollama_models if ollama_models else ["无可用模型或无法连接到Ollama服务器"],
            value=state["ollama_model"] if ollama_models else "",
            interactive=bool(ollama_models)
        )
        
        return api_key_update, ollama_settings_update, ollama_model_selector_update, ollama_params_update
    
    def update_ollama_server(server_url, state):
        """更新Ollama服务器URL并刷新模型列表"""
        state["ollama_server_url"] = server_url
        
        # 获取新服务器上的模型列表
        ollama_models = get_ollama_models(server_url)
        if ollama_models:
            state["ollama_model"] = ollama_models[0]
        else:
            state["ollama_model"] = ""
        
        # 更新模型选择器
        ollama_model_selector_update = gr.update(
            choices=ollama_models if ollama_models else ["无可用模型或无法连接到Ollama服务器"],
            value=state["ollama_model"] if ollama_models else "",
            interactive=bool(ollama_models)
        )
        
        return ollama_model_selector_update

    def refresh_ollama_models(state):
        """刷新Ollama模型列表"""
        server_url = state["ollama_server_url"]
        ollama_models = get_ollama_models(server_url)
        
        if ollama_models:
            if not state["ollama_model"] or state["ollama_model"] not in ollama_models:
                state["ollama_model"] = ollama_models[0]
        else:
            state["ollama_model"] = ""
        
        # 更新模型选择器
        return gr.update(
            choices=ollama_models if ollama_models else ["无可用模型或无法连接到Ollama服务器"],
            value=state["ollama_model"] if ollama_models else "",
            interactive=bool(ollama_models)
        )

    def update_ollama_model(model_name, state):
        """更新选中的Ollama模型"""
        state["ollama_model"] = model_name

    def update_api_key(api_key_value, state):
        state["api_key"] = api_key_value
        state[f'{state["provider"]}_api_key'] = api_key_value

    def clear_chat(state):
        # Reset message-related state
        state["messages"] = []
        state["responses"] = {}
        state["tools"] = {}
        state['chatbot_messages'] = []
        return state['chatbot_messages']

    def update_ollama_params(temperature_val, num_ctx_val, top_p_val, top_k_val, state):
        """更新Ollama参数设置"""
        state["ollama_params"] = {
            "temperature": temperature_val,
            "num_ctx": num_ctx_val,
            "top_p": top_p_val,
            "top_k": top_k_val
        }
        print(f"Ollama参数已更新: {state['ollama_params']}")

    model.change(fn=update_model, inputs=[model, state], outputs=[provider, api_key, ollama_settings, ollama_model_selector, ollama_params_panel])
    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    provider.change(fn=update_provider, inputs=[provider, state], outputs=[api_key, ollama_settings, ollama_model_selector, ollama_params_panel])
    ollama_server.change(fn=update_ollama_server, inputs=[ollama_server, state], outputs=ollama_model_selector)
    ollama_model_selector.change(fn=update_ollama_model, inputs=[ollama_model_selector, state], outputs=None)
    refresh_models_btn.click(fn=refresh_ollama_models, inputs=[state], outputs=ollama_model_selector)
    api_key.change(fn=update_api_key, inputs=[api_key, state], outputs=None)
    chatbot.change(fn=clear_chat, inputs=[state], outputs=[chatbot])

    # 添加Ollama参数更新事件
    temperature.change(fn=update_ollama_params, inputs=[temperature, num_ctx, top_p, top_k, state], outputs=None)
    num_ctx.change(fn=update_ollama_params, inputs=[temperature, num_ctx, top_p, top_k, state], outputs=None)
    top_p.change(fn=update_ollama_params, inputs=[temperature, num_ctx, top_p, top_k, state], outputs=None)
    top_k.change(fn=update_ollama_params, inputs=[temperature, num_ctx, top_p, top_k, state], outputs=None)

    submit_button.click(process_input, [chat_input, state], chatbot)
    stop_button.click(stop_app, [state], None)
    
if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7888)