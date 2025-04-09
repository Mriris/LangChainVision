import json
from collections.abc import Callable
from typing import cast, Callable
import uuid
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import requests

from anthropic import APIResponse
from anthropic.types import ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam, BetaUsage

from agent.llm_utils.oaiclient import run_oai_interleaved
from agent.llm_utils.groqclient import run_groq_interleaved
from agent.llm_utils.ollamaclient import run_ollama_interleaved
from agent.llm_utils.utils import is_image_path
import time
import re
from json_repair import repair_json

OUTPUT_DIR = "./tmp/outputs"

def extract_data(input_string, data_type):
    # Regular expression to extract content starting from '```python' until the end if there are no closing backticks
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    # Extract content
    # re.DOTALL allows '.' to match newlines as well
    matches = re.findall(pattern, input_string, re.DOTALL)
    # Return the first match if exists, trimming whitespace and ignoring potential closing backticks
    return matches[0][0].strip() if matches else input_string

class VLMAgent:
    def __init__(
        self,
        model: str, 
        provider: str, 
        api_key: str,
        output_callback: Callable, 
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
        ollama_server_url: str = None,
        ollama_model: str = None,
    ):
        if model == "omniparser + gpt-4o":
            self.model = "gpt-4o-2024-11-20"
        elif model == "omniparser + R1":
            self.model = "deepseek-r1-distill-llama-70b"
        elif model == "omniparser + qwen2.5vl":
            self.model = "qwen2.5-vl-72b-instruct"
        elif model == "omniparser + o1":
            self.model = "o1"
        elif model == "omniparser + o3-mini":
            self.model = "o3-mini"
        elif model == "omniparser + ollama":
            self.model = ollama_model or "llama3"
        else:
            raise ValueError(f"Model {model} not supported")
        

        self.provider = provider
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback
        self.ollama_server_url = ollama_server_url or "http://localhost:11434"
        self.ollama_model = ollama_model or "llama3"

        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0

        self.system = ''
           
    def _api_response_callback(self, response: APIResponse):
        if hasattr(self, "api_response_callback") and callable(self.api_response_callback):
            self.api_response_callback(response)
           
    def __call__(self, messages: list, parsed_screen: list[str, list, dict], ollama_params=None):
        self.step_count += 1
        image_base64 = parsed_screen['original_screenshot_base64']
        latency_omniparser = parsed_screen['latency']
        self.output_callback(f'-- Step {self.step_count}: --', sender="bot")
        screen_info = str(parsed_screen['screen_info'])
        screenshot_uuid = parsed_screen['screenshot_uuid']
        screen_width, screen_height = parsed_screen['width'], parsed_screen['height']

        boxids_and_labels = parsed_screen["screen_info"]
        system = self._get_system_prompt(boxids_and_labels)

        # drop looping actions msg, byte image etc
        planner_messages = messages
        _remove_som_images(planner_messages)
        _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)

        if isinstance(planner_messages[-1], dict):
            if not isinstance(planner_messages[-1]["content"], list):
                planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_{screenshot_uuid}.png")
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_som_{screenshot_uuid}.png")

        start = time.time()
        # 首先判断提供者是否为Ollama，优先使用Ollama API
        if self.provider == "ollama":
            # 调用Ollama API
            vlm_response, token_usage = run_ollama_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.ollama_model,
                server_url=self.ollama_server_url,
                max_tokens=self.max_tokens,
                temperature=0,
                ollama_params=ollama_params
            )
            print(f"ollama token usage: {token_usage}")
            self.total_token_usage += token_usage
            # Ollama是本地运行的，没有成本
            self.total_cost += 0
        # 然后根据模型名称判断使用哪个API
        elif "gpt" in self.model or "o1" in self.model or "o3-mini" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                provider_base_url="https://api.openai.com/v1",
                temperature=0,
            )
            print(f"oai token usage: {token_usage}")
            self.total_token_usage += token_usage
            if 'gpt' in self.model:
                self.total_cost += (token_usage * 2.5 / 1000000)  # https://openai.com/api/pricing/
            elif 'o1' in self.model:
                self.total_cost += (token_usage * 15 / 1000000)  # https://openai.com/api/pricing/
            elif 'o3-mini' in self.model:
                self.total_cost += (token_usage * 1.1 / 1000000)  # https://openai.com/api/pricing/
        elif "r1" in self.model:
            vlm_response, token_usage = run_groq_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
            )
            print(f"groq token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 0.99 / 1000000)
        elif "qwen" in self.model:
            try:
                # 限制最大输入长度，减少输入token数量的方法
                # 1. 只保留更少的历史图像
                qwen_only_n_most_recent_images = min(2, self.only_n_most_recent_images if self.only_n_most_recent_images is not None else 2)
                _maybe_filter_to_n_most_recent_images(planner_messages, qwen_only_n_most_recent_images)
                print(f"为Qwen模型限制历史图像数量为{qwen_only_n_most_recent_images}")
                
                # 2. 使用较小的max_tokens值
                qwen_max_tokens = min(1024, self.max_tokens)  # 进一步降低max_tokens值
                
                vlm_response, token_usage = run_oai_interleaved(
                    messages=planner_messages,
                    system=system,
                    model_name=self.model,
                    api_key=self.api_key,
                    max_tokens=qwen_max_tokens,
                    provider_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    temperature=0,
                )
                print(f"qwen token usage: {token_usage}")
                self.total_token_usage += token_usage
                self.total_cost += (token_usage * 2.2 / 1000000)  # https://help.aliyun.com/zh/model-studio/getting-started/models?spm=a2c4g.11186623.0.0.74b04823CGnPv7#fe96cfb1a422a
            except Exception as e:
                print(f"Error in qwen API call: {e}")
                # 为确保在API调用失败时程序仍能继续运行，创建一个默认响应
                vlm_response = f"Error: {e}"
                token_usage = 0
        else:
            raise ValueError(f"Model {self.model} not supported")
        latency_vlm = time.time() - start
        self.output_callback(f"LLM: {latency_vlm:.2f}s, OmniParser: {latency_omniparser:.2f}s", sender="bot")

        print(f"{vlm_response}")
        
        if self.print_usage:
            print(f"Total token so far: {self.total_token_usage}. Total cost so far: $USD{self.total_cost:.5f}")
        
        # 添加异常处理，确保vlm_response_json是字典类型
        try:
            vlm_response_json = extract_data(vlm_response, "json")
            try:
                vlm_response_json = json.loads(repair_json(vlm_response_json))
                
                # 如果vlm_response_json不是字典类型，则创建一个默认字典
                if not isinstance(vlm_response_json, dict):
                    print(f"Warning: 返回的不是有效的JSON对象: {vlm_response_json}")
                    vlm_response_json = {
                        "Reasoning": f"LLM返回了无效的响应: {vlm_response}",
                        "Next Action": "None"
                    }
            except Exception as e:
                print(f"Warning: JSON解析错误: {e}, 原始响应: {vlm_response}")
                vlm_response_json = {
                    "Reasoning": f"JSON解析错误: {e}\n原始响应: {vlm_response}",
                    "Next Action": "None"
                }
        except Exception as e:
            print(f"Error extracting data: {e}, 原始响应: {vlm_response}")
            vlm_response_json = {
                "Reasoning": f"数据提取错误: {e}\n原始响应: {vlm_response}",
                "Next Action": "None"
            }

        img_to_show_base64 = parsed_screen["som_image_base64"]
        if "Box ID" in vlm_response_json:
            try:
                bbox = parsed_screen["parsed_content_list"][int(vlm_response_json["Box ID"])]["bbox"]
                vlm_response_json["box_centroid_coordinate"] = [int((bbox[0] + bbox[2]) / 2 * screen_width), int((bbox[1] + bbox[3]) / 2 * screen_height)]
                img_to_show_data = base64.b64decode(img_to_show_base64)
                img_to_show = Image.open(BytesIO(img_to_show_data))

                draw = ImageDraw.Draw(img_to_show)
                x, y = vlm_response_json["box_centroid_coordinate"] 
                radius = 10
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
                draw.ellipse((x - radius*3, y - radius*3, x + radius*3, y + radius*3), fill=None, outline='red', width=2)

                buffered = BytesIO()
                img_to_show.save(buffered, format="PNG")
                img_to_show_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except:
                print(f"Error parsing: {vlm_response_json}")
                pass
        self.output_callback(f'<img src="data:image/png;base64,{img_to_show_base64}">', sender="bot")
        self.output_callback(
                    f'<details>'
                    f'  <summary>Parsed Screen elemetns by OmniParser</summary>'
                    f'  <pre>{screen_info}</pre>'
                    f'</details>',
                    sender="bot"
                )
        vlm_plan_str = ""
        for key, value in vlm_response_json.items():
            if key == "Reasoning":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'

        # construct the response so that anthropicExcutor can execute the tool
        response_content = [BetaTextBlock(text=vlm_plan_str, type='text')]
        if 'box_centroid_coordinate' in vlm_response_json:
            move_cursor_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': 'mouse_move', 'coordinate': vlm_response_json["box_centroid_coordinate"]},
                                            name='computer', type='tool_use')
            response_content.append(move_cursor_block)

        # 检查 Next Action 是否存在，如果不存在则设置为 "None"
        if "Next Action" not in vlm_response_json:
            vlm_response_json["Next Action"] = "None"
            print("Warning: 'Next Action' not found in response, defaulting to 'None'")

        if vlm_response_json["Next Action"] == "None":
            print("Task paused/completed.")
        elif vlm_response_json["Next Action"] == "type":
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                        input={'action': vlm_response_json["Next Action"], 'text': vlm_response_json["value"]},
                                        name='computer', type='tool_use')
            response_content.append(sim_content_block)
        else:
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': vlm_response_json["Next Action"]},
                                            name='computer', type='tool_use')
            response_content.append(sim_content_block)
        response_message = BetaMessage(id=f'toolu_{uuid.uuid4()}', content=response_content, model='', role='assistant', type='message', stop_reason='tool_use', usage=BetaUsage(input_tokens=0, output_tokens=0))
        return response_message, vlm_response_json

    def _get_system_prompt(self, screen_info: str = ""):
        # 检查是否为纯文本模型
        is_text_only_model = (self.provider == "ollama" and 
                            not any(x in self.ollama_model.lower() for x in ["vl", "vision", "visual", "llava", "image"]))
        
        # 针对不同模型使用不同的系统提示词
        if "qwen" in self.ollama_model.lower() and self.provider == "ollama":
            if is_text_only_model:
                # 针对纯文本Qwen模型的简化提示词
                main_section = f"""
你是一个计算机界面操作助手。以下是当前屏幕上可交互元素的描述：

{screen_info}

请分析上述界面元素，并确定下一步操作。你可以执行以下操作：
- type: 输入文本
- left_click: 点击左键
- right_click: 点击右键
- double_click: 双击左键
- hover: 鼠标悬停
- scroll_up: 向上滚动
- scroll_down: 向下滚动
- wait: 等待1秒

请以JSON格式输出你的分析结果，例如：
```json
{{
    "Reasoning": "分析当前屏幕内容...",
    "Next Action": "操作类型",
    "Box ID": 数字,
    "value": "文本内容"  // 仅当操作为type时
}}
```
"""
            else:
                # 原有的Qwen模型提示词（支持多模态）
                main_section = f"""
你是一个计算机视觉界面操作助手。通过分析屏幕截图和可交互元素，你能帮助用户完成Windows系统上的各种任务。

你正在操作Windows设备，可以使用鼠标和键盘与计算机界面交互，但仅限于GUI界面操作（不能访问终端或应用程序菜单）。

当前屏幕上所有可检测到的边界框元素及其描述如下:
{screen_info}

你可用的"下一步操作"仅包括：
- type: 输入文本字符串
- left_click: 将鼠标移动到指定ID的元素并单击左键
- right_click: 将鼠标移动到指定ID的元素并单击右键
- double_click: 将鼠标移动到指定ID的元素并双击左键
- hover: 将鼠标悬停在指定ID的元素上
- scroll_up: 向上滚动屏幕以查看之前的内容
- scroll_down: 向下滚动屏幕，适用于所需按钮不可见或需要查看更多内容的情况
- wait: 等待1秒钟让设备加载或响应

基于屏幕截图中的视觉信息和检测到的边界框，确定下一步操作、要操作的边界框ID（如果操作是"type"、"hover"、"scroll_up"、"scroll_down"或"wait"，则不需要指定边界框ID），以及要输入的值（如果操作是"type"）。

输出格式必须是如下JSON:
```json
{{
    "Reasoning": "分析当前屏幕内容，考虑历史记录，然后详细描述你的逐步思考过程，每次从可用操作中选择一个操作。",
    "Next Action": "操作类型" | "None", 
    "Box ID": 数字,
    "value": "xxx" // 仅当操作为type时才提供value字段，否则不要包含value键
}}
```

示例:
```json
{{  
    "Reasoning": "当前屏幕显示了谷歌搜索结果，我在之前的操作中已经在谷歌上搜索了amazon。现在我需要点击第一个搜索结果前往amazon.com网站。",
    "Next Action": "left_click",
    "Box ID": 5
}}
```

另一个示例:
```json
{{
    "Reasoning": "当前屏幕显示了亚马逊的首页。没有之前的操作记录。因此我需要在搜索栏中输入"Apple watch"。",
    "Next Action": "type",
    "Box ID": 3,
    "value": "Apple watch"
}}
```

再一个示例:
```json
{{
    "Reasoning": "当前屏幕没有显示'提交'按钮，我需要向下滚动查看按钮是否可用。",
    "Next Action": "scroll_down"
}}
```

重要提示:
1. 每次只执行一个操作。
2. 详细分析当前屏幕，并通过查看历史记录反思已完成的操作，然后描述你实现任务的逐步思考过程。
3. 在"Next Action"中明确指定下一步操作。
4. 不要包含其他操作，如键盘快捷键。
5. 当任务完成时不要执行额外操作，应在json字段中说明"Next Action": "None"。
6. 如果遇到登录信息页面、验证码页面，或认为需要用户许可才能执行下一步操作，应在json字段中说明"Next Action": "None"。
7. 避免连续多次选择相同的操作/元素，如果发生这种情况，考虑可能出错的原因，并预测不同的操作。
"""
        else:
            # 默认系统提示词
            if is_text_only_model:
                # 针对非Qwen的纯文本模型的简化提示词
                main_section = f"""
You are a computer interface operation assistant. Below is a description of the interactive elements on the current screen:

{screen_info}

Please analyze the above interface elements and determine the next action. You can perform the following actions:
- type: Input text
- left_click: Click the left mouse button
- right_click: Click the right mouse button
- double_click: Double click the left mouse button
- hover: Hover the mouse
- scroll_up: Scroll up
- scroll_down: Scroll down
- wait: Wait for 1 second

Please output your analysis in JSON format, for example:
```json
{{
    "Reasoning": "Analyze the current screen...",
    "Next Action": "action_type",
    "Box ID": number,
    "value": "text content"  // only when the action is type
}}
```
"""
            else:
                # 原有的默认提示词（支持多模态）
                main_section = f"""
You are using a Windows device.
You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
You can only interact with the desktop GUI (no terminal or application menu access).

You may be given some history plan and actions, this is the response from the previous loop.
You should carefully consider your plan base on the task, screenshot, and history actions.

Here is the list of all detected bounding boxes by IDs on the screen and their description:{screen_info}

Your available "Next Action" only include:
- type: types a string of text.
- left_click: move mouse to box id and left clicks.
- right_click: move mouse to box id and right clicks.
- double_click: move mouse to box id and double clicks.
- hover: move mouse to box id.
- scroll_up: scrolls the screen up to view previous content.
- scroll_down: scrolls the screen down, when the desired button is not visible, or you need to see more content. 
- wait: waits for 1 second for the device to load or respond.

Based on the visual information from the screenshot image and the detected bounding boxes, please determine the next action, the Box ID you should operate on (if action is one of 'type', 'hover', 'scroll_up', 'scroll_down', 'wait', there should be no Box ID field), and the value (if the action is 'type') in order to complete the task.

Output format:
```json
{{
    "Reasoning": str, # describe what is in the current screen, taking into account the history, then describe your step-by-step thoughts on how to achieve the task, choose one action from available actions at a time.
    "Next Action": "action_type, action description" | "None" # one action at a time, describe it in short and precisely. 
    "Box ID": n,
    "value": "xxx" # only provide value field if the action is type, else don't include value key
}}
```

One Example:
```json
{{  
    "Reasoning": "The current screen shows google result of amazon, in previous action I have searched amazon on google. Then I need to click on the first search results to go to amazon.com.",
    "Next Action": "left_click",
    "Box ID": m
}}
```

Another Example:
```json
{{
    "Reasoning": "The current screen shows the front page of amazon. There is no previous action. Therefore I need to type "Apple watch" in the search bar.",
    "Next Action": "type",
    "Box ID": n,
    "value": "Apple watch"
}}
```

Another Example:
```json
{{
    "Reasoning": "The current screen does not show 'submit' button, I need to scroll down to see if the button is available.",
    "Next Action": "scroll_down",
}}
```

IMPORTANT NOTES:
1. You should only give a single action at a time.
"""

        thinking_model = "r1" in self.model
        if not thinking_model:
            main_section += """
2. You should give an analysis to the current screen, and reflect on what has been done by looking at the history, then describe your step-by-step thoughts on how to achieve the task.

"""
        else:
            main_section += """
2. In <think> XML tags give an analysis to the current screen, and reflect on what has been done by looking at the history, then describe your step-by-step thoughts on how to achieve the task. In <output> XML tags put the next action prediction JSON.

"""
        # 通用附加说明
        if "qwen" not in self.ollama_model.lower() or self.provider != "ollama":
            main_section += """
3. Attach the next action prediction in the "Next Action".
4. You should not include other actions, such as keyboard shortcuts.
5. When the task is completed, don't complete additional actions. You should say "Next Action": "None" in the json field.
6. The tasks involve buying multiple products or navigating through multiple pages. You should break it into subgoals and complete each subgoal one by one in the order of the instructions.
7. avoid choosing the same action/elements multiple times in a row, if it happens, reflect to yourself, what may have gone wrong, and predict a different action.
8. If you are prompted with login information page or captcha page, or you think it need user's permission to do the next action, you should say "Next Action": "None" in the json field.
""" 

        return main_section

def _remove_som_images(messages):
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            msg["content"] = [
                cnt for cnt in msg_content 
                if not (isinstance(cnt, str) and 'som' in cnt and is_image_path(cnt))
            ]


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place
    """
    if images_to_keep is None:
        return messages

    total_images = 0
    for msg in messages:
        for cnt in msg.get("content", []):
            if isinstance(cnt, str) and is_image_path(cnt):
                total_images += 1
            elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                for content in cnt.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        total_images += 1

    images_to_remove = total_images - images_to_keep
    
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            new_content = []
            for cnt in msg_content:
                # Remove images from SOM or screenshot as needed
                if isinstance(cnt, str) and is_image_path(cnt):
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                # VLM shouldn't use anthropic screenshot tool so shouldn't have these but in case it does, remove as needed
                elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                    new_tool_result_content = []
                    for tool_result_entry in cnt.get("content", []):
                        if isinstance(tool_result_entry, dict) and tool_result_entry.get("type") == "image":
                            if images_to_remove > 0:
                                images_to_remove -= 1
                                continue
                        new_tool_result_content.append(tool_result_entry)
                    cnt["content"] = new_tool_result_content
                # Append fixed content to current message's content list
                new_content.append(cnt)
            msg["content"] = new_content