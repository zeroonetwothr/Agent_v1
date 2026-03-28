from openai import OpenAI
import os
import json
from datetime import datetime


print("代码已重新加载")


API_KEY = os.getenv("ARK_API_KEY")
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_ID = "doubao-seed-2-0-mini-260215"

if not API_KEY:
    raise ValueError("请先设置环境变量 ARK_API_KEY")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)
class BaseTool:
    name = ""
    description = ""
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False
    }

    def run(self, **kwargs):
        raise NotImplementedError

    def to_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "查询指定城市的天气情况"
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，比如北京"
            }
        },
        "required": ["city"],
        "additionalProperties": False
    }

    def run(self, city):
        fake_weather_data = {
            "北京": "北京今天晴，28摄氏度，微风。",
            "上海": "上海今天多云，30摄氏度，空气湿润。",
            "广州": "广州今天阵雨，29摄氏度，请带伞。"
        }
        return fake_weather_data.get(city, f"暂时查不到 {city} 的天气。")



class TimeTool(BaseTool):
    name = "get_current_time"
    description = "获取当前系统时间"
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False
    }

    def run(self):
        now = datetime.now()
        return now.strftime("现在时间是：%Y-%m-%d %H:%M:%S")


class CalculatorTool(BaseTool):
    name = "calculate"
    description = "计算一个数学表达式，比如 25*4+10"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要计算的数学表达式，比如 25*4+10"
            }
        },
        "required": ["expression"],
        "additionalProperties": False
    }

    def run(self, expression):
        try:
            allowed_chars = "0123456789+-*/(). "
            for char in expression:
                if char not in allowed_chars:
                    return "表达式中包含不允许的字符。"

            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果是：{result}"
        except Exception:
            return "计算失败，请检查表达式格式。"


tool_instances = [
    WeatherTool(),
    TimeTool(),
    CalculatorTool()
]

TOOLS_MAP = {tool.name: tool for tool in tool_instances}
tools = [tool.to_schema() for tool in tool_instances]

def call_model(messages, tools=None):
    """统一调用模型"""
    params = {
        "model": MODEL_ID,
        "messages": messages
    }

    if tools is not None:
        params["tools"] = tools
        params["tool_choice"] = "auto"

    response = client.chat.completions.create(**params)
    return response.choices[0].message


def execute_tool_call(tool_call):
    """执行单个工具调用"""
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    tool = TOOLS_MAP.get(function_name)

    if not tool:
        return "未知工具"

    try:
        return tool.run(**arguments)
    except Exception as e:
        return f"工具执行失败：{str(e)}"


def build_tool_calls_message(assistant_message):
    """把模型返回的 tool_calls 转成可写入 messages 的标准格式"""
    tool_calls_list = []

    for tool_call in assistant_message.tool_calls:
        item = {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments
            }
        }
        tool_calls_list.append(item)

    return {
        "role": "assistant",
        "content": assistant_message.content or "",
        "tool_calls": tool_calls_list
    }


messages = [
    {
        "role": "system",
        "content": (
            "你是一个多功能助手。"
            "当用户询问天气时，调用 get_weather。"
            "当用户询问当前时间时，调用 get_current_time。"
            "当用户提到算式、加减乘除、等于多少、结果是多少时，调用 calculate。"
            "如果问题不需要工具，就直接正常回答。"
        )
    }
]


MAX_STEPS = 5

while True:
    user_input = input("你：").strip()

    if user_input.lower() in ["exit", "quit", "q"]:
        print("Agent：再见！")
        break

    if not user_input:
        print("Agent：请输入内容。")
        continue

    messages.append({
        "role": "user",
        "content": user_input
    })

    step_count = 0

    while step_count < MAX_STEPS:
        step_count += 1

        assistant_message = call_model(messages, tools=tools)

        if assistant_message.tool_calls:
            messages.append(build_tool_calls_message(assistant_message))

            for tool_call in assistant_message.tool_calls:
                tool_result = execute_tool_call(tool_call)

                print(f"[第{step_count}步] 调用工具: {tool_call.function.name}")
                print(f"[第{step_count}步] 工具结果: {tool_result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

            continue

        final_answer = assistant_message.content or "我暂时不知道怎么回答。"
        print("Agent：", final_answer)

        messages.append({
            "role": "assistant",
            "content": final_answer
        })
        break

    else:
        fallback_answer = "思考步数过多，我先停下来。"
        print("Agent：", fallback_answer)

        messages.append({
            "role": "assistant",
            "content": fallback_answer
        })