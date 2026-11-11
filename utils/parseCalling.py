import regex as re
import json
import inspect

def extract_tool_calls(response, on_error=None):
    """解析 <FUNCTION>/<ARGS> 形式的工具调用。

    on_error 用于把参数解析失败交给调用方处理（界面里不该直接 print）。
    """
    pattern = r"<FUNCTION>\s*(\w+)\s+<ARGS>\s*(\{.*?\})"

    matches = re.findall(pattern, response, re.DOTALL)
    tool_calls = []

    for match in matches:
        tool_name = match[0]
        args_str = match[1]

        try:
            # 确保是合法 JSON 字符串
            args = json.loads(args_str)
            tool_calls.append({
                "name": tool_name,
                "args": args
            })
        except Exception as e:
            if on_error is not None:
                on_error(e)
    
    return tool_calls

def has_config_parameter(func):
    return 'config' in inspect.signature(func).parameters

