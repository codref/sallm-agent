DEFAULT_MODEL = "ollama/gemma4:e4b-it-qat"
DEFAULT_API_BASE = "http://localhost:11434"


def user(content):
    return {"role": "user", "content": content}


def assistant(content, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def system(content):
    return {"role": "system", "content": content}


def tool(content, tool_call_id):
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }
