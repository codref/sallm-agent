"""Tool registry helpers and an example tool (`echo`).

Consumers pass their own `{name: callable}` map into `Agent(tools=...)`.

Tools may return an intermediate observation by prefixing the result with
`[intermediate]` (see `INTERMEDIATE_PREFIX`). The agent then injects a
continuation nudge and keeps the tool loop going.
"""

import inspect
import json

# Tools return this prefix when more tool rounds are required before a final answer.
INTERMEDIATE_PREFIX = "[intermediate]"


def is_intermediate(observation):
    """True if a tool result is an intermediate step (not a finished result)."""
    text = str(observation or "").lstrip()
    return text.lower().startswith(INTERMEDIATE_PREFIX)


def intermediate(message):
    """Build an intermediate tool observation string."""
    message = str(message or "").strip()
    if is_intermediate(message):
        return message
    return f"{INTERMEDIATE_PREFIX} {message}".rstrip()


def echo(text=""):
    """Echo text back unchanged.

    Use only when the user explicitly asks to echo or repeat text.
    Never use this to phrase your own reply.
    """
    return str(text)


# Example registry — not loaded by Agent unless the consumer passes it.
EXAMPLE_TOOLS = {
    "echo": echo,
}


def _string_arg_fallback(fn, text):
    """Map a bare string to the first string-like parameter when JSON is missing."""
    for pname, param in inspect.signature(fn).parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        return {pname: text}
    return text


def run_tool(tools, name, args):
    """Look up and call a tool. Unknown tools return an error string."""
    fn = tools.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'. Available: {', '.join(tools) or '(none)'}"
    try:
        if isinstance(args, str):
            args = args.strip()
            if not args:
                args = {}
            else:
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = _string_arg_fallback(fn, args)
        if isinstance(args, dict):
            return str(fn(**args))
        return str(fn(args))
    except Exception as exc:
        return f"Error running tool '{name}': {exc}"


def _param_schema(param):
    annotation = param.annotation
    if annotation is inspect.Parameter.empty or annotation is str:
        json_type = "string"
    elif annotation is int:
        json_type = "integer"
    elif annotation is float:
        json_type = "number"
    elif annotation is bool:
        json_type = "boolean"
    else:
        json_type = "string"
    schema = {"type": json_type}
    if param.default is not inspect.Parameter.empty:
        schema["default"] = param.default
    return schema


def tool_schemas(tools):
    """Build OpenAI-style tool definitions for litellm completion(tools=...)."""
    schemas = []
    for name, fn in tools.items():
        doc = (fn.__doc__ or "").strip()
        # Collapse docstring so providers get full usage guidance, not only the summary line
        description = " ".join(line.strip() for line in doc.splitlines() if line.strip()) or name
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for pname, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            properties[pname] = _param_schema(param)
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return schemas


def tool_descriptions(tools):
    """Human-readable tool list for the system prompt (from each tool's docstring)."""
    if not tools:
        return "(none)"
    lines = []
    for name, fn in tools.items():
        doc = (fn.__doc__ or "").strip() or "no description"
        # Indent continuation lines under the tool name
        doc_lines = [ln.rstrip() for ln in doc.splitlines()]
        summary = doc_lines[0].strip() if doc_lines else "no description"
        block = [f"- {name}: {summary}"]
        for ln in doc_lines[1:]:
            text = ln.strip()
            if text:
                block.append(f"  {text}")
        lines.append("\n".join(block))
    return "\n".join(lines)
