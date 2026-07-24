import ast
import inspect
import json
import math
import operator

def echo(text=""):
    """Echo text back unchanged.

    Use only when the user explicitly asks to echo or repeat text.
    Never use this to phrase your own reply.
    """
    return str(text)


# --- math sandbox -----------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCS = {
    name: getattr(math, name)
    for name in (
        "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh",
        "ceil", "comb", "copysign", "cos", "cosh", "degrees", "dist",
        "erf", "erfc", "exp", "expm1", "fabs", "factorial", "floor",
        "fmod", "frexp", "fsum", "gamma", "gcd", "hypot", "isclose",
        "isfinite", "isinf", "isnan", "isqrt", "lcm", "ldexp", "lgamma",
        "log", "log10", "log1p", "log2", "modf", "nextafter", "perm",
        "pow", "prod", "radians", "remainder", "sin", "sinh", "sqrt",
        "tan", "tanh", "trunc", "ulp",
    )
    if hasattr(math, name)
}
_SAFE_FUNCS.update(
    {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }
)
_SAFE_CONSTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
}


class _SafeMath(ast.NodeVisitor):
    """Evaluate an expression AST using only whitelisted ops/names."""

    def visit(self, node):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise ValueError(f"disallowed expression: {type(node).__name__}")
        return method(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"disallowed constant: {node.value!r}")

    def visit_Name(self, node):
        if node.id in _SAFE_CONSTS:
            return _SAFE_CONSTS[node.id]
        if node.id in _SAFE_FUNCS:
            return _SAFE_FUNCS[node.id]
        raise ValueError(f"unknown name: {node.id}")

    def visit_BinOp(self, node):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"disallowed operator: {type(node.op).__name__}")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"disallowed unary: {type(node.op).__name__}")
        return op(self.visit(node.operand))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only simple function calls are allowed")
        fn = _SAFE_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"disallowed function: {node.func.id}")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        args = [self.visit(a) for a in node.args]
        return fn(*args)

    def visit_Compare(self, node):
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            else:
                raise ValueError(f"disallowed compare: {type(op).__name__}")
            if not ok:
                return False
            left = right
        return True


def calc(expression=""):
    """Evaluate a math expression in a sandbox.

    Use only for numeric or math evaluation the user asked you to compute.
    Examples: 'sqrt(2) + 3**2', 'sin(pi/2)', '2**10'.
    Do not use for non-math questions.
    """
    expression = (expression or "").strip()
    if not expression:
        return "Error: empty expression"
    try:
        tree = ast.parse(expression, mode="eval")
        result = _SafeMath().visit(tree)
    except Exception as exc:
        return f"Error: {exc}"
    return str(result)


DEFAULT_TOOLS = {
    "calc": calc,
}

# Optional demo tool — not registered by default (small models over-call it).
OPTIONAL_TOOLS = {
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
