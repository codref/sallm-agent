"""Chat-app tools — registered by the CLI, not by the sallm library."""

import ast
import math
import operator

from sallm.tools import intermediate

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


CHAT_TOOLS = {
    "calc": calc,
}


# --- fake multi-step tool ---------------------------------------------------

_DIG_PROGRESS = {}


def reset_dig_state():
    """Clear dig progress (e.g. on /clear)."""
    _DIG_PROGRESS.clear()


def dig(site="default"):
    """Dig at a site for treasure. Needs several digs at the same site.

    Early calls return intermediate results (prefixed with [intermediate]).
    Keep calling dig with the same site until you get a final non-intermediate result.
    Do not invent the treasure — only report what dig returns.
    """
    site = (site or "default").strip() or "default"
    n = _DIG_PROGRESS.get(site, 0) + 1
    _DIG_PROGRESS[site] = n
    if n == 1:
        return intermediate(
            f"At '{site}' you found loose soil. Dig again at the same site."
        )
    if n == 2:
        return intermediate(
            f"At '{site}' you uncovered a locked chest. Dig once more at the same site."
        )
    _DIG_PROGRESS[site] = 0
    return f"At '{site}' you found gold coins worth 42."


CHAT_TOOLS["dig"] = dig
