"""calc — evaluate a sandboxed math expression."""

from __future__ import annotations

import argparse
import ast
import math
import operator
import sys

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
_SAFE_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
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
        return fn(*[self.visit(a) for a in node.args])

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


def evaluate(expression: str) -> str:
    expression = (expression or "").strip()
    if not expression:
        return "Error: empty expression"
    try:
        tree = ast.parse(expression, mode="eval")
        result = _SafeMath().visit(tree)
    except Exception as exc:
        return f"Error: {exc}"
    return str(result)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="calc",
        description=(
            "Evaluate a math expression in a sandbox. "
            "Examples: 'sqrt(2) + 3**2', 'sin(pi/2)', '2**10'."
        ),
    )
    parser.add_argument(
        "--expression",
        "-e",
        required=True,
        help="Math expression to evaluate",
    )
    args = parser.parse_args(argv)
    out = evaluate(args.expression)
    sys.stdout.write(out + ("\n" if not out.endswith("\n") else ""))
    return 1 if out.startswith("Error:") else 0


if __name__ == "__main__":
    raise SystemExit(main())
