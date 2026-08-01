"""Public package exports — agent + tool runner types."""

from .agent import Agent
from .consciousness import ToolAdvisor
from .messages import DEFAULT_API_BASE, DEFAULT_MODEL
from .tools import CliTool, ToolResult

__all__ = [
    "Agent",
    "CliTool",
    "ToolResult",
    "ToolAdvisor",
    "DEFAULT_MODEL",
    "DEFAULT_API_BASE",
]
