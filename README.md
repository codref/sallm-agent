# sallm-agent

Minimal tool-calling agent library (`sallm`) over [LiteLLM](https://github.com/BerriAI/litellm), aimed at local models via Ollama.

## Setup

```bash
# pull the default local model
ollama pull gemma4:e4b-it-qat

# install the package (editable)
uv sync
```

## Library

Tools are provided by the consumer — the library ships `echo` only as an example.

```python
from sallm import Agent
from sallm.tools import EXAMPLE_TOOLS, echo

# No tools by default
agent = Agent()

# Or register your own (and/or the example echo tool)
agent = Agent(tools={"echo": echo})
# same as: Agent(tools=EXAMPLE_TOOLS)
```

## CLI

The chat app registers its own tools (sandboxed `calc`, multi-step demo `dig`) in `sallm.cli.tools`.

Tools can return an intermediate result by prefixing with `[intermediate]`; the agent then nudges the model to continue.

```bash
uv run sallm chat
uv run sallm chat --model ollama/gemma4:e4b-it-qat
```

Slash commands in the REPL: `/help`, `/clear`, `/history`, `/quit`.

After each turn the CLI shows decide/tool panels, the assistant reply, and metrics.
