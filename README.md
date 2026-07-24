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

```python
from sallm import Agent

agent = Agent()  # ollama/gemma4:e4b-it-qat by default; tools: calc
result = agent.ask("What is sqrt(2) + 3**2?")
print(result["answer"])
print(result["metrics"])

# Optional: include the echo demo tool
from sallm.tools import DEFAULT_TOOLS, OPTIONAL_TOOLS
agent = Agent(tools={**DEFAULT_TOOLS, **OPTIONAL_TOOLS})
```

## CLI

```bash
uv run sallm chat
uv run sallm chat --model ollama/gemma4:e4b-it-qat
```

Slash commands in the REPL: `/help`, `/clear`, `/history`, `/quit`.

After each turn the CLI shows input/output tokens, elapsed time, context size, and any tool call steps.
