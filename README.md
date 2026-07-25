# sallm-agent

Minimal tool-calling agent library (`sallm`) over [LiteLLM](https://github.com/BerriAI/litellm), aimed at local models via Ollama.

Tools are **small CLI programs** run as subprocesses. The model emits shell-style commands in a fenced `run` block — no JSON tool payloads, no native provider tool APIs.

## Setup

```bash
# pull the default local model
ollama pull gemma4:e4b-it-qat

# install the package (editable) + test deps
uv sync --extra dev
```

## Tool contract

| Rule | Detail |
|------|--------|
| Identity | Tool name = first argv token (`calc`, `dig`, `echo`) |
| Help | Every tool supports `--help` |
| Args | CLI flags / positionals only (never JSON blobs) |
| Success | exit 0; result = stdout |
| Failure | non-zero exit; stderr/stdout returned as the observation |
| Intermediate | stdout may start with `[intermediate]` (agent keeps going) |

The model invokes tools like this (multiple lines = concurrent processes):

````markdown
```run
calc --expression "2**10"
echo --text hello
```
````

If unsure of flags, it can run `toolname --help` inside a `run` block.

## Library

Register `CliTool` instances (argv prefix + summary). The library ships example CLI apps under `sallm.toolapps`.

```python
import sys
from sallm import Agent
from sallm.tools import CliTool

calc = CliTool(
    name="calc",
    argv=[sys.executable, "-m", "sallm.toolapps.calc"],
    summary="Evaluate a math expression. Flags: --expression EXPR.",
)
agent = Agent(tools={"calc": calc})
print(agent.ask("What is 2**10? Use the calc tool.")["answer"])
```

Runner helpers live in `sallm.tools`: `parse_run_blocks`, `run_tool`, `run_many`, `help_text`.

## CLI

The chat app registers `echo`, `calc`, and multi-step `dig` (file-backed state) from `sallm.cli.tools`.

```bash
uv run sallm chat
uv run sallm chat --model ollama/gemma4:e4b-it-qat
```

Standalone tool binaries (optional): `sallm-echo`, `sallm-calc`, `sallm-dig`.

Slash commands in the REPL: `/help`, `/clear`, `/history`, `/quit`.

## Tests

```bash
# runner tests always; e2e needs local Ollama + default model
uv run pytest tests/ -v
```
