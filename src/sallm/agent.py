"""Agent: LLM turns + ```run tool execution (observations land in the transcript).

Tool I/O contract:
  - Model emits fenced ```run blocks → parse_run_blocks → run_many
  - Observations append as user messages prefixed with RESULTS_PREFIX
  - stdout starting with [intermediate] triggers a continue nudge
Context optimizers (optional) only reshape the prompt view; they do not run tools.
Consciousness loops (optional) inject ephemeral system addenda (tool advice only).
"""

import time

from . import metrics as metrics_mod
from .consciousness import join_addenda, normalize_consciousness
from .llm import complete
from .messages import DEFAULT_API_BASE, DEFAULT_MODEL, assistant, system, user
from .tools import (
    format_observations,
    normalize_registry,
    parse_run_blocks,
    run_many,
    tool_descriptions,
)

MULTI_STEP_ON = """Multi-step mode is ON.
If the user asked for several sequential operations, run one tool (or one batch) at a time.
After each tool result, either emit another ```run block or answer in plain text when finished.
If a tool result starts with [intermediate], the work is not done — run that tool again.
Do not invent tool output."""

MULTI_STEP_OFF = """Multi-step mode is OFF.
Prefer a single ```run block when possible.
Exception: if a tool result starts with [intermediate], you must run that tool again
until you get a final (non-intermediate) result — then answer in plain text.
Do not invent tool output."""

CONTINUE_NUDGE = (
    "The previous tool result was intermediate (not finished). "
    "Emit another ```run block to call the same or next required tool. "
    "Do not give a final answer yet."
)

EARLY_ANSWER_NUDGE = (
    "You replied with text before the tool work finished. "
    "An intermediate tool result is still pending. "
    "Emit a ```run block now. Do not answer the user yet."
)

SYSTEM = """You are a helpful assistant.

Default behavior: answer the user directly in plain text.
Do not run tools for greetings, identity questions, opinions, explanations,
or anything you can answer from your own knowledge.

When a tool-advice section is appended below, treat it as guidance for
this turn: prefer the named tools when relevant; still do not invent tool output.

Tools are small command-line programs. To use them, reply with a fenced run block
containing one command per line (shell-style flags, no JSON):

```run
toolname --flag value
another --flag value
```

Multiple lines run as concurrent processes in one step.
If you are unsure of a tool's flags, run `toolname --help` inside a ```run block first.
Never invent tool output. After finished tool results, reply in short plain text.

{multi_step_policy}

Available tools:
{tools}
"""

RESULTS_PREFIX = "Tool results:\n"


class Agent:
    def __init__(
        self,
        model=None,
        api_base=None,
        tools=None,
        system=None,
        max_steps=5,
        multi_step=True,
        trace=None,
        context=None,
        consciousness=None,
    ):
        self.model = model or DEFAULT_MODEL
        self.api_base = api_base or DEFAULT_API_BASE
        self.tools = normalize_registry(tools)
        self.max_steps = max_steps
        self.multi_step = multi_step
        self.system = system
        self.trace = trace  # Tracer | None — never touch when None
        self.context = context  # optimizer | None — prepare() view for LLM
        self.consciousness = normalize_consciousness(consciousness)
        self._turn_addendum = ""  # ephemeral; set per ask()
        self.messages = []
        self._ensure_system()

    def _multi_step_policy(self):
        return MULTI_STEP_ON if self.multi_step else MULTI_STEP_OFF

    def _base_system_content(self):
        base = SYSTEM.format(
            tools=tool_descriptions(self.tools),
            multi_step_policy=self._multi_step_policy(),
        )
        if self.system:
            return self.system.rstrip() + "\n\n" + base
        return base

    def _ensure_system(self):
        """Keep transcript system message as stable base (no consciousness addenda)."""
        content = self._base_system_content()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = system(content)
        else:
            self.messages.insert(0, system(content))

    def clear(self):
        self.messages = []
        self._turn_addendum = ""
        self._ensure_system()
        ctx = self.context
        if ctx is not None:
            on_clear = getattr(ctx, "on_clear", None)
            if on_clear is not None:
                on_clear()

    def _run_consciousness(self) -> str:
        parts = []
        for layer in self.consciousness:
            advise = getattr(layer, "advise", None)
            if advise is None:
                continue
            part = advise(self.messages, self.tools)
            if part and str(part).strip():
                parts.append(str(part).strip())
        return join_addenda(parts)

    def _prompt_messages(self, messages):
        view = list(messages)
        if self._turn_addendum and view and view[0].get("role") == "system":
            base = view[0].get("content") or ""
            view[0] = system(base.rstrip() + "\n\n" + self._turn_addendum)
        ctx = self.context
        if ctx is None:
            return view
        return ctx.prepare(view)

    def _complete(self, messages, turn_metrics):
        prompt = self._prompt_messages(messages)
        result = complete(
            model=self.model,
            messages=prompt,
            api_base=self.api_base,
        )
        step_metrics = metrics_mod.from_llm_result(result)
        turn_metrics = metrics_mod.add_usage(turn_metrics, step_metrics)
        tr = self.trace
        if tr is not None:
            tr.llm(
                model=self.model,
                metrics=step_metrics,
                content=result.get("content") or "",
                reasoning=result.get("reasoning"),
                messages=prompt,
            )
        return result, step_metrics, turn_metrics

    def _run_tools(self, commands):
        tr = self.trace
        if tr is None:
            return run_many(self.tools, commands)
        started = time.perf_counter()
        results = run_many(self.tools, commands)
        batch_ms = (time.perf_counter() - started) * 1000
        per = batch_ms / max(len(results), 1)
        for r in results:
            tr.tool(
                name=r.name,
                command=r.command,
                observation=r.observation,
                stdout=r.stdout,
                stderr=r.stderr,
                returncode=r.returncode,
                intermediate=r.intermediate,
                elapsed_ms=per,
            )
        return results

    def _finish(self, answer, steps, turn_metrics, stopped=None):
        self.messages.append(assistant(answer))
        prompt_view = self._prompt_messages(self.messages)
        summary = metrics_mod.summarize(
            turn_metrics,
            context_messages=len(self.messages),
            prompt_messages=len(prompt_view),
        )
        tr = self.trace
        if tr is not None:
            tr.turn_end(
                answer=answer,
                metrics=summary,
                messages=self.messages,
                stopped=stopped,
            )
        out = {
            "answer": answer,
            "steps": steps,
            "metrics": summary,
        }
        if stopped:
            out["stopped"] = stopped
        if self._turn_addendum:
            out["consciousness"] = self._turn_addendum
        return out

    def _inject_continue_nudge(self):
        self.messages.append(user(CONTINUE_NUDGE))
        tr = self.trace
        if tr is not None:
            tr.nudge(CONTINUE_NUDGE)

    def ask(self, user_text):
        self.messages.append(user(user_text))
        steps = []
        turn_metrics = metrics_mod.empty_usage()
        acted = False
        awaiting_continue = False

        tr = self.trace
        if tr is not None:
            tr.turn_start(user_text, self.messages, model=self.model)

        self._turn_addendum = self._run_consciousness()
        if self._turn_addendum and tr is not None:
            tr.consciousness(self._turn_addendum)

        for _ in range(self.max_steps):
            result, step_metrics, turn_metrics = self._complete(
                self.messages, turn_metrics
            )
            content = result.get("content") or ""
            commands = parse_run_blocks(content) if self.tools else []

            if commands:
                self.messages.append(assistant(content))
                results = self._run_tools(commands)
                observation = format_observations(results)
                self.messages.append(user(RESULTS_PREFIX + observation))
                pending = any(r.intermediate for r in results)
                steps.append(
                    {
                        "kind": "action",
                        "raw": content,
                        "reasoning": result.get("reasoning"),
                        "commands": [list(r.command) for r in results],
                        "tool_calls": [
                            {
                                "action": r.name,
                                "action_input": " ".join(r.command[1:]),
                                "observation": r.observation,
                                "intermediate": r.intermediate,
                                "returncode": r.returncode,
                            }
                            for r in results
                        ],
                        "observation": observation,
                        "intermediate": pending,
                        "metrics": step_metrics,
                    }
                )
                acted = True

                if pending:
                    self._inject_continue_nudge()
                    steps.append({"kind": "nudge", "raw": CONTINUE_NUDGE})
                    awaiting_continue = True
                    continue

                awaiting_continue = False
                if self.multi_step:
                    continue
                break

            # Plain text reply (no run block).
            if awaiting_continue:
                self.messages.append(user(EARLY_ANSWER_NUDGE))
                steps.append(
                    {
                        "kind": "rejected",
                        "raw": content,
                        "nudge": EARLY_ANSWER_NUDGE,
                        "metrics": step_metrics,
                    }
                )
                if tr is not None:
                    tr.rejected(content, nudge=EARLY_ANSWER_NUDGE)
                continue

            steps.append(
                {
                    "kind": "final",
                    "raw": content,
                    "answer": content,
                    "reasoning": result.get("reasoning"),
                    "metrics": step_metrics,
                }
            )
            return self._finish(content, steps, turn_metrics)

        # Hit max_steps still in tool mode → force a text answer.
        if acted:
            result, step_metrics, turn_metrics = self._complete(
                self.messages, turn_metrics
            )
            answer = result.get("content") or ""
            # Strip any trailing run blocks from the forced answer path.
            if parse_run_blocks(answer):
                # One last execute, then ask again without tools expectation.
                self.messages.append(assistant(answer))
                results = self._run_tools(parse_run_blocks(answer))
                self.messages.append(
                    user(RESULTS_PREFIX + format_observations(results))
                )
                steps.append(
                    {
                        "kind": "action",
                        "raw": answer,
                        "tool_calls": [
                            {
                                "action": r.name,
                                "action_input": " ".join(r.command[1:]),
                                "observation": r.observation,
                                "intermediate": r.intermediate,
                            }
                            for r in results
                        ],
                        "metrics": step_metrics,
                    }
                )
                result, step_metrics, turn_metrics = self._complete(
                    self.messages, turn_metrics
                )
                answer = result.get("content") or ""
            steps.append(
                {
                    "kind": "final",
                    "raw": answer,
                    "answer": answer,
                    "reasoning": result.get("reasoning"),
                    "metrics": step_metrics,
                }
            )
            return self._finish(answer, steps, turn_metrics, stopped="max_steps")

        fallback = "(no tool call produced)"
        steps.append({"kind": "final", "raw": fallback, "answer": fallback})
        return self._finish(fallback, steps, turn_metrics, stopped="max_steps")
