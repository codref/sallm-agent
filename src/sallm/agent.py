"""Agent: LLM turns + ```run tool execution (observations land in the transcript).

Owns: ask() loop, message transcript, tool rounds.
Uses Prompt for all wording; last_prompt is the troubleshoot hook (exact LLM view).
Does not embed or index — tools are subprocess CLIs; context only reshapes the prompt view.

Tool I/O contract:
  - Model emits fenced ```run blocks → parse_run_blocks → run_many
  - Observations append as user messages prefixed with Prompt.RESULTS_PREFIX
  - stdout starting with [intermediate] triggers a continue nudge
"""

import time

from . import metrics as metrics_mod
from .llm import complete
from .messages import DEFAULT_API_BASE, DEFAULT_MODEL, assistant, system, user
from .prompt import Prompt
from .tools import (
    format_observations,
    normalize_registry,
    parse_run_blocks,
    run_many,
    tool_descriptions,
)


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
    ):
        self.model = model or DEFAULT_MODEL
        self.api_base = api_base or DEFAULT_API_BASE
        self.tools = normalize_registry(tools)
        self.max_steps = max_steps
        self.multi_step = multi_step
        self.system = system  # optional extra prefix on the system prompt
        self.trace = trace  # Tracer | None — never touch when None
        self.context = context  # optimizer | None — prepare() view for LLM
        self.prompt = self._build_prompt()
        self.last_prompt = None  # list[dict] | None — last messages sent to complete()
        self.messages = []
        self._ensure_system()

    def _build_prompt(self) -> Prompt:
        return Prompt(
            tools_text=tool_descriptions(self.tools),
            multi_step=self.multi_step,
            extra=self.system,
        )

    def _ensure_system(self):
        """Keep transcript system message aligned with current Prompt.render."""
        self.prompt = self._build_prompt()
        content = self.prompt.system()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = system(content)
        else:
            self.messages.insert(0, system(content))

    def clear(self):
        self.messages = []
        self.last_prompt = None
        self._ensure_system()
        ctx = self.context
        if ctx is not None:
            on_clear = getattr(ctx, "on_clear", None)
            if on_clear is not None:
                on_clear()

    def _prompt_messages(self, messages):
        view = list(messages)
        ctx = self.context
        if ctx is None:
            return view
        return ctx.prepare(view)

    def _complete(self, messages, turn_metrics):
        prompt = self._prompt_messages(messages)
        self.last_prompt = prompt  # exact payload for /prompt troubleshooting
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

    def _record_action(self, content, result, step_metrics, results):
        """Append assistant + tool observations to transcript; return step dict."""
        observation = format_observations(results)
        self.messages.append(assistant(content))
        self.messages.append(user(self.prompt.RESULTS_PREFIX + observation))
        pending = any(r.intermediate for r in results)
        return {
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
        return out

    def _inject_continue_nudge(self):
        nudge = self.prompt.CONTINUE_NUDGE
        self.messages.append(user(nudge))
        tr = self.trace
        if tr is not None:
            tr.nudge(nudge)

    def _force_answer_after_tools(self, steps, turn_metrics):
        """max_steps exhausted after tool use — one more complete; strip trailing runs."""
        result, step_metrics, turn_metrics = self._complete(
            self.messages, turn_metrics
        )
        answer = result.get("content") or ""
        # Forced path may still emit ```run — execute once more, then complete again.
        if parse_run_blocks(answer):
            results = self._run_tools(parse_run_blocks(answer))
            steps.append(
                self._record_action(answer, result, step_metrics, results)
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

    def ask(self, user_text):
        self.messages.append(user(user_text))
        steps = []
        turn_metrics = metrics_mod.empty_usage()
        acted = False
        awaiting_continue = False

        tr = self.trace
        if tr is not None:
            tr.turn_start(user_text, self.messages, model=self.model)

        for _ in range(self.max_steps):
            result, step_metrics, turn_metrics = self._complete(
                self.messages, turn_metrics
            )
            content = result.get("content") or ""
            commands = parse_run_blocks(content) if self.tools else []

            if commands:
                results = self._run_tools(commands)
                step = self._record_action(content, result, step_metrics, results)
                steps.append(step)
                acted = True

                # Intermediate stdout → nudge and keep looping until a final result.
                if step["intermediate"]:
                    self._inject_continue_nudge()
                    steps.append(
                        {"kind": "nudge", "raw": self.prompt.CONTINUE_NUDGE}
                    )
                    awaiting_continue = True
                    continue

                awaiting_continue = False
                if self.multi_step:
                    continue
                break

            # Plain text while an intermediate result is pending → reject and nudge.
            if awaiting_continue:
                nudge = self.prompt.EARLY_ANSWER_NUDGE
                self.messages.append(user(nudge))
                steps.append(
                    {
                        "kind": "rejected",
                        "raw": content,
                        "nudge": nudge,
                        "metrics": step_metrics,
                    }
                )
                if tr is not None:
                    tr.rejected(content, nudge=nudge)
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

        if acted:
            return self._force_answer_after_tools(steps, turn_metrics)

        fallback = "(no tool call produced)"
        steps.append({"kind": "final", "raw": fallback, "answer": fallback})
        return self._finish(fallback, steps, turn_metrics, stopped="max_steps")
