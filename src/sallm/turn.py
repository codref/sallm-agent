"""One agent turn: control → retrieve → ReAct → extract/index (or enqueue)."""

from __future__ import annotations

import time

from sallm import metrics as metrics_mod
from sallm.control import ControlDecision
from sallm.llm import complete
from sallm.messages import assistant, user
from sallm.receipt import compile_prompt_messages
from sallm.tools import (
    format_observations,
    parse_run_blocks,
    run_many,
)


def apply_stack_decision(repo, session_id: str, decision: ControlDecision, registry):
    """Mutate persisted skill stack from a validated control decision."""
    known = set(registry.names())
    skill = decision.skill if decision.skill in known else repo.active_skill(session_id)
    action = decision.action
    if action == "push" and skill != repo.active_skill(session_id):
        repo.push_skill(session_id, skill)
    elif action == "replace":
        repo.replace_skill(session_id, skill)
    elif action == "pop":
        repo.pop_skill(session_id)
    # keep: no stack change
    if decision.goal:
        repo.set_goal(session_id, decision.goal)
    return repo.active_skill(session_id)


class TurnRunner:
    """Owns the ReAct loop for one ask(); Agent holds durable wiring."""

    def __init__(self, agent):
        self.agent = agent

    def _complete(self, messages, turn_metrics, *, max_tokens=None):
        kwargs = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        self.agent.last_prompt = list(messages)
        result = complete(
            model=self.agent.model,
            messages=messages,
            api_base=self.agent.api_base,
            **kwargs,
        )
        step_metrics = metrics_mod.from_llm_result(result)
        turn_metrics = metrics_mod.add_usage(turn_metrics, step_metrics)
        tr = self.agent.trace
        if tr is not None:
            tr.llm(
                model=self.agent.model,
                metrics=step_metrics,
                content=result.get("content") or "",
                reasoning=result.get("reasoning"),
                messages=messages,
            )
        return result, step_metrics, turn_metrics

    def _run_tools(self, commands, tools):
        tr = self.agent.trace
        if tr is None:
            return run_many(tools, commands)
        started = time.perf_counter()
        results = run_many(tools, commands)
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
        observation = format_observations(results)
        self.agent.messages.append(assistant(content))
        self.agent.messages.append(
            user(self.agent.prompt.RESULTS_PREFIX + observation)
        )
        # Persist to SQLite when a repository is present.
        repo = self.agent.repo
        if repo is not None:
            repo.append_message(
                self.agent.session_id, role="assistant", content=content, kind="tool"
            )
            repo.append_message(
                self.agent.session_id,
                role="user",
                content=self.agent.prompt.RESULTS_PREFIX + observation,
                kind="tool",
            )
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

    def run_react(self, prompt_messages, tools, turn_metrics):
        """Existing ReAct loop over an already-compiled prompt view."""
        steps = []
        acted = False
        awaiting_continue = False
        # Seed agent.messages system from prompt if empty aside from system.
        for _ in range(self.agent.max_steps):
            # Rebuild view each iteration from full transcript + last receipt memory.
            view, receipt = compile_prompt_messages(
                profile=self.agent.profile,
                prompt=self.agent.prompt,
                recent_messages=self.agent.messages,
                hits=self.agent._last_hits,
            )
            self.agent.last_receipt = receipt
            result, step_metrics, turn_metrics = self._complete(view, turn_metrics)
            content = result.get("content") or ""
            commands = parse_run_blocks(content) if tools else []

            if commands:
                results = self._run_tools(commands, tools)
                step = self._record_action(content, result, step_metrics, results)
                steps.append(step)
                acted = True
                if step["intermediate"]:
                    nudge = self.agent.prompt.CONTINUE_NUDGE
                    self.agent.messages.append(user(nudge))
                    steps.append({"kind": "nudge", "raw": nudge})
                    tr = self.agent.trace
                    if tr is not None:
                        tr.nudge(nudge)
                    awaiting_continue = True
                    continue
                awaiting_continue = False
                if self.agent.multi_step:
                    continue
                break

            if awaiting_continue:
                nudge = self.agent.prompt.EARLY_ANSWER_NUDGE
                self.agent.messages.append(user(nudge))
                steps.append(
                    {
                        "kind": "rejected",
                        "raw": content,
                        "nudge": nudge,
                        "metrics": step_metrics,
                    }
                )
                tr = self.agent.trace
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
            return content, steps, turn_metrics, None

        if acted:
            result, step_metrics, turn_metrics = self._complete(
                compile_prompt_messages(
                    profile=self.agent.profile,
                    prompt=self.agent.prompt,
                    recent_messages=self.agent.messages,
                    hits=self.agent._last_hits,
                )[0],
                turn_metrics,
            )
            answer = result.get("content") or ""
            if parse_run_blocks(answer) and tools:
                results = self._run_tools(parse_run_blocks(answer), tools)
                steps.append(
                    self._record_action(answer, result, step_metrics, results)
                )
                result, step_metrics, turn_metrics = self._complete(
                    compile_prompt_messages(
                        profile=self.agent.profile,
                        prompt=self.agent.prompt,
                        recent_messages=self.agent.messages,
                        hits=self.agent._last_hits,
                    )[0],
                    turn_metrics,
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
            return answer, steps, turn_metrics, "max_steps"

        fallback = "(no tool call produced)"
        steps.append({"kind": "final", "raw": fallback, "answer": fallback})
        return fallback, steps, turn_metrics, "max_steps"
