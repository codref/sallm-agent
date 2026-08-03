"""Legacy in-memory ReAct ask() — kept for tests without state_path."""

from __future__ import annotations

import time

from sallm import metrics as metrics_mod
from sallm.llm import complete
from sallm.messages import assistant, user
from sallm.tools import format_observations, parse_run_blocks, run_many


def ask_legacy(agent, user_text: str):
    agent.messages.append(user(user_text))
    steps = []
    turn_metrics = metrics_mod.empty_usage()
    acted = False
    awaiting_continue = False
    tr = agent.trace
    if tr is not None:
        tr.turn_start(user_text, agent.messages, model=agent.model)

    def _finish(answer, stopped=None):
        agent.messages.append(assistant(answer))
        # Metrics compare full transcript vs the optimizer view of that transcript.
        prompt_view = agent._prompt_messages(agent.messages)
        summary = metrics_mod.summarize(
            turn_metrics,
            context_messages=len(agent.messages),
            prompt_messages=len(prompt_view),
        )
        if tr is not None:
            tr.turn_end(
                answer=answer,
                metrics=summary,
                messages=agent.messages,
                stopped=stopped,
            )
        out = {"answer": answer, "steps": steps, "metrics": summary}
        if stopped:
            out["stopped"] = stopped
        return out

    for _ in range(agent.max_steps):
        prompt = agent._prompt_messages(agent.messages)
        agent.last_prompt = prompt
        result = complete(
            model=agent.model, messages=prompt, api_base=agent.api_base
        )
        step_metrics = metrics_mod.from_llm_result(result)
        turn_metrics = metrics_mod.add_usage(turn_metrics, step_metrics)
        if tr is not None:
            tr.llm(
                model=agent.model,
                metrics=step_metrics,
                content=result.get("content") or "",
                reasoning=result.get("reasoning"),
                messages=prompt,
            )
        content = result.get("content") or ""
        commands = parse_run_blocks(content) if agent.tools else []
        if commands:
            started = time.perf_counter()
            results = run_many(agent.tools, commands)
            if tr is not None:
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
            observation = format_observations(results)
            agent.messages.append(assistant(content))
            agent.messages.append(user(agent.prompt.RESULTS_PREFIX + observation))
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
                nudge = agent.prompt.CONTINUE_NUDGE
                agent.messages.append(user(nudge))
                steps.append({"kind": "nudge", "raw": nudge})
                if tr is not None:
                    tr.nudge(nudge)
                awaiting_continue = True
                continue
            awaiting_continue = False
            if agent.multi_step:
                continue
            break
        if awaiting_continue:
            nudge = agent.prompt.EARLY_ANSWER_NUDGE
            agent.messages.append(user(nudge))
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
        return _finish(content)

    if acted:
        prompt = agent._prompt_messages(agent.messages)
        agent.last_prompt = prompt
        result = complete(
            model=agent.model, messages=prompt, api_base=agent.api_base
        )
        step_metrics = metrics_mod.from_llm_result(result)
        turn_metrics = metrics_mod.add_usage(turn_metrics, step_metrics)
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
        return _finish(answer, stopped="max_steps")

    fallback = "(no tool call produced)"
    steps.append({"kind": "final", "raw": fallback, "answer": fallback})
    return _finish(fallback, stopped="max_steps")
