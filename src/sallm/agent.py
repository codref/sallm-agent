import json
import re

from . import metrics as metrics_mod
from .llm import complete
from .messages import DEFAULT_API_BASE, DEFAULT_MODEL, assistant, system, tool, user
from .tools import is_intermediate, run_tool, tool_descriptions, tool_schemas

MULTI_STEP_ON = """Multi-step mode is ON.
If the user asked for several sequential operations, call tools one step at a time.
After each tool result, either call the next tool or answer in plain text when finished.
If a tool result starts with [intermediate], the work is not done — call the tool again
(or the next required tool). Do not give a final answer yet.
Never invent tools. Never wrap answers as JSON tool payloads."""

MULTI_STEP_OFF = """Multi-step mode is OFF.
Prefer a single tool call when possible (combine work into one call if you can).
Exception: if a tool result starts with [intermediate], you must call that tool again
until you get a final (non-intermediate) result — then answer in plain text.
Never invent tools. Never wrap answers as JSON tool payloads."""

CONTINUE_NUDGE = (
    "The previous tool result was intermediate (not finished). "
    "Call the same or next required tool again. "
    "Do not give a final answer yet. Do not invent JSON tool formats."
)

EARLY_ANSWER_NUDGE = (
    "You replied with text before the tool work finished. "
    "An intermediate tool result is still pending. "
    "Call the required tool again now. Do not answer the user yet. "
    "Do not invent JSON."
)

SYSTEM = """You are a helpful assistant.

Default behavior: answer the user directly in plain text.
Do not call tools for greetings, identity questions, opinions, explanations,
or anything you can answer from your own knowledge.

When tools are offered by the API, call them only if they clearly match the
user's request. Follow each tool's own description.
Never wrap a normal answer as a tool argument.
After a finished tool result, reply to the user in short plain text (not JSON).

{multi_step_policy}

Available tools:
{tools}
"""

DECIDE_PROMPT = """Decide whether you need a tool for the user's latest message.

Reply with ONLY a JSON object, no other text:
{{"use_tools": false}}
{{"use_tools": true}}
{{"use_tools": true, "tools": ["tool_name"]}}

Rules:
- use_tools=true only if an available tool is required to answer correctly.
- use_tools=false for greetings, identity, opinions, explanations, and general chat.
- If use_tools=true, optional "tools" is a list of tool names to enable (subset of available tools).
- Do not answer the user in this step. JSON only.

Available tools:
{tools}
"""

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_decision(text):
    """Parse decide-step JSON. Defaults to use_tools=False on failure."""
    text = (text or "").strip()
    candidates = []
    try:
        candidates.append(json.loads(text))
    except json.JSONDecodeError:
        pass
    for match in _JSON_RE.finditer(text):
        try:
            candidates.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    for data in candidates:
        if not isinstance(data, dict) or "use_tools" not in data:
            continue
        use = data.get("use_tools")
        if isinstance(use, str):
            use = use.strip().lower() in ("1", "true", "yes")
        else:
            use = bool(use)
        names = data.get("tools")
        if names is not None and not isinstance(names, list):
            names = None
        if names is not None:
            names = [str(n) for n in names]
        return {"use_tools": use, "tools": names, "raw": text}
    return {"use_tools": False, "tools": None, "raw": text}


class Agent:
    def __init__(
        self,
        model=None,
        api_base=None,
        tools=None,
        system=None,
        max_steps=5,
        multi_step=True,
    ):
        self.model = model or DEFAULT_MODEL
        self.api_base = api_base or DEFAULT_API_BASE
        self.tools = dict(tools or {})
        self.tool_defs = tool_schemas(self.tools)
        self.max_steps = max_steps
        self.multi_step = multi_step
        self.system = system
        self.messages = []
        self._ensure_system()

    def _multi_step_policy(self):
        return MULTI_STEP_ON if self.multi_step else MULTI_STEP_OFF

    def _ensure_system(self):
        base = SYSTEM.format(
            tools=tool_descriptions(self.tools),
            multi_step_policy=self._multi_step_policy(),
        )
        if self.system:
            content = self.system.rstrip() + "\n\n" + base
        else:
            content = base
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = system(content)
        else:
            self.messages.insert(0, system(content))

    def clear(self):
        self.messages = []
        self._ensure_system()

    def _select_tool_defs(self, names):
        if not names:
            return self.tool_defs or None
        allowed = set(names) & set(self.tools)
        if not allowed:
            return self.tool_defs or None
        subset = {k: v for k, v in self.tools.items() if k in allowed}
        return tool_schemas(subset) or None

    def _complete(self, messages, turn_metrics, tools=None):
        result = complete(
            model=self.model,
            messages=messages,
            api_base=self.api_base,
            tools=tools,
        )
        step_metrics = metrics_mod.from_llm_result(result)
        turn_metrics = metrics_mod.add_usage(turn_metrics, step_metrics)
        return result, step_metrics, turn_metrics

    def _finish(self, answer, steps, turn_metrics):
        self.messages.append(assistant(answer))
        return {
            "answer": answer,
            "steps": steps,
            "metrics": metrics_mod.summarize(
                turn_metrics,
                context_messages=len(self.messages),
            ),
        }

    def _run_tool_calls(self, tool_calls, content=None):
        observations = []
        self.messages.append(assistant(content, tool_calls=tool_calls))
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments") or "{}"
            observation = run_tool(self.tools, name, args)
            observations.append(
                {
                    "id": tc.get("id"),
                    "action": name,
                    "action_input": args,
                    "observation": observation,
                    "intermediate": is_intermediate(observation),
                }
            )
            self.messages.append(tool(observation, tool_call_id=tc.get("id")))
        return observations

    def _inject_continue_nudge(self):
        self.messages.append(user(CONTINUE_NUDGE))

    def ask(self, user_text):
        self.messages.append(user(user_text))
        steps = []
        turn_metrics = metrics_mod.empty_usage()

        # --- 1) DECIDE (never pass tools=) ---------------------------------
        decide_messages = list(self.messages) + [
            user(DECIDE_PROMPT.format(tools=tool_descriptions(self.tools)))
        ]
        decide_result, decide_metrics, turn_metrics = self._complete(
            decide_messages, turn_metrics, tools=None
        )
        decision = parse_decision(decide_result.get("content"))
        steps.append(
            {
                "kind": "decide",
                "raw": decide_result.get("content"),
                "reasoning": decide_result.get("reasoning"),
                "use_tools": decision["use_tools"],
                "tools": decision.get("tools"),
                "metrics": decide_metrics,
            }
        )

        # --- 2a) No tools → answer in plain chat --------------------------
        if not decision["use_tools"] or not self.tool_defs:
            result, step_metrics, turn_metrics = self._complete(
                self.messages, turn_metrics, tools=None
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
            return self._finish(answer, steps, turn_metrics)

        # --- 2b) Yes → native tool loop -----------------------------------
        tool_defs = self._select_tool_defs(decision.get("tools"))
        acted = False
        awaiting_continue = False

        for _ in range(self.max_steps):
            result, step_metrics, turn_metrics = self._complete(
                self.messages, turn_metrics, tools=tool_defs
            )
            content = result.get("content") or ""
            tool_calls = result.get("tool_calls") or []

            if tool_calls:
                observations = self._run_tool_calls(tool_calls, content=content or None)
                pending = any(o.get("intermediate") for o in observations)
                step = {
                    "kind": "action",
                    "raw": content,
                    "reasoning": result.get("reasoning"),
                    "tool_calls": observations,
                    "intermediate": pending,
                    "metrics": step_metrics,
                }
                if len(observations) == 1:
                    step["action"] = observations[0]["action"]
                    step["action_input"] = observations[0]["action_input"]
                    step["observation"] = observations[0]["observation"]
                steps.append(step)
                acted = True

                # Intermediate tool results always require another round.
                if pending:
                    self._inject_continue_nudge()
                    steps.append({"kind": "nudge", "raw": CONTINUE_NUDGE})
                    awaiting_continue = True
                    continue

                awaiting_continue = False
                if self.multi_step:
                    continue
                break

            # Model returned text instead of a tool call.
            if awaiting_continue:
                # Reject early "answers" (common with small models: JSON chatter).
                self.messages.append(user(EARLY_ANSWER_NUDGE))
                steps.append(
                    {
                        "kind": "rejected",
                        "raw": content,
                        "nudge": EARLY_ANSWER_NUDGE,
                        "metrics": step_metrics,
                    }
                )
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

        # --- 3) Hit max_steps still in tool mode → force a text answer ----
        if acted:
            result, step_metrics, turn_metrics = self._complete(
                self.messages, turn_metrics, tools=None
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
            out = self._finish(answer, steps, turn_metrics)
            out["stopped"] = "max_steps"
            return out

        fallback = "(no tool call produced)"
        steps.append({"kind": "final", "raw": fallback, "answer": fallback})
        return {
            **self._finish(fallback, steps, turn_metrics),
            "stopped": "max_steps",
        }
