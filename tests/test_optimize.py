"""Offline optimizer — successive halving without DSPy."""

from __future__ import annotations

from sallm.optimization import (
    Candidate,
    Case,
    iterate_on_failures,
    load_artifact,
    merge_task_instruction,
    propose_instructions,
    save_artifact,
    score_case,
    successive_halving,
)


def test_propose_passes_think_false():
    from unittest.mock import patch

    fake = {
        "content": "Be shorter.",
        "reasoning": None,
        "usage": {"total_tokens": 8},
        "elapsed_ms": 1.0,
    }
    with patch("sallm.optimization.search.complete", return_value=fake) as mocked:
        texts = propose_instructions(
            baseline="Base instruction.",
            task="converse",
            model="ollama_chat/qwen3.5:0.8b",
            api_base="http://localhost",
            n=2,
            seed=0,
        )
    assert mocked.call_args.kwargs.get("think") is False
    assert texts[0] == "Base instruction."
    assert "Be shorter." in texts


def test_propose_with_fake_teacher():
    def teacher(prompt, i):
        return f"Instruction variant {i}: be clear."

    texts = propose_instructions(
        baseline="Base instruction.",
        task="controller",
        model="x",
        api_base="http://localhost",
        n=3,
        seed=1,
        teacher_fn=teacher,
    )
    assert len(texts) == 3
    assert texts[0] == "Base instruction."


def test_successive_halving_and_mandatory():
    cases = [
        Case(
            id="1",
            task="controller",
            input={"user": "hi"},
            expected={"action": "keep"},
            mandatory=True,
        ),
        Case(
            id="2",
            task="controller",
            input={"user": "bye"},
            expected={"action": "keep"},
        ),
    ]

    def predict(case, instruction, demos):
        # Candidate "good" always returns keep; "bad" fails mandatory.
        if "BAD" in instruction:
            return {"action": "push"}, {"total_tokens": 10, "elapsed_ms": 1}
        return {"action": "keep"}, {"total_tokens": 5, "elapsed_ms": 1}

    cands = [
        Candidate(name="good", instruction="GOOD keep skill"),
        Candidate(name="bad", instruction="BAD always push"),
        Candidate(name="good2", instruction="GOOD2 keep"),
        Candidate(name="noisy", instruction="GOOD noisy keep"),
    ]
    winner, report = successive_halving(
        cands, cases, predict_fn=predict, seed=0, min_keep=1
    )
    assert winner.name != "bad"
    assert "final" in report
    # bad should be heavily penalized on full eval if it survives
    assert report["winner"] == winner.name


def test_reject_avg_win_with_mandatory_fail():
    case = Case(
        id="m",
        task="t",
        input={},
        expected={"x": 1},
        mandatory=True,
    )

    def predict(case, instruction, demos):
        return {"x": 0}, {"total_tokens": 1, "elapsed_ms": 1}

    s = score_case(case, instruction="i", demos="", predict_fn=predict)
    assert s.mandatory_fail
    assert s.total < -1e8


def test_artifact_roundtrip(tmp_path):
    path = tmp_path / "p.json"
    save_artifact(
        path,
        target_model="ollama/gemma4:e4b-it-qat",
        instructions={"controller": "Do X"},
        demonstrations={"controller": ""},
        budgets={"prompt_budget": 4096},
        dataset_fingerprint="abc",
        metrics={"good": {"score": 1.0}},
        seed=7,
    )
    data = load_artifact(path)
    assert data["instructions"]["controller"] == "Do X"
    assert data["metadata"]["seed"] == 7
    assert data["metadata"]["content_digest"]


def test_converse_messages_use_react_system():
    from sallm.optimization.program import converse_messages
    from sallm.prompt import CompiledProfile

    compiled = CompiledProfile(
        target_model="x",
        instructions={"converse": "old"},
        demonstrations={},
        budgets={},
        metadata={},
    )
    case = Case(
        id="v",
        task="converse",
        input={"user": "What is 17 times 19? Use the calc tool."},
        expected={"tool": "calc"},
        mandatory=True,
    )
    messages = converse_messages(
        case,
        "WHEN CALC, FENCE_ONLY",
        compiled,
        tools_text="- calc: math",
    )
    assert messages[0]["role"] == "system"
    assert "WHEN CALC, FENCE_ONLY" in messages[0]["content"]
    assert "```run" in messages[0]["content"]
    assert "- calc: math" in messages[0]["content"]
    assert messages[0]["content"].index("Available tools:") < messages[0][
        "content"
    ].index("WHEN CALC, FENCE_ONLY")
    assert messages[1]["content"].startswith("What is 17")


def test_converse_messages_include_history():
    from sallm.optimization.program import converse_messages
    from sallm.prompt import CompiledProfile

    compiled = CompiledProfile(
        target_model="x",
        instructions={"converse": "ACK"},
        demonstrations={},
        budgets={},
        metadata={},
    )
    case = Case(
        id="v",
        task="converse",
        input={
            "history": [
                {"role": "assistant", "content": "The result is 323."},
            ],
            "user": "Please remember that my lab code is ORANGE-19.",
        },
        expected={"contains": ["ORANGE-19"]},
        mandatory=True,
    )
    messages = converse_messages(case, "ACK", compiled, tools_text="(none)")
    assert [m["role"] for m in messages] == ["system", "assistant", "user"]
    assert messages[1]["content"] == "The result is 323."
    assert "ORANGE-19" in messages[2]["content"]


def test_converse_messages_include_retrieved():
    from sallm.optimization.program import converse_messages
    from sallm.prompt import CompiledProfile

    compiled = CompiledProfile(
        target_model="x",
        instructions={"converse": "ACK"},
        demonstrations={},
        budgets={},
        metadata={},
    )
    case = Case(
        id="v",
        task="converse",
        input={
            "retrieved": "User lab code is ORANGE-19.",
            "user": "What lab code did I tell you earlier?",
        },
        expected={"contains": ["ORANGE-19"]},
        mandatory=True,
    )
    messages = converse_messages(case, "ACK", compiled, tools_text="(none)")
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].startswith("[Retrieved memory]")
    assert "ORANGE-19" in messages[1]["content"]
    assert messages[2]["content"].startswith("What lab code")


def test_converse_scores_raw_text_not_json():
    case = Case(
        id="v",
        task="converse",
        input={"user": "calc"},
        expected={"contains": ["```run", "calc"]},
        mandatory=True,
    )

    def predict(case, instruction, demos):
        return "```run\ncalc -e '2**8'\n``` {not: json}", {
            "total_tokens": 8,
            "elapsed_ms": 1,
        }

    s = score_case(case, instruction="i", demos="", predict_fn=predict)
    assert s.quality == 1.0
    assert not s.mandatory_fail


def test_converse_scores_parsed_tool_and_observation():
    from sallm.optimization.metrics import score_program_output

    got = {
        "content": "```run\ncalc -e '17*19'\n```",
        "commands": [["calc", "-e", "17*19"]],
        "observation": "$ calc -e '17*19'\n323\n",
    }
    q = score_program_output(
        got,
        {
            "tool": "calc",
            "argv_contains": ["17", "19"],
            "observation_contains": ["323"],
        },
    )
    assert q == 1.0
    q_bad = score_program_output(
        {
            "content": "323",
            "commands": [],
            "observation": "",
        },
        {"tool": "calc", "observation_contains": ["323"]},
    )
    assert q_bad == 0.0


def test_converse_scores_single_tool_rejects_fake_memory_tool():
    from sallm.optimization.metrics import score_program_output

    bad = {
        "content": "```run\ncalc -e '3*7'\nORANGE-19\n```",
        "commands": [["calc", "-e", "3*7"], ["ORANGE-19"]],
        "observation": "21\nunknown tool",
    }
    q = score_program_output(
        bad,
        {
            "tool": "calc",
            "single_tool": True,
            "argv_contains": ["3", "7"],
            "observation_contains": ["21"],
        },
    )
    assert q < 1.0
    good = {
        "content": "```run\ncalc -e '3*7'\n```",
        "commands": [["calc", "-e", "3*7"]],
        "observation": "$ calc -e '3*7'\n21\n",
    }
    assert (
        score_program_output(
            good,
            {
                "tool": "calc",
                "single_tool": True,
                "argv_contains": ["3", "7"],
                "observation_contains": ["21"],
            },
        )
        == 1.0
    )


def test_converse_scores_absent_and_no_tool():
    from sallm.optimization.metrics import score_program_output

    ok = score_program_output(
        "Recalled your lab code as ORANGE-19.",
        {
            "contains": ["ORANGE-19"],
            "absent": ["The result is", "Tongyi"],
            "no_tool": True,
        },
    )
    assert ok == 1.0
    leak = score_program_output(
        "The result is ORANGE-19.",
        {
            "contains": ["ORANGE-19"],
            "absent": ["The result is"],
            "no_tool": True,
        },
    )
    assert leak < 1.0
    toolish = score_program_output(
        "```run\necho --text ORANGE-19\n```",
        {"contains": ["ORANGE-19"], "no_tool": True},
    )
    assert toolish < 1.0


def test_json_scores_retrieval_query_and_facts_contains():
    from sallm.optimization.metrics import score_json_output

    keep = score_json_output(
        {"action": "keep", "skill": "converse", "retrieval_query": ""},
        {"action": "keep", "skill": "converse", "retrieval_query": ""},
    )
    assert keep == 1.0
    recall = score_json_output(
        {"action": "keep", "skill": "converse", "retrieval_query": "lab code ORANGE"},
        {
            "action": "keep",
            "skill": "converse",
            "retrieval_query_contains": ["lab"],
        },
    )
    assert recall == 1.0
    empty_q = score_json_output(
        {"action": "keep", "skill": "converse", "retrieval_query": ""},
        {"action": "keep", "retrieval_query_nonempty": True},
    )
    assert empty_q < 1.0
    facts = score_json_output(
        {"facts": [{"text": "User lab code is ORANGE-19.", "source_message_ids": [10]}]},
        {"contains": ["ORANGE-19"], "absent": ["Tongyi"]},
    )
    assert facts == 1.0


def test_materialize_got_runs_calc():
    from sallm.optimization.program import materialize_got
    from sallm.tools import builtin_tools

    got = materialize_got(
        "```run\ncalc -e '17*19'\n```",
        builtin_tools("calc"),
        {"tool": "calc", "observation_contains": ["323"]},
    )
    assert got["commands"] == [["calc", "-e", "17*19"]]
    assert "323" in got["observation"]


def test_converse_teacher_prompt_mentions_run_contract():
    from sallm.optimization import propose_from_failures

    prompt = propose_from_failures(
        instruction="Be brief.",
        task="converse",
        failures=[
            {
                "id": "v-calc",
                "expected": {"tool": "calc"},
                "got": "256 | commands: (none)",
            }
        ],
        model="x",
        api_base="http://localhost",
        round_index=1,
        teacher_fn=lambda p, i: p,
    )
    assert "```run" in prompt
    assert "ORANGE-19" in prompt
    assert "The result is" in prompt
    assert "commands: (none)" in prompt
    assert "Be brief." in prompt


def test_iterate_on_failures_rewrites_until_pass():
    case = Case(
        id="v",
        task="converse",
        input={"user": "hi"},
        expected={"contains": ["ok"]},
        mandatory=True,
    )

    def predict(case, instruction, demos):
        text = "ok" if "FIXED" in instruction else "nope"
        return text, {"total_tokens": 2, "elapsed_ms": 1}

    def teacher(prompt, i):
        assert "nope" in prompt
        return "FIXED be brief"

    winner, report = iterate_on_failures(
        baseline="bad",
        task="converse",
        cases=[case],
        predict_fn=predict,
        model="x",
        api_base="http://localhost",
        rounds=3,
        teacher_fn=teacher,
    )
    assert winner == "FIXED be brief"
    assert report["passed"] is True


def test_iterate_on_failures_emits_progress_events():
    case = Case(
        id="v",
        task="converse",
        input={"user": "hi"},
        expected={"contains": ["ok"]},
        mandatory=True,
    )

    def predict(case, instruction, demos):
        text = "ok" if "FIXED" in instruction else "nope"
        return text, {"total_tokens": 2, "elapsed_ms": 1}

    events = []

    def on_event(kind, **payload):
        events.append(kind)

    winner, report = iterate_on_failures(
        baseline="bad",
        task="converse",
        cases=[case],
        predict_fn=predict,
        model="x",
        api_base="http://localhost",
        rounds=2,
        teacher_fn=lambda p, i: "FIXED be brief",
        on_event=on_event,
    )
    assert winner == "FIXED be brief"
    assert "eval_begin" in events
    assert "case_begin" in events
    assert "case" in events
    assert "eval_end" in events
    assert "teacher_begin" in events
    assert "teacher_end" in events
    assert report["passed"] is True


def test_merge_task_keeps_sibling_keys_and_budgets():
    data = {
        "schema_version": 1,
        "target_model": "ollama_chat/qwen",
        "instructions": {"converse": "old", "controller": "keep-me"},
        "demonstrations": {"controller": "demo"},
        "budgets": {"think": False, "temperature": 0.2},
        "metadata": {"note": "x"},
    }
    merged = merge_task_instruction(
        data,
        task="converse",
        instruction="new converse",
        dataset_fingerprint="abc",
        metrics={"best": {"quality": 1.0}},
        seed=2,
    )
    assert merged["instructions"]["converse"] == "new converse"
    assert merged["instructions"]["controller"] == "keep-me"
    assert merged["demonstrations"]["controller"] == "demo"
    assert merged["budgets"]["think"] is False
    assert merged["metadata"]["note"] == "x"
    assert merged["metadata"]["seed"] == 2
    assert merged["metadata"]["dataset_fingerprint"] == "abc"


def test_optimize_evaluate_only_uses_profile_instruction(tmp_path):
    import json
    from unittest.mock import patch

    from typer.testing import CliRunner

    from sallm.cli.chat import app

    profile = {
        "schema_version": 1,
        "target_model": "ollama_chat/qwen3.5:0.8b",
        "instructions": {"converse": "Say qwen briefly."},
        "demonstrations": {},
        "budgets": {"think": False, "max_output_tokens": 64},
        "metadata": {},
    }
    ppath = tmp_path / "p.json"
    ppath.write_text(json.dumps(profile), encoding="utf-8")
    dpath = tmp_path / "c.jsonl"
    dpath.write_text(
        json.dumps(
            {
                "id": "v-greet",
                "task": "converse",
                "input": {"user": "hi"},
                "expected": {"contains": ["qwen"]},
                "mandatory": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fake = {
        "content": "I am Qwen.",
        "usage": {"total_tokens": 4},
        "elapsed_ms": 1,
        "reasoning": None,
    }
    with patch("sallm.cli.optimize_cmd.complete", return_value=fake) as mocked:
        result = CliRunner().invoke(
            app,
            [
                "optimize",
                "--dataset",
                str(dpath),
                "--profile",
                str(ppath),
                "--task",
                "converse",
                "--evaluate-only",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "quality=1.000" in result.output
    messages = mocked.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    system = messages[0]["content"]
    assert "Say qwen briefly." in system
    assert "```run" in system
    assert system.index("Available tools:") < system.index("Say qwen briefly.")
    assert messages[1] == {"role": "user", "content": "hi"}
    assert mocked.call_args.kwargs.get("think") is False
    assert mocked.call_args.kwargs.get("max_tokens") == 64
    assert json.loads(ppath.read_text())["instructions"]["converse"] == "Say qwen briefly."


def _opt_paths(tmp_path):
    import json

    profile = {
        "schema_version": 1,
        "target_model": "ollama_chat/qwen3.5:0.8b",
        "instructions": {"converse": "Say qwen briefly."},
        "demonstrations": {},
        "budgets": {"think": False, "max_output_tokens": 64},
        "metadata": {"note": "seed"},
    }
    ppath = tmp_path / "p.json"
    ppath.write_text(json.dumps(profile), encoding="utf-8")
    dpath = tmp_path / "c.jsonl"
    dpath.write_text(
        json.dumps(
            {
                "id": "v-greet",
                "task": "converse",
                "input": {"user": "hi"},
                "expected": {"contains": ["qwen"]},
                "mandatory": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return ppath, dpath


def test_optimize_search_does_not_write_without_out(tmp_path):
    from unittest.mock import patch

    from typer.testing import CliRunner

    from sallm.cli.chat import app

    ppath, dpath = _opt_paths(tmp_path)
    original = ppath.read_text(encoding="utf-8")
    fake = {
        "content": "I am Qwen.",
        "usage": {"total_tokens": 4},
        "elapsed_ms": 1,
        "reasoning": None,
    }
    with patch("sallm.cli.optimize_cmd.complete", return_value=fake):
        result = CliRunner().invoke(
            app,
            [
                "optimize",
                "--dataset",
                str(dpath),
                "--profile",
                str(ppath),
                "--task",
                "converse",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "not written" in result.output
    assert ppath.read_text(encoding="utf-8") == original


def test_optimize_search_writes_only_to_out(tmp_path):
    import json
    from unittest.mock import patch

    from typer.testing import CliRunner

    from sallm.cli.chat import app

    ppath, dpath = _opt_paths(tmp_path)
    original = ppath.read_text(encoding="utf-8")
    dest = tmp_path / "opt.json"
    fake = {
        "content": "I am Qwen.",
        "usage": {"total_tokens": 4},
        "elapsed_ms": 1,
        "reasoning": None,
    }
    with patch("sallm.cli.optimize_cmd.complete", return_value=fake):
        result = CliRunner().invoke(
            app,
            [
                "optimize",
                "--dataset",
                str(dpath),
                "--profile",
                str(ppath),
                "--task",
                "converse",
                "--out",
                str(dest),
            ],
        )
    assert result.exit_code == 0, result.output
    assert dest.exists()
    assert json.loads(dest.read_text())["instructions"]["converse"] == "Say qwen briefly."
    assert ppath.read_text(encoding="utf-8") == original


def test_optimize_in_place_overwrites_profile(tmp_path):
    import json
    from unittest.mock import patch

    from typer.testing import CliRunner

    from sallm.cli.chat import app

    ppath, dpath = _opt_paths(tmp_path)
    fake = {
        "content": "I am Qwen.",
        "usage": {"total_tokens": 4},
        "elapsed_ms": 1,
        "reasoning": None,
    }
    with patch("sallm.cli.optimize_cmd.complete", return_value=fake):
        result = CliRunner().invoke(
            app,
            [
                "optimize",
                "--dataset",
                str(dpath),
                "--profile",
                str(ppath),
                "--task",
                "converse",
                "--in-place",
            ],
        )
    assert result.exit_code == 0, result.output
    data = json.loads(ppath.read_text(encoding="utf-8"))
    assert data["instructions"]["converse"] == "Say qwen briefly."
    assert data["metadata"]["note"] == "seed"
    assert data["metadata"]["optimize_task"] == "converse"


def test_expand_budget_candidates_ota():
    from sallm.optimization.budgets import expand_budget_candidates

    base = {
        "temperature": 0.2,
        "max_output_tokens": 512,
        "think": False,
        "prompt_budget": 2048,
    }
    cands = expand_budget_candidates(base, task="converse", seed=0)
    assert cands[0] == base
    assert len(cands) > 1
    # One-at-a-time: each non-baseline differs in exactly one searched key.
    searched = {"temperature", "max_output_tokens", "think"}
    for cand in cands[1:]:
        diffs = [k for k in searched if cand.get(k) != base.get(k)]
        assert len(diffs) == 1
        assert cand.get("prompt_budget") == 2048


def test_search_budgets_keeps_better_temperature():
    from sallm.optimization import Case, search_budgets

    case = Case(
        id="v",
        task="converse",
        input={"user": "hi"},
        expected={"contains": ["ok"]},
        mandatory=True,
    )

    def factory(budgets):
        def predict(case, instruction, demos):
            # temperature 0.0 is the only setting that passes.
            text = "ok" if budgets.get("temperature") == 0.0 else "nope"
            return text, {"total_tokens": 2, "elapsed_ms": 1}

        return predict

    winner, report = search_budgets(
        baseline={"temperature": 0.2, "max_output_tokens": 256, "think": False},
        task="converse",
        cases=[case],
        instruction="be brief",
        predict_factory=factory,
        seed=0,
    )
    assert winner["temperature"] == 0.0
    assert report["passed"] is True
    assert report["final"]["best"]["mandatory_misses"] == 0


def test_merge_budgets_overlays_keys():
    from sallm.optimization.artifacts import merge_budgets

    data = {
        "schema_version": 1,
        "target_model": "x",
        "instructions": {"converse": "keep"},
        "demonstrations": {},
        "budgets": {"think": False, "temperature": 0.2, "max_output_tokens": 512},
        "metadata": {"note": "x"},
    }
    merged = merge_budgets(
        data,
        budgets={"temperature": 0.0, "max_output_tokens": 128},
        dataset_fingerprint="abc",
        metrics={"best": {"quality": 1.0}},
        seed=1,
    )
    assert merged["instructions"]["converse"] == "keep"
    assert merged["budgets"]["think"] is False
    assert merged["budgets"]["temperature"] == 0.0
    assert merged["budgets"]["max_output_tokens"] == 128
    assert merged["metadata"]["optimize_task"] == "budgets"
    assert merged["metadata"]["note"] == "x"


def test_optimize_budgets_only_writes_budgets(tmp_path):
    import json
    from unittest.mock import patch

    from typer.testing import CliRunner

    from sallm.cli.chat import app

    profile = {
        "schema_version": 1,
        "target_model": "ollama_chat/qwen3.5:0.8b",
        "instructions": {"converse": "Say qwen briefly."},
        "demonstrations": {},
        "budgets": {"think": False, "temperature": 0.2, "max_output_tokens": 64},
        "metadata": {},
    }
    ppath = tmp_path / "p.json"
    ppath.write_text(json.dumps(profile), encoding="utf-8")
    dpath = tmp_path / "c.jsonl"
    dpath.write_text(
        json.dumps(
            {
                "id": "v-greet",
                "task": "converse",
                "input": {"user": "hi"},
                "expected": {"contains": ["qwen"]},
                "mandatory": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.json"
    fake = {
        "content": "I am Qwen.",
        "usage": {"total_tokens": 4},
        "elapsed_ms": 1,
        "reasoning": None,
    }
    with patch("sallm.cli.optimize_cmd.complete", return_value=fake):
        result = CliRunner().invoke(
            app,
            [
                "optimize",
                "--dataset",
                str(dpath),
                "--profile",
                str(ppath),
                "--task",
                "converse",
                "--budgets-only",
                "--out",
                str(dest),
            ],
        )
    assert result.exit_code == 0, result.output
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["instructions"]["converse"] == "Say qwen briefly."
    assert "budgets" in data
    assert data["metadata"]["optimize_task"].startswith("budgets")


def test_optimize_defaults_to_packaged_profile(tmp_path):
    import json
    from unittest.mock import patch

    from typer.testing import CliRunner

    from sallm.cli.chat import app
    from sallm.prompt import DEFAULT_PROFILE_PATH

    assert DEFAULT_PROFILE_PATH.is_file()
    dpath = tmp_path / "c.jsonl"
    dpath.write_text(
        json.dumps(
            {
                "id": "c1",
                "task": "controller",
                "input": {"user": "hi"},
                "expected": {"action": "keep", "skill": "converse"},
                "mandatory": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fake = {
        "content": '{"goal":"g","action":"keep","skill":"converse","retrieval_query":""}',
        "usage": {"total_tokens": 4},
        "elapsed_ms": 1,
        "reasoning": None,
    }
    with patch("sallm.cli.optimize_cmd.complete", return_value=fake):
        result = CliRunner().invoke(
            app,
            [
                "optimize",
                "--dataset",
                str(dpath),
                "--task",
                "controller",
                "--evaluate-only",
            ],
        )
    assert result.exit_code == 0, result.output
    # Default packaged profile target_model (Rich may wrap long paths).
    assert "gemma4:e4b-it-qat" in result.output
    assert "passed" in result.output.lower() or "quality=" in result.output


def test_optimize_out_preserves_sibling_tasks(tmp_path):
    """Writing controller into an existing --out must keep converse."""
    import json
    from unittest.mock import patch

    from typer.testing import CliRunner

    from sallm.cli.chat import app

    seed = {
        "schema_version": 1,
        "target_model": "ollama/gemma4:e4b-it-qat",
        "instructions": {
            "converse": "KEEP_CONVERSE",
            "controller": "old controller",
        },
        "demonstrations": {},
        "budgets": {},
        "metadata": {},
    }
    # Input profile has empty converse — bug was merging only from --profile.
    pin = tmp_path / "in.json"
    pin.write_text(
        json.dumps(
            {
                **seed,
                "instructions": {"controller": "seed controller", "converse": ""},
            }
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "out.json"
    dest.write_text(json.dumps(seed), encoding="utf-8")
    dpath = tmp_path / "c.jsonl"
    dpath.write_text(
        json.dumps(
            {
                "id": "c1",
                "task": "controller",
                "input": {"user": "hi"},
                "expected": {"action": "keep", "skill": "converse"},
                "mandatory": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fake = {
        "content": '{"goal":"g","action":"keep","skill":"converse","retrieval_query":""}',
        "usage": {"total_tokens": 4},
        "elapsed_ms": 1,
        "reasoning": None,
    }
    with patch("sallm.cli.optimize_cmd.complete", return_value=fake):
        result = CliRunner().invoke(
            app,
            [
                "optimize",
                "--dataset",
                str(dpath),
                "--profile",
                str(pin),
                "--task",
                "controller",
                "--rounds",
                "1",
                "--out",
                str(dest),
            ],
        )
    assert result.exit_code == 0, result.output
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert data["instructions"]["converse"] == "KEEP_CONVERSE"
    assert data["instructions"]["controller"]


def test_optimize_out_without_profile_continues_artifact(tmp_path):
    """Omit --profile when --out exists: load that artifact as input."""
    import json
    from unittest.mock import patch

    from typer.testing import CliRunner

    from sallm.cli.chat import app

    dest = tmp_path / "artifact.json"
    dest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target_model": "ollama/gemma4:e4b-it-qat",
                "instructions": {
                    "converse": "KEEP_CONVERSE",
                    "controller": "Reply JSON keep/converse only.",
                },
                "demonstrations": {},
                "budgets": {},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    dpath = tmp_path / "c.jsonl"
    dpath.write_text(
        json.dumps(
            {
                "id": "c1",
                "task": "controller",
                "input": {"user": "hi"},
                "expected": {"action": "keep", "skill": "converse"},
                "mandatory": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fake = {
        "content": '{"goal":"g","action":"keep","skill":"converse","retrieval_query":""}',
        "usage": {"total_tokens": 4},
        "elapsed_ms": 1,
        "reasoning": None,
    }
    with patch("sallm.cli.optimize_cmd.complete", return_value=fake):
        result = CliRunner().invoke(
            app,
            [
                "optimize",
                "--dataset",
                str(dpath),
                "--task",
                "controller",
                "--rounds",
                "1",
                "--out",
                str(dest),
            ],
        )
    assert result.exit_code == 0, result.output
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert data["instructions"]["converse"] == "KEEP_CONVERSE"
    assert data["instructions"]["controller"]
    # Continued from --out, not the empty packaged default.
    assert data["target_model"] == "ollama/gemma4:e4b-it-qat"


def test_small_chat_opt_cases_cover_tasks():
    from pathlib import Path

    from sallm.optimization import load_jsonl

    path = Path(__file__).resolve().parents[1] / "examples/small_chat/opt_cases.jsonl"
    cases = load_jsonl(path)
    by_task = {}
    for c in cases:
        by_task.setdefault(c.task, []).append(c.id)
    assert len(by_task["converse"]) >= 16
    assert len(by_task["controller"]) >= 12
    assert len(by_task["extractor"]) >= 6
    assert any(c.input.get("retrieved") for c in cases if c.task == "converse")
    assert any(
        "retrieval_query_contains" in c.expected
        for c in cases
        if c.task == "controller"
    )
    ids = {c.id for c in cases}
    assert "v-calc-and-recall-after-tool" in ids
    assert "v-echo-and-recall-after-tool" in ids
    assert "c-echo-and-recall" in ids
    after = next(c for c in cases if c.id == "v-calc-and-recall-after-tool")
    assert after.input.get("user", "").startswith("Tool results:")
    assert "ORANGE-19" in after.expected.get("contains", [])
    assert after.expected.get("no_tool") is True
    first = next(c for c in cases if c.id == "v-calc-and-recall-first")
    assert first.expected.get("single_tool") is True
