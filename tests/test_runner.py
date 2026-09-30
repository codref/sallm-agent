"""Runner tests — no LLM required."""

from __future__ import annotations

import sys

from sallm.tools import (
    builtin_tools,
    format_observations,
    help_text,
    parse_file_blocks,
    parse_run_blocks,
    prepare_commands,
    reset_dig_state,
    run_many,
    run_tool,
)

TOOLS = builtin_tools(("echo", "calc", "dig"))


def test_parse_run_blocks_multi():
    text = """
I'll run both tools.

```run
calc --expression "2+2"
echo --text hi
```
"""
    cmds = parse_run_blocks(text)
    assert cmds == [
        ["calc", "--expression", "2+2"],
        ["echo", "--text", "hi"],
    ]


def test_parse_run_blocks_empty():
    assert parse_run_blocks("just text") == []
    assert parse_run_blocks("```run\n\n```") == []


def test_parse_run_blocks_same_line():
    cmds = parse_run_blocks("```run calc -e '2**8' ```")
    assert cmds == [["calc", "-e", "2**8"]]


def test_parse_run_line_apostrophe_does_not_collapse_command():
    """Tom's breaks shlex; must not become unknown tool '<whole line>'."""
    text = """```run
echo --text Tom's code diagram missing component
```"""
    cmds = parse_run_blocks(text)
    assert len(cmds) == 1
    assert cmds[0][0] == "echo"
    assert "--text" in cmds[0]
    assert "Tom's" in cmds[0]


def test_parse_file_blocks_multiline_and_apostrophe():
    text = """```file note
# Tom's notes

Line two with "quotes"
```"""
    files = parse_file_blocks(text)
    assert "note" in files
    assert "Tom's notes" in files["note"]
    assert 'Line two with "quotes"' in files["note"]


def test_prepare_commands_substitutes_at_name_and_cleans_up():
    text = """```run
echo --text @note
```
```file note
hello
world
```"""
    prepared = prepare_commands(text)
    try:
        assert prepared.error is None
        assert len(prepared.commands) == 1
        path = prepared.commands[0][-1]
        assert path.endswith(".md")
        from pathlib import Path

        assert Path(path).read_text(encoding="utf-8") == "hello\nworld\n"
        temps = list(prepared._temps)
    finally:
        prepared.cleanup()
    from pathlib import Path

    for path in temps:
        assert not Path(path).exists()


def test_prepare_commands_unknown_ref_errors_without_temps():
    text = """```run
echo --text @missing
```"""
    prepared = prepare_commands(text)
    try:
        assert prepared.error is not None
        assert "@missing" in prepared.error
        assert prepared._temps == []
    finally:
        prepared.cleanup()


def test_prepare_commands_runs_tool_with_file_body():
    text = """```run
echo --text @note
```
```file note
payload
```"""
    prepared = prepare_commands(text)
    try:
        path = prepared.commands[0][-1]
        results = run_many(TOOLS, [["echo", "--text", path]])
        assert results[0].returncode == 0
        assert path in results[0].observation
    finally:
        prepared.cleanup()


def test_calc_subprocess():
    result = run_tool(TOOLS["calc"], ["--expression", "2+2"])
    assert result.returncode == 0
    assert result.observation == "4"


def test_calc_power():
    result = run_tool(TOOLS["calc"], ["--expression", "2**10"])
    assert result.observation == "1024"


def test_echo_subprocess():
    result = run_tool(TOOLS["echo"], ["--text", "hello"])
    assert result.returncode == 0
    assert result.observation == "hello"


def test_unknown_tool():
    results = run_many(TOOLS, [["nope", "--x", "1"]])
    assert len(results) == 1
    assert "unknown tool" in results[0].observation


def test_help_text():
    text = help_text(TOOLS["calc"])
    assert "--expression" in text or "-e" in text


def test_run_many_concurrent():
    results = run_many(
        TOOLS,
        [
            ["calc", "--expression", "1+1"],
            ["echo", "--text", "z"],
        ],
    )
    assert len(results) == 2
    by_name = {r.name: r.observation for r in results}
    assert by_name["calc"] == "2"
    assert by_name["echo"] == "z"
    obs = format_observations(results)
    assert "$ calc" in obs
    assert "$ echo" in obs


def test_dig_intermediate_then_final():
    reset_dig_state()
    site = "test-site-runner"
    r1 = run_tool(TOOLS["dig"], ["--site", site])
    assert r1.intermediate
    assert "loose soil" in r1.observation

    r2 = run_tool(TOOLS["dig"], ["--site", site])
    assert r2.intermediate
    assert "chest" in r2.observation

    r3 = run_tool(TOOLS["dig"], ["--site", site])
    assert not r3.intermediate
    assert "42" in r3.observation
    reset_dig_state()


def test_dig_summary_is_entropy_safe():
    summary = TOOLS["dig"].summary.lower()
    assert "never" in summary
    assert "document" in summary or "transcript" in summary


def test_module_argv_uses_python():
    assert TOOLS["calc"].argv[0] == sys.executable
    assert TOOLS["calc"].argv[1] == "-m"
    assert "sallm.tools.calc" in TOOLS["calc"].argv


def test_builtin_tools_none_and_unknown():
    assert builtin_tools("none") == {}
    try:
        builtin_tools("nope")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown" in str(exc).lower()
