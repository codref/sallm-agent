"""Runner tests — no LLM required."""

from __future__ import annotations

import sys

from sallm.cli.tools import CHAT_TOOLS
from sallm.toolapps.dig import reset_dig_state
from sallm.tools import (
    format_observations,
    help_text,
    parse_run_blocks,
    run_many,
    run_tool,
)


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


def test_calc_subprocess():
    result = run_tool(CHAT_TOOLS["calc"], ["--expression", "2+2"])
    assert result.returncode == 0
    assert result.observation == "4"


def test_calc_power():
    result = run_tool(CHAT_TOOLS["calc"], ["--expression", "2**10"])
    assert result.observation == "1024"


def test_echo_subprocess():
    result = run_tool(CHAT_TOOLS["echo"], ["--text", "hello"])
    assert result.returncode == 0
    assert result.observation == "hello"


def test_unknown_tool():
    results = run_many(CHAT_TOOLS, [["nope", "--x", "1"]])
    assert len(results) == 1
    assert "unknown tool" in results[0].observation


def test_help_text():
    text = help_text(CHAT_TOOLS["calc"])
    assert "--expression" in text or "-e" in text


def test_run_many_concurrent():
    results = run_many(
        CHAT_TOOLS,
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
    r1 = run_tool(CHAT_TOOLS["dig"], ["--site", site])
    assert r1.intermediate
    assert "loose soil" in r1.observation

    r2 = run_tool(CHAT_TOOLS["dig"], ["--site", site])
    assert r2.intermediate
    assert "chest" in r2.observation

    r3 = run_tool(CHAT_TOOLS["dig"], ["--site", site])
    assert not r3.intermediate
    assert "42" in r3.observation
    reset_dig_state()


def test_module_argv_uses_python():
    assert CHAT_TOOLS["calc"].argv[0] == sys.executable
    assert CHAT_TOOLS["calc"].argv[1] == "-m"
