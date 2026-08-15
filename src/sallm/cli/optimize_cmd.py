"""sallm optimize — offline prompt profile search."""

from __future__ import annotations

from pathlib import Path

import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from sallm.llm import ThinkingTruncated, complete
from sallm.messages import DEFAULT_API_BASE
from sallm.prompt import DEFAULT_PROFILE_PATH, CompiledProfile
from sallm.tools import builtin_tools

console = Console()

JSON_TASKS = frozenset({"controller", "extractor", "ingest", "remember"})


def _case_mark(score) -> str:
    if score.mandatory_fail:
        return f"[red]FAIL[/] q={score.quality:.2f}"
    if score.quality >= 1.0:
        return "[green]ok[/]  "
    return f"[yellow]weak[/] q={score.quality:.2f}"


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def register(app: typer.Typer):
    @app.command("optimize")
    def optimize_cmd(
        dataset: str = typer.Option(..., "--dataset", help="JSONL train/dev cases"),
        profile: str = typer.Option(
            "",
            "--profile",
            help=(
                "Input CompiledProfile JSON (instructions + budgets). "
                f"Default: --out if it exists, else packaged {DEFAULT_PROFILE_PATH.name}"
            ),
        ),
        out: str = typer.Option(
            "",
            "--out",
            help=(
                "Write merged profile here (preserves other tasks already in the file). "
                "Default: do not write."
            ),
        ),
        in_place: bool = typer.Option(
            False,
            "--in-place",
            help="Write the winner back into --profile (or the default profile)",
        ),
        model: str = typer.Option(
            "",
            "--model",
            "-m",
            help="Student model (default: profile target_model)",
        ),
        teacher: str = typer.Option(
            "",
            "--teacher",
            help="Teacher model for rewrites (default: student --model)",
        ),
        api_base: str = typer.Option(DEFAULT_API_BASE, "--api-base"),
        task: str = typer.Option(
            "controller",
            "--task",
            help="controller|extractor|ingest|converse|rewriter",
        ),
        tools: str = typer.Option(
            "echo,calc",
            "--tools",
            help="Built-in tools for converse eval (same ```run contract as chat)",
        ),
        rounds: int = typer.Option(
            6, "--rounds", help="Failure-driven rewrite iterations"
        ),
        seed: int = typer.Option(0, "--seed"),
        evaluate_only: bool = typer.Option(
            False, "--evaluate-only/--search", help="Score profile instruction only"
        ),
        search_budgets: bool = typer.Option(
            False,
            "--search-budgets/--no-search-budgets",
            help="Search generation budgets (temperature, max tokens, think)",
        ),
        budgets_only: bool = typer.Option(
            False,
            "--budgets-only",
            help="Only search budgets (skip instruction rewrite)",
        ),
    ):
        """Search/evaluate a compiled prompt profile (offline)."""
        from sallm.control import (
            CONTROL_INSTRUCTION,
            EXTRACT_INSTRUCTION,
            INGEST_INSTRUCTION,
            _parse_json,
        )
        from sallm.models import resolve_model_profile
        from sallm.optimization import (
            iterate_on_failures,
            load_jsonl,
            merge_task_instruction,
            score_case,
            write_profile,
        )
        from sallm.optimization.artifacts import merge_budgets
        from sallm.optimization.budgets import search_budgets as run_budget_search
        from sallm.optimization.program import (
            control_user_prompt,
            converse_messages,
            extract_user_prompt,
            materialize_got,
            tools_text_for,
        )

        if budgets_only:
            search_budgets = True
        if evaluate_only and budgets_only:
            raise typer.BadParameter(
                "use --budgets-only (with optional --out) or --evaluate-only, not both"
            )

        out_path = (out or "").strip()
        explicit_profile = (profile or "").strip()
        if explicit_profile:
            profile_path = explicit_profile
        elif out_path and Path(out_path).is_file():
            # Continue the artifact being written (multi-task optimize).
            profile_path = out_path
        else:
            profile_path = str(DEFAULT_PROFILE_PATH)
        if not Path(profile_path).is_file():
            raise typer.BadParameter(f"profile not found: {profile_path}")
        compiled = CompiledProfile.load(profile_path)
        base_model_profile = resolve_model_profile(compiled.target_model or None)
        model_profile = compiled.apply_budgets(base_model_profile)
        student_model = (model or compiled.target_model or "").strip()
        if not student_model:
            raise typer.BadParameter("set --model or profile.target_model")
        teacher_model = (teacher or student_model).strip()

        cases = [
            c
            for c in load_jsonl(dataset)
            if task == "all"
            or c.task == task
            or (task in ("ingest", "remember") and c.task in ("ingest", "remember"))
        ]
        if not cases:
            raise typer.BadParameter(f"no cases for task={task!r} in {dataset}")

        baselines = {
            "controller": CONTROL_INSTRUCTION,
            "extractor": EXTRACT_INSTRUCTION,
            "ingest": INGEST_INSTRUCTION,
            "remember": INGEST_INSTRUCTION,
            "converse": "Answer the user clearly and briefly.",
            "rewriter": "Rewrite the user turn as a short retrieval query sentence.",
        }
        if task not in baselines and task != "all":
            raise typer.BadParameter(
                f"unknown task={task!r}; choose from "
                f"{', '.join(k for k in baselines if k != 'remember')}"
            )
        profile_task = "ingest" if task == "remember" else task
        baseline = str(compiled.instructions.get(profile_task) or "").strip() or (
            baselines.get(profile_task) or baselines["converse"]
        )
        demos = str(compiled.demonstrations.get(profile_task) or "")

        json_mode = profile_task in JSON_TASKS

        tools_flag = (tools or "").strip().lower()
        try:
            registry = builtin_tools(
                "none" if tools_flag in ("", "none", "off") else tools
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        tools_text = tools_text_for(registry)

        def complete_kw_for(budgets: dict) -> dict:
            overlay = CompiledProfile(
                target_model=compiled.target_model,
                instructions=compiled.instructions,
                demonstrations=compiled.demonstrations,
                budgets=dict(budgets or {}),
                metadata={},
            )
            mp = overlay.apply_budgets(base_model_profile)
            if json_mode:
                max_tokens = (
                    mp.extract_max_tokens
                    if profile_task in ("extractor", "ingest")
                    else mp.control_max_tokens
                )
            else:
                max_tokens = mp.max_output_tokens
            return mp.complete_kwargs(max_tokens=max_tokens, json_mode=json_mode)

        def make_predict_fn(budgets: dict):
            complete_kw = complete_kw_for(budgets)

            def predict_fn(case, instruction, demos_text):
                if profile_task == "converse":
                    messages = converse_messages(
                        case,
                        instruction,
                        compiled,
                        tools_text=tools_text,
                        demos=demos_text,
                    )
                    try:
                        result = complete(
                            model=student_model,
                            messages=messages,
                            api_base=api_base,
                            **complete_kw,
                        )
                    except ThinkingTruncated:
                        result = {"content": "", "usage": {}, "elapsed_ms": 0}
                    content = result.get("content") or ""
                    got = materialize_got(content, registry, case.expected)
                    return got, {
                        "total_tokens": (result.get("usage") or {}).get(
                            "total_tokens", 0
                        ),
                        "elapsed_ms": result.get("elapsed_ms", 0),
                    }
                if profile_task == "controller":
                    prompt = control_user_prompt(case, instruction, demos_text)
                elif profile_task in ("extractor", "ingest"):
                    prompt = extract_user_prompt(case, instruction, demos_text)
                else:
                    prompt = instruction
                    if demos_text:
                        prompt += "\nExamples:\n" + demos_text
                    prompt += "\nInput:\n" + str(case.input)
                result = complete(
                    model=student_model,
                    messages=[{"role": "user", "content": prompt}],
                    api_base=api_base,
                    **complete_kw,
                )
                content = result.get("content") or ""
                if json_mode:
                    parsed = _parse_json(content)
                    got = parsed if parsed is not None else content
                else:
                    got = content
                return got, {
                    "total_tokens": (result.get("usage") or {}).get("total_tokens", 0),
                    "elapsed_ms": result.get("elapsed_ms", 0),
                }

            return predict_fn

        predict_fn = make_predict_fn(compiled.budgets)

        if budgets_only:
            mode = "budgets-only"
        elif evaluate_only:
            mode = "evaluate-only"
        elif search_budgets:
            mode = "search+budgets"
        else:
            mode = "search"
        console.print(
            Panel(
                f"[bold]{mode}[/]  task=[cyan]{profile_task}[/]  cases={len(cases)}\n"
                f"student [cyan]{student_model}[/]\n"
                f"teacher [cyan]{teacher_model if not (evaluate_only or budgets_only) else '—'}[/]\n"
                f"budgets [cyan]{compiled.budgets}[/]\n"
                f"profile {profile_path}",
                title="optimize",
                border_style="cyan",
            )
        )

        if evaluate_only:
            scores = []
            with _progress() as progress:
                tid = progress.add_task("evaluate", total=len(cases))
                for c in cases:
                    progress.update(tid, description=f"eval [cyan]{c.id}[/]")
                    s = score_case(
                        c, instruction=baseline, demos=demos, predict_fn=predict_fn
                    )
                    scores.append(s)
                    progress.console.print(f"  {_case_mark(s)}  [dim]{c.id}[/]")
                    progress.advance(tid)
            avg_q = sum(s.quality for s in scores) / len(scores)
            lines = [
                f"profile: {profile_path}",
                f"task: {profile_task}",
                f"student: {student_model}",
                f"quality={avg_q:.3f} n={len(scores)}",
            ]
            fails = []
            for c, s in zip(cases, scores):
                if s.mandatory_fail:
                    flag = "FAIL"
                    fails.append(c.id)
                else:
                    flag = f"q={s.quality:.2f}"
                lines.append(f"  {c.id}: {flag}")
            if fails:
                lines.append(f"mandatory misses: {', '.join(fails)}")
            console.print(
                Panel(
                    "\n".join(lines),
                    title="evaluate",
                    border_style="red" if fails else "green",
                )
            )
            return

        dest = out_path
        if in_place and dest:
            raise typer.BadParameter("use --out PATH or --in-place, not both")
        if in_place:
            dest = profile_path

        state = {"bar": None, "tid": None, "status": None}

        def on_event(kind: str, **payload):
            if kind == "eval_begin":
                label = payload.get("label") or "eval"
                n = int(payload.get("n") or 0)
                console.print(f"\n[bold]▸[/] scoring [cyan]{label}[/] ({n} cases)")
                progress = _progress()
                progress.start()
                tid = progress.add_task(f"score {label}", total=n)
                state["bar"] = progress
                state["tid"] = tid
            elif kind == "case_begin":
                case = payload["case"]
                bar = state.get("bar")
                tid = state.get("tid")
                if bar is not None and tid is not None:
                    bar.update(tid, description=f"score [cyan]{case.id}[/]")
            elif kind == "case":
                case = payload["case"]
                score = payload["score"]
                bar = state.get("bar")
                tid = state.get("tid")
                if bar is not None and tid is not None:
                    bar.console.print(f"  {_case_mark(score)}  [dim]{case.id}[/]")
                    bar.advance(tid)
            elif kind == "eval_end":
                bar = state.get("bar")
                if bar is not None:
                    bar.stop()
                state["bar"] = None
                state["tid"] = None
                snap = payload.get("snapshot") or {}
                kept = payload.get("kept")
                label = payload.get("label") or snap.get("name") or "?"
                miss = snap.get("mandatory_misses", "?")
                quality = snap.get("quality", 0)
                fails = ", ".join(snap.get("fails") or []) or "none"
                deltas = snap.get("deltas") or {}
                keep_txt = ""
                if label not in ("baseline", "budgets:baseline"):
                    keep_txt = "  [green]kept[/]" if kept else "  [dim]rejected[/]"
                extra = ""
                if deltas:
                    extra = f"\n  try: [cyan]{deltas}[/]"
                console.print(
                    f"[bold]◂[/] {label}: quality={quality:.3f}  "
                    f"mandatory_misses={miss}{keep_txt}"
                    f"{extra}\n"
                    f"  fails: [dim]{fails}[/]"
                )
            elif kind == "teacher_begin":
                r = payload.get("round")
                fail_ids = payload.get("fail_ids") or []
                console.print(
                    f"\n[bold]▸[/] teacher rewrite round [cyan]{r}[/] "
                    f"({len(fail_ids)} fails) → [cyan]{teacher_model}[/]"
                )
                status = console.status(
                    f"[dim]teacher rewriting with {teacher_model}…[/]",
                    spinner="dots",
                )
                status.start()
                state["status"] = status
            elif kind == "teacher_end":
                status = state.get("status")
                if status is not None:
                    status.stop()
                state["status"] = None
                console.print("[dim]teacher done[/]")
            elif kind == "round_skip":
                console.print(
                    f"[yellow]round {payload.get('round')} skipped[/] "
                    f"({payload.get('reason')})"
                )

        winner_instruction = baseline
        instruction_report: dict = {}
        budget_report: dict = {}
        winner_budgets = dict(compiled.budgets or {})

        try:
            if not budgets_only:
                winner_instruction, instruction_report = iterate_on_failures(
                    baseline=baseline,
                    task=profile_task,
                    cases=cases,
                    predict_fn=predict_fn,
                    model=teacher_model,
                    api_base=api_base,
                    rounds=rounds,
                    seed=seed,
                    demos=demos,
                    teacher_max_tokens=512 if profile_task == "converse" else 256,
                    on_event=on_event,
                )

            if search_budgets:
                console.print(
                    f"\n[bold]▸[/] budget search for [cyan]{profile_task}[/] "
                    f"(instruction fixed)"
                )
                winner_budgets, budget_report = run_budget_search(
                    baseline=dict(compiled.budgets or {}),
                    task=profile_task,
                    cases=cases,
                    instruction=winner_instruction,
                    predict_factory=make_predict_fn,
                    demos=demos,
                    seed=seed,
                    on_event=on_event,
                )
        finally:
            bar = state.get("bar")
            if bar is not None:
                bar.stop()
            status = state.get("status")
            if status is not None:
                status.stop()

        # Merge into the destination if it already exists so sibling tasks
        # (e.g. converse) survive a later --task controller --out same-file.
        merge_src = dest if dest and Path(dest).is_file() else profile_path
        raw = Path(merge_src).read_text(encoding="utf-8")
        data = json.loads(raw)
        if not budgets_only:
            data = merge_task_instruction(
                data,
                task=profile_task,
                instruction=winner_instruction,
                dataset_fingerprint=instruction_report.get("dataset_fingerprint")
                or "",
                metrics=instruction_report.get("final") or {},
                seed=seed,
            )
        if search_budgets:
            data = merge_budgets(
                data,
                budgets=winner_budgets,
                dataset_fingerprint=budget_report.get("dataset_fingerprint") or "",
                metrics=budget_report.get("final") or {},
                seed=seed,
                optimize_task=f"budgets:{profile_task}",
            )

        if dest:
            write_profile(dest, data)
            parts = []
            if not budgets_only:
                parts.append(f"instructions.{profile_task}")
            if search_budgets:
                parts.append("budgets")
            wrote = f"wrote {', '.join(parts)} → [cyan]{dest}[/]"
        else:
            wrote = "not written (pass --out PATH or --in-place)"
            if not budgets_only:
                wrote += f"\nwinner instructions.{profile_task}:\n{winner_instruction}"
            if search_budgets:
                wrote += f"\nwinner budgets:\n{json.dumps(winner_budgets, indent=2)}"

        final = {}
        if search_budgets:
            final = (budget_report.get("final") or {}).get("best") or {}
        elif instruction_report:
            final = (instruction_report.get("final") or {}).get("best") or (
                instruction_report.get("final") or {}
            ).get("baseline") or {}

        remaining = instruction_report.get("remaining_fails") or []
        if search_budgets:
            passed = bool(budget_report.get("passed"))
        else:
            passed = bool(instruction_report.get("passed"))
        status = "passed" if passed else "best-so-far"
        extra = ""
        if remaining and not budgets_only:
            extra = f"\nmandatory still failing: {', '.join(remaining)}"
        if search_budgets:
            extra += f"\nbudgets: {winner_budgets}"

        round_lines = []
        for snap in instruction_report.get("rounds") or []:
            if snap.get("skipped"):
                round_lines.append(
                    f"  {snap.get('name')}: skipped ({snap.get('skipped')})"
                )
                continue
            round_lines.append(
                f"  {snap.get('name')}: q={snap.get('quality', 0):.3f}  "
                f"miss={snap.get('mandatory_misses', 0)}  "
                f"fails={', '.join(snap.get('fails') or []) or '—'}"
            )
        for snap in budget_report.get("rounds") or []:
            deltas = snap.get("deltas") or {}
            delta_s = f"  Δ{deltas}" if deltas else ""
            mark = ""
            if snap.get("name") != "baseline" and "kept" in snap:
                mark = " kept" if snap.get("kept") else " rejected"
            round_lines.append(
                f"  {snap.get('name')}: q={snap.get('quality', 0):.3f}  "
                f"miss={snap.get('mandatory_misses', 0)}{mark}{delta_s}"
            )
        rounds_blob = "\n".join(round_lines) if round_lines else "  (none)"
        console.print(
            Panel(
                f"{status}\n"
                f"student {student_model}  teacher {teacher_model}\n"
                f"{wrote}\n"
                f"quality={final.get('quality', 0):.3f}  "
                f"mandatory_misses={final.get('mandatory_misses', 0)}"
                f"{extra}\n\n"
                f"rounds:\n{rounds_blob}",
                title="optimize",
                border_style="green" if passed else "yellow",
            )
        )
