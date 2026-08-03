"""sallm optimize — offline prompt profile search."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from sallm.llm import complete
from sallm.messages import DEFAULT_API_BASE, DEFAULT_MODEL

console = Console()


def register(app: typer.Typer):
    @app.command("optimize")
    def optimize_cmd(
        dataset: str = typer.Option(..., "--dataset", help="JSONL train/dev cases"),
        out: str = typer.Option(..., "--out", help="Output profile JSON path"),
        model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
        api_base: str = typer.Option(DEFAULT_API_BASE, "--api-base"),
        task: str = typer.Option(
            "controller", "--task", help="controller|extractor|converse"
        ),
        candidates: int = typer.Option(4, "--candidates"),
        seed: int = typer.Option(0, "--seed"),
        evaluate_only: bool = typer.Option(
            False, "--evaluate-only/--search", help="Score baseline only"
        ),
    ):
        """Search/evaluate a neutral compiled prompt profile (offline)."""
        from sallm.control import CONTROL_INSTRUCTION, EXTRACT_INSTRUCTION, _parse_json
        from sallm.optimization import (
            Candidate,
            load_jsonl,
            propose_instructions,
            save_artifact,
            successive_halving,
        )

        cases = [c for c in load_jsonl(dataset) if c.task == task or task == "all"]
        if not cases:
            raise typer.BadParameter(f"no cases for task={task!r} in {dataset}")

        baselines = {
            "controller": CONTROL_INSTRUCTION,
            "extractor": EXTRACT_INSTRUCTION,
            "converse": "Answer the user clearly and briefly.",
            "rewriter": "Rewrite the user turn as a short retrieval query sentence.",
        }
        baseline = baselines.get(task, baselines["converse"])

        def predict_fn(case, instruction, demos):
            prompt = instruction
            if demos:
                prompt += "\nExamples:\n" + demos
            prompt += "\nInput:\n" + str(case.input)
            result = complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_base=api_base,
                max_tokens=256,
            )
            content = result.get("content") or ""
            parsed = _parse_json(content)
            got = parsed if parsed is not None else content
            return got, {
                "total_tokens": (result.get("usage") or {}).get("total_tokens", 0),
                "elapsed_ms": result.get("elapsed_ms", 0),
            }

        if evaluate_only:
            from sallm.optimization.search import score_case

            scores = [
                score_case(
                    c, instruction=baseline, demos="", predict_fn=predict_fn
                )
                for c in cases
            ]
            avg_q = sum(s.quality for s in scores) / len(scores)
            console.print(f"baseline quality={avg_q:.3f} n={len(scores)}")
            return

        texts = propose_instructions(
            baseline=baseline,
            task=task,
            model=model,
            api_base=api_base,
            n=candidates,
            seed=seed,
        )
        cands = [
            Candidate(name=f"c{i}", instruction=t) for i, t in enumerate(texts)
        ]
        winner, report = successive_halving(
            cands, cases, predict_fn=predict_fn, seed=seed
        )
        save_artifact(
            out,
            target_model=model,
            instructions={task: winner.instruction},
            demonstrations={task: winner.demos},
            budgets={},
            dataset_fingerprint=report["dataset_fingerprint"],
            metrics=report.get("final") or {},
            seed=seed,
        )
        console.print(
            Panel(
                f"winner: [cyan]{winner.name}[/]\n"
                f"wrote: [cyan]{out}[/]\n"
                f"fingerprint: {report['dataset_fingerprint']}",
                title="optimize",
                border_style="green",
            )
        )
