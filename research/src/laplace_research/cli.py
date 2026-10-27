"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from laplace_research import __version__

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
CACHE = ROOT / ".cache"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laplace-research")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    majors = sub.add_parser("majors", help="blinded Majors backtest (PoC 1)")
    majors.add_argument("action", choices=["run", "report"])
    majors.add_argument("--run-id", required=True)
    majors.add_argument("--days", type=int, default=30)
    majors.add_argument("--step-hours", type=int, default=6)
    majors.add_argument("--concurrency", type=int, default=16)
    majors.add_argument("--agents", help="comma-separated model/directive ids (default: all)")

    evolve = sub.add_parser("evolve", help="evolutionary search over forecasting programs (PoC 2)")
    evolve.add_argument("action", choices=["init", "run", "report", "senses", "events", "rolling"])
    evolve.add_argument("--run-id", required=True)
    evolve.add_argument("--generations", type=int, default=4)
    evolve.add_argument("--children", type=int, default=2, help="children per island per generation")
    evolve.add_argument("--workers", type=int, default=8, help="evaluation processes")
    evolve.add_argument(
        "--ages", action="store_true", help="rolling: also score every fit on every later month (rolling-ages.json)"
    )
    evolve.add_argument(
        "--seed", type=Path, help="init: the program every island starts from (default: the naive seed)"
    )
    evolve.add_argument(
        "--reference",
        choices=["naive", "ev-har"],
        default="naive",
        help="init: what a program's gain is measured against",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "majors":
        from laplace_research.experiments import majors, report

        run_dir = RUNS / args.run_id
        if args.action == "run":
            agents = majors.default_agents()
            if args.agents:
                wanted = set(args.agents.split(","))
                agents = [a for a in agents if a.agent_id in wanted]
            majors.run(
                run_dir,
                days=args.days,
                step_hours=args.step_hours,
                concurrency=args.concurrency,
                agents=agents,
            )
        print(report.render(report.report(run_dir)))
    if args.command == "evolve":
        # Evaluation runs many single-threaded workers; keep numerical libraries from spawning thread pools in each.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        import asyncio

        from laplace_research.evolve import report as evolve_report
        from laplace_research.evolve.run import Run

        run_dir = RUNS / f"evolve-{args.run_id}"
        if args.action == "init":
            from laplace_research.evolve.islands import SEED_PROGRAM

            seed_code = args.seed.read_text() if args.seed else SEED_PROGRAM
            Run.init(run_dir, CACHE / "candles", seed_code=seed_code, reference=args.reference)
        elif args.action == "senses":
            from laplace_research.evolve import senses
            from laplace_research.evolve.panel import Panel

            panel = Panel.load(run_dir / "panel.npz")
            panel.with_senses(senses.build(panel, CACHE / "senses")).save(run_dir / "panel.npz")
            for name, values in Panel.load(run_dir / "panel.npz").senses.items():
                print(f"{name}: {np.isfinite(values).mean(axis=1).round(3).tolist()} of hours known, per asset")
        elif args.action == "events":
            import json

            from laplace_research.data import events
            from laplace_research.evolve.event_study import event_study, overlay_screen, render, render_screen

            run = Run(run_dir)
            study = event_study(run.panel, events.load())
            print(render(study))
            if "rv" in run.panel.senses and "dvol" in run.panel.senses:
                study["overlay_screen"] = overlay_screen(run.panel, run.config, events.load())
                print()
                print(render_screen(study["overlay_screen"]))
            (run_dir / "events.json").write_text(json.dumps(study, indent=1) + "\n")
        elif args.action == "rolling":
            import json
            from datetime import UTC, datetime

            from laplace_research.data import events
            from laplace_research.evolve.rolling import render as render_rolling
            from laplace_research.evolve.rolling import rolling

            run = Run(run_dir)
            first = datetime.fromisoformat(run.config.search[0]).replace(tzinfo=UTC)
            end = datetime.fromtimestamp(int(run.panel.ts[-1]) + 3600, UTC)
            boundaries = [datetime(2026, m, 1, tzinfo=UTC) for m in range(3, 13)]
            boundaries = [b for b in boundaries if b < end] + [end]
            result = rolling(run.panel, events.load(), first, boundaries, ages=args.ages)
            name = "rolling-ages.json" if args.ages else "rolling.json"
            (run_dir / name).write_text(json.dumps(result, indent=1) + "\n")
            print(render_rolling(result))
        elif args.action == "run":
            asyncio.run(Run(run_dir).evolve(args.generations, args.children, args.workers))
        else:
            print(evolve_report.render(evolve_report.report(run_dir, workers=args.workers)))
    return 0
