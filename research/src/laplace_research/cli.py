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
    evolve.add_argument("action", choices=["init", "run", "report", "senses"])
    evolve.add_argument("--run-id", required=True)
    evolve.add_argument("--generations", type=int, default=4)
    evolve.add_argument("--children", type=int, default=2, help="children per island per generation")
    evolve.add_argument("--workers", type=int, default=8, help="evaluation processes")
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
            Run.init(run_dir, CACHE / "candles")
        elif args.action == "senses":
            from laplace_research.evolve import senses
            from laplace_research.evolve.panel import Panel

            panel = Panel.load(run_dir / "panel.npz")
            panel.with_senses(senses.build(panel, CACHE / "senses")).save(run_dir / "panel.npz")
            for name, values in Panel.load(run_dir / "panel.npz").senses.items():
                print(f"{name}: {np.isfinite(values).mean(axis=1).round(3).tolist()} of hours known, per asset")
        elif args.action == "run":
            asyncio.run(Run(run_dir).evolve(args.generations, args.children, args.workers))
        else:
            print(evolve_report.render(evolve_report.report(run_dir, workers=args.workers)))
    return 0
