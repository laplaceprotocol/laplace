"""Command line: freeze a release; seal, commit, publish and settle issues; verify an issue; scores; recent issues."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from eth_hash.auto import keccak

from laplace_engine import candles, chain, files, issue, notify, publish, release, settle, verify
from laplace_research.data.candles import fetch_hourly
from laplace_research.evolve.feed import HORIZONS, product

log = logging.getLogger("laplace_engine")


def issue_dir(data: Path, number: int) -> Path:
    return data / "issues" / str(number)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laplace-engine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("release", help="freeze a research run into a release")
    freeze.add_argument("--run", type=Path, required=True, help="research run directory (with report.json)")
    freeze.add_argument("--name", required=True)
    freeze.add_argument("--out", type=Path, required=True, help="release directory (public part, in the repo)")
    freeze.add_argument("--private", type=Path, required=True, help="where private program sources are kept")

    extend = sub.add_parser("extend", help="add shadow forecasters to a release: the monkey and pools fitted on a run")
    extend.add_argument("--from", dest="source", type=Path, required=True, help="the release to extend")
    extend.add_argument("--private", type=Path, required=True)
    extend.add_argument("--run", type=Path, required=True, help="research run whose validation window fits the pools")
    extend.add_argument("--name", required=True)
    extend.add_argument("--out", type=Path, required=True)
    extend.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))

    seal = sub.add_parser("issue", help="seal the current (or a given) issue; optionally commit and publish it")
    seal.add_argument("--release", type=Path, required=True)
    seal.add_argument("--private", type=Path, required=True)
    seal.add_argument("--data", type=Path, required=True, help="engine data directory (candle cache, issues)")
    seal.add_argument("--issue", type=int, help="issue number (hours since the epoch); default: the current hour")
    seal.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    seal.add_argument("--wait-minutes", type=float, default=10.0, help="how long to wait for the candles")
    seal.add_argument("--commit", action="store_true", help="commit the root on-chain (key: LAPLACE_COMMIT_KEY_FILE)")
    seal.add_argument("--publish", action="store_true", help="publish committed issues (credentials: LAPLACE_R2_*)")
    seal.add_argument("--settle", action="store_true", help="settle and score every question that has become due")
    seal.add_argument(
        "--judging",
        type=Path,
        action="append",
        default=[],
        help="a pre-registered rule in force (hashed into manifests)",
    )
    seal.add_argument("--rpc", default=",".join(chain.RPC_URLS), help="RPC endpoints, comma-separated, tried in turn")
    seal.add_argument("--ledger", default=chain.LEDGER)

    check = sub.add_parser("verify", help="check an issue's files, and that they were committed in time")
    check.add_argument("--issue", type=int, required=True)
    check.add_argument("--data", type=Path, help="read the files from this data directory instead of the public URL")
    check.add_argument("--url", default=publish.PUBLIC_URL)
    check.add_argument("--offline", action="store_true", help="check the files only, not the chain")
    check.add_argument("--rpc", default=",".join(chain.RPC_URLS))
    check.add_argument("--ledger", default=chain.LEDGER)

    board = sub.add_parser("scoreboard", help="mean log scores and gains over the baselines, from settled questions")
    board.add_argument("--data", type=Path, required=True)
    board.add_argument("--horizon", type=int, choices=HORIZONS, help="gains at one horizon instead of overall")

    alert = sub.add_parser("alert", help="report a crashed hourly run (systemd OnFailure)")
    alert.add_argument("--unit", default="laplace-issue.service")

    digest = sub.add_parser("digest", help="send the daily summary of the record")
    digest.add_argument("--data", type=Path, required=True)
    digest.add_argument("--rpc", default=",".join(chain.RPC_URLS))
    digest.add_argument("--ledger", default=chain.LEDGER)
    digest.add_argument("--dry-run", action="store_true", help="print the digest without sending it")

    status = sub.add_parser("status", help="recent issues: sealed, committed, published")
    status.add_argument("--data", type=Path, required=True)
    status.add_argument("--hours", type=int, default=24)
    return parser


def cmd_release(args: argparse.Namespace) -> int:
    from laplace_engine.freeze import freeze

    frozen = freeze(args.run, args.name)
    release.write(frozen, args.out, args.private)
    for entry in frozen.describe():
        print(f"{entry['name']:12s} {entry['kind']:8s} {entry['id']}")
    return 0


REPO = Path(__file__).resolve().parents[3]


def judging_entries(paths: list[Path], alerts: notify.Alerts) -> list[dict[str, str]]:
    """Each rule's path (relative to the repository) and keccak256. A missing rule is alerted and left out rather than
    costing the hour."""
    entries = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            alerts.add(f"judging rule {path} not found; sealed without it", key=f"judging:{path}", every=HOURS_6)
            continue
        name = resolved.relative_to(REPO).as_posix() if resolved.is_relative_to(REPO) else resolved.name
        entries.append({"path": name, "keccak256": "0x" + bytes(keccak(resolved.read_bytes())).hex()})
    return entries


def cmd_extend(args: argparse.Namespace) -> int:
    """Release ``--name``: every forecaster of ``--from`` unchanged, plus the monkey and two pools of all programs
    fitted on the run's validation window (one set of weights, and one per horizon), saved with a holdout screening."""
    from laplace_engine import baselines, fit

    base = release.load(args.source, args.private)
    monkey = release.Program("monkey", baselines.MONKEY, True, "baseline: equal odds on every bin")
    programs = (*base.programs, monkey)
    windows = fit.run_windows(args.run, programs, workers=args.workers)
    everyone = {p.name: windows.validation[p.name] for p in programs}
    plus = release.Pool("demon-plus", fit.fit_weights(everyone, windows.realized_validation, HORIZONS))
    by_horizon = tuple((h, fit.fit_weights(everyone, windows.realized_validation, (h,))) for h in HORIZONS)
    notes = (
        f"{base.name}'s forecasters unchanged, plus shadow forecasters: the monkey (equal odds on every bin) and "
        f"pools of all {len(programs)} programs fitted by EM on the validation window of {args.run.name}: "
        "demon-plus (one set of weights, left out if it fits to an existing pool's weights) and demon-h (its own "
        "weights at each horizon). Exploratory: no rule judges them until they get one."
    )
    # A fitted pool identical to an existing one (EM can land on the same weights) would duplicate its leaves.
    program_ids = {p.name: p.id for p in programs}
    taken = set(program_ids.values()) | {pool.id(program_ids) for pool in base.pools}
    new_pools: list[release.AnyPool] = []
    for pool in (plus, release.HorizonPool("demon-h", by_horizon)):
        if pool.id(program_ids) in taken:
            log.warning("%s fits to the same weights as an existing forecaster; left out", pool.name)
            continue
        taken.add(pool.id(program_ids))
        new_pools.append(pool)
    extended = release.Release(args.name, notes, programs, (*base.pools, *new_pools))
    release.write(extended, args.out, args.private)

    holdout = {p.name: windows.holdout[p.name] for p in programs}
    candidates = {pool.name: fit.mix(holdout, pool.weights_at) for pool in extended.pools}
    screening = {
        "run": args.run.name,
        "note": "holdout gains as development evidence (the holdout was inspected before; the live record is the test)",
        "weights": {pool.name: {str(h): dict(pool.weights_at(h)) for h in HORIZONS} for pool in extended.pools},
        "holdout": fit.compare(windows, candidates),
    }
    (args.out / "screening.json").write_text(json.dumps(screening, indent=1) + "\n")
    for name, by_opponent in screening["holdout"].items():
        for opponent, row in by_opponent.items():
            cells = "  ".join(
                f"{scope} {r['mean']:+.4f} ({r['interval95'][0]:+.4f}..{r['interval95'][1]:+.4f})"
                for scope, r in row.items()
            )
            print(f"{name:10s} vs {opponent:5s}  {cells}")
    for entry in extended.describe():
        print(f"{entry['name']:12s} {entry['kind']:8s} {entry['id']}")
    return 0


def seal_issue(args: argparse.Namespace, number: int, target: Path, judging: list[dict[str, str]]) -> None:
    started = time.monotonic()
    rel = release.load(args.release, args.private)
    args.data.mkdir(parents=True, exist_ok=True)
    # A live issue waits for its candles until the deadline; a past issue (a dry run) only needs a moment.
    now = datetime.now(UTC)
    deadline = max(candles.issue_time(number) + timedelta(minutes=args.wait_minutes), now + timedelta(minutes=1))
    panel = candles.wait_for(number, args.data / "candles", deadline=deadline)
    with tempfile.TemporaryDirectory(dir=args.data) as workdir:
        finals, failures = issue.run_release(rel, panel, Path(workdir), workers=args.workers)
    ledger = issue.Ledger(chain.CHAIN_ID, args.ledger)
    sealed = issue.seal(
        number, rel, panel, finals, failures, ledger=ledger, commit=issue.engine_commit(), judging=judging
    )
    files.write_atomic(target / "forecasts.json", sealed.forecasts)
    files.write_atomic(target / "manifest.json", sealed.manifest)
    for name, error in sealed.failures.items():
        log.warning("issue %d: %s failed: %s", number, name, error)
    root, manifest = sealed.root.hex(), sealed.manifest_hash.hex()
    seconds = time.monotonic() - started
    print(f"issue {number}: {sealed.leaves} forecasts, root 0x{root}, manifest 0x{manifest} ({seconds:.1f}s)")


JOURNAL = "commit-tx.json"  # the signed commit transaction's hash, written before it is sent
GAP = "gap.json"  # an issue whose window closed without a commitment
CONFLICT = "conflict.json"  # an issue committed with a root other than its local files (never published)
SOFT_FAILURE = 3  # the run finished but something needed attention, and the operator was alerted
HOURS_6 = timedelta(hours=6)


def commit_issue(ledger: chain.Ledger, number: int, target: Path, alerts: notify.Alerts) -> bool:
    """Commit the issue, retrying transient failures until shortly before its window closes."""
    closes = candles.issue_time(number) + timedelta(seconds=chain.COMMIT_WINDOW_S)
    if (closes - datetime.now(UTC)).total_seconds() <= 30:
        alerts.add(f"issue {number} not committed: its window closed at {closes:%H:%M:%S} UTC before it was sealed")
        return False
    key_file = os.environ.get("LAPLACE_COMMIT_KEY_FILE")
    if not key_file:
        raise RuntimeError("LAPLACE_COMMIT_KEY_FILE is not set")
    account = chain.load_account(Path(key_file))
    manifest = (target / "manifest.json").read_bytes()
    body = json.loads(manifest)
    last_error = "no attempt"
    while True:
        remaining = (closes - datetime.now(UTC)).total_seconds()
        if remaining <= 30:
            alerts.add(f"issue {number} not committed before its window closed ({last_error[:200]})")
            return False
        try:
            commitment = ledger.commit(
                account,
                number,
                bytes.fromhex(body["root"][2:]),
                bytes(keccak(manifest)),
                int(body["leaves"]),
                journal=target / JOURNAL,
                timeout_s=min(120.0, remaining - 20),
            )
        except chain.LedgerRevert as exc:
            # The chain's clock can trail ours by a few seconds at the start of the hour; any other refusal is final.
            if exc.reason == "OutsideCommitWindow" and remaining > chain.COMMIT_WINDOW_S - 120:
                time.sleep(5.0)
                continue
            alerts.add(f"issue {number} not committed: {exc}")
            return False
        except Exception as exc:
            log.exception("issue %d: commit attempt failed; retrying", number)
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(15.0, max(remaining - 30, 1.0)))
            continue
        files.write_atomic(target / "commitment.json", commitment.to_json())
        print(f"issue {number}: committed in {commitment.tx} (block {commitment.block})")
        try:
            balance = ledger.balance(account.address)
            if balance < chain.LOW_BALANCE_WEI:
                alerts.add(
                    f"the commit key holds {balance / 1e18:.6f} ETH; top it up", key="balance", every=timedelta(days=1)
                )
        except Exception:
            log.exception("could not read the commit key's balance")
        return True


def reconcile(ledger: chain.Ledger, issues_dir: Path, current: int, alerts: notify.Alerts) -> None:
    """Resolve every sealed issue without a recorded commitment once its window has closed: a commitment that reached
    the chain without being recorded (a lost reply, a crash) is recovered; one that never did is marked as a gap."""
    if not issues_dir.exists():
        return
    for issue_dir in sorted((d for d in issues_dir.iterdir() if d.name.isdigit()), key=lambda d: int(d.name)):
        number = int(issue_dir.name)
        resolved = any((issue_dir / m).exists() for m in ("commitment.json", GAP, CONFLICT))
        if number >= current or resolved or not (issue_dir / "manifest.json").exists():
            continue
        try:
            manifest = (issue_dir / "manifest.json").read_bytes()
            body = json.loads(manifest)
            root, manifest_hash = bytes.fromhex(body["root"][2:]), bytes(keccak(manifest))
            onchain = ledger.root_of(number)
            if onchain == root:
                found = ledger.recover(number, root, manifest_hash, int(body["leaves"]), journal=issue_dir / JOURNAL)
                files.write_atomic(issue_dir / "commitment.json", found.to_json())
                alerts.add(f"issue {number}: recovered a commitment that had not been recorded ({found.tx})")
            elif onchain == bytes(32):
                files.write_atomic(issue_dir / GAP, release.canonical({"issue": number, "reason": "not committed"}))
                alerts.add(f"issue {number} was never committed: a gap in the record")
            else:
                record = {"issue": number, "local_root": body["root"], "onchain_root": "0x" + onchain.hex()}
                files.write_atomic(issue_dir / CONFLICT, release.canonical(record))
                alerts.add(f"issue {number}: CONFLICT, the committed root differs from the local files; not published")
        except Exception as exc:
            log.exception("issue %d: could not reconcile; will retry", number)
            alerts.add(f"issue {number}: could not reconcile ({exc})"[:300], key=f"reconcile:{number}", every=HOURS_6)


@contextmanager
def run_lock(path: Path) -> Iterator[bool]:
    """An exclusive, non-blocking lock: two runs must never seal or commit the same hour."""
    with path.open("a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def cmd_issue(args: argparse.Namespace) -> int:
    args.data.mkdir(parents=True, exist_ok=True)
    with run_lock(args.data / ".lock") as locked:
        if not locked:
            log.warning("another run holds the lock; leaving this hour to it")
            return 0
        return run_issue(args)


def run_issue(args: argparse.Namespace) -> int:
    """Seal, commit, reconcile, publish, settle. Each step is isolated, so one failure never blocks the backlog; what
    went wrong is sent to the operator in one message at the end."""
    number = args.issue if args.issue is not None else candles.current_issue()
    target = issue_dir(args.data, number)
    issues = args.data / "issues"
    alerts = notify.Alerts(notify.Telegram.from_env(), args.data / "alerts.json")
    ok = True
    if (target / "manifest.json").exists():
        log.info("issue %d is already sealed", number)  # never reseal: what may be committed is on disk
    else:
        try:
            seal_issue(args, number, target, judging_entries(args.judging, alerts))
            for name, error in json.loads((target / "manifest.json").read_bytes())["failed"].items():
                alerts.add(f"{name} produced no forecast: {error}"[:300], key=f"forecaster:{name}", every=HOURS_6)
        except Exception as exc:
            log.exception("issue %d: sealing failed", number)
            alerts.add(f"issue {number} could not be sealed: {type(exc).__name__}: {exc}"[:300])
            ok = False
    if args.commit:
        ledger = chain.Ledger(chain.RPC(args.rpc.split(",")), args.ledger)
        if (target / "manifest.json").exists() and not (target / "commitment.json").exists():
            try:
                ok = commit_issue(ledger, number, target, alerts) and ok
            except Exception as exc:
                log.exception("issue %d: commit failed", number)
                alerts.add(f"issue {number} not committed: {type(exc).__name__}: {exc}"[:300])
                ok = False
        reconcile(ledger, issues, candles.current_issue(), alerts)
    store = None
    if args.publish:
        try:
            store = publish.R2.from_env()
            name = json.loads((args.release / release.RELEASE_FILE).read_text())["release"]
            publish.publish_release(store, args.release, name, args.data / "releases" / f"{name}.published")
            publish.publish_pending(store, issues)
        except Exception as exc:
            log.exception("publishing failed; will retry next run")
            alerts.add(f"publishing failed: {type(exc).__name__}: {exc}"[:300], key="publish", every=HOURS_6)
            ok = False
    if args.settle and issues.exists():
        problems: list[tuple[str, str]] = []
        settled = settle.settle_due(
            issues, candles.current_issue(), coinbase_close(args.data / "candles"), problems=problems
        )
        for key, message in problems:
            alerts.add(message[:300], key=key, every=timedelta(hours=12))
        if settled:
            log.info("settled %d question sets", len(settled))
        if store is not None:
            try:
                if publish.publish_settlements(store, issues):
                    publish.publish_scoreboard(store, settle.scoreboard(settle.load_settlements(issues)))
            except Exception as exc:
                log.exception("publishing settlements failed; will retry next run")
                alerts.add(f"publishing settlements failed: {exc}"[:300], key="publish-settlements", every=HOURS_6)
                ok = False
    alerts.flush(f"⚠️ Laplace · issue {number} ({candles.issue_time(number):%Y-%m-%d %H:%M} UTC)")
    return 0 if ok else SOFT_FAILURE


def cmd_alert(args: argparse.Namespace) -> int:
    """Called by systemd when the hourly unit fails (OnFailure=). A run that already reported its own problems exits
    with SOFT_FAILURE and is not reported twice; anything else (a crash, a timeout, a kill) is."""
    status = os.environ.get("MONITOR_EXIT_STATUS", "")
    if status == str(SOFT_FAILURE):
        return 0
    result = os.environ.get("MONITOR_SERVICE_RESULT", "unknown")
    telegram = notify.Telegram.from_env()
    text = (
        f"🚨 Laplace · the hourly run crashed (result: {result}, exit status: {status or 'none'}). "
        f"The next run retries and reconciles automatically. Logs: journalctl --user -u {args.unit} -n 80"
    )
    print(text)
    if telegram is not None:
        telegram.send(text)
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    """A daily summary: the last 24 hours of the record, the scores so far, and the commit key's runway."""
    newest = candles.current_issue()
    window = range(newest - 23, newest + 1)
    committed = [n for n in window if (issue_dir(args.data, n) / "commitment.json").exists()]
    published = [n for n in committed if (issue_dir(args.data, n) / publish.PUBLISHED).exists()]
    missing = [n for n in window if n not in committed]
    lines = [f"📊 Laplace · daily digest · {datetime.now(UTC):%Y-%m-%d %H:%M} UTC"]
    lines.append(
        f"Record: {len(committed)}/24 hours committed, {len(published)} published (issues {window[0]} to {window[-1]})"
        + (f"; missing: {', '.join(map(str, missing))}" if missing else "; no gaps")
    )
    board = settle.scoreboard(settle.load_settlements(args.data / "issues"))
    demons = [e for e in board["forecasters"].values() if e["name"] == "demon"]
    if demons:
        demon = max(demons, key=lambda e: e["questions"])
        for base in ("garch", "har"):
            by_h = demon["vs"].get(base, {}).get("by_horizon", {})
            cells = [
                f"{h}h {by_h[str(h)]['mean']:+.3f} (n={by_h[str(h)]['issues']})" for h in HORIZONS if str(h) in by_h
            ]
            lines.append(f"Demon vs {base.upper()}: " + (", ".join(cells) if cells else "nothing settled yet"))
        if demon["vs"].get("garch", {}).get("interval95") is None:
            lines.append(f"(Directions only: intervals appear after {2 * settle.BLOCK_ISSUES} fully settled issues.)")
    try:
        ledger = chain.Ledger(chain.RPC(args.rpc.split(",")), args.ledger)
        key_file = os.environ.get("LAPLACE_COMMIT_KEY_FILE")
        if key_file and committed:
            account = chain.load_account(Path(key_file))
            balance = ledger.balance(account.address)
            last = chain.Commitment.from_json((issue_dir(args.data, committed[-1]) / "commitment.json").read_bytes())
            cost = ledger.tx_cost(last.tx)
            days = balance / cost / 24 if cost else float("inf")
            lines.append(f"Commit key: {balance / 1e18:.5f} ETH ≈ {days:.0f} days at the last commit's cost")
    except Exception as exc:
        lines.append(f"Commit key: balance unavailable ({type(exc).__name__})")
    text = "\n".join(lines)
    print(text)
    telegram = notify.Telegram.from_env()
    if telegram is not None and not args.dry_run:
        telegram.send(text)
    return 0


def coinbase_close(cache_dir: Path) -> settle.FetchClose:
    """The close of the last public exchange candle opening at or before an hour (an hour without trades repeats the
    previous close), for questions whose settling issue was never committed."""
    client = httpx.Client(timeout=30.0, headers={"User-Agent": "laplace-engine"})

    def fetch(asset: str, open_time: int) -> float:
        hour = datetime.fromtimestamp(open_time, UTC)
        window = fetch_hourly(
            product(asset),
            hour - timedelta(hours=12),
            hour + timedelta(hours=1),
            client=client,
            cache_dir=cache_dir,
            max_filled_hours=12,
        )
        found = [c for c in window if c.ts <= open_time]
        if not found:
            raise ValueError(f"{asset}: no candle in the 12 hours up to {hour:%Y-%m-%d %H:%M}")
        return found[-1].close

    return fetch


def cmd_verify(args: argparse.Namespace) -> int:
    if args.data is not None:
        target = issue_dir(args.data, args.issue)
        manifest, forecasts = (target / "manifest.json").read_bytes(), (target / "forecasts.json").read_bytes()
    else:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            fetched = [
                client.get(f"{args.url}/issues/{args.issue}/{name}") for name in ("manifest.json", "forecasts.json")
            ]
        for response in fetched:
            response.raise_for_status()
        manifest, forecasts = fetched[0].content, fetched[1].content
    result = verify.verify(manifest, forecasts)
    print(f"issue {result.issue}: {result.leaves} forecasts and {result.pools_checked} pool(s) check out")
    print(f"  root 0x{result.root.hex()}, manifest 0x{result.manifest_hash.hex()}")
    if args.offline:
        return 0
    found = chain.Ledger(chain.RPC(args.rpc.split(",")), args.ledger).find(args.issue)
    if found is None:
        print(f"issue {args.issue}: no commitment on-chain")
        return 1
    checks = {
        "root": found.root == "0x" + result.root.hex(),
        "manifest hash": found.manifest == "0x" + result.manifest_hash.hex(),
        "leaf count": found.leaves == result.leaves,
        "commit window": found.in_window,
    }
    stamp = datetime.fromtimestamp(found.timestamp, UTC)
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print(f"committed in {found.tx}, block {found.block}, {stamp:%Y-%m-%d %H:%M:%S} UTC")
    return 0 if all(checks.values()) else 1


def cmd_scoreboard(args: argparse.Namespace) -> int:
    board = settle.scoreboard(settle.load_settlements(args.data / "issues"))
    scope = f"at {args.horizon}h" if args.horizon else "over issues settled at every horizon"
    print(f"{board['settled_issues']} settled issues ({board['first_issue']} to {board['last_issue']}); gains {scope}")
    columns = "  ".join(f"vs {b:>6s}" for b in settle.BASELINES)
    print(f"{'forecaster':12s} {'release':7s} {'questions':>9s} {'log score':>9s}  {columns}")
    for entry in sorted(board["forecasters"].values(), key=lambda e: -e["mean_log_score"]):
        gains = "  ".join(_gain(entry["vs"].get(b), args.horizon) for b in settle.BASELINES)
        print(
            f"{entry['name']:12s} {entry['release']:7s} {entry['questions']:9d} {entry['mean_log_score']:9.4f}  {gains}"
        )
    return 0


def _gain(comparison: dict[str, Any] | None, horizon: int | None) -> str:
    """A gain for the table: '-' where there is nothing to compare yet."""
    if comparison is not None and horizon is not None:
        comparison = comparison["by_horizon"].get(str(horizon))
    if comparison is None or comparison["mean"] is None:
        return f"{'-':>9s}"
    return f"{comparison['mean']:+9.4f}"


def cmd_status(args: argparse.Namespace) -> int:
    newest = candles.current_issue()
    for number in range(newest - args.hours + 1, newest + 1):
        target = issue_dir(args.data, number)
        flags = [
            name if (target / f).exists() else "-" * len(name)
            for name, f in (("sealed", "manifest.json"), ("committed", "commitment.json"), ("published", "published"))
        ]
        flags += [name for name, f in (("GAP", GAP), ("CONFLICT", CONFLICT)) if (target / f).exists()]
        failed = ""
        if (target / "manifest.json").exists():
            failures = json.loads((target / "manifest.json").read_bytes())["failed"]
            failed = f"  failed: {', '.join(failures)}" if failures else ""
        print(f"{number} {candles.issue_time(number):%Y-%m-%d %H:%M}  {' '.join(flags)}{failed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    commands = {
        "release": cmd_release,
        "extend": cmd_extend,
        "issue": cmd_issue,
        "verify": cmd_verify,
        "scoreboard": cmd_scoreboard,
        "status": cmd_status,
        "alert": cmd_alert,
        "digest": cmd_digest,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
