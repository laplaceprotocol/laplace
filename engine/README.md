# engine

The live side of Laplace. Every hour it runs the current release's forecasters on the candles that have just
closed, rounds each forecast to whole basis points, and commits the Merkle root of the hour's forecasts to
`LaplaceLedger` on Robinhood Chain before any of them can settle. The forecasts and a manifest are then published, so
anyone can rebuild the root and check it against the chain. The design is in
[`docs/design/live-record.md`](../docs/design/live-record.md).

```bash
uv sync
uv run pytest

# freeze a research run into a release (public part here, private champions elsewhere)
uv run laplace-engine release --run ../research/runs/evolve-poc2 --name r1 --out releases/r1 --private PRIVATE_DIR
# extend a release with the monkey and pools fitted on a run's validation window (r1 → r1.1)
uv run laplace-engine extend --from releases/r1 --private PRIVATE_DIR --run ../research/runs/evolve-poc2 \
  --name r1.1 --out releases/r1.1
# seal the current hour's issue into DATA_DIR/issues/<issue>/ (or a past one with --issue)
uv run laplace-engine issue --release releases/r1 --private PRIVATE_DIR --data DATA_DIR
# check a published issue: rebuild it from its files, then check its commitment on-chain
uv run laplace-engine verify --issue ISSUE
```

## Operations
The hourly run is a systemd user timer (`deploy/laplace-issue.timer`, 7 minutes after every hour) that runs
`laplace-engine issue --commit --publish` from a dedicated worktree, so development never changes what runs.
`deploy/deploy.sh [ref]` checks out a ref there (default `origin/main`), syncs the locked dependencies and installs the
units. The service reads its secrets from an environment file outside the repository:

| variable | |
|---|---|
| `LAPLACE_COMMIT_KEY_FILE` | JSON file with the commit key (`privateKey`); the key only pays gas for `commit` |
| `LAPLACE_R2_ENDPOINT`, `LAPLACE_R2_BUCKET` | the public bucket's S3 endpoint and name |
| `LAPLACE_R2_ACCESS_KEY_ID`, `LAPLACE_R2_SECRET_ACCESS_KEY` | write access to that bucket |
| `LAPLACE_TG_TOKEN`, `LAPLACE_TG_CHAT` | Telegram alerts and the daily digest (optional: without them, problems are only logged) |

```bash
systemctl --user list-timers laplace-issue.timer    # next run
journalctl --user -u laplace-issue -n 50            # last runs
uv run laplace-engine status --data DATA_DIR        # sealed, committed, published, per hour
uv run laplace-engine scoreboard --data DATA_DIR    # mean log scores and gains over the baselines
uv run laplace-engine digest --data DATA_DIR --dry-run   # the daily digest, printed instead of sent
```

Each run holds a lock, seals the issue (never resealing one that exists on disk), commits it while the window is
open (journaled, retried, and read back from the chain), reconciles older issues (recovering commitments that were
never recorded, marking real gaps), publishes every committed issue that is not yet public after checking it against
its commitment, then settles every question that has become due and publishes the settlements and the scoreboard.
Every step is isolated and retried on the next run; `status` shows gaps and conflicts.

**Monitoring.** A run sends the operator one Telegram message listing whatever needed attention: an issue not sealed
or not committed, a recovered commitment, a gap, a conflict, a forecaster that produced nothing, a question that could
not be settled, a failed upload, a low key balance (repeats are throttled). If the run itself crashes, times out or is
killed, `laplace-alert.service` (systemd `OnFailure=`) reports it; runs that already reported their problems exit with
status 3 and are not reported twice. `laplace-digest.timer` sends a daily digest at 08:15 UTC: the last 24 hours, the
scores so far, the key's runway. Independently of this machine, the Cloudflare Worker in
[`../ops/watch`](../ops/watch) checks the public record every hour and alerts if the current hour is missing.

| module | |
|---|---|
| `record.py` | leaves and the Merkle tree, mirrored by `LaplaceLedger.leafHash` and `verifyForecast` |
| `quantize.py` | rounding to basis points; exact integer pooling of rounded forecasts |
| `release.py`, `freeze.py`, `fit.py` | releases: forecasters identified by their source or pool specification; freezing a research run; fitting new pools |
| `baselines.py` | the baseline ladder as programs with frozen parameters |
| `candles.py`, `issue.py` | an issue's candles, its forecasts, root and manifest |
| `verify.py` | the reference verifier |
| `chain.py`, `publish.py` | committing to the ledger; publishing to `data.laplaceprotocol.com` |
| `settle.py` | settling questions from the committed record and the scoreboard |
| `vectors.py` | test vectors shared with the contracts (`python -m laplace_engine.vectors`) |

The research package (`../research`) supplies the feed definition, the sandboxed evaluator and the baselines; this
package adds only what the live record needs.
