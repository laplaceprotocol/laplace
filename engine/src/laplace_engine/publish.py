"""Publishing the record to the public bucket served at https://data.laplaceprotocol.com.

Only committed issues are published. Issue files never change once written, so they are cached for a year;
``latest.json`` points at the newest published issue and is cached for a minute. Local marker files make publishing
idempotent, and every run first catches up on committed issues whose upload failed.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from laplace_engine.release import canonical
from laplace_engine.verify import verify

PREFIX = "v1"
PUBLIC_URL = "https://data.laplaceprotocol.com/" + PREFIX
IMMUTABLE = "public, max-age=31536000, immutable"
FRESH = "public, max-age=60"
ISSUE_FILES = ("manifest.json", "forecasts.json", "commitment.json")
PUBLISHED = "published"

log = logging.getLogger(__name__)


class Store(Protocol):
    def put(self, key: str, body: bytes, *, content_type: str, cache_control: str) -> None: ...


class R2:
    """Cloudflare R2 through its S3 API."""

    def __init__(self, endpoint: str, access_key_id: str, secret_access_key: str, bucket: str) -> None:
        import boto3
        from botocore.config import Config

        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
            config=Config(retries={"max_attempts": 5, "mode": "standard"}),
        )
        self._bucket = bucket

    @classmethod
    def from_env(cls) -> R2:
        names = ("LAPLACE_R2_ENDPOINT", "LAPLACE_R2_ACCESS_KEY_ID", "LAPLACE_R2_SECRET_ACCESS_KEY", "LAPLACE_R2_BUCKET")
        missing = [n for n in names if not os.environ.get(n)]
        if missing:
            raise RuntimeError(f"missing environment variables: {', '.join(missing)}")
        return cls(*(os.environ[n] for n in names))

    def put(self, key: str, body: bytes, *, content_type: str, cache_control: str) -> None:
        self._s3.put_object(
            Bucket=self._bucket, Key=key, Body=body, ContentType=content_type, CacheControl=cache_control
        )


def check(issue_dir: Path) -> None:
    """Refuse files that do not rebuild exactly the committed root, manifest hash and size."""
    result = verify((issue_dir / "manifest.json").read_bytes(), (issue_dir / "forecasts.json").read_bytes())
    commitment = json.loads((issue_dir / "commitment.json").read_bytes())
    local = ("0x" + result.root.hex(), "0x" + result.manifest_hash.hex(), result.leaves)
    if (commitment["root"], commitment["manifest"], commitment["leaves"]) != local:
        raise ValueError(f"issue {issue_dir.name}: the files do not match the commitment")


def publish_issue(store: Store, issue_dir: Path, *, latest: bool) -> None:
    """Upload a committed issue's files after checking them against the commitment; with ``latest``, also point
    ``latest.json`` at it."""
    check(issue_dir)
    issue = int(issue_dir.name)
    for name in ISSUE_FILES:
        key = f"{PREFIX}/issues/{issue}/{name}"
        store.put(key, (issue_dir / name).read_bytes(), content_type="application/json", cache_control=IMMUTABLE)
    if latest:
        commitment = json.loads((issue_dir / "commitment.json").read_bytes())
        pointer = {k: commitment[k] for k in ("issue", "root", "manifest", "tx", "block", "timestamp")}
        pointer["files"] = {name: f"{PUBLIC_URL}/issues/{issue}/{name}" for name in ISSUE_FILES}
        store.put(f"{PREFIX}/latest.json", canonical(pointer), content_type="application/json", cache_control=FRESH)
    (issue_dir / PUBLISHED).touch()


def pending(issues_dir: Path) -> list[Path]:
    """Committed issues not yet published, oldest first, however old."""
    dirs = sorted((d for d in issues_dir.iterdir() if d.name.isdigit()), key=lambda d: int(d.name))
    return [d for d in dirs if (d / "commitment.json").exists() and not (d / PUBLISHED).exists()]


def publish_pending(store: Store, issues_dir: Path) -> list[int]:
    """Publish every pending issue. ``latest.json`` only ever points at the newest committed issue, so catching up on
    an older one never moves it back."""
    if not issues_dir.exists():
        return []
    committed = [int(d.name) for d in issues_dir.iterdir() if d.name.isdigit() and (d / "commitment.json").exists()]
    newest = max(committed, default=None)
    done = []
    for issue_dir in pending(issues_dir):
        try:
            publish_issue(store, issue_dir, latest=int(issue_dir.name) == newest)
        except Exception:
            log.exception("issue %s: not published; will retry", issue_dir.name)
            continue
        log.info("published issue %s", issue_dir.name)
        done.append(int(issue_dir.name))
    return done


def publish_release(store: Store, release_dir: Path, name: str, marker: Path) -> None:
    """Upload a release's file, public programs and parameter files once."""
    if marker.exists():
        return
    programs = release_dir / "programs"
    files: Iterable[Path] = [
        release_dir / "release.json",
        *sorted(programs.glob("*.py")),
        *sorted(programs.glob("*.params.json")),
    ]
    for path in files:
        key = f"{PREFIX}/releases/{name}/{path.relative_to(release_dir).as_posix()}"
        content_type = "application/json" if path.suffix == ".json" else "text/x-python; charset=utf-8"
        store.put(key, path.read_bytes(), content_type=content_type, cache_control=IMMUTABLE)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def publish_settlements(store: Store, issues_dir: Path) -> int:
    """Upload every settlement file not uploaded yet; returns how many were."""
    count = 0
    for path in sorted(issues_dir.glob("*/settlement-*h.json")):
        marker = path.with_name(path.name + "." + PUBLISHED)
        if marker.exists():
            continue
        key = f"{PREFIX}/issues/{path.parent.name}/{path.name}"
        store.put(key, path.read_bytes(), content_type="application/json", cache_control=IMMUTABLE)
        marker.touch()
        count += 1
    return count


def publish_scoreboard(store: Store, board: dict[str, Any]) -> None:
    store.put(f"{PREFIX}/scoreboard.json", canonical(board), content_type="application/json", cache_control=FRESH)
