from pathlib import Path

import httpx
import pytest
from eth_abi.abi import encode
from eth_account import Account
from eth_account.signers.local import LocalAccount

from laplace_engine.chain import (
    COMMIT,
    ERRORS,
    LEDGER,
    LEDGER_BLOCK,
    RPC,
    Ledger,
    LedgerRevert,
    RPCError,
    selector,
)

from .conftest import ISSUE
from .fakes import FakeNode

ROOT, MANIFEST = b"\x11" * 32, b"\x22" * 32


def setup(head_timestamp: int = ISSUE * 3600 + 120) -> tuple[FakeNode, Ledger, LocalAccount]:
    """A ledger on a fake node whose latest block is two minutes into the issue's commit window."""
    account: LocalAccount = Account.create()
    node = FakeNode(account.address, head_timestamp=head_timestamp)
    rpc = RPC("https://rpc.test", client=httpx.Client(transport=node.transport()))
    return node, Ledger(rpc), account


def test_commit_signs_one_commit_transaction_to_the_ledger(tmp_path: Path) -> None:
    node, ledger, account = setup()
    commitment = ledger.commit(account, ISSUE, ROOT, MANIFEST, 273, journal=tmp_path / "tx.json")
    assert len(node.sent) == 1
    tx = node.sent[0]
    assert tx["from"] == account.address
    assert bytes(tx["to"]) == bytes.fromhex(LEDGER[2:])
    assert tx["chainId"] == 4663
    assert tx["value"] == 0
    assert tx["maxPriorityFeePerGas"] == 0
    assert bytes(tx["data"]) == COMMIT + encode(
        ["uint64", "bytes32", "bytes32", "uint32"], [ISSUE, ROOT, MANIFEST, 273]
    )
    assert tx["gas"] == 51_000 * 13 // 10
    assert commitment.root == "0x" + ROOT.hex()
    assert commitment.manifest == "0x" + MANIFEST.hex()
    assert commitment.leaves == 273
    assert commitment.in_window
    assert commitment.tx in (tmp_path / "tx.json").read_text()  # journaled before it was sent


def test_a_lost_reply_is_resolved_from_the_chain(tmp_path: Path) -> None:
    # The node accepts the transaction but the reply is lost; the client's resend reads "already known".
    node, ledger, account = setup()
    node.lost_replies = 1
    commitment = ledger.commit(account, ISSUE, ROOT, MANIFEST, 273, journal=tmp_path / "tx.json")
    assert len(node.sent) == 1
    assert commitment.tx in node.receipts


def test_a_rerun_recovers_the_commitment_instead_of_sending_again(tmp_path: Path) -> None:
    node, ledger, account = setup()
    first = ledger.commit(account, ISSUE, ROOT, MANIFEST, 273, journal=tmp_path / "tx.json")
    again = ledger.commit(account, ISSUE, ROOT, MANIFEST, 273, journal=tmp_path / "tx.json")
    assert again == first
    assert len(node.sent) == 1
    assert ledger.find(ISSUE) == first


def test_recovery_falls_back_to_the_journal_and_checks_the_manifest(tmp_path: Path) -> None:
    node, ledger, account = setup()
    first = ledger.commit(account, ISSUE, ROOT, MANIFEST, 273, journal=tmp_path / "tx.json")
    node.logs.clear()  # an RPC that cannot find the event
    assert ledger.recover(ISSUE, ROOT, MANIFEST, 273, journal=tmp_path / "tx.json") == first
    with pytest.raises(RPCError, match="not found"):
        ledger.recover(ISSUE, ROOT, MANIFEST, 273, journal=tmp_path / "missing.json")
    node.logs.clear()
    with pytest.raises(RPCError, match="another root, manifest or size"):
        ledger.recover(ISSUE, ROOT, b"\x44" * 32, 273, journal=tmp_path / "tx.json")


def test_refuses_to_commit_over_another_root_or_with_another_key(tmp_path: Path) -> None:
    node, ledger, account = setup()
    node.roots[ISSUE] = b"\x33" * 32
    with pytest.raises(RPCError, match="another root"):
        ledger.commit(account, ISSUE, ROOT, MANIFEST, 273, journal=tmp_path / "tx.json")
    node.roots.clear()
    node.committer = Account.create().address
    with pytest.raises(ValueError, match="committer"):
        ledger.commit(account, ISSUE, ROOT, MANIFEST, 273, journal=tmp_path / "tx.json")
    assert not node.sent


def test_contract_errors_are_named(tmp_path: Path) -> None:
    node, ledger, account = setup()
    window = selector("OutsideCommitWindow(uint64,uint256)")
    assert ERRORS[window] == "OutsideCommitWindow"
    node.revert = "0x" + (window + encode(["uint64", "uint256"], [ISSUE, ISSUE * 3600 + 900])).hex()
    with pytest.raises(LedgerRevert) as raised:
        ledger.commit(account, ISSUE, ROOT, MANIFEST, 273, journal=tmp_path / "tx.json")
    assert raised.value.reason == "OutsideCommitWindow"
    assert not node.sent


def test_rpc_retries_rate_limits_then_gives_up() -> None:
    node, ledger, _ = setup()
    node.rate_limited = 2
    assert ledger.root_of(ISSUE) == bytes(32)
    impatient = Ledger(RPC("https://rpc.test", client=httpx.Client(transport=node.transport()), attempts=2))
    node.rate_limited = 5
    with pytest.raises(RPCError, match="429"):
        impatient.root_of(ISSUE)


def test_find_searches_exactly_the_commit_window(tmp_path: Path) -> None:
    # The head sits on the window's first second, so the commit lands in a block stamped exactly at the hour.
    _, ledger, account = setup(head_timestamp=ISSUE * 3600)
    commitment = ledger.commit(account, ISSUE, ROOT, MANIFEST, 273, journal=tmp_path / "tx.json")
    assert ledger.find(ISSUE) == commitment
    assert ledger.find(ISSUE + 1) is None


def test_block_at_is_the_last_block_at_or_before_a_time() -> None:
    node, ledger, _ = setup()
    t = node.timestamps[LEDGER_BLOCK + 1_234]
    assert ledger.block_at(t) == LEDGER_BLOCK + 1_234
    assert ledger.block_at(10**12) == node.head


def test_rpc_moves_to_the_next_endpoint_when_one_throttles_or_fails() -> None:
    node, _, _ = setup()
    served = []

    def route(request: httpx.Request) -> httpx.Response:
        served.append(request.url.host)
        if request.url.host == "down.test":
            return httpx.Response(503)
        if request.url.host == "throttled.test":  # a throttled endpoint that says so in a JSON-RPC error
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "error": {"message": "method does not exist"}})
        return node.handle(request)

    client = httpx.Client(transport=httpx.MockTransport(route))
    ledger = Ledger(RPC(["https://down.test", "https://throttled.test", "https://up.test"], client=client))
    assert ledger.root_of(ISSUE) == bytes(32)
    assert served == ["down.test", "throttled.test", "up.test"]


def test_a_revert_is_the_chain_s_answer_and_is_not_retried_elsewhere(tmp_path: Path) -> None:
    node, ledger, account = setup()
    node.revert = "0x" + selector("NotCommitter()").hex()
    before = node.requests
    with pytest.raises(LedgerRevert):
        ledger.commit(account, ISSUE, ROOT, MANIFEST, 273, journal=tmp_path / "tx.json")
    assert node.requests - before == 3  # rootOf, committer, then estimateGas once: nothing is resent
