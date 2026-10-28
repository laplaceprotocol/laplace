from datetime import UTC, datetime

import numpy as np
import pytest

from laplace_engine import looks
from laplace_engine.settle import Settlement

ASSETS = ["BTC", "ETH"]


def test_looks_match_the_tables_of_both_rules() -> None:
    r1, r2 = looks.rule_named("r1"), looks.rule_named("r2")
    assert looks.issues_of(r1, 90)[-1] == 500179
    assert looks.computed_after(r1, 90) == datetime(2027, 1, 23, tzinfo=UTC)
    assert looks.issues_of(r1, 30)[-1] == 498739
    assert looks.computed_after(r1, 30) == datetime(2026, 11, 24, tzinfo=UTC)
    assert looks.issues_of(r2, 365)[-1] == 506854
    assert looks.computed_after(r2, 365) == datetime(2027, 10, 29, tzinfo=UTC)
    assert looks.computed_after(r2, 30) == datetime(2026, 11, 28, tzinfo=UTC)
    with pytest.raises(KeyError):
        looks.rule_named("r9")


def test_a_look_counts_left_out_issues_and_claims_only_with_every_opponent() -> None:
    rng = np.random.default_rng(3)
    ids = {"demon": "0x" + "d" * 64, "garch": "0x" + "6" * 64, "har": "0x" + "7" * 64}
    rule = {
        "rule": "docs/judging/r1.md",
        "start_issue": 10_000,
        "judged": ("demon", ids["demon"]),
        "opponents": (("garch", ids["garch"]), ("har", ids["har"])),
    }
    settlements = []
    for issue in range(9_990, 10_000 + 24 * 90 + 10):  # some issues before and after the window
        for h in (1, 4, 24):
            base = {a: float(rng.normal(-2.0, 0.4)) for a in ASSETS}
            scores = {
                "demon": {a: base[a] + 0.05 + float(rng.normal(0, 0.05)) for a in ASSETS},
                "garch": base,
                "har": {a: base[a] + 0.05 + float(rng.normal(0, 0.05)) for a in ASSETS},  # as good as the Demon
            }
            if issue == 10_100 and h == 4:
                scores["demon"].pop("ETH")  # a missing forecast: the issue is left out
            if issue == 10_200:
                continue  # a gap: nothing settled for this issue
            settlements.append(Settlement(issue, h, "test", {}, {}, dict.fromkeys(ASSETS, 5), scores, "r1", ids))
    result = looks.look(settlements, rule, 90)
    garch, har = result["vs"]["garch"], result["vs"]["har"]
    assert garch["issues_used"] == 24 * 90 - 2
    assert garch["issues_left_out"] == 2
    assert garch["gain"] == pytest.approx(0.05, abs=0.005)
    assert garch["z"]["72"] > 4.36
    assert garch["z"]["168"] > 4.36
    assert garch["verdict"] == "beats"
    assert har["verdict"] == "no claim at this look"
    assert result["claims"] == {"the Demon beats the classical baselines": "not made at this look"}
    assert looks.look(settlements, rule, 30)["vs"]["garch"]["verdict"] == "direction only"
    text = looks.render(result, preview=True)
    assert "a preview, not the look" in text
    assert "| garch | +0.0" in text
    assert "| beats |" in text
