"""EV-GBM: EV-HAR corrected by gradient-boosted trees, frozen as arrays that a program walks with numpy.

LightGBM fits one multiclass model per horizon over the ten bins; its loss is exactly the binned log score. Boosting
starts from EV-HAR's log-probabilities (``init_score``), so the trees only learn where EV-HAR is wrong:

    logits = log EV-HAR(bins) + per bin, the sum of its trees' leaves

The inputs are EV-NET's (``learned.NAMES``) plus the asset's index and the raw hour and weekday. Each model keeps the
round that scores best on a later validation window, and no trees at all if none beats EV-HAR there. The trees are
exported to flat arrays (``Trees``), and every forecast, research included, walks those arrays (``walk``): research
forecasts are what the frozen program computes. LightGBM is only needed to fit (the ``gbm`` extra).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from laplace_research.data.events import Calendar
from laplace_research.evolve import learned
from laplace_research.evolve.baselines import Baseline
from laplace_research.evolve.feed import HORIZONS, N_BINS, FloatArray, IntArray

HOUR = 3600
DAY = 86_400
NAMES = (*learned.NAMES, "asset", "hour", "weekday")
SETTINGS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": N_BINS,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "seed": 0,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
MAX_ROUNDS = 1500
PATIENCE = 100  # rounds without a better validation score before boosting stops
FIELDS = ("root", "feature", "threshold", "left", "right", "value")


def design(
    pw: learned.Panelwise, calendar: Calendar, ts: IntArray, times: IntArray, sd: FloatArray, horizon: int
) -> FloatArray:
    """``[asset, sample, input]``: EV-NET's inputs, then the asset's index, the UTC hour and the weekday (Monday 0) of
    the forecast time; ``sd`` is EV-HAR's standard deviation ``[asset, sample]``."""
    x = learned.inputs(pw, calendar, ts, times, sd, horizon)
    assets, n, _ = x.shape
    at = ts[times] + HOUR
    columns = [np.arange(assets)[:, None], ((at // HOUR) % 24)[None, :], (((at // DAY) + 3) % 7)[None, :]]
    extra = np.stack([np.broadcast_to(c, (assets, n)) for c in columns], axis=-1).astype(np.float64)
    out: FloatArray = np.concatenate([x, extra], axis=-1)
    return out


@dataclass(frozen=True)
class Trees:
    """Every tree of one horizon's model, flattened.

    Internal node ``k`` sends an input row to ``left[k]`` when ``row[feature[k]] <= threshold[k]``, else to
    ``right[k]``; a child ``c >= 0`` is a node and ``c < 0`` is leaf ``~c``, worth ``value[~c]``. Tree ``t`` starts
    at ``root[t]`` and adds its leaf to bin ``t % 10`` (LightGBM orders a multiclass model round by round).
    """

    root: IntArray
    feature: IntArray
    threshold: FloatArray
    left: IntArray
    right: IntArray
    value: FloatArray

    def arrays(self, prefix: str) -> dict[str, np.ndarray[Any, Any]]:
        return {f"{prefix}.{name}": getattr(self, name) for name in FIELDS}

    @classmethod
    def of(cls, arrays: Any, prefix: str) -> Trees:
        return cls(*(np.asarray(arrays[f"{prefix}.{name}"]) for name in FIELDS))


def walk(trees: Trees, x: FloatArray) -> FloatArray:
    """Raw scores ``[sample, bin]`` for rows ``x`` ``[sample, input]``: per bin, the sum of its trees' leaves."""
    n = x.shape[0]
    node = np.repeat(trees.root[:, None], n, axis=1)  # [tree, sample]
    rows = np.arange(n)[None, :]
    inner = node >= 0
    while inner.any():
        k = np.where(inner, node, 0)
        step = np.where(x[rows, trees.feature[k]] <= trees.threshold[k], trees.left[k], trees.right[k])
        node = np.where(inner, step, node)
        inner = node >= 0
    out: FloatArray = trees.value[~node].reshape(-1, N_BINS, n).sum(axis=0).T
    return out


def export(booster: Any, rounds: int) -> Trees:
    """The first ``rounds`` rounds of a LightGBM multiclass booster as flat arrays (numerical splits only)."""
    root: list[int] = []
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []

    def visit(node: dict[str, Any]) -> int:
        if "leaf_value" in node:
            value.append(float(node["leaf_value"]))
            return ~(len(value) - 1)
        # Inputs are always finite, so a NaN default never applies; zero-as-missing or categorical splits would.
        if node["decision_type"] != "<=" or node["missing_type"] not in ("None", "NaN"):
            raise ValueError(f"unsupported split {node['decision_type']} / missing {node['missing_type']}")
        k = len(feature)
        feature.append(int(node["split_feature"]))
        threshold.append(float(node["threshold"]))
        left.append(0)
        right.append(0)
        left[k] = visit(node["left_child"])
        right[k] = visit(node["right_child"])
        return k

    if rounds > 0:
        for tree in booster.dump_model(num_iteration=rounds)["tree_info"]:
            root.append(visit(tree["tree_structure"]))
    return Trees(
        np.array(root, dtype=np.int32),
        np.array(feature, dtype=np.int32),
        np.array(threshold, dtype=np.float64),
        np.array(left, dtype=np.int32),
        np.array(right, dtype=np.int32),
        np.array(value, dtype=np.float64),
    )


@dataclass
class Gbm:
    """EV-HAR corrected by one boosted model per horizon: ``probs(times)`` gives ``[asset, horizon, sample, bin]`` raw
    probabilities (finalize them once before scoring), exactly what the frozen program returns."""

    name: str
    ev: Baseline
    panelwise: learned.Panelwise
    calendar: Calendar
    ts: IntArray
    trees: dict[int, Trees]
    params: dict[str, Any] = field(default_factory=dict)

    def probs(self, times: IntArray) -> FloatArray:
        base = learned.final(self.ev.probs(times))
        sd = self.ev.sd(times)
        out = np.empty_like(base)
        for j, h in enumerate(HORIZONS):
            x = design(self.panelwise, self.calendar, self.ts, times, sd[:, j], h)
            raw = walk(self.trees[h], x.reshape(-1, len(NAMES))).reshape(x.shape[0], x.shape[1], N_BINS)
            out[:, j] = learned.softmax(np.log(base[:, j]) + raw)
        return out

    def arrays(self) -> dict[str, np.ndarray[Any, Any]]:
        """Every horizon's trees, for a parameter file (``evolve.params``)."""
        out: dict[str, np.ndarray[Any, Any]] = {}
        for h in HORIZONS:
            out |= self.trees[h].arrays(f"h{h}")
        return out


def _log_loss(raw: FloatArray, base: FloatArray, y: IntArray) -> float:
    p = learned.softmax(np.log(base) + raw)
    return -float(np.mean(np.log(p[np.arange(len(y)), y])))


def ev_gbm(
    ev: Baseline,
    panelwise: learned.Panelwise,
    calendar: Calendar,
    ts: IntArray,
    fit_hours: IntArray,
    valid_hours: IntArray,
    realized: Callable[[IntArray], IntArray],
) -> Gbm:
    """Boost EV-HAR's correction on every hour of ``fit_hours``, keeping the round best on ``valid_hours``."""
    import lightgbm as lgb  # the gbm extra: only fitting needs it

    rows = {}
    for name, hours in (("fit", fit_hours), ("valid", valid_hours)):
        rows[name] = (hours, learned.final(ev.probs(hours)), ev.sd(hours), realized(hours))
    trees: dict[int, Trees] = {}
    rounds: dict[str, int] = {}
    for j, h in enumerate(HORIZONS):
        data = {}
        for name, (hours, base, sd, y) in rows.items():
            x = design(panelwise, calendar, ts, hours, sd[:, j], h).reshape(-1, len(NAMES))
            data[name] = (x, base[:, j].reshape(-1, N_BINS), y[:, j].reshape(-1))
        (x, base, y), (xv, basev, yv) = data["fit"], data["valid"]
        train = lgb.Dataset(x, label=y, init_score=np.log(base), feature_name=list(NAMES), free_raw_data=False)
        valid = lgb.Dataset(xv, label=yv, init_score=np.log(basev), reference=train)
        booster = lgb.train(
            SETTINGS,
            train,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[valid],
            callbacks=[lgb.early_stopping(PATIENCE, verbose=False)],
        )
        best = export(booster, booster.best_iteration)
        # Trees that do not beat EV-HAR on validation are not kept: the model is then EV-HAR itself.
        if _log_loss(walk(best, xv), basev, yv) >= _log_loss(np.zeros_like(basev), basev, yv):
            best, rounds[str(h)] = export(booster, 0), 0
        else:
            rounds[str(h)] = booster.best_iteration
        trees[h] = best
    params = ev.params | {
        "gbm": {"settings": SETTINGS, "max_rounds": MAX_ROUNDS, "patience": PATIENCE, "rounds": rounds},
        "inputs": list(NAMES),
    }
    return Gbm("ev-gbm", ev, panelwise, calendar, ts, trees, params)
