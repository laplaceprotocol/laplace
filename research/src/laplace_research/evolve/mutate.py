"""Mutation: ask a language model for an improved child of a parent program."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class ModelRunner(Protocol):
    async def run(
        self, model: str, payload: dict[str, Any], metadata: Mapping[str, str | int | float | bool] | None = None
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Mutator:
    name: str
    model: str
    options: dict[str, Any]
    usd_per_m_input: float
    usd_per_m_output: float

    def cost(self, usage: dict[str, Any]) -> float:
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)
        return (tokens_in * self.usd_per_m_input + tokens_out * self.usd_per_m_output) / 1e6


MUTATORS = {
    m.name: m
    for m in (
        Mutator(
            "glm-5.3-flash",
            "@cf/zai-org/glm-5.3-flash",
            {"reasoning_effort": "low", "max_completion_tokens": 16000},
            0.15,
            0.5,
        ),
        Mutator(
            "deepseek-v4.1-flash",
            "@cf/deepseek-ai/deepseek-v4.1-flash-1017",
            {"reasoning_effort": "low", "max_completion_tokens": 16000},
            0.30,
            1.20,
        ),
        Mutator(
            "glm-5.3", "@cf/zai-org/glm-5.3", {"reasoning_effort": "high", "max_completion_tokens": 20000}, 1.4, 4.4
        ),
    )
}

SYSTEM = (
    "You are a quantitative researcher inside an evolutionary search over forecasting programs. Each program "
    "forecasts the probability distribution of a crypto asset's log return over the next 1, 4 or 24 hours, as "
    "probabilities for ten fixed bins. Programs are scored on years of hourly data by the mean log probability they "
    "give the bin that actually happened, a strictly proper score: honest, calibrated probabilities maximize it. "
    "Direction is close to unpredictable at these horizons; most of the achievable gain is in the scale and shape of "
    "the distribution."
)

INTERFACE = """\
def forecast(ctx) -> a sequence of 10 non-negative numbers, one per bin (normalized for you)

ctx.asset       "BTC" | "ETH" | "SOL" | "XRP" | "DOGE" | "AVAX" | "LINK"
ctx.horizon     1, 4 or 24 (hours)
ctx.edges       numpy array of the 9 inner bin edges in log-return units, ascending; bins are
                (-inf, e0), [e0, e1), ..., [e8, +inf). Edges: 1h +-0.3/0.7/1.5/3%, 4h +-0.6/1.4/3/6%,
                24h +-1.5/3.5/7/14%, and 0.
ctx.hour        UTC hour (0-23) at forecast time; ctx.weekday 0 = Monday
ctx.open, ctx.high, ctx.low, ctx.close, ctx.volume
                read-only numpy float arrays: this asset's last 720 hourly candles, oldest first;
                index -1 is the candle that just closed
ctx.peers       dict asset -> object with .open .high .low .close .volume for every asset (same aligned window)
ctx.senses      dict of read-only arrays aligned with ctx.close (NaN where unknown): "rv" (each hour's realized
                variance from 5-minute returns), "dvol" (30-day implied volatility index, annualized %, BTC and ETH
                only), "funding" (perpetual funding rate), "oi" (perpetual open interest in coins, at each hour's start)
ctx.peer_senses dict asset -> the same senses for every asset
ctx.events      tuple of (hours from now, kind) for scheduled events from 24 hours before to 48 hours after the
                forecast, oldest first; kinds "fomc", "cpi", "nfp", "us-open", "us-close"; an event with
                0 <= hours < ctx.horizon falls inside the forecast's window
ctx.lib.normal_probs(sd, edges, loc=0.0)
ctx.lib.laplace_probs(sd, edges, loc=0.0)
ctx.lib.student_t_probs(sd, nu, edges, loc=0.0)
                bin probabilities for a distribution whose standard deviation is sd (of the horizon's log return)
ctx.lib.mixture(list_of_bin_probabilities, weights)
ctx.lib.log_returns(close) -> hourly log returns (one element shorter)"""

RULES = """\
- Only `import math` and `import numpy as np`. No I/O, randomness, classes, global state or private attributes.
- Fast: the program is called once per asset, horizon and sampled hour, 10,000 to 50,000 times per evaluation,
  within a 90-second budget (aim for under 1 ms a call).
- Deterministic and robust: never return NaN or inf; handle flat or zero-volume hours.
- No constants tuned to one asset or period unless you can justify them: the program is later judged on data it
  never saw, and overfitting is the most common way children fail."""


@dataclass(frozen=True)
class Scored:
    code: str
    gain: float
    breakdown: dict[str, float]
    reference: str = "the naive baseline"

    def describe(self) -> str:
        horizons = ", ".join(f"{k} {v:+.4f}" for k, v in self.breakdown.items() if not k.startswith("seg:"))
        text = f"{self.gain:+.4f} nats per question vs {self.reference} ({horizons})"
        segments = [(k[4:], v) for k, v in self.breakdown.items() if k.startswith("seg:")]
        if segments:
            text += "; where it gains and loses: " + ", ".join(f"{k} {v:+.4f}" for k, v in segments)
        return text


def build_prompt(directive: str, parent: Scored, inspirations: list[Scored], failures: list[str]) -> str:
    sections = [
        "## Interface\n```\n" + INTERFACE + "\n```",
        "## Rules\n" + RULES,
        "## Island hypothesis\n" + directive,
        f"## Parent program\nScore: {parent.describe()}\n```python\n{parent.code.strip()}\n```",
    ]
    for n, other in enumerate(inspirations, 1):
        sections.append(
            f"## Inspiration {n} (from the population)\nScore: {other.describe()}\n```python\n{other.code.strip()}\n```"
        )
    if failures:
        sections.append(
            "## Recent failed children on this island (do not repeat)\n" + "\n".join(f"- {f}" for f in failures)
        )
    sections.append(
        "## Task\nPropose one focused improvement to the parent, guided by the island hypothesis. Reply with:\n"
        "TITLE: <at most 8 words>\nWHY: <one or two sentences>\nthen the complete new program in a single "
        "```python block."
    )
    return "\n\n".join(sections)


@dataclass(frozen=True)
class Proposal:
    code: str
    title: str
    why: str
    cost_usd: float
    usage: dict[str, Any]


class NoProgram(ValueError):
    """The model answered but gave no usable program. The call was still billed, so the cost travels with the error."""

    def __init__(self, message: str, cost_usd: float = 0.0) -> None:
        super().__init__(message)
        self.cost_usd = cost_usd


def parse_reply(content: str) -> tuple[str, str, str]:
    blocks = re.findall(r"```python\s*\n(.*?)```", content, re.DOTALL)
    if not blocks:
        raise NoProgram("reply has no python block")
    title = re.search(r"^TITLE:\s*(.+)$", content, re.MULTILINE)
    why = re.search(r"^WHY:\s*(.+)$", content, re.MULTILINE)
    return (
        blocks[-1].strip() + "\n",
        (title.group(1).strip() if title else "")[:80],
        (why.group(1).strip() if why else "")[:400],
    )


async def propose(
    ai: ModelRunner, mutator: Mutator, prompt: str, metadata: Mapping[str, str | int | float | bool] | None = None
) -> Proposal:
    """Ask ``mutator`` for a child program. ``metadata`` tags the request in the gateway's logs."""
    result = await ai.run(
        mutator.model,
        {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], **mutator.options},
        metadata,
    )
    usage = result.get("usage", {})
    cost = mutator.cost(usage)
    choice = result["choices"][0]
    content = choice["message"].get("content") or ""
    if not content and choice.get("finish_reason") == "length":
        raise NoProgram("token budget spent on reasoning", cost)
    try:
        code, title, why = parse_reply(content)
    except NoProgram as exc:
        raise NoProgram(str(exc), cost) from None
    return Proposal(code, title, why, cost, usage)
