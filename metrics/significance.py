"""Confidence intervals and the *right* significance test for this comparison.

Two decisions here, both of which matter more than they look:

**Wilson, not Wald.** The Wald interval ``p ± 1.96·sqrt(p(1-p)/n)`` misbehaves near 0 and 1 —
it can extend past 1.0 and its coverage degrades exactly where accuracy figures live. Wilson
is the standard fix and is what ``statsmodels.stats.proportion.proportion_confint(method=
"wilson")`` returns; ``tests/test_metrics.py`` asserts agreement with it.

**McNemar, not two independent proportion tests.** Both models are evaluated on the *same*
1,000 examples, so the predictions are paired. A two-sample proportion test assumes
independent samples, ignores the pairing, and therefore throws away the information that
matters: how often the two models disagree. McNemar's exact test uses only the discordant
pairs, which is the whole question — "when they differ, who is right more often?"

With n = 1,000 and accuracy near 0.93 the Wilson interval is roughly ±1.6 percentage points,
so a two-point gap is not resolvable. Reporting a bare point estimate at that sample size is
misleading, which is why nothing in this repo publishes one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A binomial proportion with its two-sided interval."""

    point: float
    low: float
    high: float
    level: float
    method: str
    n: int
    successes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def pp_halfwidth(self) -> float:
        """Half-width in percentage points — the number the README quotes as ``±x.x pp``."""
        return 100.0 * (self.high - self.low) / 2.0

    def format(self, digits: int = 4) -> str:
        return f"{self.point:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def wilson_interval(successes: int, n: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Implemented directly rather than delegating, so the repo does not depend on statsmodels
    at *runtime* for its headline number; the test asserts it matches statsmodels exactly.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} outside [0, {n}]")
    p = successes / n
    z = _z_for(level)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(
        point=p,
        low=max(0.0, centre - half),
        high=min(1.0, centre + half),
        level=level,
        method="wilson",
        n=n,
        successes=successes,
    )


def _z_for(level: float) -> float:
    """Two-sided normal quantile. Uses scipy when available, else a fixed table."""
    try:
        from scipy.stats import norm

        return float(norm.ppf(0.5 + level / 2.0))
    except ImportError:  # pragma: no cover - scipy is a sklearn dependency
        table = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
        if level not in table:
            raise ValueError(f"no tabulated z for level={level} and scipy is unavailable") from None
        return table[level]


def accuracy_interval(
    y_true: "np.ndarray | list[int]", y_pred: "np.ndarray | list[int]", level: float = 0.95
) -> Interval:
    """Wilson interval around an accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return wilson_interval(int((y_true == y_pred).sum()), int(y_true.size), level)


@dataclass(frozen=True)
class McNemarResult:
    """The 2×2 discordance table and the exact test on it.

    ``b`` = model A right, model B wrong. ``c`` = model A wrong, model B right. The exact test
    is a two-sided binomial test of ``b`` out of ``b + c`` against p = 0.5, so ``n_discordant``
    is the effective sample size — not the 1,000 test examples.
    """

    a_both_correct: int
    b_only_a_correct: int
    c_only_b_correct: int
    d_both_wrong: int
    statistic: float
    p_value: float
    exact: bool
    n_discordant: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def table(self) -> list[list[int]]:
        return [[self.a_both_correct, self.b_only_a_correct], [self.c_only_b_correct, self.d_both_wrong]]


def mcnemar_test(
    y_true: "np.ndarray | list[int]",
    pred_a: "np.ndarray | list[int]",
    pred_b: "np.ndarray | list[int]",
    exact: bool = True,
) -> McNemarResult:
    """Exact McNemar test on two models' predictions over the *same* examples."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if not (y_true.shape == pred_a.shape == pred_b.shape):
        raise ValueError("y_true, pred_a and pred_b must have identical shapes")

    ok_a = pred_a == y_true
    ok_b = pred_b == y_true
    a = int((ok_a & ok_b).sum())
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    d = int((~ok_a & ~ok_b).sum())

    from statsmodels.stats.contingency_tables import mcnemar

    result = mcnemar([[a, b], [c, d]], exact=exact, correction=not exact)
    return McNemarResult(
        a_both_correct=a,
        b_only_a_correct=b,
        c_only_b_correct=c,
        d_both_wrong=d,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        exact=exact,
        n_discordant=b + c,
    )


def significance_sentence(
    name_a: str,
    name_b: str,
    acc_a: Interval,
    acc_b: Interval,
    mc: McNemarResult,
    alpha: float = 0.05,
) -> str:
    """The plain-English sentence the README is required to print beneath the table.

    Generated rather than written, so it can never drift from the numbers above it.
    """
    gap_pp = 100.0 * (acc_a.point - acc_b.point)
    if gap_pp >= 0:
        leader, trailer, hi, lo = name_a, name_b, acc_a, acc_b
    else:
        leader, trailer, hi, lo = name_b, name_a, acc_b, acc_a
    verdict = "is" if mc.p_value < alpha else "is not"
    return (
        f"{leader} leads {trailer} by {abs(gap_pp):.1f} percentage points "
        f"({hi.point:.4f} vs {lo.point:.4f}) on {acc_a.n} test examples. "
        f"They disagree on {mc.n_discordant} of them; exact McNemar gives p = {mc.p_value:.4g}, "
        f"so at alpha = {alpha} the gap {verdict} distinguishable from zero."
    )
