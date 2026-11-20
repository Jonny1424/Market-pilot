"""Performance metrics in R (multiples of initial planned risk)."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(r_multiples: Iterable[float]) -> dict:
    r = np.asarray(list(r_multiples), dtype=float)
    n = len(r)
    if n == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "win_rate_lo": 0.0,
                "win_rate_hi": 1.0, "avg_win_r": 0.0, "avg_loss_r": 0.0, "win_loss_ratio": 0.0,
                "expectancy_r": 0.0, "profit_factor": 0.0, "total_r": 0.0, "max_drawdown_r": 0.0,
                "t_stat": 0.0}
    wins, losses = r[r > 0], r[r <= 0]
    lo, hi = wilson_interval(len(wins), n)
    gross_win, gross_loss = wins.sum(), -losses.sum()
    equity = np.cumsum(r)
    dd = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:] - equity
    return {
        "trades": n,
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": len(wins) / n,
        "win_rate_lo": lo,
        "win_rate_hi": hi,
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "win_loss_ratio": len(wins) / max(len(losses), 1),
        "expectancy_r": float(r.mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "total_r": float(r.sum()),
        "max_drawdown_r": float(dd.max()) if n else 0.0,
        # t-statistic of mean R: how many standard errors the average is above zero
        "t_stat": float(r.mean() / (r.std(ddof=1) / np.sqrt(n))) if n > 1 and r.std(ddof=1) > 0 else 0.0,
    }
