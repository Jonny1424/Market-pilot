"""Market Regime Bot: trending / ranging / volatile, from causal indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd

TREND_ER = 0.35
VOLATILE_RANK = 0.90


def classify_row(row) -> str:
    rank, er = row.get("atr_pct_rank"), row.get("er20")
    if rank is not None and np.isfinite(rank) and rank >= VOLATILE_RANK:
        return "volatile"
    if er is not None and np.isfinite(er) and er >= TREND_ER:
        return "trending"
    return "ranging"


def classify_series(ind: pd.DataFrame) -> pd.Series:
    out = np.where(ind["atr_pct_rank"].fillna(0) >= VOLATILE_RANK, "volatile",
                   np.where(ind["er20"].fillna(0) >= TREND_ER, "trending", "ranging"))
    return pd.Series(out, index=ind.index, name="regime")
