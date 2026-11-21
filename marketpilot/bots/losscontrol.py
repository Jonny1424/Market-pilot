"""Loss Control bot — keep losses small rather than chase a high win rate.

Win rate is a poor target: a strategy can win 80% of the time and still lose
money. MarketPilot's goal is positive expectancy with shallow drawdowns. This
bot sets how much each new trade may risk, cutting size when the odds of a big
or clustered loss are higher:

1. Volatility scaling — when a market's daily range is in the top 20% of its
   own past year, risk is halved (top 5%: quartered). Research (Moreira & Muir
   2017; Dreyer & Hubrich) finds this mainly reduces tail losses.
2. Drawdown brake — when the account is 3% below its peak, new trades risk
   half; at 6% below, a quarter. Size comes back as the account recovers.
3. Losing-streak brake — after 4 losses in a row, risk is halved until a win.
4. Concentration caps — at most 3 open positions per market group, and total
   open risk capped at 3% of the account, because related markets tend to lose
   together.

The multipliers combine (multiply) with a floor of 0.25x, so one rule can't
wipe out position size on its own interaction with the others.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {
    "enabled": True,
    "vol_high_rank": 0.80, "vol_high_mult": 0.5,
    "vol_extreme_rank": 0.95, "vol_extreme_mult": 0.25,
    "dd_soft_pct": 3.0, "dd_soft_mult": 0.5,
    "dd_hard_pct": 6.0, "dd_hard_mult": 0.25,
    "streak_losses": 4, "streak_mult": 0.5,
    "min_mult": 0.25,
    "max_positions_per_group": 3,
    "max_open_risk_pct": 3.0,
}


def config(settings_raw: dict) -> dict:
    return {**DEFAULTS, **(settings_raw.get("loss_control") or {})}


def vol_multiplier(vol_rank, cfg: dict) -> tuple[float, str | None]:
    try:
        v = float(vol_rank)
    except (TypeError, ValueError):
        return 1.0, None
    if not np.isfinite(v):
        return 1.0, None
    if v >= cfg["vol_extreme_rank"]:
        return cfg["vol_extreme_mult"], f"volatility in top {1 - cfg['vol_extreme_rank']:.0%} of the year"
    if v >= cfg["vol_high_rank"]:
        return cfg["vol_high_mult"], f"volatility in top {1 - cfg['vol_high_rank']:.0%} of the year"
    return 1.0, None


def account_multiplier(drawdown_pct: float, streak: int, cfg: dict) -> tuple[float, list[str]]:
    """Account-wide brakes (same for every market this cycle)."""
    mult, notes = 1.0, []
    if drawdown_pct >= cfg["dd_hard_pct"]:
        mult *= cfg["dd_hard_mult"]
        notes.append(f"account {drawdown_pct:.1f}% below its peak")
    elif drawdown_pct >= cfg["dd_soft_pct"]:
        mult *= cfg["dd_soft_mult"]
        notes.append(f"account {drawdown_pct:.1f}% below its peak")
    if streak >= cfg["streak_losses"]:
        mult *= cfg["streak_mult"]
        notes.append(f"{streak} losses in a row")
    return mult, notes


def multiplier(cfg: dict, *, vol_rank=None, drawdown_pct: float = 0.0, streak: int = 0) -> tuple[float, list[str]]:
    if not cfg.get("enabled", True):
        return 1.0, []
    m_acc, notes = account_multiplier(drawdown_pct, streak, cfg)
    m_vol, vnote = vol_multiplier(vol_rank, cfg)
    if vnote:
        notes.append(vnote)
    mult = m_acc * m_vol
    if mult < 1.0:
        mult = max(mult, cfg["min_mult"])
    return float(mult), notes


def drawdown_pct(equity_history: list[float], equity_now: float) -> float:
    vals = [float(x) for x in equity_history if x is not None] + [float(equity_now)]
    peak = max(vals) if vals else equity_now
    return max(0.0, (peak - equity_now) / peak * 100) if peak > 0 else 0.0


def loss_streak(closed_trades: list[dict]) -> int:
    rows = sorted((t for t in closed_trades if t.get("r_multiple") is not None and t.get("closed_at")),
                  key=lambda t: str(t["closed_at"]))
    n = 0
    for t in reversed(rows):
        if float(t["r_multiple"]) <= 0:
            n += 1
        else:
            break
    return n


def concentration_reasons(asset_class: str, open_trades: list[dict], new_risk_gbp: float,
                          equity: float, cfg: dict) -> list[str]:
    if not cfg.get("enabled", True):
        return []
    reasons = []
    same = [t for t in open_trades if t.get("asset_class") == asset_class]
    if len(same) >= int(cfg["max_positions_per_group"]):
        reasons.append(f"already {len(same)} open {asset_class} positions (max {cfg['max_positions_per_group']})")
    open_risk = sum(float(t.get("risk_gbp") or 0) for t in open_trades)
    cap = equity * float(cfg["max_open_risk_pct"]) / 100
    if open_risk + new_risk_gbp > cap + 1e-9:
        reasons.append(f"total open risk would be £{open_risk + new_risk_gbp:.2f} "
                       f"(cap {cfg['max_open_risk_pct']}% = £{cap:.2f})")
    return reasons


def compare(trades, cfg: dict) -> dict:
    """Backtest check: the same trades in R, flat size vs volatility-scaled size.
    `trades` need r_multiple and vol_rank. Account brakes are path-dependent, so the
    comparison covers the volatility rule (the part that can be tested trade by trade)."""
    r = np.array([float(t.r_multiple) for t in trades], dtype=float)
    if len(r) == 0:
        return {}
    w = np.array([vol_multiplier(getattr(t, "vol_rank", None), cfg)[0] for t in trades], dtype=float)

    def stats(x):
        eq = np.cumsum(x)
        dd = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq
        losses = x[x < 0]
        return {"total_r": float(x.sum()), "max_drawdown_r": float(dd.max()),
                "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
                "worst_r": float(x.min()), "return_per_risk": float(x.sum() / dd.max()) if dd.max() > 0 else float("inf")}

    flat, scaled = stats(r), stats(r * w)
    return {"flat": flat, "scaled": scaled, "share_reduced": float((w < 1).mean()),
            "dd_cut_pct": float(1 - scaled["max_drawdown_r"] / flat["max_drawdown_r"]) if flat["max_drawdown_r"] > 0 else 0.0}
