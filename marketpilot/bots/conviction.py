"""Conviction grade — confluence first, Time Machine as the final confirmation.

The Time Machine only gets a vote on high-confluence setups (6, 7 or 8 of the
8 factors agree). Below that, its odds are still recorded for learning, but
they don't change the grade.

  A+  confluence 6-8 AND the Time Machine confirms (simulated average R and
      target-first odds clear the bar in settings)
  A   confluence 6-8, Time Machine does not confirm (or had no data)
  B   confluence 4-5
  C   confluence 0-3

The grade is recorded on every trade and practice trade. It only blocks trades
if `conviction.min_grade` is set, which should wait until the Learning Lab
shows A+ trades really do better.
"""
from __future__ import annotations

GRADES = ["C", "B", "A", "A+"]          # weakest -> strongest
RANK = {g: i for i, g in enumerate(GRADES)}


def confirms(odds, cfg: dict | None = None) -> bool:
    """Does the Time Machine back this setup?"""
    if odds is None:
        return False
    cfg = cfg or {}
    min_r = float(cfg.get("confirm_min_exp_r", 0.0))
    min_p = float(cfg.get("confirm_min_p_target", 0.0))
    return odds.exp_r > min_r and odds.p_target >= min_p and odds.p_target >= odds.p_stop * float(
        cfg.get("confirm_target_vs_stop", 0.0))


def grade(confluence: int | None, odds=None, cfg: dict | None = None) -> str:
    cfg = cfg or {}
    c = int(confluence or 0)
    high = int(cfg.get("high_confluence", 6))
    if c >= high:
        return "A+" if confirms(odds, cfg) else "A"
    if c >= int(cfg.get("medium_confluence", 4)):
        return "B"
    return "C"


def passes(g: str, min_grade: str | None) -> bool:
    return not min_grade or RANK.get(g, 0) >= RANK.get(str(min_grade), 0)


def describe(g: str, confluence: int | None, odds) -> str:
    base = f"grade {g} (confluence {int(confluence or 0)}/8"
    if g in ("A+", "A"):
        base += ", time machine " + ("confirms" if g == "A+" else "does not confirm")
    return base + ")"


def grade_stats(trades: list[dict]):
    """Closed trades grouped by grade: does A+ really win more?"""
    import pandas as pd

    from ..metrics import summarize
    rows = [t for t in trades if t.get("grade") and t.get("r_multiple") is not None
            and t.get("status") == "closed"]
    cols = ["grade", "trades", "win_rate", "expectancy_r", "profit_factor", "t_stat"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    out = []
    for g, part in df.groupby("grade"):
        s = summarize(part["r_multiple"].astype(float))
        out.append({"grade": g, **{k: s[k] for k in cols[1:]}})
    return pd.DataFrame(sorted(out, key=lambda r: -RANK.get(r["grade"], 0)), columns=cols)
