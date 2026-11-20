"""Trade Decision Bot — combines the other bots and may ABSTAIN.

Expected value is estimated from evidence, not from how good a chart looks:
  * out-of-sample backtest expectancy (already net of costs),
  * blended with this strategy's live paper results on the symbol once available,
  * shrunk toward zero by n/(n + k) so thin evidence counts for less.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..metrics import summarize


@dataclass
class Decision:
    trade: bool
    expected_r: float
    reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


class DecisionBot:
    def __init__(self, settings):
        self.cfg = settings["decision"]

    def decide(self, *, approval: dict, regime: str, strategy_regimes, scan: dict,
               news_vetoes: list[str], live_r: list[float]) -> Decision:
        reasons = []
        oos = approval["oos"]
        n_oos, e_oos = oos["trades"], oos["expectancy_r"]
        live = summarize(live_r)
        n = n_oos + live["trades"]
        blended = (n_oos * e_oos + live["trades"] * live["expectancy_r"]) / n if n else 0.0
        ev = blended * n / (n + self.cfg["shrinkage_trades"])
        evidence = {"oos_trades": n_oos, "oos_expectancy_r": e_oos, "oos_win_rate": oos["win_rate"],
                    "live_trades": live["trades"], "live_expectancy_r": live["expectancy_r"],
                    "shrunk_ev_r": ev, "regime": regime, "scan": scan}
        if not scan["ok"]:
            reasons += scan["reasons"]
        if regime not in strategy_regimes:
            reasons.append(f"regime '{regime}' not suited to this strategy")
        reasons += news_vetoes
        if ev < self.cfg["min_expected_r"]:
            reasons.append(f"estimated edge {ev:+.3f}R below {self.cfg['min_expected_r']}R")
        return Decision(not reasons, ev, reasons, evidence)
