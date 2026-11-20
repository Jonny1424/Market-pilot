"""Transaction cost and fill model — shared by the backtester and the paper broker,
so backtests and forward paper trades pay the same costs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    commission_pct: float = 0.0     # % of notional per side
    commission_fixed: float = 0.0   # account currency per side

    @property
    def half_spread(self) -> float:
        return self.spread_bps / 2 / 10_000

    @property
    def slip(self) -> float:
        return self.slippage_bps / 10_000

    def market_fill(self, mid: float, side: int) -> float:
        """Price paid (side=+1 buy) or received (side=-1 sell) for a market order."""
        return mid * (1 + side * (self.half_spread + self.slip))

    def limit_fill(self, level: float, side: int) -> float:
        """Take-profit limit: crosses the spread but no slippage."""
        return level * (1 + side * self.half_spread)

    def commission_per_unit(self, price: float) -> float:
        return price * self.commission_pct / 100

    def round_trip_cost_frac(self) -> float:
        """Approximate total cost of entering and exiting, as a fraction of price."""
        return 2 * (self.half_spread + self.slip + self.commission_pct / 100)


def cost_model_for(settings, asset_class: str) -> CostModel:
    return CostModel(**settings["costs"][asset_class])
