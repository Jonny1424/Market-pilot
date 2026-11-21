"""Risk Manager Bot — deterministic position sizing and hard limits."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskResult:
    ok: bool
    quantity: float = 0.0
    risk_gbp: float = 0.0
    notional_gbp: float = 0.0
    cost_r: float = 0.0
    reasons: list[str] = field(default_factory=list)
    risk_mult: float = 1.0
    risk_notes: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, settings):
        self.a = settings.account
        self.costs = settings["costs"]
        self.max_daily_trades = settings["safety"]["max_daily_trades"]
        from .losscontrol import config
        self.lc = config(settings.raw)

    def evaluate(self, *, symbol: str, asset_class: str, equity: float, realized_today: float,
                 trades_today: int, open_trades: list[dict], entry_est: float,
                 risk_per_unit: float, fx_to_gbp: float, kill_switch: bool,
                 vol_rank=None, drawdown_pct: float = 0.0, loss_streak: int = 0) -> RiskResult:
        from .losscontrol import concentration_reasons, multiplier
        a, reasons = self.a, []
        if kill_switch:
            reasons.append("kill switch is ON")
        if realized_today <= -equity * a["daily_loss_limit_pct"] / 100:
            reasons.append(f"daily loss limit hit (£{realized_today:.2f} today)")
        if trades_today >= self.max_daily_trades:
            reasons.append(f"max {self.max_daily_trades} new trades per day reached")
        active = [t for t in open_trades if t["status"] in ("open", "pending")]
        if len(active) >= a["max_open_positions"]:
            reasons.append(f"max {a['max_open_positions']} open positions reached")
        if any(t["symbol"] == symbol for t in active):
            reasons.append(f"already have a position in {symbol}")
        if risk_per_unit <= 0 or entry_est <= 0:
            reasons.append("invalid stop distance")
        if reasons:
            return RiskResult(False, reasons=reasons)

        mult, notes = multiplier(self.lc, vol_rank=vol_rank, drawdown_pct=drawdown_pct, streak=loss_streak)
        risk_gbp = equity * a["risk_per_trade_pct"] / 100 * mult
        conc = concentration_reasons(asset_class, active, risk_gbp, equity, self.lc)
        if conc:
            return RiskResult(False, reasons=conc, risk_mult=mult, risk_notes=notes)
        qty = risk_gbp / (risk_per_unit * fx_to_gbp)
        notional = qty * entry_est * fx_to_gbp
        used = sum(float(t.get("notional_gbp") or 0) for t in active)
        cap = equity * a["max_leverage"] - used
        if notional > cap:
            if cap <= 0:
                return RiskResult(False, reasons=["no capital left without leverage"])
            scale = cap / notional
            if scale < 0.25:
                return RiskResult(False, reasons=[
                    f"position would need {notional:,.0f} GBP notional; only {cap:,.0f} available "
                    "without leverage (risk would fall below 25% of plan)"])
            qty, notional, risk_gbp = qty * scale, cap, risk_gbp * scale
        c = self.costs[asset_class]
        frac = 2 * (c["spread_bps"] / 2 + c["slippage_bps"]) / 10_000 + 2 * c["commission_pct"] / 100
        cost_gbp = notional * frac + 2 * c["commission_fixed"]
        cost_r = cost_gbp / risk_gbp if risk_gbp else 1.0
        if cost_r > a["max_cost_fraction_of_risk"]:
            return RiskResult(False, cost_r=cost_r, reasons=[
                f"costs ≈ £{cost_gbp:.2f} = {cost_r:.0%} of the £{risk_gbp:.2f} risk "
                f"(limit {a['max_cost_fraction_of_risk']:.0%})"])
        return RiskResult(True, qty, risk_gbp, notional, cost_r, risk_mult=mult, risk_notes=notes)
