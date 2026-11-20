"""MarketPilot command line.

  python -m marketpilot research            # Strategy Research + Backtesting bots (weekly)
  python -m marketpilot run-once            # one trading cycle (schedule hourly)
  python -m marketpilot loop --minutes 60   # keep cycling (for a VPS / always-on box)
  python -m marketpilot report              # print win rate & performance
  python -m marketpilot kill | resume       # system-wide kill switch
  python -m marketpilot notify-test         # send a test push notification (needs NTFY_TOPIC)
  python -m marketpilot demo --days 30      # offline end-to-end demo on synthetic data

Storage: --db supabase   (env SUPABASE_URL + SUPABASE_SERVICE_KEY)
         --db sqlite:///marketpilot.db (default)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

from .config import load_settings
from .storage import make_store


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="marketpilot")
    ap.add_argument("command", choices=["research", "run-once", "loop", "report", "kill", "resume", "demo", "notify-test"])
    ap.add_argument("--db", default=None)
    ap.add_argument("--provider", default="yfinance", help="yfinance | synthetic | ccxt:kraken")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--minutes", type=int, default=60)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--relaxed", action="store_true", help="demo only: loosen approval gate")
    args = ap.parse_args(argv)

    settings = load_settings()
    store = make_store(args.db)
    now = datetime.now(timezone.utc)

    if args.command == "notify-test":
        from . import notify
        from .bots.performance import report
        ok = notify.send("MarketPilot is connected", "Test notification: your phone will get trade alerts "
                         "and a daily summary from the paper-trading bots.", tags="tada")
        rep = report(store.trades(), settings.account["starting_balance"])["overall"]
        print("sent" if ok else "NOT sent - is NTFY_TOPIC set?", rep["trades"], "closed trades so far")
        return 0 if ok else 1
    if args.command == "kill":
        from .bots.safety import SafetyBot
        SafetyBot(store, settings).engage("manual kill switch from CLI")
        print("kill switch ON — no new trades; open positions still managed")
        return 0
    if args.command == "resume":
        from .bots.safety import SafetyBot
        SafetyBot(store, settings).release()
        print("kill switch OFF")
        return 0
    if args.command == "report":
        from .bots.performance import report
        r = report(store.trades(), settings.account["starting_balance"])["overall"]
        target = settings["goals"]["target_win_rate"]
        print(f"closed trades   {r['trades']}  (wins {r['wins']} / losses {r['losses']})")
        print(f"win rate        {r['win_rate']:.1%}  95% range {r['win_rate_lo']:.0%}–{r['win_rate_hi']:.0%}"
              f"  (goal {target:.0%})")
        print(f"win/loss ratio  {r['win_loss_ratio']:.2f}   avg win {r['avg_win_r']:+.2f}R  avg loss {r['avg_loss_r']:+.2f}R")
        print(f"expectancy      {r['expectancy_r']:+.3f}R per trade   profit factor {r['profit_factor']:.2f}")
        print(f"P&L             £{r['pnl_gbp']:+.2f} ({r['return_pct']:+.2f}%)  max drawdown {r['max_drawdown_r']:.2f}R")
        return 0
    if args.command == "demo":
        from .engine import run_demo
        start = datetime(2026, 6, 1, tzinfo=timezone.utc)
        r = run_demo(settings, store, start, args.days, args.relaxed)
        o = r["overall"]
        print(f"\nDEMO (synthetic data): {r['new_orders']} orders, {o['trades']} closed, "
              f"win rate {o['win_rate']:.0%}, expectancy {o['expectancy_r']:+.3f}R, P&L £{o['pnl_gbp']:+.2f}")
        return 0

    from .data import make_provider
    provider = make_provider(args.provider)
    if args.command == "research":
        from .bots.research import research
        research(settings, provider, store, now, symbols=args.symbols)
        return 0

    from .engine import Engine
    eng = Engine(settings, provider, store, live=args.provider != "synthetic")
    while True:
        s = eng.cycle(datetime.now(timezone.utc))
        print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} cycle: {s}")
        if args.command == "run-once":
            return 0
        time.sleep(args.minutes * 60)


if __name__ == "__main__":
    sys.exit(main())
