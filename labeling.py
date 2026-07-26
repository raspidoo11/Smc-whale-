"""Triple-barrier labeling (López de Prado, *Advances in Financial ML*, 2018).

Why this exists
---------------
The classifier's label used to be the bot's own realized outcome: whatever the
trade manager happened to do — trailing stop, time stop, breakeven move,
manual reconcile — decided whether a setup was labeled a win. That conflates
two very different questions:

  1. Was this a good SETUP?              <- what the model should predict
  2. Did our EXIT MANAGEMENT work?       <- a separate engineering problem

So every change to trailing/time-stop logic silently relabeled the entire
training corpus, and the model was partly learning to predict the exit code's
behaviour rather than the market's.

Triple-barrier labeling fixes that by defining a fixed, explicit exit policy
and asking which of three barriers price touches FIRST:

  * profit barrier   - the take-profit level
  * stop barrier     - the invalidation level
  * vertical barrier - a maximum holding time

The label is then a property of the market path after entry, not of our exit
code. It is also defined for setups we never traded, which is what makes
meta-labeling possible: the SMC engine picks the side, and a secondary model
learns P(profit | features) over these labels.

Barrier choice
--------------
Barriers default to the trade's ACTUAL sl/tp, not an ATR band. For meta-labeling
that is the correct choice: the secondary model has to predict whether *this
strategy's* geometry works, and López de Prado's own meta-labeling example uses
the primary model's barriers. ATR-scaled barriers are available via
`barriers_from_atr` for research that needs labels comparable across geometry
changes.

Intrabar ambiguity
------------------
OHLC data cannot say whether the high or the low came first within one candle.
When both barriers fall inside the same bar the outcome is recorded as a STOP,
and `ambiguous` is set. That is the pessimistic assumption, and it matches the
backtester's existing convention that an SL breach beats a limit fill sharing a
candle. Optimism here is how a backtest invents edge that does not exist: it
would resolve every coin-flip candle in your favour.
"""

from dataclasses import dataclass, asdict

LABEL_PROFIT = 1
LABEL_STOP = -1
LABEL_TIMEOUT = 0

BARRIER_PROFIT = "profit"
BARRIER_STOP = "stop"
BARRIER_VERTICAL = "vertical"


@dataclass(frozen=True)
class BarrierOutcome:
    """Result of walking price forward from an entry until a barrier is hit."""

    label: int          # LABEL_PROFIT / LABEL_STOP / LABEL_TIMEOUT
    barrier: str        # which barrier resolved it
    bars_to_touch: int  # bars from entry to resolution (>=1; 0 only if no data)
    touch_price: float  # price at resolution
    realized_r: float   # signed R-multiple at resolution
    ambiguous: bool     # both barriers sat inside the resolving candle

    def as_dict(self, prefix="tb_"):
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def barriers_from_atr(direction, entry, atr, profit_mult, stop_mult):
    """Volatility-scaled barriers: entry +/- multiple x ATR.

    Alternative to the trade's own sl/tp when labels need to stay comparable
    across changes to the strategy's risk geometry.
    """
    if atr <= 0:
        raise ValueError(f"atr must be positive, got {atr}")
    if direction == "LONG":
        return entry + atr * profit_mult, entry - atr * stop_mult
    return entry - atr * profit_mult, entry + atr * stop_mult


def apply_triple_barrier(
    direction,
    entry,
    profit_price,
    stop_price,
    highs,
    lows,
    closes,
    max_bars,
):
    """Walk forward bars from an entry and report the first barrier touched.

    `highs`/`lows`/`closes` must start at the FIRST BAR AFTER the fill (or the
    fill bar itself if a stop can legitimately trigger on it) and be aligned
    with each other. `max_bars` is the vertical barrier in bars.

    Returns a BarrierOutcome. When the vertical barrier is reached first, the
    label is LABEL_TIMEOUT and `realized_r` carries the mark-to-market R at
    that point — callers decide how to binarize (see WIN_LABEL_MIN_R), so
    there is exactly one binarization rule in the codebase.
    """
    if direction not in ("LONG", "SHORT"):
        raise ValueError(f"direction must be LONG or SHORT, got {direction!r}")
    if max_bars < 1:
        raise ValueError(f"max_bars must be >= 1, got {max_bars}")

    risk = abs(entry - stop_price)
    if risk <= 0:
        raise ValueError(
            f"degenerate stop: entry={entry} stop={stop_price} (zero risk distance)"
        )

    is_long = direction == "LONG"

    def signed_r(price):
        move = (price - entry) if is_long else (entry - price)
        return round(move / risk, 4)

    n = min(len(highs), len(lows), max_bars)

    for k in range(n):
        high, low = highs[k], lows[k]
        hit_profit = high >= profit_price if is_long else low <= profit_price
        hit_stop = low <= stop_price if is_long else high >= stop_price

        if hit_profit and hit_stop:
            # Ambiguous candle -> assume the adverse side resolved first.
            return BarrierOutcome(
                label=LABEL_STOP,
                barrier=BARRIER_STOP,
                bars_to_touch=k + 1,
                touch_price=float(stop_price),
                realized_r=signed_r(stop_price),
                ambiguous=True,
            )
        if hit_stop:
            return BarrierOutcome(
                label=LABEL_STOP,
                barrier=BARRIER_STOP,
                bars_to_touch=k + 1,
                touch_price=float(stop_price),
                realized_r=signed_r(stop_price),
                ambiguous=False,
            )
        if hit_profit:
            return BarrierOutcome(
                label=LABEL_PROFIT,
                barrier=BARRIER_PROFIT,
                bars_to_touch=k + 1,
                touch_price=float(profit_price),
                realized_r=signed_r(profit_price),
                ambiguous=False,
            )

    # Vertical barrier: mark to market at the last bar we could see.
    if n <= 0:
        # No forward data at all — unlabelable, not a timeout at breakeven.
        return BarrierOutcome(
            label=LABEL_TIMEOUT,
            barrier=BARRIER_VERTICAL,
            bars_to_touch=0,
            touch_price=float(entry),
            realized_r=0.0,
            ambiguous=False,
        )

    last_close = float(closes[n - 1])
    return BarrierOutcome(
        label=LABEL_TIMEOUT,
        barrier=BARRIER_VERTICAL,
        bars_to_touch=n,
        touch_price=last_close,
        realized_r=signed_r(last_close),
        ambiguous=False,
    )


def label_trade(trade, highs, lows, closes, max_bars, profit_atr=None, stop_atr=None):
    """Convenience wrapper: label one trade dict from its own sl/tp geometry.

    Pass `profit_atr`/`stop_atr` to override with ATR-scaled barriers instead.
    Returns None when the trade lacks the fields needed to label it honestly.
    """
    direction = trade.get("direction")
    try:
        entry = float(trade["entry"])
    except (KeyError, TypeError, ValueError):
        return None

    if profit_atr is not None and stop_atr is not None:
        atr = float(trade.get("atr") or 0)
        if atr <= 0:
            return None
        profit_price, stop_price = barriers_from_atr(
            direction, entry, atr, profit_atr, stop_atr
        )
    else:
        try:
            profit_price = float(trade["tp"])
            stop_price = float(trade["sl"])
        except (KeyError, TypeError, ValueError):
            return None

    try:
        return apply_triple_barrier(
            direction, entry, profit_price, stop_price,
            highs, lows, closes, max_bars,
        )
    except ValueError:
        return None
