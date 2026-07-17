# SMC Whale — AI-Assisted Crypto Futures Bot

An automated **Bybit** USDT-perp trading bot that fuses **Smart Money Concepts**
(liquidity sweeps, fair-value gaps, break-of-structure, displacement, volume
spikes) with a self-retraining **XGBoost + LightGBM** probability model. It
scans the market, scores setups, sizes and executes trades (or paper-trades),
manages exits with trailing stops, and continuously retrains on its own
realized outcomes. Runs 24/7 as a Railway worker and alerts to Telegram.

---

## Architecture

```
scanner.py        Pulls live USDT pairs from Bybit, filters meme/stable/low-vol,
                  fetches 15m (bias) + 5m (entry) OHLCV.
        │
        ▼
strategy.py       Signal engine. Builds SMC features, scores confluence, blends
                  in the AI probability, emits LONG/SHORT with entry/SL/TP.
        │           └── uses the SAME featurizer as the trainer (parity).
        ▼
paper_trader.py   Position sizing + PnL/fees (paper).
bybit_executor.py Live order placement, AI risk sizing, trailing stops.
        │
        ▼
trade_manager.py  Persistence (atomic JSON in DATA_DIR): balance, open trades,
                  history, signal hashes, cooldowns.
        │
        ▼
trade_monitor.py  Polls open trades for SL/TP, activates trailing stops, closes.
        │
        ▼
xgboost_trainer.py  Walk-forward feature frame, calibrated 2-model ensemble,
                    champion/challenger promotion, expected-R regression,
                    feature-importance decay, self-diagnostics.

main.py           Scheduler: scan (5m), monitor (35s), retrain (10m),
                  daily reset (00:00 UTC).
config.py         Single source of truth for paths + risk knobs.
```

## Signal → trade lifecycle

1. `scan()` pulls fresh symbols (excluding ones with open trades).
2. `get_signal()` computes features and returns a **signal snapshot** — a single
   canonical dict with every raw field (regime, atr_percentile, EMA/VWAP
   distances, body/volume, session/time, SMC flags).
3. Top signals by confidence are sized, executed/paper-opened, and persisted.
4. `monitor_trades()` watches price; on SL it closes, on TP it arms a trailing
   stop, then records the outcome (WIN/LOSS + realized PnL/fees).
5. On its own 10-minute cadence, `train_model_incremental()` retrains on the
   rolling window of closed trades and promotes a new model only if it beats
   the incumbent on a held-out AUC.

## Two modes (`USE_XGBOOST`)

- **SMC mode** (`false`): looser thresholds, generates more trades to bootstrap
  training data. Confidence = raw SMC score.
- **AI mode** (`true`): confidence = `0.6·SMC + 0.4·AI_probability`, with:
  - an **adaptive confidence threshold** (`get_dynamic_confidence_threshold`) —
    looser in clean trends, stricter in chop/high-volatility and after a losing
    run — instead of a fixed cutoff;
  - an **expected-R entry filter** — the auxiliary regression must predict at
    least `MIN_EXPECTED_R` before an otherwise-confident signal is taken;
  - position risk scaled by model confidence (`get_ai_risk_percent`).

## Train / serve feature parity

The single most important invariant: **inference and training build features
with the same function** (`extract_pro_features_from_trade`). `get_signal`
persists the full raw snapshot on every trade, so the trainer reads real values
(not defaults) and the live model is scored on the exact feature distribution it
learned. `tests/test_feature_parity.py` guards this contract.

---

## Configuration

Copy `.env.example` → `.env`. Environment variables:

| Var | Default | Purpose |
|-----|---------|---------|
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | — | Alerts |
| `BYBIT_API_KEY` / `BYBIT_API_SECRET` | — | Trading (live/demo/testnet) |
| `TRADE_MODE` | `demo` | `demo` \| `testnet` \| `live` |
| `EXECUTE_TRADES` | `true` | `false` = paper only, no Bybit orders |
| `USE_XGBOOST` | `false` | Enable the AI probability layer |
| `START_BALANCE` | `100` | Paper starting balance |
| `RISK_PERCENT` | `0.5` | Live base risk % of balance |
| `DAILY_LOSS_LIMIT` | `15` | Pause trading after this daily loss |
| `MAX_OPEN_TRADES` | `10` | Concurrent open positions cap |
| `MAX_PORTFOLIO_RISK_PCT` | `5.0` | Total open risk ("heat") cap, % of balance |
| `MAX_TRADES_PER_DIRECTION` | `6` | Max simultaneous same-direction positions |
| `MAX_ALT_POSITIONS` | `8` | Max concurrent non-major (alt) positions |
| `MIN_EXPECTED_R` | `0.0` | Min model-expected R to accept a trade (AI mode) |
| `TRAIL_ACTIVATION_RATIO` | `0.97` | Fraction of entry→TP at which TP is cancelled and trailing arms |
| `TRAIL_PERCENT` | `0.5` | Trailing distance as % of price |
| `ENTRY_MODE` | `limit` | `limit` = prediction zones (OB / FVG / ATR pullback); `market` = chase close |
| `LIMIT_TTL_MINUTES` | `180` | How long a resting prediction may wait (desk-style) |
| `INVALIDATE_PENDING_ON_STRUCTURE` | `true` | Cancel unfilled limits if structure breaks before fill |
| `RETRACE_ATR_FRACTION` | `0.45` | Fallback pullback depth (×ATR) when no FVG/OB |
| `MIN_SL_ATR` | `1.15` | Minimum stop distance in ATRs (anti stop-hunt floor) |
| `STRUCTURE_SL_BUFFER_ATR` | `0.25` | Extra room beyond structural swing (×ATR) so equal-high/low sweeps don't tag SL |
| `STRUCTURE_SWING_LOOKBACK` | `20` | Bars per TF for structural swing; SL uses the *wider* of entry (5m) + bias (15m) |
| `SPREAD_MAX_FRACTION_OF_RISK` | `0.15` | Skip entry if bid-ask spread eats more of the risk than this |
| `CONFIDENCE_REQUIRED_SMC` | `40` | Confidence bar in pure-SMC *market* mode |
| `CONFIDENCE_REQUIRED_LIMIT` | `28` | Softer bar for resting prediction limits |
| `LIMIT_MIN_SETUP_SCORE` | `20` | Min soft setup score (HTF bias + edge) to place a limit |
| `NEWS_FILTER_ENABLED` | `false` | Pause entries in high-impact macro windows (edit news_filter.py first) |
| `BLOCKED_SESSIONS` | *(none)* | UTC sessions with no new entries, e.g. `asian` or `london,quiet` (valid: asian/london/ny/quiet) |
| `MAX_HOLD_MINUTES` | `0` (off) | Time stop: market-close any OPEN trade held this long (trailing winners exempt) |
| `SL_MAX_ATR_MULT` | `0` (off) | Cap stop distance at N×ATR (keep above MIN_SL_ATR); faster resolution, more stop-outs |
| `AI_MAX_WEIGHT` | `0.40` | Ceiling of the model's vote in final confidence (trust ramp still applies) |
| `AI_WEIGHT_FULL_AT` | `150` | Real closed trades at which the model reaches its full vote |
| `SLIPPAGE_PCT` | `0.02` | Adverse slippage per side in backtests |
| `STORAGE_BACKEND` | `sqlite` | `sqlite` (auto-migrates JSON) or `json` |
| `DATA_DIR` | `data` | State/model root (Railway volume: `/app/data`) |

> **Railway note:** `railway.json` mounts a volume at `/app/data`. The worker
> runs from `/app`, so the default relative `DATA_DIR=data` resolves onto that
> volume. Set `DATA_DIR=/app/data` explicitly if you change the workdir.

## Running

```bash
# install
pip install -r requirements.txt          # prod
pip install -r requirements-dev.txt       # + pytest for tests

# run the worker
python main.py

# run tests
python -m pytest tests/ -q
```

## Tests

- `tests/test_money_math.py` — quantity sizing, fees, realized-R, and that trade
  status is derived from PnL sign (not fragile exit-reason strings).
- `tests/test_feature_parity.py` — the train/serve feature contract, signal-hash
  symbol uniqueness, and EMA/VWAP presence.

## Backtesting

Replay historical candles through the **live** signal engine + a realistic fill
simulator (SL/TP, ratcheting trailing stop, taker fees):

```bash
python backtester.py "BTC/USDT:USDT" --candles 3000
python backtester.py "ETH/USDT:USDT" --candles 5000 --xgboost
```

Reports win rate, profit factor, expectancy, avg R, max drawdown, and Sharpe.
The simulation core (`backtester.simulate`) is pure/offline and unit-tested.
Fills model adverse slippage (`SLIPPAGE_PCT`) and, in limit mode, resting-order
fills/expiries — so backtest numbers track what the live bot would experience.

Tune the hand-picked thresholds empirically with the optimizer:

```bash
python optimize.py "BTC/USDT:USDT" "ETH/USDT:USDT" --candles 3000
```

It grid-searches confidence bar, retrace depth, and trailing parameters across
both entry modes and prints the best combination as ready-to-set env vars.

## Precision entries

- **Closed candles only** — the scanner drops the currently-forming candle, so
  signals can't repaint; scans fire within ~20s of each 5m candle close.
- **Retrace limit entries** (`ENTRY_MODE=limit`, default) — instead of chasing
  the displacement candle's close, a limit order rests at the FVG midpoint (or
  an ATR-fraction pullback) as a `PENDING` trade. Telegram alerts fire on
  placement, fill, and expiry; unfilled orders cancel after `LIMIT_TTL_MINUTES`
  and never pollute training history.
- **Spread gate** — entries are skipped when the bid-ask spread eats more than
  `SPREAD_MAX_FRACTION_OF_RISK` of the planned stop distance.
- **Market-context features** — signals are enriched with funding rate,
  open-interest change, BTC 15m structure, spread, and per-symbol win rate;
  all persisted for training so the model sees exactly what inference saw.

## Portfolio risk & reconciliation

- **`risk_manager.py`** gates every new trade on total open "heat", direction
  concentration, and alt concentration — not just `MAX_OPEN_TRADES`.
- **`reconcile.py`** (live/demo only, every 2 min) closes local trades that
  Bybit already closed exchange-side (recording real exit/PnL) and snaps local
  balance to real wallet equity, so local state can't drift from the exchange.

## Safety notes

- Start in `EXECUTE_TRADES=false` (paper) or `TRADE_MODE=demo`. Only move to
  `live` after validating behavior and running a backtest.
- Circuit breakers: `DAILY_LOSS_LIMIT` (daily) + portfolio heat cap (per-scan).

## Roadmap (next)

- Multi-symbol portfolio backtest (current backtester is per-symbol).
- Correlation matrix from returns (alt-concentration is a heuristic proxy today).
- Slippage/funding modeling in both paper fills and backtests.
