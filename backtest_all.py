import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta

from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# =========================
# SETTINGS
# =========================
TIMEFRAME = "1h"
CONFIRM_TIMEFRAME = "4h"
DAYS_BACK = 120
LOOKAHEAD = 24

PROB_THRESHOLD = 0.85
TOP_SYMBOLS = 500
SHORTLIST_SIZE = 25
RR_MIN = 1.30

MIN_STOP_PCT = 0.0030
ATR_STOP_MULT = 1.50
MAX_STOP_PCT = 0.0200

TP1_RR = 1.0
TP2_RR = 2.0
TP3_RR = 2.4

DRAW_LIMIT = 0.30

# =========================
# MODEL
# =========================
model = xgb.XGBClassifier()
model.load_model("ai_model_tp_sl_bot_1h.json")

FEATURES = [
    "rsi", "atr_pct", "bb_width", "ret_3", "ret_10",
    "vol_ratio", "dist_from_high", "mom_5", "mom_5_acc", "vol_surge"
]

exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# =========================
# FEATURES
# =========================
def add_features(df):
    df = df.copy()

    df["rsi"] = RSIIndicator(df["close"], 14).rsi()

    df["atr"] = AverageTrueRange(
        df["high"],
        df["low"],
        df["close"],
        14
    ).average_true_range()

    bb = BollingerBands(df["close"], 20, 2)

    df["bb_width"] = bb.bollinger_wband()
    df["bb_lband"] = bb.bollinger_lband()
    df["bb_uband"] = bb.bollinger_hband()

    df["atr_pct"] = df["atr"] / df["close"]

    df["ret_3"] = df["close"].pct_change(3)
    df["ret_10"] = df["close"].pct_change(10)

    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    df["recent_high_20"] = df["high"].rolling(20).max()

    df["dist_from_high"] = (
        df["close"] - df["recent_high_20"]
    ) / df["recent_high_20"]

    df["mom_5"] = df["close"].pct_change(5)
    df["mom_5_acc"] = df["mom_5"] - df["mom_5"].shift(1)

    df["vol_surge"] = df["volume"] / df["volume"].rolling(50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    df.dropna(inplace=True)
    return df

def detect_regime(df: pd.DataFrame) -> str:
    close = df["close"]
    ema50 = close.ewm(span=50).mean()

    if len(ema50) < 60:
        return "RANGE"

    slope = (ema50.iloc[-1] - ema50.iloc[-10]) / max(abs(ema50.iloc[-10]), 1e-9)
    atr_pct = float(df["atr_pct"].iloc[-1])
    bb_width = float(df["bb_width"].iloc[-1])
    bb_mean = float(df["bb_width"].rolling(100).mean().iloc[-1])

    if atr_pct > 0.02 or (bb_mean > 0 and bb_width > bb_mean * 1.3):
        return "HIGH_VOL"
    if abs(slope) > 0.002:
        return "TREND"
    return "RANGE"

def pump_score(row, regime: str) -> float:
    score = 0.0
    score += clamp01((float(row["vol_surge"]) - 1.05) / 0.8) * 0.35
    score += clamp01((float(row["vol_ratio"]) - 1.0) / 0.7) * 0.20
    score += clamp01((float(row["mom_5"]) - 0.005) / 0.03) * 0.15
    score += clamp01((float(row["mom_5_acc"]) + 0.01) / 0.04) * 0.10
    score += clamp01((0.08 - abs(float(row["dist_from_high"]))) / 0.08) * 0.10
    score += clamp01((0.025 - float(row["atr_pct"])) / 0.02) * 0.05
    score += clamp01((15.0 - float(row["bb_width"])) / 15.0) * 0.05

    if regime == "TREND":
        score += 0.06
    elif regime == "HIGH_VOL":
        score -= 0.05

    return clamp01(score)

def btc_market_filter(df4_btc_cut: pd.DataFrame) -> bool:
    if len(df4_btc_cut) < 50:
        return True
    row4 = df4_btc_cut.iloc[-1]
    return not (
        float(row4["close"]) < float(row4["ema200"])
        and float(row4["rsi"]) < 40
        and float(row4["mom_5"]) < 0
    )

def compute_levels(entry, atr_pct, regime):
    stop_pct = max(MIN_STOP_PCT, atr_pct * ATR_STOP_MULT)
    stop_pct = min(stop_pct, MAX_STOP_PCT)

    stop = entry * (1 - stop_pct)

    rr1 = TP1_RR
    rr2 = TP2_RR
    rr3 = TP3_RR

    if regime == "TREND":
        rr2 *= 1.03
        rr3 *= 1.05
    elif regime == "HIGH_VOL":
        rr1 *= 0.95
        rr2 *= 0.90
        rr3 *= 0.85

    tp1 = entry * (1 + stop_pct * rr1)
    tp2 = entry * (1 + stop_pct * rr2)
    tp3 = entry * (1 + stop_pct * rr3)

    return stop_pct, stop, tp1, tp2, tp3, rr1, rr2, rr3

def candidate_score(prob, rr2, rr3, regime, confirm_ok, pump):
    regime_bonus = {"TREND": 2.5, "RANGE": 0.0, "HIGH_VOL": -1.8}.get(regime, 0.0)
    confirm_bonus = 4.0 if confirm_ok else -2.5
    return (prob * 100.0) + (rr2 * 7.0) + (min(rr3, 3.0) * 2.0) + regime_bonus + confirm_bonus + (pump * 12.0)

# =========================
# LOAD DATA WINDOW
# =========================
since = exchange.parse8601(
    (datetime.utcnow() - timedelta(days=DAYS_BACK))
    .strftime("%Y-%m-%dT%H:%M:%S")
)

markets = exchange.load_markets()

symbols = [
    m["symbol"] for m in markets.values()
    if m.get("active")
    and m.get("spot")
    and m.get("quote") == "USDT"
]

symbols = symbols[:TOP_SYMBOLS]

data_1h = {}
data_4h = {}

# حمّل BTC دائمًا لفلتر السوق العام
symbols_for_fetch = list(dict.fromkeys(symbols + ["BTC/USDT"]))

for symbol in symbols_for_fetch:
    try:
        ohlcv1 = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=1000)
        df1 = pd.DataFrame(ohlcv1, columns=["ts", "open", "high", "low", "close", "volume"])
        df1["ts"] = pd.to_datetime(df1["ts"], unit="ms")
        df1.set_index("ts", inplace=True)
        df1 = add_features(df1)
        if len(df1) >= 250:
            data_1h[symbol] = df1

        ohlcv4 = exchange.fetch_ohlcv(symbol, timeframe=CONFIRM_TIMEFRAME, since=since, limit=1000)
        df4 = pd.DataFrame(ohlcv4, columns=["ts", "open", "high", "low", "close", "volume"])
        df4["ts"] = pd.to_datetime(df4["ts"], unit="ms")
        df4.set_index("ts", inplace=True)
        df4 = add_features(df4)
        if len(df4) >= 220:
            data_4h[symbol] = df4
    except Exception:
        pass

# =========================
# RESULTS
# =========================
wins = 0
losses = 0
draws = 0

tp1_hits = 0
tp2_hits = 0
tp3_hits = 0

total = 0
skipped_no_confirm = 0
skipped_by_btc = 0

# =========================
# BACKTEST LOOP
# =========================
common_symbols = [s for s in symbols if s in data_1h and s in data_4h]
max_i = min([len(data_1h[s]) for s in common_symbols]) if common_symbols else 0

for i in range(220, max_i - LOOKAHEAD):
    candidates = []

    btc_ok = True
    if "BTC/USDT" in data_4h and "BTC/USDT" in data_1h:
        ts_ref = data_1h[common_symbols[0]].index[i]
        df4_btc_cut = data_4h["BTC/USDT"][data_4h["BTC/USDT"].index <= ts_ref]
        btc_ok = btc_market_filter(df4_btc_cut)

    for symbol in common_symbols:
        try:
            df = data_1h[symbol]
            row = df.iloc[i]

            entry = float(row["close"])
            atr_pct = float(row["atr_pct"])
            if atr_pct < MIN_STOP_PCT:
                continue

            ema200 = float(row["ema200"])
            if entry < ema200:
                continue

            if float(row["rsi"]) > 65:
                continue

            X = row[FEATURES].values.reshape(1, -1)
            prob = model.predict_proba(X)[0][1]
            if prob < PROB_THRESHOLD:
                continue

            regime = detect_regime(df.iloc[:i+1])
            stop_pct, stop, tp1, tp2, tp3, rr1, rr2, rr3 = compute_levels(entry, atr_pct, regime)
            if rr3 < RR_MIN:
                continue

            ts = df.index[i]
            df4 = data_4h[symbol]
            df4_cut = df4[df4.index <= ts]
            if len(df4_cut) < 50:
                continue
            row4 = df4_cut.iloc[-1]
            confirm_ok = (
                float(row4["close"]) >= float(row4["ema200"])
                and float(row4["rsi"]) < 72
                and float(row4["mom_5"]) > -0.03
            )

            pump = pump_score(row, regime)

            if (not btc_ok) and prob < 0.92 and pump < 0.62:
                skipped_by_btc += 1
                continue

            if (not confirm_ok) and prob < 0.88 and pump < 0.62:
                skipped_no_confirm += 1
                continue

            score = candidate_score(prob, rr2, rr3, regime, confirm_ok, pump)

            candidates.append({
                "symbol": symbol,
                "i": i,
                "entry": entry,
                "stop": stop,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "prob": prob,
                "rr3": rr3,
                "score": score,
                "confirm_ok": confirm_ok,
                "pump": pump,
            })

        except Exception:
            continue

    if not candidates:
        continue

    candidates.sort(key=lambda x: x["score"], reverse=True)
    shortlist = candidates[:SHORTLIST_SIZE]
    shortlist.sort(
        key=lambda x: (x["confirm_ok"], x["pump"], x["prob"], x["score"]),
        reverse=True,
    )
    best = shortlist[0]

    symbol = best["symbol"]
    df = data_1h[symbol]
    future = df.iloc[i+1:i+1+LOOKAHEAD]

    entry = best["entry"]
    stop = best["stop"]
    tp1 = best["tp1"]
    tp2 = best["tp2"]
    tp3 = best["tp3"]

    max_tp = 0
    result = None

    for _, frow in future.iterrows():
        # نفضل TP3 إذا لُمس مع الوقف في نفس الشمعة لتقليل الانحياز
        if frow["high"] >= tp3:
            max_tp = 3
            result = "WIN"
            break

        if frow["low"] <= stop:
            result = "LOSS"
            break

        if frow["high"] >= tp1:
            max_tp = max(max_tp, 1)

        if frow["high"] >= tp2:
            max_tp = max(max_tp, 2)

    if result is None:
        last_price = future.iloc[-1]["close"]
        pnl_pct = (last_price - entry) / entry * 100

        if max_tp == 3:
            result = "WIN"
        elif max_tp == 2:
            result = "WIN"
            tp2_hits += 1
        elif max_tp == 1:
            result = "WIN"
            tp1_hits += 1
        else:
            if abs(pnl_pct) < DRAW_LIMIT:
                result = "DRAW"
                draws += 1
            else:
                result = "LOSS"

    if result == "WIN":
        wins += 1
        if max_tp == 3:
            tp3_hits += 1
    elif result == "LOSS":
        losses += 1

    total += 1

# =========================
# STATS
# =========================
if total > 0:
    winrate = wins / total * 100

    print("\n========== BACKTEST RESULTS ==========\n")
    print("Symbols tested:", len(common_symbols))
    print("Total trades:", total)

    print("\nWins:", wins)
    print("Losses:", losses)
    print("Draws:", draws)

    print("\nWinrate:", round(winrate, 2))

    print("\nTP1 hits:", tp1_hits)
    print("TP2 hits:", tp2_hits)
    print("TP3 hits:", tp3_hits)
    print("Skipped by weak 4h confirmation:", skipped_no_confirm)
    print("Skipped by BTC market filter:", skipped_by_btc)
else:
    print("No trades were generated.")
