import os
import math
import datetime as dt
import re
import time
import asyncio
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import ccxt
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from xgboost import XGBClassifier

import db

# =========================
# Token from ENV
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("Missing BOT_TOKEN env var. ضع التوكن في متغير البيئة BOT_TOKEN.")

# =========================
# Binance Spot (Public only)
# =========================
EX = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
})

# =========================
# Scan settings
# =========================
QUOTE = "USDT"
TIMEFRAME = "1h"
CONFIRM_TIMEFRAME = "4h"
LIMIT = 510
TOP_SYMBOLS = 500
SHORTLIST_SIZE = 25
ALERT_MINUTES = 10

# ✅ Subscribers (persisted in DB settings)
SUBSCRIBERS = set()
SUBSCRIBERS_SETTINGS_KEY = "broadcast_chat_ids"  # comma-separated int ids

# Cache السيولة
LIQ_CACHE = {"ts": 0.0, "symbols": []}
LIQ_CACHE_TTL = 60 * 60  # ساعة

# =========================
# Concurrency + Dedup (6h)
# =========================
SCAN_LOCK = asyncio.Lock()

DEDUP_HOURS = 6
DEDUP_SECONDS = DEDUP_HOURS * 3600
LAST_SENT_BY_SYMBOL = {}  # symbol -> unix_ts

OPEN_TRADE_EXPIRY_HOURS = 24
OPEN_TRADE_EXPIRY_SECONDS = OPEN_TRADE_EXPIRY_HOURS * 3600

# إذا لم تصل لأي هدف خلال 24 ساعة ولم تلمس الستوب:
# إن كانت الخسارة صغيرة => DRAW
DRAW_MAX_LOSS_PCT = 0.30

# إيقاف إعادة التدريب مؤقتًا
RETRAIN_ENABLED = False
RETRAIN_EVERY_HOURS = 48
RETRAIN_MIN_SAMPLES = 50
RETRAIN_LAST_KEY = "last_retrain_time"

# =========================
# AI Model
# =========================
MODEL = XGBClassifier()
MODEL.load_model("ai_model_tp_sl_bot_1h.json")

FEATURES = [
    "rsi", "atr_pct", "bb_width", "ret_3", "ret_10", "vol_ratio",
    "dist_from_high", "mom_5", "mom_5_acc", "vol_surge",
]

# =========================
# Helpers
# =========================
def now_str():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def mecca_time_str():
    tz = ZoneInfo("Asia/Riyadh")
    return dt.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

def f(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

def fmt_symbol(sym: str) -> str:
    return sym

def is_safe_symbol(sym: str) -> bool:
    if not isinstance(sym, str):
        return False
    if not sym.isascii():
        return False
    return re.match(r"^[A-Z0-9]{2,20}/USDT$", sym) is not None

def strategy_ar(s: str) -> str:
    return {
        "BREAKOUT": "اختراق",
        "PULLBACK": "ارتداد",
        "VOL_EXPANSION": "انفجار تذبذب",
        "MEAN_REVERSION": "عودة للمتوسط",
        "AI_ONLY": "ذكاء اصطناعي فقط",
    }.get(s, s)

def tier_ar(t: str) -> str:
    return {"NORMAL": "عادي", "STRONG": "قوي", "ULTRA": "فائق"}.get(t, t)

def regime_ar(r: str) -> str:
    return {"TREND": "اتجاه", "RANGE": "تذبذب", "HIGH_VOL": "تذبذب عالي"}.get(r, r)

def time_to_rise_ar(regime: str, tier: str, prob: float = 0.0, confirm_ok: bool = False, pump_score: float = 0.0) -> str:
    regime = (regime or "").upper()
    tier = (tier or "").upper()
    prob = float(prob or 0.0)
    pump_score = float(pump_score or 0.0)

    if tier == "ULTRA" or prob >= 0.92:
        base = (1, 4)
    elif tier == "STRONG" or prob >= 0.88:
        base = (3, 8)
    elif regime == "TREND":
        base = (6, 12)
    elif regime == "HIGH_VOL":
        base = (8, 16)
    else:
        base = (10, 18)

    lo, hi = base
    if confirm_ok:
        lo = max(1, lo - 1)
        hi = max(lo + 1, hi - 2)
    if pump_score >= 0.75:
        lo = max(1, lo - 1)
        hi = max(lo + 1, hi - 2)

    return f"{lo} - {hi} ساعة"

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def compute_analysis_scores(features: dict) -> dict:
    rsi = float(features.get("rsi", 50.0))
    vol_surge = float(features.get("vol_surge", 1.0))
    vol_ratio = float(features.get("vol_ratio", 1.0))
    dist_from_high = float(features.get("dist_from_high", 0.0))
    bb_width = float(features.get("bb_width", 0.0))
    atr_pct = float(features.get("atr_pct", 0.0))

    accumulation = clamp01(1.0 - abs(dist_from_high) * 4.0)
    accumulation = clamp01(accumulation * (1.0 - clamp01(bb_width / 20.0)))

    vol_mom = clamp01((max(vol_surge, vol_ratio) - 0.9) / 0.8)
    tech_setup = clamp01(1.0 - abs(rsi - 50.0) / 35.0)
    market_pos = clamp01(1.0 - abs(dist_from_high) * 3.0)

    if atr_pct > 0.02:
        accumulation *= 0.85
        market_pos *= 0.85

    return {
        "accumulation": accumulation,
        "vol_mom": vol_mom,
        "tech_setup": tech_setup,
        "market_pos": market_pos,
    }

def detected_patterns(strategy: str, regime: str, features: dict) -> list[str]:
    pats = []
    if float(features.get("vol_surge", 1.0)) >= 1.3:
        pats.append("📈 تباعد حجْمي إيجابي")
    if strategy == "MEAN_REVERSION":
        pats.append("🌀 ارتداد من منطقة تشبع/بولنجر")
    if strategy == "BREAKOUT":
        pats.append("🚀 اختراق محتمل مع دعم حجم")
    if strategy == "VOL_EXPANSION":
        pats.append("🌪️ تقلص بولنجر ثم توسع (انفجار تذبذب)")
    if regime == "HIGH_VOL":
        pats.append("⚡ تقلب مرتفع (التزم بالوقف)")
    if not pats:
        pats.append("📊 لا توجد أنماط إضافية واضحة")
    return pats

def should_send_symbol(symbol: str) -> bool:
    now = time.time()
    last_ts = LAST_SENT_BY_SYMBOL.get(symbol)
    if last_ts and (now - last_ts) < DEDUP_SECONDS:
        return False
    LAST_SENT_BY_SYMBOL[symbol] = now
    return True

def parse_time(s: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

# =========================
# Subscribers persistence (DB settings)
# =========================
def load_subscribers_from_db():
    global SUBSCRIBERS
    raw = db.get_setting(SUBSCRIBERS_SETTINGS_KEY) or ""
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except Exception:
            continue
    SUBSCRIBERS = ids

def save_subscribers_to_db():
    raw = ",".join(str(int(x)) for x in sorted(SUBSCRIBERS))
    db.set_setting(SUBSCRIBERS_SETTINGS_KEY, raw)

def add_subscriber(chat_id: int):
    SUBSCRIBERS.add(int(chat_id))
    save_subscribers_to_db()

def remove_subscriber(chat_id: int):
    if int(chat_id) in SUBSCRIBERS:
        SUBSCRIBERS.remove(int(chat_id))
        save_subscribers_to_db()

# =========================
# Settings
# =========================
def get_settings():
    prob_threshold = float(db.get_setting("prob_threshold") or "0.85")
    rr_min = float(db.get_setting("rr_min") or "1.30")
    cooldown_minutes = int(float(db.get_setting("cooldown_minutes") or "60"))
    max_open_trades = int(float(db.get_setting("max_open_trades") or "1000"))

    adaptive_enabled = (db.get_setting("adaptive_enabled") or "0") == "1"
    adaptive_window = int(float(db.get_setting("adaptive_window") or "30"))
    low_wr = float(db.get_setting("adaptive_low_wr") or "0.45")
    high_wr = float(db.get_setting("adaptive_high_wr") or "0.60")
    step = float(db.get_setting("adaptive_step") or "0.02")

    prob_min = float(db.get_setting("prob_min") or "0.68")
    prob_max = float(db.get_setting("prob_max") or "0.90")

    min_stop_pct = float(db.get_setting("min_stop_pct") or "0.0030")
    atr_stop_mult = float(db.get_setting("atr_stop_mult") or "1.50")
    max_stop_pct = float(db.get_setting("max_stop_pct") or "0.0200")

    tp1_rr = float(db.get_setting("tp1_rr") or "1.0")
    tp2_rr = float(db.get_setting("tp2_rr") or "2.0")
    tp3_rr = float(db.get_setting("tp3_rr") or "2.4")

    trend_filter_enabled = (db.get_setting("trend_filter_enabled") or "1") == "1"
    symbol_cooldown_minutes = int(float(db.get_setting("symbol_cooldown_minutes") or "30"))

    strong_prob = float(db.get_setting("strong_prob") or "0.88")
    ultra_prob = float(db.get_setting("ultra_prob") or "0.92")

    btc_filter_enabled = (db.get_setting("btc_filter_enabled") or "1") == "1"
    results_broadcast_enabled = (db.get_setting("results_broadcast_enabled") or "0") == "1"

    return {
        "prob_threshold": prob_threshold,
        "rr_min": rr_min,
        "cooldown_minutes": cooldown_minutes,
        "max_open_trades": max_open_trades,

        "adaptive_enabled": adaptive_enabled,
        "adaptive_window": adaptive_window,
        "low_wr": low_wr,
        "high_wr": high_wr,
        "step": step,
        "prob_min": prob_min,
        "prob_max": prob_max,

        "min_stop_pct": min_stop_pct,
        "atr_stop_mult": atr_stop_mult,
        "max_stop_pct": max_stop_pct,

        "tp1_rr": tp1_rr,
        "tp2_rr": tp2_rr,
        "tp3_rr": tp3_rr,

        "trend_filter_enabled": trend_filter_enabled,
        "symbol_cooldown_minutes": symbol_cooldown_minutes,

        "strong_prob": strong_prob,
        "ultra_prob": ultra_prob,

        "btc_filter_enabled": btc_filter_enabled,
        "results_broadcast_enabled": results_broadcast_enabled,
    }

# =========================
# Data + Features
# =========================
def fetch_ohlcv(symbol: str, timeframe: str = TIMEFRAME, limit: int = LIMIT) -> pd.DataFrame:
    ohlcv = EX.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rsi"] = RSIIndicator(df["close"], 14).rsi()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()

    bb = BollingerBands(df["close"], 20, 2)
    df["bb_width"] = bb.bollinger_wband()
    df["bb_lband"] = bb.bollinger_lband()
    df["bb_uband"] = bb.bollinger_hband()

    df["atr_pct"] = df["atr"] / df["close"]
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_10"] = df["close"].pct_change(10)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    df["recent_high_20"] = df["high"].rolling(20).max()
    df["dist_from_high"] = (df["close"] - df["recent_high_20"]) / df["recent_high_20"]

    df["mom_5"] = df["close"].pct_change(5)
    df["mom_5_acc"] = df["mom_5"] - df["mom_5"].shift(1)

    df["vol_surge"] = df["volume"] / df["volume"].rolling(50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    df.dropna(inplace=True)
    return df

def ai_probability(last_row: pd.Series) -> float:
    x = np.array([[f(last_row[c]) for c in FEATURES]], dtype=float)
    return float(MODEL.predict_proba(x)[0][1])

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

def btc_market_filter() -> tuple[bool, dict]:
    try:
        df4 = add_features(fetch_ohlcv("BTC/USDT", timeframe=CONFIRM_TIMEFRAME, limit=260))
        last4 = df4.iloc[-1]
        close4 = float(last4["close"])
        ema200_4h = float(last4["ema200"])
        rsi4 = float(last4["rsi"])
        mom4 = float(last4["mom_5"])
        regime4 = detect_regime(df4)

        ok = not (close4 < ema200_4h and rsi4 < 40 and mom4 < 0)
        info = {
            "btc_ok": ok,
            "btc_regime": regime4,
            "btc_close_4h": close4,
            "btc_ema200_4h": ema200_4h,
            "btc_rsi_4h": rsi4,
            "btc_mom_5_4h": mom4,
        }
        return ok, info
    except Exception:
        return True, {
            "btc_ok": True,
            "btc_regime": "UNKNOWN",
            "btc_close_4h": None,
            "btc_ema200_4h": None,
            "btc_rsi_4h": None,
            "btc_mom_5_4h": None,
        }

def confirm_trend_4h(symbol: str) -> tuple[bool, dict]:
    try:
        df4 = add_features(fetch_ohlcv(symbol, timeframe=CONFIRM_TIMEFRAME, limit=260))
        last4 = df4.iloc[-1]
        close4 = float(last4["close"])
        ema200_4h = float(last4["ema200"])
        rsi4 = float(last4["rsi"])
        mom4 = float(last4["mom_5"])
        regime4 = detect_regime(df4)
        ok = (close4 >= ema200_4h and rsi4 < 72 and mom4 > -0.03)
        info = {
            "confirm_ok": ok,
            "regime_4h": regime4,
            "close_4h": close4,
            "ema200_4h": ema200_4h,
            "rsi_4h": rsi4,
            "mom_5_4h": mom4,
        }
        return ok, info
    except Exception:
        return False, {
            "confirm_ok": False,
            "regime_4h": "UNKNOWN",
            "close_4h": None,
            "ema200_4h": None,
            "rsi_4h": None,
            "mom_5_4h": None,
        }

def pump_detector(features: dict, regime: str, strategy: str) -> dict:
    vol_surge = float(features.get("vol_surge", 1.0))
    vol_ratio = float(features.get("vol_ratio", 1.0))
    mom_5 = float(features.get("mom_5", 0.0))
    mom_5_acc = float(features.get("mom_5_acc", 0.0))
    dist_from_high = abs(float(features.get("dist_from_high", 0.0)))
    bb_width = float(features.get("bb_width", 0.0))
    atr_pct = float(features.get("atr_pct", 0.0))

    score = 0.0
    score += clamp01((vol_surge - 1.05) / 0.8) * 0.35
    score += clamp01((vol_ratio - 1.0) / 0.7) * 0.20
    score += clamp01((mom_5 - 0.005) / 0.03) * 0.15
    score += clamp01((mom_5_acc + 0.01) / 0.04) * 0.10
    score += clamp01((0.08 - dist_from_high) / 0.08) * 0.10
    score += clamp01((0.025 - atr_pct) / 0.02) * 0.05
    score += clamp01((15.0 - bb_width) / 15.0) * 0.05

    if strategy == "BREAKOUT":
        score += 0.08
    if regime == "TREND":
        score += 0.06
    elif regime == "HIGH_VOL":
        score -= 0.05

    score = clamp01(score)
    explosive = score >= 0.62
    label = "HIGH" if score >= 0.78 else ("MEDIUM" if score >= 0.62 else "LOW")
    return {"pump_score": score, "pump_label": label, "explosive": explosive}

def candidate_score(prob: float, rr2: float, rr3: float, regime: str, tier: str, confirm_ok: bool, pump_score: float, strategy: str) -> float:
    tier_bonus = {"NORMAL": 0.0, "STRONG": 3.0, "ULTRA": 6.0}.get(tier, 0.0)
    regime_bonus = {"TREND": 2.5, "RANGE": 0.0, "HIGH_VOL": -1.8}.get(regime, 0.0)
    confirm_bonus = 4.0 if confirm_ok else -2.5
    strategy_bonus = {
        "BREAKOUT": 0.5,
        "PULLBACK": 1.2,
        "VOL_EXPANSION": 1.8,
        "MEAN_REVERSION": -1.0,
        "AI_ONLY": 0.0,
    }.get(strategy, 0.0)

    return (
        prob * 100.0
        + rr2 * 7.0
        + min(rr3, 3.0) * 2.0
        + tier_bonus
        + regime_bonus
        + confirm_bonus
        + strategy_bonus
        + pump_score * 12.0
    )

# =========================
# Universe selection
# =========================
def get_spot_usdt_symbols():
    markets = EX.load_markets()
    out = []
    for m in markets.values():
        if m.get("spot") is True and m.get("active") is True and m.get("quote") == QUOTE:
            sym = m.get("symbol")
            if sym and is_safe_symbol(sym):
                out.append(sym)
    return out

def get_top_liquid_symbols(n: int):
    now = time.time()
    if LIQ_CACHE["symbols"] and (now - LIQ_CACHE["ts"]) < LIQ_CACHE_TTL:
        return LIQ_CACHE["symbols"][:n]

    syms = get_spot_usdt_symbols()
    sample = syms[:min(len(syms), n)]
    rows = []
    chunk = 80

    for i in range(0, len(sample), chunk):
        part = sample[i:i + chunk]
        tickers = {}
        try:
            tickers = EX.fetch_tickers(part)
        except Exception:
            for s in part:
                try:
                    tickers[s] = EX.fetch_ticker(s)
                except Exception:
                    pass

        for s in part:
            t = tickers.get(s)
            if not t:
                continue
            qv = f(t.get("quoteVolume"), 0.0)
            if qv < 300_000:
                continue
            rows.append((s, qv))

    rows.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, _ in rows[:n]]

    LIQ_CACHE["symbols"] = top
    LIQ_CACHE["ts"] = now
    return top

# =========================
# Strategies (labeling only)
# =========================
def strategy_breakout(df: pd.DataFrame):
    last = df.iloc[-1]
    recent_high = df["high"].iloc[-21:-1].max()
    if float(last["close"]) > float(recent_high) and float(last["vol_surge"]) >= 1.3:
        return "BREAKOUT"
    return None

def strategy_pullback(df: pd.DataFrame):
    last = df.iloc[-1]
    sma50 = df["close"].rolling(50).mean().iloc[-1]
    if float(last["close"]) > float(sma50) and 40 <= float(last["rsi"]) <= 56:
        return "PULLBACK"
    return None

def strategy_vol_expansion(df: pd.DataFrame):
    last = df.iloc[-1]
    avg_width = df["bb_width"].rolling(50).mean().iloc[-1]
    if float(last["bb_width"]) < float(avg_width) * 0.7:
        return "VOL_EXPANSION"
    return None

def strategy_mean_reversion(df: pd.DataFrame):
    last = df.iloc[-1]
    if float(last["close"]) < float(last["bb_lband"]) and float(last["rsi"]) < 32:
        return "MEAN_REVERSION"
    return None

STRATEGIES = [strategy_breakout, strategy_pullback, strategy_vol_expansion, strategy_mean_reversion]

# =========================
# Cooldown & adaptive
# =========================
def in_cooldown(symbol: str, cooldown_minutes: int) -> bool:
    last_time = db.last_trade_time(symbol)
    if not last_time:
        return False
    try:
        t = dt.datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")
        return (dt.datetime.now() - t).total_seconds() < cooldown_minutes * 60
    except Exception:
        return False

def adaptive_update_threshold():
    s = get_settings()
    if not s["adaptive_enabled"]:
        return

    recent = db.get_recent_closed(s["adaptive_window"])
    if len(recent) < max(10, s["adaptive_window"] // 3):
        return

    wins = sum(1 for r in recent if r["result"] == "WIN")
    wr = wins / len(recent)

    cur = s["prob_threshold"]
    new = cur

    if wr < s["low_wr"]:
        new = min(s["prob_max"], cur + s["step"])
    elif wr > s["high_wr"]:
        new = max(s["prob_min"], cur - s["step"])

    if abs(new - cur) >= 1e-9:
        db.set_setting("prob_threshold", f"{new:.2f}")

# =========================
# Dynamic levels
# =========================
def compute_smart_levels(df: pd.DataFrame, entry: float, settings: dict):
    last = df.iloc[-1]
    atr_pct = float(last["atr_pct"])
    bb_width = float(last["bb_width"])
    mom_5 = float(last["mom_5"])
    regime = detect_regime(df)

    base = atr_pct * float(settings["atr_stop_mult"])
    bb_component = clamp01(bb_width / 20.0) * 0.6 * atr_pct

    mom_abs = abs(mom_5)
    mom_component = min(mom_abs * 2.5, 0.015)

    regime_mult = {"TREND": 1.05, "RANGE": 0.95, "HIGH_VOL": 1.10}.get(regime, 1.0)
    stop_pct = (base + bb_component + mom_component) * regime_mult

    stop_pct = max(stop_pct, float(settings["min_stop_pct"]))
    max_stop = float(settings["max_stop_pct"])
    if max_stop > 0:
        stop_pct = min(stop_pct, max_stop)

    stop = entry * (1.0 - stop_pct)

    rr1 = float(settings["tp1_rr"])
    rr2 = float(settings["tp2_rr"])
    rr3 = float(settings["tp3_rr"])

    if regime == "TREND":
        rr2 *= 1.03
        rr3 *= 1.05
    elif regime == "HIGH_VOL":
        rr1 *= 0.95
        rr2 *= 0.90
        rr3 *= 0.85

    if mom_abs > 0.02:
        rr3 *= 1.03

    rr3 = max(1.4, min(rr3, 3.0))
    rr2 = max(1.1, min(rr2, rr3 - 0.2))
    rr1 = max(0.8, min(rr1, rr2 - 0.2))

    tp1 = entry * (1.0 + stop_pct * rr1)
    tp2 = entry * (1.0 + stop_pct * rr2)
    tp3 = entry * (1.0 + stop_pct * rr3)

    return stop_pct, stop, tp1, tp2, tp3, rr1, rr2, rr3, regime

def prob_tier(prob: float, settings: dict) -> str:
    if prob >= float(settings["ultra_prob"]):
        return "ULTRA"
    if prob >= float(settings["strong_prob"]):
        return "STRONG"
    return "NORMAL"

# =========================
# OHLC-based open trade tracking
# =========================
def _recent_candles_since(symbol: str, t_open: dt.datetime | None):
    limit = max(12, OPEN_TRADE_EXPIRY_HOURS + 8)
    ohlcv = EX.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
    if not ohlcv:
        return []
    rows = []
    for candle in ohlcv:
        c_ts = dt.datetime.fromtimestamp(candle[0] / 1000.0)
        if t_open is None or c_ts >= t_open:
            rows.append({
                "ts": c_ts,
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
            })
    return rows

def update_open_trades() -> list:
    open_trades = db.get_open_trades()
    closed_now = []
    now = dt.datetime.now()

    for tr in open_trades:
        trade_id = int(tr["id"])
        symbol = tr["symbol"]

        try:
            t_open = parse_time(tr["time_open"])
            age = (now - t_open).total_seconds() if t_open is not None else 0

            entry = float(tr["entry"])
            stop = float(tr["stop"])
            tp1 = float(tr["tp1"]) if tr["tp1"] is not None else None
            tp2 = float(tr["tp2"]) if tr["tp2"] is not None else None
            tp3 = float(tr["tp3"]) if tr["tp3"] is not None else None

            candles = _recent_candles_since(symbol, t_open)
            if not candles:
                continue

            hit_tp1 = int(tr["hit_tp1"]) if tr["hit_tp1"] is not None else 0
            hit_tp2 = int(tr["hit_tp2"]) if tr["hit_tp2"] is not None else 0
            hit_tp3 = int(tr["hit_tp3"]) if tr["hit_tp3"] is not None else 0
            max_tp_hit = int(tr["max_tp_hit"]) if tr["max_tp_hit"] is not None else 0

            last_close = candles[-1]["close"]

            for candle in candles:
                high = candle["high"]
                low = candle["low"]

                fields = {}
                if tp1 is not None and high >= tp1 and hit_tp1 == 0:
                    fields["hit_tp1"] = 1
                    hit_tp1 = 1
                    max_tp_hit = max(max_tp_hit, 1)

                if tp2 is not None and high >= tp2 and hit_tp2 == 0:
                    fields["hit_tp2"] = 1
                    hit_tp2 = 1
                    max_tp_hit = max(max_tp_hit, 2)

                if tp3 is not None and high >= tp3 and hit_tp3 == 0:
                    fields["hit_tp3"] = 1
                    hit_tp3 = 1
                    max_tp_hit = max(max_tp_hit, 3)

                if fields:
                    fields["max_tp_hit"] = max_tp_hit
                    db.update_trade_fields(trade_id, fields)

                if tp3 is not None and high >= tp3:
                    pnl_pct = (tp3 - entry) / entry * 100.0
                    db.close_trade(trade_id, now_str(), "WIN", tp3, pnl_usdt=None, pnl_pct=pnl_pct, close_reason="TP3")
                    closed_now.append({"symbol": symbol, "result": "WIN", "close_reason": "TP3", "pnl_pct": pnl_pct})
                    break

                if low <= stop:
                    pnl_pct = (stop - entry) / entry * 100.0
                    db.close_trade(trade_id, now_str(), "LOSS", stop, pnl_usdt=None, pnl_pct=pnl_pct, close_reason="STOP")
                    closed_now.append({"symbol": symbol, "result": "LOSS", "close_reason": "STOP", "pnl_pct": pnl_pct})
                    break
            else:
                if age >= OPEN_TRADE_EXPIRY_SECONDS:
                    if max_tp_hit >= 3 and tp3 is not None:
                        pnl_pct = (tp3 - entry) / entry * 100.0
                        db.close_trade(trade_id, now_str(), "WIN", tp3, pnl_usdt=None, pnl_pct=pnl_pct, close_reason="TP3")
                        closed_now.append({"symbol": symbol, "result": "WIN", "close_reason": "TP3", "pnl_pct": pnl_pct})

                    elif max_tp_hit == 2 and tp2 is not None:
                        pnl_pct = (tp2 - entry) / entry * 100.0
                        db.close_trade(trade_id, now_str(), "WIN", tp2, pnl_usdt=None, pnl_pct=pnl_pct, close_reason="TP2")
                        closed_now.append({"symbol": symbol, "result": "WIN", "close_reason": "TP2", "pnl_pct": pnl_pct})

                    elif max_tp_hit == 1 and tp1 is not None:
                        pnl_pct = (tp1 - entry) / entry * 100.0
                        db.close_trade(trade_id, now_str(), "WIN", tp1, pnl_usdt=None, pnl_pct=pnl_pct, close_reason="TP1")
                        closed_now.append({"symbol": symbol, "result": "WIN", "close_reason": "TP1", "pnl_pct": pnl_pct})

                    else:
                        pnl_pct_now = (last_close - entry) / entry * 100.0
                        if pnl_pct_now <= -abs(DRAW_MAX_LOSS_PCT):
                            db.close_trade(trade_id, now_str(), "LOSS", last_close, pnl_usdt=None, pnl_pct=pnl_pct_now, close_reason="TIMEOUT")
                            closed_now.append({"symbol": symbol, "result": "LOSS", "close_reason": "TIMEOUT", "pnl_pct": pnl_pct_now})
                        else:
                            db.close_trade(trade_id, now_str(), "DRAW", last_close, pnl_usdt=None, pnl_pct=pnl_pct_now, close_reason="DRAW")
                            closed_now.append({"symbol": symbol, "result": "DRAW", "close_reason": "DRAW", "pnl_pct": pnl_pct_now})

        except Exception as e:
            print("update_open_trades error:", symbol, repr(e))
            continue

    if closed_now:
        adaptive_update_threshold()

    return closed_now

def _format_lines_with_total(rows: list, icon: str, title: str) -> str:
    if not rows:
        return f"{icon} {title}: لا يوجد"
    total = sum(float(x.get("pnl_pct", 0.0)) for x in rows)
    parts = [f"{icon} {title}:"]
    for x in rows:
        sym = fmt_symbol(x.get("symbol", ""))
        pnl = float(x.get("pnl_pct", 0.0))
        parts.append(f"{sym} | {pnl:+.2f}%")
    parts.append(f"المجموع الكلي: {total:+.2f}%")
    return "\n".join(parts)

def format_closed_report(closed_list: list, title: str = "📌 تقرير النتائج") -> str:
    if not closed_list:
        return ""

    wins = [x for x in closed_list if (x.get("result") or "").upper() == "WIN"]
    losses = [x for x in closed_list if (x.get("result") or "").upper() == "LOSS"]
    draws = [x for x in closed_list if (x.get("result") or "").upper() == "DRAW"]

    parts = [title]
    parts.append(_format_lines_with_total(wins, "✅", "الصفقات الرابحة"))
    parts.append("")
    parts.append(_format_lines_with_total(losses, "❌", "الصفقات الخاسرة"))
    if draws:
        parts.append("")
        parts.append(_format_lines_with_total(draws, "➖", "الصفقات المتعادلة"))
    return "\n".join(parts)

def _parse_dt(s: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def retrain_model_from_db_if_due() -> str:
    if not RETRAIN_ENABLED:
        return ""

    last = db.get_setting(RETRAIN_LAST_KEY)
    now = dt.datetime.now()
    if last:
        last_dt = _parse_dt(last)
        if last_dt is not None:
            hours = (now - last_dt).total_seconds() / 3600.0
            if hours < RETRAIN_EVERY_HOURS:
                return ""

    conn = sqlite3.connect("bot.db")
    q = f"""
        SELECT
            {','.join(FEATURES)},
            close_reason
        FROM trades
        WHERE status='CLOSED'
          AND close_reason IS NOT NULL
          AND ({' AND '.join([f'{c} IS NOT NULL' for c in FEATURES])})
        ORDER BY time_close ASC
    """
    df = pd.read_sql_query(q, conn)
    conn.close()

    if df.empty:
        return ""

    reason = df["close_reason"].astype(str).str.upper()
    keep = reason.isin(["TP2", "TP3", "STOP"])
    df2 = df.loc[keep].copy()
    if len(df2) < RETRAIN_MIN_SAMPLES:
        return ""

    y = reason.loc[keep].isin(["TP2", "TP3"]).astype(int)
    X = df2[FEATURES].astype(float)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = (n_neg / max(n_pos, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    m = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=spw,
        reg_lambda=1.0,
        n_jobs=4,
        eval_metric="logloss",
    )
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    report = classification_report(y_test, pred, zero_division=0)

    m.save_model("ai_model_tp_sl_bot_1h.json")
    try:
        MODEL.load_model("ai_model_tp_sl_bot_1h.json")
    except Exception:
        pass

    db.set_setting(RETRAIN_LAST_KEY, now_str())

    return (
        "🧠 تم إعادة تدريب النموذج من الصفقات المغلقة.\n"
        f"• samples={len(df2)} | pos={n_pos} | neg={n_neg}\n"
        "\n" + report
    )

# =========================
# Analytics helpers
# =========================
def get_closed_rows_from_db(limit: int = 30):
    conn = sqlite3.connect("bot.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT symbol, result, pnl_pct, close_reason, time_close
        FROM trades
        WHERE status='CLOSED'
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return rows

def get_closed_rows_today():
    conn = sqlite3.connect("bot.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    today = dt.datetime.now().strftime("%Y-%m-%d")
    rows = cur.execute(
        """
        SELECT symbol, result, pnl_pct, close_reason, time_close
        FROM trades
        WHERE status='CLOSED'
          AND time_close IS NOT NULL
          AND substr(time_close, 1, 10) = ?
        ORDER BY id DESC
        """,
        (today,),
    ).fetchall()
    conn.close()
    return rows

def _to_closed_list(rows):
    return [
        {
            "symbol": r["symbol"],
            "result": r["result"],
            "pnl_pct": float(r["pnl_pct"] or 0.0),
            "close_reason": r["close_reason"],
        }
        for r in rows
    ]

def coin_rankings(limit: int = 10, reverse: bool = True, min_trades: int = 3) -> str:
    conn = sqlite3.connect("bot.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT
            symbol,
            COUNT(*) as trades,
            SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) as losses,
            SUM(COALESCE(pnl_pct, 0)) as total_pnl
        FROM trades
        WHERE status='CLOSED'
        GROUP BY symbol
        HAVING COUNT(*) >= ?
        """,
        (min_trades,),
    ).fetchall()
    conn.close()

    stats = []
    for r in rows:
        trades = int(r["trades"] or 0)
        wins = int(r["wins"] or 0)
        wr = wins / trades if trades else 0.0
        stats.append({
            "symbol": r["symbol"],
            "trades": trades,
            "wins": wins,
            "losses": int(r["losses"] or 0),
            "win_rate": wr,
            "total_pnl": float(r["total_pnl"] or 0.0),
        })

    stats.sort(key=lambda x: (x["win_rate"], x["total_pnl"]), reverse=reverse)
    chosen = stats[:limit]

    title = "🏆 أفضل العملات" if reverse else "⚠️ أسوأ العملات"
    if not chosen:
        return f"{title}\n\nلا توجد بيانات كافية."

    parts = [title]
    for x in chosen:
        parts.append(
            f"{fmt_symbol(x['symbol'])} | WinRate: {x['win_rate'] * 100:.1f}% | "
            f"Trades: {x['trades']} | PnL: {x['total_pnl']:+.2f}%"
        )
    return "\n".join(parts)

# =========================
# Scan best 1
# =========================
def scan_best1():
    s = get_settings()

    btc_ok = True
    btc_info = {}
    if s["btc_filter_enabled"]:
        btc_ok, btc_info = btc_market_filter()
        if not btc_ok:
            print("BTC filter soft-gating market:", btc_info)

    open_count = db.count_open_trades()
    if open_count >= s["max_open_trades"]:
        print("Max open reached:", open_count, "/", s["max_open_trades"], "| continue scanning (no stop).")

    symbols = get_top_liquid_symbols(TOP_SYMBOLS)
    print("Scanning symbols:", len(symbols), "| threshold:", s["prob_threshold"], "| shortlist:", SHORTLIST_SIZE)

    candidates = []

    for symbol in symbols:
        try:
            if db.has_open_trade(symbol):
                continue
        except Exception:
            pass

        if in_cooldown(symbol, s["symbol_cooldown_minutes"]):
            continue

        try:
            df = add_features(fetch_ohlcv(symbol))
            last = df.iloc[-1]

            entry = float(last["close"])
            atr_pct = float(last["atr_pct"])

            if atr_pct < float(s["min_stop_pct"]):
                continue

            if s["trend_filter_enabled"]:
                ema200 = float(last["ema200"])
                if entry < ema200:
                    continue

            rsi = float(last["rsi"])
            if rsi > 65:
                continue

            prob = ai_probability(last)
            if prob < s["prob_threshold"]:
                continue

            strat_name = None
            for strat_fn in STRATEGIES:
                sn = strat_fn(df)
                if sn:
                    strat_name = sn
                    break
            if not strat_name:
                strat_name = "AI_ONLY"

            regime = detect_regime(df)
            if regime == "HIGH_VOL" and strat_name == "MEAN_REVERSION":
                continue

            stop_pct, stop, tp1, tp2, tp3, rr1, rr2, rr3, regime2 = compute_smart_levels(df, entry, s)
            regime = regime2

            if rr3 < float(s["rr_min"]):
                continue

            tier = prob_tier(prob, s)
            features = {k: f(last[k]) for k in FEATURES}
            pump = pump_detector(features, regime, strat_name)
            confirm_ok, confirm_info = confirm_trend_4h(symbol)

            # Soft gate حسب حالة BTC
            if (not btc_ok) and prob < float(s["ultra_prob"]) and not pump["explosive"]:
                continue

            if (not confirm_ok) and prob < max(float(s["strong_prob"]), float(s["prob_threshold"]) + 0.03) and not pump["explosive"]:
                continue

            score = candidate_score(prob, rr2, rr3, regime, tier, confirm_ok, pump["pump_score"], strat_name)

            candidates.append({
                "symbol": symbol,
                "strategy": strat_name,
                "entry": entry,
                "stop": float(stop),
                "tp1": float(tp1),
                "tp2": float(tp2),
                "tp3": float(tp3),
                "tp": float(tp3),
                "stop_pct": float(stop_pct),
                "rr": float(rr3),
                "rr1": float(rr1),
                "rr2": float(rr2),
                "rr3": float(rr3),
                "prob": float(prob),
                "tier": tier,
                "regime": regime,
                "score": score,
                "confirm_ok": bool(confirm_ok),
                "pump_score": float(pump["pump_score"]),
                "pump_label": pump["pump_label"],
                "regime_4h": confirm_info.get("regime_4h"),
                "features": features,
            })

        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)
    shortlist = candidates[:SHORTLIST_SIZE]

    shortlist.sort(
        key=lambda x: (
            x["confirm_ok"],
            x["tier"] == "ULTRA",
            x["pump_score"],
            x["prob"],
            x["rr2"],
            x["score"],
        ),
        reverse=True,
    )
    return shortlist[0]

# =========================
# Message format
# =========================
def format_signal(sig: dict) -> str:
    entry = float(sig["entry"])
    stop = float(sig["stop"])
    tp1 = float(sig.get("tp1", sig.get("tp")))
    tp2 = float(sig.get("tp2", sig.get("tp")))
    tp3 = float(sig.get("tp3", sig.get("tp")))

    symbol = fmt_symbol(sig["symbol"])
    tier = (sig.get("tier") or "NORMAL")
    regime = (sig.get("regime") or "RANGE")
    confirm_ok = bool(sig.get("confirm_ok", False))
    eta = time_to_rise_ar(regime, tier, float(sig.get("prob", 0.0)), confirm_ok, float(sig.get("pump_score", 0.0)))
    confidence_pct = float(sig.get("prob", 0.0)) * 100.0
    confirm_txt = "✅ متوافق" if confirm_ok else "⚠️ ضعيف"
    pump_txt = sig.get("pump_label") or "LOW"

    return (
        f"💎 {symbol}\n\n"
        f"📍 الدخول: {entry:.6f}\n"
        f"🛑 وقف الخسارة: {stop:.6f}\n\n"
        f"🎯 الهدف 1: {tp1:.6f}\n"
        f"🎯 الهدف 2: {tp2:.6f}\n"
        f"🎯 الهدف 3: {tp3:.6f}\n\n"
        f"🧠 الثقة: {confidence_pct:.1f}%\n"
        f"📈 تأكيد 4h: {confirm_txt}\n"
        f"🚀 Pump Score: {pump_txt} ({float(sig.get('pump_score', 0.0)) * 100.0:.0f}%)\n"
        f"⏳ وقت الصعود المتوقع: {eta}\n"
    )

def persist_trade(sig: dict):
    try:
        if db.has_open_trade(sig["symbol"]):
            return
    except Exception:
        pass

    payload = {
        "time_open": now_str(),
        "symbol": sig["symbol"],
        "strategy": sig["strategy"],
        "timeframe": TIMEFRAME,
        "entry": sig["entry"],
        "stop": sig["stop"],
        "tp": sig["tp"],

        "rr": sig["rr"],
        "prob": sig["prob"],
        "status": "OPEN",
        "result": None,
        "close_price": None,

        "tp1": sig.get("tp1"),
        "tp2": sig.get("tp2"),
        "tp3": sig.get("tp3"),
        "hit_tp1": 0,
        "hit_tp2": 0,
        "hit_tp3": 0,
        "max_tp_hit": 0,
        "trail_stop": None,

        "tier": sig.get("tier"),
        "regime": sig.get("regime"),

        "qty": None,
        "risk_usdt": None,
        "pnl_usdt": None,
        "pnl_pct": None,
        "close_reason": None,

        **sig["features"]
    }
    db.insert_trade(payload)

# =========================
# Telegram Commands
# =========================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    add_subscriber(update.effective_chat.id)

    await update.message.reply_text(
    "✅ تم تفعيل نظام الإشارات.\n\n"
    f"• إغلاق تلقائي بعد {OPEN_TRADE_EXPIRY_HOURS} ساعة\n"
    "• تتبع OHLC للأهداف والوقف\n"
    "• هدف 1 / هدف 2 / هدف 3\n"
    "• النتائج لا تُرسل تلقائيًا إلا بطلبك\n"
    "• 1h Entry + 4h Confirmation + BTC Market Filter\n\n"
    "📘 شرح الأوامر:\n\n"
    "🔹 /subscribe\n"
    "للاشتراك في استقبال إشارات البوت داخل هذا الشات.\n\n"
    "🔹 /unsubscribe\n"
    "لإلغاء الاشتراك وإيقاف استقبال الإشارات لهذا الشات.\n\n"
    "🔹 /addchannel -100xxxxxxxxxx\n"
    "لإضافة قناة أو مجموعة إلى قائمة الإرسال باستخدام chat_id.\n\n"
    "🔹 /signal\n"
    "لفحص السوق الآن وإرسال أفضل فرصة متاحة حاليًا إذا وُجدت.\n\n"
    "🔹 /stats\n"
    "لعرض لوحة إحصائيات البوت مثل عدد الصفقات ونسبة الفوز ومتوسط RR.\n\n"
    "🔹 /results [30]\n"
    "لعرض آخر الصفقات المغلقة مع إجمالي الربح والخسارة. يمكن تغيير العدد مثل /results 50.\n\n"
    "🔹 /results_today\n"
    "لعرض نتائج الصفقات المغلقة الخاصة باليوم الحالي فقط.\n\n"
    "🔹 /topcoins\n"
    "لعرض أفضل العملات أداءً مع البوت بناءً على السجل الموجود في قاعدة البيانات.\n\n"
    "🔹 /worstcoins\n"
    "لعرض أسوأ العملات أداءً مع البوت لمعرفة العملات التي يجب الحذر منها.\n\n"
    "🔹 /settings\n"
    "لعرض الإعدادات الحالية للبوت مثل threshold و RR والفلترات المفعلة.\n\n"
    "🔹 /set ...\n"
    "لتعديل إعدادات البوت يدويًا مثل threshold و rr_min و btc filter وغيرها.\n"
)

async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    add_subscriber(cid)
    await update.message.reply_text(f"✅ تم الاشتراك بنجاح.\nchat_id = {cid}")

async def unsubscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    remove_subscriber(cid)
    await update.message.reply_text("✅ تم إلغاء الاشتراك لهذا الشات.")

async def addchannel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("استخدم: /addchannel -100xxxxxxxxxx")
        return
    raw = context.args[0].strip()
    try:
        cid = int(raw)
    except Exception:
        await update.message.reply_text("❌ chat_id غير صحيح.")
        return

    add_subscriber(cid)
    await update.message.reply_text(f"✅ تم إضافة القناة.\nchat_id = {cid}")

async def listsubs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not SUBSCRIBERS:
        await update.message.reply_text("لا يوجد مشتركين حاليا.")
        return
    txt = "📌 قائمة الإرسال الحالية:\n" + "\n".join([f"- {x}" for x in sorted(SUBSCRIBERS)])
    await update.message.reply_text(txt)

async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 جاري البحث عن أفضل فرصة...")
    async with SCAN_LOCK:
        def work():
            update_open_trades()
            sig = scan_best1()
            return sig

        sig = await asyncio.to_thread(work)

        if not sig:
            await update.message.reply_text("لا توجد فرصة قوية الآن حسب الشروط الحالية.")
            return

        if not should_send_symbol(sig["symbol"]):
            await update.message.reply_text(
                f"⚠️ تم تجاهل الإشارة: نفس العملة تم إرسالها خلال آخر {DEDUP_HOURS} ساعات."
            )
            return

        persist_trade(sig)
        await update.message.reply_text(format_signal(sig))

async def results_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    limit = 30
    if context.args:
        try:
            limit = max(1, min(200, int(context.args[0])))
        except Exception:
            limit = 30

    rows = get_closed_rows_from_db(limit)
    if not rows:
        await update.message.reply_text("لا توجد صفقات مغلقة حتى الآن.")
        return

    await update.message.reply_text(format_closed_report(_to_closed_list(rows), title=f"📌 تقرير النتائج (آخر {limit} صفقة)"))

async def results_today_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = get_closed_rows_today()
    if not rows:
        await update.message.reply_text("لا توجد صفقات مغلقة اليوم.")
        return
    await update.message.reply_text(format_closed_report(_to_closed_list(rows), title="📌 تقرير نتائج اليوم"))

async def topcoins_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(coin_rankings(limit=10, reverse=True, min_trades=3))

async def worstcoins_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(coin_rankings(limit=10, reverse=False, min_trades=3))

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await asyncio.to_thread(update_open_trades)
    st = db.get_stats()
    s = get_settings()

    wr = "—" if st["win_rate"] is None else f"{st['win_rate']:.3f}"
    avg_rr = "—" if st["avg_rr"] is None else f"{st['avg_rr']:.2f}"

    msg = (
        "📊 لوحة التحكم\n\n"
        f"🔔 Threshold: {s['prob_threshold']:.2f}\n"
        f"🧊 Cooldown per-symbol: {s['symbol_cooldown_minutes']} دقيقة\n"
        f"📌 max_open_trades: {s['max_open_trades']}\n"
        f"📈 Trend filter (EMA200): {'ON' if s['trend_filter_enabled'] else 'OFF'}\n"
        f"₿ BTC market filter: {'ON' if s['btc_filter_enabled'] else 'OFF'}\n"
        f"🧯 Dedup: {DEDUP_HOURS} ساعات | Expiry: {OPEN_TRADE_EXPIRY_HOURS} ساعة\n"
        f"📣 Subscribers: {len(SUBSCRIBERS)}\n\n"
        f"📦 إجمالي الصفقات: {st['total']}\n"
        f"🟨 مفتوحة: {st['open']}\n"
        f"🟩 مغلقة: {st['closed']}\n"
        f"✅ رابحة: {st['wins']} | ❌ خاسرة: {st['losses']} | ➖ تعادل: {st['draws']}\n"
        f"🎯 نسبة الفوز: {wr}\n"
        f"⚖️ متوسط RR: {avg_rr}\n"
    )
    await update.message.reply_text(msg)

async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = get_settings()
    msg = (
        "⚙️ الإعدادات الحالية\n\n"
        f"prob_threshold = {s['prob_threshold']:.2f}\n"
        f"rr_min = {s['rr_min']:.2f}\n"
        f"symbol_cooldown_minutes = {s['symbol_cooldown_minutes']}\n"
        f"max_open_trades = {s['max_open_trades']}\n"
        f"adaptive_enabled = {1 if s['adaptive_enabled'] else 0}\n\n"
        f"atr_stop_mult = {s['atr_stop_mult']:.2f}\n"
        f"min_stop_pct = {s['min_stop_pct']:.4f}\n"
        f"max_stop_pct = {s['max_stop_pct']:.4f}\n"
        f"tp1_rr = {s['tp1_rr']:.2f}\n"
        f"tp2_rr = {s['tp2_rr']:.2f}\n"
        f"tp3_rr = {s['tp3_rr']:.2f}\n"
        f"trend_filter_enabled = {1 if s['trend_filter_enabled'] else 0}\n"
        f"btc_filter_enabled = {1 if s['btc_filter_enabled'] else 0}\n"
        f"results_broadcast_enabled = {1 if s['results_broadcast_enabled'] else 0}\n\n"
        f"Subscribers (saved): {db.get_setting(SUBSCRIBERS_SETTINGS_KEY) or ''}\n"
    )
    await update.message.reply_text(msg)

async def set_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "استخدم: /set prob 0.85 أو /set symcool 30 أو /set maxopen 1000\n"
            "إضافي: /set atrmult 1.5 | /set tp1 1.0 | /set tp2 2.0 | /set tp3 2.4 | "
            "/set trend 1/0 | /set btcfilter 1/0 | /set resultsbroadcast 1/0 | "
            "/set rrmin 1.3 | /set minstop 0.003 | /set maxstop 0.02"
        )
        return

    key = context.args[0].lower()
    val = context.args[1]

    mapping = {
        "prob": "prob_threshold",
        "symcool": "symbol_cooldown_minutes",
        "maxopen": "max_open_trades",
        "adaptive": "adaptive_enabled",
        "atrmult": "atr_stop_mult",
        "minstop": "min_stop_pct",
        "maxstop": "max_stop_pct",
        "tp1": "tp1_rr",
        "tp2": "tp2_rr",
        "tp3": "tp3_rr",
        "trend": "trend_filter_enabled",
        "strong": "strong_prob",
        "ultra": "ultra_prob",
        "rrmin": "rr_min",
        "btcfilter": "btc_filter_enabled",
        "resultsbroadcast": "results_broadcast_enabled",
    }

    if key not in mapping:
        await update.message.reply_text("المفاتيح المتاحة: prob / symcool / maxopen / adaptive / atrmult / minstop / maxstop / tp1 / tp2 / tp3 / trend / strong / ultra / rrmin / btcfilter / resultsbroadcast")
        return

    if key == "maxstop":
        try:
            v = float(val)
            v = min(v, 0.0200)
            val = f"{v:.4f}"
        except Exception:
            val = "0.0200"

    db.set_setting(mapping[key], str(val))
    await update.message.reply_text(f"✅ تم تحديث الإعداد: {mapping[key]} = {val}")

# =========================
# Scheduled Job
# =========================
async def scheduled_job(context: ContextTypes.DEFAULT_TYPE):
    print("TICK", dt.datetime.utcnow())
    if not SUBSCRIBERS:
        return

    async with SCAN_LOCK:
        def work():
            closed_now = update_open_trades()
            retrain_msg = retrain_model_from_db_if_due()
            sig = scan_best1()
            return closed_now, retrain_msg, sig

        closed_now, retrain_msg, sig = await asyncio.to_thread(work)

        if retrain_msg:
            for cid in list(SUBSCRIBERS):
                try:
                    await context.bot.send_message(chat_id=cid, text=retrain_msg)
                except Exception:
                    pass

        s = get_settings()
        if closed_now and s["results_broadcast_enabled"]:
            rep = format_closed_report(closed_now)
            for cid in list(SUBSCRIBERS):
                try:
                    await context.bot.send_message(chat_id=cid, text=rep)
                except Exception:
                    pass

        if not sig:
            return

        if not should_send_symbol(sig["symbol"]):
            return

        persist_trade(sig)
        text = format_signal(sig)

        for cid in list(SUBSCRIBERS):
            try:
                await context.bot.send_message(chat_id=cid, text=text)
            except Exception:
                pass

# =========================
# Main
# =========================
def main():
    db.init_db()
    load_subscribers_from_db()

    db.set_setting("prob_threshold", db.get_setting("prob_threshold") or "0.85")
    db.set_setting("rr_min", db.get_setting("rr_min") or "1.30")
    db.set_setting("atr_stop_mult", db.get_setting("atr_stop_mult") or "1.50")
    db.set_setting("max_stop_pct", db.get_setting("max_stop_pct") or "0.0200")
    db.set_setting("tp1_rr", db.get_setting("tp1_rr") or "1.0")
    db.set_setting("tp2_rr", db.get_setting("tp2_rr") or "2.0")
    db.set_setting("tp3_rr", db.get_setting("tp3_rr") or "2.4")
    db.set_setting("trend_filter_enabled", db.get_setting("trend_filter_enabled") or "1")
    db.set_setting("symbol_cooldown_minutes", db.get_setting("symbol_cooldown_minutes") or "30")
    db.set_setting("strong_prob", db.get_setting("strong_prob") or "0.88")
    db.set_setting("ultra_prob", db.get_setting("ultra_prob") or "0.92")
    db.set_setting("max_open_trades", db.get_setting("max_open_trades") or "1000")
    db.set_setting("btc_filter_enabled", db.get_setting("btc_filter_enabled") or "1")
    db.set_setting("results_broadcast_enabled", db.get_setting("results_broadcast_enabled") or "0")

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("subscribe", subscribe_cmd))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe_cmd))
    app.add_handler(CommandHandler("addchannel", addchannel_cmd))
    app.add_handler(CommandHandler("listsubs", listsubs_cmd))
    app.add_handler(CommandHandler("signal", signal_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("results", results_cmd))
    app.add_handler(CommandHandler("results_today", results_today_cmd))
    app.add_handler(CommandHandler("topcoins", topcoins_cmd))
    app.add_handler(CommandHandler("worstcoins", worstcoins_cmd))
    app.add_handler(CommandHandler("settings", settings_cmd))
    app.add_handler(CommandHandler("set", set_cmd))

    if app.job_queue is None:
        raise RuntimeError("JobQueue غير متاح. ثبّت: pip install 'python-telegram-bot[job-queue]'")

    app.job_queue.run_repeating(
        scheduled_job,
        interval=ALERT_MINUTES * 60,
        first=10
    )

    print(f"✅ البوت يعمل الآن (broadcast subscribers={len(SUBSCRIBERS)}) ...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
