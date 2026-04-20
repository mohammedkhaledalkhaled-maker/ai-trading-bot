# db.py
import sqlite3
from typing import Optional, Dict, Any, List
from pathlib import Path
import datetime as dt

DB_PATH = Path("bot.db")


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(table: str, col: str, col_type: str):
    conn = connect()
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = {r["name"] for r in cur.fetchall()}
    if col not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
        conn.commit()
    conn.close()


def init_db():
    conn = connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time_open TEXT NOT NULL,
        time_close TEXT,
        symbol TEXT NOT NULL,
        strategy TEXT NOT NULL,
        timeframe TEXT NOT NULL,

        entry REAL NOT NULL,
        stop REAL NOT NULL,
        tp REAL NOT NULL,
        rr REAL NOT NULL,
        prob REAL NOT NULL,

        status TEXT NOT NULL,      -- OPEN / CLOSED
        result TEXT,               -- WIN / LOSS / DRAW / TIMEOUT
        close_price REAL,

        -- Features at entry (for future learning)
        rsi REAL,
        atr_pct REAL,
        bb_width REAL,
        ret_3 REAL,
        ret_10 REAL,
        vol_ratio REAL,
        dist_from_high REAL,
        mom_5 REAL,
        mom_5_acc REAL,
        vol_surge REAL
    )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, time_open)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")

    conn.commit()
    conn.close()

    # ============================
    # Defaults
    # ============================
    set_default("prob_threshold", "0.80")
    set_default("rr_min", "1.30")
    set_default("cooldown_minutes", "60")
    set_default("max_open_trades", "1000")   # حتى لا يوقف البوت عمليًا

    # Adaptive (اختياري)
    set_default("adaptive_enabled", "0")
    set_default("adaptive_window", "30")
    set_default("adaptive_low_wr", "0.45")
    set_default("adaptive_high_wr", "0.60")
    set_default("adaptive_step", "0.02")
    set_default("prob_min", "0.68")
    set_default("prob_max", "0.90")

    # ===== Scanner dynamic SL/TP settings =====
    set_default("min_stop_pct", "0.0030")        # 0.30%
    set_default("atr_stop_mult", "1.50")

    # ✅ سقف الوقف 2% (حسب طلبك)
    set_default("max_stop_pct", "0.0200")

    set_default("tp1_rr", "1.0")
    set_default("tp2_rr", "2.0")
    set_default("tp3_rr", "3.0")

    set_default("trend_filter_enabled", "1")     # EMA200
    set_default("symbol_cooldown_minutes", "30") # منع سبام نفس العملة

    set_default("strong_prob", "0.85")
    set_default("ultra_prob", "0.90")

    # ============================
    # Migrations / new columns
    # ============================
    ensure_column("trades", "qty", "REAL")
    ensure_column("trades", "risk_usdt", "REAL")

    ensure_column("trades", "tp1", "REAL")
    ensure_column("trades", "tp2", "REAL")
    ensure_column("trades", "tp3", "REAL")   # ✅ مهم جدًا (كان ناقص)

    ensure_column("trades", "hit_tp1", "INTEGER")   # 0/1
    ensure_column("trades", "hit_tp2", "INTEGER")   # 0/1
    ensure_column("trades", "hit_tp3", "INTEGER")   # 0/1
    ensure_column("trades", "max_tp_hit", "INTEGER")  # 0/1/2/3

    ensure_column("trades", "trail_stop", "REAL")
    ensure_column("trades", "tier", "TEXT")         # NORMAL/STRONG/ULTRA
    ensure_column("trades", "regime", "TEXT")       # TREND/RANGE/HIGH_VOL

    ensure_column("trades", "pnl_usdt", "REAL")

    # ✅ نسبة الربح/الخسارة رقمياً (بدون علامة %)
    ensure_column("trades", "pnl_pct", "REAL")

    # ✅ سبب الإغلاق (STOP / TP1 / TP2 / TP3 / TIMEOUT / DRAW)
    ensure_column("trades", "close_reason", "TEXT")


def set_default(key: str, value: str):
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM settings WHERE key=?", (key,))
    if cur.fetchone() is None:
        cur.execute("INSERT INTO settings(key,value) VALUES(?,?)", (key, value))
        conn.commit()
    conn.close()


def get_setting(key: str) -> Optional[str]:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None


def set_setting(key: str, value: str):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO settings(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value)
    )
    conn.commit()
    conn.close()


def insert_trade(payload: Dict[str, Any]) -> int:
    conn = connect()
    cur = conn.cursor()
    cols = ", ".join(payload.keys())
    qs = ", ".join(["?"] * len(payload))
    cur.execute(f"INSERT INTO trades ({cols}) VALUES ({qs})", tuple(payload.values()))
    trade_id = cur.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def update_trade_fields(trade_id: int, fields: Dict[str, Any]):
    if not fields:
        return
    conn = connect()
    cur = conn.cursor()
    sets = ", ".join([f"{k}=?" for k in fields.keys()])
    values = list(fields.values()) + [trade_id]
    cur.execute(f"UPDATE trades SET {sets} WHERE id=?", values)
    conn.commit()
    conn.close()


def get_open_trades() -> List[sqlite3.Row]:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM trades WHERE status='OPEN' ORDER BY time_open ASC")
    rows = cur.fetchall()
    conn.close()
    return rows


def close_trade(
    trade_id: int,
    time_close: str,
    result: str,
    close_price: float,
    pnl_usdt: Optional[float] = None,
    pnl_pct: Optional[float] = None,
    close_reason: Optional[str] = None,
):
    """
    يغلق الصفقة ويضمن كتابة pnl_pct و close_reason.
    إذا لم يتم تمرير pnl_pct نحسبها تلقائياً من entry و close_price.
    وإذا لم يتم تمرير pnl_usdt نحسبها تلقائيًا لو qty موجودة.
    """
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT entry, qty FROM trades WHERE id=?", (trade_id,))
    row = cur.fetchone()

    entry = float(row["entry"]) if row and row["entry"] is not None else None
    qty = float(row["qty"]) if row and row["qty"] is not None else None

    if pnl_pct is None and entry and entry != 0:
        pnl_pct = (float(close_price) - entry) / entry * 100.0

    if pnl_usdt is None and qty is not None and entry is not None:
        pnl_usdt = (float(close_price) - entry) * qty

    cur.execute("""
        UPDATE trades
        SET status='CLOSED',
            time_close=?,
            result=?,
            close_price=?,
            pnl_usdt=?,
            pnl_pct=?,
            close_reason=?
        WHERE id=?
    """, (time_close, result, float(close_price), pnl_usdt, pnl_pct, close_reason, trade_id))

    conn.commit()
    conn.close()


def count_open_trades() -> int:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM trades WHERE status='OPEN'")
    c = cur.fetchone()["c"]
    conn.close()
    return int(c)


def last_trade_time(symbol: str) -> Optional[str]:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT time_open FROM trades WHERE symbol=? ORDER BY time_open DESC LIMIT 1", (symbol,))
    row = cur.fetchone()
    conn.close()
    return row["time_open"] if row else None


def get_recent_closed(n: int) -> List[sqlite3.Row]:
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM trades
        WHERE status='CLOSED'
        ORDER BY time_close DESC
        LIMIT ?
    """, (n,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_stats() -> Dict[str, Any]:
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS c FROM trades")
    total = int(cur.fetchone()["c"])

    cur.execute("SELECT COUNT(*) AS c FROM trades WHERE status='OPEN'")
    open_ = int(cur.fetchone()["c"])

    cur.execute("SELECT COUNT(*) AS c FROM trades WHERE status='CLOSED'")
    closed = int(cur.fetchone()["c"])

    cur.execute("SELECT COUNT(*) AS c FROM trades WHERE status='CLOSED' AND result='WIN'")
    wins = int(cur.fetchone()["c"])

    cur.execute("SELECT COUNT(*) AS c FROM trades WHERE status='CLOSED' AND result='LOSS'")
    losses = int(cur.fetchone()["c"])

    cur.execute("SELECT COUNT(*) AS c FROM trades WHERE status='CLOSED' AND result='DRAW'")
    draws = int(cur.fetchone()["c"])

    avg_rr = None
    avg_pnl = None
    if closed > 0:
        cur.execute("SELECT AVG(rr) AS a FROM trades WHERE status='CLOSED'")
        row = cur.fetchone()
        avg_rr = float(row["a"]) if row and row["a"] is not None else None

        cur.execute("SELECT AVG(pnl_usdt) AS a FROM trades WHERE status='CLOSED' AND pnl_usdt IS NOT NULL")
        row = cur.fetchone()
        avg_pnl = float(row["a"]) if row and row["a"] is not None else None

    conn.close()

    win_rate = (wins / closed) if closed > 0 else None

    return {
        "total": total,
        "open": open_,
        "closed": closed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "avg_rr": avg_rr,
        "avg_pnl": avg_pnl,
    }


def get_today_pnl() -> float:
    today = dt.datetime.now().strftime("%Y-%m-%d")
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT COALESCE(SUM(pnl_usdt), 0) AS s
        FROM trades
        WHERE status='CLOSED'
          AND pnl_usdt IS NOT NULL
          AND time_close LIKE ?
    """, (today + "%",))
    s = float(cur.fetchone()["s"])
    conn.close()
    return s


def has_open_trade(symbol: str) -> bool:
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM trades WHERE status='OPEN' AND symbol=? LIMIT 1", (symbol,))
    row = cur.fetchone()
    conn.close()
    return row is not None