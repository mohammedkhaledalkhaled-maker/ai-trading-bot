import sqlite3
import datetime as dt

DB_PATH = "bot.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ===============================
# last N results
# ===============================

def results(limit=50):

    conn = get_conn()
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT symbol, result, pnl_pct, close_time
        FROM trades
        WHERE result IS NOT NULL
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()

    wins = []
    losses = []

    total_profit = 0
    total_loss = 0

    for r in rows:

        sym = r["symbol"]
        pnl = r["pnl_pct"]

        if r["result"] in ["TP1", "TP2", "TP3"]:
            wins.append(f"{sym} : +{round(pnl,2)}%")
            total_profit += pnl

        if r["result"] == "STOP":
            losses.append(f"{sym} : {round(pnl,2)}%")
            total_loss += pnl

    msg = "📊 آخر الصفقات\n\n"

    msg += "📈 الرابحة\n"
    msg += "\n".join(wins) if wins else "لا يوجد"
    msg += "\n\n"

    msg += "📉 الخاسرة\n"
    msg += "\n".join(losses) if losses else "لا يوجد"
    msg += "\n\n"

    msg += f"إجمالي الربح : {round(total_profit,2)}%\n"
    msg += f"إجمالي الخسارة : {round(total_loss,2)}%\n"

    return msg


# ===============================
# today results
# ===============================

def results_today():

    conn = get_conn()
    cur = conn.cursor()

    today = dt.date.today().isoformat()

    rows = cur.execute(
        """
        SELECT symbol, result, pnl_pct
        FROM trades
        WHERE date(close_time)=?
        """,
        (today,)
    ).fetchall()

    wins = []
    losses = []

    total_profit = 0
    total_loss = 0

    for r in rows:

        sym = r["symbol"]
        pnl = r["pnl_pct"]

        if r["result"] in ["TP1","TP2","TP3"]:
            wins.append(f"{sym} : +{round(pnl,2)}%")
            total_profit += pnl

        if r["result"] == "STOP":
            losses.append(f"{sym} : {round(pnl,2)}%")
            total_loss += pnl

    msg = "📊 نتائج اليوم\n\n"

    msg += "📈 الرابحة\n"
    msg += "\n".join(wins) if wins else "لا يوجد"
    msg += "\n\n"

    msg += "📉 الخاسرة\n"
    msg += "\n".join(losses) if losses else "لا يوجد"
    msg += "\n\n"

    msg += f"إجمالي الربح اليوم : {round(total_profit,2)}%\n"
    msg += f"إجمالي الخسارة اليوم : {round(total_loss,2)}%\n"

    return msg


# ===============================
# best coins
# ===============================

def topcoins(limit=10):

    conn = get_conn()
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT symbol,
        COUNT(*) as trades,
        SUM(CASE WHEN result IN ('TP1','TP2','TP3') THEN 1 ELSE 0 END) as wins
        FROM trades
        GROUP BY symbol
        HAVING trades >= 5
        """).fetchall()

    stats = []

    for r in rows:

        winrate = r["wins"] / r["trades"]

        stats.append(
            (
                r["symbol"],
                r["trades"],
                round(winrate*100,1)
            )
        )

    stats.sort(key=lambda x: x[2], reverse=True)

    msg = "🏆 أفضل العملات\n\n"

    for s in stats[:limit]:
        msg += f"{s[0]} : {s[2]}% winrate ({s[1]} trades)\n"

    return msg


# ===============================
# worst coins
# ===============================

def worstcoins(limit=10):

    conn = get_conn()
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT symbol,
        COUNT(*) as trades,
        SUM(CASE WHEN result IN ('TP1','TP2','TP3') THEN 1 ELSE 0 END) as wins
        FROM trades
        GROUP BY symbol
        HAVING trades >= 5
        """).fetchall()

    stats = []

    for r in rows:

        winrate = r["wins"] / r["trades"]

        stats.append(
            (
                r["symbol"],
                r["trades"],
                round(winrate*100,1)
            )
        )

    stats.sort(key=lambda x: x[2])

    msg = "⚠️ أسوأ العملات\n\n"

    for s in stats[:limit]:
        msg += f"{s[0]} : {s[2]}% winrate ({s[1]} trades)\n"

    return msg