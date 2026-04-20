import sqlite3

DB_PATH = "bot.db"

def col_exists(cur, table, col):
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def add_col(cur, table, col, col_type):
    cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # الأعمدة المطلوبة للتحديثات الجديدة
    needed_cols = [
        ("close_reason", "TEXT"),
        ("pnl_pct", "REAL"),
        ("hit_tp2", "INTEGER"),
        ("hit_tp3", "INTEGER"),
        ("max_tp_hit", "INTEGER"),
    ]

    for col, typ in needed_cols:
        if not col_exists(cur, "trades", col):
            add_col(cur, "trades", col, typ)
            print(f"[OK] Added column: {col}")

    conn.commit()

    # --- Backfill للصفقات القديمة المغلقة ---
    # 1) حساب pnl_pct إذا كان NULL
    cur.execute("""
        UPDATE trades
        SET pnl_pct = ((close_price - entry) / entry) * 100.0
        WHERE status='CLOSED'
          AND pnl_pct IS NULL
          AND close_price IS NOT NULL
          AND entry IS NOT NULL
          AND entry != 0
    """)
    print(f"[OK] Backfilled pnl_pct for CLOSED trades. rows={cur.rowcount}")

    # 2) close_reason من result للصفقات القديمة
    cur.execute("""
        UPDATE trades
        SET close_reason =
            CASE
                WHEN UPPER(COALESCE(result,''))='WIN' THEN 'TP3'
                WHEN UPPER(COALESCE(result,''))='LOSS' THEN 'STOP'
                WHEN UPPER(COALESCE(result,''))='TIMEOUT' THEN 'TIMEOUT'
                ELSE 'TIMEOUT'
            END
        WHERE status='CLOSED'
          AND (close_reason IS NULL OR TRIM(close_reason)='')
    """)
    print(f"[OK] Backfilled close_reason for CLOSED trades. rows={cur.rowcount}")

    # 3) max_tp_hit: من hit_tp1 أو من close_reason/result
    cur.execute("""
        UPDATE trades
        SET max_tp_hit =
            CASE
                WHEN close_reason='TP3' THEN 3
                WHEN close_reason='TP2' THEN 2
                WHEN close_reason='TP1' THEN 1
                WHEN COALESCE(hit_tp1,0)=1 THEN 1
                ELSE 0
            END
        WHERE max_tp_hit IS NULL
    """)
    print(f"[OK] Backfilled max_tp_hit. rows={cur.rowcount}")

    # 4) تأكيد قيم hit_tp2/hit_tp3 الافتراضية
    cur.execute("UPDATE trades SET hit_tp2=COALESCE(hit_tp2,0) WHERE hit_tp2 IS NULL")
    cur.execute("UPDATE trades SET hit_tp3=COALESCE(hit_tp3,0) WHERE hit_tp3 IS NULL")
    conn.commit()

    # تقرير سريع
    cur.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl_pct IS NOT NULL")
    closed_with_pnl = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND close_reason IS NOT NULL")
    closed_with_reason = cur.fetchone()[0]

    print("\n=== SUMMARY ===")
    print("CLOSED with pnl_pct:", closed_with_pnl)
    print("CLOSED with close_reason:", closed_with_reason)

    conn.close()

if __name__ == "__main__":
    main()