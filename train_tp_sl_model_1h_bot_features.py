import sqlite3
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DB_PATH = "bot.db"

FEATURES = [
    "rsi", "atr_pct", "bb_width", "ret_3", "ret_10", "vol_ratio",
    "dist_from_high", "mom_5", "mom_5_acc", "vol_surge",
]

MIN_SAMPLES = 50

# ✅ تدريب أكثر تحفظًا:
# نعتبر TP2 / TP3 نجاحًا حقيقيًا
# و STOP خسارة
WIN_REASONS = {"TP2", "TP3"}
LOSS_REASONS = {"STOP"}

def load_closed_trades() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    q = f"""
        SELECT
            {",".join(FEATURES)},
            close_reason
        FROM trades
        WHERE status='CLOSED'
          AND close_reason IS NOT NULL
          AND {" AND ".join([f"{c} IS NOT NULL" for c in FEATURES])}
        ORDER BY time_close ASC
    """

    df = pd.read_sql_query(q, conn)
    conn.close()
    return df

def make_labels(df: pd.DataFrame):
    """
    - WIN إذا close_reason من TP2 / TP3
    - LOSS إذا close_reason = STOP
    - نستبعد TP1 / DRAW / TIMEOUT وغيرها
    """
    df = df.copy()
    df["close_reason"] = df["close_reason"].astype(str).str.upper().str.strip()

    mask = df["close_reason"].isin(WIN_REASONS | LOSS_REASONS)
    df2 = df.loc[mask].copy()

    y = df2["close_reason"].isin(WIN_REASONS).astype(int)
    X = df2[FEATURES].astype(float)

    return X, y, len(df), len(df2)

def main():
    df = load_closed_trades()

    if df.empty:
        print("⚠️ Skipped retrain: no CLOSED trades with features/close_reason found.")
        return

    X, y, total_rows, used_rows = make_labels(df)

    if used_rows < MIN_SAMPLES:
        print(
            "⚠️ Skipped retrain: not enough usable CLOSED trades.\n"
            f"Found total CLOSED rows with features: {total_rows}\n"
            f"Usable (TP2/TP3/STOP only): {used_rows} (need >= {MIN_SAMPLES})\n"
            "Tip: run bot longer until it closes more trades with TP2/TP3/STOP outcomes."
        )
        return

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = (n_neg / max(n_pos, 1))

    print("Total rows:", total_rows)
    print("Used rows:", used_rows)
    print("Class balance | neg:", n_neg, "pos:", n_pos, "scale_pos_weight:", round(spw, 3))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = xgb.XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=spw,
        reg_lambda=1.2,
        min_child_weight=4,
        n_jobs=4,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(classification_report(y_test, pred, zero_division=0))
    model.save_model("ai_model_tp_sl_bot_1h.json")
    print("✅ Model saved as ai_model_tp_sl_bot_1h.json")

if __name__ == "__main__":
    main()
