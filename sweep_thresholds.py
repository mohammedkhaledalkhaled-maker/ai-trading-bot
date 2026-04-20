import subprocess
import re

thresholds = [0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]
results = []

with open("backtest_all.py", "r", encoding="utf-8") as f:
    original = f.read()

pattern = r"PROB_THRESHOLD\s*=\s*[0-9.]+"
if not re.search(pattern, original):
    raise SystemExit("لم أجد السطر PROB_THRESHOLD = ... داخل backtest_all.py. أضفه بصيغة واضحة ثم أعد التشغيل.")

for th in thresholds:
    modified = re.sub(pattern, f"PROB_THRESHOLD = {th}", original)

    with open("backtest_all.py", "w", encoding="utf-8") as f:
        f.write(modified)

    out = subprocess.check_output(["python", "backtest_all.py"], text=True, encoding="utf-8", errors="ignore")

    total = int(re.search(r"Total trades:\s*(\d+)", out).group(1))
    wins = int(re.search(r"Wins:\s*(\d+)", out).group(1))
    winrate = float(re.search(r"Winrate:\s*([0-9.]+)%", out).group(1))
    ev = float(re.search(r"Expected Value:\s*([-0-9.]+)", out).group(1))

    results.append((th, total, wins, winrate, ev))
    print(f"TH={th:.2f} | trades={total} | wins={wins} | winrate={winrate:.2f}% | EV={ev:.4f}")

with open("backtest_all.py", "w", encoding="utf-8") as f:
    f.write(original)

print("\n=== SUMMARY (sorted by EV) ===")
for th, total, wins, winrate, ev in sorted(results, key=lambda x: x[4], reverse=True):
    print(f"TH={th:.2f} | trades={total} | winrate={winrate:.2f}% | EV={ev:.4f}")