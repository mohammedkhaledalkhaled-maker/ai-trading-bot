import sqlite3

conn = sqlite3.connect("bot.db")
c = conn.cursor()

# عدد الصفقات المغلقة
c.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED'")
total = c.fetchone()[0]

# عدد الرابحة
c.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND result='WIN'")
wins = c.fetchone()[0]

# عدد الخاسرة
c.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND result='LOSS'")
losses = c.fetchone()[0]

# متوسط RR
c.execute("SELECT AVG(rr) FROM trades WHERE status='CLOSED'")
avg_rr = c.fetchone()[0]

# متوسط الربح بالدولار
c.execute("SELECT AVG(pnl_usdt) FROM trades WHERE status='CLOSED'")
avg_pnl = c.fetchone()[0]

winrate = (wins / total * 100) if total > 0 else 0

print("----- STATISTICS -----")
print("Closed trades:", total)
print("Wins:", wins)
print("Losses:", losses)
print("Winrate: %.2f%%" % winrate)
print("Average RR:", avg_rr)
print("Average PnL (USDT):", avg_pnl)

conn.close()