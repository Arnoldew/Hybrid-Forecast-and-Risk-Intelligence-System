import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

print("=== Total baris data ===")
cursor.execute("SELECT COUNT(*) FROM price_data")
print("Total:", cursor.fetchone()[0])

print("\n=== Tanggal pertama dan terakhir ===")
cursor.execute("SELECT MIN(date), MAX(date) FROM price_data")
print(cursor.fetchone())

print("\n=== 10 data terakhir ===")
cursor.execute("""
    SELECT date, price FROM price_data
    ORDER BY date DESC
    LIMIT 10
""")
for r in cursor.fetchall():
    print(r)

print("\n=== Cek data Jan 2026 ===")
cursor.execute("""
    SELECT date, price FROM price_data
    WHERE date LIKE '2026%'
    ORDER BY date
""")
rows = cursor.fetchall()
print(f"Jumlah data 2026: {len(rows)}")
for r in rows:
    print(r)

conn.close()