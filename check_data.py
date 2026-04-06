import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

print("=== Data dengan harga < 5000 ===")
cursor.execute("""
    SELECT date, price FROM price_data
    WHERE price < 5000
    ORDER BY date
""")
rows = cursor.fetchall()
for r in rows:
    print(r)

print("\n=== Data Nov 2024 - Mar 2025 ===")
cursor.execute("""
    SELECT date, price FROM price_data
    WHERE date BETWEEN '2024-11-01' AND '2025-03-31'
    ORDER BY date
""")
rows = cursor.fetchall()
for r in rows:
    print(r)

conn.close()