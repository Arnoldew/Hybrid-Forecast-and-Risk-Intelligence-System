import sqlite3

conn = sqlite3.connect("database/ews.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM harga")
conn.commit()
conn.close()

print("Table harga cleared.")