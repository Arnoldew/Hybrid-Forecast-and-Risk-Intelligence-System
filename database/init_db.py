import sqlite3

def init_db():
    conn = sqlite3.connect("database/ews.db")
    cursor = conn.cursor()

    # Tabel harga historis
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS harga (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        price REAL NOT NULL
    )
    """)

    # Tabel forecast historis
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS forecast (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        forecast_date TEXT,
        horizon INTEGER,
        predicted_price REAL,
        upper_bound REAL,
        lower_bound REAL,
        model_used TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Tabel risk log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS risk_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        risk_status TEXT,
        risk_score INTEGER,
        fdi REAL,
        volatility REAL,
        confidence_breach INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")