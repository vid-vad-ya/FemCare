import psycopg2

try:
    conn = psycopg2.connect(
        host     = "localhost",
        port     = 5432,
        dbname   = "femcare_db",
        user     = "postgres",
        password = "KAVIYA"    # ← type your actual postgres password here
    )
    print("✓ Connected to femcare_db successfully!")
    conn.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")