# A database connection that enforces rules itself: only reads, preferably only the view, and not
# for too long. SQLite's authorizer and progress handler, standing in for grants and timeouts —
# and showing where a stand-in stops being enough.

import sqlite3
import time

DB = "data/meridian/warehouse/meridian.db"
ALLOWED_TABLES = {"v_sales"}
DEADLINE_SECONDS = 1.0


clock = {"started": 0.0}                                     # reset before each statement


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    def authorizer(action, table, column, database, source):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            if table in ALLOWED_TABLES or source in ALLOWED_TABLES:
                return sqlite3.SQLITE_OK            # the view, or a table read on its behalf
            if column == "":
                return sqlite3.SQLITE_OK            # see the note in the output
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_DENY                  # INSERT, UPDATE, DELETE, PRAGMA, ATTACH ...

    con.set_authorizer(authorizer)
    def watchdog():
        return 1 if time.perf_counter() - clock["started"] > DEADLINE_SECONDS else 0   # abort

    con.set_progress_handler(watchdog, 10_000)
    return con


con = connect()
TESTS = [
    ("a normal question", "SELECT region, ROUND(SUM(revenue)) FROM v_sales WHERE year = 2025 "
                          "GROUP BY region ORDER BY 2 DESC LIMIT 2"),
    ("a base table's columns", "SELECT name, segment FROM customers LIMIT 2"),
    ("a base table, counted", "SELECT COUNT(*) FROM customers"),
    ("a write", "DELETE FROM orders WHERE order_date < '2024-01-01'"),
    ("a settings change", "PRAGMA writable_schema = ON"),
    ("an accidental cross join", "SELECT COUNT(*) FROM v_sales a, v_sales b"),
]
for label, sql in TESTS:
    clock["started"] = time.perf_counter()
    try:
        rows = con.execute(sql).fetchall()
        print(f"  {label:26} allowed   {rows}")
    except sqlite3.DatabaseError as error:
        print(f"  {label:26} REFUSED   {type(error).__name__}: {error}  "
              f"({time.perf_counter() - clock['started']:.2f}s)")

plan = con.execute("EXPLAIN QUERY PLAN SELECT SUM(revenue) FROM v_sales WHERE year = 2025").fetchall()
print("\nEXPLAIN QUERY PLAN before running a generated query:")
for row in plan[:4]:
    print("   ", row[-1])

print("\nThe count got through. SQLite reports a read with no column both for COUNT(*) on a base")
print("table and for tables a view uses internally, so an authorizer cannot tell them apart.")
print("It is a useful guard against writes; it is not a grant. Postgres grants and row-level")
print("security enforce table access in the database, where no query can get around them.")
