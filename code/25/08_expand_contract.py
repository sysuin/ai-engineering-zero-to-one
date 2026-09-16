# Renaming a column on a live system. The old code and the new code are both running during a
# deploy; which of them breaks, at which step, done in place and done by expand and contract?

import sqlite3


def fresh() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, question TEXT)")
    db.executemany("INSERT INTO runs (question) VALUES (?)", [("revenue 2024 Q3",), ("margin 2025 Q1",)])
    return db


# Two versions of the application, as they would be deployed side by side.
def old_write(db):  db.execute("INSERT INTO runs (question) VALUES ('old writer')")
def old_read(db):   return db.execute("SELECT question FROM runs ORDER BY id DESC LIMIT 1").fetchone()[0]
def dual_write(db): db.execute("INSERT INTO runs (question, prompt) VALUES ('dual writer', 'dual writer')")
def new_write(db):  db.execute("INSERT INTO runs (prompt) VALUES ('new writer')")
def new_read(db):   return db.execute("SELECT prompt FROM runs ORDER BY id DESC LIMIT 1").fetchone()[0]


def check(db, label: str, writers: list, readers: list) -> None:
    results = []
    for fn in writers + readers:
        try:
            value = fn(db)
            results.append(f"{fn.__name__} {'ok' if value is None else repr(value)}")
        except sqlite3.Error as error:
            reason = "no such column" if "column" in str(error) else "error"
            results.append(f"{fn.__name__} FAILS: {reason}")
    print(f"  {label}\n      " + "   ".join(results))


print("in place: one migration renames the column while old code is still serving\n")
db = fresh()
check(db, "before", [old_write], [old_read])
db.execute("ALTER TABLE runs RENAME COLUMN question TO prompt")
check(db, "renamed; old and new code both running", [old_write, new_write], [old_read, new_read])

print("\nexpand and contract: five deploys, each safe to roll back\n")
db = fresh()
db.execute("ALTER TABLE runs ADD COLUMN prompt TEXT")
check(db, "1 expand: add the new column", [old_write], [old_read])
check(db, "2 deploy code that writes both", [old_write, dual_write], [old_read])
db.execute("UPDATE runs SET prompt = question WHERE prompt IS NULL")
check(db, "3 backfill the old rows", [dual_write], [old_read, new_read])
check(db, "4 deploy code that reads the new column", [dual_write], [new_read])
db.execute("ALTER TABLE runs DROP COLUMN question")
check(db, "5 contract: drop the old column", [new_write], [new_read])
missing = db.execute("SELECT COUNT(*) FROM runs WHERE prompt IS NULL").fetchone()[0]
print(f"\n  rows without a value in the new column at the end: {missing}")
