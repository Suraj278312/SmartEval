import sqlite3

conn = sqlite3.connect('instance/smarteval.db')
for r in conn.execute("SELECT type, name, sql FROM sqlite_master WHERE tbl_name='answer_evaluation'"):
    print(f"TYPE: {r[0]} | NAME: {r[1]}")
    print(f"SQL:\n{r[2]}\n")

print("\n--- PRAGMA table_info ---")
for col in conn.execute("PRAGMA table_info(answer_evaluation)"):
    print(col)

conn.close()
