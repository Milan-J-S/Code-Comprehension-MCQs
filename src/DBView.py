import sqlite3

con = sqlite3.connect("database.db")
cur = con.cursor()
cur.execute("SELECT * from Tags")
rows = cur.fetchall()

print(rows)

