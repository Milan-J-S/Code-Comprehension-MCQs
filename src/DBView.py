import sqlite3

con = sqlite3.connect("database.db")
cur = con.cursor()
cur.execute("SELECT * from CodeViews")
rows = cur.fetchall()

print(rows)

