import sqlite3

con = sqlite3.connect("database.db")
cur = con.cursor()
cur.execute("SELECT * from Points")
rows = cur.fetchall()

print(rows)

