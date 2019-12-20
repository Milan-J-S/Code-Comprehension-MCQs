import sqlite3

con = sqlite3.connect("database.db")
cur = con.cursor()
# cur.execute("SELECT * from CodeComments, Codes where CodeComments.code = Codes.filename")
# rows = cur.fetchall()
#
# for row in rows:
#     print(row[1])

cur.execute("SELECT * from GoodCodes")
rows = cur.fetchall()
print(rows)

cur.execute("SELECT * from GoodComments")
rows = cur.fetchall()
print(rows)




