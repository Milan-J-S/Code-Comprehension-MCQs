import sqlite3

con = sqlite3.connect("database.db")
cur = con.cursor()
# cur.execute("SELECT * from CodeComments, Codes where CodeComments.code = Codes.filename")
# rows = cur.fetchall()
#
# for row in rows:
#     print(row[1])

# cur.execute("SELECT * from ModifiedDistractors")
# rows = cur.fetchall()
# print(rows)

cur.execute("SELECT * from GoodComments")
rows = cur.fetchall()

# rows = set(map(lambda x: x[0], cur.execute("SELECT filename from Codes").fetchall()))

print(rows)




