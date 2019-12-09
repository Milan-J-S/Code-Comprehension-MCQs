import sqlite3

con = sqlite3.connect("database.db")
cur = con.cursor()
# cur.execute("DROP TABLE ModifiedDistractors")
# cur.execute("CREATE TABLE ModifiedDistractors ( user TEXT, timeDiff NUMBER, filename TEXT, option1 TEXT, option2 TEXT, option3 TEXT, originalOption1 TEXT, originalOption2 TEXT, originalOption3 TEXT )")
cur.execute("CREATE TABLE GoodCodes SELECT * FROM Codes;")
con.commit()
