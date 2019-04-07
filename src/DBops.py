import sqlite3
con = sqlite3.connect("database.db")

# con.execute("DROP TABLE Convos")
# con.execute("CREATE TABLE Convos (filename TEXT, id NUMBER, comment TEXT, user TEXT)")
# print("Table created succesfully")
#
# con.execute("DROP TABLE Codes")
# con.execute("CREATE TABLE Codes (poster TEXT, filename TEXT, description TEXT, lang TEXT)")
# print("Table created succesfully")
#
# con.execute("DROP TABLE Login")
# con.execute("CREATE TABLE Login ( email TEXT, pw TEXT )")
# print("Table created succesfully")
#
# con.execute("DROP TABLE Tags")
# con.execute("CREATE TABLE Tags( tag TEXT, PRIMARY KEY(tag) ) ")
# print("Table created succesfully")
#
# con.execute("DROP TABLE CodeTags")
# con.execute("CREATE TABLE CodeTags( code TEXT references Codes(filename), tag TEXT references Tags(tag))")
# print("Table created succesfully")
#
con.execute("DROP TABLE CodeViews")
con.execute("CREATE TABLE CodeViews( code TEXT references Codes(filename), user TEXT references Login(email), difficulty NUMERIC, views NUMERIC)")
print("Table created succesfully")
#
# con.execute("DROP TABLE CodeComments")
# con.execute("CREATE TABLE CodeComments( code TEXT references Codes(filename), comment TEXT references Comments(comment), line NUMERIC)")
# print("Table created succesfully")

# con.execute("DROP TABLE Comments")
# con.execute("CREATE TABLE Comments( comment TEXT, PRIMARY KEY(comment))")
# print("Table created succesfully")

con.close()
