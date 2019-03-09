import sqlite3
con = sqlite3.connect("database.db");

con.execute("CREATE TABLE Convos (filename TEXT, id NUMBER, comment TEXT, user TEXT)")
print("Table created succesfully");

con.close()
