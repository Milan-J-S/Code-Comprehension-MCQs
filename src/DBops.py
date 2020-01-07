import sqlite3

con = sqlite3.connect("database.db")
cur = con.cursor()
# cur.execute("DROP TABLE ModifiedDistractors")
# cur.execute("CREATE TABLE ModifiedDistractors ( user TEXT, timeDiff NUMBER, filename TEXT, option1 TEXT, option2 TEXT, option3 TEXT, originalOption1 TEXT, originalOption2 TEXT, originalOption3 TEXT, diff1 TEXT, diff2 TEXT, diff3 TEXT )")
# cur.execute("DROP TABLE CreatedDistractors")
# cur.execute("CREATE TABLE CreatedDistractors ( user TEXT, timeDiff NUMBER, option1 TEXT, option2 TEXT, option3 TEXT, filename TEXT, diff1 TEXT, diff2 TEXT, diff3 TEXT)")

cur.execute("CREATE TABLE GoodComments AS SELECT * FROM Comments;")

# cur.execute("CREATE TABLE Synonyms (answer TEXT, synonym TEXT)")

# cur.execute("DROP TABLE Comments")
# cur.execute("CREATE TABLE Comments ( comment TEXT)")
# cur.execute("INSERT INTO Comments values (?)",("Reverse a given array",))
# cur.execute("INSERT INTO Comments values (?)",("Replace all occurrences of a given integer in an array with another integer",))
# cur.execute("INSERT INTO Comments values (?)",("Count the number of times an integer appears in an array",))
# cur.execute("INSERT INTO Comments values (?)",("Count the number of array elements in a given range",))

# cur.execute("DELETE FROM Codes")

# entries = [('cccrvlcqyg',), ('zqkerplfph',), ('rezxuytlac',), ('amdxxgrdxl',), ('aiuqwxggdl',), ('vqwfsnsyns',), ('dinbahqulh',), ('elurfearoz',), ('sbnxjxfsvz',), ('zqwafyxmfa',), ('vqbbmbtztc',), ('anjqhdtlyz',), ('kaoozevbct',), ('etadekqyjf',), ('cucjxvynws',), ('mykefmctbn',), ('gkebhblrud',), ('zipmfenkfi',), ('zxvtqhvccc',), ('idksdhnlbf',), ('tlsxgzituz',), ('nwdfdklgvt',), ('jwvhpyyxmt',), ('xyjghrqawl',), ('gscqaoykkd',), ('golplmgpdn',), ('ilauuelzey',)]

# for entry in entries:
#     cur.execute("INSERT INTO Codes values(?, ?, ? , ?)", (None, entry[0],"Some Description" ,'text/x-csrc'))


con.commit()
