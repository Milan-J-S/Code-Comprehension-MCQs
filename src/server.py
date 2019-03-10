import random
import string

from flask import Flask, request, jsonify, session, g, redirect, url_for, abort, \
    render_template, flash, send_from_directory
import sqlite3
from os import listdir

app = Flask(__name__)

print("connection recieved")
# CORS(app)

def convertToDict(x):
    obj = {}
    obj['filename'] = x[0];
    obj['title'] = x[1];
    return obj;

def generateRandomFilename():
    filename = ''
    for i in range(10):
        filename+=(chr(random.randrange(97,123)))
    return filename


@app.route("/")
def start():
    return render_template('codeUpload.html')


@app.route("/showCode", methods=["GET", "POST"])
def showCode():
    filename = request.args['filename']
    rows = fetchConvos(filename)
    full_filename = 'static/data/' + filename
    print(full_filename)



    f=  open(full_filename)
    return render_template('codeView.html', data=str(f.read()), rows = rows, filename = filename)


@app.route("/showAll")
def showAll():
    items = []
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT filename, description FROM Codes")
    rows = cur.fetchall()

    items = map( convertToDict, rows)

    return render_template('showResources.html', items=items)


def fetchConvos(filename):
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT user, comment FROM Convos WHERE filename =(?)", (filename,))
    rows = cur.fetchall()

    print(rows)

    return(rows)


@app.route("/putConvos", methods= ["POST"])
def putConvos():
    filename = request.form['filename']
    id = request.form['id']
    comment = request.form['comment']
    user = request.form['user']

    print("filename = "+filename)

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("INSERT INTO Convos VALUES (?,?,?,?)",(filename, id, comment, user) )
    con.commit()
    return redirect(url_for('showCode',filename = filename))

@app.route("/putCode", methods= ["POST"])
def putCode():
    poster = request.form['poster']
    content = request.form['content']
    lang = request.form['lang']
    description = request.form['description']


    filename = generateRandomFilename()

    print(content)

    file = open('static/data/'+filename, "w+")
    file.write(content)
    file.close()

    print("written to file ", filename );

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("INSERT INTO Codes VALUES (?,?,?,?)", ( poster, filename, description, lang ))
    con.commit()

    return jsonify(filename = filename)

@app.route("/userExists", methods= ["POST"])
def userExists():
    email = request.form['email']
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM Login where email = (?)", (email,))
    rows = cur.fetchall()

    return jsonify(success = len(rows))

@app.route("/login", methods= ["POST"])
def login():
    email = request.form['email']
    pw = request.form['password']
    operation = request.form['operation']

    if(operation == 'login'):
        con = sqlite3.connect("database.db")
        cur = con.cursor()
        cur.execute("SELECT pw FROM Login where email = (?)", (email,))
        rows = cur.fetchall()

        return jsonify(auth = (rows[0].pw == pw))

    else:
        con = sqlite3.connect("database.db")
        cur = con.cursor()
        cur.execute("INSERT INTO Login values = (?,?)", (email, pw))
        con.commit()

        return jsonify(auth=True)



if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', threaded=True)
