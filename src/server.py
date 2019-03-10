import random
import string

from flask import Flask, request, jsonify, session, g, redirect, url_for, abort, \
    render_template, flash, send_from_directory
import sqlite3
from os import listdir

app = Flask(__name__)


# CORS(app)

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
    for f in listdir('static/data'):
        print(f)
        di = {}
        di['title'] = f
        items.append(di)
    print(items)
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

    filename = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

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



if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0')
