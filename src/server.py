import random
import string

from flask import Flask, request, jsonify, session, g, redirect, url_for, abort, \
    render_template, make_response
import sqlite3
import numpy as np
import re



from sklearn.neighbors import NearestNeighbors

from os import listdir
import json

from nltk.tokenize import word_tokenize

import nltk
nltk.download('corpus')

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

import os

app = Flask(__name__)

print("connection recieved")


# CORS(app)

def convertToDict(x):
    obj = {}
    obj['filename'] = x[0]
    obj['title'] = x[1]
    return obj


def generateRandomFilename():
    filename = ''
    for i in range(10):
        filename += (chr(random.randrange(97, 123)))
    return filename


def generateTags(code):

    code= re.sub(r"#include.*<.+>",'',code)

    print(code)
    f = open("cleaned.txt", 'w+')
    f.write(str(code))
    f.close()

    os.system("pcpp cleaned.txt --line-directive > test.txt")
    os.system("python parseToJson.py test.txt > test1.txt")

    f = open("test1.txt", "r")
    AST = json.loads(f.read())
    print(AST)

    tags = set()

    for item in AST['ext']:
        curfunc = str(item)
        curfunc = re.sub(r"\'coord\': [^,]+,", "", curfunc)
        curfunc = curfunc.replace("\'", "")
        for item in KNN(curfunc):
            tags.add(item)

    return list(tags)


@app.route("/upload")
def start():
    return render_template('codeUpload.html', username=request.cookies.get("user", "Login/Sign Up").split("@")[0])


@app.route("/showCode", methods=["GET", "POST"])
def showCode():
    filename = request.args['filename']
    rows = fetchConvos(filename)
    full_filename = 'static/data/' + filename
    print(full_filename)

    f = open(full_filename)
    return render_template('codeView.html', data=str(f.read()), rows=rows, filename=filename,
                           username=request.cookies.get("user", "Login/Sign Up").split("@")[0])


@app.route("/")
def showAll():
    items = []
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT filename, description FROM Codes")
    rows = cur.fetchall()

    items = map(convertToDict, rows)

    resp = make_response(render_template('showResources.html', items=items,
                                         username=request.cookies.get("user", "Login/Sign Up").split("@")[0]))
    resp.set_cookie("test", "test")

    return (resp)


def fetchConvos(filename):
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT user, comment FROM Convos WHERE filename =(?)", (filename,))
    rows = cur.fetchall()
    print(rows)
    return (rows)


@app.route("/putConvos", methods=["POST"])
def putConvos():
    filename = request.form['filename']
    id = request.form['id']
    comment = request.form['comment']

    user = request.cookies.get("user")

    print("filename = " + filename)

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("INSERT INTO Convos VALUES (?,?,?,?)", (filename, id, comment, user))
    con.commit()
    return redirect(url_for('showCode', filename=filename))


@app.route("/putCode", methods=["POST"])
def putCode():
    content = request.form['content']
    lang = request.form['lang']
    description = request.form['description']

    user = request.cookies.get("user")

    print(request.cookies)

    filename = generateRandomFilename()

    print(content)

    file = open('static/data/' + filename, "w+")
    file.write(content)
    file.close()

    print("written to file ", filename)

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("INSERT INTO Codes VALUES (?,?,?,?)", (user, filename, description, lang))
    con.commit()

    print(content)

    global code_tensors
    tags = generateTags(content)

    return jsonify(filename=filename, tags=tags)

@app.route("/getTags",methods=["POST"])
def getTags():
    content = request.form['content']
    tags = generateTags(content)
    return jsonify(tags=tags)

@app.route("/userExists", methods=["POST"])
def userExists():
    email = request.form['email']
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT * FROM Login where email = (?)", (email,))
    rows = cur.fetchall()

    return jsonify(success=len(rows))


@app.route("/login", methods=["POST"])
def login():
    email = request.form['email']
    pw = request.form['password']
    operation = request.form['operation']

    if (operation == 'login'):
        con = sqlite3.connect("database.db")
        cur = con.cursor()
        cur.execute("SELECT pw FROM Login where email = (?)", (email,))
        rows = cur.fetchall()

        print(rows)

        resp = jsonify(auth=(rows[0][0] == pw))



    else:
        con = sqlite3.connect("database.db")
        cur = con.cursor()
        cur.execute("INSERT INTO Login values (?,?)", (email, pw))
        con.commit()

        resp = jsonify(auth=True)

    resp.set_cookie("user", email.split("@")[0])
    return (resp)


def KNN(code):
    global code_tensors
    global commments

    global code_dict

    code_tensor = np.zeros(750)
    for i in range(min(750, len(code))):
        if code[i] in code_dict:
            code_tensor[i] = code_dict[code[i]]


    # print(comments)

    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(code_tensors)
    distances, indices = nbrs.kneighbors(np.asarray([code_tensor]))

    print("distances and indices")
    # print(distances)
    print(indices[0][0])

    words = dict()
    # print(comments[indices[0][0]],"\n\n")
    for j in range(1, len(indices[0])):

        index = indices[0][j]
        print(comments[index])

        for word in comments[index].split():
            if (word not in stop):
                if (word not in words):
                    words[word] = 0
                # if(distances[i][j] != 0):
                words[word] += 1000 / (distances[0][j] + 1)

    words_ordered = sorted(words.items(), key=lambda kv: kv[1], reverse=True)
    tags = [x[0] for x in words_ordered[:10]]
    print(tags, "\n\n\n")
    return tags


code_tensors = []
comments = []
code_dict = dict()


def clusterCodes():
    global code_tensors
    global comments
    filepath = 'code'
    codes = []
    comments = []
    code_vocab = set()
    for root, dirs, files in os.walk(filepath):
        for f in files:
            code = open('code/' + f).read()
            code = re.sub("\"[^\"]*\"", "0", code)
            code = re.sub("name: [^,}]+", "name", code)
            code = re.sub("value: [^,}]+", "value", code)
            codes.append(word_tokenize(code))

            for word in word_tokenize(code):
                code_vocab.add(word)

            comment = open('comments/' + f).read()
            comments.append(comment)

    code_tensors = []

    i = 0
    for word in code_vocab:
        code_dict[word] = i
        i += 1

    for code in codes:
        code_tensor = np.zeros(750)
        for i in range(min(750, len(code))):
            code_tensor[i] = code_dict[code[i]]
        code_tensors.append(code_tensor)
    print(code_tensors)


clusterCodes()

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', threaded=True)
