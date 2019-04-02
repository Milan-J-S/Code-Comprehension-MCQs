import random
import string

from flask import Flask, request, jsonify, session, g, redirect, url_for, abort, \
    render_template, make_response
import sqlite3
import numpy as np
import re

user_codes_matrix = []

from sklearn.neighbors import NearestNeighbors

from os import listdir
import json

from nltk.tokenize import word_tokenize

import nltk

nltk.download('corpus')

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

import legacy
from legacy import AttentionDecoder

import os

import pickle
from keras.models import load_model

app = Flask(__name__)


# print("connection recieved")


# CORS(app)

def convertToDict(x):
    obj = {}
    obj['filename'] = x[0]
    obj['title'] = x[1]
    return obj


new_users_dict = {}
new_codes_dict = {}

new_users_reverse_map = []
new_codes_reverse_map = []


def generateRandomFilename():
    filename = ''
    for i in range(10):
        filename += (chr(random.randrange(97, 123)))
    return filename

automodel = load_model("encoder (1).h5")

attnmodel = load_model("attention (2).h5", custom_objects={'AttentionDecoder': AttentionDecoder})

code_dict_p = pickle.load(open("code_dict (3).pickle", "rb+"))
print(code_dict_p)


comments_reverse_map_p = pickle.load(open("comments_reverse_map (3).pickle", "rb+"))

def generateComments(code):
    global automodel
    global attnmodel
    global code_dict_p
    global comments_reverse_map_p
    code_tensors = []

    code = re.sub("\"[^\"]*\"", "0", code)
    code = re.sub("name: [^,}]+", "name", code)
    code = re.sub("value: [^,}]+", "value", code)

    code_tensor = np.zeros(751)
    item = word_tokenize(code)
    for i in range(min(len(item), 751)):
        code_tensor[i] = code_dict_p[item[i]] / 107
    code_tensors.append(code_tensor)

    result = automodel.predict(np.asarray(code_tensors))

    comment = attnmodel.predict(result.reshape(1,20,1))

    res = []

    for item in comment[0]:
        #print(np.argmax(item))
        if(np.argmax(item)-3>0):
            res.append(comments_reverse_map_p[np.argmax(item)-3])

    return res


def generateTags(code):
    code = re.sub(r"#include.*<.+>", '', code)

    # print(code)
    f = open("cleaned.txt", 'w+')
    f.write(str(code))
    f.close()

    os.system("pcpp cleaned.txt --line-directive > test.txt")
    os.system("python parseToJson.py test.txt > test1.txt")

    f = open("test1.txt", "r")
    AST = json.loads(f.read())
    # print(AST)

    tags = set()

    for item in AST['ext']:
        curfunc = str(item)
        curfunc = re.sub(r"\'coord\': [^,]+,", "", curfunc)
        curfunc = curfunc.replace("\'", "")

        for item in KNN(curfunc):
            tags.add(item)

    return list(tags)

def generateFunctionComment(code):
    code = re.sub(r"#include.*<.+>", '', code)

    # print(code)
    f = open("cleaned.txt", 'w+')
    f.write(str(code))
    f.close()

    os.system("pcpp cleaned.txt --line-directive > test.txt")
    os.system("python parseToJson.py test.txt > test1.txt")

    f = open("test1.txt", "r")
    AST = json.loads(f.read())
    # print(AST)

    funComment = []

    for item in AST['ext']:
        curfunc = str(item)
        curfunc = re.sub(r"\'coord\': [^,]+,", "", curfunc)
        curfunc = curfunc.replace("\'", "")

        resss = generateComments(curfunc)
        funComment.append(resss)

    return funComment


@app.route("/upload")
def start():
    return render_template('codeUpload.html', username=request.cookies.get("user", "Login/Sign Up").split("@")[0])


@app.route("/showCode", methods=["GET", "POST"])
def showCode():
    global user_codes_matrix
    filename = request.args['filename']
    rows = fetchConvos(filename)
    full_filename = 'static/data/' + filename
    # print(full_filename)

    f = open(full_filename)

    username = request.cookies.get("user", "Login/Sign Up").split("@")[0]

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("INSERT into CodeViews values (?,?,?) ", (filename, username, 0))
    con.commit()

    # user_codes_matrix[new_users_dict[username][new_codes_dict[filename]]] += 1

    # user_codes_matrix[use]

    return render_template('codeView.html', data=str(f.read()), rows=rows, filename=filename, username=username
                           )


def recommendCodes(user):
    user_index = new_users_dict[user]

    user_has_viewed = set()

    for i in range(len(new_codes_dict)):
        if user_codes_matrix[user_index][i] > 0:
            user_has_viewed.add(i)

    user_stats_dict = {}

    for j in range(len(new_users_dict)):
        user_stats_dict[j] = 0
        for item in user_has_viewed:
            user_stats_dict[j] += user_codes_matrix[j][item]

    users_ordered = sorted(user_stats_dict.items(), key=lambda kv: kv[1], reverse=True)[1:4]

    code_to_recommend = {}

    for i in range(len(new_codes_dict)):
        if (i in user_has_viewed):
            continue
        code_to_recommend[i] = 0
        for user in users_ordered:
            code_to_recommend[i] += user_codes_matrix[user[0]][i]

    codes_ordered = sorted(code_to_recommend.items(), key=lambda kv: kv[1], reverse=True)[1:]

    print(codes_ordered)

    for code in codes_ordered:
        print(new_codes_reverse_map[code[0]])

    return codes_ordered


@app.route("/")
def showAll():
    items = []
    code_desc = {}
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT filename, description FROM Codes")
    rows = cur.fetchall()
    username = request.cookies.get("user", "Login/Sign Up").split("@")[0]

    for row in rows:
        code_desc[new_codes_dict[row[0]]] = row[1]
    rows = recommendCodes(username)

    rows = [(x[0], code_desc[x[0]]) for x in rows]

    items = map(convertToDict, rows)

    resp = make_response(render_template('showResources.html', items=items, username=username))
    resp.set_cookie("test", "test")

    return (resp)


def fetchConvos(filename):
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT user, comment FROM Convos WHERE filename =(?)", (filename,))
    rows = cur.fetchall()
    # print(rows)
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
    payload = request.json

    # print(payload)

    content = payload.get('content', '')
    lang = payload.get('lang', '')
    description = payload.get('description', '')

    tags = list(payload.get('tags', ''))

    user = request.cookies.get("user")

    # print(request.cookies)

    filename = generateRandomFilename()

    print("tags = ", tags)

    file = open('static/data/' + filename, "w+")
    file.write(content)
    file.close()

    print("written to file ", filename)

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("INSERT INTO Codes VALUES (?,?,?,?)", (user, filename, description, lang))
    con.commit()

    for tag in tags:
        cur = con.cursor()
        exists = cur.execute("SELECT * from tags where tag = (?)", (tag,)).fetchall()
        if (len(exists) == 0):
            cur.execute("INSERT INTO Tags VALUES (?)", (tag,))
        cur.execute("INSERT INTO CodeTags VALUES (?,?)", (filename, tag))
    con.commit()
    con.close()

    print(content)

    global user_codes_matrix
    global new_codes_dict
    global new_codes_reverse_map

    for i in range(len(new_users_dict)):
        user_codes_matrix[i].extend(0)

    new_codes_dict[filename] = len(new_codes_dict)
    new_codes_reverse_map.append(filename)


    # global code_tensors
    # tags = generateTags(content)

    return jsonify(filename=filename)


@app.route("/getTags", methods=["POST"])
def getTags():
    content = request.form['content']
    tags = generateTags(content)

    print("tags generated = ", tags)

    return jsonify(tags=tags)

@app.route("/getFuncComments", methods=["POST"])
def getFuncComments():
    content = request.form['content']
    funCom = generateFunctionComment(content)

    return jsonify(funCom=funCom)


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

        # print(rows)

        resp = jsonify(auth=(rows[0][0] == pw))



    else:
        con = sqlite3.connect("database.db")
        cur = con.cursor()
        cur.execute("INSERT INTO Login values (?,?)", (email, pw))
        con.commit()

        resp = jsonify(auth=True)

    resp.set_cookie("user", email.split("@")[0])
    np.append(user_codes_matrix, np.zeros(len(new_codes_dict)))
    new_users_dict[email.split("@")[0]] = len(new_users_dict)
    return (resp)


def prepareUserMatrix():
    global user_codes_matrix
    global new_users_dict
    global new_codes_dict
    global new_codes_reverse_map
    global new_users_reverse_map

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT DISTINCT filename FROM Codes")
    rows = cur.fetchall()

    print(rows)

    new_codes_dict = {}

    i = 0
    for row in rows:
        new_codes_dict[row[0]] = i
        new_codes_reverse_map.append(row[0])
        i += 1

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT DISTINCT email FROM Login")
    rows = cur.fetchall()

    new_users_dict = {}

    i = 0
    for row in rows:
        print(row)
        new_users_dict[row[0].split("@")[0]] = i
        new_users_reverse_map.append(row[0].split("@")[0])
        i += 1

    new_users_dict['Login/Sign Up'] = i

    user_codes_matrix = np.zeros((len(new_users_dict), len(new_codes_dict)))

    print(user_codes_matrix)
    print(user_codes_matrix.shape)

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT code,user FROM CodeViews")
    rows = cur.fetchall()

    for row in rows:
        user_codes_matrix[new_users_dict[row[1]]][new_codes_dict[row[0]]] += 1

    print(user_codes_matrix)


def KNN(code):
    global code_tensors
    global commments

    global code_dict

    code = re.sub("\"[^\"]*\"", "0", code)
    code = re.sub("name: [^,}]+", "name", code)
    code = re.sub("value: [^,}]+", "value", code)
    code = word_tokenize(code)

    code_tensor = np.zeros(750)
    for i in range(min(750, len(code))):
        if code[i] in code_dict:
            code_tensor[i] = code_dict[code[i]]

    distances, indices = nbrs.kneighbors(np.asarray([code_tensor]))

    print("distances and indices")
    print(distances)
    print(indices[0][0])

    words = dict()
    print(comments[indices[0][0]], "\n\n")
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
    # print(tags, "\n\n\n")
    return tags


code_tensors = []
comments = []
code_dict = dict()
nbrs = None
indices = []

codes_reverse_map = []
codes_dict = {}


def recommendForUser(user):
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT code from CodeViews WHERE user = (?)", (user,))

    rows = cur.fetchall()

    reco = {}

    print(rows)
    for item in rows:
        print(codes_dict[item[0]])
        for code in indices[codes_dict[item[0]]]:
            if (code not in reco):
                reco[code] = 0
            reco[code] += 1

    print(reco)

    reco_ordered = sorted(reco.items(), key=lambda kv: kv[1], reverse=True)
    reco_ordered = [x[0] for x in reco_ordered[:10]]

    for rec in reco_ordered:
        print(codes_reverse_map[rec])


def clusterCodes():
    global code_tensors
    global comments
    global nbrs
    global codes_reverse_map
    global indices
    filepath = 'code'
    codes = []
    comments = []
    code_vocab = set()
    i = 0
    for root, dirs, files in os.walk(filepath):
        for f in files:
            code = open('code/' + f).read()
            code = re.sub("\"[^\"]*\"", "0", code)
            code = re.sub("name: [^,}]+", "name", code)
            code = re.sub("value: [^,}]+", "value", code)
            codes.append(word_tokenize(code))
            codes_reverse_map.append(f)
            codes_dict[f] = i
            i += 1

            for word in word_tokenize(code):
                code_vocab.add(word)

            comment = open('comments/' + f).read()
            comments.append(comment)

    code_tensors = []

    i = 0
    for word in code_vocab:
        code_dict[word] = i
        i += 1

    j = 0
    for code in codes:
        code_tensor = np.zeros(750)
        for i in range(min(750, len(code))):
            code_tensor[i] = code_dict[code[i]]
        code_tensors.append(code_tensor)

    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(code_tensors)

    distances, indices = nbrs.kneighbors(np.asarray(code_tensors))

    print(indices)
    print(nbrs)

    # recommendForUser('milan.j.srinivas')

    # print(code_ten sors)


clusterCodes()

prepareUserMatrix()


@app.route("/search", methods=["POST"])
def search():
    con = sqlite3.connect("database.db")
    cur = con.cursor()

    global new_codes_dict

    searchTerms = request.form.get("searchTerms", "").split()

    print(searchTerms)

    files_that_match = {}

    for searchTerm in searchTerms:
        cur.execute("SELECT code from CodeTags WHERE tag = (?)", (searchTerm,))
        rows = cur.fetchall()
        for row in rows:
            if (row[0] not in files_that_match):
                files_that_match[row[0]] = 0
            files_that_match[row[0]] += 1

    files_ordered = sorted(files_that_match.items(), key=lambda kv: kv[1], reverse=True)


    print(files_ordered)

    code_desc = {}
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT filename, description FROM Codes")
    rows = cur.fetchall()
    # username = request.cookies.get("user", "Login/Sign Up").split("@")[0]

    for row in rows:
        code_desc[row[0]] = row[1]
    # rows = recommendCodes(username)

    print(code_desc)


    files_ordered = [(x[0], code_desc[x[0]]) for x in files_ordered]

    items = list(map(convertToDict, files_ordered))

    return jsonify(files=items)


# search(["tree", "print"])
resullt = generateComments('{_nodetype: FuncDef, body: {_nodetype: Compound, block_items: [{_nodetype: For, cond: {_nodetype: BinaryOp,  left: {_nodetype: ID,  name: i}, op: <, right: {_nodetype: ID,  name: n}},  init: {_nodetype: DeclList,  decls: [{_nodetype: Decl, bitsize: None,  funcspec: [], init: {_nodetype: Constant,  type: int, value: 0}, name: i, quals: [], storage: [], type: {_nodetype: TypeDecl,  declname: i, quals: [], type: {_nodetype: IdentifierType,  names: [int]}}}]}, next: {_nodetype: UnaryOp,  expr: {_nodetype: ID,  name: i}, op: p++}, stmt: {_nodetype: FuncCall, args: {_nodetype: ExprList,  exprs: [{_nodetype: Constant,  type: string, value: "%d"}, {_nodetype: ArrayRef,  name: {_nodetype: ID,  name: a}, subscript: {_nodetype: ID,  name: i}}]},  name: {_nodetype: ID,  name: printf}}}],   decl: {_nodetype: Decl, bitsize: None,  funcspec: [], init: None, name: printAll, quals: [], storage: [], type: {_nodetype: FuncDecl, args: {_nodetype: ParamList,  params: [{_nodetype: Decl, bitsize: None,  funcspec: [], init: None, name: n, quals: [], storage: [], type: {_nodetype: TypeDecl,  declname: n, quals: [], type: {_nodetype: IdentifierType,  names: [int]}}}, {_nodetype: Decl, bitsize: None,  funcspec: [], init: None, name: a, quals: [], storage: [], type: {_nodetype: ArrayDecl,  dim: {_nodetype: ID,  name: n}, dim_quals: [], type: {_nodetype: TypeDecl,  declname: a, quals: [], type: {_nodetype: IdentifierType,  names: [int]}}}}]},  type: {_nodetype: TypeDecl,  declname: printAll, quals: [], type: {_nodetype: IdentifierType,  names: [void]}}}}, param_decls: None}'
)

print(resullt)

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', threaded=True)
