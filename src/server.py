import random
import string

from flask import Flask, request, jsonify, session, g, redirect, url_for, abort, \
    render_template, make_response
import sqlite3
import numpy as np
import re
from random import shuffle

user_codes_matrix = []

import json
from nltk.tokenize import word_tokenize
import nltk
nltk.download('corpus')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
import legacy
from legacy import AttentionDecoder
import os
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from keras.models import load_model
app = Flask(__name__)

code_to_mode = {
'text/x-csrc': "C",
'python' : "Python",
    'text/x-c++src' : "C++",
'text/html' : 'HTML',
    'text/x-java': "Java",
    'javascript' : "Javascript"

}

code_to_ext = {
'text/x-csrc': "c",
'python' : "py",
    'text/x-c++src' : "cpp",
'text/html' : 'html',
    'text/x-java': "java",
    'javascript' : "js"

}
# print("connection recieved")
# CORS(app)

def convertToDict(x):
    obj = {}
    obj['filename'] = x[0]
    obj['title'] = x[1]
    if len(x)>2:
        obj['difficulty'] = x[2]
        if len(x)>3:
            obj['lang'] = code_to_mode[x[3]]
    return obj

def convertToFilesDict(x):
    obj = {}
    obj['description'] = x[0]
    obj['type'] = code_to_mode[x[1]]
    obj['filename'] = x[2]
    return obj



new_users_dict = {}
new_codes_dict = {}

difficulty_matrix = []
adaptive_difficulty_matrix = []

new_users_reverse_map = []
new_codes_reverse_map = []

doc2vec_model = None


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

    return ' '.join(res)




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
    print(AST)

    tags = dict()

    comments_Set = re.findall("//.*\n.*", code)
    print(comments_Set)

    comments_Set.extend(re.findall("/*[^\*]*\*/\n.*", code))
    print(comments_Set)



    comments = []

    for item in AST['ext']:

        if item['_nodetype'] == "FuncDef":

            curfunc = str(item)
            curfunc = re.sub(r"\'coord\': [^,]+,", "", curfunc)
            curfunc = curfunc.replace("\'", "")

            for tag in KNN(curfunc):
                if(tag not in tags):
                    tags[tag] = 0
                tags[tag]+=1

            name = item['decl']['name']

            existing = ''

            for comment in comments_Set:
                if(name in comment):
                    if(comment[1] == '/'):
                        existing = comment.split("\n")[0][2:]
                    else:
                        existing = comment.split("*/")[0]

            extractFunction(name, code)

            if(existing  == ''):
                comments.append((generateComments(curfunc), name ))
            else:
                comments.append( (existing, name) )

    tags = map(lambda kv: kv[0], sorted(tags.items(), key=lambda kv: kv[1], reverse=True))
    return (list(tags), comments)


@app.route("/upload")
def start():
    return render_template('codeUpload.html', username=request.cookies.get("user", "Login/Sign Up").split("@")[0])


@app.route("/showCode", methods=["GET", "POST"])
def showCode():
    global user_codes_matrix
    filename = request.args['filename']
    difficulty = request.args.get('difficulty',0)
    rows = fetchConvos(filename)
    full_filename = 'static/data/' + filename
    # print(full_filename)

    f = open(full_filename)

    username = request.cookies.get("user", "Login/Sign Up").split("@")[0]

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT Views FROM CodeViews WHERE user=(?) AND code=(?)", (username, filename))
    codeviews = cur.fetchall()
    if (len(codeviews) > 0):
        cur.execute("UPDATE CodeViews SET views=(?) WHERE user=(?) AND code=(?)",
                    (codeviews[0][0] + 1, username, filename))
    else:
        cur.execute("INSERT into CodeViews values (?,?,?,?) ", (filename, username, 0, 1))
    con.commit()

    user_codes_matrix[new_users_dict[username]][new_codes_dict[filename]] += 1


    options_per_func =[]

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT comment,line  FROM CodeComments WHERE code = (?)",(filename,))
    comments_options = cur.fetchall()

    data = str(f.read())

    print(comments_options)
    for row in comments_options:

        if(row[1] == ""):
            function = data
        else:
            function = extractFunction(row[1], data)
            function = function.replace(row[1], "foo")

        doc2vec_model = Doc2Vec.load("d2v.model")
        v1 = doc2vec_model.infer_vector(list(filter(lambda x: x not in stop,word_tokenize(row[0].lower()))), steps=1000)


        similar_docs = doc2vec_model.docvecs.most_similar([v1], topn=len(doc2vec_model.docvecs))

        options = []
        print(row[0])
        options.append((row[0],1))
        options_actual = [row[0]]
        i = 1
        while(len(options)!=4):
            if(comments[int(similar_docs[i][0])] not in options):
                options.append((comments[int(similar_docs[i][0])], 0))
                options_actual.append(comments[int(similar_docs[i][0])])
            i+=1


        shuffle(options)


        options_per_func.append((row[1], options, function))

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT description, lang FROM Codes WHERE filename=(?)", (filename,))

    description = cur.fetchall()
    new_description = 'no_desc'
    if (description != []):
        new_description = '_'.join(description[0][0].split())+"."+code_to_ext[description[0][1]]

    print(description)
    return render_template('codeView.html',
                           data=data,
                           rows=rows,
                           filename=filename,
                           username=username,
                           options = options_per_func,
                           difficulty = float(difficulty),
                           description = new_description,
                            )

def extractFunction(name, code):

    str = re.compile(".*"+name+"\(.*\)[^;]", re.MULTILINE)

    matches = re.search(str, code)
    print(matches.end())
    print(matches)
    end = matches.end()
    while(code[end]!='{'):
        end+=1

    relevant_code = code[end:]
    counter = 0

    code_actual = ''

    for char in relevant_code:
        if(char == '{'):
            counter+=1
        elif(char == '}'):
            counter-=1
        code_actual+=char
        if(counter==0):
            break

    # print(matches.expand())
    res = matches.group(0)+code_actual
    print(res)
    return res



def recommendCodes(user):
    user_index = new_users_dict[user]

    user_has_viewed = set()

    global user_codes_matrix

    print(user_codes_matrix)
    for i in range(len(new_codes_dict)):
        print(i)
        print(user_codes_matrix[user_index])
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


def prepareAll(username, lang, difficulty):
    code_desc = {}
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    if(lang == ''):
        cur.execute("SELECT filename, description,lang  FROM Codes")

    else:
        cur.execute("SELECT filename, description,lang  FROM Codes where lang=(?)", (lang,))

    rows = cur.fetchall()

    for row in rows:
        code_desc[new_codes_dict[row[0]]] = (row[1], row[2])
    rows = recommendCodes(username)

    global difficulty_matrix

    users_to_match = dict()

    for i in range(len(difficulty_matrix[new_users_dict[username]])):
        if (difficulty_matrix[new_users_dict[username]][i] > 0):
            for j in range(len(new_users_dict)):
                if (difficulty_matrix[j][i] > 0):
                    if (j not in users_to_match):
                        users_to_match[j] = 0
                    users_to_match[j] += abs(difficulty_matrix[j][i] - difficulty_matrix[new_users_dict[username]][i])

    print(users_to_match)
    sorted_users = sorted(users_to_match.items(), key=lambda kv: kv[1])[1:]

    indices = list(map(lambda x: x[0], sorted_users))
    print(indices)

    code_difficulties = []

    print(difficulty_matrix)

    for i in range(len(new_codes_dict)):
        if (difficulty_matrix[new_users_dict[username]][i] == 0):
            adaptive_score = 0
            count = 0

            for index in indices:
                if (difficulty_matrix[index][i] > 0):
                    count += 1
                adaptive_score += difficulty_matrix[index][i]

            if (count > 0):
                code_difficulties.append(adaptive_score / count)
            else:
                code_difficulties.append(0)

        else:
            code_difficulties.append(difficulty_matrix[new_users_dict[username]][i])

        adaptive_difficulty_matrix[new_users_dict[username]][i] = code_difficulties[-1]

    print(code_difficulties)

    rows = [(new_codes_reverse_map[x[0]], code_desc[x[0]][0], code_difficulties[x[0]], code_desc[x[0]][1]) for x in
            rows if x[0] in code_desc ]

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    if(lang == ''):
        cur.execute(
        "SELECT filename, description, lang FROM Codes c INNER JOIN CodeViews v WHERE user=(?) AND c.filename=v.code",
        (username,))
    else:
        cur.execute(
            "SELECT filename, description, lang FROM Codes c INNER JOIN CodeViews v WHERE user=(?) AND c.filename=v.code AND lang = (?)",
            (username,lang))
    seen = cur.fetchall()

    rows.extend([(x[0], x[1], code_difficulties[new_codes_dict[x[0]]], x[2]) for x in seen])

    items = list(map(convertToDict, rows))


    if(difficulty != ''):
        items = list(filter( lambda x: checkDifficulty(x,difficulty) , items ))

    return(items)

def checkDifficulty( item, difficulty ):
    difficulty = int(difficulty)
    if(difficulty == 1):
        return item['difficulty'] <= 1.5
    elif(difficulty == 3):
        return item['difficulty'] >= 2.5
    elif(difficulty == 2):
        return (item['difficulty'] < 2.5 and item['difficulty'] > 2.5)


@app.route("/")
def showAll():
    username = request.cookies.get("user", "Login/Sign Up").split("@")[0]
    items = prepareAll(username, '','')

    resp = make_response(render_template('showResources.html', items=items, username=username))

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
    comments = list(payload.get('comments', ''))
    func = list(payload.get('func', ''))

    print(tags)
    print(comments)

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

    print("tags for this code are :", tags)

    for tag in tags:
        cur = con.cursor()
        exists = cur.execute("SELECT * from tags where tag = (?)", (tag,)).fetchall()
        if len(exists) == 0:
            cur.execute("INSERT INTO Tags VALUES (?)", (tag,))
        cur.execute("INSERT INTO CodeTags VALUES (?,?)", (filename, tag))
    con.commit()

    point = 10

    for i in range(len(comments)):
        point += 1
        cur = con.cursor()
        comment = comments[i]
        exists = cur.execute("SELECT * from comments where comment = (?)", (comment,)).fetchall()
        if len(exists) == 0:
            cur.execute("INSERT INTO Comments VALUES (?)", (comment,))
        cur.execute("INSERT INTO CodeComments VALUES (?,?,?)", (filename, comment, func[i]))
    con.commit()
    con.close()

    updatePoints(user, point )

    print(content)

    global user_codes_matrix
    global difficulty_matrix
    global adaptive_difficulty_matrix
    global new_codes_dict
    global new_codes_reverse_map

    print(user_codes_matrix.shape)

    print(len(new_users_dict))

    user_codes_matrix = np.insert(user_codes_matrix, len(new_codes_dict), 0, axis=1)
    difficulty_matrix = np.insert(difficulty_matrix, len(new_codes_dict), 0, axis=1)
    adaptive_difficulty_matrix = np.insert(adaptive_difficulty_matrix, len(new_codes_dict), 0, axis=1)

    print(user_codes_matrix.shape)

    # for i in range(len(new_users_dict)):
    #     np.concatenate(np.array(user_codes_matrix)[i],np.zeros(1))

    new_codes_dict[filename] = len(new_codes_dict)
    new_codes_reverse_map.append(filename)


    # global code_tensors
    # tags = generateTags(content)

    return jsonify(filename=filename)


@app.route("/getTags", methods=["POST"])
def getTags():
    content = request.form['content']
    ctype = request.form['ctype']

    print(ctype)

    if(ctype=="text/x-csrc"):
        (tags,comments) = generateTags(content)
    else:
        tags = []
        comments = []

    print("tags generated = ", tags)
    print("comments_generated = ", comments)

    return jsonify(tags = tags, comments = comments)




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
    global user_codes_matrix
    global difficulty_matrix
    global adaptive_difficulty_matrix
    global new_users_dict
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

        user_codes_matrix = np.append(user_codes_matrix, [np.zeros(len(new_codes_dict))], axis = 0)
        print(user_codes_matrix)

        difficulty_matrix = np.append(difficulty_matrix, [np.zeros(len(new_codes_dict))], axis = 0)
        adaptive_difficulty_matrix = np.append(adaptive_difficulty_matrix, [np.zeros(len(new_codes_dict))], axis = 0)
        new_users_dict[email.split("@")[0]] = len(new_users_dict)

        resp = jsonify(auth=True)

    resp.set_cookie("user", email.split("@")[0])

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
    cur.execute("SELECT code,user, difficulty, views FROM CodeViews")
    rows = cur.fetchall()

    global difficulty_matrix
    global adaptive_difficulty_matrix
    # global difficulty_nbrs

    difficulty_matrix = np.zeros((len(new_users_dict),len(new_codes_dict)))
    adaptive_difficulty_matrix = np.zeros((len(new_users_dict),len(new_codes_dict)))

    for row in rows:
        user_codes_matrix[new_users_dict[row[1]]][new_codes_dict[row[0]]] = row[3]
        difficulty_matrix[new_users_dict[row[1]]][new_codes_dict[row[0]]] = row[2]
        adaptive_difficulty_matrix[new_users_dict[row[1]]][new_codes_dict[row[0]]] = row[2]


    # difficulty_nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(difficulty_matrix)


    print(user_codes_matrix)


def KNN(code):
    global code_tensors
    global commments

    global code_dict

    code = re.sub("\"[^\"]*\"", "0", code)
    code = re.sub("name: [^,}]+", "name", code)
    code = re.sub("value: [^,}]+", "value", code)
    # code = word_tokenize(code)

    # code_tensor = np.zeros(750)
    # for i in range(min(750, len(code))):
    #     if code[i] in code_dict:
    #         code_tensor[i] = code_dict[code[i]]
    #
    code_vector = code2vec_model.infer_vector([code], steps=1000)

    similar_docs = code2vec_model.docvecs.most_similar([code_vector], topn=10)

    print(similar_docs)

    indices = list(map(lambda x: int(x[0]), similar_docs))
    distances = list(map(lambda x: 1/x[1], similar_docs))

    # distances, indices = nbrs.kneighbors(np.asarray([code_tensor]))

    # print("distances and indices")
    # print(distances)
    # print(indices[0][0])

    words = dict()
    print(comments[indices[0]], "\n\n")
    for j in range(1, len(indices)):

        index = indices[j]
        print(comments[index])

        for word in comments[index].split():
            if (word not in stop):
                if (word not in words):
                    words[word] = 0
                # if(distances[i][j] != 0):
                words[word] += 1000 / ((distances[j] + 1)*(distances[j] + 1))

    words_ordered = sorted(words.items(), key=lambda kv: kv[1], reverse=True)
    tags = [x[0] for x in words_ordered[:5]]
    # print(tags, "\n\n\n")
    return tags


code_tensors = []
comments = []
code_dict = dict()
nbrs = None
difficulty_nbrs = None
indices = []

codes_reverse_map = []
codes_dict = {}


# def recommendForUser(user):
#     con = sqlite3.connect("database.db")
#     cur = con.cursor()
#     cur.execute("SELECT code from CodeViews WHERE user = (?)", (user,))
#
#     rows = cur.fetchall()
#
#     reco = {}
#
#     print(rows)
#     for item in rows:
#         print(codes_dict[item[0]])
#         for code in indices[codes_dict[item[0]]]:
#             if (code not in reco):
#                 reco[code] = 0
#             reco[code] += 1
#
#     print(reco)
#
#     reco_ordered = sorted(reco.items(), key=lambda kv: kv[1], reverse=True)
#     reco_ordered = [x[0] for x in reco_ordered[:10]]
#
#     for rec in reco_ordered:
#         print(codes_reverse_map[rec])


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
            codes.append(code)
            codes_reverse_map.append(f)
            codes_dict[f] = i
            i += 1

            for word in word_tokenize(code):
                code_vocab.add(word)

            comment = open('comments/' + f).read()
            comments.append(comment)

    code_tensors = []

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT comment from Comments" )
    new_comments = cur.fetchall()
    print(new_comments)
    comments.extend(list(map(lambda x: x[0],new_comments)))


    i = 0
    for word in code_vocab:
        code_dict[word] = i
        i += 1

    j = 0


    tagged_data = [TaggedDocument(list(filter(lambda x: x not in stop,word_tokenize(d.lower()))), tags=[str(i)]) for i, d in enumerate(comments)]

    tagged_code = [TaggedDocument([d.lower()], tags=[str(i)]) for i, d in enumerate(codes)]

    max_epochs = 3
    vec_size = 100
    alpha = 0.125

    global doc2vec_model

    doc2vec_model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    doc2vec_model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        doc2vec_model.train(tagged_data,
                    total_examples=doc2vec_model.corpus_count,
                    epochs=doc2vec_model.iter)
        # decrease the learning rate
        doc2vec_model.alpha -= 0.0002
        # fix the learning rate, no decay
        doc2vec_model.min_alpha = doc2vec_model.alpha

    doc2vec_model.save("d2v.model")
    print("Model Saved")


    global code2vec_model

    code2vec_model = Doc2Vec(size=100,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    code2vec_model.build_vocab(tagged_code)

    for epoch in range(max_epochs):
        code2vec_model.train(tagged_code,
                    total_examples=code2vec_model.corpus_count,
                    epochs=code2vec_model.iter)
        # decrease the learning rate
        code2vec_model.alpha -= 0.0002
        # fix the learning rate, no decay
        code2vec_model.min_alpha = code2vec_model.alpha

    code2vec_model.save("c2v.model")
    print("Model Saved")


clusterCodes()

prepareUserMatrix()


@app.route("/search", methods=["POST"])
def search():
    con = sqlite3.connect("database.db")
    cur = con.cursor()

    global new_codes_dict
    global adaptive_difficulty_matrix

    searchTerms = request.form.get("searchTerms", "").split()
    lang = request.form.get("lang", "")
    difficulty = request.form.get("difficulty", "")

    print("lang = ", lang)

    if searchTerms == []:
        items = prepareAll(request.cookies.get("username","Login/Sign Up").split("@")[0], lang, difficulty)
        return jsonify(files=items)


    print(searchTerms)

    files_that_match = {}

    for searchTerm in searchTerms:
        if not lang == '':
            cur.execute("SELECT code from CodeTags JOIN Codes WHERE tag like (?) AND lang = (?) and code=filename", (searchTerm+'%',lang))
        else:
            print("searching all")
            cur.execute("SELECT code from CodeTags WHERE tag like (?)", (searchTerm+'%',))
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
    cur.execute("SELECT filename, description, lang FROM Codes")
    rows = cur.fetchall()
    username = request.cookies.get("user", "Login/Sign Up").split("@")[0]

    for row in rows:
        code_desc[row[0]] = (row[1], row[2])
    # rows = recommendCodes(username)

    print(code_desc)
    files_ordered = [(x[0],
                      code_desc[x[0]][0],
                      adaptive_difficulty_matrix[new_users_dict[username]][new_codes_dict[x[0]]],
                      code_desc[x[0]][1]) for x in files_ordered]

    items = list(map(convertToDict, files_ordered))

    return jsonify(files=items)

@app.route("/difficulty", methods = ["GET","POST"])
def setDifficulty():
    global difficulty_matrix

    user = request.cookies.get('user','').split("@")[0]
    filename = request.form.get('filename')
    difficulty = request.form.get('difficulty', 0)

    difficulty_matrix[new_users_dict[user]][new_codes_dict[filename]] = difficulty

    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("UPDATE CodeViews SET difficulty=(?) WHERE user=(?) AND code=(?)", (difficulty, user, filename))
    con.commit()
    con.close()

    return jsonify(success="success")

@app.route("/profile", methods=["GET","POST"])
def profile():
    username = request.cookies.get("user", "Login/Sign Up")
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT score FROM Points WHERE user=(?)", (username,))
    rows = cur.fetchall()
    score=0
    if len(rows) == 0:
        score = 0
    else:
        score = int(rows[0][0])
    cur.execute("SELECT * FROM Points ORDER BY score DESC")
    ranked = cur.fetchall()
    print(ranked)
    con.commit()
    con.close()
    username = username.split("@")[0]

    viewed,stats = getViewed(username)

    return render_template('profile.html',
                           username=username,
                           score=score,
                           ranked=ranked,
                           viewed = viewed,
                           stats = stats)

@app.route("/addPoints", methods = ["GET","POST"])
def addPoints():
    user = request.cookies.get('user','')
    point = int(request.form.get('addscore'))

    updatePoints(user,point)

    return jsonify(success="success")

def updatePoints(user,point):
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT score FROM Points WHERE user=(?)", (user,))
    rows = cur.fetchall()
    if len(rows) == 0:
        print("CURRENT SCORE IS: ", point)
        cur.execute("INSERT INTO Points VALUES (?,?)", (user, point))
    else:
        score = int(rows[0][0])
        print("EXISTING SCORE IS: ", score)
        score = score + point
        cur.execute("UPDATE Points SET score=(?) WHERE user=(?)", (score, user))
    con.commit()
    con.close()

#you will at some point think this is useless and delete. Bad idea
result = generateComments('{_nodetype: FuncDef, body: {_nodetype: Compound, block_items: [{_nodetype: For, cond: {_nodetype: BinaryOp,  left: {_nodetype: ID,  name: i}, op: <, right: {_nodetype: ID,  name: n}},  init: {_nodetype: DeclList,  decls: [{_nodetype: Decl, bitsize: None,  funcspec: [], init: {_nodetype: Constant,  type: int, value: 0}, name: i, quals: [], storage: [], type: {_nodetype: TypeDecl,  declname: i, quals: [], type: {_nodetype: IdentifierType,  names: [int]}}}]}, next: {_nodetype: UnaryOp,  expr: {_nodetype: ID,  name: i}, op: p++}, stmt: {_nodetype: FuncCall, args: {_nodetype: ExprList,  exprs: [{_nodetype: Constant,  type: string, value: "%d"}, {_nodetype: ArrayRef,  name: {_nodetype: ID,  name: a}, subscript: {_nodetype: ID,  name: i}}]},  name: {_nodetype: ID,  name: printf}}}],   decl: {_nodetype: Decl, bitsize: None,  funcspec: [], init: None, name: printAll, quals: [], storage: [], type: {_nodetype: FuncDecl, args: {_nodetype: ParamList,  params: [{_nodetype: Decl, bitsize: None,  funcspec: [], init: None, name: n, quals: [], storage: [], type: {_nodetype: TypeDecl,  declname: n, quals: [], type: {_nodetype: IdentifierType,  names: [int]}}}, {_nodetype: Decl, bitsize: None,  funcspec: [], init: None, name: a, quals: [], storage: [], type: {_nodetype: ArrayDecl,  dim: {_nodetype: ID,  name: n}, dim_quals: [], type: {_nodetype: TypeDecl,  declname: a, quals: [], type: {_nodetype: IdentifierType,  names: [int]}}}}]},  type: {_nodetype: TypeDecl,  declname: printAll, quals: [], type: {_nodetype: IdentifierType,  names: [void]}}}}, param_decls: None}'
)

@app.route("/download", methods=["GET"])
def download():
    return app.send_static_file('data/'+request.args.get("filename",""))

def getViewed(user):
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("SELECT DISTINCT description, lang, filename FROM CodeViews v INNER JOIN Codes c WHERE user=(?)", (user,))
    rows = cur.fetchall()

    rows = list(map(convertToFilesDict, rows))

    stats = {}

    for row in rows:
        if(row['type'] not in stats):
            stats[row['type']] = 0
        stats[row['type']] += 1

    print("rows = ", rows)
    print("stats = " , stats )

    return (rows, stats)

def __init__():
    return app

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', threaded=True)
