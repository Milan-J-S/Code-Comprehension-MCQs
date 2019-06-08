import sqlite3
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import re
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
from random import shuffle




def extractFunction(name, code):

    str = re.compile(".*"+name+"\(.*\)[^;]", re.MULTILINE)

    matches = re.search(str, code)
    
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
    # print(res)
    return res

options_per_func =[]

con = sqlite3.connect("database.db")
cur = con.cursor()
cur.execute("SELECT comment,line,code  FROM CodeComments")
comments_options = cur.fetchall()

for row in comments_options:

        data = str(open('static/data/' + row[2]).read())

        if(row[1] == ""):
            function = data
        else:
            function = extractFunction(row[1], data)
            function = function.replace(row[1], "foo")

        doc2vec_model = Doc2Vec.load("d2v.model")
        v1 = doc2vec_model.infer_vector(list(filter(lambda x: x not in stop,word_tokenize(row[0].lower()))), steps=1000)


        similar_docs = doc2vec_model.docvecs.most_similar([v1], topn=len(doc2vec_model.docvecs))

        options = []
        # print(row[0])
        options.append((row[0],1))
        options_actual = [' '.join(list(filter(lambda x: x not in stop, row[0].split())))]
        i = 1
        while(len(options)!=6):
            if(' '.join(list(filter(lambda x: x not in stop, similar_docs[i][0].split()))) not in options_actual):
                options.append((similar_docs[i][0], 0))
                options_actual.append(' '.join(list(filter(lambda x: x not in stop, similar_docs[i][0].split()))))
            i+=1


        shuffle(options)


        options_per_func.append((row[1], options, function))

        print(function)
        print("\n\n")
        print(options_actual)
        print("------------------------------------------\n\n\n")

# print(options_per_func)