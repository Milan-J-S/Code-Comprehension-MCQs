from flask import Flask, request, jsonify,session, g, redirect, url_for, abort, \
    render_template, flash , send_from_directory
import os
import random 
import json
import string
import re
app = Flask(__name__)


@app.route("/")
def start():
   return render_template('helper.html')

@app.route("/submit", methods = ["POST", "GET"])
def data():
    code = request.form.get("code","")
    comments = request.form.get("comments","")

    doProcessing(code, comments)

    print(code, comments)
    return render_template('helper.html')

def doProcessing( code , comments ):
    f = open("cleaned.txt",'w+')
    f.write(str(code))
    
    f.close()
    os.system("pcpp cleaned.txt --line-directive > preprocessed.txt")
    os.system("python parseToJson.py preprocessed.txt > extracted.txt")

    f = open("extracted.txt", "r")
    AST = json.loads(f.read())
    print(AST)

    newFilename = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

    curfunc = str(AST['ext'][0])
    curfunc = re.sub(r"\'coord\': [^,]+,","", curfunc)
    curfunc = curfunc.replace("\'","")

    toWrite = open('code/'+newFilename,'w+')
    toWrite.write(curfunc)
    commentWrite = open('comments/'+newFilename,'w+')
    commentWrite.write( comments )


if __name__ == '__main__':
   
   app.config['TEMPLATES_AUTO_RELOAD'] = True
   app.run(host = '0.0.0.0')
