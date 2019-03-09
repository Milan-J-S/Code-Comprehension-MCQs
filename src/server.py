from flask import Flask, request, jsonify, session, g, redirect, url_for, abort, \
    render_template, flash, send_from_directory

from os import listdir

app = Flask(__name__)


# CORS(app)

@app.route("/")
def start():
    return render_template('codeUpload.html')


@app.route("/showCode", methods=["GET", "POST"])
def showCode():
    filename = request.args['filebirthname']

    filename = 'static/data/' + filename
    print(filename)
    with open(filename) as f:
        return render_template('codeView.html', data=str(f.read()))

    # @app.route("/getFile", methods=["GET","POST"])


# def getFile():
#     with open('static/data/test.c') as f : 
#         return jsonify(data = str(f.read()), mode = "text/x-csrc" )

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


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0')
