from flask import Flask, request, jsonify,session, g, redirect, url_for, abort, \
    render_template, flash

app = Flask(__name__)
# CORS(app)

@app.route("/")
def start():
   return render_template('codeUpload.html')

if __name__ == '__main__':
   
   app.config['TEMPLATES_AUTO_RELOAD'] = True
   app.run(host = '0.0.0.0')
