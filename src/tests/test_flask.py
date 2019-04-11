import flask

app = flask.Flask(__name__)

def test_something():
    with app.test_request_context('/?name=Peter'):
        assert flask.request.path == '/'
        assert flask.request.args['name'] == 'Peter'