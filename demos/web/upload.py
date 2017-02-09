#!/usr/bin/env python

import os
from flask import Flask, request, redirect, url_for, g
from werkzeug import secure_filename
import json
import numpy as np
from compare import getRep


UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
store = dict()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/list", methods=['GET', 'POST'])
def people():
    """List stored peoples"""
    return json.dumps(store.keys())


@app.route("/distance/<u1>/<u2>", methods=['GET', 'POST'])
def distance(u1, u2):
    """Evaluate the distance between two existing peoples.

        :param  u1 - first person's filename given from /list
        :param  u2 - second person's filename given from /list
    """
    assert u1 in store
    assert u2 in store
    return json.dumps(np.dot(store[u1], store[u2]))


@app.route("/upload", methods=['GET', 'POST'])
def upload():
    """Upload one or more photos to the local store.

    :return:
    """
    if request.method == 'POST':
        file_ = request.files['file']
        assert file_
        assert allowed_file(file_.filename)
        if file_ and allowed_file(file_.filename):
            filename = secure_filename(file_.filename)
            dpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_.save(dpath)
            file_.close()
            fname, ext = os.path.splitext(filename)
            store[fname] = getRep(dpath)
            return redirect(url_for('upload'))
        else:
            raise NotImplementedError

    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    <p>%s</p>
    """ % "<br>".join(os.listdir(app.config['UPLOAD_FOLDER'],))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
