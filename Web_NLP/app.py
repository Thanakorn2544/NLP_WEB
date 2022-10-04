from operator import le
import os, shutil
from flaskext.markdown import Markdown
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from natural_language_processing import process as pro
import pathlib

app = Flask(__name__)
Markdown(app)
app.config["UPLOAD_PATH"] = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/uploads/"
path = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/uploads/"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_file", methods=["GET","POST"])
def upload_file():
    if request.method == 'POST' and len(request.files.getlist('file_name')) != 0:
        process = pro(request.files.getlist('file_name'))
        process.clear_folder_uploads()
        process.upload_file(request.files.getlist('file_name'))
        process.preprocessed()


@app.route('/search_word', methods=["GET","POST"])
def search_word():


@app.route('/fake_news_detection', methods=["GET","POST"])
def fake_news_detection():


if __name__ == '__main__':
    app.run(debug=True)