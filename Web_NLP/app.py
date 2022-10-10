from operator import le
import os, shutil
from flaskext.markdown import Markdown
from flask import Flask, render_template, request, url_for, Markup, redirect, session
from flask_session import Session
from werkzeug.utils import secure_filename
from natural_language_processing import process as pro
import pathlib

app = Flask(__name__)
Markdown(app)
app.config["UPLOAD_PATH"] = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/uploads/"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
path = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/uploads/"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_file", methods=["GET","POST"])
def upload_file():
    if request.method == 'POST' and len(request.files.getlist('file_name')) != 0:
        session["name"] = request.files.getlist('file_name')
        pro.clear_folder_uploads()
        processing = pro(session["name"])
    return render_template("index.html",top_word = processing.bag_of_words(), top_tfidf = processing.TF_IDF(), result_spy = processing.name_entity_recognition())


@app.route('/search_word', methods=["GET","POST"])
def search_word():
    filename = session["name"]
    processing = pro(filename)
    return render_template("index.html",top_word = processing.bag_of_words(), top_tfidf = processing.TF_IDF(), result_spy = processing.name_entity_recognition(), key_words = processing.search_word(request.form['key_word']))

@app.route('/fake_news_detection', methods=["GET","POST"])
def fake_news_detection():
    filename = session["name"]
    processing = pro(filename)
    return render_template("index.html",top_word = processing.bag_of_words(), top_tfidf = processing.TF_IDF(), result_spy = processing.name_entity_recognition(),show_news = processing.fake_news_detection(request.form['text_news']))

if __name__ == '__main__':
    app.run(debug=True)