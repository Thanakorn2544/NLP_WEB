from gensim.models.tfidfmodel import TfidfModel
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
import os, shutil
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import itertools
import spacy
from spacy import displacy
import pandas as pd
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import pathlib

class process:
    def __init__(self):
        self.preprocessed = self.preprocess()
    
    def preprocess(self):
        file = request.files.getlist('file_name')
        articles = []
        for article in textList:
            filename = secure_filename(file.filename)
            path = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/uploads/{filename}"
            f = open(path,"r")
            article = f.read()
            tokens = word_tokenize(article)
            lower_tokens = [t.lower() for t in tokens]
            alpha_only = [t for t in lower_tokens if t.isalpha()]
            no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
            wordnet_lemmatizer = WordNetLemmatizer()
            lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
            articles.append(lemmatized)
        return articles

    def clear_folder_uploads(self):
        path = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/uploads/"
        for file in os.listdir(path) :
            try:
                if os.path.isfile(path+file) or os.path.islink(path+file):
                    os.unlink(path+file)
                elif os.path.isdir(path+file):
                    shutil.rmtree(path+file)
            except Exception as e:
                gg = 0

    def save_file(self, filename):
        files = request.files.getlist('file_name')
        for file in files :
            filename = secure_filename(file.filename)
            path = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/uploads/{filename}"
            file.save(path)
    
    def bag_of_words(self):
        articles = self.preprocessed
        dictionary = Dictionary(articles)
        corpus = [dictionary.doc2bow(a) for a in articles]
        total_word_count = defaultdict(int)
        for word_id, word_count in itertools.chain.from_iterable(corpus):
            total_word_count[word_id] += word_count
        topBOW = []
        sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1],reverse=True)
        for word_id, word_count in sorted_word_count[:5]:
            topBOW.append(f"{dictionary.get(word_id)} {word_count}")
        return topBOW
    
    def TF_IDF(self):
        articles = self.preprocessed
        dictionary = Dictionary(articles)
        corpus = [dictionary.doc2bow(a) for a in articles]
        tfidf = TfidfModel(corpus)
        tfidf_weights = []
        for doc in corpus:
            tfidf_weights += tfidf[doc]
        topTFIDF = []
        sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
        for term_id, weight in sorted_tfidf_weights[:5]:
            topTFIDF.append(f'{dictionary.get(term_id)} {weight}')
        return topTFIDF

    def search_word(self, word):
        word = str(word).lower().strip()
        articles = self.preprocessed

        dictionary = Dictionary(articles)
        wordid =  dictionary.token2id.get(word)
        corpus = [dictionary.doc2bow(a) for a in articles]

        total_word_count = defaultdict(int)
        for word_id, word_count in itertools.chain.from_iterable(corpus):
            total_word_count[word_id] += word_count

        count_of_word = 0
        sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1],reverse=True)
        for word_id, word_count in sorted_word_count:
            if (dictionary.get(word_id) == word):
                count_of_word = word_count
                break

        result = dictionary.get(wordid)
        if result:
            return result,count_of_word
        else:
            return '',0
            #return f'Couldn\'t find the word {word} in the entire article.'

    def name_entity_recognition(self,textlist):
        nlp = spacy.load('en_core_web_sm')

        NERhteml = []
        for text in textlist:
            doc = nlp(text)
            NERhteml.append(displacy.render(doc, style="ent"))
        return NERhteml
    
    def fake_news_detection(self,news,convert_to_label=False):
        model_path = f"{str(pathlib.Path(__file__).parent.resolve().as_posix())}/model/fake-news-bert-base-uncased"
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # prepare our text into tokenized sequence
        inputs = tokenizer(news, padding=True, truncation=True, max_length=512,return_tensors="pt")
        # perform inference to our model
        outputs = model(**inputs)
        # get output probabilities by doing softmax
        probs = outputs[0].softmax(1)
        # executing argmax function to get the candidate label
        d = {
            0: "reliable",
            1: "fake"
        }
        if convert_to_label:
            return d[int(probs.argmax())]
        else:
            return int(probs.argmax())