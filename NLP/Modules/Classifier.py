
import re
import nltk
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class Classifier(object):
    def __init__(self, language, train=False):
        warnings.simplefilter("ignore", UserWarning)
        print("Instantiating classifier for language {}...".format(language))
        self.language = language
        self.dataPaths = {
            "es": "./Data/ONU/subset_spanish.csv",
            "en": "./Data/ONU/subset_english.csv",
            "fr": "./Data/ONU/subset_french.csv",
            "ru": "./Data/ONU/subset_russian.csv"
        }
        self.vocabPaths = {
            "es": "./Data/ONU/vocab_spanish.json",
            "en": "./Data/ONU/vocab_english.json",
            "fr": "./Data/ONU/vocab_french.json",
            "ru": "./Data/ONU/vocab_russian.json"
        }
        self.la2lang = {'es':'spanish','en':'english','fr':'french','ru':'russian'}
        self.dataset = pd.read_csv(self.dataPaths[self.language])
        with open(self.vocabPaths[self.language], "r", encoding="utf8") as f: self.vocab = json.load(f)
        self.stemmer = SnowballStemmer(self.la2lang[self.language])
        self.stopwords = stopwords.words(self.la2lang[self.language])
        self.pDocs, self.y = self.processCorpus()
        self.features, self.wordrel = self.featurize()
        self.model = RandomForestClassifier(max_depth=100, n_estimators=600, max_features=320) 
        if train: self.train()
        else: self.load()
        print("{} model loaded...".format(language))

    def featurize(self):
        vectorizer = CountVectorizer(max_features=3500, min_df=5, max_df=0.7, stop_words=self.stopwords)
        X_f = vectorizer.fit_transform(self.pDocs).toarray()
        words = vectorizer.get_feature_names()

        X = np.zeros( (len(self.pDocs), len(self.vocab)) )
        for dIdx, docCount in enumerate(X_f):
            for wIdx, word in enumerate(self.vocab):
                if word in words:
                    wIdx2 = words.index(word)
                    X[dIdx][wIdx] = docCount[wIdx2]

        tfidftransformer = TfidfTransformer()
        X = tfidftransformer.fit_transform(X).toarray()

        wordrel = []
        for dIdx, dTfIdf in enumerate(X):
            docWordrel = {}
            for wIdx, wScore in enumerate(dTfIdf):
                if wScore == 0: continue
                docWordrel[words[wIdx]] = wScore
            wordrel.append(docWordrel)
        return X, wordrel

    def processCorpus(self):
        rawDocs, pDocs = self.dataset.text, []
        y = self.dataset.Category
        for doc in rawDocs:
            document = self.processDoc(doc)
            pDocs.append(document)
        return pDocs, y

    def getTrainingData(self):
        X = self.featurize()
        X = { idx: doc for idx,doc in enumerate(X) }
        X = pd.DataFrame(X)
        y = pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=21)
        return X_train, X_test, y_train, y_test

    def train(self):
        X_train, X_test, y_train, y_test = self.getTrainingData()
        self.model.fit(X_train, y_train)
        with open("./modelsDta/Classifiers/{}_classifier.pkl".format(self.language), 'wb') as f: 
            pickle.dump(self.model, f)

    def load(self):
        with open("./modelsData/Classifiers/{}_classifier.pkl".format(self.language), 'rb') as f: 
            self.model = pickle.load(f)

    def processDoc(self, doc):
        if not isinstance(doc, str): doc = doc.text
        document = re.sub(r'\W', ' ', str(doc))
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        document = document.lower()
        document = document.split()
        document = [self.stemmer.stem(word) for word in document]
        document = ' '.join(document)
        return document

    def getPredictions(self, idx):
        x = self.features[idx]
        x = x.reshape(1, -1)
        y_pred = self.model.predict( x )
        # return self.y[idx], y_pred[0], self.dataset.text[idx], self.wordrel[idx]
        return {
            "realLabel": self.y[idx],
            "predLabel": y_pred[0],
            "text": self.dataset.text[idx],
            "wordRel": self.wordrel[idx]
        }
