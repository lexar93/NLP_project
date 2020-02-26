import pickle
import pyLDAvis
import warnings
import numpy as np
from NLP.utils import SentenceProcessor
from pyLDAvis import sklearn as sklearn_lda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as SkLearnLDA


class LDA(object):
    def __init__(self, corpus, numTopics=20, load=True, language='en'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            print("Instantiating Latent Dirichlet Allocation (Topic Modeling)")
            if not load:
                # self.SP = SentenceProcessor(language=language)
                # self.countVectorizer = CountVectorizer(stop_words=self.SP.stopWords, lowercase=True, strip_accents='ascii')
                # self.countData = self.countVectorizer.fit_transform( [self.SP.getProcessedSentence(doc.text) for doc in corpus.docList] )
                # with open("./modelsData/LDA/countVectorizer_59k.pkl", 'wb') as f: pickle.dump(self.countVectorizer, f)
                # with open("./modelsData/LDA/countData_59k.pkl", 'wb') as f: pickle.dump(self.countData, f)

                with open("./modelsData/LDA/countVectorizer_59k.pkl", 'wb') as f: self.countVectorizer = pickle.load(f)
                with open("./modelsData/LDA/countData_59k.pkl", 'wb') as f: self.countData = pickle.load(f)

                self.lda = SkLearnLDA(n_components=numTopics, n_jobs=3, max_iter=100, verbose=1, random_state=0)
                self.lda.fit(self.countData)
                with open("./modelsData/LDA/SKLearnLDAModel.pkl", 'wb') as f: pickle.dump(self.lda, f)
                self.ldaModel = sklearn_lda.prepare(self.lda, self.countData, self.countVectorizer)
                with open("./modelsData/LDA/SKLearnLDA.pkl", 'wb') as f: pickle.dump(self.ldaModel, f)
            else: 
                with open("./modelsData/LDA/SKLearnLDA_59k_100it.pkl", "rb") as f: self.ldaModel = pickle.load(f)
                with open("./modelsData/LDA/SKLearnLDAModel_59k_100it.pkl", 'rb') as f: self.lda =  pickle.load(f)       
    
    def viz(self):
        html = pyLDAvis.prepared_data_to_html(self.ldaModel)
        return html
