import numpy as np 
import pandas as pd
from gensim import corpora
from gensim import similarities
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from gensim.models import TfidfModel

from NLP.Cluster import Cluster
from NLP.utils import SentenceProcessor

class AClustering(object):
    def __init__(self, corpus):
        self.SP = SentenceProcessor(language='es')
        self.simDf, self.sims = self.getClusters(corpus)

    def getSimilarityMatrixV(self, corpus):
        mat = np.ones( (len(corpus.docList), len(corpus.docList)) )
        for i, doc_i in enumerate(corpus.docList):
            for j, doc_j in enumerate(corpus.docList):
                if j <= i: continue
                mat[i,j] = doc_i.getSimilarity(doc_j)
                mat[j,i] = doc_j.getSimilarity(doc_i)
        return mat

    def getSimilarityMatrixA(self, corpus):        
        pTxts = [self.SP.getTokenizedSentence(doc.text) for doc in corpus.docList]
        dictionary = corpora.Dictionary(pTxts)
        bows = [dictionary.doc2bow(txt) for txt in pTxts]
        model = TfidfModel(bows)
        sims = similarities.MatrixSimilarity(model[bows])
        return sims

    def getClusters(self, corpus):
        sims = self.getSimilarityMatrixV(corpus)
        sim_df = pd.DataFrame(sims)
        sim_df.columns = range(1,len(corpus.docList)+1)
        sim_df.index = range(1,len(corpus.docList)+1)
        return sim_df, sims

    def getSimilarDocs(self, i):
        v = self.simDf[i+1]
        v = v.drop([i+1])
        v_sorted = v.sort_values()
        return v_sorted

    def getChart(self, i, doc):
        v_sorted = self.getSimilarDocs(i)
        v_sorted.plot.barh(x='lab', y='val', rot=0).plot()
        plt.xlabel("Score")
        plt.ylabel("News")
        plt.xlim((0,1))
        plt.title("Similarity")
        plt.savefig("./Vizs/htmls/img/{}.png".format(doc.id), bbox_inches='tight')
        # plt.savefig("./Vizs/htmls/img/{}.png".format(i), bbox_inches='tight')
        plt.clf()

    def getJson(self, i, doc):
        self.getChart(i, doc)
        return "./Vizs/htmls/img/{}.png".format(doc.id)

    def getDendogram(self):
        z = hierarchy.linkage(self.sims, 'ward')
        hierarchy.dendrogram(z, leaf_font_size=8, labels=self.simDf.index, orientation='left')
        plt.savefig("./Vizs/htmls/img/dendogram.png", bbox_inches='tight')
        plt.clf()

    def viz(self, i, doc):
        self.getChart(i,doc)
        html = """
            <div id="images" class="_flex _row">
                {}{}
            </div>""".format('<img src="img/dendogram.png" class="cluster" />', '<img src="img/{}.png" class="cluster" />'.format(i))
        return html


class VClustering(object):
    """
        Helper class computing clusters with a corpus of doc
        Attributes:
            SP (class doc.SentenceProcessor): Helper class for processing text
    """
    def __init__(self):
        print("Instantiating Clustering")
        self.docList = None
        self.clusters = {}
        self.id2idx = { }

    def findClusters(self, similarityThreshold = 0.70):
        clusters = []
        for doc in self.docList:
            isClustered = False
            for i, cluster in enumerate(clusters):
                for docId in cluster:
                    clusterdoc = self.docList[self.id2idx[docId]]
                    if doc.getSimilarity(clusterdoc) > similarityThreshold:
                        cluster.append(doc.id)
                        isClustered = True
                        break
            if not isClustered: clusters.append([doc.id])
        return clusters
        

     
    def getClusters(self, corpus, similarityThreshold=0.7):
        finalClusters, self.docList = [], corpus.docList
        self.id2idx = { doc.id: idx for idx, doc in enumerate(self.docList) }
        clusterList = self.findClusters(similarityThreshold=similarityThreshold)
        for cluster in clusterList:
            clusterdocs = [ self.docList[self.id2idx[nId]] for nId in cluster ]
            finalClusters.append(Cluster(docList = clusterdocs))
        return finalClusters
