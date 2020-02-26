import uuid

from NLP.Document import Document
from NLP.utils import PageRank, SentenceProcessor, SimilarityCalculator

#
import progressbar
#

class Corpus:
    def __init__(self, docJson):
        self.SP = SentenceProcessor()
        self.SC = SimilarityCalculator(self.SP)
        self.docList, self.similarityMatrix, self.clusters =  [], None, None
        self.deleteDuplicates(docJson)

    def deleteDuplicates(self, docJson):
        bar = progressbar.ProgressBar(len(docJson)-1)
        bar.start()
        for i, doc in enumerate(docJson):
            id = doc['id'] if 'id' in doc.keys() else str(uuid.uuid4())
            title = doc['title'] if 'title' in doc.keys() else "No title"
            self.docList.append(Document({'text': doc['text'], 'id': id, 'title': title}, self.SP, self.SC))
            bar.update(i)

    def setClusters(self, clusters):
        self.clusters = clusters

    def findCluster(self, docId):
        for cluster in self.clusters:
            if docId in [doc.id for doc in cluster.docList]: return cluster
        return None