import uuid

import nltk
from nltk.stem.snowball import SnowballStemmer


class Cluster(object):
    def __init__(self, docList):
        self.id = str(uuid.uuid4())
        self.docList = docList

    def __str__(self):
        summaries = ""
        for doc in self.docList: summaries += "\t{}: {}\n".format(doc.id, doc.text[:50])
        return "Cluster ID: {}, contains {} documents, which are:\n{}".format(self.id, len(self.docList), summaries)