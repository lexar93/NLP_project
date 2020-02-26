import numpy as np

from NLP.utils import PageRank, SentenceProcessor, SimilarityCalculator

class Summarizer(object):
    def __init__(self):
        print("Instantiating Summarizer")
        self.SP = SentenceProcessor()
        self.SC = SimilarityCalculator(self.SP)


    # Computes the np.array that represents the graph and returns it
    def computeGraph(self, body, maxSentences=100, duplicateCoef=0.7):
        #Document preprocessing
        bodyCopy = ' '.join("{}".format(body).replace('\n', ' ').split())
        ppd = list(self.SP.getPreprocessedDocument(bodyCopy))
        originalSents, processedSents = ([doc[0] for doc in ppd], [doc[1] for doc in ppd])
        if len(processedSents) > maxSentences:
            originalSents = originalSents[:maxSentences]
            processedSents = processedSents[:maxSentences]
        #Graph Computation  
        edgeWeights = np.zeros ( (len(processedSents), len(processedSents)) )
        duplIndxs = []
        for i, sent in list(enumerate(processedSents)):
            for j, sent1 in list(enumerate(processedSents))[i+1:]:
                weight = self.SC.getJaccardSimilarity(sent, sent1)
                if weight > duplicateCoef:
                    if not j in duplIndxs: duplIndxs.append(j)
                else:
                    edgeWeights[i, j] = weight
                    edgeWeights[j, i] = weight
        #Duplicate deletion
        edgeWeights = np.delete(edgeWeights, duplIndxs, axis=0) 
        edgeWeights = np.delete(edgeWeights, duplIndxs, axis=1) 
        for i, idx in enumerate(duplIndxs): del(originalSents[idx-i])
        return edgeWeights, originalSents

    # Takes the body and computes its summary of a maximum of numSentences sentences
    def getSummary(self, doc, numSentences=3):
        text = doc.text
        docGraph, originalSentences = self.computeGraph(text)
        if not originalSentences: return ''
        PR = PageRank(docGraph)
        sentsProbs = list(PR.powerIteration())

        ranking = [ (rank, idx) for idx, rank in list(enumerate(sentsProbs)) ] #Pair (rank, idx) list
        maxVal = max([rank for rank, idx in ranking])
        ranking.sort(reverse=True) # Sort decr by rank
        ranking = ranking[:numSentences] # Get the numSentences with higher rank
        ranking.sort(key=lambda i: i[1]) # Re-Sort as they appear in the body

        summarySentences = [originalSentences[idx] for rank, idx in ranking ]
        metadata = [ (sent, rank/maxVal) for sent, (rank, idx) in zip(summarySentences, ranking)]

        summary = ' '.join(summarySentences)
        return summary, metadata

    def getJson(self, doc):
        summary, sentScores = self.getSummary(doc)
        return {"summary": summary, "sentenceScores": sentScores}


    def viz(self, doc):
        html = """<p>{}</p>""".format(doc.viz())

        html = ' '.join([c for c in html.split() if c])

        try: _, metadata = self.getSummary(doc)
        except: return ""

        for phrase, punct in metadata:
            html = html.replace(phrase, "<span class='important-phrase'>{} <span class='punctuation'>{}</span></span>".format(phrase, round(punct, 3)))

        return html