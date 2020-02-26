import json

from NLP.Viz import Viz
from NLP.Corpus import Corpus
from NLP.Modules.LDA import LDA
from NLP.Document import Document
from NLP.Modules.NER import SpacyNER
from NLP.Modules.Summarizer import Summarizer
from NLP.Modules.Clustering import AClustering as Clustering
from NLP.Modules.Classifier import Classifier
from NLP.Modules.SentimentAnalizer import SentimentAnalyzer

#
import os
import progressbar
#



if __name__ == "__main__":
    """
    print("\nLoading news corpus...")
    with open("./Data/news/newsCorpus893.json", "r", encoding="utf8") as f: corpus = Corpus(json.load(f)[222:242])
    
    # print("\nLoading restaurant reviews corpus...")
    # with open("./Data/reviews/reviewsCorpus56_5K.json", "r", encoding="utf8") as f: restRevCorpus = Corpus(json.load(f))
    
    print("\nCorpus loaded...\nInstantiating tools...")

    clustering = Clustering(corpus)
    LDA = LDA(corpus=None, load=True, language='en', numTopics=12)
    sentimentAnalyzer = SentimentAnalyzer()
    entityRecognizer = SpacyNER()

    summarizer = Summarizer()

    print("Pocessing NEWS documents...")
    bar = progressbar.ProgressBar(len(corpus.docList)-1)
    bar.start()

    newsData = []
    clustering.getDendogram()
    for index, doc in enumerate(corpus.docList):
        id = doc.id
        text = doc.text
        clusteringResult = clustering.getJson(index, doc)
        sentAnalysisResult = sentimentAnalyzer.getJson(doc)
        nerResult = entityRecognizer.getJson(doc)
        summaryResult = summarizer.getJson(doc)
        newsData.append({
            "id": id,
            "text": text,
            "clustering": clusteringResult,
            "sentimentAnalysis": sentAnalysisResult,
            "ner": nerResult,
            "summary": summaryResult
        })
        bar.update(index)
    with open("./generatedData/newsData.json", "w", encoding="utf8") as f:
        json.dump(newsData, f, indent=4)
        
    print()
    """



    onuData = []
    classifierES = Classifier('es')
    classifierEN = Classifier('en')
    classifierFR = Classifier('fr')
    classifierRU = Classifier('ru')

    eshit, enhit, frhit, ruhit = 0,0,0,0

    print("Pocessing ONU documents...")
    bar = progressbar.ProgressBar( 79 )
    bar.start()
    for i in range(80):
        predES = classifierES.getPredictions(i)
        predEN = classifierEN.getPredictions(i)
        predFR = classifierFR.getPredictions(i)
        predRU = classifierRU.getPredictions(i)
        if predES['realLabel'] == predES['predLabel']: eshit+=1
        if predEN['realLabel'] == predEN['predLabel']: enhit+=1
        if predFR['realLabel'] == predFR['predLabel']: frhit+=1
        if predRU['realLabel'] == predRU['predLabel']: ruhit+=1
        onuData.append({
            "es": predES,
            "en": predEN,
            "fr": predFR,
            "ru": predRU
        })
        bar.update(i)
    print("Spanish accuracy: {}/80 -> {}".format( eshit, eshit/80 ))
    print("English accuracy: {}/80 -> {}".format( enhit, enhit/80 ))
    print("French accuracy: {}/80 -> {}".format( frhit, frhit/80 ))
    print("Russian accuracy: {}/80 -> {}".format( ruhit, ruhit/80 ))

    with open("./generatedData/onuData.json", "w", encoding="utf8") as f:
        json.dump(onuData, f, indent=4)


    

    ##### HTML GENERATION #####

    # # clustering.getDendogram()
    # for index, doc in enumerate(corpus.docList):
    #     summaryViz = summarizer.viz(doc)
    #     sentimentViz = sentimentAnalyzer.viz(doc)
    #     entityViz = entityRecognizer.viz(doc)
    #     # classes = classifier.getClasses(doc)
    #     clusteringViz = clustering.viz(index, doc)

    #     viz = Viz(document=doc.viz(), summary=summaryViz, entities=entityViz, sentiment=sentimentViz, topicModeling=LDA.viz(), clustering=clusteringViz)
    #     viz.viz(index, len(corpus.docList), doc.title)
    #     bar.update(index)