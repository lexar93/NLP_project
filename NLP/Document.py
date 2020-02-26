class Document(object):
    """
        News representation
        Computed Attributes:
            processedBody (obj str): processed version of body (lowercase, no accents, words in root form)
            processedTitle (obj str): processed version of title (lowercase, no accents, words in root form)
            titleEmbedding (class numpy.array): (1, 300) shaped np.array containing the doc embeddings for the title
    """    
    def __init__(self, docInfo, SP, SC):
        self.SP = SP
        self.SC = SC
        self.id = docInfo['id']
        self.text = docInfo['text']
        self.title = docInfo['title']
        self.embeddedText = self.embedText()
    
    # Takes the body attribute and returns it processed
    def processText(self):
        return self.SP.getProcessedSentence(self.text)

    # Takes the title and infers its doc vector
    def embedText(self):
        return self.SP.getInferedVector(self.text)

    # Computes the similarity between itself and the news in the parameter
    def getSimilarity(self, otherDoc):
        return self.SC.getVecCosineSimilarity(self.embeddedText, otherDoc.embeddedText)

    def viz(self):
        return self.text.replace('\n', '<br/>')

