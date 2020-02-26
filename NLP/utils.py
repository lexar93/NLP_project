
import json
import string
import unidecode
import numpy as np 
import nltk.tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
from nltk.stem.snowball import SnowballStemmer


class SentenceProcessor(object):
    """
        Helper class for processing text
        Attributes:
            stemmer (class nltk.stem.snowball): stemmer to convert words to its root form
            punctuation (obj list[str]): list of all characters representing punctuation
            doc2vec (class gensim.models.doc2vec): custom doc-based doc2vec vectorspace model
            stopwords (obj list[str]): list of several spanish stopwords
    """
    def __init__(self, d2vModelPath='./modelsData/Embeddings/model_docsimilarity.doc2vec', language='es'):
        if language == 'es': self.stemmer = SnowballStemmer("spanish")
        else: self.stemmer = SnowballStemmer("english")
        self.punctuation = string.punctuation+'¿'
        self.doc2vec = Doc2Vec.load(d2vModelPath)
        # self.stopWords = ["‘", "’", "“", "”", "«", "»", "más", "o", "a", "e", "le", "al", "el", "la", "los", "las", "y", "este", "ante", "cabo", "con", "contra", "de" ,"desde", "hacia", "hasta", "para", "por", "según", "sin", "sobre", "tras", "se", "un", "una", "unas", "unos", "uno", "sobre", "todo", "también", "tras", "de", "del", "a", "otro", "algún", "alguno", "alguna", "algunos", "algunas", "ser", "es", "soy", "eres", "somos", "sois", "estoy", "esta", "estamos", "estais", "estan", "en", "para", "atras", "estado", "estaba", "ante", "antes", "siendo", "ambos", "pero", "por", "poder", "puede", "puedo", "podemos", "podeis", "pueden", "fui", "fue", "fuimos", "fueron", "hacer", "hago", "hace", "hacemos", "haceis", "hacen", "cada", "fin", "incluso", "primero", "desde", "conseguir", "consigo", "consigue", "consigues", "conseguimos", "consiguen", "ir", "voy", "va", "vamos", "vais", "van", "vaya", "gueno", "ha", "tener", "tengo", "tiene", "tenemos", "teneis", "tienen", "el", "la", "lo", "las", "los", "su", "aqui", "mio", "tuyo", "ellos", "ellas", "nos", "nosotros", "vosotros", "vosotras", "si", "dentro", "solo", "solamente", "saber", "sabes", "sabe", "sabemos", "sabeis", "saben", "ultimo", "largo", "bastante", "haces", "muchos", "aquellos", "aquellas", "sus", "entonces", "tiempo", "verdad", "verdadero", "verdadera", "cierto", "ciertos", "cierta", "ciertas", "intentar", "intento", "intenta", "intentas", "intentamos", "intentais", "intentan", "dos", "bajo", "arriba", "encima", "usar", "uso", "usas", "usa", "usamos", "usais", "usan", "emplear", "empleo", "empleas", "emplean", "ampleamos", "empleais", "valor", "muy", "era", "eras", "eramos", "eran", "modo", "bien", "mientras", "quien", "con", "entre", "sin", "trabajo", "trabajar", "trabajas", "trabaja", "trabajamos", "trabajais", "trabajan", "podria", "podrias", "podriamos", "podrian", "podriais", "yo", "aquel", "porque", "porqué", "por qué", "cual", "cuál", "cuando", "cuándo", "donde", "dónde", "como", "cómo", "que", "qué", "quien", "quién", "cuanto", "cúanto", "cuantos", "cuántos", "cuales", "cuáles"]
        if language == 'es': self.stopWords = list(stopwords.words('spanish'))
        else: self.stopWords = list(stopwords.words('english'))
        
    # Normalizes str in sentence to take out accents
    def stripAccents(self, sentence):
        return unidecode.unidecode(sentence)

    # Returns a list of every token in the sentence
    # If stemming is set to True, every word is reduced to its root form
    def getTokenizedSentence(self, sentence, stemming=True):
        words = sentence.lower().split()
        tSentence = []
        for token in [ t for t in words if not (t in self.stopWords) and (not (t in self.punctuation)) ]:
            pToken = token
            for punctuation in self.punctuation: pToken = pToken.replace(punctuation, '')
            if pToken.isdigit() or len(pToken) <= 3: continue
            if stemming: tSentence.append( self.stripAccents(self.stemmer.stem(pToken)) )
            else: tSentence.append( self.stripAccents(pToken) )
        return tSentence

    # Returns a list of every sentence in the document
    def getTokenizedDocument(self, document):
        return nltk.tokenize.sent_tokenize(document)

    # Returns normalized sentence as str
    def getProcessedSentence(self, sentence):
        return ' '.join(self.getTokenizedSentence(sentence, stemming=False))

    # Returns np.array shaped (1,300) as a result of doc2vec inferring
    def getInferedVector(self, sentence):
        pSentence = self.getProcessedSentence(sentence)
        return self.doc2vec.infer_vector(pSentence.split(), epochs=50)

    # Returns whether the parameter sentence has to be filtered or not
    # minTokens, maxTokens are the boundaries for number of tokens
    # maxLenToken is the boundary for the length of a single token
    # maxPropUppercase is the maximum proportion of uppercase words in the sentence
    def sentenceIsFiltrable(self, tokens, minTokens=3, maxTokens=55, maxLenToken=20, maxPropUppercase=0.2):
        if not ( minTokens < len(tokens) < maxTokens ): return True 
        else:
            num_punct, num_uppercase = (0, 0)
            for token in tokens:
                if len(token) > maxLenToken: return True
                if '@' in token: return True
                if token.isupper(): num_uppercase += 1
            if num_uppercase/len(tokens) >= maxPropUppercase: return True
        return False

    # Computes the pairs original sentence-processed sentence for all not filtered sentences in document
    def getPreprocessedDocument(self, document):
        sents = self.getTokenizedDocument(document)
        for originalSent in sents:
            tokens = self.getTokenizedSentence(originalSent)
            if not self.sentenceIsFiltrable(tokens):
                processedSent= ' '.join(tokens)
                yield originalSent, processedSent

    def NERPreprocess(self, sentence):
        sentence = nltk.word_tokenize(sentence)
        sentence = nltk.pos_tag(sentence)
        return sentence
    

class SimilarityCalculator(object):
    """
        Helper class calculating similarities over text or vectors
        Attributes:
            SP (class doc.SentenceProcessor): Helper class for processing text
    """
    def __init__(self, SP):
        self.SP = SP

    # Computes the cosine similarity between two str sentences
    def getCosineSimilarity(self, sent1, sent2):    
        wv1 = self.SP.getInferedVector(sent1)
        wv2 = self.SP.getInferedVector(sent2)
        return self.getVecCosineSimilarity(wv1, wv2)

    # Computes Jaccard similarity between two str sentences
    def getJaccardSimilarity(self, sent1, sent2):
        ws1 = self.SP.getTokenizedSentence(sent1)
        ws2 = self.SP.getTokenizedSentence(sent2)
        intersection = len(list(set(ws1) & set(ws2)))
        union = len(list(set(ws1) | set(ws2)))
        weight = float(intersection / union)
        return weight

    # Computes the cosine similarity among two numpy arrays
    def getVecCosineSimilarity(self, vec1, vec2): 
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


class PageRank(object):
    def __init__(self, transitionWeights):
        self.matrix = transitionWeights
        self.nodes = range(transitionWeights.shape[0])
        self.ensureRowsPositive()
        
    def ensureRowsPositive(self):
        for i, row in enumerate(self.matrix):
            if sum(row) == 0.0: self.matrix[i,:] = np.ones(len(row))

    def normalizeRows(self):
        rowSum = self.matrix.sum(axis=1)
        return self.matrix/rowSum[:, np.newaxis]

    def integrateRandomSurfer(self, transitionProbs, dampingFactor):
        alpha = 1.0 / float(len(self.nodes)) * (1-dampingFactor)
        return np.multiply(transitionProbs, dampingFactor) + alpha

    def euclideanNorm(self, delta):
        return np.sqrt(delta.dot(delta))

    def computeStartState(self):
        startProb = 1.0 / float(len(self.nodes))
        return np.full ( (len(self.nodes)), startProb )  

    def powerIteration(self, dampingFactor=0.85, epsilon=1e-7, maxIterations=1000):
        transitionProbs = self.normalizeRows()
        transitionProbs = self.integrateRandomSurfer(transitionProbs, dampingFactor)
        state = self.computeStartState()
        for _ in range(maxIterations):
            oldState = state.copy()
            state = state.dot(transitionProbs)
            delta = state-oldState
            if self.euclideanNorm(delta) < epsilon: break
        return state
