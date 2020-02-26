import json
import string
import unidecode
import numpy as np 

class SentimentAnalyzer(object):
    def __init__(self, pathToDict="./modelsData/SA/customDictionary.json"):
        print("Instantiating Sentiment Analyzer")
        with open(pathToDict, "r", encoding="utf8") as f: self.dictionary = json.load(f)

    def getDictionary(self, pathToDict):
        word2sentiment = {}
        with open(pathToDict, "r", encoding="utf8") as f: dictionary = json.load(f)["senticon"]["layer"]
        for level in dictionary:
            for word in level["positive"]["lemma"]:
                text = self.processSentence(word["#text"][1:-1])
                pol = float(word["-pol"])*0.5 + 0.5
                if not text in word2sentiment.keys(): word2sentiment[text] = (float(pol), word["#text"][1:-1].replace('_',' '))
                
            for word in level["negative"]["lemma"]:
                text = self.processSentence(word["#text"][1:-1])
                pol = (float(word["-pol"])+1)*0.5
                if not text in word2sentiment.keys(): word2sentiment[text] = (float(pol), word["#text"][1:-1].replace('_',' '))
        with open("./modelsData/dictionary.json", "w", encoding="utf8") as f: json.dump(word2sentiment, f, indent=4)
        return word2sentiment

    def processSentence(self, sentence):
        sentence = sentence.replace('<br/>',' ')
        sentence = unidecode.unidecode(sentence.replace('_',' ')).lower()
        for char in string.punctuation: sentence = sentence.replace(char, ' ')
        sentence = ' '.join( [part for part in sentence.split() if part] )
        return sentence

    def applyCoef(self, x, slope=2):
        if x >= 0.5: coef =  1 - 10* (x-0.5)**3
        else: coef = slope - slope*x
        return x**coef

    def bsa(self, text):
        try:
            words = []
            values = []
            for w, (sw, ow) in self.dictionary.items():
                if len(w) < 4: continue
                pw = self.processSentence(w)
                tw = text.count(pw)
                if tw > 0: #If a word match is found
                    iw = text.index(pw)
                    if text[iw-1] != " ": continue #It has to start a word
                    if iw in [i for l,o,t,i,s in values]: continue
                        
                    otw = text[iw:iw+len(pw)] + text[iw+len(pw)-1:].split()[0][1:]
                    if otw[-1] in string.punctuation: otw = otw[:-1]
                    lw = len(otw)

                    if lw - len(pw) <= 2: 
                        values.append( (lw, otw, tw, iw, sw) )
                        words.append(w)

            sum =  np.sum( [s*t for l,w,t,i,s in values] )
            times = np.sum( [t for l,w,t,i,s in values] )
            return sum/times, [(w,t,s) for l,w,t,i,s in values], words
        except Exception as e: return 0.5, [], []

    def pondPunct(self, text):
        punctScore = 0
        if ":)" in text: punctScore += 0.5
        if ":P" in text: punctScore += 0.5
        if ":D" in text: punctScore += 0.7
        if ":(" in text: punctScore -= 0.5
        if "..." in text: punctScore -= 0.25
        return punctScore        

    def getSentiment(self, doc, punctFact=0.15):
        text = doc.text
        bsaScore, words, dictWords = self.bsa(text)
        punctScore = self.pondPunct(text)
        if '!' in text: bsaScore = self.applyCoef(bsaScore, slope=2)
        score = max( min( self.applyCoef(bsaScore) + (punctScore*punctFact) , 1 ) , 0 )   
        return score, words, dictWords

    def getJson(self, doc):
        sentValue, wordValues, _ = self.getSentiment(doc)
        return {"value": sentValue, "wordScores": wordValues}

    def viz_word(self, sentiment):
        if sentiment <= 0.25:
            return "sentiment_very_dissatisfied"
        elif sentiment <= 0.5:
            return "sentiment_dissatisfied"
        elif sentiment <= 0.75:
            return "sentiment_satisfied"
        elif sentiment <= 1.0:
            return "sentiment_very_satisfied"

    def viz(self, doc):
        sentValue, wordValues, _ = self.getSentiment(doc)
        
        html = """
        <p style='margin-bottom: 50px;'>El sentimiento general de la notícia es <b>{}</b></p>
        <p style='line-height: 2.5;'>{}</p>
        """
        html = html.format(round(sentValue,3), doc.viz())

        word_template = """
            <mark class='{}'>
                {}
                <span>{}</span>
            </mark>
        """

        for w,_,s in wordValues:
            sentiment_class = self.viz_word(s)
            html = html.replace(w, word_template.format(sentiment_class, w, round(s, 3)))

        return html