import spacy
from spacy import displacy

class SpacyNER(object):
    def __init__(self):
        print("Instantiating Entity Recognizer")
        self.nlp = spacy.load('./modelsData/NER/es_core_news_md_custom1')

    def getEntities(self, doc):
        docText = self.nlp(doc.text.replace('\n',' '))
        ents = [(e.text, e.label_) for e in docText.ents]
        return ents

    def getJson(self, doc):
        return self.getEntities(doc)
        
    def viz_word(self, type):
        if type == "ORG":
            return "sentiment_very_dissatisfied"
        elif type == "PER":
            return "sentiment_dissatisfied"
        elif type == "LOC":
            return "sentiment_satisfied"
        elif type == "MISC":
            return "sentiment_very_satisfied"

    def viz(self, document):
        text = document.viz()
        entities = self.getEntities(document)
        html = """
        <p style='line-height: 2.5;'>{}</p>
        """
        html = html.format(text)

        word_template = """
            <mark class='{}'>
                {}
                <span>{}</span>
            </mark>
        """

        for i, (w,t) in enumerate(entities):
            processed, toProcess = '</mark>'.join(html.split('</mark>')[:-1])+'</mark>', html.split('</mark>')[-1]
            sIdx = toProcess.find(w)
            eIdx = sIdx + len(w)
            before = toProcess[:sIdx]
            after = toProcess[eIdx:]
            
            sentiment_class = self.viz_word(t)
            html = processed + before + word_template.format(sentiment_class, w, t) + after
        return html
