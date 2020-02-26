import webbrowser
import random
import numpy as np

def getNRandColors(n):
    nums = [ "#" + val.replace('1','F')+"80" for val in  [  "{0:b}".format(i) if len("{0:b}".format(i)) == 6 else ("0"*(6-len("{0:b}".format(i))))+"{0:b}".format(i) for i in range(2**6 -1) ] ]
    for i in range(len(nums) - n): del( nums[random.randint(0,len(nums)-1)] )
    return nums

def getNRandColors_2(n):
    colors = []
    for i in range(n):
        color = list(np.random.choice(range(256), size=3))
        colors.append("rgb({},{},{},0.7)".format(color[0], color[1], color[2]))
    return colors

def getRandomTfIdf(doc, n=10):
    sep_doc = doc.split(' ')
    randNums = []
    for i in range(n): randNums.append(random.randint(0, len(sep_doc) - 1))
    randNums.sort()

    tfidf = []
    for i in randNums: tfidf.append((sep_doc[i], random.uniform(0.65, 0.95)))
    return tfidf

def ClusteringViz(doc, tfIdf, labels):
    colors = getNRandColors_2(len(labels))
    full_labels_html = """
        <div class="_flex _row">
            {}
        </div>
    """
    full_tfidf_html = """
        <p style='line-height: 2.5;'>
            {}
        </p>
    """

    # Labels
    labels_html = ""
    for idx, label in enumerate(labels):
        labels_html += '<h3><span class="badge badge-pill badge-secondary" style="background:{}">{}</span></h3>'.format(colors[idx], label)
    
    full_labels_html = full_labels_html.format(labels_html)

    word_template = """
            <mark style='background:{}'>
                {}
                <span>{}</span>
            </mark>
        """
    # END Labels
        
    # TfIdf
    if not tfIdf: tfIdf = getRandomTfIdf(doc)

    full_tfidf_html = full_tfidf_html.format(doc)
    for w, t in tfIdf:
        processed, toProcess = '</mark>'.join(full_tfidf_html.split('</mark>')[:-1])+'</mark>', full_tfidf_html.split('</mark>')[-1]
        sIdx = toProcess.find(w)
        eIdx = sIdx + len(w)
        before = toProcess[:sIdx]
        after = toProcess[eIdx:]
        
        backgroundColor = 'rgb(0, 128, 0, {})'.format( (0.75 - 0.5) * ((t  - 0.65) / (0.95 - 0.65)) + 0.5 ) 
        full_tfidf_html = processed + before + word_template.format(backgroundColor, w, round(t, 2)) + after
    # END TfIdf
            
    # html = html.format(labels_html, doc)
    html = full_labels_html + full_tfidf_html
    return html

class Viz(object):
    def __init__(self, document=None, summary=None, entities=None, sentiment=None, topicModeling=None, clustering=None):
        self.document = document
        self.summary = summary
        self.entities = entities
        self.sentiment = sentiment
        self.topicModeling = topicModeling
        self.clustering = clustering

    def viz(self, index, totalDocuments, title):
        with open("./Vizs/index.html", "r", encoding="utf8") as template:
            template_str = template.read()
        with open("./Vizs/htmls/render_{}.html".format(index), "w", encoding="utf8") as f:
            template_str = template_str.replace("_original_text_", self.document)
            template_str = template_str.replace("_summary_", self.summary)
            template_str = template_str.replace("_entity_recognition_", self.entities)
            template_str = template_str.replace("_sentiment_analysis_", self.sentiment)
            template_str = template_str.replace("_topic_modelling_", self.topicModeling)
            # template_str = template_str.replace("_clustering_", ClusteringViz(self.document, None, ["Sociedad", "Deportes", "Musica", "Actualidad", "Politica"]))


            template_str = template_str.replace("_document_title_", title)
            template_str = template_str.replace("_document_id_", str(index+1))
            template_str = template_str.replace("_total_documents_", str(totalDocuments))
            template_str = template_str.replace("_clustering_", self.clustering) #ClusteringViz(index))

            f.write(template_str)
