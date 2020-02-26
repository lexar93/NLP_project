import json
from NLP.Corpus import Corpus
from NLP.Modules.SentimentAnalizer import SentimentAnalyzer

with open("./modelsData/customDictionary.json", "r", encoding="utf8") as f: data = json.load(f)

sentimentAnalyzer = SentimentAnalyzer()
with open("./Data/largeCorpus.json", "r", encoding="utf8") as f: corpus = Corpus(json.load(f)[:])


words = []
for doc in corpus.docList:
    print("Computing SA id: {}".format(doc.id))
    sent, wordsS, dictWords = sentimentAnalyzer.getSentiment(doc)
    for w in dictWords: 
        if not w in words: words.append(w)


words.sort()

try: 
    with open("./modelsData/customDictionaryP1.json", "r", encoding="utf8") as f: dataSaving = json.load(f)
except: dataSaving = {}

try:
    dsk = list(dataSaving.keys())
    dsk.sort()
    lastIdx = words.index( dsk[-1] )
    currentIdx = lastIdx+1
except:
    currentIdx = 0

print("The current index is {}".format(currentIdx))
print("There are a total of {} words".format(len(words)))
while currentIdx < len(words):
    w = words[currentIdx]

    if w in dataSaving.keys(): 
        currentIdx += 1
        continue

    (s, rw) = data[w]
    try:
        option = int(input("Palabra: {} -> {} - {}\n\t1. OK\n\t2. Borrar\n\t3. Cambiar score\n\t4. Anterior\n> ".format(currentIdx, w, s)))
        while not option in [1,2,3,4]: option = int(input("Palabra: {} - {}\n\t1. OK\n\t2. Borrar\n\t3. Cambiar score\n\t4. Anterior\n".format(currentIdx, w, s)))
    except Exception as e: 
        print(e)
        exit()
    
    if option == 1: dataSaving[w] = (s,rw)
    elif option == 3:
        try: s = float(input("Cuál es la nueva puntuación?\n> "))
        except: s = float(input("Cuál es la nueva puntuación?\n> "))
        dataSaving[w] = (s,rw)
    elif option == 4:
        currentIdx -= 2
    currentIdx+=1

    with open("./modelsData/customDictionaryP1.json","w", encoding="utf8") as f: json.dump(dataSaving, f, indent=4)   





###########################################################################################################################


# dataItems = list(data.items())
# dataItems.sort(key= lambda x: x[0])
# dataItemsKeys = [k for k,(s,w) in dataItems]

# if not dataSaving: currentIdx = 0
# else: 
#     dataSKeys = list(dataSaving.keys())
#     dataSKeys.sort()
#     last = dataSKeys[-1]
#     currentIdx = dataItemsKeys.index(last)+1

# while currentIdx < len(dataItemsKeys):
#     w, (s, rw) = dataItems[currentIdx]
#     try:
#         option = int(input("Palabra: {} -> {} - {}\n\t1. OK\n\t2. Borrar\n\t3. Cambiar score\n\t4. Anterior\n> ".format(currentIdx, w, s)))
#         while not option in [1,2,3,4]: option = int(input("Palabra: {} - {}\n\t1. OK\n\t2. Borrar\n\t3. Cambiar score\n\t4. Anterior\n".format(currentIdx, w, s)))
#     except: exit()
    
#     if option == 1: dataSaving[w] = (s,rw)
#     elif option == 3:
#         try: s = float(input("Cuál es la nueva puntuación?\n> "))
#         except: s = float(input("Cuál es la nueva puntuación?\n> "))
#         dataSaving[w] = (s,rw)
#     elif option == 4:
#         currentIdx -= 2
#     currentIdx += 1

#     with open("./modelsData/customDictionaryP.json","w", encoding="utf8") as f: data = json.dump(dataSaving, f, indent=4)