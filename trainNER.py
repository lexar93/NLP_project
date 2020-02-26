from __future__ import unicode_literals, print_function

import plac
import json
import spacy
import random
from pathlib import Path
from spacy.util import minibatch, compounding


with open("./nerAnnotator/AnnotatedData/dataset.json", encoding="utf8") as f: TRAIN_DATA = json.load(f)

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model="es_core_news_md", output_dir="./modelsData/NER/es_core_news_md_custom1", n_iter=100):
    nlp = spacy.load(model) 
    print("Loaded model '%s'" % model)

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else: ner = nlp.get_pipe("ner")

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"): ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        for itn in range(n_iter):
            print("\n\nIteration {}\n--------------------------------".format(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    drop=0.3,
                    losses=losses,
                )
            print("Losses", losses)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


if __name__ == "__main__":
    plac.call(main)
