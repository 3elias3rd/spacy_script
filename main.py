import random
import spacy
from pathlib import Path
from spacy.training.example import Example
from spacy.util import compounding, minibatch

from training_data import TRAIN_DATA

MODEL_DIR = Path("model")/"intent_bot"

def create_nlp_pipeline():
    nlp = spacy.load("en")

    textcat = nlp.add_pipe("textcat", last=True)

    labels = ["farewell", "greeting", "location", "pricing"]
    for label in labels:
        textcat.add_label(label)
    return nlp

def train(nlp, training_data, n_iter=20):
    examples = []
    for text, ann in training_data:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, {"cats": ann["cats"]}))
    
    print("Training is now begining")

    optimizer = nlp.initialize(lambda:examples)

    for epoch in range(n_iter):
        random.shuffle(examples)
        losses = {}
        batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses)
    print(f"Iteration: {epoch+1} | Loss: {losses.get('textcat', 0.0):.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_DIR)
    print(f"Model trained and uploaded to {MODEL_DIR.resolve()}")

if __name__ == "__main__":
    nlp = create_nlp_pipeline()
    train(nlp, TRAIN_DATA, n_iter=20)
    
        
    