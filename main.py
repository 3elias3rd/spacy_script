import random
import spacy
from pathlib import Path
from spacy.training.example import Example
from spacy.util import compounding, minibatch

from training_data import TRAIN_DATA

# Create path in the directory to model 
MODEL_DIR = Path("model")/"nlp_intent_bot"

# Funtion to create the pipeline with textcat componet and all 4 labels.
def create_nlp_pipeline():
    nlp = spacy.load("en")

    textcat = nlp.add_pipe("textcat", last=True)

    labels = ["farewell", "greeting", "location", "pricing"]

    # Add each label to the textcat component
    for label in labels:
        textcat.add_label(label)
    return nlp

# Function takes the pipeline and runs text that has been converted into a doc object and makes predictions based on training.
def train(nlp, trained_data, n_iter=30):
    examples = []
    for text, ann in trained_data:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, {"cats": ann["cats"]}))
    
    print("Training is now begining")

    optimizer = nlp.initialize(lambda:examples)

    # For each run through the training data
    for epoch in range(n_iter):
        random.shuffle(examples)
        losses = {}

        # Create smaller beatches from the list of example that increase in size slowly with each batch for 4 Example objects to 32 Example objects.
        batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses)

        # Print Losses after each run through the data. (losses should go down over time)
        print(f"Iteration: {epoch+1} | Loss: {losses.get('textcat', 0.0):.2f}")

    # Upload the model to the directory. Make sure MODEL_DIR exists and create if necessary
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_DIR)
    print(f"Model trained and uploaded to {MODEL_DIR.resolve()}")

if __name__ == "__main__":
    nlp = create_nlp_pipeline()
    train(nlp, TRAIN_DATA, n_iter=30)