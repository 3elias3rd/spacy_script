import random
import spacy
from pathlib import Path
from spacy.training.example import Example
from spacy.util import compounding, minibatch

from training_data import TRAIN_DATA

MODEL_DIR = Path("model")/"intent_bot"
