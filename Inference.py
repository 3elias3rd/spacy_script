from pathlib import Path
from typing import Dict, Any
import spacy

MODEL_DIR = Path("model")/"nlp_intent_bot"
FALLBACK_THRESHOLD = 0.6

class IntentClassifier:
    def __init__(self, model_dir: Path=MODEL_DIR):
        if not model_dir.exists():
            raise FileExistsError(f"Model {model_dir} has not been trained and uploaded to the directory.")
        
        self.nlp =spacy.load(model_dir)

        if "textcat" not in self.nlp.pipe_names:
            raise ValueError(f"{model_dir} has no textcat component.")
        
    def predict(self, text: str) -> Dict[str,Any]:
        if not isinstance(text, str) or not text.strip():
            return{"intent": None, "confidence": 0.0, "fallback": True}
        
        doc = self.nlp(text)
        cats = doc.cats
        top_label = max(cats, key=cats.get)

        confidence = float(cats[top_label])
        fallback = confidence < FALLBACK_THRESHOLD

        return {"intent": top_label, "confidence": confidence, "fallback": fallback}
    