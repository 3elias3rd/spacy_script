from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from Inference import IntentClassifier, MODEL_DIR

app = FastAPI(title="NLP Intent classifier")

class PredictRequest(BaseModel):
    message: str

class PredictResponse(BaseModel):
    intent: Optional[str]
    confidence: float
    fallback: bool

try:
    classifier = IntentClassifier()
except Exception as e:
    classifier = None
    load_error = str(e)

else:
    load_error = None

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    if classifier is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")
    result = classifier.predict(req.message)
    return result

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
    
