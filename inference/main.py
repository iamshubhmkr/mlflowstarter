from fastapi import FastAPI
from logger import get_logger
from inference.predict import predict_survival
from inference.schema import PassengerInput, PredictionResponse

logger = get_logger("inference_main")
app = FastAPI(title="Titanic Inference API")

@app.get("/")
def read_root():
    logger.info("✅ Health check called — API is up.")
    return {"msg": "Titanic ML Inference API is up!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PassengerInput):
    logger.info(f"📥 Received input: {input_data}")
    prediction = predict_survival(input_data)
    logger.info(f"📤 Prediction response: {prediction}")
    return prediction