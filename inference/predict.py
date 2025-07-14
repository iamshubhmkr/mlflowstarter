import pandas as pd
from inference.model_loader import load_best_model
from inference.schema import PassengerInput, PredictionResponse
from logger import get_logger

logger = get_logger("predict")

# Load model and expected columns once
model = load_best_model()
try:
    expected_columns = model.metadata.get_input_schema().input_names()
    logger.info(f"Expected input columns from model signature: {expected_columns}")
except Exception as e:
    logger.warning(f"❌ Failed to extract input schema from model metadata: {e}")
    # Fallback to default columns based on data version 1.2 (all features minus PassengerId)
    expected_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]
    logger.info(f"Using fallback columns: {expected_columns}")

def predict_survival(input_data: PassengerInput) -> PredictionResponse:
    # Convert input to DataFrame
    data = pd.DataFrame([input_data.dict()])
    logger.debug(f"🧾 Raw input DataFrame: \n{data}")

    try:
        if not expected_columns:
            logger.error("❌ No expected columns defined in model signature or fallback.")
            raise ValueError("No expected columns available for prediction.")

        # Identify available and missing columns
        input_columns = [col for col in data.columns if not data[col].isna().all()]
        available_columns = [col for col in expected_columns if col in input_columns]
        missing_columns = [col for col in expected_columns if col not in input_columns]

        if not available_columns:
            logger.error(f"❌ No matching columns between input {input_columns} and expected {expected_columns}.")
            raise ValueError(f"No input columns match expected columns: {expected_columns}")

        # Log column information
        logger.info(f"✅ Available columns: {available_columns}")
        if missing_columns:
            logger.warning(f"⚠️ Missing columns: {missing_columns}. Filling with default value (0.0).")

        # Create a DataFrame with expected columns, filling missing ones with 0.0
        filtered_data = pd.DataFrame(0.0, index=[0], columns=expected_columns)
        for col in available_columns:
            filtered_data[col] = data[col].astype(float)  # Ensure float type for consistency

        logger.debug(f"✅ Filtered input (columns ordered): \n{filtered_data}")

        # Make prediction
        prediction = model.predict(filtered_data)[0]
        logger.info(f"🎯 Model prediction: {prediction}")
        return PredictionResponse(prediction=int(prediction))

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise