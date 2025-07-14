from pydantic import BaseModel
from typing import Optional

class PassengerInput(BaseModel):
    Pclass: Optional[int] = None
    Sex: Optional[int] = None
    Age: Optional[float] = None
    SibSp: Optional[int] = None
    Parch: Optional[int] = None
    Fare: Optional[float] = None
    Embarked: Optional[int] = None
    Title: Optional[int] = None
    FamilySize: Optional[int] = None

class PredictionResponse(BaseModel):
    prediction: int