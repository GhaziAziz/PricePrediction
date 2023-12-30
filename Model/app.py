from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np

# Load the XGBoost model
model = xgb.XGBRegressor()
model.load_model(r'C:\Users\Lenovo\Desktop\Model\Price.rgb')
#C:\Users\Lenovo\AppData\Roaming\Python\Python312\Scripts

app = FastAPI()

class Item(BaseModel):
    Property_Id: int
    Purpose: str
    City: str
    Province: str
    Property_Type: str
    Size_Zameen_com: float
    Bedrooms: int
    Baths: int
    Area_Type: str
    Size_Marla_Kanal_Sq_Yd: float
    Long_Location: float
    Creation_date: str

@app.post("/predict")
def predict(item: Item):
    # Convert input data to a numpy array for prediction
    input_data = np.array([[
        item.Property_Id, item.Size_Zameen_com,
        item.Bedrooms, item.Baths, item.Size_Marla_Kanal_Sq_Yd, item.Long_Location
    ]])

    # Make predictions using the loaded model
    prediction = model.predict(input_data)

    # Return the prediction
    return {"prediction": prediction[0]}
