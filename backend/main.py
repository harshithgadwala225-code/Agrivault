import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # It allows your frontend and backend to communicate even when they are running on different ports.

# Pydantic checks whether data has the correct structure and data types.
from pydantic import BaseModel
import joblib # It is used to save the model and load it later models like agrivault_model.pkl and scaler.pkl
import numpy as np

app = FastAPI(title="AgriVault Decision API")

# Enable CORS so the HTML frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware, # Add CORS functionality to my application
    allow_origins=["*"], # Allow requests from any origin.
    allow_credentials=True, # This allows credentials such as cookies or authorization information to be included in cross-origin requests.
    allow_methods=["*"], # Allows HTTP methods like GET, POST, etc.
    allow_headers=["*"], # Allows all HTTP headers
)

# Robust path resolution to locate artifacts inside the ML/ directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# for examples paths not the exact paths

# __file__ means location of the current Python file
    # backend/main.py
# abspath converts the path into absolute path.
    # C:\Users\Harshith\AgriVault\backend\main.py
# first os.path.dirname() gets the folder containing the main.py
    # C:\Users\Harshith\AgriVault\backend
# Second os.path.dirname() gets the folder containing the ML folder goes one folder up
    # C:\Users\Harshith\AgriVault
# BASE_DIR becomes your main project directory.

MODEL_PATH = os.path.join(BASE_DIR, 'ML', 'agrivault_model.pkl')
# os.path.join this join and safely combines the folders names and adds the ML\agrivault_model.pkl in the path of the BASE_DIR

SCALER_PATH = os.path.join(BASE_DIR, 'ML', 'scaler.pkl')
# this also joins the scaler.pkl in the ML folder in the same path.

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class PredictionInput(BaseModel):
    current_price_per_kg: float
    price_30d_lag: float
    quantity_kg: float
    storage_cost_per_kg_month: float = 2.0
    spoilage_rate_pct: float = 8.0

@app.post("/predict") # The backend needs a specific address where the frontend can send that data so that address is /predict
def predict_decision(data: PredictionInput): # data : predictionInput means data coming into this function must follow the structure defined by PredictionInput

    # 1. Feature Engineering
    # this stores the percentage change in prices and stored in the decimal ratio 0.25 rather than 25.
    price_change_30d_pct = (data.current_price_per_kg - data.price_30d_lag) / data.price_30d_lag
    
    features = np.array([[
        data.current_price_per_kg,
        data.price_30d_lag,
        price_change_30d_pct
    ]])




    # 2. Model Prediction
    scaled_features = scaler.transform(features)
    prob_store = float(model.predict_proba(scaled_features)[0][1]) # [0] → probability of class 0 [1] → probability of class 1
    # predict_proba() predicts the probibility and [0][1] → probability of class 1
    # and it becomes as a float decimal value ex 0.75 

    decision_class = int(prob_store >= 0.5) # here if the prob_store is greater than or equal to 0.5 (50%) then it is true then that true is converted into 1 STORE and false is converted into 0 SELL




    # 3. Financial Calculations
    sell_now_revenue = data.quantity_kg * data.current_price_per_kg
    
    # Project stored price using prediction confidence
    est_future_price = data.current_price_per_kg * (1 + (prob_store * 0.35)) # 0.35 means we are allowing a maximum estimated price increase of 35%.
    # Here we are calculating the the estimated future price 
    
    surviving_quantity = data.quantity_kg * (1 - (data.spoilage_rate_pct / 100.0))
    stored_revenue = surviving_quantity * est_future_price
    total_storage_cost = data.quantity_kg * (data.storage_cost_per_kg_month * 2)
    net_stored_profit = stored_revenue - total_storage_cost
    potential_gain = net_stored_profit - sell_now_revenue
    # potiential_gain:
    # A positive number means the storage scenario produces more money under this calculation.
    # A negative number means the storage scenario produces less.


    # This returns a Python dictionary.
    # FastAPI automatically converts it into JSON and give it back to frontend
    return {
        "decision": "STORE FOR 60 DAYS" if decision_class == 1 else "SELL TODAY",
        "recommendation_code": decision_class,
        "confidence_percentage": round(prob_store * 100, 1),
        "financials": {
            "immediate_sale_revenue": round(sell_now_revenue, 2),
            "estimated_stored_revenue": round(stored_revenue, 2),
            "total_storage_cost": round(total_storage_cost, 2),
            "expected_spoilage_kg": round(data.quantity_kg * (data.spoilage_rate_pct / 100.0), 1),
            "net_gain_or_loss": round(potential_gain, 2) # tells the difference between the net money after storage and money selling today 
        }
    }