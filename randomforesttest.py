import pandas as pd
import joblib

model = joblib.load("trained.pkl")

result = model.predict_proba([[52,1,6050,805.0,9,200,5,0,0]])

print(result)



