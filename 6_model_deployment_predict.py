  
import joblib
import pandas as pd
import numpy as np

enc = joblib.load("xx_model_deployment_encoding.pkl")
model = joblib.load("xx_model_deployment_model.pkl")

def predict(ipt):
  ipt_df = pd.DataFrame([ipt])
  
  ipt_num = ipt_df[["Distance", "CRSDepTime"]]
  ipt_onehot = ipt_df[["UniqueCarrier", "Origin", "Dest"]]
  ipt_transform = enc.transform(ipt_onehot)
  
  ipt_final = np.hstack([ipt_transform, ipt_num])
  return {"delay_probability": model.predict_proba(ipt_final)[0,1]}
    
#test
#ipt = {
#  "CRSDepTime": 630,
#  "Dest": "SJC",
#  "Distance": 308,
#  "Origin": "LAX",
#  "UniqueCarrier": "PS"
#}
#
#predict(ipt)
