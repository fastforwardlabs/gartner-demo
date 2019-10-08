  
import joblib
import pandas as pd
import numpy as np

model_path = '/home/cdsw/models/'
model_name = 'lr_model.pkl'
enc_name = 'lr_enc.pkl'

import os, shutil
if ( os.path.exists('/home/cdsw/'+enc_name)):
  shutil.move('/home/cdsw/'+enc_name, model_path+enc_name)
enc = joblib.load(model_path+enc_name)

if ( os.path.exists('/home/cdsw/'+model_name)):
  shutil.move('/home/cdsw/'+model_name, model_path+model_name)
model = joblib.load(model_path+model_name)

def predict(ipt):
  ipt_df = pd.DataFrame([ipt])
  
  ipt_num = ipt_df[["Distance", "CRSDepTime"]]
  ipt_onehot = ipt_df[["UniqueCarrier", "Origin", "Dest"]]
  ipt_transform = enc.transform(ipt_onehot)
  
  ipt_final = np.hstack([ipt_transform, ipt_num])
  return {"delay_probability": model.predict_proba(ipt_final)[0,1]}
    
#test
#ipt = {"CRSDepTime": 630, "Dest": "SJC", "Distance": 308, "Origin": "LAX", "UniqueCarrier": "PS"}
#predict(ipt)
