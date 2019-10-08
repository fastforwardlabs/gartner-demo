
#NOTE: In CDP find the HMS warehouse directory and external table directory by browsing to:
# Environment -> <env name> ->  Data Lake Cluster -> Cloud Storage
# copy and paste the external location to the config setting below.

#Temporary workaround for MLX-975
#In utils/hive-site.xml edit hive.metastore.warehouse.dir and hive.metastore.warehouse.external.dir based on settings in CDP Data Lake -> Cloud Storage
import os, shutil
if ( not os.path.exists('/etc/hadoop/conf/hive-site.xml')):
  shutil.copyfile("/home/cdsw/utils/hive-site.xml", "/etc/hadoop/conf/hive-site.xml")

import cdsw
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

spark = SparkSession\
    .builder\
    .appName("Airline sklearn")\
    .config("spark.executor.memory", "16g")\
    .config("spark.executor.cores","4")\
    .config("spark.driver.memory","6g")\
    .config("spark.executor.instances", 5)\
    .config("spark.driver.maxResultSize","16g")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://ml-field/demo/flight-analysis/data/")\
    .getOrCreate()
spark.sql("SHOW databases").show()
spark.sql("USE default")
spark.sql("SHOW tables").show()
spark.sql("DESCRIBE flights").show()

model_path = '/home/cdsw/models/'
model_name = 'lr_model.pkl'
enc_name = 'lr_enc.pkl'

# Read the data into Spark
#flight_df=spark.read.parquet("s3a://ml-field/demo/flight-analysis/data/airline_parquet_2/",)

#try data set_size via Experiments
if len (sys.argv) == 2:
  if sys.argv[1].split(sep='=')[0]=='set_size' and isinstance(float(sys.argv[1].split(sep='=')[1]), float):
    set_size = float(sys.argv[1].split(sep='=')[1])
  else:
    sys.exit("Invalid Arguments passed to Experiment")
else:
    set_size = 5000

cdsw.track_metric("Set Size", set_size)


# Pull a sample of the dataset into an in-memory
# Pandas dataframe. Use a smaller dataset for a quick demo.
flight_df_local = spark.sql("SELECT * FROM `default`.`flights`").limit(set_size).toPandas()

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Put the data into the array format required by scikit-learn's linreg.
# Use one-hot encoding for the categorical variables
X_onehot = flight_df_local[["UniqueCarrier", "Origin", "Dest"]]
X_num = flight_df_local[["Distance", "CRSDepTime"]]
y = (flight_df_local["DepDelay"] > 0)

enc = OneHotEncoder(sparse=False)
X_transform = enc.fit_transform(X_onehot)

reg = LogisticRegression().fit(np.hstack([X_transform, X_num]), y)

accuracy = reg.score(np.hstack([X_transform, X_num]), y)
cdsw.track_metric("Accuracy", accuracy)


import joblib
with open(model_path+model_name, 'wb') as fid:
    joblib.dump(reg, fid)
cdsw.track_file(model_path+model_name)

with open(model_path+enc_name, 'wb') as fid:
    joblib.dump(enc, fid)
cdsw.track_file(model_path+enc_name)
