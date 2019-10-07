#NOTE: In CDP find the HMS warehouse directory and external table directory by browsing to:
# Environment -> <env name> ->  Data Lake Cluster -> Cloud Storage
# copy and paste the external location to the config setting below.

#Temporary workaround for MLX-975
#In utils/hive-site.xml edit hive.metastore.warehouse.dir and hive.metastore.warehouse.external.dir based on settings in CDP Data Lake -> Cloud Storage
import os, shutil
if ( not os.path.exists('/etc/hadoop/conf/hive-site.xml')):
  shutil.copyfile("/home/cdsw/utils/hive-site.xml", "/etc/hadoop/conf/hive-site.xml")
  
#Data taken from http://stat-computing.org/dataexpo/2009/the-data.html
#!for i in `seq 1987 2008`; do wget http://stat-computing.org/dataexpo/2009/$i.csv.bz2; bunzip2 $i.csv.bz2; sed -i '1d' $i.csv; aws s3 cp $i.csv s3://ml-field/demo/flight-analysis/data/flights_csv/; rm $i.csv; done

import cdsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import LongTensor, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from pyspark.sql import SparkSession


spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.executor.memory", "4g")\
    .config("spark.executor.instances", 5)\
    .config("spark.yarn.access.hadoopFileSystems","s3a://ml-field/demo/flight-analysis/data/")\
    .config("spark.driver.maxResultSize","4g")\
    .getOrCreate()

spark.sql("SHOW databases").show()
spark.sql("USE default")
spark.sql("SHOW tables").show()

#!pip3 install pyarrow==0.15.0
#spark.conf.set("spark.sql.execution.arrow.enabled", "true")

model_path = '/home/cdsw/models/'
model_name = 'nn_model.pkl'

## Use GPU if available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using CPU" if torch.cuda.is_available() else "Using GPU")

## Load representative subset of rows from dataset
flights = spark.sql("SELECT * FROM `default`.`flights`").limit(500000).toPandas()

# Select useful features and build scale/one-hot-encode
X = flights[['DayOfWeek', 'DayofMonth', 'Month', 'Origin', 'Dest', 'UniqueCarrier', 'CRSDepTime']].copy()
y = flights[['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']].sum(axis=1)
one_hot_matrices = []
for col in filter(lambda col: col != 'CRSDepTime', X.columns):
    one_hot_matrices.append(pd.get_dummies(X[col]))
one_hot_matrix = np.concatenate(one_hot_matrices, axis=1)
X = np.hstack([X['CRSDepTime'].values.reshape(-1, 1), one_hot_matrix])
X = StandardScaler().fit_transform(X)

## Define target variable: binary outcome "was the flight late?"
y = (y > 0).values.astype(int)

## Build pytorch datasets and dataloaders for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y)
dataset_train = TensorDataset(Tensor(X_train), LongTensor(y_train))
dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
dataset_test = TensorDataset(Tensor(X_test), LongTensor(y_test))
dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))


## Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 144)
        self.fc2 = nn.Linear(144, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        return x


## Define a couple of utility functions
def calc_accuracy(output, target):
    pred = output.argmax(dim=1)
    correct = (pred == target).sum().item()
    return correct / len(target)


def losscurve():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot([x[1] for x in train_losses])
    ax[0].set_xlabel("batch")
    ax[0].set_ylabel("accuracy")
    ax[1].plot([x[1] for x in test_losses])
    ax[1].set_xlabel("epoch")


## Train!
train_losses = []
test_losses = []
model = Net()
loss_function = nn.NLLLoss()


#try learning rates 1e-2, 1e-3, 1e-4,1e-5 via Experiments
if len (sys.argv) == 2:
  if sys.argv[1].split(sep='=')[0]=='learning_rate' and isinstance(float(sys.argv[1].split(sep='=')[1]), float):
    learning_rate = float(sys.argv[1].split(sep='=')[1])
  else:
    sys.exit("Invalid Arguments passed to Experiment")
else:
    learning_rate = 1e-3

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

for epoch in range(3):
    for batch, (data, target) in enumerate(dataloader_train):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        accuracy = calc_accuracy(output, target)
        train_losses.append((loss.item(), accuracy))

        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            if epoch == 0 and batch == 0:
                print(f"epoch  batch      loss  accuracy")
            print(f"{epoch:5d}{batch:7d}{loss.item():10.4g}{accuracy:10.4g}")

            
    test_loss = 0
    with torch.no_grad():
        for test_data, test_target in dataloader_test:
            test_data, test_target = test_data.to(device), test_target.to(device)
            test_output = model(test_data)
            test_loss += loss_function(test_output, test_target).item()
    test_loss /= len(dataset_test)
    test_accuracy = calc_accuracy(test_output, test_target)
    test_losses.append((test_loss, test_accuracy))

print(f"\nFinal test set accuracy = {100*test_accuracy:.4g}%")
cdsw.track_metric("Test Accuracy", test_accuracy*100)

losscurve()

#torch.save(model.state_dict(), model_path+model_name)
#cdsw.track_file(model_path+model_name)
