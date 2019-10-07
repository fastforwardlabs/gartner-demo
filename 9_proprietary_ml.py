#NOTE: In CDP find the HMS warehouse directory and external table directory by browsing to:
# Environment -> <env name> ->  Data Lake Cluster -> Cloud Storage
# copy and paste the external location to the config setting below.

#Temporary workaround for MLX-975
#In utils/hive-site.xml edit hive.metastore.warehouse.dir and hive.metastore.warehouse.external.dir based on settings in CDP Data Lake -> Cloud Storage
if ( not os.path.exists('/etc/hadoop/conf/hive-site.xml')):
  !cp /home/cdsw/utils/hive-site.xml /etc/hadoop/conf/

  
#from __future__ import print_function

import sys
import os
import time
import copy

sys.path.append("/home/cdsw/") 
from utils.auger_api import AugerAPI

from pyspark.sql import SparkSession
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession\
    .builder\
    .appName("MLAugerSample")\
    .config("spark.executor.memory", "8g")\
    .config("spark.executor.instances", 5)\
    .config("spark.yarn.access.hadoopFileSystems","s3a://ml-field/demo/flight-analysis/data/")\
    .config("spark.driver.maxResultSize","8g")\
    .getOrCreate()

augerAPI = AugerAPI("https://app.auger.ai")
augerAPI.login(email=os.environ.get('AUGER_EMAIL'), password=os.environ.get('AUGER_PW'))

search_space = {
    "pyspark.ml.classification.RandomForestClassifier": {
        "maxDepth": {"bounds": [5, 20], "type": "int"},
        "maxBins": {"bounds": [4, 16], "type": "int"},
        "impurity": {"values": ["gini", "entropy"], "type": "categorical"},
        "numTrees": {"bounds": [4, 16], "type": "int"}
    },
    "pyspark.ml.classification.GBTClassifier": {
        "maxIter": {"bounds": [4, 16], "type": "int"},
        "maxBins": {"bounds": [4, 16], "type": "int"},        
        "stepSize": {"bounds": [0.1, 1.0], "type": "float"},
        "featureSubsetStrategy": {"values": ['auto', 'all', 'sqrt', 'log2'], "type": "categorical"}        
    }
}

trials_total_count = 20
augerAPI.create_trial_search(trials_total_count=trials_total_count, search_space=search_space)
next_trials = augerAPI.continue_trial_search(trials_limit=4)


data = spark.read.format("libsvm").load("/home/cdsw/optimizer_search/sample_libsvm_data.txt")
#sample_size = 50000
#num_flights = spark.sql("SELECT COUNT(*) FROM `default`.`flights`").count()
#sample_ratio = sample_size / num_flights

flights_DF = spark.sql("SELECT * FROM `default`.`flights`").sample(.004).toPandas()

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

(training_data, test_data) = data.randomSplit([0.8, 0.2])

print("Start execute trials: %s"%next_trials)
trials_history = []
while len(trials_history) < trials_total_count:
	#Execute trials to get score, bigger is better (0.0..1.0)
	#It may be run in parallel
	for trial in next_trials:
		algo_params = copy.deepcopy(trial.get('algorithm_params'))
		algo_params['labelCol'] = "indexedLabel"
		algo_params['featuresCol'] = "indexedFeatures"

		ml_algo = AugerAPI.create_object_by_class(trial.get('algorithm_name'), algo_params)
		pipeline = Pipeline(stages=[labelIndexer, featureIndexer, ml_algo])

		start_fit_time = time.time()
		ml_model = pipeline.fit(training_data)

		history_item = copy.deepcopy(trial)
		history_item['evaluation_time'] = time.time() - start_fit_time

		predictions = ml_model.transform(test_data)
		evaluator = MulticlassClassificationEvaluator(
		    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

		history_item['score'] = evaluator.evaluate(predictions)

		print("Executed trial: %s"%history_item)
		trials_history.append(history_item)

	next_trials = augerAPI.continue_trial_search(trials_limit=4, trials_history=trials_history)
	
trials_history.sort(key=lambda t: t['score'], reverse=True)

print("Best trial: %s"%trials_history[0])
#spark.stop()

