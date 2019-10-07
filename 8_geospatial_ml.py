# !pip3 install scikit-gstat

#Temporary workaround for MLX-975
#In utils/hive-site.xml edit hive.metastore.warehouse.dir and hive.metastore.warehouse.external.dir based on settings in CDP Data Lake -> Cloud Storage
import os, shutil
if ( not os.path.exists('/etc/hadoop/conf/hive-site.xml')):
  shutil.copyfile("/home/cdsw/utils/hive-site.xml", "/etc/hadoop/conf/hive-site.xml")

#Data taken from http://stat-computing.org/dataexpo/2009/the-html
#!for i in `seq 1987 2008`; do wget http://stat-computing.org/dataexpo/2009/$i.csv.bz2; bunzip2 $i.csv.bz2; sed -i '1d' $i.csv; aws s3 cp $i.csv s3://ml-field/demo/flight-analysis/data/flights_csv/; rm $i.csv; done

from __future__ import print_function
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType

spark = SparkSession\
    .builder\
    .appName("GeoAnalysis")\
    .config("spark.executor.memory", "4g")\
    .config("spark.executor.instances", 5)\
    .config("spark.yarn.access.hadoopFileSystems","s3a://ml-field/demo/flight-analysis/data/")\
    .getOrCreate()

# Find the frequency of all trips (preserving direction)
flights_airport = spark.sql("select Origin as origin, Dest as destination, count(1) from flights group by Origin, Dest").toPandas()

# Get the airports as a local dataframe
airports = spark.sql("SELECT * FROM airports").toPandas()

# Display an interactive map of airports with connections using
# d3.js, vega-lite and altair
import altair as alt
from vega_datasets import data

states = alt.topo_feature(data.us_10m.url, feature="states")

# Create mouseover selection
select_city = alt.selection_single(
    on="mouseover", nearest=True, fields=["origin"], empty="none"
)

# Define which attributes to lookup from airports.csv
lookup_data = alt.LookupData(
    airports, key="iata", fields=["state", "lat", "long"]
)

background = alt.Chart(states).mark_geoshape(
    fill="lightgray",
    stroke="white"
).properties(
    width=750,
    height=500
).project("albersUsa")

connections = alt.Chart(flights_airport).mark_rule(opacity=0.35).encode(
    latitude="lat:Q",
    longitude="long:Q",
    latitude2="lat2:Q",
    longitude2="lon2:Q"
).transform_lookup(
    lookup="origin",
    from_=lookup_data
).transform_lookup(
    lookup="destination",
    from_=lookup_data,
    as_=["state", "lat2", "lon2"]
).transform_filter(
    select_city
)

points = alt.Chart(flights_airport).mark_circle().encode(
    latitude="lat:Q",
    longitude="long:Q",
    size=alt.Size("routes:Q", scale=alt.Scale(range=[0, 1000]), legend=None),
    order=alt.Order("routes:Q", sort="descending"),
    tooltip=["origin:N", "routes:Q"]
).transform_aggregate(
    routes="count()",
    groupby=["origin"]
).transform_lookup(
    lookup="origin",
    from_=lookup_data
).transform_filter(
    (alt.datum.state != "PR") & (alt.datum.state != "VI")
).add_selection(
    select_city
)

cht = (background + connections + points).configure_view(stroke=None)
cht.save("/cdn/altair.html")

from IPython.display import HTML

HTML("<iframe src=altair.html width=900 height=600>")


flights_delayed = spark.sql("SELECT Origin, count(1) as total, sum(case when DepDelay > 0 then 1 else 0 end) as delayed from flights group by Origin").toPandas()
delayed_fraction = flights_delayed["delayed"].astype("float64") / flights_delayed["total"]
import matplotlib.pyplot as plt
import numpy as np

#plt.figure()
#plt.hist(delayed_fraction, bins=100)

import pandas as pd
airports_with_delays = pd.merge(flights_delayed, airports, left_on=['Origin'], right_on=['iata'], how='left')

# Investigate the spatial dependence of propensity to delay
# with a variogram. The variogram doesn't show strong evidence
# of spatial dependence.
import skgstat
v = skgstat.Variogram(np.vstack([airports_with_delays["long"], airports_with_delays["lat"]]).transpose(), delayed_fraction).plot()
