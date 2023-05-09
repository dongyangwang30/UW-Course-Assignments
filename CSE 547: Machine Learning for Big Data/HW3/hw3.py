'''
!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

from google.colab import drive
drive.mount('/content/drive')

'''

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from pyspark.rdd import RDD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

conf = SparkConf().set("spark.ui.port", "4050")
sc = SparkContext(conf=conf)
spark = SparkSession.builder \
.appName('app_name') \
.master('local[*]') \
.config('spark.sql.execution.arrow.pyspark.enabled', True) \
.config('spark.sql.session.timeZone', 'UTC') \
.config('spark.driver.memory','32G') \
.config('spark.ui.showConsoleProgress', True) \
.config('spark.sql.repl.eagerEval.enabled', True) \
.getOrCreate()

##### Q2

#### a

# Initialize
# edges = sc.textFile('/content/drive/MyDrive/graph-small.txt').map(lambda line: line.split()).map(lambda item: (int(item[0]), int(item[1])))
edges = sc.textFile('/content/drive/MyDrive/graph-full.txt').map(lambda line: line.split()).map(lambda item: (int(item[0]), int(item[1])))
n = 1000
beta = 0.8

# Edges with degree
distinct_edges = edges.distinct()
out_degree = distinct_edges.map(
  lambda x: (x[0], 1)).reduceByKey(
    lambda a, b: a + b)
edges_with_degree = distinct_edges.join(out_degree)

edges_with_degree.collect()

r = sc.parallelize([(i+1, 1/n) for i in range(0, n)])

for i in range(40):
    r = r.join(edges_with_degree).map(
        lambda x: (x[1][1][0], x[1][0] * beta / x[1][1][1])).reduceByKey(
            lambda a, b: a +b).mapValues(
                lambda v: v + (1 - beta) / n)

res = r.collect()
res = sorted(res, key=lambda x: x[1])
res[:5]

print("The top 5 nodes are ", [i for i,j in res[::-1][:5]])
print("The bottom 5 nodes are ", [i for i,j in res[:5]])

#### b

# Initialize
lmbda = 1
mu = 1
h = [(i + 1, 1) for i in range(0, n)]
h = sc.parallelize(h)

for i in range(40):
   
    h = edges.join(h).map(
        lambda x: (x[1][0], mu * x[1][1])).reduceByKey(
            lambda a, b: a + b)
    max_value_a = h.max(lambda x: x[1])[0]
    a = h.map(lambda x: (x[0], x[1] / max_value_a))

    a = edges.map(
        lambda x: (x[1], x[0])).join(a).map(
            lambda x: (x[1][0], lmbda * x[1][1])).reduceByKey(
                lambda a, b: a + b)
    max_value_h = a.max(lambda x: x[1])[0]
    h = a.map(lambda x: (x[0], x[1] / max_value_h))

res = h
res = sorted(res, key=lambda x: x[1])
res[:5]

print("The top 5 hubbiness score nodes are ", [i for i,j in res[::-1][:5]])
print("The bottom 5 hubbiness score nodes  are ", [i for i,j in res[:5]])

res = a.collect()
res = sorted(res, key=lambda x: x[1])
res[:5]

print("The top 5 authority score nodes are ", [i for i,j in res[::-1][:5]])
print("The bottom 5 authority score nodes  are ", [i for i,j in res[:5]])

##### Q3
df = pd.read_csv('/content/drive/MyDrive/songs.csv')
df.head()
df.info()

# Calculate distance for dataset
from scipy.spatial.distance import cdist
dist = cdist(df, df, metric='euclidean')

# Calculate the adjancancy matrix A
A = dist.copy()
for i in range(len(dist)):
  for j in range(len(dist)):
    if dist[i,j] < 1:
      A[i,j] = 1
    else:
      A[i,j] = 0

# Calculate D and L
D = np.diag(np.sum(A, axis = 1))
L = D-A

# Calculate the D^{1/2} and L_hat
D_half = np.linalg.inv(np.sqrt(D))
L_hat = D_half.dot(L).dot(D_half)

# Calculate eigenvalues, eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(L_hat)

# Second smallest eigenvector and optimal solution
v = eigenvectors[:, np.argpartition(eigenvalues, 2)[1]]
x = D_half.dot(v)

# Get first 5 users clustering result
for i in range(5):
  print("The number " + str(i+1) + " user belongs to group " + str(int(x[i] < 0) + 1) + "\n")

comparison = pd.concat([df[x >= 0].sum(axis = 0), df[x < 0].sum(axis = 0)], axis = 1)
comparison['difference'] = np.abs(comparison[0] - comparison[1])
comparison = comparison.sort_values(by = "difference", ascending=False)
for i in range(3):
  print("The number " + str(i + 1) + " feature is " + comparison.index[i] + " with value " + str(comparison["difference"][i]) + "\n")

for i in range(3):
  col_name = comparison.index[i]
  result = scipy.stats.ttest_ind(df[x >= 0][col_name], df[x < 0][col_name], equal_var = True)
  print("The p-value for the feature " + col_name + " is " + str(result.pvalue) + "\n")

