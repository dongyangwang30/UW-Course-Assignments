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

##### Q1
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

##b

df_q1 = pd.read_csv('/content/drive/MyDrive/faces.csv', header = None)
df_q1 = df_q1.to_numpy()
df_q1t = np.transpose(df_q1)

sigma = np.matmul(df_q1t, df_q1) / len(df_q1)
eigen_val, eigen_vec = np.linalg.eig(sigma)

for i in [1,2,10,30,50]:
    print('The ' + str(i) + 'th  Eigenvalue is ' +  str(eigen_val[i-1]))

trace = np.trace(sigma)
print('The trace of the sigma matrix is ' +  str(trace))

k = np.arange(1,51)
eigen_val_50 = eigen_val[:50]
recon_error = []

for i in k:
  score = 1 - sum(eigen_val_50[:i]).real/trace
  recon_error.append(score)

plt.plot(k,recon_error)
plt.ylabel('Fractional Reconstruction Error')
plt.xlabel('k')
plt.title('Fractional Reconstruction Error vs k')

##c

eigen_vec_real = eigen_vec.real

fig, axs = plt.subplots(2, 5)
ind = 0
for i in range(1,3):
  for j in range(1,6):
    axs[i-1, j-1].imshow(eigen_vec_real.transpose()[ind].reshape(84,96).transpose(),cmap='gray')
    axs[i-1, j-1].set_title('Eigenvector' + str(ind+1))
    ind += 1

fig.set_size_inches(18.5, 10.5)

##d
k = [1,2,5,10,50]
imgs = [0, 23, 64, 67, 256]

fig1, axs1 = plt.subplots(5, 6)

for i in range(1,6):
  ind = 0
  for j in range(1,6):
    axs1[j-1, 0].imshow(df_q1[imgs[j-1]].reshape(84,96).transpose(), cmap='gray', vmin=0, vmax=1)
    axs1[j-1, 0].set_title('Original ' + 'image ' + str(imgs[j-1]))
  reconstruction_matrix = np.matmul(eigen_vec_real[:,:k[i-1]], eigen_vec_real[:,:k[i-1]].transpose())
  for j in range(1,6):
    result = np.matmul(df_q1[imgs[j-1]], reconstruction_matrix)
    axs1[j-1, i].imshow(result.reshape(84,96).transpose(), cmap='gray', vmin=0, vmax=1)
    axs1[j-1, i].set_title('k = ' + str(k[i-1]) )

fig1.set_size_inches(18.5, 18.5)

##### Q2
 
data_q2 = sc.textFile('/content/drive/MyDrive/data.txt').map(lambda line: [float(i) for i in line.split(" ")])
c1 = sc.textFile('/content/drive/MyDrive/c1.txt').map(lambda line: [float(i) for i in line.split(" ")])
c2 = sc.textFile('/content/drive/MyDrive/c2.txt').map(lambda line: [float(i) for i in line.split(" ")])
rdd_data = data_q2.zipWithIndex()
rdd_c1 = c1.zipWithIndex()
rdd_c2 = c2.zipWithIndex()

# Euclidean

centroids = rdd_c1
iterations = 20
cost_res_E1 = []
index1 = 0

while index1 <= iterations:

  # Generate all possible pairs
  all_pairs = rdd_data.cartesian(centroids)

  # Include distance
  pairs_with_distance = all_pairs.map(lambda x: 
                                      (x[0][1],(x[0][0], x[1][1],
                                      np.sum( [ (a - b)**2 for a, b in zip(x[0][0], x[1][0])  ] )
                                      )))

  # Keep only closest pairs
  closest_pairs = pairs_with_distance.reduceByKey(lambda x,y: x if x[2]<y[2] else y)

  # Cost calculation
  cost = closest_pairs.map(lambda x : x[1][-1]).sum()
  cost_res_E1.append(cost)

  # Group centroids by the points paired with them
  centroid_groups = closest_pairs.map(lambda x: (x[1][1], x[1][0])).groupByKey().mapValues(list)

  # Now update the centroids
  centroids = centroid_groups.map(lambda x: ( [*map(np.mean, zip(*x[1]) ) ] , x[0])).sortBy(lambda x: x[1])

  index1 += 1

  all_pairs.unpersist()
  pairs_with_distance.unpersist()
  closest_pairs.unpersist()
  centroid_groups.unpersist()

centroids = rdd_c2
cost_res_E2 = []
index1 = 0

while index1 <= iterations:

  # Generate all possible pairs
  all_pairs = rdd_data.cartesian(centroids)

  # Include distance
  pairs_with_distance = all_pairs.map(lambda x: 
                                      (x[0][1],(x[0][0], x[1][1],
                                      np.sum( [ (a - b)**2 for a, b in zip(x[0][0], x[1][0])  ] )
                                      )))

  # Keep only closest pairs
  closest_pairs = pairs_with_distance.reduceByKey(lambda x,y: x if x[2]<y[2] else y)

  # Cost calculation
  cost = closest_pairs.map(lambda x : x[1][-1]).reduce(lambda x,y: x+y)
  cost_res_E2.append(cost)

  # Group centroids by the points paired with them
  centroid_groups = closest_pairs.map(lambda x: (x[1][1], x[1][0])).groupByKey().mapValues(list)

  # Now update the centroids
  centroids = centroid_groups.map(lambda x: ( [*map(np.mean, zip(*x[1]) ) ] , x[0])).sortBy(lambda x: x[1])

  index1 += 1

  all_pairs.unpersist()
  pairs_with_distance.unpersist()
  closest_pairs.unpersist()
  centroid_groups.unpersist()

plt.plot(k, cost_res_E1, label = "c1")
plt.plot(k, cost_res_E2, label = "c2")
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.legend()
plt.title('Euclidean Distance Results')
plt.style.use('_classic_test_patch')

print ("The percentage of error reduction for c1 is: ", (1- cost_res_E1[10]/cost_res_E1[0])*100, "%")
print ("The percentage of error reduction for c2 is: ", (1- cost_res_E2[10]/cost_res_E2[0] )*100, "%")

# Manhattan
centroids = rdd_c1
cost_res_M1 = []
index1 = 0

while index1 <= iterations:

  # Generate all possible pairs
  all_pairs = rdd_data.cartesian(centroids)

  # Include distance
  pairs_with_distance = all_pairs.map(lambda x: 
                                      (x[0][1],(x[0][0], x[1][1],
                                      np.sum( [ ((a - b)**2)**0.5 for a, b in zip(x[0][0], x[1][0])  ] )
                                      )))

  # Keep only closest pairs
  closest_pairs = pairs_with_distance.reduceByKey(lambda x,y: x if x[2]<y[2] else y)

  # Cost calculation
  cost = closest_pairs.map(lambda x : x[1][-1]).reduce(lambda x,y: x+y)
  cost_res_M1.append(cost)

  # Group centroids by the points paired with them
  centroid_groups = closest_pairs.map(lambda x: (x[1][1], x[1][0])).groupByKey().mapValues(list)

  # Now update the centroids
  centroids = centroid_groups.map(lambda x: ( [*map(np.median, zip(*x[1]) ) ] , x[0])).sortBy(lambda x: x[1])

  index1 += 1

  all_pairs.unpersist()
  pairs_with_distance.unpersist()
  closest_pairs.unpersist()
  centroid_groups.unpersist()

centroids = rdd_c2
cost_res_M2 = []
index1 = 0

while index1 <= iterations:

  # Generate all possible pairs
  all_pairs = rdd_data.cartesian(centroids)

  # Include distance
  pairs_with_distance = all_pairs.map(lambda x: 
                                      (x[0][1],(x[0][0], x[1][1],
                                      np.sum( [ ((a - b)**2)**0.5 for a, b in zip(x[0][0], x[1][0])  ] )
                                      )))

  # Keep only closest pairs
  closest_pairs = pairs_with_distance.reduceByKey(lambda x,y: x if x[2]<y[2] else y)

  # Cost calculation
  cost = closest_pairs.map(lambda x : x[1][-1]).reduce(lambda x,y: x+y)
  cost_res_M2.append(cost)

  # Group centroids by the points paired with them
  centroid_groups = closest_pairs.map(lambda x: (x[1][1], x[1][0])).groupByKey().mapValues(list)

  # Now update the centroids
  centroids = centroid_groups.map(lambda x: ( [*map(np.median, zip(*x[1]) ) ] , x[0])).sortBy(lambda x: x[1])

  index1 += 1

  all_pairs.unpersist()
  pairs_with_distance.unpersist()
  closest_pairs.unpersist()
  centroid_groups.unpersist()

plt.plot(k, cost_res_M1, label = "c1")
plt.plot(k, cost_res_M2, label = "c2")
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.legend()
plt.title('Manhattan Distance Results')
plt.style.use('_classic_test_patch')

print ("The percentage of error reduction for c1 is: ", (1- cost_res_M1[10]/cost_res_M1[0])*100, "%")
print ("The percentage of error reduction for c2 is: ", (1- cost_res_M2[10]/cost_res_M2[0] )*100, "%")

##### Q3

user_artists = np.genfromtxt('user_artists.txt', delimiter='\t', names=True, dtype=None)
smallset = np.genfromtxt('user_artists_small.txt', delimiter='\t', names=True, dtype=None)
artists = np.genfromtxt('artists.txt', delimiter='\t', names=True, dtype=None)

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse

cur_data = user_artists

m = 1882
n = 3000
# m = 100
# n = 100

f = 3
lamda = 0.01

# construct lambda * I
lamda_eye = sparse.csr_matrix(np.eye(f)*lamda)

# Set up X, Y, R
X = sparse.csr_matrix(np.zeros((m, f)) + 0.5)  
Y = sparse.csr_matrix(np.zeros((n, f))) 
R = sparse.lil_matrix(np.zeros((m, n)))

with open('user_artists.txt' , 'r') as file:
  next(file)
  data = file.readlines()

for line in data:
  R[int(line.strip().split("\t") [0]), int(line.strip().split("\t") [1])] = float(line.strip().split("\t") [2])

# Set up P, C, sparsity ratio
P = (R > 0).astype(float)

sparsity_ratio = cur_data.size / ((m*n) - cur_data.size)

C = sparse.lil_matrix(np.zeros((m, n)) + 1.) +sparsity_ratio*R

sparsity_ratio

def ALS(X, Y, P, C, m, n, f, lamda, cost_all, MAX_ITER = 100):
  """"
  X: initial user matrix
  Y: initial item matrix
  P: preference matrix
  C: confidence matrix
  """

  for t in range(MAX_ITER):

    # first calculate cost
    cost = (P-X.dot(Y.T)).power(2).multiply(C).sum() + lamda*(np.sqrt(X.power(2).sum()) + np.sqrt(Y.power(2).sum()))
    cost_all.append(cost)

    # update item matrix
    xTx = X.T.dot(X)
    for item in range(n):
      p_i = P[:, item]
      C_i_I =  sparse.dia_matrix((np.array(C.T.getrow(item).todense()),0), shape=(m,m))-sparse.eye(m) 
      left = 	xTx + X.T.dot(C_i_I).dot(X)+ lamda_eye 
      right = X.T.dot(C_i_I).dot(p_i)+ X.T.dot(p_i)
      y_i = spsolve(left, right)
      Y[item] = y_i

    # update user matrix   
    yTy = Y.T.dot(Y)
    for user in range(m):
      p_u = P[user, :]
      C_u_I = sparse.dia_matrix((np.array(C.getrow(user).todense()),0), shape=(n,n))-sparse.eye(n)
      left = 	yTy + Y.T.dot(C_u_I).dot(Y)+ lamda_eye	   
      right = Y.T.dot(C_u_I).dot(p_u.T)+ Y.T.dot(p_u.T)
      x_u = spsolve(left, right)
      X[user] = x_u
  
  # calculate cost for the most recent one
  cost = (P-X.dot(Y.T)).power(2).multiply(C).sum() + lamda*(np.sqrt(X.power(2).sum()) + np.sqrt(Y.power(2).sum()))
  cost_all.append(cost)

  return X, Y

# Initialize X, Y
X = sparse.csr_matrix(np.zeros((m, f)) + 0.5)  
Y = sparse.csr_matrix(np.zeros((n, f)))

# Cost List
cost_all1 = []

X1, Y1 = ALS(X, Y, P, C, m, n, f, lamda, cost_all = cost_all1, MAX_ITER = 1)

cost_all1

# Initialize X, Y
X = sparse.csr_matrix(np.zeros((m, f)) + 0.5)  
Y = sparse.csr_matrix(np.zeros((n, f)))

# Cost List
cost_all100 = []

X100, Y100 = ALS(X, Y, P, C, m, n, f, lamda, cost_all = cost_all100, MAX_ITER = 100)
cost_all100

user0_1 = np.argpartition(np.array((X1.dot(Y1.T)).getrow(0).todense()[0]).flatten(), -2)[-2:]
user20_1 = np.argpartition(np.array((X1.dot(Y1.T)).getrow(20).todense()[0]).flatten(), -2)[-2:]

user0_100 = np.argpartition(np.array((X100.dot(Y100.T)).getrow(0).todense()[0]).flatten(), -2)[-2:]
user20_100 = np.argpartition(np.array((X100.dot(Y100.T)).getrow(20).todense()[0]).flatten(), -2)[-2:]

print('Recommendations for user 0 after 1 iteration are ' + 
      str(artists[user0_1][0][1]) + ' and ' +
      str(artists[user0_1][1][1]))

print('Recommendations for user 20 after 1 iteration are ' + 
      str(artists[user20_1][0][1]) + ' and ' +
      str(artists[user20_1][1][1]))

print('Recommendations for user 0 after 100 iteration are ' + 
      str(artists[user0_100][0][1]) + ' and ' +
      str(artists[user0_100][1][1]))

print('Recommendations for user 20 after 100 iteration are ' + 
      str(artists[user20_100][0][1]) + ' and ' +
      str(artists[user20_100][1][1]))

k = np.arange(0,101)

plt.plot(k, cost_all100)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.title('ALS Results')