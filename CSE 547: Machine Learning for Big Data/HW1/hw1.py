##### Q1
 
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

file = sc.textFile("/content/drive/MyDrive/soc-LiveJournal1Adj.txt")

def read_f(line):
  row = line.split('\t')
  friends = row[1].split(',') if len(row[1]) > 0 else []
  return (row[0],friends)

friends_list = file.map(lambda line: read_f(line))
#friends_list.collect()

friends_list_Dict = friends_list.collectAsMap()

def mutual_friends(friends_list):
  res = {}
  for person in friends_list:
    #print(person)
    mutual = {}
    for friend in friends_list[person]:
      for two_friend in friends_list[friend]:
        if two_friend != friend:
          if two_friend in mutual:
            mutual[two_friend] += 1
          else:
            mutual[two_friend] = 1
    temp = []
    mutual = sorted(mutual.items(), key=lambda x:( -x[1], x[0]))
    res[person] = mutual
  result = []
  for i in res.keys():
    result.append((i, res[i]))
  return result
  
mutualfriends = mutual_friends(friends_list_Dict)

user_ids = ['924', '8941', '8942', '9019', '9020', '9021', '9022', '9990', '9992', '9993']
#user_ids = ['11']

res = []
for item in mutualfriends:
  if item[0] in user_ids:
    res.append(item)

recommendations = []
for i in res:
  recommendation = []
  for item in i[1]:
    if item[0] not in friends_list_Dict[i[0]] and item[0] != i[0]:
      recommendation.append(item[0])
  recommendations.append((i[0], recommendation[:10]))

#for i in range(len(recommendations)):
#  recommendations[i][1] = recommendations[i][1][:10]
print(recommendations)

##### Q2

q2data = open("/content/drive/MyDrive/browsing.txt", "r")
support = 100

filtered_items = {}

for line in q2data:
  items = line.strip().split(" ")
  for item in items:
    if item not in filtered_items:
      filtered_items[item] = 1
    else: 
      filtered_items[item] += 1

filtered_items_final = {}
for item in filtered_items:
  if filtered_items[item] >= 100:
    filtered_items_final[item] = filtered_items[item]
filtered_items_final

q2data = open("/content/drive/MyDrive/browsing.txt", "r")

pair_items = {}

for line in q2data:
  tokens = line.strip().split(" ")
  for i in range(0, len(tokens) - 1):
    for j in range(i + 1, len(tokens)):
      if tokens[i] in filtered_items_final and tokens[j] in filtered_items_final:
        key = tuple(sorted((tokens[i], tokens[j])))
        if key not in pair_items:
          pair_items[key] = 1
        else: 
          pair_items[key] += 1

pair_items_final = {}
for item in pair_items:
  if pair_items[item] >= support: pair_items_final[item] = pair_items[item]
pair_items_final

res_2d = []
for key, val in pair_items_final.items():
  res_2d.append( [key[0], key[1], val, val/filtered_items_final[key[0]]] )
  res_2d.append( [key[1], key[0], val, val/filtered_items_final[key[1]]])

print(sorted(res_2d, key =  lambda x: -x[-1])[:5])

q2data = open("/content/drive/MyDrive/browsing.txt", "r")

triple_items = {}

for line in q2data:
  tokens = line.strip().split(" ")
  for i in range(0, len(tokens) - 2):
    for j in range(i + 1, len(tokens) - 1):
      for k in range(j + 1, len(tokens)):
        key1 = tuple(sorted((tokens[i], tokens[j])))
        key2 = tuple(sorted((tokens[i], tokens[k])))
        key3 = tuple(sorted((tokens[j], tokens[k])))
        if key1 in pair_items_final and key2 in pair_items_final and key3 in pair_items_final:
          key = tuple(sorted((tokens[i], tokens[j], tokens[k])))
          if key not in triple_items:
            triple_items[key] = 1
          else: 
            triple_items[key] += 1

triple_items_final = {}
for item in triple_items:
  if triple_items[item] >= 100: triple_items_final[item] = triple_items[item]
triple_items_final

res_2e = []
for key, val in triple_items_final.items():
  #print((key[0], key[1]))
  res_2e.append( [ (key[0], key[1]), key[2], val/pair_items_final[(key[0], key[1]) ] ] )
  res_2e.append( [(key[1], key[2]), key[0], val/pair_items_final[(key[1], key[2])]] )
  res_2e.append( [(key[0], key[2]), key[1], val/pair_items_final[(key[0], key[2])]] )

print(sorted(res_2e, key =  lambda x: (-x[-1], x[0]))[:5])