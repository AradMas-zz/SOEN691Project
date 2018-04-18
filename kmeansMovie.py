import sys
import random
from pyspark.sql import Row
from pyspark import SparkConf, SparkContext
from scipy.spatial import distance

sparkConf=SparkConf().setAppName("").setMaster("local")
sc = SparkContext(conf = sparkConf)

data_file1="D:/movies.dat"
data_file2="D:/tags.dat"
data_file3="D:/tag_relevance.dat"
output_file="D:/result.txt"
k="5"
random_seed="123"

lines = sc.textFile(data_file3)
parts = lines.map(lambda row: row.split(","))
ratingsRDD = parts.map(lambda p: Row(movieId=int(p[0]),tagId=int(p[1]),tagRelevance=float(p[2])))

movieArr = []
tagRelArr = []
i=0
tmpMovieId=0
for x in ratingsRDD.collect():
    if i<1128 and (tmpMovieId==0 or tmpMovieId==x.movieId):
        tmpMovieId=x.movieId
        tagRelArr.append(x.tagRelevance)
        if i==1127:
            movieArr.append([x.movieId,tagRelArr])
            i=-1
            tagRelArr=[]
            tmpMovieId=0
        i=i+1
rdd = sc.parallelize(movieArr)
random.seed(int(random_seed))
init_states=random.sample(movieArr,  int(k))
centroids = rdd.filter(lambda x: x in init_states).collect()

def dist(a, b):
    return int(round(distance.euclidean(a, b)** 2))
distArr=[None] * len(centroids) 
cluster = [[] for _ in range(len(centroids))]
oldClusters=[[] for _ in range(len(centroids))]

for x in rdd.collect():
    mindis=0
    minindex=0
    for i in range(len(centroids)):
        distArr[i] = dist(x[1], centroids[i][1])
    for i in range(len(centroids)):   
        if i==0 or distArr[i]<=mindis:
            mindis=distArr[i]
            minindex=i
    cluster[minindex].append(x[0])

def Diff(li1, li2):
    retVal=True
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    if li1!=[] and li_dif==[]:
        retVal=False
    return retVal

def makeCentroids(clusterTmp,rdd):
    newCentroids=[]
    for i in range(len(clusterTmp)):
        clusterItem = rdd.filter(lambda x: x[0] in clusterTmp[i]).map(lambda x:x[1])
        newrdd=list(map(lambda x:sum(x)/float(len(x)), zip(*clusterItem.collect())))
        newCentroids.append(newrdd)
    return newCentroids        
               
while Diff(cluster,oldClusters):
    oldClusters=cluster
    newCentroids=makeCentroids(cluster,rdd)
    cluster = [[] for _ in range(len(centroids))]
    for x in rdd.collect():
        mindis=0
        minindex=0
        for i in range(len(newCentroids)):
            distArr[i] = dist(x[1], newCentroids[i])
        for i in range(len(newCentroids)):   
            if i==0 or distArr[i]<=mindis:
                mindis=distArr[i]
                minindex=i
        cluster[minindex].append(x[0])

for i in range(len(sorted(cluster))):
    cluster[i].sort()
cluster.sort()
f = open(output_file,'w')
for i in range(len(sorted(cluster))):
    cluster[i].sort()
    f.write("* Class "+str(i)+"\n")
    for j in range(len(cluster[i])):
        f.write(str(cluster[i][j])+" ")
    f.write("\n")
f.close()
