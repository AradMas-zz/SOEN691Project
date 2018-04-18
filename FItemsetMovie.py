import sys
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark import SparkConf, SparkContext
sparkConf=SparkConf().setAppName("").setMaster("local")
sc = SparkContext(conf = sparkConf)
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import size,desc
spark = SparkSession.builder.getOrCreate()
dataFileAds="D:/ratings.csv"
n=15
s=0.1
c=0.8

if sys.version >= '3':
    long = int
lines = sc.textFile(dataFileAds)
parts = lines.map(lambda row: row.split(","))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]),movieId=int(p[1]),rating=float(p[2])))

userArr = []
tagMovArr = []
i=0
tmpUserId=0
for x in ratingsRDD.collect():
    if tmpUserId==0 or tmpUserId==x.userId:
        tmpUserId=x.userId
        if x.rating>3:
            tagMovArr.append(x.movieId)
    elif tmpUserId!=x.userId:
        userArr.append([tmpUserId,tagMovArr])
        tagMovArr=[]
        if x.rating>3:
            tagMovArr.append(x.movieId)
        tmpUserId=x.userId
rdd = sc.parallelize(userArr) 
df_raw = spark.createDataFrame(rdd)

df = df_raw.select(df_raw[0].alias('user'),df_raw[1].alias('items')).withColumn("id", monotonically_increasing_id())
dfCount=df.count()
fpGrowth = FPGrowth(itemsCol="items", minSupport=s, minConfidence=c)
model = fpGrowth.fit(df)
mf=model.associationRules
mf2=model.freqItemsets
 
mf=mf.withColumn("count", size(mf.antecedent))
mf=mf.join(mf2, mf.consequent == mf2.items, 'inner')
mf=mf.select(mf.antecedent,mf.consequent,mf.confidence,mf2.items,mf2.freq,
             (mf.confidence-(mf2["freq"]/dfCount)).alias('interest'))\
             .sort(desc("count"),desc("interest"))
mf.show(1000)
print(mf.count())