#%%
#import findspark
#findspark.init('/opt/anaconda3/lib/python3.8/site-packages/pyspark')

import pyspark
from pyspark.sql import SQLContext, HiveContext
from pyspark.sql import functions as fn
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml import Pipeline

import numpy as np
import pandas as pd
from nltk.corpus import stopwords

conf = pyspark.SparkConf()\
    .setAppName('movie-sentiment-analysis')\
    .setMaster('local[*]')

sc = pyspark.SparkContext(conf=conf)
sqlContext = HiveContext(sc)

train_df = sqlContext.read.csv('train.csv', header=True)
train_df = train_df.selectExpr("cast(index as int) index","cast(reviews as string) reviews","cast(labels as int) labels")
test_df = sqlContext.read.csv('test.csv', header=True)
test_df = test_df.selectExpr("cast(index as int) index","cast(reviews as string) reviews","cast(labels as int) labels")
#%%
tokenizer = Tokenizer().setInputCol("reviews")\
  .setOutputCol("words")

stop_words = stopwords.words('english')
stop_words.append('br')
stopwords_processor = StopWordsRemover() \
                      .setStopWords(stop_words)\
                      .setCaseSensitive(False)\
                      .setInputCol("words").setOutputCol("filtered_words")
            
cv = CountVectorizer(minTF=1., minDF=5., vocabSize=2**10)\
                    .setInputCol('filtered_words')\
                    .setOutputCol('features')

cv_pipeline = Pipeline(stages=[tokenizer, stopwords_processor, cv]).fit(train_df)

# %%
from pyspark.ml.feature import IDF
from pyspark.ml.classification import LogisticRegression

idf = IDF().\
    setInputCol('features').\
    setOutputCol('tfidf')

idf_pipeline = Pipeline(stages=[cv_pipeline, idf]).fit(train_df)

#%%
lr = LogisticRegression().\
    setLabelCol('labels').\
    setFeaturesCol('features').\
    setRegParam(0.0).\
    setMaxIter(100).\
    setElasticNetParam(0.)

lr_pipeline = Pipeline(stages=[idf_pipeline, lr]).fit(train_df)

lr_pipeline.transform(test_df).\
    select(fn.expr('float(prediction = labels)').alias('correct')).\
    select(fn.avg('correct')).show()

# %%
vocabulary = idf_pipeline.stages[0].stages[-1].vocabulary
weights = lr_pipeline.stages[-1].coefficients.toArray()
coeffs_df = pd.DataFrame({'word': vocabulary, 'weight': weights})

# Weight for words more likely to contribute to negative sentiment
coeffs_df.sort_values('weight').head(10)

# Weight for words more likely to contribute to positive sentiment
coeffs_df.sort_values('weight', ascending=False).head(10)

coeffs_df.query('weight == 0.0').shape #0

# %%
# MPC - Feedforward ANN
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 2 (classes)

feature_len = len((idf_pipeline.transform(train_df).toPandas())['tfidf'][0])
layers = [feature_len, int(feature_len/4), 16, 2]

mpc = MultilayerPerceptronClassifier(featuresCol = 'tfidf',labelCol='labels', maxIter=100, layers=layers, blockSize=128, seed=7) 

mpc_pipeline = Pipeline(stages=[idf_pipeline, mpc]).fit(train_df)

result = mpc_pipeline.transform(test_df)

# compute accuracy on the test set
predictionAndLabels = result.select("prediction", "labels")
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",labelCol="labels")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))