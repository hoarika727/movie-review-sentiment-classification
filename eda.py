#%%
import findspark
findspark.init('/opt/anaconda3/lib/python3.8/site-packages/pyspark')

import pyspark
import numpy as np

#%%
conf = pyspark.SparkConf().\
    setAppName('movie-sentiment-analysis').\
    setMaster('local[*]')

# %%
from pyspark.sql import SQLContext, HiveContext
from pyspark.sql import functions as fn
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml import Pipeline

sc = pyspark.SparkContext(conf=conf)
sqlContext = HiveContext(sc)
# %%
train_df = sqlContext.read.csv('train.csv', header=True)
train_df.printSchema()

test_df = sqlContext.read.csv('test.csv', header=True)
test_df.printSchema()

#%%
import matplotlib.pyplot as plt
import seaborn as sns

label_count = train_df.groupBy('labels').agg(fn.count('*')).toPandas()
label_count.columns = ['labels','count']

p = plt.bar(label_count['labels'],label_count['count'], width=0.5, align='center')
p[0].set_color('r')
plt.yticks(np.arange(0, 13000, step=2000))
plt.title('Labels Count')
plt.show()

#%%
# Feature engineering + term frequency for EDA
tokenizer = Tokenizer().setInputCol("reviews")\
  .setOutputCol("words")

review_raw_tokens = tokenizer.transform(train_df)

word_raw_freq = review_raw_tokens.withColumn("words", fn.explode(fn.col("words"))) \
  .groupBy("words") \
  .agg(fn.count("*")).toPandas()
word_raw_freq.columns = ['words','count']
word_raw_freq = word_raw_freq.sort_values(by=['count'], axis=0, ascending=False)

word_raw_freq_trim = word_raw_freq.reset_index(drop=True)
word_raw_freq_trim = word_raw_freq_trim[:25]

sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x='words', y='count', data=word_raw_freq_trim, palette='Blues_d')
plt.xticks(rotation=45, fontsize=15)
plt.xlabel(xlabel="Words", fontsize=18)
plt.yticks(fontsize=12)
plt.ylabel(ylabel = "Count", fontsize=18)
plt.title('Word frequency in movie reviews', fontsize=20, pad=5)
plt.show()

#%%
from nltk.corpus import stopwords
from wordcloud import WordCloud
stop_words = stopwords.words('english')
stop_words.append('br')
stopwords_processor = StopWordsRemover() \
                      .setStopWords(stop_words)\
                      .setCaseSensitive(False)\
                      .setInputCol("words").setOutputCol("filtered_words")
review_tokens = stopwords_processor.transform(review_raw_tokens)
word_freq = review_tokens.withColumn("filtered_words", fn.explode(fn.col("filtered_words"))) \
  .groupBy("filtered_words") \
  .agg(fn.count("*")).toPandas()
word_freq.columns = ['words','count']
word_freq = word_freq.sort_values(by=['count'], axis=0, ascending=False)

word_freq_trim = word_freq.reset_index(drop=True)
word_freq_trim = word_freq_trim[:25]

sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x='words', y='count', data=word_freq_trim, palette='BuGn_d')
plt.xticks(rotation=45, fontsize=15)
plt.xlabel(xlabel="Words", fontsize=18)
plt.yticks(fontsize=12)
plt.ylabel(ylabel = "Count", fontsize=18)
plt.title('Word frequency in movie reviews', fontsize=20, pad=5)
plt.show()

sns.distplot(a=np.log(word_freq['count']))
plt.xticks(rotation=45, fontsize=15)
plt.xlabel(xlabel="Word Frequency (logarithmic)", fontsize=18)
plt.yticks(fontsize=12)
plt.ylabel(ylabel = "Distribution Ratio", fontsize=18)
plt.title('Word frequency distribution in movie reviews', fontsize=20, pad=5)
plt.show()

reviews = review_tokens.toPandas()
word_list = ''
for i in range(len(reviews)):
  word_list += ' '.join(reviews['filtered_words'][i])+' '

wordcloud = WordCloud(width = 480, height = 320, 
                background_color ='white', 
                stopwords = stop_words, 
                max_font_size = 90,
                min_font_size = 10).generate(word_list) 
  
# plot the WordCloud image                        
plt.figure(figsize = (6, 4), facecolor = 'w') 
plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off") 
plt.margins(x=0, y=0)
plt.tight_layout(pad = 0) 
plt.show() 

#%%
pos_train_df = train_df.where(fn.col('labels') == 1)
neg_train_df = train_df.where(fn.col('labels') == 0)

pos_pipeline = Pipeline(stages=[tokenizer, stopwords_processor]).fit(pos_train_df)
neg_pipeline = Pipeline(stages=[tokenizer, stopwords_processor]).fit(neg_train_df)

pos = pos_pipeline.transform(pos_train_df)
neg = neg_pipeline.transform(neg_train_df)

word_freq_pos = pos.withColumn("filtered_words", fn.explode(fn.col("filtered_words"))) \
  .groupBy("filtered_words") \
  .agg(fn.count("*")).toPandas()
word_freq_pos.columns = ['words','count']
word_freq_pos = word_freq_pos.sort_values(by=['count'], axis=0, ascending=False)

word_freq_pos_trim = word_freq_pos.reset_index(drop=True)
word_freq_pos_trim = word_freq_pos_trim[:25]

sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x='words', y='count', data=word_freq_pos_trim, palette='BuGn_d')
plt.xticks(rotation=45, fontsize=15)
plt.xlabel(xlabel="Words", fontsize=18)
plt.yticks(fontsize=12)
plt.ylabel(ylabel = "Count", fontsize=18)
plt.title('Word frequency in positive class movie reviews', fontsize=20, pad=5)
plt.show()

sns.distplot(a=np.log(word_freq_pos['count']))
plt.xticks(rotation=45, fontsize=15)
plt.xlabel(xlabel="Word Frequency (logarithmic)", fontsize=18)
plt.yticks(fontsize=12)
plt.ylabel(ylabel = "Distribution Ratio", fontsize=18)
plt.title('Word frequency distribution in positive class movie reviews', fontsize=20, pad=5)
plt.show()

word_freq_neg = neg.withColumn("filtered_words", fn.explode(fn.col("filtered_words"))) \
  .groupBy("filtered_words") \
  .agg(fn.count("*")).toPandas()
word_freq_neg.columns = ['words','count']
word_freq_neg = word_freq_neg.sort_values(by=['count'], axis=0, ascending=False)

word_freq_neg_trim = word_freq_neg.reset_index(drop=True)
word_freq_neg_trim = word_freq_neg_trim[:25]

sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x='words', y='count', data=word_freq_neg_trim, palette='BuGn_d')
plt.xticks(rotation=45, fontsize=15)
plt.xlabel(xlabel="Words", fontsize=18)
plt.yticks(fontsize=12)
plt.ylabel(ylabel = "Count", fontsize=18)
plt.title('Word frequency in negative class movie reviews', fontsize=20, pad=5)
plt.show()

sns.distplot(a=np.log(word_freq_neg['count']))
plt.xticks(rotation=45, fontsize=15)
plt.xlabel(xlabel="Word Frequency (logarithmic)", fontsize=18)
plt.yticks(fontsize=12)
plt.ylabel(ylabel = "Distribution Ratio", fontsize=18)
plt.title('Word frequency distribution in negative class movie reviews', fontsize=20, pad=5)
plt.show()
#%%
from collections import Counter
freq = Counter(word_freq['count'])
freq