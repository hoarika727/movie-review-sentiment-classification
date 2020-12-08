
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import re
import csv

imdb = keras.datasets.imdb

# load the data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=100000)
word_index = imdb.get_word_index(path="imdb_word_index.json")

# prepare the data for other models
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])

def decode(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

for i in tqdm(range(len(train_data))):
    train_data[i] = decode(train_data[i])
    test_data[i] = decode(test_data[i])

train_data = [re.sub('<START> ','',review) for review in train_data]
test_data = [re.sub('<START> ','',review) for review in test_data]

train = pd.DataFrame({'reviews':train_data, 'labels':train_labels})
test = pd.DataFrame({'reviews':test_data, 'labels':test_labels})
train.to_csv('train.csv')
test.to_csv('test.csv')

train = pd.read_csv('train.csv', index_col='index')
test = pd.read_csv('test.csv', index_col='index')

'''
explore the data
# existing word and the index
word_index = imdb.get_word_index()

# get max length of the review and its index, mean length, frequecies
total, max, idx = 0, 0, 0
for i in range(len(train_data)):
    total += len(train_data[i])
    if (len(train_data[i]) > max):
        max = len(train_data[i]) #2494
        idx = i                  #17934
mean = total/len(train_data)     #239

import matplotlib.pyplot as plt
plt.hist(word_length)            #250 the most
'''

# Build a baseline TensorFlow CNN model
# Transform the data into the same length 
for i in range(len(train_data)):
    if len(train_data[i]) < 250:
        train_data[i] = train_data[i]+[0]*(250-len(train_data[i]))
    else:
        train_data[i] = train_data[i][0:250]

for j in range(len(test_data)):
    if len(test_data[j]) < 250:
        test_data[j] = test_data[j]+[0]*(250-len(test_data[j]))
    else:
        test_data[j] = test_data[j][0:250]

# Convert data into tensors
train_tf = [item for item in train_data]
train_tf = tf.convert_to_tensor(train_tf, dtype=tf.int32)

test_tf = [item for item in test_data]
test_tf = tf.convert_to_tensor(test_tf, dtype=tf.int32)

# Split the training data into train set and validation set for validation
x_val = train_tf[:10000]
partial_x_train = train_tf[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Construct the model
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.Conv1D(16, 5, padding="valid", activation="relu", strides=3))
model.add(keras.layers.GlobalAveragePooling1D())
#model.add(keras.layers.Dense(8, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

model.summary()
# Track the training history
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=256,
                    validation_data=(x_val, y_val),
                    verbose=1)

#Evaluate the model with test test
results = model.evaluate(test_tf, test_labels)
print(results) 