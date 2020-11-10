
import numpy as np
import tensorflow as tf
from tensorflow import keras

imdb = keras.datasets.imdb

# load the data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

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