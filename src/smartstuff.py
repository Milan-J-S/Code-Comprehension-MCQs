import keras
from keras.layers import Input
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import numpy as np

import os

file_path = 'C:\PES\8TH SEM\semi-intelligent-discussion-forum-for-code-master\semi-intelligent-discussion-forum-for-code-master\src\code'

codes = []
comments = []
vocab = set()
comments_vector = {}

for root,dirs,files in os.walk(file_path):
    for fl in files:
        codes.append(open('code/'+fl).read())
        comment = open('comments/'+fl).read()
        for c in comment.split():
            vocab.add(c)
        comments.append(comment)


print(codes)
print(comments)

i=0

for item in vocab:
    comments_vector[item]=i
    i=i+1

comments_tensor = []
for item in comments:
    comment_tensor = []
    for word in item.split():
        comment = np.zeros(len(vocab))
        comment[comments_vector[word]] = 1
        comment_tensor.append(comment)
    comments_tensor.append(comment_tensor)

print(comments_tensor)

model = Sequential()
# model.add(Input(shape = (2000,)))

model.add(Embedding(10, 100, input_length=50))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(TimeDistributed(Dense(len(vocab))))

model.compile(loss = 'categorical_crossentropy', optimizer='RMSProp', metrics = ['accuracy'])

print(model.summary())






