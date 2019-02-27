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

from nltk.tokenize import word_tokenize
import os

file_path = 'code'

codes = []
comments = []
vocab = set()
comments_vector = {}

for root,dirs,files in os.walk(file_path):
    for fl in files:
        code = open('code/'+fl).read()
        code_tensor =  np.asarray(word_tokenize(code)[:1000])
        print(len(code_tensor), "--->")
        if(len(code_tensor) < 1000):
            code_tensor = np.pad(code_tensor, (0, 1000 - len(code_tensor)) , 'constant' )
        print(len(code_tensor))
        codes.append(code_tensor)
        comment = open('comments/'+fl).read()
        for c in comment.split():
            vocab.add(c)
        comments.append(comment)


# print(comments)

print(np.asarray(codes).shape)

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

print(np.asarray(comments_tensor).shape)

model = Sequential()
# model.add(Input(shape = (2000,)))

model.add(Embedding(10, 100, input_length=1000))
model.add(LSTM(500, return_sequences=True))
model.add(LSTM(500, return_sequences=True))
model.add(TimeDistributed(Dense(len(vocab))))

model.compile(loss = 'categorical_crossentropy', optimizer='RMSProp', metrics = ['accuracy'])

print(model.summary())

model.fit(np.asarray(codes), np.asarray(comments_tensor), epochs=2, verbose = 1)








