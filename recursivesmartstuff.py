import keras
from keras.layers import Input
from keras.datasets import imdb
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import concatenate
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence

import numpy as np

from nltk.tokenize import word_tokenize
import os

file_path = 'code'

codes = []
comments = []
vocab = set()
comments_vector = {}

all_codes_tensor = []
all_codes = []

for root,dirs,files in os.walk(file_path):
    for fl in files:
        code = open('code/'+fl).read()
        all_codes.append(code)
        code_tensor =  np.asarray(word_tokenize(code)[:1000])
        all_codes_tensor.extend(code_tensor)
        if(len(code_tensor) < 1000):
            code_tensor = np.pad(code_tensor, (0, 1000 - len(code_tensor)) , 'constant' )
        print(len(code_tensor))
        codes.append(code_tensor)
        comment = open('comments/'+fl).read()
        for c in comment.split():
            vocab.add(c)
        comments.append(comment)

rec_codes = []

i=0
for item in vocab:
    comments_vector[item]=i
    i=i+1

comments_tensor = []
output_tensor = []
output_tensor = []
for i in range(len(comments)):
    comment_tensor = []
    for k in range(10):
        comment_tensor.append(np.zeros(len(vocab)))

    prevlen = 0
    for word in comments[i].split():
        # print(len(comment_tensor))
        comments_tensor.append(comment_tensor)
        comment_tensor = comment_tensor[:prevlen] 
        rec_codes.append(all_codes[i])
        comment = np.zeros(len(vocab))
        comment[comments_vector[word]] = 1
        output_tensor.append(comment)
        comment_tensor.append(comment)
        # print(len(comment_tensor))
        for j in range(10-len(comment_tensor)):
            comment_tensor.append(np.zeros(len(vocab)))
        prevlen+=1

all_codes_tensor = set(all_codes_tensor)

for i in range(len(rec_codes)):
    rec_codes[i] = one_hot(rec_codes[i], round(len(all_codes_tensor)*1.3))
    rec_codes[i] = np.pad(rec_codes[i], (0, 1000 - len(rec_codes[i])) ,'constant')

print(np.asarray(rec_codes).shape)

# print(np.asarray(comments_tensor).shape)
        
inputs1 = Input(shape=(1000,))
am1 = Embedding( round(len(all_codes_tensor)*1.3), 128 )(inputs1)
am2 = LSTM(256, return_sequences = True)(am1)
am2 = LSTM(512)(am2)
inputs2 = Input(shape=(10,78))
flat = Flatten()(inputs2)
sm1 = Embedding(len(vocab), 128)(flat)
sm2 = LSTM(256, return_sequences = True)(sm1)
sm2 = LSTM(512)(sm2)
decoder1 = concatenate([am2, sm2])
outputs = Dense(len(vocab), activation='softmax')(decoder1)


# tie it together [article, summary] [word]
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.fit([np.asarray(rec_codes), np.asarray(comments_tensor)], np.asarray(output_tensor), epochs = 2, verbose = 1)

        