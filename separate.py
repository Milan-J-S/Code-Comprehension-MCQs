import keras
from keras.layers import Input
from keras.datasets import imdb
from keras.models import Model
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
from legacy import AttentionDecoder


from nltk.tokenize import word_tokenize
import os

file_path = 'code'

codes = []
comments = []
vocab = set()
comments_vector = {}
all_codes_tensor = []

for root,dirs,files in os.walk(file_path):
    for fl in files:
        code = open('code/'+fl).read()
        code_tensor =  np.asarray(word_tokenize(code)[:1000])
        all_codes_tensor.extend(code_tensor)
        # print(len(code_tensor), "--->")
        if(len(code_tensor) < 1000):
            code_tensor = np.pad(code_tensor, (0, 1000 - len(code_tensor)) , 'constant' )
        print(len(code_tensor))
        codes.append(code)
        comment = open('comments/'+fl).read()
        for c in comment.split():
            vocab.add(c)
        comments.append(comment)





i=0

for item in vocab:
    comments_vector[item]=i
    i=i+1

comments_tensor1 = []
comments_tensor2 = []
comments_tensor3 = []
comments_tensor4 = []
comments_tensor5 = []
comments_tensor6 = []
comments_tensor7 = []
comments_tensor8 = []
comments_tensor9 = []
comments_tensor10 = []

tensor = [comments_tensor1,comments_tensor2,comments_tensor3,comments_tensor4,comments_tensor5,comments_tensor6,comments_tensor7,comments_tensor8,comments_tensor9,comments_tensor10]

# for item in comments:
#     comment_tensor = []
#     for word in item.split():
#         comment = np.zeros(len(vocab))
#         comment[comments_vector[word]] = 1
#         comment_tensor.append(comment)
#     print(len(comment_tensor)) 
#     for i in range(10-len(comment_tensor)):
#            comment_tensor.append(np.zeros(len(vocab)))
#     comments_tensor.append(comment_tensor)
for item in comments:
    comment = item.split()
    for i in range(10):
        comment_tensor = np.zeros(len(vocab))
        if(i < len(comment)):
            comment_tensor[comments_vector[comment[i]]] = 1
        tensor[i].append(comment_tensor)

tensor = [np.asarray(comments_tensor1),np.asarray(comments_tensor2),np.asarray(comments_tensor3),np.asarray(comments_tensor4),np.asarray(comments_tensor5),np.asarray(comments_tensor6),np.asarray(comments_tensor7),np.asarray(comments_tensor8),np.asarray(comments_tensor9),np.asarray(comments_tensor10)]
            
print(np.asarray(comments_tensor1).shape)


for i in range(len(codes)):
    codes[i] = one_hot(codes[i], round(len(all_codes_tensor)*1.3))[:1000]
    codes[i] = np.pad(codes[i], (0, 1000 - len(codes[i])), 'constant')
    
print(np.asarray(codes).shape)

inp = Input(shape=(1000,))
hidden = Embedding(round(len(all_codes_tensor)*1.3), 128 )(inp)
hidden = LSTM(512, return_sequences = True )(hidden)
hidden = Dropout(0.5)(hidden)
hidden = LSTM(256 )(hidden)
hidden = Dropout(0.5)(hidden)
out1 = Dense(len(vocab))(hidden)
out2 = Dense(len(vocab))(hidden)
out3 = Dense(len(vocab))(hidden)
out4 = Dense(len(vocab))(hidden)
out5 = Dense(len(vocab))(hidden)
out6 = Dense(len(vocab))(hidden)
out7 = Dense(len(vocab))(hidden)
out8 = Dense(len(vocab))(hidden)
out9 = Dense(len(vocab))(hidden)
out10 = Dense(len(vocab))(hidden)

model = Model(inputs = inp, outputs = [out1,out2,out3,out4,out5,out6,out7,out8,out9,out10])


model.compile(loss = 'categorical_crossentropy', optimizer='RMSProp', metrics = ['accuracy'])

print(model.summary())

model.fit(np.asarray(codes), tensor, epochs=2, verbose = 1)








