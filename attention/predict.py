# my imports
import re
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import Sequential
import os
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

import pickle

from legacy import AttentionDecoder

from keras.models import load_model



def convert_to_onehot(c):
    tensor = np.zeros(128)
    tensor[ord(c)] = 1
    return tensor


code_tensors = []
codes = []
comments = []
comments_reverse_map = []
vocab = set()
code_vocab = set()
comments_dict = {}
code_dict = {}

file_path = 'code'
for root, dirs, files in os.walk(file_path):
    for fl in files:
        code = open('code/' + fl).read()
        code = re.sub("\"[^\"]*\"", "0", code)
        code = re.sub("name: [^,}]+", "name", code)
        code = re.sub("value: [^,}]+", "value", code)
        print(code)

        codes.append(word_tokenize(code))
        for word in (word_tokenize(code)):
            code_vocab.add(word)


i = 0

for item in code_vocab:
    code_dict[item] = i
    i = i + 1

print(len(code_vocab))



for item in codes:
    code_tensor = np.zeros(751)
    for i in range(min(len(item), 751)):
        code_tensor[i] = code_dict[item[i]] / 107
    code_tensors.append(code_tensor)

# print(code_tensors[3])

print(np.asarray(code_tensors).shape)

automodel = load_model("autoencoder.h5")

attnmodel = load_model("attention.h5", custom_objects={'AttentionDecoder': AttentionDecoder})

code_dict_p = pickle.load(open("code_dict.pickle", "rb+"))
print(code_dict_p)

comment_tensors_p = pickle.load(open("code_dict.pickle", "rb+"))
print(comment_tensors_p)

comments_reverse_map_p = pickle.load(open("comments_reverse_map.pickle", "rb+"))
print(comments_reverse_map_p)

code_tensors = []

code = open("file.txt").read()
code = re.sub("\"[^\"]*\"", "0", code)
code = re.sub("name: [^,}]+", "name", code)
code = re.sub("value: [^,}]+", "value", code)

code_tensor = np.zeros(751)
item = word_tokenize(code)
for i in range(min(len(item), 751)):
    code_tensor[i] = code_dict[item[i]] / 107
code_tensors.append(code_tensor)

result = automodel.predict(np.asarray(code_tensors))

print(result)
print(result.shape)
