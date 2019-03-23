from keras.models import load_model

encoder_model = load_model("enc.h5")
decoder_model = load_model("dec.h5")

import numpy as np
from nltk.tokenize import word_tokenize


import pickle

code_tensors = []
codes = []
code_vocab = set()

code = open("code.txt","r+").read()
codes.append(word_tokenize(code))
for word in (word_tokenize(code)):
    code_vocab.add(word)

code_dict = pickle.load(open("code_dict.pickle", "rb+"))

print(code_dict)

for item in codes:
    code_tensor = []
    code_word = np.zeros(len(code_vocab)+4)
    code_word[1] = 1
    code_tensor.append(code_word)
    for i in range(750):
        code_word = np.zeros(len(code_vocab)+4)
        if(i < len(item)):
            code_word[code_dict[item[i]]+4] = 1
        elif(i == len(item)):
            code_word[2] = 1
        else:
            code_word[0] = 1
        code_tensor.append(code_word)
    code_tensors.append(code_tensor)




comments_reverse_map = pickle.load(open("comments_reverse_map.pickle", "rb+"))

for i in range(1):
    states_value = encoder_model.predict(np.asarray([code_tensors[i]]))
    # print(states_value)

    target_seq = np.zeros((1, 1, len(comments_reverse_map) + 3))

    target_seq[0, 0, 1] = 1

    stop_condition = False
    i = 0
    decoded_sentence = ''
    print('\n')
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        target_seq = np.zeros((1, 1, len(comments_reverse_map) + 3))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]
        # print(sampled_token_index)
        i += 1
        if (i == 10 or sampled_token_index == 2):
            break

        print(comments_reverse_map[sampled_token_index - 3], end=' ')
        if (i == 10):
            stop_condition = True

    print('\n')
