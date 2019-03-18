from keras.models import load_model

encoder_model = load_model("enc.h5")
decoder_model = load_model("dec.h5")

import numpy as np

import pickle

code_tensors = []


def convert_to_onehot(c):
    tensor = np.zeros(len(code_dict)+3)
    return tensor


code = open('new_code.txt', 'r+').read()
code_tensor = convert_to_onehot(code)
code_tensor = list(map(convert_to_onehot, code_tensor))
code_tensors.append(code_tensor)

code_tensors = pickle.load(open("code_tensors.pickle", "rb+"))
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
