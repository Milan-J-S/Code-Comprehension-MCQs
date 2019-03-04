#import modules
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import os
import numpy as np 

def convert_to_onehot( c ):
    tensor = np.zeros(128)
    tensor[ord(c)] = 1
    return tensor

code_tensors = []
vocab = set()
comments_dict = {}
comments = []


file_path = 'code'
for root,dirs,files in os.walk(file_path):
    for fl in files:
        code = open('code/'+fl).read()
        code_tensor = [x for x in code][:1000]
        for i in range(1000 - len(code_tensor)):
            code_tensor.append(' ')
        code_tensor = list(map(convert_to_onehot ,code_tensor))
        code_tensors.append(code_tensor)
        comment = open('comments/'+fl).read()
        for com in comment.split():
            vocab.add(com)
        comments.append(comment)

i=0
for item in vocab:
    comments_dict[item]=i
    i=i+1

comment_tensors_input = []
comment_tensors_target = []

for item in comments:
    comment_tensors = []
    comment = item.split()
    for i in range(10):
        comment_tensor = np.zeros(len(vocab)+1)
        if(i < len(comment)):
            comment_tensor[comments_dict[comment[i]]+1] = 1
        else:
            comment_tensor[0] = 1
        comment_tensors.append(comment_tensor)
    comment_tensors_input.append(comment_tensors)
    comment_targets = comment_tensors[1:]
    zero_vec = np.zeros(len(vocab)+1)
    zero_vec[0] = 1 
    comment_targets.append(zero_vec)
    print(len(comment_targets))
    comment_tensors_target.append(comment_targets)


        
        


print(np.asarray(code_tensors).shape)
print(np.asarray(comment_tensors_input).shape)
print(np.asarray(comment_tensors_target).shape)


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, 128))
encoder = LSTM(500, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, len(vocab)+1))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the 
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(500, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(len(vocab)+1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])



encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(500,))
decoder_state_input_c = Input(shape=(500,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

decoder_model.save("dec.py")
encoder_model.save("enc.py")

print(model.summary())
print(encoder_model.summary())
print(decoder_model.summary())

model.fit([np.asarray(code_tensors), np.asarray(comment_tensors_input)], np.asarray(comment_tensors_target),
          batch_size=5,
          epochs=100,
          validation_split=0.2)

model.save("encdec.h5")


# from keras.models import load_model

# model = load_model("encdec.h5")
# prediction = model.predict([np.asarray(code_tensors)[0]])
# print(prediction)
