from keras.models import Sequential
from  keras.layers import LSTM

from legacy import AttentionDecoder

model = Sequential()

model.add(LSTM(150, input_shape=(20,1), return_sequences = True))
model.add(AttentionDecoder(150, 3))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])

config = model.get_config()
print(config)
# model.fit(np.asarray(code_tensors_rep), np.asarray(comment_tensors_target), verbose = 1, epochs = 100, validation_split = 0.2 )

model.save("test.h5")

from keras.models import load_model

model = load_model("test.h5", custom_objects={'AttentionDecoder': AttentionDecoder})