from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

import numpy as np
def vectorize_sequences(sequences, dim = 10000):
  results = np.zeros((len(sequences), dim))
  for i, seq in enumerate(sequences):
    results[i, seq] = 1.
  return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

from keras.utils.np_utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=512, validation_split=0.4, verbose=0)

import utils
utils.plot(history.history)


results = model.evaluate(x_test, y_test, verbose=0)
print(results)

predictions = model.predict(x_test)
print(predictions[0])
print(np.argmax(predictions[0]))