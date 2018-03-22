from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# normalize data
import numpy as np
mean = np.mean(train_data)
train_data -= mean
std = train_data.std()
train_data /= std
test_data -= mean
test_data /= std

from keras import models
from keras import layers

def build_model():
  model = models.Sequential()
  model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(1))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
  return model


#k-fold
k=4
num_val_samples = len(train_data) // k
num_epochs = 10
all_mae_histories = []
for i in range(k):
  print('processing fold #', i)
  x_val = train_data[i*num_val_samples:(i+1)*num_val_samples]
  y_val = train_targets[i*num_val_samples:(i+1)*num_val_samples]
  x_train = np.concatenate([train_data[:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis=0)
  y_train = np.concatenate([train_targets[:i*num_val_samples], train_targets[(i+1)*num_val_samples:]], axis=0)

  model = build_model()
  history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(x_val, y_val))
  mae_history = history.history['val_mean_absolute_error']
  all_mae_histories.append(mae_history)  

average_mae_history = [
np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()