import os, shutil

# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = './kaggle/train'

# The directory where we will
# store our smaller dataset
base_dir = './kaggle/cats_and_dogs_small'
# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# Directory with our validation cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')  
# Directory with our validation dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')  

def copyFile():
	if os.path.exists(base_dir):
		return

	os.mkdir(base_dir)

	os.mkdir(train_dir)
	os.mkdir(validation_dir)
	os.mkdir(test_dir)

	os.mkdir(train_cats_dir)
	os.mkdir(train_dogs_dir)

	os.mkdir(validation_cats_dir)
	os.mkdir(validation_dogs_dir)

	os.mkdir(test_cats_dir)
	os.mkdir(test_dogs_dir)

	# Copy first 1000 cat images to train_cats_dir
	fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(train_cats_dir, fname)
		shutil.copyfile(src, dst)

	# Copy next 500 cat images to validation_cats_dir
	fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(validation_cats_dir, fname)
		shutil.copyfile(src, dst)

	# Copy next 500 cat images to test_cats_dir
	fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(test_cats_dir, fname)
		shutil.copyfile(src, dst)

	# Copy first 1000 dog images to train_dogs_dir
	fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(train_dogs_dir, fname)
		shutil.copyfile(src, dst)

	# Copy next 500 dog images to validation_dogs_dir
	fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(validation_dogs_dir, fname)
		shutil.copyfile(src, dst)

	# Copy next 500 dog images to test_dogs_dir
	fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(test_dogs_dir, fname)
		shutil.copyfile(src, dst)


copyFile()

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

from keras import optimizers

model.compile(
	loss='binary_crossentropy',
	optimizer=optimizers.RMSprop(lr=1e-4),
	metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
	# This is the target directory
	train_dir,
	# All images will be resized to 150x150
	target_size=(150, 150),
	batch_size=20,
	# Since we use binary_crossentropy loss, we need binary labels
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

for data_batch, labels_batch in train_generator:
	print('data batch shape:', data_batch.shape)
	print('labels batch shape:', labels_batch.shape)
	break

from keras.callbacks import TensorBoard

callbacks=[
	TensorBoard(log_dir='my_log_dir', histogram_freq=1)
]

history = model.fit_generator(
  train_generator,
  steps_per_epoch=100,
  epochs=30,
  validation_data=validation_generator,
  validation_steps=50,
	callbacks = callbacks)

model.save('cats_and_dogs_small_1.h5')

import utils
utils.plot(history.history)
