import tensorflow as tf
import keras
from keras import layers
import keras.datasets.mnist as mnist

(train_image, train_label), (test_image, test_label) = mnist.load_data()
print(train_image.shape,test_image.shape)

import matplotlib.pyplot as plt
plt.imshow(train_image[4])
plt.show()
print(train_label[4])

import numpy as np
train_image = np.expand_dims(train_image, axis=-1)
test_image = np.expand_dims(test_image, axis=-1)
print(train_image.shape,train_label.shape,test_image.shape)


model = keras.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv_1'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', name='conv_2'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', name='dense_1'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax', name='dense_2'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_image, train_label, epochs=3, batch_size=512)
print(model.evaluate(test_image, test_label))
model.save_weights('model/my_model_weights.h5')