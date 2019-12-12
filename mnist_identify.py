import keras
from keras import layers
import matplotlib.pyplot as plt
import keras.datasets.mnist as mnist

# https://s3.amazonaws.com/img-datasets/mnist.npz下载到C:\Users\codewj\.keras\datasets\mnist.npz
(train_image, train_label), (test_image, test_label) = mnist.load_data()
print(train_image.shape,train_label[1000]) # (60000, 28, 28) 0
plt.imshow(train_image[1000])
plt.show()
print(test_image.shape, test_label.shape,train_label) # (10000, 28, 28) (10000,) [5 0 4 ... 5 6 8]

model = keras.Sequential()
model.add(layers.Flatten())     # (60000, 28, 28) --->  (60000, 28*28)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)
model.fit(train_image, train_label, epochs=50, batch_size=512)
print(model.evaluate(test_image, test_label))
print(model.evaluate(train_image, train_label))


import numpy as np
print(np.argmax(model.predict(test_image[:10]), axis=1)) # test_image为10000 28*28取前10条数据，预测得ndarray(10,10)的one-hot结果,axis 1行0列，取每行最大值索引位置返回ndarray列表
print(test_label[:10])


# 优化模型
model = keras.Sequential()
model.add(layers.Flatten())     # (60000, 28, 28) --->  (60000, 28*28)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)
model.fit(train_image, train_label, epochs=50, batch_size=512, validation_data=(test_image, test_label))


# 再优化
model = keras.Sequential()
model.add(layers.Flatten())     # (60000, 28, 28) --->  (60000, 28*28)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)
model.fit(train_image, train_label, epochs=200, batch_size=512, validation_data=(test_image, test_label))