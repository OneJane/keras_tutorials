from tensorflow import keras
from tensorflow.keras import layers

# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz下载解压到C:\Users\codewj\.keras\datasets\cifar-10-batches-py
cifar = keras.datasets.cifar10 # 10分类
(train_image, train_label), (test_image, test_label) = cifar.load_data()
print(train_image.shape, test_image.shape,) # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(train_label)

# 归一化处理
train_image = train_image/255
test_image = test_image/255

model = keras.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
"""
BatchNormalization:
    将该层特征值分布重新拉回标准正态分布，特征值将落在激活函数对于输入较为敏感的区间，输入的小变化可导致损失函数较大的变化，使得梯度变大，避免梯度消失，同时也可加快收敛。
    需要计算均值与方差，不适合动态网络或者RNN。计算均值方差依赖每批次，因此数据最好足够打乱
"""

model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (1, 1), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(128))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)
history = model.fit(train_image, train_label, epochs=30, batch_size=128)
print(model.evaluate(test_image, test_label))
print(model.evaluate(test_image, test_label))