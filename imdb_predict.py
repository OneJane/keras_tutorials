import keras
from keras import layers
import matplotlib.pyplot as plt

# https://s3.amazonaws.com/text-datasets/imdb.npz下到C:\Users\codewj\.keras\datasets
data = keras.datasets.imdb
max_word = 10000 # 只保留训练集中最常出现的前10000个词，不经常出现的单词被抛弃，最终所有评论的维度保持相同。
# pip install numpy==1.16.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
(x_train, y_train), (x_test, y_test) = data.load_data(num_words=max_word) # x电影评论转换成了一系列数字，每个数字代表字典中的一个单词，y标识积极消极评论类型
print(x_train.shape, y_train.shape)  # 都为list,1行25000列,每列是list

word_index = data.get_word_index()
index_word = dict((value, key) for key, value in word_index.items())
print([index_word.get(index - 3, '?') for index in x_train[0]])  # dict.get(key, default=None)
print(max([max(seq) for seq in x_train]), [len(seq) for seq in x_train])

# 文本向量化
import numpy as np
def k_hot(seqs, dim=10000):  # list,int
    result = np.zeros((len(seqs), dim)) # ndarray大小250000000，25000行，10000列作为词
    for i, seq in enumerate(seqs):
        result[i, seq] = 1
    return result
x_train = k_hot(x_train)  # 25000列句子，用1/0表示句子的单词
x_test = k_hot(x_test)
print(x_train.shape,y_train.shape)  # (25000, 10000) (25000,)

model = keras.Sequential()
model.add(layers.Dense(32, input_dim=10000, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
history = model.fit(x_train, y_train, epochs=15, batch_size=256, validation_data=(x_test, y_test))
plt.plot(history.epoch, history.history.get('loss'), c='r', label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), c='b', label='val_loss')
plt.legend()
plt.show()
plt.plot(history.epoch, history.history.get('acc'), c='r', label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), c='b', label='val_acc')
plt.legend()
plt.show()