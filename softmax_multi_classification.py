import keras
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./dataset/iris.csv')
print(data.head())
print(data.info())

print(data.Species.unique())
data = data.join(pd.get_dummies(data.Species)) # 将Species以unique后分为n列，转为one-hot矩阵拼接
del data['Species']
index = np.random.permutation(len(data))  # permutation随机打乱数据
data = data.iloc[index] # 根据打乱后顺序生成数据

x = data[data.columns[1: -3]]  # 获取2~-3列数据作为x,150*4
y = data.iloc[:, -3:] # 获取所有行，-3列~最后列数据组成的矩阵，150*3
print(x.shape, y.shape)

# 定义模型
model = keras.Sequential()
# 输入4维特征，输出3维one-hot
model.add(layers.Dense(3, input_dim=4, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc']
)      #  目标数据做独热编码，用 categorical_crossentropy 来计算softmax交叉熵
history = model.fit(x, y, epochs=50)

# 不用one-hot
data = pd.read_csv('./dataset/iris.csv')
print(data.Species.unique())
spc_dic = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
data['Species'] = data.Species.map(spc_dic)
x = data[data.columns[1:-1]]
y = data.Species

model = keras.Sequential()
model.add(layers.Dense(3, input_dim=4, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
)      #  目标数据做顺序编码，用 sparse_categorical_crossentropy 来计算softmax交叉熵
model.fit(x, y, epochs=50)