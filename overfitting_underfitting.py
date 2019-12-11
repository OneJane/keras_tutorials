import keras
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./dataset/credit-a.csv', header=None)
print(data.iloc[:, -1].unique())
x = data.iloc[:, :-1].values  # 取到最后一列
y = data.iloc[: , -1].replace(-1, 0).values.reshape(-1, 1) # 取最后一列
print(y.shape, x.shape) # (653, 1) (653, 15)

model = keras.Sequential()
model.add(layers.Dense(128, input_dim=15, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
"""
binary_cross_entropy是二分类的交叉熵，实际是多分类softmax_cross_entropy的一种特殊情况
Adam:同时获得了 AdaGrad 和 RMSProp 算法的优点
    适应性梯度算法（AdaGrad）为每一个参数保留一个学习率以提升在稀疏梯度
    均方根传播（RMSProp）基于权重梯度最近量级的均值为每一个参数适应性地保留学习率
"""
history = model.fit(x, y, epochs=1000)
plt.plot(history.epoch, history.history.get('loss'), c='r')
plt.plot(history.epoch, history.history.get('acc'), c='b')
plt.show()


# 判断拟合评价标准： 对未见过数据的预测
x_train = x[:int(len(x)*0.75)]
x_test = x[int(len(x)*0.75):]
y_train = y[:int(len(x)*0.75)]
y_test = y[int(len(x)*0.75):]
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
model = keras.Sequential()
model.add(layers.Dense(128, input_dim=15, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))
plt.plot(history.epoch, history.history.get('val_acc'), c='r', label='val_acc')
plt.plot(history.epoch, history.history.get('acc'), c='b', label='acc')
plt.legend() # 给图加上图例
plt.show()

print(model.evaluate(x_train, y_train)) # evaluate评估模型返回loss,accuracy
print(model.evaluate(x_test, y_test))

y_pred = model.predict(x_test,batch_size = 1) # 模型预测,输入测试集,输出预测结果
# 过拟合：在训练数据正确率非常高， 在测试数据上比较低

# 随机断开百分比神经元连接
model = keras.Sequential()
model.add(layers.Dense(128, input_dim=15, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))
print(model.evaluate(x_train, y_train)) # evaluate评估模型返回loss,accuracy
print(model.evaluate(x_test, y_test))
plt.plot(history.epoch, history.history.get('val_acc'), c='r', label='val_acc')
plt.plot(history.epoch, history.history.get('acc'), c='b', label='acc')
plt.legend()
plt.show()

# 正则化 L1 L2
from keras import regularizers
model = keras.Sequential()
model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), input_dim=15, activation='relu'))
model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))
print(model.evaluate(x_train, y_train)) # evaluate评估模型返回loss,accuracy
print(model.evaluate(x_test, y_test))
plt.plot(history.epoch, history.history.get('val_acc'), c='r', label='val_acc')
plt.plot(history.epoch, history.history.get('acc'), c='b', label='acc')
plt.legend()
plt.show()


model = keras.Sequential()
model.add(layers.Dense(4, input_dim=15, activation='relu'))  # 修改隐藏层神经元数
model.add(layers.Dense(4,  activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))
print(model.evaluate(x_train, y_train)) # evaluate评估模型返回loss,accuracy
print(model.evaluate(x_test, y_test))
plt.plot(history.epoch, history.history.get('val_acc'), c='r', label='val_acc')
plt.plot(history.epoch, history.history.get('acc'), c='b', label='acc')
plt.legend()
plt.show()


model = keras.Sequential()
model.add(layers.Dense(4, input_dim=15, activation='relu'))
model.add(layers.Dense(1,  activation='relu'))  # 修改隐藏层神经元数
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
history = model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))
plt.plot(history.epoch, history.history.get('val_acc'), c='r', label='val_acc')
plt.plot(history.epoch, history.history.get('acc'), c='b', label='acc')
plt.legend()
plt.show()