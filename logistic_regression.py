import pandas as pd
import keras
data = pd.read_csv('./dataset/train.csv')
print(data.head()) # 前五行数据
print(data.info()) # 数据列信息

y = data.Survived  # 返回指定列的Series数据
x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']] # 获取指定列数据
x.Embarked.unique() # 返回DataFrame数据的Embarked列的唯一索引值

# 独热编码
x['Embarked_S'] = (x.Embarked == 'S').astype('int')
x.loc[:, 'Embarked_C'] = (x.Embarked == 'C').astype('int') # loc['a',0] 取第'a'行和第一行
x.loc[:, 'Embarked_Q'] = (x.Embarked == 'Q').astype('int') # loc[:,['a']] 取a列所有行，如没有则新增
del x['Embarked'] # 删除指定列
x['Sex'] = (x.Sex == 'male').astype('int')
print(x.info())

x['Age'] = x.Age.fillna(x.Age.mean()) # 平均数填充空值
x['p1'] = (x.Pclass == 1).astype('int')
x['p2'] = (x.Pclass == 2).astype('int')
x['p3'] = (x.Pclass == 3).astype('int')
del x['Pclass']
print("x.shape为：",x.shape, "y.shape为",y.shape)

# 定义模型
model = keras.Sequential()
from keras import layers
"""
激活函数：每个神经网络层都需要一个激活函数
    sigmoid
    tanh
    relu
"""
# 输入11列个特征x1...xn->1个输出y_pre
model.add(layers.Dense(1, input_dim=11, activation='sigmoid'))     #y_pre = (w1*x1 + w2*x2 + ... + w11*x11 + b)
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
history = model.fit(x, y, epochs=300)

import matplotlib.pyplot as plt
plt.plot(range(300), history.history.get('loss'))
plt.plot(range(300), history.history.get('acc'))
plt.show()

