import pandas as pd
import keras
import numpy as np

data = pd.read_csv('./dataset/train.csv')
print(data.head(),data.info())
y = data.Survived
x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
print(x.Embarked.unique())

# 新增字段独热编码
x.loc[:, 'Embarked_S'] = (x.Embarked == 'S').astype('int')
x.loc[:, 'Embarked_C'] = (x.Embarked == 'C').astype('int')
x.loc[:, 'Embarked_Q'] = (x.Embarked == 'Q').astype('int')

# 删除字段
del x['Embarked']

x.loc[:, 'Sex'] = (x.Sex == 'male').astype('int')
x.loc[:, 'Age'] = x.Age.fillna(x.Age.mean())
x.loc[:, 'p1'] = (x.Pclass == 1).astype('int')
x.loc[:, 'p2'] = (x.Pclass == 2).astype('int')
x.loc[:, 'p3'] = (x.Pclass == 3).astype('int')
del x['Pclass']
print(x.info())
print(x.shape, y.shape) # (891, 11) (891,)


model = keras.Sequential()
from keras import layers
# 输入11维特征，输出32维 11-32-32-1，其中32 32为隐藏层神经元数量
model.add(layers.Dense(32, input_dim=11, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
model.fit(x, y, epochs=300)