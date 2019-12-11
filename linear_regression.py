import keras
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 100, 30)  # 0~100 30份array数据
y = 3 * x + 7 + np.random.randn(30) * 6  # np.random.randn(30)1行30列随机标准正态分布数
plt.scatter(x, y)
plt.show()  # 绘图

model = keras.Sequential()  # 顺序堆叠模型
from keras import layers
"""
Dense:指定输入数据形状 全连接层，bias为偏置向量，只有当use_bias=True才会添加，units：大于0的整数，代表该层的输出维度
    输入形如(batch_size, ..., input_dim)的nD张量，最常见的情况为(batch_size, input_dim)的2D张量。
    输出形如(batch_size, ..., units)的nD张量，最常见的情况为(batch_size, units)的2D张量。
Flatten层:输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡
dropout层:为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开一定百分比（p）的输入神经元连接，Dropout层用于防止过拟合
Conv1D层:一维卷积层（即时域卷积），用以在一维输入信号上进行邻域滤波。作为首层时，需要提供关键字参数input_shape.
    输入shape形如（samples，steps，input_dim）的3D张量。
    输出shape形如（samples，new_steps，nb_filter）的3D张量，因为有向量填充的原因，steps的值会改变。
Conv2D层:二维卷积层，即对图像的空域卷积。该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数,
    例如input_shape = (128,128,3)代表128*128的彩色RGB图像
Conv3D层:三维卷积对三维的输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数。
    例如input_shape = (3,10,128,128)代表对10帧128*128的彩色RGB图像进行卷积
MaxPooling1D层:对时域1D信号进行最大值池化
MaxPooling2D层:为空域信号施加最大值池化
AveragePooling1D层:对时域1D信号进行平均值池化
AveragePooling2D层:为空域信号施加平均值池化
循环层（Recurrent）:LSTM、GRU和SimpleRNN。
    SimpleRNN:全连接RNN网络，RNN的输出会被回馈到输入
    LSTM:长短期记忆模型
    GRU:门限循环单元
Embedding层:只能作为模型第一层
"""
model.add(layers.Dense(1, input_dim=1))
model.summary()  # 打印流程

"""
优化器：调整每个节点权重
    SGD随机梯度下降
    RMSprop
    Adam
    Adagrad
损失函数/目标函数: 计算神经网络的输出与样本标记的差  
    mse均方误差
    categorical_crossentropy
"""
model.compile(optimizer='adam', loss='mse')  # 编译模型使用adam优化器,mse均方误差损失作为损失函数

model.fit(x, y, epochs=3000)  # 开始训练模型3000轮

model.predict(x)  # 预测模型返回对应x得到的y值

plt.scatter(x, y, c='r')  # r红色
plt.plot(x, model.predict(x))
plt.show()  # 在原图绘制
model.predict([150])  # 指定x预测
