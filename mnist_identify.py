import keras
from keras import layers
import matplotlib.pyplot as plt
import keras.datasets.mnist as mnist

# https://s3.amazonaws.com/img-datasets/mnist.npz下载到C:\Users\codewj\.keras\datasets\mnist.npz
(train_image, train_label), (test_image, test_label) = mnist.load_data()
print(train_image.shape) # (60000, 28, 28)
plt.imshow(train_image[1000])
