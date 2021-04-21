import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers,models
import sys
import numpy as np
from tensorflow import keras
fs=keras.datasets.fashion_mnist
(train_img,train_lab),(test_img,test_lab)=fs.load_data()
name=['衣服','裤子','毛衣','裙子','外套','女鞋','女装','运动鞋','袋子','鞋子']
loca=0
k=55
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_img[i+k], cmap=plt.cm.binary)
    # 由于 CIFAR 的标签是 array， 
    # 因此您需要额外的索引（index）。
    plt.xlabel(train_lab[i+k])
plt.show()
train_img= np.expand_dims(train_img, axis=-1)
train_lab= np.expand_dims(train_lab, axis=-1)
test_img= np.expand_dims(train_img, axis=-1)
test_lab= np.expand_dims(train_lab, axis=-1)
print(tf.shape(train_img))
model = models.Sequential()
def build_discriminator():
    model.add(layers.Conv2D(32, (2, 2), activation='tanh', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
build_discriminator()
print(model.summary())
model.fit(train_img, train_lab, epochs=2, validation_data=(test_img, test_lab))
