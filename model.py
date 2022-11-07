import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image

def display_digit(X):
    fig, ax = plt.subplots(1,1, figsize=(0.5,0.5))
    ax.imshow(X, cmap='gray')
    plt.show()

(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

display_digit(x_train[1015])

#visualize the dataset

m, n, x, y = x_train.shape

fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) 

for i,ax in enumerate(axes.flat):
    
    random_index = np.random.randint(m)

    ax.imshow(x_train[random_index], cmap='gray')
    
    ax.set_title(y_train[random_index])
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize=14)
    
input_shape = (28,28,1)
batch_size = 128
num_classes = 10
epochs = 50

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train.shape : ',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
hist = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
print('The model has succesfully trained')

model.summary()

score = model.evaluate(x_test,y_test,verbose=0)
print('Test loss: ',score[0])
print('Test accuracy: ',score[1])

model.save('DigitRecognizer')
print("Saving the model as DigitRecognizer")


#checking some predictions
m, n, x, y = x_train.shape

fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91])

for i,ax in enumerate(axes.flat):

    random_index = np.random.randint(m)
    
    ax.imshow(x_train[random_index], cmap='gray')
    
    prediction = model.predict(x_train[random_index].reshape(1,28,28,1),verbose=0)
    prediction_p = tf.nn.softmax(prediction)
    y_pred = np.argmax(prediction_p)
    
    ax.set_title(f"{np.argmax(y_train[random_index]},{y_pred}",fontsize=10)
    ax.set_axis_off()
fig.suptitle("Label, y_pred", fontsize=14)
plt.show()
