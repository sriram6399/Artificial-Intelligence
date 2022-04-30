import keras as kr
from tensorflow.keras import optimizers
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.models import Model
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from random import shuffle
from tqdm import tqdm
from _ctypes import sizeof
from random import randint
import time
import os
from tensorflow.keras.datasets import cifar100

def get_error(y,yh):
    # Threshold 
    yht = np.zeros(np.shape(yh))
    yht[np.arange(len(yh)), yh.argmax(1)] = 1
    # Evaluate Error
    error = np.count_nonzero(np.count_nonzero(y-yht,1))/len(y)
    return error


def one_hot(y):
    n_values = np.max(y) + 1
    y_new = np.eye(n_values)[y[:,0]]
    return y_new


def fine_model():
    net = Conv2D(1024, 1, strides=1, padding='same', activation='elu')(model.layers[-8].output)
    net = Conv2D(1152, 2, strides=1, padding='same', activation='elu')(net)
    net = Dropout(.6)(net)
    net = MaxPooling2D((2, 2), padding='same')(net)

    net = Flatten()(net)
    net = Dense(1152, activation='elu')(net)
    out_fine = Dense(100, activation='softmax')(net)
    model_fine = Model(inputs=in_layer,outputs=out_fine)
    model_fine.compile(optimizer= sgd_coarse,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    for i in range(len(model_fine.layers)-1):
        model_fine.layers[i].set_weights(model.layers[i].get_weights())
    return model_fine


def eval_hdcnn(X, y):
    yh = np.zeros(np.shape(y))
    
    yh_s = model.predict(X, batch_size=batch)
    
    print('Single Classifier Error: '+str(get_error(y,yh_s)))
    
    yh_c = model_c.predict(X, batch_size=batch)
    y_c = np.dot(y,fine2coarse)
    
    print('Coarse Classifier Error: '+str(get_error(y_c,yh_c)))

    for i in range(coarse_categories):    
        fine_models['yhf'][i] = fine_models['models'][i].predict(X, batch_size=batch)
        yh += np.multiply(yh_c[:,i].reshape((len(y)),1), fine_models['yhf'][i])
    
    print('Overall Error: '+str(get_error(y,yh)))
    return yh


# The number of coarse categories
coarse_categories = 20

# The number of fine categories
fine_categories = 100

IMG_SIZE = 32

(X, y_c), (x_test, y_c_test) = cifar100.load_data(label_mode='coarse')
(X, y), (x_test, y_test) = cifar100.load_data(label_mode='fine')

fine2coarse = np.zeros((fine_categories,coarse_categories))
for i in range(coarse_categories):
    index = np.where(y_c_test[:,0] == i)[0]
    fine_cat = np.unique([y_test[j,0] for j in index])
    for j in fine_cat:
        fine2coarse[j,i] = 1

             
y_c = 0; # Clear y_c in interest of saving mem
y_c_test=0;



y=one_hot(y)
y_test=one_hot(y_test)



x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=.1, random_state=0)
X = 0
y = 0

tf.compat.v1.reset_default_graph()

in_layer = Input(shape=(32, 32, 3), dtype='float32', name='main_input')

net = Conv2D(384, 3, strides=1, padding='same', activation='elu')(in_layer)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(384, 1, strides=1, padding='same', activation='elu')(net)
net = Conv2D(384, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(640, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(640, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.2)(net)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(640, 1, strides=1, padding='same', activation='elu')(net)
net = Conv2D(768, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(768, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(768, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.3)(net)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(768, 1, strides=1, padding='same', activation='elu')(net)
net = Conv2D(896, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(896, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.4)(net)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(896, 3, strides=1, padding='same', activation='elu')(net)
net = Conv2D(1024, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(1024, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.5)(net)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(1024, 1, strides=1, padding='same', activation='elu')(net)
net = Conv2D(1152, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.6)(net)
net = MaxPooling2D((2, 2), padding='same')(net)

net = Flatten()(net)
net = Dense(1152, activation='elu')(net)
net = Dense(100, activation='softmax')(net)

model = Model(inputs=in_layer,outputs=net)
sgd_coarse = optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer= sgd_coarse, loss='categorical_crossentropy', metrics=['accuracy'])


tbCallBack = kr.callbacks.TensorBoard(log_dir='C:/Users/gutti/Documents/AI project/logs2', histogram_freq=0, write_graph=True, write_images=True)
tbCallBack1 = kr.callbacks.TensorBoard(log_dir='C:/Users/gutti/Documents/AI project/logs1', histogram_freq=0, write_graph=True, write_images=True)
batch = 50
index= 0
step = 1
stop = 2

while index < stop:
    model.fit(x_train, y_train, batch_size=batch, initial_epoch=index, epochs=index+step, validation_data=(x_val, y_val), callbacks=[tbCallBack])
    index += step
    model.save_weights('data/models/model_coarse'+str(index))
save_index = index


sgd_fine = optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
for i in range(len(model.layers)):
    model.layers[i].trainable=False
    
y_train_c = np.dot(y_train,fine2coarse)
y_val_c = np.dot(y_val,fine2coarse)

net = Conv2D(1024, 1, strides=1, padding='same', activation='elu')(model.layers[-8].output)
net = Conv2D(1152, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.6)(net)
net = MaxPooling2D((2, 2), padding='same')(net)

net = Flatten()(net)
net = Dense(1152, activation='elu')(net)
out_coarse = Dense(20, activation='softmax')(net)

model_c = Model(inputs=in_layer,outputs=out_coarse)
model_c.compile(optimizer= sgd_coarse, loss='categorical_crossentropy', metrics=['accuracy'])

for i in range(len(model_c.layers)-1):
    model_c.layers[i].set_weights(model.layers[i].get_weights())
    
index = 2
step = 1
stop = 3

while index < stop:
    model_c.fit(x_train, y_train_c, batch_size=batch, initial_epoch=index, epochs=index+step, validation_data=(x_val, y_val_c), callbacks=[tbCallBack])
    index += step
    
model_c.compile(optimizer=sgd_fine, loss='categorical_crossentropy', metrics=['accuracy'])
stop = 4

while index < stop:
    model_c.fit(x_train, y_train_c, batch_size=batch, initial_epoch=index, epochs=index+step, validation_data=(x_val, y_val_c), callbacks=[tbCallBack])
    index += step
    

fine_models = {'models' : [{} for i in range(coarse_categories)], 'yhf' : [{} for i in range(coarse_categories)]}
for i in range(coarse_categories):
    model_i = fine_model()
    fine_models['models'][i] = model_i
    

for i in range(coarse_categories):
    index= 0
    step = 1
    stop = 1
    
    # Get all training data for the coarse category
    ix = np.where([(y_train[:,j]==1) for j in [k for k, e in enumerate(fine2coarse[:,i]) if e != 0]])[1]
    x_tix = x_train[ix]
    y_tix = y_train[ix]
    
    # Get all validation data for the coarse category
    ix_v = np.where([(y_val[:,j]==1) for j in [k for k, e in enumerate(fine2coarse[:,i]) if e != 0]])[1]
    x_vix = x_val[ix_v]
    y_vix = y_val[ix_v]
    
    while index < stop:
        fine_models['models'][i].fit(x_tix, y_tix, batch_size=batch, initial_epoch=index, epochs=index+step, validation_data=(x_vix, y_vix))
        index += step
    
    fine_models['models'][i].compile(optimizer=sgd_fine, loss='categorical_crossentropy', metrics=['accuracy'])
    stop = 10

    while index < stop:
        fine_models['models'][i].fit(x_tix, y_tix, batch_size=batch, initial_epoch=index, epochs=index+step, validation_data=(x_vix, y_vix))
        index += step
        
    yh_f = fine_models['models'][i].predict(x_val[ix_v], batch_size=batch)
    print('Fine Classifier '+str(i)+' Error: '+str(get_error(y_val[ix_v],yh_f)))
    
  

yh = eval_hdcnn(x_val,y_val) 
