import keras
import tensorflow as tf
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, LSTM, Activation
from keras.callbacks import CSVLogger
from keras.optimizers import Adam


data = np.load('oneHotEncoded.npy')
input_data = data[:,:-1,:]
output_data = data[:,1:,:]
batch_size = 32
epochs = 10
# input_data = np.reshape(input_data,(-1, input_data.shape[2]))

print data.shape
model = Sequential()
model.add(LSTM(150, input_shape= (39, 150)))

model.add(Activation('softmax'))

adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy']
              )
model.summary()
model.fit(input_data, output_data,
	epochs = epochs,
    batch_size=batch_size,
	callbacks=[CSVLogger('output.csv', append=False)])

