# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:09:35 2020

@author: kbhandari
"""

import os
wd = "C:/Users/kbhandari/OneDrive - Epsilon/Desktop/Not Just OLS/Keras Generator/"
os.chdir(wd)
import numpy as np
import pandas as pd
import math

import keras
from keras.models import Sequential
from keras.layers import Dense

train_nrows = len(pd.read_csv('Data.csv'))
val_nrows = len(pd.read_csv('Validation.csv'))

class MyGenerator(): #keras.utils.Sequence
    'Generates data for Keras'
    def __init__(self, train_dir, nrows, chunksize = 100, batch_size=32):
        'Initialization'
        self.train_dir = train_dir
        self.chunksize = chunksize
        self.batch_size = batch_size
        self.nrows = nrows
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nrows / self.batch_size))

    def __getitem__(self):
        while True:
            # load data
            data = pd.read_csv(self.train_dir, chunksize=self.chunksize)
            for df_chunk in data:
                chunk_steps = math.ceil(len(df_chunk) / self.batch_size)
                for i in range(chunk_steps):
                    batch = df_chunk[i * self.batch_size:(i + 1) * self.batch_size]
                    print(batch.index)
                    Y = batch['Target'].values
                    X = batch[[col for col in batch.columns if 'Target' not in col]].values                    
                    yield X, Y

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# fit the keras model on the dataset
#model.fit(X, y, epochs=150, batch_size=10)

batch_size = 32
train_chunksize = 32*3
val_chunksize = 50

trainGen = MyGenerator('Data.csv', train_nrows, train_chunksize, batch_size)
nextTrainGen = trainGen.__getitem__()

valGen = MyGenerator('Validation.csv', val_nrows, val_chunksize, batch_size)
nextValGen = valGen.__getitem__()

history = model.fit_generator(nextTrainGen, validation_data = nextValGen, epochs=10, steps_per_epoch=train_nrows//batch_size, validation_steps=val_nrows//batch_size)

trainGen.count
valGen.count





history = model.fit_generator(nextTrainGen, epochs=1, steps_per_epoch=train_nrows//batch_size)



