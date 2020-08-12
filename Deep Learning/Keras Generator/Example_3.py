# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:26:14 2020

@author: kbhandari
"""

import os
wd = "C:/Users/kbhandari/OneDrive - Epsilon/Desktop/Not Just OLS/Keras Generator/Example_3/"
os.chdir(wd)
import numpy as np
import pandas as pd
import math
import glob

import keras
from keras.models import Sequential
from keras.layers import Dense

train_nrows = 768+764
validation_nrows = 631

#class DataGenerator(keras.utils.Sequence):
#    'Generates data for Keras'
#    def __init__(self, ids, train_dir, nrows, batch_size=32):
#        'Initialization'
#        self.ids = ids
#        self.train_dir = train_dir
#        self.nrows = nrows
#        self.batch_size = batch_size
#        
#    def __len__(self):
#        'Denotes the number of batches per epoch'
##        return len(self.ids)
#        return int(np.ceil(self.nrows / self.batch_size))
#    def __getitem__(self, ids):
#        batch_id = self.ids
#        # load data
#        data = pd.read_csv(self.train_dir + 'data_' + str(batch_id) + '.csv')
#        X = data[[col for col in data.columns if 'Target' not in col]].values
#        Y = data['Target'].values
#            
#        return X, Y


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, train_dir, nrows, batch_size=32, shuffle=True):
        'Initialization'
        self.train_dir = train_dir
        self.nrows = nrows
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nrows / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        df = self.__data_generation()
        print(index)
        # Generate indexes of the batch
        batch = df[index*self.batch_size:(index+1)*self.batch_size]
        Y = batch['Target'].values
        X = batch[[col for col in batch.columns if 'Target' not in col]].values
        
        return X,Y

    def on_epoch_end(self):
        print("training generator: epoch end")
    
    def __data_generation(self):
        'Generates data containing batch_size samples'
        
        csv_files = glob.glob(wd + '/*.csv')
        for file in csv_files:
            df = pd.read_csv(file)
#            df_steps = math.ceil(len(df) / self.batch_size)
#            for i in range(df_steps):
#                batch = df[i * self.batch_size:(i + 1) * self.batch_size]
#                Y = batch['Target'].values
#                X = batch[[col for col in batch.columns if 'Target' not in col]].values
        return df


# Parameters
params = {'train_dir': 'C:/Users/kbhandari/OneDrive - Epsilon/Desktop/Not Just OLS/Keras Generator/Example_3/',
          'nrows': train_nrows,
          'batch_size': 16,
          'shuffle': False}
        
#params = {'train_dir': 'C:/Users/kbhandari/OneDrive - Epsilon/Desktop/Not Just OLS/Keras Generator/Example_3/'}

# Datasets
partition = {'train': ['data_1', 'data_2'], 'validation': ['data_3']}

# Generators
training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)

batch_size = 32

#training_generator = DataGenerator(1, wd, train_nrows, batch_size)
#validation_generator = DataGenerator(3, wd, validation_nrows, batch_size)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train model on dataset
#history = model.fit_generator(generator=training_generator,
#                    validation_data=validation_generator,
#                    epochs=10,
#                    verbose=1,
#                    shuffle = True)

# Train model on dataset
history = model.fit_generator(generator=training_generator,
                    epochs=10,
                    verbose=1)


#history = model.fit_generator(generator=training_generator,
#                    validation_data=validation_generator,
#                    epochs=10,
#                    use_multiprocessing=True,
#                    workers=2,
#                    verbose=1)