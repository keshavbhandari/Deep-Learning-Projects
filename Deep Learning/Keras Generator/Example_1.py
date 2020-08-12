# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:27:01 2020

@author: kbhandari
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import math
import numpy as np
from keras.models import Model
from keras.utils import Sequence
from keras.layers import Input, Dense, LSTM


class Generator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(self, x_set, y_set, batch_size=256):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]  # Line A
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# dummy model
input_1 = Input(shape=(None, 10))
x = LSTM(90)(input_1)
x = Dense(10)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(input_1, x)
print(model.summary())

# Compile and fit_generator
model.compile(optimizer='adam', loss='binary_crossentropy')

x1_train = np.random.rand(1590, 20, 10)
x1_test = np.random.rand(90, 20, 10)
y_train = np.random.rand(1590, 1)
y_test = np.random.rand(90, 1)

train_data_gen = Generator(x1_train, y_train, 256)
test_data_gen = Generator(x1_test, y_test, 256)

model.fit_generator(generator=train_data_gen,
                    validation_data=test_data_gen,
                    epochs=5,
                    shuffle=False,
                    verbose=1)

loss = model.evaluate_generator(generator=test_data_gen)
print('Test Loss: %0.5f' % loss)