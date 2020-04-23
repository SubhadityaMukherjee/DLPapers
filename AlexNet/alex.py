# %%
import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense,Dropout
import  matplotlib.pyplot as plt
import numpy as np

# %%

inputs = Input(shape = (32, 32, 3))
x = Conv2D(filters=96,kernel_size=(3,3),strides=(4,4),input_shape=input_shape, activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
x = Conv2D(256,(5,5),padding='same',activation='relu')(x)

x = Conv2D(384,(3,3),padding='same',activation='relu')(x)
x = Conv2D(384,(3,3),padding='same',activation='relu')(x)
x = Conv2D(256,(3,3),padding='same',activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

x = Flatten()(x)
x = Dense(4096,activation='relu')(x)
x = Dropout(.4)(x)
x = Dense(4096,activation='relu')(x)
x = Dropout(.4)(x)
x = Dense(10,activation='softmax')(x)


model = Model(inputs = inputs, outputs = x)

# %%
(x, y), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# %%
x.shape

# %%
y.shape
#%%
y[0]

# %%
plt.figure()
plt.imshow(x[1])
plt.colorbar()
# %%
x = x/255.0
x_test = x_test/255.0

# %%
model.summary()

# %% [markdown]
## Compile

# %%
model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
#%%
# x = np.expand_dims(x, -1)
# %%
model.fit(x=x, y=y,
          batch_size=64, epochs=10,
          verbose=1, 
          steps_per_epoch=1)

# %%


# %%
