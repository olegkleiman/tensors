import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("{} version: {}".format(tf.__name__, tf.__version__))

inputs = keras.Input(shape=(1,), name='dense_input')
denseLayer = layers.Dense(units=2, activation='linear')
output = denseLayer(inputs)
model = keras.Model(inputs=inputs, outputs=output, name='my_model')