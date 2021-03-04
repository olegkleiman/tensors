from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from plot_helpers import plot_to_image

print("Num GPU Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.gpu_device_name())
print(device_lib.list_local_devices())

X = np.arange(0.0, 1.0, 0.001, dtype='float32')
# DATA_SIZE = 100
# X = tf.random.uniform(shape=(DATA_SIZE,), minval=0.0, maxval=5.0)
perturb = tf.random.normal(shape=(len(X),), stddev=0.01)
y = 0.2 + 0.4 * X ** 2 + 0.3 * X * tf.math.sin(15 * X) + 0.05 * tf.math.cos(50 * X) + perturb

# callbacks
logdir = "logs/keras/hint/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, profile_batch='500,520')
# early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=True)

# model
model = keras.Sequential([
    keras.layers.Dense(256, activation=tf.nn.sigmoid, input_dim=1),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(64, activation=tf.nn.sigmoid),
    keras.layers.Dense(1)  # output layer that returns a single, continuous value
])

# training
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(lr=0.00001))
model.fit(X, y, epochs=1000,
          callbacks=[tensorboard_callback])  # , early_stopping_callback])
print(model.summary())
# predictions
predictions = model.predict(X)

file_writer = tf.summary.create_file_writer(logdir + "/metrics")
# plot
with file_writer.as_default():
    figure = plt.figure(figsize=(10, 10))
    plt.scatter(X, y, edgecolors='g')
    plt.plot(X, predictions, 'r')
    plt.legend(['Predicted Y', 'Actual Y'])
    image = plot_to_image(figure)
    tf.summary.image("sigmoidal", image, step=0)
