from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from plot_helpers import plot_to_image

print("Num GPU Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.gpu_device_name())

X = np.arange(0.0, 5.0, 0.01, dtype='float32')
# DATA_SIZE = 100
# X = tf.random.uniform(shape=(DATA_SIZE,), minval=0.0, maxval=5.0)
perturb = tf.random.normal(shape=(len(X),), stddev=0.1)
y = 5 * tf.experimental.numpy.power(X, 2) + perturb

# callbacks
logdir = "logs/keras/hint/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, profile_batch='500,520')
# early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=True)

# model
model = keras.Sequential([
    keras.layers.Dense(50, activation=tf.nn.sigmoid, input_dim=1),
    keras.layers.Dense(30, activation=tf.nn.sigmoid),
    keras.layers.Dense(1)  # output layer that returns a single, continuous value
])

# training
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(lr=0.001))
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
