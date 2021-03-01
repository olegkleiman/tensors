import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

physical_devices = tf.config.list_physical_devices('GPU')

DATA_SIZE = 1000
slope = 0.4
bias = 1.5

x_train = tf.constant(tf.random.uniform(shape=(DATA_SIZE, 2)))

# z_train = tf.math.pow(x_train[:, 0], 2) + x_train[:, 1]
z_train = tf.constant(x_train[:, 0] + x_train[:, 1] - 4.5)
print(z_train.shape)
#
# fig = plt.figure()
# ax = fig.add_subplot(111) #, projection='3d')
# ax = Axes3D(fig)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.scatter(x_train[:, 0], x_train[:, 1], z_train) #, c='r', marker='o')
#
# plt.show()

logdir = "logs/keras/linreg_multi/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")

early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=True )
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model = tf.keras.models.Sequential([
    # tf.keras.layers.Dense(units=1, activation=tf.nn.relu, input_dim=2),
    tf.keras.layers.Dense(units=1, input_dim=2)
])
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(0.1))
history = model.fit(x_train, z_train,
                    verbose=True,
                    epochs=100,
                    batch_size=8,
                    callbacks=[tensorboard_callback, early_stopping_callback])
model.summary()
print(model.trainable_variables)
print(model.metrics_names)
print(model.predict([
    [2.7, 0.3],
    [4.8, 1.2]]))
