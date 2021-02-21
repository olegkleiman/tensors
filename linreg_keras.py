from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from custom_tf_callbacks import CustomCallbacks

print("Num GPU Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.gpu_device_name())

slope = 0.4
bias = 1.5
data_size = 1000
# 80% of the data is for training.
train_pct = 0.8

x_train = tf.random.uniform(shape=(data_size,))
perturb = tf.random.normal(shape=(len(x_train),), stddev=0.1)
y_train = slope * x_train + bias + perturb

data_size = 800
train_size = int(data_size * train_pct)
x_test, y_test = x_train[train_size:], y_train[train_size:]

logdir = "logs/keras/linreg/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1)
])
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(x_train, y_train,
                    epochs=100,
                    verbose=False,  # Suppress chatty output; use Tensorboard instead
                    callbacks=[tensorboard_callback],
                    validation_data=(x_test, y_test),
                    validation_split=0.2, shuffle=True)
print("Average test loss: ", np.average(history.history['loss']))
weights = model.get_weights()
print(weights)

model.summary()
print(model.predict([60, 25, 2]))