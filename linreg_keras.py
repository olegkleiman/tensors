# This excerpt demonstrates the linear regression implementation
# with help of simplest Keras model
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

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

# learning_rate = 0.05


def lr_schedule(epoch, lr):
    if epoch < 10:
        lr = 0.05
    else:
        lr = lr * tf.math.exp(-0.1)

    tf.summary.scalar('learning rate', data=lr, step=epoch)
    return lr


logdir = "logs/keras/linreg/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# At the beginning of every epoch, this callback gets the updated learning rate value from lr_schedule
lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1)
])
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(), # lr=learning_rate)
              # optimizer=tf.keras.optimizers.Adam(0.1)
              )

history = model.fit(x_train, y_train,
                    epochs=100,
                    verbose=True,  # Suppress chatty output; use Tensorboard instead
                    callbacks=[tensorboard_callback, lr_callback],
                    validation_data=(x_test, y_test),
                    validation_split=0.2, shuffle=True)
print("Average test loss: ", np.average(history.history['loss']))
weights = model.get_weights()
print(weights)

model.summary()
print(model.predict([60, 25, 2]))
