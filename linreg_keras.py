# This excerpt demonstrates the linear regression implementation
# with help of simplest Keras model
from datetime import datetime
from time import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import io
import matplotlib.pyplot as plt

print("Num GPU Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.gpu_device_name())

slope = 0.4
bias = 1.5
data_size = 1000
# 80% of the data is for training.
train_pct = 0.8


def generate_plot(x, y):
    plt.figure()
    plt.scatter(x, y)
    plt.title('y=0.4*x+1.5')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


x_train = tf.random.uniform(shape=(data_size,))
perturb = tf.random.normal(shape=(len(x_train),), stddev=0.1)
y_train = slope * x_train + bias + perturb

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
data_train = train_dataset.take(800)
data_test = train_dataset.skip(800)

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

# Add image to TensorBoard
plot_buf = generate_plot(x_train, y_train)
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
# Add the batch dimension
image = tf.expand_dims(image, 0)
tf.summary.image("plot", image, step=0)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# At the beginning of every epoch, this callback gets the updated learning rate value from lr_schedule
lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
# This callback will stop the learning process if there is no improvement in minimizing loss for 3 epochs
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=True)

# output = Dense(1, activation='relu')(input_vals)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1)
])
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(),  # lr=learning_rate),
              # optimizer=tf.keras.optimizers.Adam(0.1)
              metrics=[tf.metrics.mean_squared_error,
                       tf.metrics.mean_absolute_error,
                       tf.metrics.mean_absolute_percentage_error]
              )

# A very rough measurement of total learning time.
# More precise measurement is use of custom callbacks
start = time()
history = model.fit(x_train, y_train,
                    epochs=100,
                    verbose=True,  # Suppress chatty output; use Tensorboard instead
                    callbacks=[tensorboard_callback, lr_callback, early_stopping_callback])
print('Total learning time: {} sec.'.format(time()-start))

print("Average test loss: ", np.average(history.history['loss']))
weights = model.get_weights()
print(weights)

model.summary()
print(model.predict([60, 25, 2]))
