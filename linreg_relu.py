from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from plot_helpers import plot_to_image

print("Num GPU Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.gpu_device_name())

DATA_SIZE = 1000
slope = 0.4
bias = 1.5
logdir = "logs/keras/linreg_relu/" + datetime.now().strftime("%Y%m%d-%H%M%S")

x_train = tf.random.uniform(shape=(DATA_SIZE,))
perturb = tf.random.normal(shape=(len(x_train),), stddev=0.1)
y_train = slope * x_train + bias + perturb

file_writer = tf.summary.create_file_writer(logdir + "/metrics")

# Create plot image and later store it into TensorBoard
figure = plt.figure(figsize=(10, 10))
ax = figure.add_subplot(1, 2, 1)
ax.scatter(x_train, y_train, color='blue')

model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[1]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)  # output layer that returns a single, continuous value
])

# Functional API
# inputs = keras.Input(shape=(1,), name='dense_input')
# output = layers.Dense(units=1, activation='linear')(inputs)
# model = keras.Model(inputs=inputs, outputs=output, name='my_model')


tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=True)

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(0.1)
              )

model.fit(x_train, y_train,
          batch_size=8,
          epochs=100,
          verbose=True,
          # validation_split=0.1,
          callbacks=[tensorboard_callback, early_stopping_callback])
model.summary()
# print('Weights: {}'.format(model.weights))
predictions = model.predict(x_train)
# print(predictions)

with file_writer.as_default():
    ax = figure.add_subplot(1, 2, 2)
    ax.scatter(x_train, predictions, color='red')

    image = plot_to_image(figure)
    tf.summary.image("plot", image, step=0)
