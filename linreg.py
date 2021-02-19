import tensorflow as tf

print("TensorFlow version: ", tf.__version__)

logdir = './logs/linreg'
writer = tf.summary.create_file_writer(logdir)

slope = 0.4
bias = 1.5
rows = 100

x_train = tf.random.uniform(shape=(rows,))
perturb = tf.random.normal(shape=(len(x_train),), stddev=0.1)
y_train = slope * x_train + bias + perturb

m = tf.Variable(0.)
b = tf.Variable(0.)


def predict_y_value(x):
    y = m * x + b
    return y


@tf.function
def squared_error(y_pred, y_true):
    print('inside loss function')
    return tf.reduce_mean(tf.square(y_pred - y_true))


steps = 400
learning_rate = 0.05

# Bracket the loss function call - squared_error()- with
# tf.summary.trace_on() and tf.summary.trace_export()
tf.summary.trace_on(graph=True, profiler=False)

with writer.as_default():
    for epoch in range(steps):
        with tf.GradientTape() as tape:
            predictions = predict_y_value(x_train)
            loss = squared_error(predictions, y_train)
            tf.summary.scalar('loss', loss, step=epoch)

        gradients = tape.gradient(loss, [m, b])
        m.assign_sub(gradients[0] * learning_rate)
        b.assign_sub(gradients[1] * learning_rate)

    tf.summary.trace_export(name='mse_loss',
                            step=0,
                            profiler_outdir=logdir)

print("m: %f, b: %f" % (m.numpy(), b.numpy()))
