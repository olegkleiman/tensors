import os
import sys
import tensorflow as tf

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

print("TensorFlow version: ", tf.__version__)
print("Num GPU Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.python import _pywrap_util_port
print("MKL enabled:", _pywrap_util_port.IsMklEnabled())

tf.debugging.set_log_device_placement(False)

logdir = './logs/func2'
writer = tf.summary.create_file_writer(logdir)


@tf.function
def my_func(ax, bx):
    return tf.matmul(ax, bx)


A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32, name='A')
x = tf.constant([[5, 6], [7, 8]], dtype=tf.float32, name='x')

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export()
tf.summary.trace_on(graph=True, profiler=False)
z = my_func(A, x)

loss = z[0, 0].numpy()
tf.print(loss, output_stream=sys.stdout)

with writer.as_default():
    for i in range(10):
        tf.summary.scalar('loss', loss+i, step=i)
    tf.summary.trace_export(name='this_summary_name',
                            step=0,
                            profiler_outdir=logdir)
