from absl import app
import tensorflow as tf

from tensorboard_plugin_example import summary_v2


tf.compat.v1.enable_eager_execution()
tf = tf.compat.v2


def main(unused_argv):
    writer = tf.summary.create_file_writer("demo_logs")
    with writer.as_default():
        summary_v2.greeting(
            "guestbook",
            "Alice",
            step=0,
            description="Sign your name!",
        )
        summary_v2.greeting(
            "guestbook", "Bob", step=1
        )  # no need for `description`
        summary_v2.greeting("guestbook", "Cheryl", step=2)
        summary_v2.greeting("more_names", "David", step=4)


if __name__ == "__main__":
    app.run(main)