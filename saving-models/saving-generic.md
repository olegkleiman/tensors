# Saving regular

The accompanying code is hosted on [Codelab](https://colab.research.google.com/drive/1aId7DB_bH6KAla529_154K_A0tITqeAS?usp=sharing).  

In order to save the model based on non-Keras code consider wrap this code with a class that inherits from`tf.Module` since it's the most generic class that implements Trackable interface.

```python
class Wrapper(tf.Module):

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32),
                                  tf.TensorSpec(shape=None, dtype=tf.float32)])
    def __call__(self, x, y):
        return tf.math.pow(x, y)
```

`_call` method here is decorated with famous `@tf.function` decorator 

