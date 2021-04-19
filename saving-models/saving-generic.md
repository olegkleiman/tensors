# Saving regular functions

![](../.gitbook/assets/colab_favicon.ico) The accompanying code is hosted on [Codelab](https://colab.research.google.com/drive/1aId7DB_bH6KAla529_154K_A0tITqeAS?usp=sharing).  

### Setup

```python
import tensorflow as tf
```

### Saving with Trackable

In order to save the model based on non-Keras code consider wrap this code with a class that inherits from`tf.Module` since it's the most generic class that derived from`TrackableBase` The instances of this class are the objects that can be stored inside checkpoint file.

```python
class Wrapper(tf.Module):

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.float32)])
    def __call__(self, x, y):
        return tf.math.pow(x, y)
```

`_call` method here is decorated with famous `@tf.function` decorator.

Then saving and loading the model is permormed straight-forward:

```python
model = Wrapper()
saved_model_dir = './saved/pow/1'
tf.saved_model.save(model, saved_model_dir)

loaded_model = tf.saved_model.load(saved_model_dir)
print(loaded_model.signatures)
```

