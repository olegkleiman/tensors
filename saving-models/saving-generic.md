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

Then saving and loading the model is performed straight-forward:

```python
model = Wrapper()
saved_model_dir = './saved/pow/1'
tf.saved_model.save(model, saved_model_dir)

loaded_model = tf.saved_model.load(saved_model_dir)
print(loaded_model.signatures)
```

```text
_SignatureMap({'serving_default': <ConcreteFunction signature_wrapper(*, x, y) at 0x1CDCF595C70>})
```

Remembering the SavedModel structure mentioned in the "Brief" section, you may analyze the saved model with the help of [`saved_model_cli`](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/saved_model.md#cli-to-inspect-and-execute-savedmodel) util :

```python
..\saved\pow> saved_model_cli show --dir ./1 --all
2021-04-20 03:20:05.579456: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: serving_default_x:0
    inputs['y'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: serving_default_y:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_0'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: PartitionedCall:0
  Method name is: tensorflow/serving/predict

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          x: TensorSpec(shape=(), dtype=tf.float32, name='x')
        Argument #2
          y: TensorSpec(shape=(), dtype=tf.float32, name='y')
```

This is typical predict SignatureDef allowed for calls to TF Serving Predict API.

