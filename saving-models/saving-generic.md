# Saving regular functions

![](../.gitbook/assets/colab_favicon.ico) The accompanying code is hosted on [Codelab](https://colab.research.google.com/drive/1aId7DB_bH6KAla529_154K_A0tITqeAS?usp=sharing).  

Saving Keras model generally does not differs from saving the generic functions except by the following points:

* Keras models already have the signature required by `SavedModel` format
* It is possible to use Subclassing API to derive the Keras model from `tf.Module` that provides `TrackableBase` class instances of which may be stored in the checkpoint file.

### Setup

```python
import tensorflow as tf
```

### Saving with Trackable

In order to save the model based on non-Keras code consider wrap this code with a class that inherits from`tf.Module`.

```python
class Wrapper(tf.Module):

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.float32)])
    def __call__(self, x, y):
        return tf.math.pow(x, y)
```

As discussed previously, decorating the method with @tf.function decorator actually attaches the execution graph \(`tf.Graph`\) to the decorated function. This graph is exposed by `ConcreteFunction` to callers.

This `ConcreteFunction` then maybe executed without Python runtime environment before **and after** the model is saved. Before the saving, it's straight-forward:

```python
model = Wrapper()

concrete_func = model.__call__.get_concrete_function()
result = concrete_func(3, 2)
print(result.numpy())
```

After the saving:

```python
saved_model_dir = './saved/pow/1'
tf.saved_model.save(model, saved_model_dir)

loaded_model = tf.saved_model.load(saved_model_dir)
loaded_model.__call__.get_concrete_function()
concrete_func = loaded_model.__call__.get_concrete_function()
result = concrete_func(3, 2)
print(result.numpy())
```

Note that we used here some previously known information about the function name and its parameters: the code executed after reloading the model assumed to know the name \_\_call and at least the types of the parameters passed to this function.

In the actual implementation, `SavedModel` format allowed to store this information to omit these assumptions.  

`SavedModel`introduces the concept of Signatures that provides all the caller needs

```python
loaded_model = tf.saved_model.load(saved_model_dir)
signature_map = loaded_model.signatures
print(signature_map.keys())
```

```text
KeysView(_SignatureMap({'serving_default': <ConcreteFunction signature_wrapper(*, x, y) at 0x1FAAC270880>}))
```



```python
concrete_func = signature_map['serving_default']
result = concrete_func(x=tf.constant(3.), y=tf.constant(2.))
print(result)
```

Remembering the `SavedModel` structure mentioned in the "Brief" section, you may analyze the saved model with the help of [`saved_model_cli`](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/saved_model.md#cli-to-inspect-and-execute-savedmodel) util :

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

This is typical predict SignatureDef allowed for calls to TF Serving Predict API. It posses no restrictions on the returned number of output Tensors.

Based on this analysis, the caller code for executing the stored graph may look like:

```python
loaded_model = tf.saved_model.load(saved_model_dir)
signature_map = loaded_model.signatures

concrete_func = signature_map['serving_default']
result = concrete_func(x=tf.constant(3.), y=tf.constant(2.))
print(result['output_0'])
```

Actually, the invocation of `concrete_func()` is passed to [TensorFlow C++ code](https://fossies.org/linux/tensorflow/tensorflow/python/tfe_wrapper.cc) \(\`TFE\__Py\__Execute\` wrapper\).

Pay attention that the result returned from the graph's invocation is a dictionary that read according to `outputs` section of `SignatureDef` 

