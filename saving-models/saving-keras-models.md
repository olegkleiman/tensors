# Saving Keras models

![](../.gitbook/assets/colab_favicon.ico) The accompanied code is hosted at [Colab](https://colab.research.google.com/drive/1ZYZb1LGmITC1e1fhojSBIaeHxuwFnjG7?usp=sharing).

### Setup

```python
import tensorflow as tf
from tensorflow import keras
from keras import layers
```

### Building and saving Sequential models with Functional API

In order to save the Keras model, it should be "built", i.e. it should be properly layered and the weights of each layer should be initialized somehow. Generally, when the Keras layer is initially created it has no associated weights. It creates its weights the first time it is called on an input.

The model initially has no weights as well. Hence, the following excerpt raises the exception

```python
model = keras.Sequential([
        keras.layers.Dense(8, activation=tf.nn.sigmoid, name="dense_1"),
        keras.layers.Dense(1, name="output_layer")
], name="My_Model")
print(model.summary())
```

{% hint style="danger" %}
ValueError: This model has not yet been built. Build the model first by calling `build()` or calling `fit()` with some data, or specify an `input_shape` argument in the first layer\(s\) for automatic build
{% endhint %}

Once the `input_shape` parameter of the first layer is added, you can call its `summary()`method:

```python
model = keras.Sequential([
        keras.layers.Dense(8, activation=tf.nn.sigmoid, name="dense_1", input_shape=(1,)),
        keras.layers.Dense(1, name="output_layer")
], name="My_Model")
print(model.summary())
```

```text
Model: "My_Model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 8)                 16        
_________________________________________________________________
output_layer (Dense)              (None, 1)                 9         
=================================================================
Total params: 25
Trainable params: 25
Non-trainable params: 0
_________________________________________________________________
None
```

Another way to initialize the weights is to call the model with some dummy but appropriately shaped input:

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
        keras.layers.Dense(8, activation=tf.nn.sigmoid, name="dense_1"),
        keras.layers.Dense(1, name="output_layer")
], name="My_Model")

input = tf.ones((1,4))
model(input)
print(model.summary())
```

In this case, the model calculates exactly the shapes, and thus, the definition of the built layer is slightly changed: we have`(1,8)`as a shape for the first Dense layer instead of `(None, 8)` in the previous run:

```text
Model: "My_Model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)             (1, 8)                    40        
_________________________________________________________________
output_layer (Dense)             (1, 1)                    9         
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________
None
```

### Use Input object     

When you know the input shape of the designed model, it's very useful to start its design with `Input` object that is actually TensorFlow symbolic tensor used as an entry point into a Network.

```python
model = keras.Sequential(name="My_Model")
model.add(keras.Input(shape=(1,4), name="input_layer"))
# or model.add(keras.layers.InputLayer(input_shape=(1, 4), name="input_layer"))
model.add(layers.Dense(8, activation=tf.nn.sigmoid, name="dense_1"))
model.add(layers.Dense(1, name="output_layer"))

print(model.summary())
```

Functionally this is the same model design as in previous examples, but internally`Input` object is used to create \(not displayed in summary\) `InputLayer` 

```text
Model: "My_Model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 1, 8)              40        
_________________________________________________________________
output_layer (Dense)         (None, 1, 1)              9         
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________
None
```

{% hint style="info" %}
`Sequential` API is not the only way to define Keras models. Actually, following this example, with the help of `Input`object, the model may be defined as more general:

```python
inputs = keras.Input(shape=(1, 4), name="input_layer")
x = keras.layers.Dense(8, activation=tf.nn.sigmoid, input_dim=1, name="dense_1")(inputs)
outputs = keras.layers.Dense(1, name="output_layer")(x)
model = tf.keras.Model(inputs, outputs, name="My_Model")
```

This is extremely useful for creating encoders/decoders.
{% endhint %}

The most important part of `Input` object is to provide the model with the description of the input in terms of `TensorSpec` similarly as we saw in `@tf.function` decorator parameters for generic function.

To sum up,  using `InputLayer` with Keras Sequential API, can be skipped moving the `input_shape` parameter to the first layer of the model

### Save a model

Once the model is built it may be saved as [TF SavedModel format](https://www.tensorflow.org/guide/saved_model):

```python
saved_model_dir = './my_model/1/'
tf.saved_model.save(model, saved_model_dir)
```

This is TensorFlow \(not Keras\) API for saving and may be applied to regular functions \(as we did in the previous section\) but Keras itself has similar functionality:generr

```python
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
```

### Save the subclassed Keras model

In many cases it's very convinient to define the custom model by inheriting \(subclassing\) from generic, not-Sequential,`keras.Model` class.

```python
class KerasModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(KerasModel, self).__init__()

        inputs = tf.keras.Input(shape=input_shape)

        x = Dense(8, activation=tf.nn.sigmoid, input_dim=1)(inputs)
        outputs = Dense(1)(x)
        self.model = tf.keras.Model(inputs, outputs, name="My_Model")
```

The models defined in this way may be safely serialized into SavedModel format by calling both tf.saved\_model.save\(\) or keras.model.save\(\), but saving into HDF5 is not possible, because

> such models are deined via the body of a Python method, which isn't safely serializable

>

{% page-ref page="saving-keras-models.md" %}

