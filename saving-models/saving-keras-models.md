# Saving Keras models

The accompanied code is hosted at [Colab](https://colab.research.google.com/drive/1ZYZb1LGmITC1e1fhojSBIaeHxuwFnjG7?usp=sharing).

In order to save the Keras model, it should be "built", i.e. it should be properly layered and the weights of each layer should be initialized somehow. Generally, when the Keras layer is initially created it has no associated weights. It creates its weights the first time it is called on an input.

The model initially has no weights as well. Hence, the following excerpt raises the exception

```python
model = keras.Sequential([
        keras.layers.Dense(8, activation=tf.nn.sigmoid, name="dense_1"),
        keras.layers.Dense(1, name="output_layer")
])
print(model.summary())
```

{% hint style="danger" %}
ValueError: This model has not yet been built. Build the model first by calling `build()` or calling `fit()` with some data, or specify an `input_shape` argument in the first layer\(s\) for automatic build
{% endhint %}

Once the input\_shape parameter of the first layer is added, you can call its `summary()`method:

```python
model = keras.Sequential([
        keras.layers.Dense(8, activation=tf.nn.sigmoid, name="dense_1", input_shape=(1,)),
        keras.layers.Dense(1, name="output_layer")
])
model._name = "My Model"
print(model.summary())
```

```text
Model: "My Model"
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

Another way to initialize the weights is to call with some dummy but appropriately shaped input:

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
        keras.layers.Dense(8, activation=tf.nn.sigmoid, name="dense_1"),
        keras.layers.Dense(1, name="output_layer")
])
model._name = "My Model"
input = tf.ones((1,4))
model(input)
print(model.summary())
```

In this case, the model calculates exactly the shapes, and thus, the definition of the built layer is slightly changed: we have`(1,8)`as a shape for the first Dense layer instead of `(None, 8)` in the previous run:

```text
Model: "My Model"
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



Once the model is built it may be saved as TF SavedModel format:

```text
saved_model_dir = './my_model/1/'
tf.saved_model.save(model, saved_model_dir)
```



{% page-ref page="saving-keras-models.md" %}

