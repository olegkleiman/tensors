---
description: >-
  TF Models may be serialized into SavedModel format. This is a protobuf used
  for further deploying the model on various Python-independent platforms like
  tensorflow.js, TF Lite, or TF Serving
---

# Brief

When saving the model the following 2 topic has an important significance:

1. Initializing model's weights
2. Defining model's signatures

### Initializing model's weights

Before Keras model is able to save it should be "built", i.e. the framework should know not only the architecture of the model \(the relationship between the layers\) but the actual size of the tensors be passed through it. This is required in order to allocate the corresponding weights of tensors within the model. Given the shape of the first input tensor, Keras is able to calculate the rest, but this input size should be specified somehow by the author of the model.

Basically, the way to describe the input tensor's size is to provide its shape explicitly by specifying the `input_shape` of the first layer or alternatively, call the model directly providing the properly dimensioned tensor as dummy input

### Defining model's signatures

SavedModel format is actually a **directory** with a pre-defined structure. In its root, it contains`saved_model.pd` file that stores the actual TensorFlow program, or model, and a set of named signatures, each identifying a function that accepts tensor inputs and produces tensor outputs.

SavedModel -&gt; MetaGraphDef \(tag-set\) -&gt; SignatureDef\[\] -&gt; 

.  

----

Models' graphs are serialized in SavedModel format that is actually a binary _protobuf_. TensorFlow gets known about the graph to be saved with a help of `@tf.function` decorator. While Keras marks the appropriate function with this decorator automatically \(when the model is built\), the generic functions should be decorated manually. Moreover, this manual decoration usually should include the input parameters description \(in terms of Tensors\) as parameters to `@tf.function` decorator. 



TF models may be saved as SavedModel or HDF5 format for further deployment onto platforms that run without Python interpreter: [TF Serving](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple), [tensorflow.js](https://www.tensorflow.org/js/tutorials), and [TensorFlow Lite](https://www.tensorflow.org/lite/guide).



