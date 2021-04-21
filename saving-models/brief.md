---
description: >-
  TF Models may be serialized into SavedModel format that is a protobuf used for
  further deploying the model on various Python-independent platforms like
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

SavedModel format is actually a **directory** with a pre-defined structure. In its root, it contains`saved_model.pd` file that is the SavedModel protocol buffer. It includes the graph definitions as `MetaGraphDef` protocol buffers that is actually the set of named signatures, each identifying a function that accepts tensor inputs and produces tensor outputs.

SavedModel -&gt; MetaGraphDef \(tag-set\) -&gt; SignatureDef\[\] -&gt; inputs/outputs

SavedModel signatures are discussed [here ](https://www.tensorflow.org/guide/saved_model)\(official TF doc\) in more detail. The valuable insight into the internals of SavedModel with signatures was published [here](https://blog.tensorflow.org/2021/03/a-tour-of-savedmodel-signatures.html?m=1) \(TensorFlow Blog\)

----

Models' graphs are serialized in SavedModel format that is actually a binary _protobuf_. TensorFlow gets known about the graph to be saved with a help of `@tf.function` decorator. While Keras marks the appropriate function with this decorator automatically \(when the model is built\), the generic functions should be decorated manually. Moreover, this manual decoration usually should include the input parameters description \(in terms of Tensors\) as parameters to `@tf.function` decorator. 







