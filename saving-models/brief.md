# Brief

Models' graphs are serialized in SavedModel format that is actually a binary _protobuf_. TensorFlow gets known about the graph to be saved with a help of `@tf.function` decorator. While Keras marks the appropriate function with this decorator automatically \(when the model is built\), the generic functions should be decorated manually. Moreover, this manual decoration usually should include the input parameters description \(in terms of Tensors\) as parameters to `@tf.function` decorator. 

TF models may be saved as SavedModel or HDF5 format for further deployment onto platforms that run without Python interpreter: TF Serving, [tensorflow.js](https://www.tensorflow.org/js/tutorials), and [TensorFlow Lite](https://www.tensorflow.org/lite/guide).



