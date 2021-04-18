---
description: >-
  Model, algorithm, gradient descent, min loss function. Automatic
  differentiation on calculation graph. TensorFlow, PyTorch, etc are automatic
  differentiation frameworks.
---

# Brief

An example hosted at [Colab](https://colab.research.google.com/drive/1Y0Ylkt15XkNEpZ057DNW5fnU5ylqwyUu?usp=sharing) demonstrates using **@tf.function** decorator as an entry point for creating the calculation graph. The first time the decorated function is called it is parsed into the graph. Next time the calls to this function are executed on the created graph. 

This graph is visualized in TensorBoard with help on `ft.summary.*` methods.

