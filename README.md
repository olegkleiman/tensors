# README

## What is this

This workspace summarises the learning curve of theoretical and practical ML learning materials primary oriented for Deep Learning and Neural Networks.

## Setup notes

As for Feb. 2021, in order to run on GeForce GTX 1660 Ti, TF 2.4.1 should be configured to use CUDA-compatible libraries as following: 1. Python 3.8 \(TF did not release anything for Python 3.9\) 2. CUDA ver 10.2 \(preferable - over 11\) 3. cuDNN ver. 8.1 \(for all related CUDA versions\) 4. CUPTI: according to [@sanjoy](https://github.com/tensorflow/tensorflow/issues/43030) \(Dec 1, 2020\): "copy cupti64\_2020.1.1.dll to cupti64\_110.dll to use the profiler on Windows. We'll fix this for TF 2.5."

## Tensorboard observations

1. Start tensorboard: $tensorboard --logdir=logs/keras/linreg\_relu

