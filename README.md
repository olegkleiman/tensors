# README

## What is this?

This workspace summarises the learning curve of theoretical and practical ML learning materials primary oriented for mastering Deep Learning and Neural Networks.

## What you should have before?

You should have a basic knowledge of the following mathematical areas:

* Linear Algebra: matrix terms \(shapes, dimensionality\) and operations \(multiplication, dot product\) 
* Probability Theory \(most important\):
  1. The properties of most known distributions: Normal \(Gauss\), Binomial, Logistic, Beta, Gamma
  2. Bayes reasoning
* Calculus: differentiation and basics of function analysis
* Information Theory:   

## Covered frameworks

This workspace primarily concentrated on TensorFlow 2.X. Other frameworks like PyTorch and Intel OpenVino are mentioned. 

Pay attention also to less known but brilliant solution JS-based for general ML:  ![](.gitbook/assets/ml5.png) ML5 \([https://learn.ml5js.org/](https://learn.ml5js.org/)\)

## Setup notes

As for Feb. 2021, in order to run on GeForce GTX 1660 Ti, TF 2.4.1 should be configured to use CUDA-compatible libraries as follows: 

* Python 3.8 \(TF did not release anything for Python 3.9\) 
* CUDA ver 10.2 \(preferable - over 11\). Get currently installed CUDA version:
  * $ [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#introduction) --version 

    > nvcc: NVIDIA \(R\) Cuda compiler driver 
    >
    > Copyright \(c\) 2005-2019 NVIDIA Corporation 
    >
    > Built on Wed\_Oct\_23\_19:32:27\_Pacific\_Daylight\_Time\_2019 
    >
    > Cuda compilation tools, **release 10.2**, V10.2.89
* cuDNN ver. 8.1 \(for all related CUDA versions\)
* CUPTI: according to [@sanjoy](https://github.com/tensorflow/tensorflow/issues/43030) \(Dec 1, 2020\): "copy cupti64\_2020.1.1.dll to cupti64\_110.dll to use the profiler on Windows. We'll fix this for TF 2.5."

## Tensorboard observations

1. Start tensorboard: $tensorboard --logdir=logs/keras/linreg\_relu

