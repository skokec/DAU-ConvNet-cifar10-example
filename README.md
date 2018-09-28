## CIFAR-10 example for DAU-ConvNet

Example for using DAU-ConvNet with simple three layer architecture on CIFAR-10 dataset. Based on original TensorFlow CIFAR-10 tutorial. Modified only network architecture in cifar10.py ```inference()``` method.

You need to have [DAU-ConvNet TensorFlow plugin](https://https://github.com/skokec/DAU-ConvNet) compiled, installed and available in python system path.
 
## Original README

**NOTE: For users interested in multi-GPU, we recommend looking at the newer [cifar10_estimator](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator) example instead.**

---

CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

Detailed instructions on how to get started available at:

http://tensorflow.org/tutorials/deep_cnn/
