#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import logging
import matplotlib.pyplot as plt
from mltools.data import Data
from NeuralNet import NeuralNet

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
np.seterr(over="raise", invalid="raise")

def load_mnist(filename):
    target, data = np.split(np.load(filename), [1], axis=1)
    data = data.astype("float64") / 255.0
    d = Data(data)
    return d

def main():
    train = load_mnist("./data/mnist_train.npy")
    test = load_mnist("./data/mnist_test.npy")

    nn = NeuralNet.load("mnist.pickle")
    recon = nn.forward(test.data[0])
    plt.imshow(test.data[0].reshape(28, 28), cmap="gray")
    plt.show()
    plt.imshow(recon.reshape(28, 28), cmap="gray")
    plt.show()

if __name__=='__main__':
    main()