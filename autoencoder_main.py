#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import logging
from mltools.optimizer import Adamax
from mltools.data import Data
from NeuralNet import NeuralNet
from ActivationFunctions import ReLU, SoftMax
from Layers import Dense, SoftMaxLayer, MeanSquareErrorLayer
from Validations import mean_square

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
np.seterr(over="raise", invalid="raise")

def load_mnist(filename):
    target, data = np.split(np.load(filename), [1], axis=1)
    data = data.astype("float64") / 255.0
    d = Data(data)
    return d

def main():
    train = load_mnist("./data/zero.npy")

    def validate(update_time, model, train, test):
        mean_square(update_time, model, train)

    nn = NeuralNet([
        Dense((784, 200), ReLU, Adamax()),
        Dense((200, 784), ReLU, Adamax()),
        MeanSquareErrorLayer()
    ], validate)
    nn.train(train, train, 50, 100)
    nn.save("zero.pickle")

if __name__=='__main__':
    main()