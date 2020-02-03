#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import logging
from mltools.optimizer import Adamax
from mltools.data import CategoricalData
from NeuralNet import NeuralNet
from ActivationFunctions import ReLU, SoftMax
from Layers import Dense, SoftMaxDense
from Validations import test_error, train_error

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
np.seterr(over="raise", invalid="raise")

def load_mnist(filename):
    target, data = np.split(np.load(filename), [1], axis=1)
    data = data.astype("float64") / 255.0
    ok_target = np.eye(10)[target.astype("int64").flatten().tolist()]
    cd = CategoricalData(data, ok_target)
    cd.original_answer = target.astype("int64").flatten()
    return cd

def main():
    test = load_mnist("./data/mnist_test.npy")
    train = load_mnist("./data/mnist_train.npy")

    def validate(update_time, model, train, test):
        test_error(update_time, model, test)
        train_error(update_time, model, train)

    nn = NeuralNet([
        Dense((784, 200), ReLU, Adamax()),
        Dense((200, 200), ReLU, Adamax()),
        SoftMaxDense((200, 10), Adamax()),
    ], validate)
    nn.train(train, test, 50, 100)

if __name__=='__main__':
    main()